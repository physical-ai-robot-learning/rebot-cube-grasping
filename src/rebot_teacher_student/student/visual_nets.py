"""Shared network modules for the reBot visual policy (SQUINT-exact), pure torch (no Isaac import).

Stage: June student networks (torch only).
Was src/rebot_sim2real/student/visual_nets.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).


Used by:
  - train_dagger_visual.py : teacher(StateOnlyActor) -> student(CNNEncoder x2 + Projection + Actor)
  - (optionally) a visual-RL trainer that also needs CriticC51 + project_target.

Architecture (SQUINT):
  CNNEncoder(16/32/64 -> 1024) ; Projection(wrist 1024->50, front 1024->50, state n->256 -> 356);
  Actor(proj -> 3x[256+LN+ReLU] -> mean/logstd) ; CriticC51(proj+act -> 3x[512+LN+ReLU] -> atoms, 2-Q).
  StateOnlyActor = the state-only teacher (Linear(n_state->256) proj, same fc/mean/logstd).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonalInit(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ── visual student modules ──
class CNNEncoder(nn.Module):
    """SQUINT CNN encoder. Input (N,H,W,3) uint8. Output (N,1024). image_size in {16,32,64}."""
    REPR_DIM = 1024

    def __init__(self, image_size):
        super().__init__()
        if image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 128:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),       # 128 -> 31
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),      # 31 -> 14
                nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),      # 14 -> 6
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())  # 6 -> 4 => 64*4*4 = 1024
        else:
            raise ValueError(f"unsupported image size {image_size}")
        self.apply(orthogonalInit)

    def forward(self, img_uint8):
        tX = img_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(tX)


class Projection(nn.Module):
    """wrist_feat(1024)->50, front_feat(1024)->50, state(n)->256, concat -> 356."""
    REPR_DIM = 50 + 50 + 256   # legacy (wrist+front+state); use instance .repr_dim

    def __init__(self, n_state, use_wrist=True):
        super().__init__()
        self.use_wrist = use_wrist
        if use_wrist:
            self.wrist_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.front_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.repr_dim = (50 if use_wrist else 0) + 50 + 256   # 356 (wrist+front) | 306 (front-only)

    def forward(self, wf, ff, state):
        tParts = [self.wrist_proj(wf)] if self.use_wrist else []
        tParts += [self.front_proj(ff), self.state_proj(state)]
        return torch.cat(tParts, dim=-1)


class Actor(nn.Module):
    """Visual actor: (wrist_feat, front_feat, state) -> tanh-squashed Gaussian action in [-1,1]."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state, n_act, hidden=256, use_wrist=True):
        super().__init__()
        self.proj = Projection(n_state, use_wrist=use_wrist)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)
        self.apply(orthogonalInit)

    def _meanLogstd(self, wf, ff, s):
        tX = self.fc(self.proj(wf, ff, s))
        tMean = self.mean(tX)
        tLogstd = torch.tanh(self.logstd(tX))
        tLogstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (tLogstd + 1)
        return tMean, tLogstd

    def meanAction(self, wf, ff, s):
        """Deterministic squashed action tanh(mean). Keeps gradient (for imitation regression)."""
        tMean, _ = self._meanLogstd(wf, ff, s)
        return torch.tanh(tMean)

    def getAction(self, wf, ff, s):
        tMean, tLogstd = self._meanLogstd(wf, ff, s)
        tStd = tLogstd.exp()
        tNormal = torch.distributions.Normal(tMean, tStd)
        tX = tNormal.rsample()
        tY = torch.tanh(tX)
        tLogp = (tNormal.log_prob(tX) - torch.log(1 - tY.pow(2) + 1e-6)).sum(1, keepdim=True)
        return tY, tLogp

    @torch.no_grad()
    def actDeterministic(self, wf, ff, s):
        tMean, _ = self._meanLogstd(wf, ff, s)
        return torch.tanh(tMean)


class StagedVisualActor(nn.Module):
    """STAGED visual student: a SHARED Projection (wrist+front CNN features + 21D proprio) feeding
    n_stages action heads + 1 stage-classifier head. At deploy the stage head predicts the current stage
    from vision+proprio (distilled from the teacher's STATE-based stage label) and the matching action head
    is run — so NO privileged state is needed on the real robot. enc_w/enc_f (CNNEncoder) are external/shared."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state, n_act, n_stages=4, hidden=256, use_wrist=True):
        super().__init__()
        self.n_stages = n_stages
        self.proj = Projection(n_state, use_wrist=use_wrist)
        tD = self.proj.repr_dim

        def body():
            return nn.Sequential(nn.Linear(tD, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.bodies = nn.ModuleList([body() for _ in range(n_stages)])
        self.means = nn.ModuleList([nn.Linear(hidden, n_act) for _ in range(n_stages)])
        self.logstds = nn.ModuleList([nn.Linear(hidden, n_act) for _ in range(n_stages)])  # RL fine-tune only
        self.stage_body = nn.Sequential(nn.Linear(tD, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.stage_out = nn.Linear(hidden, n_stages)
        self.apply(orthogonalInit)

    def feat(self, wf, ff, s):
        return self.proj(wf, ff, s)

    def stageLogits(self, feat):
        return self.stage_out(self.stage_body(feat))

    def allActions(self, feat):
        """(N, n_stages, n_act) = tanh(mean) for every stage head."""
        return torch.stack([torch.tanh(self.means[tI](self.bodies[tI](feat))) for tI in range(self.n_stages)], dim=1)

    def actionFor(self, feat, stageIdx):
        """(N, n_act): each env's action from its `stage_idx` head."""
        tActs = self.allActions(feat)
        return tActs.gather(1, stageIdx.view(-1, 1, 1).expand(-1, 1, tActs.shape[-1])).squeeze(1)

    @torch.no_grad()
    def actDeploy(self, wf, ff, s):
        """deploy: predict stage from vision+proprio -> run the matching head. returns (action(N,n_act), stage(N,))."""
        tFeat = self.feat(wf, ff, s)
        tStage = self.stageLogits(tFeat).argmax(dim=-1)
        return self.actionFor(tFeat, tStage), tStage

    # ── stochastic per-stage head (visual-RL fine-tune ONLY; deploy/distill use the deterministic means) ──
    def _meanLogstdStage(self, feat, s):
        tX = self.bodies[s](feat)
        tMean = self.means[s](tX)
        tLs = torch.tanh(self.logstds[s](tX))
        tLs = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (tLs + 1)
        return tMean, tLs

    def getActionStage(self, feat, s):
        """tanh-squashed Gaussian sample for a batch ALL in stage s. Returns (action(N,n_act), logp(N,1))."""
        tMean, tLs = self._meanLogstdStage(feat, s)
        tStd = tLs.exp()
        tNormal = torch.distributions.Normal(tMean, tStd)
        tX = tNormal.rsample()
        tY = torch.tanh(tX)
        tLogp = (tNormal.log_prob(tX) - torch.log(1 - tY.pow(2) + 1e-6)).sum(1, keepdim=True)
        return tY, tLogp


class CriticC51(nn.Module):
    """Distributional C51 critic, 2-Q ensemble (for optional visual-RL fine-tune)."""
    def __init__(self, n_state, n_act, num_atoms, v_min, v_max, num_q=2, hidden=512, use_wrist=True):
        super().__init__()
        self.num_atoms, self.num_q, self.v_min, self.v_max = num_atoms, num_q, v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.proj = Projection(n_state, use_wrist=use_wrist)

        def qnet():
            return nn.Sequential(
                nn.Linear(self.proj.repr_dim + n_act, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, num_atoms))
        self.qs = nn.ModuleList([qnet() for _ in range(num_q)])
        self.apply(orthogonalInit)

    def logits(self, wf, ff, s, a):
        tX = torch.cat([self.proj(wf, ff, s), a], dim=-1)
        return torch.stack([tQ2(tX) for tQ2 in self.qs], dim=0)

    def qValues(self, wf, ff, s, a):
        tProbs = F.softmax(self.logits(wf, ff, s, a), dim=-1)
        return (tProbs * self.support).sum(-1)


@torch.no_grad()
def projectTarget(targetCritic, nwf, nff, ns, na, rewards, bootstrap, gamma):
    """C51 categorical projection for the visual critic. Returns target dists (num_q, N, atoms).
    (Used by the optional visual-RL trainer; the DAgger student is supervised and does not need it.)"""
    tLogits = targetCritic.logits(nwf, nff, ns, na)
    tProbs = F.softmax(tLogits, dim=-1)
    tSupport = targetCritic.support
    tVMin, tVMax, tA = targetCritic.v_min, targetCritic.v_max, targetCritic.num_atoms
    tQ, tNn, _ = tProbs.shape
    tDz = (tVMax - tVMin) / (tA - 1)
    tTz = rewards.view(1, tNn, 1) + bootstrap.view(1, tNn, 1) * gamma * tSupport.view(1, 1, tA)
    tTz = tTz.clamp(tVMin, tVMax).expand(tQ, tNn, tA)
    tB = (tTz - tVMin) / tDz
    tL = tB.floor().long(); tU = tB.ceil().long()
    tIsInt = (tU == tL)
    tL = torch.where((tL > 0) & tIsInt, tL - 1, tL)
    tU = torch.where((tL == 0) & tIsInt, tU + 1, tU)
    tL = tL.clamp(0, tA - 1); tU = tU.clamp(0, tA - 1)
    tM = torch.zeros_like(tProbs)
    tM.scatter_add_(2, tL, tProbs * (tU.float() - tB))
    tM.scatter_add_(2, tU, tProbs * (tB - tL.float()))
    return tM


# ── state-only teacher ──
class StateOnlyActor(nn.Module):
    """The trained state-only C51 actor (privileged 43D obs -> action). Used as the DAgger teacher."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state, n_act, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)

    @torch.no_grad()
    def actDeterministic(self, s):
        tX = self.fc(self.proj(s))
        return torch.tanh(self.mean(tX))

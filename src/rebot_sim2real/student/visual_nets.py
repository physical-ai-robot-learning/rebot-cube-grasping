"""Shared network modules for the reBot visual policy (SQUINT-exact), pure torch (no Isaac import).

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


def orthogonal_init(m):
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
        self.apply(orthogonal_init)

    def forward(self, img_uint8):
        x = img_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(x)


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
        parts = [self.wrist_proj(wf)] if self.use_wrist else []
        parts += [self.front_proj(ff), self.state_proj(state)]
        return torch.cat(parts, dim=-1)


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
        self.apply(orthogonal_init)

    def _mean_logstd(self, wf, ff, s):
        x = self.fc(self.proj(wf, ff, s))
        mean = self.mean(x)
        logstd = torch.tanh(self.logstd(x))
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)
        return mean, logstd

    def mean_action(self, wf, ff, s):
        """Deterministic squashed action tanh(mean). Keeps gradient (for imitation regression)."""
        mean, _ = self._mean_logstd(wf, ff, s)
        return torch.tanh(mean)

    def get_action(self, wf, ff, s):
        mean, logstd = self._mean_logstd(wf, ff, s)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        logp = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y, logp

    @torch.no_grad()
    def act_deterministic(self, wf, ff, s):
        mean, _ = self._mean_logstd(wf, ff, s)
        return torch.tanh(mean)


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
        d = self.proj.repr_dim

        def body():
            return nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.bodies = nn.ModuleList([body() for _ in range(n_stages)])
        self.means = nn.ModuleList([nn.Linear(hidden, n_act) for _ in range(n_stages)])
        self.logstds = nn.ModuleList([nn.Linear(hidden, n_act) for _ in range(n_stages)])  # RL fine-tune only
        self.stage_body = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.stage_out = nn.Linear(hidden, n_stages)
        self.apply(orthogonal_init)

    def feat(self, wf, ff, s):
        return self.proj(wf, ff, s)

    def stage_logits(self, feat):
        return self.stage_out(self.stage_body(feat))

    def all_actions(self, feat):
        """(N, n_stages, n_act) = tanh(mean) for every stage head."""
        return torch.stack([torch.tanh(self.means[i](self.bodies[i](feat))) for i in range(self.n_stages)], dim=1)

    def action_for(self, feat, stage_idx):
        """(N, n_act): each env's action from its `stage_idx` head."""
        acts = self.all_actions(feat)
        return acts.gather(1, stage_idx.view(-1, 1, 1).expand(-1, 1, acts.shape[-1])).squeeze(1)

    @torch.no_grad()
    def act_deploy(self, wf, ff, s):
        """deploy: predict stage from vision+proprio -> run the matching head. returns (action(N,n_act), stage(N,))."""
        feat = self.feat(wf, ff, s)
        stage = self.stage_logits(feat).argmax(dim=-1)
        return self.action_for(feat, stage), stage

    # ── stochastic per-stage head (visual-RL fine-tune ONLY; deploy/distill use the deterministic means) ──
    def _mean_logstd_stage(self, feat, s):
        x = self.bodies[s](feat)
        mean = self.means[s](x)
        ls = torch.tanh(self.logstds[s](x))
        ls = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (ls + 1)
        return mean, ls

    def get_action_stage(self, feat, s):
        """tanh-squashed Gaussian sample for a batch ALL in stage s. Returns (action(N,n_act), logp(N,1))."""
        mean, ls = self._mean_logstd_stage(feat, s)
        std = ls.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        logp = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y, logp


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
        self.apply(orthogonal_init)

    def logits(self, wf, ff, s, a):
        x = torch.cat([self.proj(wf, ff, s), a], dim=-1)
        return torch.stack([q(x) for q in self.qs], dim=0)

    def q_values(self, wf, ff, s, a):
        probs = F.softmax(self.logits(wf, ff, s, a), dim=-1)
        return (probs * self.support).sum(-1)


@torch.no_grad()
def project_target(target_critic, nwf, nff, ns, na, rewards, bootstrap, gamma):
    """C51 categorical projection for the visual critic. Returns target dists (num_q, N, atoms).
    (Used by the optional visual-RL trainer; the DAgger student is supervised and does not need it.)"""
    logits = target_critic.logits(nwf, nff, ns, na)
    probs = F.softmax(logits, dim=-1)
    support = target_critic.support
    v_min, v_max, A = target_critic.v_min, target_critic.v_max, target_critic.num_atoms
    Q, Nn, _ = probs.shape
    dz = (v_max - v_min) / (A - 1)
    tz = rewards.view(1, Nn, 1) + bootstrap.view(1, Nn, 1) * gamma * support.view(1, 1, A)
    tz = tz.clamp(v_min, v_max).expand(Q, Nn, A)
    b = (tz - v_min) / dz
    l = b.floor().long(); u = b.ceil().long()
    is_int = (u == l)
    l = torch.where((l > 0) & is_int, l - 1, l)
    u = torch.where((l == 0) & is_int, u + 1, u)
    l = l.clamp(0, A - 1); u = u.clamp(0, A - 1)
    m = torch.zeros_like(probs)
    m.scatter_add_(2, l, probs * (u.float() - b))
    m.scatter_add_(2, u, probs * (b - l.float()))
    return m


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
    def act_deterministic(self, s):
        x = self.fc(self.proj(s))
        return torch.tanh(self.mean(x))

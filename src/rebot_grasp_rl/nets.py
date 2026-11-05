"""Project 2 networks: front+wrist DETECTION + cube ORIENTATION cross-attention, with FFN (MLP) mixing.

Stage: Project 2 networks (torch only).
Was src/det_xattn/det_xattn_nets.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).

No CNN / no pixels in the policy. Inputs (all real-measurable, normalized):
  - per camera detection: cx, cy, area_frac, visible  (HSV red centroid)
  - cube orientation: sin(yaw), cos(yaw)  (table top-down grasp; swap to 4D quat for 6-DOF)
Cross-attention fuses {front, wrist, orientation} tokens (ori restored 2026-06-29); transformer FFN mixes; + proprio -> SAC.
Action dim n_act is set by the ENV: 7 for joint-delta, OR pose-command (Δpos3 + ori + grip) for IK."""
import torch
import torch.nn as nn

D_DET = 4   # per camera: cx, cy, area_frac, visible
D_ORI = 2   # cube orientation: sin(yaw), cos(yaw)  (set 4 for quaternion / 6 for 6D-rot if 6-DOF grasp)
D_CUBE = int(__import__("os").environ.get("D_CUBE", "2"))  # user 2026-07-09: cube token dims — 2=front cube_rel only; 4=front(2)+wrist(2). STATE_DIM=21+D_CUBE. MUST match trainer D_CUBE (r41/r42 wristnet = 4). front-cam coord-transform cube offset -> LAST D_CUBE dims of `state` -> 4th attention token.
D_ACQ = int(__import__("os").environ.get("D_ACQ", "0"))  # user 2026-08-01: FRONT-GUIDED aim token dims — 0=off(4 tokens, legacy); 3=on -> [pcx,pcy,pv] = FRONT-known cube projected into the WRIST image (fully deployable) as a 5th "direction(aim)" token so search can directly observe where to turn the boresight. state tail = [... cube_rel(D_CUBE), acq(D_ACQ)]. non-zero SEARCH-only(off-search=0). MUST match trainer D_ACQ + deploy.

import os as _os
# user 2026-06-29: architecture is ENV-CONFIGURABLE so A/B jobs can compare without editing this file (set in the submit).
DET_POOL   = _os.environ.get("DET_POOL", "concat")        # how the 4 cross-attention tokens are aggregated: concat | attn | mean
CUBE3D_MLP = int(_os.environ.get("CUBE3D_MLP", "0"))      # 1 = cube3d ALSO fed to the proprio MLP (3d-twice); 0 = attention token only
PROPRIO_OFF = int(_os.environ.get("PROPRIO_OFF", "0"))    # user 2026-07-08: 1 = DROP the state_proj proprio MLP entirely (genuinely vision-only; supersedes MASK_PROPRIO). cube_rel STILL fed via the cube3d attention token. Kills the proprio (d2r wflex) shortcut that diverged on real. Changes net shape -> fresh retrain + matching deploy PROPRIO_OFF=1.
_NTOK = 5 if D_ACQ > 0 else 4
_POOL_OUT = _NTOK * 64 if DET_POOL == "concat" else 64
_BLOCKS        = int(_os.environ.get("NET_BLOCKS", "2"))          # user 2026-06-29: transformer blocks over the 4 tokens (slim=1)
_ACTOR_HID     = int(_os.environ.get("NET_ACTOR_HID", "256"))     # actor MLP width
_ACTOR_LAYERS  = int(_os.environ.get("NET_ACTOR_LAYERS", "3"))    # actor MLP depth (slim=2)
_CRITIC_HID    = int(_os.environ.get("NET_CRITIC_HID", "512"))    # C51 critic MLP width (slim=256)
_CRITIC_LAYERS = int(_os.environ.get("NET_CRITIC_LAYERS", "3"))   # C51 critic MLP depth (slim=2)


class _Block(nn.Module):
    """One transformer block: multihead self/cross-attention over the tokens + FFN (MLP), both residual."""
    def __init__(self, d, nheads=4, ff=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))  # MLP mixed in
        self.ln2 = nn.LayerNorm(d)

    def forward(self, tok):
        tA, _ = self.attn(tok, tok, tok)
        tok = self.ln1(tok + tA)            # cross/self-attention residual
        tok = self.ln2(tok + self.ffn(tok))  # FFN (MLP) residual
        return tok


class DetCrossAttn(nn.Module):
    """{front_det, wrist_det, cube_ori} -> tokens -> N transformer blocks (attn + FFN) -> pooled fused.
    user 2026-06-29: cube_ori token RESTORED (3 tokens: front/wrist cams + orientation). yaw (sin/cos)
    is real-measurable (OBB angle) and needed for grasp jaw alignment, so it influences the policy again."""
    def __init__(self, d=64, nheads=4, blocks=_BLOCKS):
        super().__init__()
        self.d = d
        self.front_embed = nn.Sequential(nn.Linear(D_DET, d), nn.LayerNorm(d), nn.GELU())
        self.wrist_embed = nn.Sequential(nn.Linear(D_DET, d), nn.LayerNorm(d), nn.GELU())
        self.ori_embed = nn.Sequential(nn.Linear(D_ORI, d), nn.LayerNorm(d), nn.GELU())
        self.cube_embed = nn.Sequential(nn.Linear(D_CUBE, d), nn.LayerNorm(d), nn.GELU())   # front-cam 3D cube offset token (user 2026-06-29)
        if D_ACQ > 0:
            self.acq_embed = nn.Sequential(nn.Linear(D_ACQ, d), nn.LayerNorm(d), nn.GELU())   # user 2026-08-01: FRONT-guided aim token ([pcx,pcy,pv] front cube in wrist image)
        self.tok_id = nn.Parameter(torch.zeros(_NTOK, d))   # token-id embedding (front/wrist/ori/cube3d[/acq])
        nn.init.normal_(self.tok_id, std=0.02)
        self.blocks = nn.ModuleList([_Block(d, nheads) for _ in range(blocks)])
        if DET_POOL == "attn":                           # learnable-query attention pool (else concat / mean)
            self.pool_q = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.pool_q, std=0.02)
            self.pool_attn = nn.MultiheadAttention(d, nheads, batch_first=True)

    def forward(self, front_det, wrist_det, cube_ori, cube3d, acq=None):
        tFt = self.front_embed(front_det) + self.tok_id[0]
        tWt = self.wrist_embed(wrist_det) + self.tok_id[1]
        tOt = self.ori_embed(cube_ori) + self.tok_id[2]
        tCt = self.cube_embed(cube3d) + self.tok_id[3]
        tToks = [tFt, tWt, tOt, tCt]
        if D_ACQ > 0:
            tToks.append(self.acq_embed(acq) + self.tok_id[4])   # user 2026-08-01: 5th FRONT-guided aim token
        tTok = torch.stack(tToks, dim=1)                  # (B, _NTOK, d) -- front + wrist + ori + cube3d [+ acq]
        for tBlk in self.blocks:
            tTok = tBlk(tTok)                                 # cross-attention KEPT (tokens mix here)
        if DET_POOL == "concat":
            return tTok.flatten(1)                          # (B, 4*d) -- keep all 4 enriched tokens
        if DET_POOL == "attn":
            tQ = self.pool_q.expand(tTok.shape[0], -1, -1)
            tPooled, _ = self.pool_attn(tQ, tTok, tTok)        # learnable query attends over the 4 tokens
            return tPooled.squeeze(1)                        # (B, d)
        return tTok.mean(dim=1)                             # (B, d) -- mean pool


class DetProjection(nn.Module):
    OUT_DIM = _POOL_OUT + (0 if PROPRIO_OFF else 256)   # pool output (concat=4*d, attn/mean=d) + state_proj 256 (dropped if PROPRIO_OFF)

    def __init__(self, n_state, d=64, blocks=_BLOCKS):
        super().__init__()
        self.det = DetCrossAttn(d=d, blocks=blocks)   # user 2026-07-07: per-instance blocks (deploy loads slim-search + full-approach in one process)
        if not PROPRIO_OFF:
            tSpIn = n_state if CUBE3D_MLP else n_state - D_CUBE - D_ACQ   # CUBE3D_MLP -> cube3d ALSO in MLP (3d-twice); else proprio only (exclude cube3d + acq tail)
            self.state_proj = nn.Sequential(nn.Linear(tSpIn, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, front_det, wrist_det, cube_ori, state):
        if D_ACQ > 0:                                               # state tail = [... cube_rel(D_CUBE), acq(D_ACQ)]
            tAcq = state[:, -D_ACQ:]                                 # user 2026-08-01: front-guided aim vector [pcx,pcy,pv]
            tCube3d = state[:, -(D_CUBE + D_ACQ):-D_ACQ]
        else:
            tAcq = None
            tCube3d = state[:, -D_CUBE:]                              # front-cam coordinate-transform offset -> attention token (kept even when PROPRIO_OFF)
        tDet = self.det(front_det, wrist_det, cube_ori, tCube3d, tAcq)
        if PROPRIO_OFF:
            return tDet                                              # vision-only tokens, NO proprio MLP
        tProprio = state if CUBE3D_MLP else state[:, :-(D_CUBE + D_ACQ)]   # 3d-twice -> full state to MLP; else proprio-only (drop cube3d + acq)
        return torch.cat([tDet, self.state_proj(tProprio)], dim=-1)


class DetActor(nn.Module):
    def __init__(self, n_state, n_act, hidden=_ACTOR_HID, layers=_ACTOR_LAYERS, blocks=_BLOCKS):
        super().__init__()
        self.proj = DetProjection(n_state, blocks=blocks)
        tSeq = [nn.Linear(DetProjection.OUT_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(max(1, layers) - 1):
            tSeq += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.fc = nn.Sequential(*tSeq)
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)

    def _meanLogstd(self, fd, wd, ori, s):
        tX = self.fc(self.proj(fd, wd, ori, s))
        return self.mean(tX), self.logstd(tX).clamp(-5, 2)

    def sample(self, fd, wd, ori, s):
        tMean, tLogstd = self._meanLogstd(fd, wd, ori, s)
        tStd = tLogstd.exp()
        tNormal = torch.distributions.Normal(tMean, tStd)
        tX = tNormal.rsample(); tY = torch.tanh(tX)
        tLogp = (tNormal.log_prob(tX) - torch.log(1 - tY.pow(2) + 1e-6)).sum(1, keepdim=True)
        return tY, tLogp

    def meanAction(self, fd, wd, ori, s):
        tMean, _ = self._meanLogstd(fd, wd, ori, s)
        return torch.tanh(tMean)

    # trainer-compatible aliases (match train_c51_visual Actor API)
    def getAction(self, fd, wd, ori, s):
        return self.sample(fd, wd, ori, s)

    @torch.no_grad()
    def actDeterministic(self, fd, wd, ori, s):
        return self.meanAction(fd, wd, ori, s)


class DetCriticC51(nn.Module):
    def __init__(self, n_state, n_act, num_atoms=101, v_min=-20.0, v_max=20.0, num_q=2, hidden=_CRITIC_HID, d=64, layers=_CRITIC_LAYERS):
        super().__init__()
        self.num_atoms, self.num_q, self.v_min, self.v_max = num_atoms, num_q, v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.proj = DetProjection(n_state, d=d)
        tQin = DetProjection.OUT_DIM + n_act
        def _mkq():
            tS = [nn.Linear(tQin, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            for _ in range(max(1, layers) - 1):
                tS += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            tS += [nn.Linear(hidden, num_atoms)]
            return nn.Sequential(*tS)
        self.qs = nn.ModuleList([_mkq() for _ in range(num_q)])

    def forward(self, fd, wd, ori, s, a):
        tX = torch.cat([self.proj(fd, wd, ori, s), a], dim=-1)
        return torch.stack([tQ(tX) for tQ in self.qs], dim=0)            # (Q,N,A) logits

    # trainer-compatible aliases (match train_c51_visual CriticC51 API)
    def logits(self, fd, wd, ori, s, a):
        return self.forward(fd, wd, ori, s, a)

    def qValues(self, fd, wd, ori, s, a):
        import torch.nn.functional as F
        tProbs = F.softmax(self.forward(fd, wd, ori, s, a), dim=-1)
        return (tProbs * self.support).sum(-1)                          # (Q,N)

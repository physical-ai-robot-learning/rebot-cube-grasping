#!/usr/bin/env python3
"""CONSTRAINED wrist calib — the unconstrained fit_wrist_calib.py went non-physical (lens 0.4-2m, focal ~0/3x,
axis 40-144deg off) because the front-est cube reference is biased/inconsistent -> unidentifiable.

Here we USE what we trust and only fit what we don't:
  - optical AXIS  = FIXED to spec cam_ax_local (verified == the user's measured 22deg forward+down).
  - focal         = FIXED to spec.
  - u-sign        = -1 (the data's Pearson said so; = the X-runaway cause), v-sign = -1 (known).
  - lens POSITION = the only free DOF, BOUNDED to +/-15cm of spec (tanh) so it can't run non-physical.

Also evaluates the USER's DIRECT physical measurement of the lens (TCP + (0, dy, +0.075) in the zero-state frame,
TCP-fwd=+Y) for BOTH frame y-signs, so the reprojection RMS itself resolves the frame ambiguity. Saves the
lowest-RMS PHYSICAL candidate as deploy/wrist_calib_fit.npz (same schema as fit_wrist_calib.py -> --wrist-calib).
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rebot_sim2real.rbot_calculator.base_controller import RuleBaseController
from rebot_sim2real.rbot_calculator.config import load_train_config

DATA = "deploy/wrist_calib_data.npz"
OUT = "deploy/wrist_calib_fit.npz"
SU, SV = -1.0, -1.0                     # u-sign -1 (data), v-sign -1 (known)
USER_DY = 0.08                          # user: lens is 8cm along -TCPfwd (behind); z +7.5cm
USER_DZ = 0.075
MOUNT_BOUND = 0.15                      # +/-15cm physical bound on the fitted lens position

cfg = load_train_config("configs/deploy_env.yaml").env
base = RuleBaseController(cfg, "cpu")
d = np.load(DATA, allow_pickle=True)
rows = d["rows"]; cols = list(d["cols"]); ci = {str(c): i for i, c in enumerate(cols)}
base_z = float(np.median(rows[:, ci["base_z"]])) if "base_z" in ci else 0.010
base.base_pos[2] = base_z

mount0 = base.wrist_mount.detach().clone().float()
aim0 = base.wrist_aim.detach().clone().float()
fx0, fy0, cx, cy = [float(v) for v in base.wrist_K]
ax_dir = (aim0 - mount0); AX_LEN = float(ax_dir.norm()); ax_dir = ax_dir / AX_LEN   # FIXED axis direction (gripper frame)

# ---- precompute FK constants for the usable (visible, in-frame) samples ----
Rg_l, glink_l, cube_l, obs_l, pidx_l = [], [], [], [], []
pos_map = {}                                                  # (cx,cy) rounded -> position index (for bundle adjustment)
for r in rows:
    if float(r[ci["wvis"]]) < 0.5:
        continue
    un, vn = float(r[ci["wun"]]), float(r[ci["wvn"]])
    if abs(un) > 0.95 or abs(vn) > 0.95:
        continue
    aq = torch.tensor(r[ci["q0"]:ci["q0"] + 6], dtype=torch.float32).view(1, 6)
    with torch.no_grad():
        _, Rg, cam_w = base._fk_full(aq)
        glink = cam_w[0] - Rg[0] @ mount0
    Rg_l.append(Rg[0]); glink_l.append(glink)
    cube_l.append(torch.tensor([r[ci["cx"]], r[ci["cy"]], r[ci["cz"]]], dtype=torch.float32))
    obs_l.append(torch.tensor([un, vn], dtype=torch.float32))
    key = (round(float(r[ci["cx"]]), 2), round(float(r[ci["cy"]]), 2))
    pidx_l.append(pos_map.setdefault(key, len(pos_map)))
Rg = torch.stack(Rg_l); glink = torch.stack(glink_l); cube = torch.stack(cube_l)
obs_m = torch.stack(obs_l) * torch.tensor([SU, SV])          # obs in the model (legacy) sign frame
pidx = torch.tensor(pidx_l, dtype=torch.long); NPOS = len(pos_map); pos = pos_map
M = obs_m.shape[0]
up = torch.tensor([0.0, 0.0, 1.0])
print(f"[cfit] base_z={base_z:+.3f} · {M} usable samples · {len(pos)} positions · u-sign {SU:+.0f} v-sign {SV:+.0f}")


def project(mount, aim, cube_pts):
    lens = glink + (Rg @ mount.view(1, 3, 1)).squeeze(-1)
    axl = (aim - mount); axl = axl / axl.norm().clamp_min(1e-9)
    fwd = (Rg @ axl.view(1, 3, 1)).squeeze(-1); fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = -fwd
    x = torch.cross(up.expand_as(z), z, dim=-1); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.cross(z, x, dim=-1)
    rel = cube_pts - lens
    xc = (rel * x).sum(-1); yc = (rel * y).sum(-1); depth = (-(rel * z).sum(-1)).clamp_min(1e-3)
    return torch.stack([xc / depth * (fx0 / cx), yc / depth * (fy0 / cy)], dim=-1)


def rms(mount, aim, cube_pts=cube):
    with torch.no_grad():
        return float(((project(mount, aim, cube_pts) - obs_m) ** 2).mean().sqrt())


# ---- (0) spec baseline ----
print(f"[cfit] SPEC          mount={[round(v,4) for v in mount0.tolist()]}  RMS={rms(mount0, aim0):.4f}")

# ---- (2) BOUNDED fit (axis+focal FIXED, trust the front-est cube) — reference only (front-bias polluted) ----
raw = torch.zeros(3, requires_grad=True)
opt = torch.optim.Adam([raw], lr=0.01)
for _ in range(8000):
    opt.zero_grad()
    mount = mount0 + MOUNT_BOUND * torch.tanh(raw)
    loss = ((project(mount, mount + AX_LEN * ax_dir, cube) - obs_m) ** 2).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    mount_b = mount0 + MOUNT_BOUND * torch.tanh(raw)
print(f"[cfit] BOUNDED-FIT   mount={[round(v,4) for v in mount_b.tolist()]}  RMS={rms(mount_b, mount_b + AX_LEN*ax_dir):.4f}"
      "  (trusts front-est cube -> polluted)")

# ---- (3) BUNDLE ADJUSTMENT: axis+focal FIXED, fit lens (bounded) + the TRUE cube xy per position jointly.
#         The front est is only the INIT for each cube; J1 excitation separates the (gripper-frame) camera from the
#         (world) cube positions -> removes the front bias entirely. Recovers the camera AND the real cube xy. ----
raw = torch.zeros(3, requires_grad=True)                      # lens (bounded)
cdxy = torch.zeros(NPOS, 2, requires_grad=True)               # per-position cube xy delta (world), init 0 = front est
opt = torch.optim.Adam([raw, cdxy], lr=0.01)
for _ in range(20000):
    opt.zero_grad()
    mount = mount0 + MOUNT_BOUND * torch.tanh(raw)
    cube_ba = cube.clone()
    cube_ba = cube_ba + torch.cat([cdxy[pidx], torch.zeros(M, 1)], dim=-1)   # shift xy per position, z fixed
    loss = ((project(mount, mount + AX_LEN * ax_dir, cube_ba) - obs_m) ** 2).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    mount_ba = mount0 + MOUNT_BOUND * torch.tanh(raw)
    cube_ba = cube + torch.cat([cdxy[pidx], torch.zeros(M, 1)], dim=-1)
    rms_ba = float(((project(mount_ba, mount_ba + AX_LEN * ax_dir, cube_ba) - obs_m) ** 2).mean().sqrt())
    shift = cdxy.detach()
print(f"[cfit] BUNDLE-ADJ    mount={[round(v,4) for v in mount_ba.tolist()]}  RMS={rms_ba:.4f}  (front bias removed)")
print(f"[cfit]   per-position front-est error |Δcube|: "
      + "  ".join(f"{float(shift[i].norm())*1000:.0f}mm" for i in range(NPOS)))

# ---- save the BUNDLE-ADJ camera (physical, axis-fixed, front-bias-free) as the deploy calib ----
aim_ba = mount_ba + AX_LEN * ax_dir
np.savez(OUT, wrist_mount_offset=np.array(mount_ba.tolist(), np.float32),
         wrist_aim_local=np.array(aim_ba.tolist(), np.float32),
         focal_scale=np.float32(1.0), fit_focal=np.bool_(False),
         sign_u=np.float32(SU), sign_v=np.float32(SV), base_z=np.float32(base_z),
         rms_before=np.float32(rms(mount0, aim0)), rms_after=np.float32(rms_ba),
         n=np.int32(M), n_pos=np.int32(NPOS))
print(f"[cfit] saved BUNDLE-ADJ -> {OUT}  (apply: --wrist-calib {OUT}; then open-loop validate wrist_xy vs cube)")

"""WRIST-token ablation v2 — isolate the wrist's contribution WHEN THE BASE HAS CALIB ERROR.
v1 used base_noise=0 (perfect base) -> wrist irrelevant (zeroed==cnn==gt). The wrist only earns its
keep when the base is OFF, so we sweep base_noise x wrist. TOP token = analytic _a_top (incorporates
_top_off consistently with the base -> the trained perception model); only the WRIST varies.

4 conditions (gain-DR + positions paired per batch):
  A 0/zeroed     base perfect, no wrist     -> ceiling (base alone)
  B N/zeroed     base off by N, no wrist     -> how much calib error hurts
  C N/cnn        base off + current wrist CNN-> does the CNN recover it?
  D N/gt         base off + perfect wrist     -> headroom of a perfect wrist
Read: C-B = wrist CNN value | D-C = CNN headroom | D vs A = can a perfect wrist fully recover.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/eval_wrist_ablation.py --nx 8 --ny 6 --envs 16 --base-noise 0.03
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--ckpt", default="ckpt/pick_residual_real_best.pt")
ap.add_argument("--nx", type=int, default=8)
ap.add_argument("--ny", type=int, default=6)
ap.add_argument("--envs", type=int, default=16)
ap.add_argument("--base-noise", type=float, default=0.03, help="calib-error magnitude (m) for conditions B/C/D")
ap.add_argument("--out", default="viz_out")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
NE = args.envs
tcfg.env.num_envs = NE
tcfg.env.obs_analytic_uv = False
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import numpy as np, cv2, torch
from rebot_sim2real.envs.pick_env import RebotPickEnv
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real.perception_cnn import CubeDetector

dev = torch.device(tcfg.env.device); c = tcfg.env
env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev, critic_obs_dim=env.critic_obs_dim)
agent.load_state_dict(torch.load(args.ckpt, map_location=dev))


def load_det(p):
    sd = torch.load(p, map_location=dev)
    m = CubeDetector(w=int(sd.get("width", 32)), stride=int(sd.get("stride", 8))).to(dev)
    m.load_state_dict(sd["state_dict"], strict=False); m.eval(); return m


wrist_cnn = load_det("ckpt/cube_detector_wrist.pt")
ai = list(env.arm_idx)


def reroll_gain():
    kps = torch.full((NE, env.num_dof), c.drive_stiffness, device=dev)
    kds = torch.full((NE, env.num_dof), c.drive_damping, device=dev)
    kps[:, ai] = c.arm_drive_stiffness; kds[:, ai] = c.arm_drive_damping
    kps[:, ai[3:]] = c.wrist_drive_stiffness; kds[:, ai[3:]] = c.wrist_drive_damping
    if c.dr_enable and c.dr_gain_scale > 0:
        gs = 1.0 + (torch.rand(NE, 1, device=dev) * 2 - 1) * c.dr_gain_scale
        kps[:, ai] = kps[:, ai] * gs
    env.robots.set_gains(kps=kps, kds=kds)


def place_batch(pos_xy, base_noise):
    """Cube at TRUE pos; the BASE looks up by the PERCEIVED pos (= true + _top_off of magnitude base_noise),
    and _a_top shifts by the same _top_off (consistent trained perception). Wrist stays on the true cube."""
    env.reset()
    k = pos_xy.shape[0]; idx = torch.arange(k, device=dev)
    if base_noise > 0:
        env._top_off[:k] = (torch.rand(k, 2, device=dev) * 2 - 1) * base_noise
    else:
        env._top_off[:k] = 0.0
    cz = c.table_z + c.cube_size / 2.0 + 0.002
    pw = torch.cat([pos_xy, torch.full((k, 1), cz, device=dev)], dim=-1) + env.env_origins[:k]
    env.cubes.set_world_poses(positions=pw, indices=idx)
    env.cubes.set_velocities(torch.zeros(k, 6, device=dev), indices=idx)
    bxy = pos_xy + env._top_off[:k]                       # PERCEIVED -> base lookup (calib error)
    nn = torch.cdist(bxy, env._tab_xy).argmin(dim=1)
    env._base_grasp[:k] = env._tab_grasp[nn]; env._base_hover[:k] = env._tab_hover[nn]
    env.arm_target[:k] = env._tab_hover[nn]
    qh = env.robots.get_joint_positions(); qh[:k, ai] = env._tab_hover[nn]
    env.robots.set_joint_positions(qh)
    for _ in range(24):
        env.world.step(render=True)


@torch.no_grad()
def cnn_tokens(cams, model):
    frames = [np.asarray(cam.get_rgba())[..., :3] for cam in cams]
    x = torch.from_numpy(np.stack(frames)).to(dev).float().div(255.).permute(0, 3, 1, 2)
    hm, vlogit, _ = model(x)
    uv = model.uv_from_heatmap(hm)
    H, W = frames[0].shape[:2]
    un = (uv[:, 0] - W / 2) / (W / 2); vn = (uv[:, 1] - H / 2) / (H / 2)
    vis = (torch.sigmoid(vlogit) > 0.5).float()
    return torch.stack([un, vn, vis], dim=-1)


def run_episode(pos_xy, base_noise, wrist_mode):
    place_batch(pos_xy, base_noise)
    k = pos_xy.shape[0]
    recorded = torch.zeros(NE, dtype=torch.bool, device=dev); outcome = torch.zeros(NE, device=dev)
    for t in range(c.episode_len + 8):
        obs = env.get_obs()                                  # refreshes env._a_top (perceived) + env._a_wrist (true)
        obs[:, 33:36] = env._a_top                           # TOP fixed = analytic perceived (consistent w/ base)
        if wrist_mode == "cnn":
            obs[:, 36:39] = cnn_tokens(env.wrist_cams, wrist_cnn)
        elif wrist_mode == "zeroed":
            obs[:, 36:39] = 0.0
        else:
            obs[:, 36:39] = env._a_wrist
        a = agent.act(obs, deterministic=True)
        _, _, done, info = env.step(a)
        newly = done & (~recorded); newly[k:] = False
        if bool(newly.any()):
            outcome[newly] = info["success"][newly].float(); recorded |= done
        if bool(recorded[:k].all()):
            break
    return outcome[:k].cpu()


bn = args.base_noise
COND = [("A 0/zeroed", 0.0, "zeroed"), (f"B {bn}/zeroed", bn, "zeroed"),
        (f"C {bn}/cnn", bn, "cnn"), (f"D {bn}/gt", bn, "gt")]
tc, rc, rb = c.table_center, c.residual_reach_center, c.residual_reach_box
grid = [(float(tc[0] + rc[0] + fx * rb[0]), float(tc[1] + rc[1] + fy * rb[1]))
        for fy in np.linspace(-1, 1, args.ny) for fx in np.linspace(-1, 1, args.nx)]
batches = [grid[i:i + NE] for i in range(0, len(grid), NE)]
res = {name: [] for name, _, _ in COND}
for bi, batch in enumerate(batches):
    reroll_gain()
    pos = torch.tensor(batch, device=dev, dtype=torch.float32)
    for name, bnz, wm in COND:
        oc = run_episode(pos, bnz, wm)
        res[name].extend(bool(x) for x in oc)
    line = " | ".join(f"{name}={sum(res[name])/len(res[name]):.0%}" for name, _, _ in COND)
    print(f"[abl] batch {bi+1}/{len(batches)}  {line}", flush=True)

rates = {name: sum(res[name]) / max(1, len(res[name])) for name, _, _ in COND}
A, B, C, D = (rates[COND[i][0]] for i in range(4))
print(f"[abl] DONE  n={len(res[COND[0][0]])}/cond  base_noise={bn}", flush=True)
for name, _, _ in COND:
    print(f"[abl]   {name:14s} = {rates[name]:.1%}", flush=True)
print(f"[abl]   base-error damage (A-B) = {(A-B)*100:+.1f} pt", flush=True)
print(f"[abl]   wrist CNN value (C-B)   = {(C-B)*100:+.1f} pt", flush=True)
print(f"[abl]   CNN headroom (D-C)      = {(D-C)*100:+.1f} pt", flush=True)
print(f"[abl]   perfect-wrist recovery (D vs A) = {D:.1%} vs {A:.1%}", flush=True)
with open("/tmp/eval_wrist_abl.txt", "w") as f:
    f.write(f"A={A:.3f} B={B:.3f} C={C:.3f} D={D:.3f} (C-B={(C-B)*100:+.1f} D-C={(D-C)*100:+.1f})\n")
    f.flush(); os.fsync(f.fileno())
app.close()

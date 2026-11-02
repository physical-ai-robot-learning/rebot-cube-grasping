"""Batched deploy-faithful eval of the residual-vision policy over a DENSE cube-position grid (~100).
top+wrist CNN in-loop; gain-DR RE-ROLLED per batch so EVERY position gets a fresh (gain,tcp_off) draw
(removes the num_envs=1 single-draw bias that made the 12-cell viz unreliable as an absolute rate).
No video -> fast. Outputs overall success rate + a dense success scatter map (success_grid100.png).

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/eval_grid100.py --nx 10 --ny 10 --envs 20
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--ckpt", default="ckpt/pick_residual_real_best.pt")
ap.add_argument("--nx", type=int, default=10)
ap.add_argument("--ny", type=int, default=10)
ap.add_argument("--envs", type=int, default=20, help="parallel envs/batch (x2 cams must stay < ~64)")
ap.add_argument("--out", default="viz_out")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
NE = args.envs
tcfg.env.num_envs = NE
tcfg.env.obs_analytic_uv = False                 # RENDER top+wrist (CNN in-loop)
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


top_cnn = load_det("ckpt/cube_detector_sharp.pt")
wrist_cnn = load_det("ckpt/cube_detector_wrist.pt")
ai = list(env.arm_idx)


def reroll_gain():
    """Re-sample the per-env arm-stiffness DR (init samples it ONCE -> fixed; re-roll => fresh draw/batch)."""
    kps = torch.full((NE, env.num_dof), c.drive_stiffness, device=dev)
    kds = torch.full((NE, env.num_dof), c.drive_damping, device=dev)
    kps[:, ai] = c.arm_drive_stiffness; kds[:, ai] = c.arm_drive_damping
    kps[:, ai[3:]] = c.wrist_drive_stiffness; kds[:, ai[3:]] = c.wrist_drive_damping
    if c.dr_enable and c.dr_gain_scale > 0:
        gs = 1.0 + (torch.rand(NE, 1, device=dev) * 2 - 1) * c.dr_gain_scale
        kps[:, ai] = kps[:, ai] * gs
    env.robots.set_gains(kps=kps, kds=kds)


def place_batch(pos_xy):
    """Override the first k cubes to env-local (gx,gy), NN-look-up the IK base, settle at hover. top_off=0."""
    env.reset()
    k = pos_xy.shape[0]; idx = torch.arange(k, device=dev)
    env._top_off[:] = 0.0
    cz = c.table_z + c.cube_size / 2.0 + 0.002
    pw = torch.cat([pos_xy, torch.full((k, 1), cz, device=dev)], dim=-1) + env.env_origins[:k]
    env.cubes.set_world_poses(positions=pw, indices=idx)
    env.cubes.set_velocities(torch.zeros(k, 6, device=dev), indices=idx)
    nn = torch.cdist(pos_xy, env._tab_xy).argmin(dim=1)
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


tc, rc, rb = c.table_center, c.residual_reach_center, c.residual_reach_box
grid = [(float(tc[0] + rc[0] + fx * rb[0]), float(tc[1] + rc[1] + fy * rb[1]))
        for fy in np.linspace(-1, 1, args.ny) for fx in np.linspace(-1, 1, args.nx)]
results = []
batches = [grid[i:i + NE] for i in range(0, len(grid), NE)]
for bi, batch in enumerate(batches):
    reroll_gain()
    pos = torch.tensor(batch, device=dev, dtype=torch.float32)
    place_batch(pos)
    k = len(batch)
    recorded = torch.zeros(NE, dtype=torch.bool, device=dev)
    outcome = torch.zeros(NE, device=dev)
    for t in range(c.episode_len + 8):           # cap > episode_len so every env hits a done (timeout worst case)
        obs = env.get_obs()
        obs[:, 33:36] = cnn_tokens(env.top_cams, top_cnn)
        obs[:, 36:39] = cnn_tokens(env.wrist_cams, wrist_cnn)
        a = agent.act(obs, deterministic=True)
        _, _, done, info = env.step(a)
        newly = done & (~recorded); newly[k:] = False
        if bool(newly.any()):
            outcome[newly] = info["success"][newly].float()
            recorded |= done
        if bool(recorded[:k].all()):
            break
    for j in range(k):
        results.append(((batch[j][0] - tc[0], batch[j][1] - tc[1]), bool(outcome[j])))
    nsucc = sum(s for _, s in results)
    print(f"[eval] batch {bi+1}/{len(batches)}  running {nsucc}/{len(results)} = {nsucc/len(results):.0%}", flush=True)

sr = sum(s for _, s in results) / max(1, len(results))
os.makedirs(args.out, exist_ok=True)
gimg = np.full((460, 480, 3), 35, np.uint8)
x0, x1 = rc[0] - rb[0], rc[0] + rb[0]; y0, y1 = rc[1] - rb[1], rc[1] + rb[1]
mapx = lambda x: int(70 + (x - x0) / (x1 - x0 + 1e-9) * 360)
mapy = lambda y: int(415 - (y - y0) / (y1 - y0 + 1e-9) * 330)
for (dx, dy), s in results:
    cv2.circle(gimg, (mapx(dx), mapy(dy)), 11, (0, 200, 0) if s else (0, 0, 220), -1)
cv2.putText(gimg, f"residual-vision (top+wrist CNN in-loop, gain-DR re-rolled/cell)", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(gimg, f"success {sr:.0%} ({sum(s for _,s in results)}/{len(results)})  green=ok red=fail",
            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(gimg, "cube xy rel. table center | +x right, +y up (back)", (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(args.out, "success_grid100.png"), gimg)
print(f"[eval] DONE  {len(results)} positions  success {sr:.1%}  -> {os.path.abspath(args.out)}/success_grid100.png", flush=True)
with open("/tmp/eval_grid100.txt", "w") as f:
    f.write(f"success={sr:.3f} n={len(results)}\n"); f.flush(); os.fsync(f.fileno())
app.close()

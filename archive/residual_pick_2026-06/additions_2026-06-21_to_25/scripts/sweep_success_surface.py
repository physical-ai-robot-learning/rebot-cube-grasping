"""Dense success-vs-cube-XY sweep -> a precise success-rate SURFACE. Deploy-faithful (top+wrist CNN
in-loop, base perfect = top_off 0). Each grid cell is sampled `reps` times with FRESH gain-DR + tcp_off
draws (round-robin: round r does all positions, so coverage is full early and refines each round).
Results stream to a CSV after every batch -> crash-resumable (re-launch continues; plot from partial too).

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/sweep_success_surface.py --nx 20 --ny 20 --reps 6 --envs 20
"""
import os, sys, argparse, csv
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--ckpt", default="ckpt/pick_residual_real_best.pt")
ap.add_argument("--nx", type=int, default=20)
ap.add_argument("--ny", type=int, default=20)
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--envs", type=int, default=20)
ap.add_argument("--spacing", type=float, default=0.0, help="m between cells, centered on workspace; 0 = fit reach-box")
ap.add_argument("--analytic", action="store_true",
                help="ANALYTIC (u,v) tokens, no cameras/render -> high-NE, ~10x faster (justified: ablation showed "
                     "wrist contributes 0 + top CNN ~= analytic). Absolute ~+10pt vs rendered deploy; SHAPE faithful.")
ap.add_argument("--base-noise", type=float, default=0.0, help="calib error (m); 0 = policy's intrinsic position map")
ap.add_argument("--csv", default="viz_out/surface/surface.csv")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
NE = args.envs
tcfg.env.num_envs = NE
tcfg.env.obs_analytic_uv = args.analytic
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import numpy as np, torch
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


top_cnn = wrist_cnn = None
if not args.analytic:
    top_cnn = load_det("ckpt/cube_detector_sharp.pt")
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
    bxy = pos_xy + env._top_off[:k]
    nn = torch.cdist(bxy, env._tab_xy).argmin(dim=1)
    env._base_grasp[:k] = env._tab_grasp[nn]; env._base_hover[:k] = env._tab_hover[nn]
    env.arm_target[:k] = env._tab_hover[nn]
    qh = env.robots.get_joint_positions(); qh[:k, ai] = env._tab_hover[nn]
    env.robots.set_joint_positions(qh)
    for _ in range(24):
        env.world.step(render=not args.analytic)


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


def run_batch(pos_xy):
    place_batch(pos_xy, args.base_noise)
    k = pos_xy.shape[0]
    recorded = torch.zeros(NE, dtype=torch.bool, device=dev); outcome = torch.zeros(NE, device=dev)
    for t in range(c.episode_len + 8):
        obs = env.get_obs()
        if not args.analytic:                                # analytic: env.get_obs already has _a_top/_a_wrist
            obs[:, 33:36] = cnn_tokens(env.top_cams, top_cnn)
            obs[:, 36:39] = cnn_tokens(env.wrist_cams, wrist_cnn)
        a = agent.act(obs, deterministic=True)
        _, _, done, info = env.step(a)
        newly = done & (~recorded); newly[k:] = False
        if bool(newly.any()):
            outcome[newly] = info["success"][newly].float(); recorded |= done
        if bool(recorded[:k].all()):
            break
    return outcome[:k].cpu().numpy()


tc, rc, rb = c.table_center, c.residual_reach_center, c.residual_reach_box
if args.spacing > 0:                                          # fixed spacing, centered on the workspace center (rc)
    gx0, gy0 = tc[0] + rc[0], tc[1] + rc[1]
    xsr = [gx0 + (i - (args.nx - 1) / 2.0) * args.spacing for i in range(args.nx)]
    ysr = [gy0 + (j - (args.ny - 1) / 2.0) * args.spacing for j in range(args.ny)]
    grid = [(float(x), float(y)) for y in ysr for x in xsr]
else:
    grid = [(float(tc[0] + rc[0] + fx * rb[0]), float(tc[1] + rc[1] + fy * rb[1]))
            for fy in np.linspace(-1, 1, args.ny) for fx in np.linspace(-1, 1, args.nx)]
P = len(grid)
tasks = [(r * P + pi, pi) for r in range(args.reps) for pi in range(P)]   # round-robin -> full coverage early

os.makedirs(os.path.dirname(args.csv), exist_ok=True)
done = set()
if os.path.exists(args.csv):
    with open(args.csv) as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                done.add(int(row[0]))
todo = [(tid, pi) for tid, pi in tasks if tid not in done]
print(f"[sweep] grid {args.nx}x{args.ny}={P}  reps={args.reps}  total={len(tasks)}  done={len(done)}  todo={len(todo)}", flush=True)

newfile = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
fcsv = open(args.csv, "a", newline="")
wr = csv.writer(fcsv)
if newfile:
    wr.writerow(["task_id", "pos_idx", "x_rel", "y_rel", "success"]); fcsv.flush()

for bstart in range(0, len(todo), NE):
    chunk = todo[bstart:bstart + NE]
    reroll_gain()
    pos = torch.tensor([grid[pi] for _, pi in chunk], device=dev, dtype=torch.float32)
    oc = run_batch(pos)
    for (tid, pi), s in zip(chunk, oc):
        wr.writerow([tid, pi, round(grid[pi][0] - tc[0], 4), round(grid[pi][1] - tc[1], 4), int(s)])
    fcsv.flush(); os.fsync(fcsv.fileno())
    ndone = len(done) + bstart + len(chunk)
    print(f"[sweep] {ndone}/{len(tasks)} ({ndone/len(tasks):.0%})  last-batch succ={oc.mean():.0%}", flush=True)

fcsv.close()
print(f"[sweep] DONE all {len(tasks)} tasks -> {os.path.abspath(args.csv)}", flush=True)
app.close()

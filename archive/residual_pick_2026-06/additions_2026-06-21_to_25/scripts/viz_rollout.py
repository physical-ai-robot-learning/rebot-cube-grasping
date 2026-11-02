"""VISUALIZE the Phase-3 residual-vision policy: rollout across diverse cube positions with the REAL
top/wrist CNNs in the loop, overlaying the CNN heatmap + detected (u,v) on each camera view, plus an
action/phase/grip flow panel. Saves one MP4 + GIF per episode and a success-vs-position grid.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/viz_rollout.py --n 9 --out viz_out
"""
import os, sys, argparse, glob, re
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--ckpt", default="ckpt/pick_residual_real_best.pt")
ap.add_argument("--nx", type=int, default=4, help="grid cols (cube x)")
ap.add_argument("--ny", type=int, default=3, help="grid rows (cube y)")
ap.add_argument("--out", default="viz_out")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
tcfg.env.num_envs = 1
tcfg.env.obs_analytic_uv = False         # RENDER top+wrist (need frames)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import numpy as np
import cv2
import imageio
import torch
from rebot_sim2real.envs.pick_env import RebotPickEnv
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real.perception_cnn import CubeDetector

dev = torch.device(tcfg.env.device)
env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
env.pos_rand = 1.0                        # full reachable box -> diverse positions
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev, critic_obs_dim=env.critic_obs_dim)
agent.load_state_dict(torch.load(args.ckpt, map_location=dev))


def load_det(path):
    sd = torch.load(path, map_location=dev)
    m = CubeDetector(w=int(sd.get("width", 32)), stride=int(sd.get("stride", 8))).to(dev)
    m.load_state_dict(sd["state_dict"], strict=False); m.eval(); return m


top_cnn = load_det("ckpt/cube_detector_sharp.pt")
wrist_cnn = load_det("ckpt/cube_detector_wrist.pt")
TOP, WR = slice(33, 36), slice(36, 39)
DW, DH = 480, 270                          # display size per view


@torch.no_grad()
def cnn_overlay(frame_u8, model):
    """frame HxWx3 uint8 -> (display BGR with heatmap+uv, (un,vn,vis))."""
    H, W = frame_u8.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(frame_u8)).to(dev).float().div(255.).permute(2, 0, 1)[None]
    hm, vlogit, _ = model(x)
    uv = model.uv_from_heatmap(hm)[0]
    u, v = float(uv[0]), float(uv[1]); vis = float(torch.sigmoid(vlogit)[0])
    hmap = hm[0, 0].cpu().numpy()
    hmap = (hmap - hmap.min()) / (hmap.ptp() + 1e-6)
    hmap = cv2.resize((hmap * 255).astype(np.uint8), (W, H))
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
    disp = cv2.addWeighted(bgr, 0.6, hmap, 0.4, 0)
    if vis > 0.5:
        cv2.circle(disp, (int(u), int(v)), 6, (255, 255, 255), 2)
        cv2.circle(disp, (int(u), int(v)), 2, (0, 0, 255), -1)
    disp = cv2.resize(disp, (DW, DH))
    un = (u - W / 2) / (W / 2); vn = (v - H / 2) / (H / 2)
    return disp, (un, vn, 1.0 if vis > 0.5 else 0.0)


def panel(phase, res_norm, grip_open, tcp_z, step, succ, cube):
    img = np.full((DH, 360, 3), 30, np.uint8)
    def put(y, t, c=(230, 230, 230)): cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    phases = ["APPROACH", "DESCEND", "GRASP", "LIFT"]
    for i, p in enumerate(phases):
        c = (0, 220, 0) if p == phase else (90, 90, 90)
        cv2.putText(img, ("> " if p == phase else "  ") + p, (12, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    put(150, f"residual |a| : {res_norm:.2f}")
    put(176, f"grip         : {'OPEN' if grip_open else 'CLOSED'}", (0,180,255) if grip_open else (0,120,255))
    put(202, f"tcp_z        : {tcp_z*100:.1f} cm")
    put(228, f"step {step}   cube({cube[0]:+.2f},{cube[1]:+.2f})")
    put(256, f"SUCCESS" if succ else "...", (0, 230, 0) if succ else (160,160,160))
    return img


os.makedirs(args.out, exist_ok=True)
tc = tcfg.env.table_center; rc = tcfg.env.residual_reach_center; rb = tcfg.env.residual_reach_box
cz = tcfg.env.table_z + tcfg.env.cube_size / 2.0 + 0.002
ix = torch.tensor([0], device=dev)
c = tcfg.env; ai = list(env.arm_idx)


def reroll_gain():
    """Fresh per-cell arm-stiffness DR draw (num_envs=1 fixes ONE gain for all cells -> biased pattern;
    re-roll => the 100 videos' success/fail match the per-cell-DR 79% map)."""
    kps = torch.full((env.num_envs, env.num_dof), c.drive_stiffness, device=dev)
    kds = torch.full((env.num_envs, env.num_dof), c.drive_damping, device=dev)
    kps[:, ai] = c.arm_drive_stiffness; kds[:, ai] = c.arm_drive_damping
    kps[:, ai[3:]] = c.wrist_drive_stiffness; kds[:, ai[3:]] = c.wrist_drive_damping
    if c.dr_enable and c.dr_gain_scale > 0:
        gs = 1.0 + (torch.rand(env.num_envs, 1, device=dev) * 2 - 1) * c.dr_gain_scale
        kps[:, ai] = kps[:, ai] * gs
    env.robots.set_gains(kps=kps, kds=kds)


def place_grid(gx, gy):
    """Override cube to env-local (gx,gy), re-look-up IK base. top_off=0 -> base AND top token both = true
    top-CNN detection (deploy-faithful: one top CNN feeds both); wrist CNN does the fine correction."""
    reroll_gain()
    env.reset()
    env._top_off[0] = 0.0
    pw = torch.tensor([[gx, gy, cz]], device=dev) + env.env_origins[0:1]
    env.cubes.set_world_poses(positions=pw, indices=ix)
    env.cubes.set_velocities(torch.zeros(1, 6, device=dev), indices=ix)
    nn = torch.cdist(torch.tensor([[gx, gy]], device=dev), env._tab_xy).argmin(dim=1)
    env._base_grasp[0:1] = env._tab_grasp[nn]; env._base_hover[0:1] = env._tab_hover[nn]
    env.arm_target[0:1] = env._tab_hover[nn]
    qh = env.robots.get_joint_positions(); qh[0, env.arm_idx] = env._tab_hover[nn][0]
    env.robots.set_joint_positions(qh)
    for _ in range(24):                    # prime cameras + FULLY settle arm at new hover (6 was a transient -> premature grip)
        env.world.step(render=True)


grid = [(float(tc[0] + rc[0] + fx * rb[0]), float(tc[1] + rc[1] + fy * rb[1]))
        for fy in np.linspace(-1, 1, args.ny) for fx in np.linspace(-1, 1, args.nx)]
for ep, (gx, gy) in enumerate(grid):
    if glob.glob(os.path.join(args.out, f"ep{ep}_*.mp4")):
        print(f"[viz] ep{ep} exists -> skip (resume)", flush=True)
        continue
    place_grid(gx, gy)
    cube_rel = (gx - tc[0], gy - tc[1])
    frames, succ = [], False
    for t in range(tcfg.env.episode_len):
        tf = np.asarray(env.top_cams[0].get_rgba())[..., :3].astype(np.uint8)
        wf = np.asarray(env.wrist_cams[0].get_rgba())[..., :3].astype(np.uint8)
        top_disp, top_tok = cnn_overlay(tf, top_cnn)
        wr_disp, wr_tok = cnn_overlay(wf, wrist_cnn)
        obs = env.get_obs()
        obs[0, TOP] = torch.tensor(top_tok, device=dev); obs[0, WR] = torch.tensor(wr_tok, device=dev)
        a = agent.act(obs, deterministic=True)
        # phase derivation
        d = min(t / max(1, tcfg.env.residual_descend_steps), 1.0)
        engaged = bool(env.grip_engaged[0]); grip_open = float(env.grip_target[0]) > 0.02
        tcp_z = float(env._last_tcp_z[0])
        phase = "LIFT" if engaged else ("GRASP" if (not grip_open) else ("DESCEND" if d > 0.15 else "APPROACH"))
        res_norm = float(a[0, :6].abs().mean())
        cv2.putText(top_disp, "TOP cam + CNN heatmap", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(wr_disp, "WRIST cam + CNN heatmap", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        comp = np.hstack([top_disp, wr_disp, panel(phase, res_norm, grip_open, tcp_z, t, succ, cube_rel)])
        frames.append(comp)                  # BGR
        _, _, done, info = env.step(a)
        if bool(info["success"][0]) or bool(env.ep_success[0]):
            succ = True
        if bool(done[0]):
            break
    tag = f"ep{ep}_x{cube_rel[0]:+.2f}_y{cube_rel[1]:+.2f}_{'SUCCESS' if succ else 'fail'}"
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(args.out, tag + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    imageio.mimsave(os.path.join(args.out, tag + ".gif"),
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[::2]], fps=10)
    print(f"[viz] ep{ep} {tag}  ({len(frames)} frames)", flush=True)

# success-vs-position map REBUILT from ALL rendered files (resume-safe -> complete even after a restart)
results = []
for fp in sorted(glob.glob(os.path.join(args.out, "ep*_x*_y*.mp4"))):
    m = re.search(r"_x([-+][\d.]+)_y([-+][\d.]+)_(SUCCESS|fail)", os.path.basename(fp))
    if m:
        results.append(((float(m.group(1)), float(m.group(2))), m.group(3) == "SUCCESS"))
sr = sum(s for _, s in results) / max(1, len(results))
gimg = np.full((460, 480, 3), 35, np.uint8)
x0, x1 = rc[0] - rb[0], rc[0] + rb[0]
y0, y1 = rc[1] - rb[1], rc[1] + rb[1]
mapx = lambda x: int(70 + (x - x0) / (x1 - x0 + 1e-9) * 360)
mapy = lambda y: int(415 - (y - y0) / (y1 - y0 + 1e-9) * 330)   # +y up
for (dx, dy), s in results:
    cv2.circle(gimg, (mapx(dx), mapy(dy)), 11, (0, 200, 0) if s else (0, 0, 220), -1)
cv2.putText(gimg, f"viz grid: success {sr:.0%} ({sum(s for _,s in results)}/{len(results)})  green=ok red=fail",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(gimg, "per-cell gain-DR re-roll | cube xy rel. table center | +x right +y up", (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(args.out, "success_grid.png"), gimg)
print(f"[viz] DONE  success {sr:.0%} ({sum(s for _,s in results)}/{len(results)})  -> {os.path.abspath(args.out)}", flush=True)
with open("/tmp/viz_rollout.txt", "w") as f:
    f.write(f"success={sr:.3f} n={len(results)} out={os.path.abspath(args.out)}\n"); f.flush(); os.fsync(f.fileno())
app.close()

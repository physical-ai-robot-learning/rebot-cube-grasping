"""AUTONOMY + VLA-LOGGING harness — the sim2real platform's core: run the trained pick policy in sim,
autonomously across N episodes (auto-reset + success label), and log each episode as a VLA-ready sample:
  observation.images.top   (RGB video, mp4)   <- the overhead view a VLA sees
  observation.images.wrist (RGB video, mp4)   <- the eye-in-hand view
  observation.state         [T, 8]  arm_qpos(6)+grip_qpos(2)   <- deployable proprio
  action                    [T, 7]  arm_target(6)+grip_target(1)  <- absolute joint command (ACT/VLA convention)
  language_instruction      e.g. "pick up the red cube"
  success                   bool (per episode)

LeRobot-COMPATIBLE layout (keys + dirs match LeRobot v2) but dependency-free (npz + cv2 mp4 + json),
so it converts cleanly to a LeRobot/RLDS dataset later. This same logger is the template the REAL-robot
deploy loop will reuse — sim generates the dataset now; real swaps the cameras/robot in.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/collect_vla_dataset.py --episodes 40 --out data/vla_pick_v0
"""
import os, sys, json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--ckpt", default="ckpt/pick_residual_real_best.pt")
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--out", default="data/vla_pick_v0")
ap.add_argument("--instruction", default="pick up the red cube")
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--success-only", action="store_true", help="keep ONLY successful episodes (clean demo set)")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
tcfg.env.num_envs = 1
tcfg.env.obs_analytic_uv = False                 # RENDER top+wrist (the VLA's vision modality)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import numpy as np, cv2, torch
from rebot_sim2real.envs.pick_env import RebotPickEnv
from rebot_sim2real.rl.sac import SAC

dev = torch.device(tcfg.env.device); c = tcfg.env
env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
env.pos_rand = 1.0                               # full reachable box -> diverse cube positions
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev, critic_obs_dim=env.critic_obs_dim)
agent.load_state_dict(torch.load(args.ckpt, map_location=dev))

OUT = args.out
for d in ["meta", "data", "videos/observation.images.top", "videos/observation.images.wrist"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)


def rgb(cam):
    return np.ascontiguousarray(np.asarray(cam.get_rgba())[..., :3].astype(np.uint8))


def vwriter(path, w, h):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))


episodes_meta, kept, n_succ = [], 0, 0
for ep in range(args.episodes):
    env.reset()
    for _ in range(6):
        env.world.step(render=True)              # prime cameras
    cp, _ = env.cubes.get_world_poses()
    cube_rel = (cp - env.env_origins)[0, :2].cpu().numpy(); tc = c.table_center
    states, actions, frames_top, frames_wrist = [], [], [], []
    succ = False
    for t in range(c.episode_len):
        ftop, fwr = rgb(env.top_cams[0]), rgb(env.wrist_cams[0])
        state = env.robots.get_joint_positions()[0].detach().cpu().numpy().copy()   # qpos8
        obs = env.get_obs()
        a = agent.act(obs, deterministic=True)
        _, _, done, info = env.step(a)
        action = torch.cat([env.arm_target[0], env.grip_target[0]], dim=-1).detach().cpu().numpy()  # [7] commanded target
        frames_top.append(ftop); frames_wrist.append(fwr)
        states.append(state.astype(np.float32)); actions.append(action.astype(np.float32))
        if bool(info["success"][0]) or bool(env.ep_success[0]):
            succ = True
        if bool(done[0]):
            break
    if succ:
        n_succ += 1
    if args.success_only and not succ:
        print(f"[vla] ep{ep} fail -> skipped (success-only)", flush=True); continue

    idx = kept
    H, W = frames_top[0].shape[:2]; Hw, Ww = frames_wrist[0].shape[:2]
    vt = vwriter(os.path.join(OUT, f"videos/observation.images.top/episode_{idx:06d}.mp4"), W, H)
    vw = vwriter(os.path.join(OUT, f"videos/observation.images.wrist/episode_{idx:06d}.mp4"), Ww, Hw)
    for f in frames_top:
        vt.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    for f in frames_wrist:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vt.release(); vw.release()
    T = len(states)
    np.savez_compressed(os.path.join(OUT, f"data/episode_{idx:06d}.npz"),
                        **{"observation.state": np.stack(states), "action": np.stack(actions),
                           "timestamp": (np.arange(T) / args.fps).astype(np.float32),
                           "frame_index": np.arange(T, dtype=np.int64)})
    episodes_meta.append({"episode_index": idx, "length": T, "task_index": 0,
                          "task": args.instruction, "success": bool(succ),
                          "cube_xy_rel": [float(cube_rel[0] - tc[0]), float(cube_rel[1] - tc[1])]})
    kept += 1
    print(f"[vla] ep{ep} -> episode_{idx:06d}  T={T}  success={succ}", flush=True)

# ---- dataset meta (LeRobot-compatible) ----
with open(os.path.join(OUT, "meta/episodes.jsonl"), "w") as f:
    for e in episodes_meta:
        f.write(json.dumps(e) + "\n")
with open(os.path.join(OUT, "meta/tasks.jsonl"), "w") as f:
    f.write(json.dumps({"task_index": 0, "task": args.instruction}) + "\n")
info = {
    "codebase": "rebot_sim2real (sim2real platform) — dependency-free LeRobot-compatible dump",
    "robot_type": "rebot_b601_dm", "fps": args.fps, "total_episodes": kept,
    "total_frames": int(sum(e["length"] for e in episodes_meta)),
    "success_rate": round(n_succ / max(1, args.episodes), 3),
    "features": {
        "observation.images.top": {"dtype": "video", "shape": [H, W, 3]},
        "observation.images.wrist": {"dtype": "video", "shape": [Hw, Ww, 3]},
        "observation.state": {"dtype": "float32", "shape": [8], "names": ["j1", "j2", "j3", "j4", "j5", "j6", "grip1", "grip2"]},
        "action": {"dtype": "float32", "shape": [7], "names": ["t1", "t2", "t3", "t4", "t5", "t6", "grip"]},
    },
    "language_instruction": args.instruction,
    "note": "videos under videos/<feature>/episode_NNNNNN.mp4; per-frame state/action in data/episode_NNNNNN.npz. "
            "Convert to a LeRobot v2 dataset by writing meta/info.json schema + parquet from the npz (keys already match).",
}
with open(os.path.join(OUT, "meta/info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(f"[vla] DONE  kept {kept}/{args.episodes} episodes  success_rate={info['success_rate']:.0%}  -> {os.path.abspath(OUT)}", flush=True)
with open("/tmp/collect_vla.txt", "w") as f:
    f.write(f"kept={kept} success_rate={info['success_rate']:.3f} out={os.path.abspath(OUT)}\n"); f.flush(); os.fsync(f.fileno())
app.close()

"""Roll out a trained C51 policy WITH cameras and save successful episodes as combined-view mp4s.

Training is state-only (no cameras), so this re-runs the policy in a camera-enabled env, detects
successful episodes (is_fold_success), and writes each as a single video with front|iso|wrist views
stacked side-by-side.

  PYTHONPATH=src .venv/bin/python -m rebot_sim2real.tools.render_successes \
      --checkpoint rl_output/rl_c51_smooth/c51_ckpt_XXXX.pt --num-envs 16 --num-videos 6 \
      --out-dir rl_output/success_videos
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=16)
ap.add_argument("--num-videos", type=int, default=6)
ap.add_argument("--max-steps", type=int, default=2000)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--out-dir", type=str, default="rl_output/success_videos")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--config", type=str, default="configs/reward_config.yaml")
args, _ = ap.parse_known_args()

import yaml
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f) or {}
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.maniskill_rewards import is_fold_success
import imageio

DEVICE = "cuda:0"
N = args.num_envs

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=args.target_cube)

# grasp contact sensors (for is_fold_success)
_t = getattr(env_cfg.scene, args.target_cube); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

# cameras: front + iso (on base_link) + wrist (env_config already put it on gripper_link/fp_camera)
_spawn = sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
env_cfg.scene.front = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.9, 0.55), rot=(0.6533, 0.2706, 0.2706, 0.6533), convention="ros"),
    data_types=["rgb"], spawn=_spawn, width=240, height=180, update_period=0)
env_cfg.scene.iso = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link/iso_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.7, 0.7, 0.6), rot=(0.5, 0.27, 0.4, 0.72), convention="ros"),
    data_types=["rgb"], spawn=_spawn, width=240, height=180, update_period=0)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)

cams = {}
for nm in ["front", "iso", "wrist"]:
    try:
        cams[nm] = env.scene[nm]
    except Exception:
        pass
print(f"[render] cameras: {list(cams.keys())}", flush=True)


# ---- policy (matches train_c51 Actor) ----
class Actor(nn.Module):
    def __init__(self, n_state=43, n_act=7, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)

    @torch.no_grad()
    def act(self, obs):
        return torch.tanh(self.mean(self.fc(self.proj(obs))))


actor = Actor().to(DEVICE)
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
actor.load_state_dict(ckpt["actor"] if "actor" in ckpt else ckpt)
actor.eval()
print(f"[render] loaded {args.checkpoint} (step {ckpt.get('step','?')})", flush=True)

os.makedirs(args.out_dir, exist_ok=True)


def grab(nm):
    """(N, H, W, 3) uint8 for camera nm, or None."""
    c = cams.get(nm)
    if c is None:
        return None
    rgb = c.data.output.get("rgb")
    if rgb is None:
        return None
    a = rgb[..., :3].detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a


def combine(i):
    """side-by-side front|iso|wrist for env i, padded to equal height."""
    tiles = []
    for nm in ["front", "iso", "wrist"]:
        f = _frames_now.get(nm)
        if f is None:
            continue
        img = f[i]
        tiles.append(img)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] != h:
            pad = np.zeros((h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        out.append(t)
        out.append(np.zeros((h, 4, 3), dtype=np.uint8))  # 4px separator
    return np.hstack(out[:-1])


obs_dict, _ = env.reset()
obs = obs_dict["policy"]
buffers = [[] for _ in range(N)]          # per-env combined frames for the current episode
succ_flag = torch.zeros(N, dtype=torch.bool, device=DEVICE)
saved = 0
_frames_now = {}

for step in range(args.max_steps):
    if saved >= args.num_videos:
        break
    with torch.no_grad():
        action = actor.act(obs)
    obs_dict, _, terminated, truncated, _ = env.step(action)
    dones = (terminated | truncated)

    # success latch (pre-reset state is valid here in ManagerBasedRLEnv.step? evaluate now)
    succ_flag |= is_fold_success(env, SceneEntityCfg(args.target_cube))

    # capture frames for all envs this step
    _frames_now = {nm: grab(nm) for nm in ["front", "iso", "wrist"]}
    for i in range(N):
        fr = combine(i)
        if fr is not None:
            buffers[i].append(fr)

    obs = obs_dict["policy"]

    # on episode end, save successful ones
    if dones.any():
        for i in torch.nonzero(dones).flatten().tolist():
            if succ_flag[i] and saved < args.num_videos and len(buffers[i]) > 5:
                path = os.path.join(args.out_dir, f"success_{saved:02d}_env{i}_step{step}.mp4")
                try:
                    imageio.mimsave(path, buffers[i], fps=args.fps, macro_block_size=1)
                    print(f"[render] saved {path} ({len(buffers[i])} frames)", flush=True)
                    saved += 1
                except Exception as e:
                    print(f"[render] save error: {e}", flush=True)
            buffers[i] = []
            succ_flag[i] = False
    if step % 50 == 0:
        print(f"[render] step {step}, saved {saved}/{args.num_videos}", flush=True)

print(f"[render] DONE: saved {saved} success videos to {args.out_dir}", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

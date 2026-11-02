"""Render the state-only TEACHER's successful episodes using OUR two cameras (configure_visual):
front (90cm/55cm/45deg head-on) + wrist (between-the-jaws), side-by-side mp4. No DR (clean scene).

  PYTHONPATH=src python -m rebot_sim2real.tools.render_teacher_2cam \
      --checkpoint rl_output/rl_c51_95/c51_final.pt --num-videos 6 --out-dir rl_output/teacher95_2cam
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
ap.add_argument("--max-steps", type=int, default=600)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--out-dir", type=str, default="rl_output/teacher95_2cam")
ap.add_argument("--fps", type=int, default=10)
args, _ = ap.parse_known_args()

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import configure_visual
from rebot_sim2real.env.maniskill_rewards import is_fold_success
import imageio

DEVICE = "cuda:0"; N = args.num_envs; TARGET = args.target_cube

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_t = getattr(env_cfg.scene, TARGET); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
# OUR cameras (front 90/55/45 + wrist between-jaws), keep 43D obs for the state-only teacher. No DR.
env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=False)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
cams = {nm: env.scene[nm] for nm in ("front", "wrist") if nm in env.scene.keys()}
print(f"[r2] cameras: {list(cams.keys())}", flush=True)


class Actor(nn.Module):   # state-only teacher (matches train_c51.py / visual_nets.StateOnlyActor)
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
ck = torch.load(args.checkpoint, map_location=DEVICE)
actor.load_state_dict(ck["actor"] if "actor" in ck else ck)
actor.eval()
print(f"[r2] loaded {args.checkpoint} (step {ck.get('step','?')})", flush=True)
os.makedirs(args.out_dir, exist_ok=True)


def grab(nm):
    rgb = cams[nm].data.output.get("rgb")
    if rgb is None:
        return None
    a = rgb[..., :3].detach().cpu().numpy()
    return a if a.dtype == np.uint8 else (a * 255).clip(0, 255).astype(np.uint8)


def combine(i, fr):
    tiles = [fr[nm][i] for nm in ("front", "wrist") if fr.get(nm) is not None]
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] != h:
            t = np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
        out.append(t); out.append(np.zeros((h, 6, 3), np.uint8))
    return np.hstack(out[:-1])


obs_dict, _ = env.reset()
obs = obs_dict["policy"]
buffers = [[] for _ in range(N)]
succ_flag = torch.zeros(N, dtype=torch.bool, device=DEVICE)
saved = 0
for step in range(args.max_steps):
    if saved >= args.num_videos:
        break
    with torch.no_grad():
        action = actor.act(obs)
    obs_dict, _, term, trunc, _ = env.step(action)
    dones = (term | trunc)
    succ_flag |= is_fold_success(env, SceneEntityCfg(TARGET))
    fr = {nm: grab(nm) for nm in cams}
    for i in range(N):
        c = combine(i, fr)
        if c is not None:
            buffers[i].append(c)
    obs = obs_dict["policy"]
    if dones.any():
        for i in torch.nonzero(dones).flatten().tolist():
            if succ_flag[i] and saved < args.num_videos and len(buffers[i]) > 5:
                p = os.path.join(args.out_dir, f"teacher_success_{saved:02d}_env{i}.mp4")
                try:
                    imageio.mimsave(p, buffers[i], fps=args.fps, macro_block_size=1)
                    print(f"[r2] saved {p} ({len(buffers[i])} frames)", flush=True); saved += 1
                except Exception as e:
                    print(f"[r2] save err: {e}", flush=True)
            buffers[i] = []; succ_flag[i] = False
    if step % 50 == 0:
        print(f"[r2] step {step}, saved {saved}/{args.num_videos}", flush=True)

print(f"[r2] DONE saved {saved} to {args.out_dir}", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

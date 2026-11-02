"""ULTRA-HD 60fps render of the state-only TEACHER's successful grasps, OUR two cameras
(front 90/55/45 + wrist between-jaws) side-by-side.

High quality = cameras rendered at --res (default 720). Smooth = capture EVERY physics sub-step
(decimation sub-steps per 10Hz control step => ~100 frames/sim-sec) by replicating env.step's
decimation loop, then save at --fps (default 60).

  PYTHONPATH=src python -m rebot_sim2real.tools.render_teacher_hq \
      --checkpoint rl_output/rl_c51_95/c51_final.pt --res 720 --fps 60 --num-videos 4
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--num-videos", type=int, default=4)
ap.add_argument("--episode-steps", type=int, default=50)
ap.add_argument("--res", type=int, default=720)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--out-dir", type=str, default="rl_output/teacher95_hq")
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
env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=False)
# ULTRA-HD: override both cameras to high resolution
for nm in ("front", "wrist"):
    c = getattr(env_cfg.scene, nm, None)
    if c is not None:
        c.width = args.res; c.height = args.res
# render every physics sub-step (decimation loop renders when counter % render_interval == 0)
env_cfg.sim.render_interval = 1

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
cams = {nm: env.scene[nm] for nm in ("front", "wrist") if nm in env.scene.keys()}
decim = env.cfg.decimation
print(f"[hq] cameras={list(cams.keys())} res={args.res} decimation={decim} fps={args.fps}", flush=True)


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
ck = torch.load(args.checkpoint, map_location=DEVICE)
actor.load_state_dict(ck["actor"] if "actor" in ck else ck)
actor.eval()
print(f"[hq] loaded {args.checkpoint} (step {ck.get('step','?')})", flush=True)
os.makedirs(args.out_dir, exist_ok=True)


def grab(nm):
    rgb = cams[nm].data.output.get("rgb")
    if rgb is None:
        return None
    a = rgb[..., :3].detach().cpu().numpy()
    return a if a.dtype == np.uint8 else (a * 255).clip(0, 255).astype(np.uint8)


def combine(i, fr):
    # fixed-size front|wrist (black tile if a cam is momentarily None) -> every frame identical WxH
    R = args.res
    sep = 8                                         # even separator
    tiles = []
    for nm in ("front", "wrist"):
        t = fr.get(nm)
        tiles.append(t[i] if t is not None else np.zeros((R, R, 3), np.uint8))
    out = np.hstack([tiles[0], np.zeros((R, sep, 3), np.uint8), tiles[1]])
    H, W = out.shape[:2]                            # ensure EVEN dims (libx264 yuv420p requirement)
    if W % 2:
        out = np.pad(out, ((0, 0), (0, 1), (0, 0)))
    if H % 2:
        out = np.pad(out, ((0, 1), (0, 0), (0, 0)))
    return out


def substep():
    """one physics sub-step + render (replicates env.step's inner loop body)."""
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.sim.render()
    env.scene.update(dt=env.physics_dt)


saved = 0
attempt = 0
while saved < args.num_videos and attempt < 4:
    attempt += 1
    obs = env.reset()[0]["policy"]
    env.action_manager.process_action(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))
    for _ in range(3):                          # settle + populate camera buffers
        substep()
    buffers = [[] for _ in range(N)]
    succ = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    for cstep in range(args.episode_steps):
        with torch.no_grad():
            action = actor.act(obs)
        env.action_manager.process_action(action.to(DEVICE))
        for _ in range(decim):                  # capture EVERY sub-step -> smooth
            substep()
            fr = {nm: grab(nm) for nm in cams}
            for i in range(N):
                c = combine(i, fr)
                if c is not None:
                    buffers[i].append(c)
        obs = env.observation_manager.compute()["policy"]
        succ |= is_fold_success(env, SceneEntityCfg(TARGET))
        if cstep % 10 == 0:
            print(f"[hq] attempt {attempt} cstep {cstep} succ={int(succ.sum())}", flush=True)
    for i in torch.nonzero(succ).flatten().tolist():
        if saved >= args.num_videos:
            break
        p = os.path.join(args.out_dir, f"teacher_hq_{saved:02d}_env{i}.mp4")
        try:
            imageio.mimsave(p, buffers[i], fps=args.fps, macro_block_size=1,
                            codec="libx264", output_params=["-pix_fmt", "yuv420p", "-crf", "16"])
            print(f"[hq] saved {p} ({len(buffers[i])} frames @ {args.fps}fps, {buffers[i][0].shape})", flush=True)
            saved += 1
        except Exception as e:
            print(f"[hq] save err: {e}", flush=True)

print(f"[hq] DONE saved {saved} to {args.out_dir}", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

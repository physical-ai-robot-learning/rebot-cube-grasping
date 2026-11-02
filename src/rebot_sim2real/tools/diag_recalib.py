"""Verify the recalibration: (a) cubes spawn fully-mixed left/center/right, (b) FRONT is head-on to
the arm, (c) WRIST offset (raise it ABOVE the gripper). Renders an 8-env grid for front + wrist and
prints each env's target-cube (x,y) at reset so the left/center/right spread is visible numerically.

The wrist offset can be nudged from the command line to find the "above the wrist" pose without
editing code each time (--wrist-dx / --wrist-dz add to WRIST_OFFSET_POS; --wrist-pos sets it fully):

  PYTHONPATH=src python -m rebot_sim2real.tools.diag_recalib \
      --checkpoint rl_output/rl_c51_95/c51_final.pt --wrist-dz 0.06 --tag up06
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
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--grab-step", type=int, default=20)   # capture frame at this control step (mid-grasp)
ap.add_argument("--wrist-dx", type=float, default=0.0)
ap.add_argument("--wrist-dz", type=float, default=0.0)
ap.add_argument("--wrist-pos", type=str, default="")   # "x,y,z" overrides WRIST_OFFSET_POS fully
ap.add_argument("--wrist-look-deg", type=float, default=-1.0,   # >=0: aim this many deg BELOW the
                help="re-aim: look this many degrees below the +x (forward) horizontal in link frame")
ap.add_argument("--wrist-aim-tcp", action="store_true",         # aim from the eye at the grasp center
                help="re-aim the wrist cam at the link-frame TCP/grasp center (overrides --wrist-look-deg)")
ap.add_argument("--wrist-tcp", type=str, default="-0.086,0,0.103")  # grasp center in gripper_link frame
ap.add_argument("--wrist-roll180", action="store_true", help="apply 180deg optical-axis roll (upright)")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--tag", type=str, default="cur")
ap.add_argument("--out-dir", type=str, default="rl_output/recalib_diag")
args, _ = ap.parse_known_args()

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env import visual_obs as VO
from rebot_sim2real.env.visual_obs import configure_visual
import imageio

DEVICE = "cuda:0"; N = args.num_envs; TARGET = "cube"

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

# Optionally nudge the wrist camera offset to test "above the gripper" poses.
import math
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_mul
wp = list(VO.WRIST_OFFSET_POS)
if args.wrist_pos:
    wp = [float(v) for v in args.wrist_pos.split(",")]
wp[0] += args.wrist_dx; wp[2] += args.wrist_dz
wrot = VO._wrist_offset_rot()
tgt = None
if args.wrist_aim_tcp:                                    # aim at the grasp center (link-frame TCP)
    tgt = tuple(float(v) for v in args.wrist_tcp.split(","))
elif args.wrist_look_deg >= 0:                            # re-aim: look N deg below +x (forward) horizontal
    a = math.radians(args.wrist_look_deg)
    tgt = (wp[0] + math.cos(a), wp[1], wp[2] - math.sin(a))
if tgt is not None:
    wrot = VO._lookat_quat_opengl(tuple(wp), tgt)
    if args.wrist_roll180:                                # 180deg about optical (local Z) axis -> upright
        import torch as _t
        q = quat_mul(_t.tensor([list(wrot)]), _t.tensor([[0.0, 0.0, 0.0, 1.0]]))
        wrot = tuple(q[0].tolist())
if getattr(env_cfg.scene, "wrist", None) is not None:
    env_cfg.scene.wrist.offset = TiledCameraCfg.OffsetCfg(
        pos=tuple(wp), rot=wrot, convention="opengl")
print(f"[diag] wrist offset pos={tuple(round(v,3) for v in wp)} rot={tuple(round(v,3) for v in wrot)} "
      f"look_deg={args.wrist_look_deg} roll180={args.wrist_roll180}", flush=True)

for nm in ("front", "wrist"):
    c = getattr(env_cfg.scene, nm, None)
    if c is not None:
        c.width = args.res; c.height = args.res

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
cams = {nm: env.scene[nm] for nm in ("front", "wrist") if nm in env.scene.keys()}
print(f"[diag] cameras={list(cams.keys())} res={args.res}", flush=True)


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
os.makedirs(args.out_dir, exist_ok=True)


def grab(nm):
    rgb = cams[nm].data.output.get("rgb")
    if rgb is None:
        return None
    a = rgb[..., :3].detach().cpu().numpy()
    return a if a.dtype == np.uint8 else (a * 255).clip(0, 255).astype(np.uint8)


def grid(frames):  # list of (H,W,3) -> 2 rows x ceil(N/2)
    R = args.res
    cols = (len(frames) + 1) // 2
    canvas = np.zeros((2 * R + 4, cols * R + (cols - 1) * 4, 3), np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        canvas[r * (R + 4):r * (R + 4) + R, c * (R + 4):c * (R + 4) + R] = f
    return canvas


obs = env.reset()[0]["policy"]
# report the target cube (x,y) in env-local frame -> confirms left/center/right mixing
cube = env.scene[TARGET]
loc = (cube.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu().numpy()
print("[diag] target-cube (x,y) per env  (base x=0.35; <0.35 = one side, >0.35 = other):", flush=True)
for i in range(N):
    side = "L" if loc[i, 0] < 0.33 else ("R" if loc[i, 0] > 0.37 else "C")
    print(f"   env{i}: x={loc[i,0]:+.3f} y={loc[i,1]:+.3f}  [{side}]", flush=True)

front_cap = wrist_cap = None
for step in range(args.steps):
    with torch.no_grad():
        action = actor.act(obs)
    obs = env.step(action)[0]["policy"]
    if step == args.grab_step:
        fr = {nm: grab(nm) for nm in cams}
        if fr.get("front") is not None:
            front_cap = grid([fr["front"][i] for i in range(N)])
        if fr.get("wrist") is not None:
            wrist_cap = grid([fr["wrist"][i] for i in range(N)])

if front_cap is not None:
    p = os.path.join(args.out_dir, f"front_grid_{args.tag}.png"); imageio.imwrite(p, front_cap)
    print(f"[diag] saved {p}", flush=True)
if wrist_cap is not None:
    p = os.path.join(args.out_dir, f"wrist_grid_{args.tag}.png"); imageio.imwrite(p, wrist_cap)
    print(f"[diag] saved {p}", flush=True)
print("[diag] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

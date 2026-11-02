"""#5 deploy enabler: emit the sim camera SPEC + a verification render set, so the deploy team can
overlay their real downsampled images 1:1. Renders front + wrist at REST with the target cube at a
fixed reference, saves full-res + 128 downsample PNGs, and prints intrinsics/extrinsics/preprocessing.
  PYTHONPATH=src python -m rebot_sim2real.tools.camera_spec_render --out-dir rl_output/cam_spec
"""
import argparse, os
import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=str, default="rl_output/cam_spec")
ap.add_argument("--cube-xy", type=str, default="0.25,-0.30")   # reference target-cube position (env frame)
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
from rebot_sim2real.env import visual_obs as VO
from rebot_sim2real.env.visual_obs import configure_visual, downsample
import imageio

DEVICE = "cuda:0"
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
_t = getattr(env_cfg.scene, "cube"); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube="cube", keep_privileged=True, dr=False)  # clean real cameras

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
env.reset()
# place the target cube at the reference; push distractors aside; hold REST (zero action)
cx, cy = (float(v) for v in args.cube_xy.split(","))
org = env.scene.env_origins[0]
def place(name, x, y):
    a = env.scene[name]
    z = a.data.default_root_state[0, 2]
    pose = torch.tensor([[org[0]+x, org[1]+y, org[2]+z, 1.0, 0.0, 0.0, 0.0]], device=DEVICE)
    a.write_root_pose_to_sim(pose); a.write_root_velocity_to_sim(torch.zeros((1,6), device=DEVICE))
for _ in range(6):
    place("cube", cx, cy); place("cube_blue", 0.50, -0.34); place("cube_purple", 0.20, -0.36)
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=DEVICE))

os.makedirs(args.out_dir, exist_ok=True)
def grab(nm):
    a = env.scene[nm].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    return a if a.dtype == np.uint8 else (a*255).clip(0,255).astype(np.uint8)
for nm, size in (("front", VO.FRONT_SIZE), ("wrist", VO.WRIST_SIZE)):
    full = grab(nm)
    ds = downsample(torch.as_tensor(full, device=DEVICE).unsqueeze(0), size)[0].cpu().numpy()
    imageio.imwrite(os.path.join(args.out_dir, f"{nm}_full.png"), full)
    imageio.imwrite(os.path.join(args.out_dir, f"{nm}_ds{size}.png"), ds)
    print(f"[spec] saved {nm}_full.png ({full.shape}) + {nm}_ds{size}.png ({ds.shape})", flush=True)

print("\n========== SIM CAMERA SPEC (match real rig to these) ==========", flush=True)
print(f"render res (native) : {VO.IMG_RENDER}x{VO.IMG_RENDER}  -> AREA-downsample to front {VO.FRONT_SIZE}, wrist {VO.WRIST_SIZE}", flush=True)
print(f"color / norm        : RGB uint8; encoder does (img/255 - 0.5)", flush=True)
print(f"FRONT intrinsics    : PinholeCamera focal_length={VO.FRONT_FOCAL} (horiz_aperture 20.955 default)", flush=True)
print(f"FRONT extrinsics    : EYE(env frame)={VO.FRONT_EYE}  look-at TARGET={VO.FRONT_TARGET}  (opengl, baked)", flush=True)
print(f"                      = ~90cm forward of arm base (y), 55cm high, ~45deg down, HEAD-ON (x=0.35 centerline)", flush=True)
print(f"WRIST intrinsics    : PinholeCamera focal_length={VO.WRIST_FOCAL}", flush=True)
print(f"WRIST extrinsics    : gripper_link-local OFFSET pos={VO.WRIST_OFFSET_POS}  rot(wxyz,opengl)={tuple(round(v,5) for v in VO._wrist_offset_rot())}", flush=True)
print(f"                      = AIM at {VO.WRIST_AIM_TARGET} (fwd+down at grasp) + 180deg roll (upright)", flush=True)
print(f"                      gripper_link axes at rest: +x->world+y(fwd) +y->world+x(side) +z->world-z(down)", flush=True)
print(f"reference cube       : target 'cube' at env-frame ({cx},{cy}); robot at REST (zero action)", flush=True)
print("[spec] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

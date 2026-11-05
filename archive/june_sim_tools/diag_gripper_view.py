"""External close-up of the gripper at REST from several angles, so we can SEE the small square
camera-mount on top of the gripper and read off where to bolt the wrist camera. Adds env-level
TiledCameras (baked look-at) aimed at the gripper rest world pos (~0.35,-0.40,0.21).
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_gripper_view
"""
import os
import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env
from rebot_sim2real.env.visual_obs import _lookat_quat_opengl
import imageio

DEVICE = "cuda:0"
GRIP = (0.35, -0.40, 0.21)   # gripper rest world pos (from diag_gripper_mounts)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
_t = getattr(env_cfg.scene, "cube"); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

# external inspection cameras (eye -> look at gripper), baked look-at, zoomed in
VIEWS = {
    "topback": (0.35, -0.20, 0.45),   # from above + robot side -> sees gripper TOP face
    "side":    (0.08, -0.40, 0.26),   # from +x side -> profile
    "front":   (0.35, -0.66, 0.30),   # from -y (where fingers reach) -> face-on
}
for nm, eye in VIEWS.items():
    setattr(env_cfg.scene, f"insp_{nm}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/insp_" + nm,
        offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=_lookat_quat_opengl(eye, GRIP), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=35.0, clipping_range=(0.01, 50.0), lock_camera=True),
        width=480, height=480, update_period=0))

env = ManagerBasedRLEnv(cfg=env_cfg)
from rebot_sim2real.env.env_config import apply_motor_limits
apply_motor_limits(env, {})
env.reset()
for _ in range(4):   # settle + populate camera buffers
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=DEVICE))

os.makedirs("rl_output/recalib_diag", exist_ok=True)
for nm in VIEWS:
    cam = env.scene[f"insp_{nm}"]
    rgb = cam.data.output.get("rgb")
    if rgb is None:
        print(f"[gv] {nm}: no rgb", flush=True); continue
    a = rgb[0, ..., :3].detach().cpu().numpy()
    a = a if a.dtype == np.uint8 else (a * 255).clip(0, 255).astype(np.uint8)
    p = f"rl_output/recalib_diag/gripper_{nm}.png"
    imageio.imwrite(p, a); print(f"[gv] saved {p}", flush=True)
print("[gv] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

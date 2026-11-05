"""Render multiple FRONT + WRIST camera candidates in one boot, native 128, for the user to pick.

FRONT candidates: env-level cameras with baked world-frame look-at (vary yaw / lateral position).
WRIST candidates: gripper_link cameras with baked LOCAL-frame look-at toward the TCP (vary the
                  'above' direction) to find the "straight down from above the gripper" view.

Saves rl_output/visual_diag/cam_<name>_128.png (env 0).
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_cam_sweep2
"""
import os
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits

DEVICE = "cuda:0"
N = 4
TARGET = "cube"
TCP = (-0.086, 0.0, 0.103)   # grasp center in gripper_link local frame


def quat(eye, tgt):
    e = torch.tensor([list(eye)], dtype=torch.float32)
    t = torch.tensor([list(tgt)], dtype=torch.float32)
    return tuple(quat_from_matrix(create_rotation_matrix_from_view(e, t, "Z", device="cpu"))[0].tolist())


# FRONT candidates (env-frame eye, target). ~90cm fwd / 55cm high / ~45deg down; vary yaw + lateral.
FRONT = {
    "F1_workspace": ((0.25, 0.26, 0.55), (0.25, -0.30, 0.06)),   # straight -y, on workspace center
    "F2_robotline": ((0.35, 0.26, 0.55), (0.35, -0.30, 0.06)),   # straight -y, on robot centerline
    "F3_yaw_plusx": ((0.20, 0.26, 0.55), (0.30, -0.30, 0.06)),   # yaw toward +x
    "F4_yaw_minusx":((0.30, 0.26, 0.55), (0.20, -0.30, 0.06)),   # yaw toward -x
}
# WRIST candidates (gripper_link LOCAL eye, looking at TCP_local). Vary the 'above' direction.
WRIST = {
    "W1_localZ":  ((TCP[0], TCP[1], TCP[2] + 0.13), TCP),        # above along local +z
    "W2_localY":  ((TCP[0], TCP[1] + 0.13, TCP[2]), TCP),        # above along local +y
    "W3_negX":    ((TCP[0] - 0.13, TCP[1], TCP[2] + 0.04), TCP),  # behind along local -x
    "W4_posX":    ((TCP[0] + 0.13, TCP[1], TCP[2] + 0.04), TCP),  # along local +x
}

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
# disable the default front/side so they don't clutter; keep env_config's wrist for reference
for nm in ("front", "side"):
    if hasattr(env_cfg.scene, nm):
        setattr(env_cfg.scene, nm, None)

_spawn = sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
for name, (eye, tgt) in FRONT.items():
    setattr(env_cfg.scene, f"f_{name}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/f_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=quat(eye, tgt), convention="opengl"),
        data_types=["rgb"], spawn=_spawn, width=128, height=128, update_period=0))
for name, (eye, tgt) in WRIST.items():
    setattr(env_cfg.scene, f"w_{name}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/w_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=quat(eye, tgt), convention="opengl"),
        data_types=["rgb"], spawn=_spawn, width=128, height=128, update_period=0))

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
obs_dict, _ = env.reset()
for _ in range(6):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

eo = env.scene.env_origins
print(f"[sweep2] env0 cube local = {(env.scene[TARGET].data.root_pos_w - eo)[0].cpu().numpy().round(3)}", flush=True)

os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
for name in list(FRONT) + list(WRIST):
    key = f"f_{name}" if name in FRONT else f"w_{name}"
    r0 = env.scene[key].data.output["rgb"][0]
    imageio.imwrite(f"rl_output/visual_diag/cam_{name}_128.png", r0[..., :3].cpu().numpy().astype("uint8"))
    print(f"[sweep2] {name}: range[{int(r0[...,:3].min())},{int(r0[...,:3].max())}]", flush=True)

print("[sweep2] saved cam_* to rl_output/visual_diag/", flush=True)
env.close(); app.close()

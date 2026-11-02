"""Compute the gripper_link-LOCAL camera offset for "directly above the gripper looking straight down"
at the grasp (TCP). Prints OffsetCfg(pos, rot opengl) to bake into the wrist camera.

Reads the gripper_link world pose at rest, places a camera 15cm above the TCP in WORLD-up, looking
(near-)straight down at the TCP, and converts that world pose into the gripper_link local frame so it
stays rigidly mounted (eye-in-hand). Also spawns that exact camera to render a confirmation image.
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_wrist_calc
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
from isaaclab.utils.math import (create_rotation_matrix_from_view, quat_from_matrix, matrix_from_quat,
                                 quat_mul, quat_conjugate)
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits

DEVICE = "cuda:0"
N = 4
TARGET = "cube"
TCP_LOCAL = torch.tensor([-0.086, 0.0, 0.103], device=DEVICE)
HEIGHT = 0.15        # camera height above TCP (world +z)
TILT = 0.03          # small horizontal look offset so it isn't perfectly vertical (gimbal)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
env.reset()
for _ in range(4):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

robot = env.scene["robot"]
bid = robot.data.body_names.index("gripper_link")
p_g = robot.data.body_pos_w[0, bid]                 # (3,)
q_g = robot.data.body_quat_w[0, bid]                # (4,) wxyz
R_g = matrix_from_quat(q_g.unsqueeze(0))[0]         # (3,3) gripper->world
tcp_world = p_g + R_g @ TCP_LOCAL

print(f"[wcalc] gripper_link pos_w={p_g.cpu().numpy().round(3)} quat_w={q_g.cpu().numpy().round(3)}", flush=True)
print(f"[wcalc] R_g local-x in world={R_g[:,0].cpu().numpy().round(2)} "
      f"y={R_g[:,1].cpu().numpy().round(2)} z={R_g[:,2].cpu().numpy().round(2)}", flush=True)
print(f"[wcalc] tcp_world={tcp_world.cpu().numpy().round(3)}", flush=True)

# camera 15cm above TCP (WORLD +z), looking near-straight-down at TCP (tiny -y tilt avoids gimbal)
eye_w = tcp_world + torch.tensor([0.0, 0.0, HEIGHT], device=DEVICE)
tgt_w = tcp_world + torch.tensor([0.0, -TILT, 0.0], device=DEVICE)
R_cam_w = create_rotation_matrix_from_view(eye_w.unsqueeze(0), tgt_w.unsqueeze(0), "Z", device=DEVICE)
q_cam_w = quat_from_matrix(R_cam_w)[0]                       # opengl world orientation

# convert to gripper_link LOCAL: pos_local = R_g^T (eye_w - p_g) ; q_local = q_g^-1 * q_cam_w
pos_local = R_g.transpose(0, 1) @ (eye_w - p_g)
q_local = quat_mul(quat_conjugate(q_g).unsqueeze(0), q_cam_w.unsqueeze(0))[0]

pos_l = tuple(round(float(v), 5) for v in pos_local.cpu().numpy())
rot_l = tuple(round(float(v), 6) for v in q_local.cpu().numpy())
print(f"[wcalc] >>> WRIST OffsetCfg(pos={pos_l}, rot={rot_l}, convention='opengl')", flush=True)

env.close(); app.close()

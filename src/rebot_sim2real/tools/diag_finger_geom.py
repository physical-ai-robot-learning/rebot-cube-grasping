"""Read the gripper finger geometry and compute a 'between-the-jaws looking down the grasp axis'
wrist-camera mount (gripper_link LOCAL OffsetCfg). Prints the offset to bake into visual_obs.

The camera sits just behind the jaw gap along the ANTI-approach direction and looks ALONG the approach
(fingertip) direction, so the two jaws frame the view and the camera looks down through the gap at the
grasp point -- a fixed eye-in-hand relationship that holds at every arm pose.
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_finger_geom
"""
import os
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
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
names = robot.data.body_names
bid = names.index("gripper_link")
lid = names.index("gripper_left")
rid = names.index("gripper_right")
p_g = robot.data.body_pos_w[0, bid]
q_g = robot.data.body_quat_w[0, bid]
R_g = matrix_from_quat(q_g.unsqueeze(0))[0]
p_l = robot.data.body_pos_w[0, lid]
p_r = robot.data.body_pos_w[0, rid]
ee = env.scene["ee_frame"]
tcp = ee.data.target_pos_w[0, 0]

mid = 0.5 * (p_l + p_r)
approach = F.normalize((mid - p_g).unsqueeze(0))[0]      # link -> jaws (fingertip / grasp direction)
opening = F.normalize((p_l - p_r).unsqueeze(0))[0]       # jaw opening axis

print(f"[fg] gripper_link={p_g.cpu().numpy().round(3)} left={p_l.cpu().numpy().round(3)} right={p_r.cpu().numpy().round(3)}", flush=True)
print(f"[fg] tcp={tcp.cpu().numpy().round(3)} jaw_mid={mid.cpu().numpy().round(3)}", flush=True)
print(f"[fg] approach(link->jaws) world={approach.cpu().numpy().round(3)}", flush=True)
print(f"[fg] opening(L-R) world={opening.cpu().numpy().round(3)}", flush=True)

# camera: behind the jaw gap along -approach, looking ALONG approach through the gap toward the grasp
BACK = 0.06     # back off behind the gap (m)
AHEAD = 0.15    # look point ahead of the gap along approach
eye_w = mid - approach * BACK
tgt_w = mid + approach * AHEAD
R_cam_w = create_rotation_matrix_from_view(eye_w.unsqueeze(0), tgt_w.unsqueeze(0), "Z", device=DEVICE)
q_cam_w = quat_from_matrix(R_cam_w)[0]

pos_local = R_g.transpose(0, 1) @ (eye_w - p_g)
q_local = quat_mul(quat_conjugate(q_g).unsqueeze(0), q_cam_w.unsqueeze(0))[0]
pos_l = tuple(round(float(v), 5) for v in pos_local.cpu().numpy())
rot_l = tuple(round(float(v), 6) for v in q_local.cpu().numpy())
print(f"[fg] >>> WRIST OffsetCfg(pos={pos_l}, rot={rot_l}, convention='opengl')", flush=True)
env.close(); app.close()

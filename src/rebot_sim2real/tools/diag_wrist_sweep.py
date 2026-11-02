"""Sweep several 'above-the-gripper looking down' wrist mounts in one boot, render native 128, pick.

Uses the known gripper_link rest orientation (q_g, from diag_wrist_calc) to compute, for each candidate
WORLD offset from the TCP (forward/up), the equivalent gripper_link-LOCAL OffsetCfg(pos,rot) -- so each
camera is rigidly mounted and looks down at the grasp. Renders all candidates for visual selection.
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_wrist_sweep
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
Q_G = torch.tensor([-0.263, -0.653, -0.658, 0.266], device=DEVICE)   # gripper_link rest quat (wxyz)
R_G = matrix_from_quat(Q_G.unsqueeze(0))[0]

# h08 (8cm straight up, looking straight DOWN) was good; user wants it a little BACK (+y, toward the
# gripper body) so the jaws come more into frame. eye_delta = world (0, back, up) from TCP; look DOWN.
# straight-down look orientation (tiny +y to avoid the vertical gimbal), in gripper local frame
_down = torch.tensor([0.0, 0.02, -1.0], device=DEVICE)
_q_down_w = quat_from_matrix(create_rotation_matrix_from_view(
    torch.zeros(1, 3, device=DEVICE), _down.unsqueeze(0), "Z", device=DEVICE))[0]
Q_LOCAL_DOWN = quat_mul(quat_conjugate(Q_G).unsqueeze(0), _q_down_w.unsqueeze(0))[0]
CANDS = {
    "back2": (0.0, 0.02, 0.08),
    "back4": (0.0, 0.04, 0.08),
    "back6": (0.0, 0.06, 0.09),
    "back8": (0.0, 0.08, 0.10),
}


def local_offset(eye_delta):
    e = torch.tensor(eye_delta, device=DEVICE, dtype=torch.float32)
    eye_local = TCP_LOCAL + R_G.transpose(0, 1) @ e
    return (tuple(float(v) for v in eye_local.cpu().numpy()),
            tuple(float(v) for v in Q_LOCAL_DOWN.cpu().numpy()))


env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

_spawn = sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.005, 50.0), lock_camera=True)
offs = {}
for name, height in CANDS.items():
    pos_l, rot_l = local_offset(height)
    offs[name] = (pos_l, rot_l)
    setattr(env_cfg.scene, f"w_{name}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/w_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=pos_l, rot=rot_l, convention="opengl"),
        data_types=["rgb"], spawn=_spawn, width=128, height=128, update_period=0))

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
env.reset()
for _ in range(6):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
for name in CANDS:
    r0 = env.scene[f"w_{name}"].data.output["rgb"][0]
    imageio.imwrite(f"rl_output/visual_diag/wsweep_{name}_128.png", r0[..., :3].cpu().numpy().astype("uint8"))
    pos_l, rot_l = offs[name]
    print(f"[wsweep] {name}: range[{int(r0[...,:3].min())},{int(r0[...,:3].max())}] "
          f"pos={tuple(round(v,4) for v in pos_l)} rot={tuple(round(v,4) for v in rot_l)}", flush=True)

print("[wsweep] saved wsweep_* to rl_output/visual_diag/", flush=True)
env.close(); app.close()

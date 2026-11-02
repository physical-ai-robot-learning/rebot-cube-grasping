"""Measure the gripper PAD GAP vs gripper qpos, to (a) report the sim ground-truth open/closed extremes
+ grasp width for the deploy team, and (b) pick a SAFE stroke_lo that still clears the 4cm cube but
never commands mechanical-max open. Sets the gripper joint across its stroke and measures the distance
between the two finger pad bodies (gripper_left, gripper_right).
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_gripper_stroke
"""
import os
import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app   # configure_env adds a TiledCamera

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env

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

env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene["robot"]
env.reset()

g_ids, g_names = robot.find_joints(["gripper", "gripper_joint2"])
cube = env.scene["cube"]
print(f"[grip] gripper joints {g_names} ids {g_ids}", flush=True)

# put the 4cm cube AT the grasp center (TCP) so sweeping qpos closes the mouth onto it -> the qpos
# where BOTH fingers register contact = the grip point (mouth gap == cube width ~4cm at that qpos).
tcp = env.scene["ee_frame"].data.target_pos_w[0, 0, :].clone()
print(f"[grip] TCP(grasp center) world = {tuple(round(v,3) for v in tcp.tolist())}", flush=True)


def set_grip_and_measure(q):
    # hold cube at TCP, set gripper qpos, settle, read contact force on both fingers
    js = robot.data.joint_pos.clone()
    for jid in g_ids:
        js[:, jid] = q
    robot.write_joint_state_to_sim(js, torch.zeros_like(js))
    cpose = torch.cat([tcp, torch.tensor([1.0, 0.0, 0.0, 0.0], device=tcp.device)]).unsqueeze(0)
    for _ in range(8):
        cube.write_root_pose_to_sim(cpose)
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=tcp.device))
        env.sim.step(render=False); env.scene.update(dt=env.physics_dt)
    gf = env.scene["gripper_contact"].data.force_matrix_w[0, 0, 0, :].norm().item()
    jf = env.scene["jaw_contact"].data.force_matrix_w[0, 0, 0, :].norm().item()
    return gf, jf


print("[grip] qpos(m) -> contact force on 4cm cube  (sq_open_approach: qpos->0=OPEN, qpos>=0.035=CLOSED):", flush=True)
for q in [0.0, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.0715]:
    gf, jf = set_grip_and_measure(q)
    grip = "GRIP" if (gf >= 0.3 and jf >= 0.3) else ""
    print(f"   qpos={q:.4f}  grip_force={gf:7.2f}  jaw_force={jf:7.2f}  {grip}", flush=True)
print("[grip] first qpos with BOTH forces>=0.3 == closes on the 4cm cube. stroke_lo must stay OPEN of that.", flush=True)
print("[grip] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

"""Run ON THE TRAINING PC (needs Isaac) to dump the EXACT joint order + obs/action layout, so you

Stage: June deploy (writes the sim joint order and observation layout).
Was src/rebot_sim2real/deploy/dump_sim_spec.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).
Run: python -m rebot_teacher_student.deploy.dump_sim_spec
Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).

can verify rebot_deploy.py's constants (JOINT_ORDER, OBS_LAYOUT) match the trained sim 1:1.

  PYTHONPATH=src .venv/bin/python -m rebot_teacher_student.deploy.dump_sim_spec
"""
import os
from rebot_grasp_common.paths import LEISAAC_ASSETS
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", LEISAAC_ASSETS)

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp
from rebot_teacher_student.env.env_config import configureEnv, applyMotorLimits

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device="cuda:0", num_envs=2)
env_cfg = configureEnv(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
_t = getattr(env_cfg.scene, "cube"); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
for cam in ["front", "side", "wrist"]:
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)

env = ManagerBasedRLEnv(cfg=env_cfg)
applyMotorLimits(env, {})
env.reset()

print("\n================ SIM SPEC (copy into rebot_deploy.py) ================")
robot = env.scene["robot"]
print("JOINT_ORDER =", list(robot.data.joint_names))
print("  n_joints =", len(robot.data.joint_names))

print("\nROBOT INIT (REST) joint_pos[env0] =")
for n, v in zip(robot.data.joint_names, robot.data.joint_pos[0].tolist()):
    print(f"    {n:18s} {v:+.4f}")

print("\nOBS TERMS (group 'policy') in order:")
om = env.observation_manager
total = 0
for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
    d = int(dim[0]) if hasattr(dim, "__len__") else int(dim)
    print(f"    {name:16s} dim={d}")
    total += d
print(f"  TOTAL obs dim = {total}  (rebot_deploy OBS_DIM must equal this)")

print("\nACTION TERMS in order:")
am = env.action_manager
for name in am.active_terms:
    term = am._terms[name]
    print(f"    {name:16s} dim={term.action_dim}")
print(f"  TOTAL action dim = {am.total_action_dim}")

print("\nee_frame TCP offset (should be (-0.086,0,0.103)):")
try:
    print("   ", env_cfg.scene.ee_frame.target_frames[0].offset.pos)
except Exception as e:
    print("    (n/a)", e)
print("======================================================================\n")
env.close()
app.close()

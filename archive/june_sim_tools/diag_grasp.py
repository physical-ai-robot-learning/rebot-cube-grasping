"""Pin a cube between the reBot jaws, close, and check whether _is_grasped fires --
and which of its 7 sub-conditions pass/fail. Tells us if the grasp detector works for reBot."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import torch  # noqa
import numpy as np  # noqa
from isaaclab.envs import ManagerBasedRLEnv  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa
from isaaclab.managers import SceneEntityCfg  # noqa
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm  # noqa
from isaaclab.sensors import ContactSensorCfg  # noqa
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp  # noqa
from rebot_sim2real.env.env_config import configure_env  # noqa
from rebot_sim2real.env import maniskill_rewards as MR  # noqa

out = []
def w(s): out.append(str(s)); print(s, flush=True)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=__import__("os").environ.get("SIM_DEVICE","cpu"), num_envs=2)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
import os as _OS
FILTER_SUFFIX = _OS.environ.get("FILTER_SUFFIX", "/cube")  # ""=parent prim(old), "/cube"=nested body
_t = env_cfg.scene.cube
_FILT = _t.prim_path + FILTER_SUFFIX
print("CONTACT FILTER =", _FILT, flush=True)
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_FILT])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_FILT])
_t.spawn.activate_contact_sensors = True
_t.spawn.rigid_props.disable_gravity = True   # float the cube at the grasp center during the close test
env_cfg.scene.robot.spawn.activate_contact_sensors = True
for cam in ("front", "iso", "wrist"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)

env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()
robot = env.scene["robot"]; cube = env.scene["cube"]; ee = env.scene["ee_frame"]
gi = robot.joint_names.index("gripper")
adim = env.action_manager.total_action_dim

def subconds():
    """Replicate _is_grasped's 7 conditions and report each."""
    gf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    jf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    grip_ok = gf.norm(dim=-1) >= 0.5
    jaw_ok = jf.norm(dim=-1) >= 0.5
    bn = robot.data.body_names
    gq = robot.data.body_quat_w[:, bn.index("gripper_left"), :]
    jq = robot.data.body_quat_w[:, bn.index("gripper_right"), :]
    gy = MR._quat_to_y_axis(gq); jy = -MR._quat_to_y_axis(jq)
    g_ang = MR._compute_angle_between(gy, gf) <= (110 * np.pi / 180)
    j_ang = MR._compute_angle_between(jy, jf) <= (110 * np.pi / 180)
    gxy = gf.clone(); gxy[:, 2] = 0; jxy = jf.clone(); jxy[:, 2] = 0
    g_h = gxy.norm(dim=-1) / (gf.norm(dim=-1) + 1e-8) > 0.3
    j_h = jxy.norm(dim=-1) / (jf.norm(dim=-1) + 1e-8) > 0.3
    gd = gxy / (gxy.norm(dim=-1, keepdim=True) + 1e-8)
    jd = jxy / (jxy.norm(dim=-1, keepdim=True) + 1e-8)
    opp = (gd * jd).sum(dim=-1) < -0.5
    return dict(gripF=round(gf[0].norm().item(), 3), jawF=round(jf[0].norm().item(), 3),
                grip_ok=bool(grip_ok[0]), jaw_ok=bool(jaw_ok[0]),
                g_ang=bool(g_ang[0]), j_ang=bool(j_ang[0]),
                g_horiz=bool(g_h[0]), j_horiz=bool(j_h[0]), opposing=bool(opp[0]))

bn = robot.data.body_names
bL = bn.index("gripper_left"); bR = bn.index("gripper_right")

def report(step):
    gq = robot.data.joint_pos[0, gi].item()
    cp = cube.data.root_pos_w[0]
    gnet = env.scene["gripper_contact"].data.net_forces_w[0, 0, :].norm().item()
    jnet = env.scene["jaw_contact"].data.net_forces_w[0, 0, :].norm().item()
    grasped = bool(MR._is_grasped(env)[0])
    w("  step %2d qpos=%.4f cube=%s netF gL=%.3f gR=%.3f grasped=%s %s" %
      (step, gq, [round(x, 3) for x in cp.tolist()], gnet, jnet, grasped, subconds()))

# confirm where the cube rigid body / collider actually lives in the stage
import omni.usd as _ou  # noqa
from pxr import Usd as _U, UsdPhysics as _UP  # noqa
_stage = _ou.get_context().get_stage()
w("--- cube prim subtree (RigidBody / Collision) ---")
for p in _U.PrimRange(_stage.GetPrimAtPath("/World/envs/env_0/cube"), _U.TraverseInstanceProxies()):
    tags = []
    if p.HasAPI(_UP.RigidBodyAPI): tags.append("RIGIDBODY")
    if p.HasAPI(_UP.CollisionAPI): tags.append("COLLISION")
    if tags: w("  %s %s" % (p.GetPath(), tags))

# OPEN pads, fix the KINEMATIC cube at the grasp center, GRIP, measure sustained force.
for _ in range(40):
    a = torch.zeros((env.num_envs, adim)); a[:, -1] = -1.0; env.step(a)
tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
rp = cube.data.root_state_w[:, :7].clone(); rp[:, :3] = tcp
cube.write_root_pose_to_sim(rp); cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6)))
for _ in range(3): env.step(torch.zeros((env.num_envs, adim)))
w("kinematic cube fixed at %s; GRIP act+1, per-step sustained force ---" % [round(x,3) for x in tcp[0].tolist()])
maxg = maxj = maxgn = maxjn = 0.0
for step in range(120):
    a = torch.zeros((env.num_envs, adim)); a[:, -1] = +1.0; env.step(a)
    gf = env.scene["gripper_contact"].data.force_matrix_w[0, 0, 0, :].norm().item()
    jf = env.scene["jaw_contact"].data.force_matrix_w[0, 0, 0, :].norm().item()
    gn = env.scene["gripper_contact"].data.net_forces_w[0, 0, :].norm().item()
    jn = env.scene["jaw_contact"].data.net_forces_w[0, 0, :].norm().item()
    maxg = max(maxg, gf); maxj = max(maxj, jf); maxgn = max(maxgn, gn); maxjn = max(maxjn, jn)
    if step % 30 == 0 or step == 119:
        w("  step %3d qpos=%.4f force_matrix gL=%.3f gR=%.3f | net gL=%.3f gR=%.3f" %
          (step, robot.data.joint_pos[0, gi].item(), gf, jf, gn, jn))
w("MAX over 120: force_matrix gL=%.3f gR=%.3f | net gL=%.3f gR=%.3f" % (maxg, maxj, maxgn, maxjn))

# PRECISE: where are the finger COLLIDERS at closed (qpos~0.0715) vs the cube?
from pxr import UsdGeom as _UG, Usd as _U2  # noqa
bbc = _UG.BBoxCache(_U2.TimeCode.Default(), [_UG.Tokens.default_, _UG.Tokens.proxy, _UG.Tokens.render, _UG.Tokens.guide])
def caabb(path):
    p = _stage.GetPrimAtPath(path)
    if not p or not p.IsValid(): return None
    r = bbc.ComputeWorldBound(p).ComputeAlignedRange()
    return [round(r.GetMin()[i],3) for i in range(3)], [round(r.GetMax()[i],3) for i in range(3)]
cp = cube.data.root_pos_w[0].tolist()
w("cube pos (closed) = %s" % [round(x,3) for x in cp])
for f in ("gripper_left", "gripper_right"):
    w("  %s collider AABB = %s" % (f, caabb("/World/envs/env_0/Robot/%s/collisions/%s/node_STL_BINARY_" % (f, f))))
w("VERDICT: if cube pos is OUTSIDE both finger collider AABBs -> pads never reach the cube (placement/geometry), not a sensor bug.")

with open("/tmp/diag_grasp.txt", "w") as f:
    f.write("\n".join(out) + "\n")
import os as _os; _os._exit(0)

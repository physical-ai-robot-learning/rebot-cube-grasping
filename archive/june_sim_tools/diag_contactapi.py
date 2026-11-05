"""Check whether PhysxContactReportAPI is present on the reBot fingers + cube colliders.
If missing, that is why force_matrix_w / net_forces_w read 0 (known IsaacLab USD-roundtrip bug)."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

from isaaclab.envs import ManagerBasedRLEnv  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa
from isaaclab.managers import SceneEntityCfg  # noqa
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm  # noqa
from isaaclab.sensors import ContactSensorCfg  # noqa
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp  # noqa
from rebot_sim2real.env.env_config import configure_env  # noqa

out = []
def w(s): out.append(str(s)); print(s, flush=True)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device="cpu", num_envs=1)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
_t = env_cfg.scene.cube
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_t.prim_path + "/cube"])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_t.prim_path + "/cube"])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
for cam in ("front", "iso", "wrist"):
    if hasattr(env_cfg.scene, cam): setattr(env_cfg.scene, cam, None)

env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()

import omni.usd  # noqa
from pxr import Usd, UsdPhysics, PhysxSchema  # noqa
stage = omni.usd.get_context().get_stage()

def check(root):
    w("--- under %s ---" % root)
    p = stage.GetPrimAtPath(root)
    if not p or not p.IsValid():
        w("  (prim not found)"); return
    for sub in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
        rb = sub.HasAPI(UsdPhysics.RigidBodyAPI)
        col = sub.HasAPI(UsdPhysics.CollisionAPI)
        cr = sub.HasAPI(PhysxSchema.PhysxContactReportAPI)
        if rb or col or cr:
            thr = None
            if cr:
                a = PhysxSchema.PhysxContactReportAPI(sub).GetThresholdAttr()
                thr = a.Get() if a else None
            w("  %s  RB=%s COL=%s CONTACT_REPORT=%s thr=%s" % (sub.GetPath(), rb, col, cr, thr))

check("/World/envs/env_0/Robot/gripper_left")
check("/World/envs/env_0/Robot/gripper_right")
check("/World/envs/env_0/cube")

with open("/tmp/diag_contactapi.txt", "w") as f:
    f.write("\n".join(out) + "\n")
import os as _os; _os._exit(0)

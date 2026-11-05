"""Inspect the gripper_link subtree to locate the physical CAMERA MOUNT (the small square on top of
the gripper where the real wrist camera attaches). Prints every descendant prim's local pose (in the
gripper_link frame) + mesh bbox size, so we can read off the mount's exact offset and bolt the wrist
camera there (instead of a hand-tuned offset). Run:
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_gripper_mounts
"""
import os
import numpy as np

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
env.reset()

import omni.usd
from pxr import Usd, UsdGeom, Gf
stage = omni.usd.get_context().get_stage()

GL = "/World/envs/env_0/Robot/gripper_link"
gprim = stage.GetPrimAtPath(GL)
print(f"[mount] gripper_link prim valid={bool(gprim and gprim.IsValid())}", flush=True)

cache = UsdGeom.XformCache(Usd.TimeCode.Default())
g_world = cache.GetLocalToWorldTransform(gprim)
g_world_inv = g_world.GetInverse()


def local_pose(prim):
    """prim transform relative to gripper_link: translation (x,y,z) in the link frame."""
    w = cache.GetLocalToWorldTransform(prim)
    rel = w * g_world_inv          # row-vector convention: world = local * parentWorld
    t = rel.ExtractTranslation()
    return (t[0], t[1], t[2])


def mesh_size(prim):
    """bbox MIN/MAX of a mesh prim's points in the gripper_link frame (intermediate xforms are
    identity, so point coords == link-frame coords). Tells us where the 'top' surface is."""
    try:
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            return None
        a = np.array([[p[0], p[1], p[2]] for p in pts])
        mn, mx = a.min(0), a.max(0)
        return (f"min=({mn[0]:+.4f},{mn[1]:+.4f},{mn[2]:+.4f}) "
                f"max=({mx[0]:+.4f},{mx[1]:+.4f},{mx[2]:+.4f})")
    except Exception:
        return None


print(f"[mount] descendants of {GL} (local xyz in gripper_link frame; size = mesh bbox):", flush=True)
for prim in Usd.PrimRange(gprim, Usd.TraverseInstanceProxies()):   # descend into instanced meshes
    if prim.GetPath() == gprim.GetPath():
        continue
    t = prim.GetTypeName()
    lp = local_pose(prim)
    sz = mesh_size(prim) if t == "Mesh" else None
    extra = f" size={sz}" if sz else ""
    depth = str(prim.GetPath()).count("/") - str(gprim.GetPath()).count("/")
    print(f"   {'  '*depth}{prim.GetName():28s} <{t}>  local=({lp[0]:+.4f},{lp[1]:+.4f},{lp[2]:+.4f}){extra}", flush=True)

# world layout at rest -> to aim an external inspection camera at the gripper
for ln in ("gripper_link", "gripper_left", "gripper_right"):
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{ln}")
    if p and p.IsValid():
        w = cache.GetLocalToWorldTransform(p).ExtractTranslation()
        print(f"[mount] {ln:14s} world=({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})", flush=True)
try:
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0, :].tolist()
    print(f"[mount] TCP            world=({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f})", flush=True)
except Exception as e:
    print(f"[mount] TCP read err: {e}", flush=True)
print(f"[mount] env_origin = {env.scene.env_origins[0].tolist()}", flush=True)
print("[mount] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

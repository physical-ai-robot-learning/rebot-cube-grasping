"""Draw the gripper_link coordinate axes (R=+x, G=+y, B=+z) + a candidate camera position (yellow)
as marker spheres, then render external close-ups. Lets us SEE the link-frame orientation and verify
whether the candidate sits on the top camera-mount square. Markers are placed at REST (static).
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_mount_marker --cam=-0.10,0,0.07
"""
import argparse, os
import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=str, default="-0.10,0,0.07")  # candidate camera link-frame pos (yellow)
ap.add_argument("--axis-len", type=float, default=0.08)
ap.add_argument("--tag", type=str, default="m")
args, _ = ap.parse_known_args()

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import _lookat_quat_opengl
import imageio

DEVICE = "cuda:0"
GRIP = (0.35, -0.40, 0.21)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
_t = getattr(env_cfg.scene, "cube"); _cp = _t.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

VIEWS = {   # 'front' looks along +y from the front -> image vertical == world z (up/down UNAMBIGUOUS)
    "front":   ((0.35, -1.05, 0.30), (0.35, -0.42, 0.18)),   # well back + wide -> whole gripper + markers
    "topdown": ((0.35, -0.435, 0.62), (0.35, -0.40, 0.20)),  # straight down -> x-y placement
}
for nm, (eye, look) in VIEWS.items():
    setattr(env_cfg.scene, f"insp_{nm}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/insp_" + nm,
        offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=_lookat_quat_opengl(eye, look), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True),
        width=520, height=520, update_period=0))

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
env.reset()
for _ in range(4):
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=DEVICE))

import omni.usd
from pxr import Usd, UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
gprim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper_link")
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
g_world = cache.GetLocalToWorldTransform(gprim)

# print link world axes (rotation rows) so the link orientation in world is explicit
R = g_world.ExtractRotationMatrix()
print(f"[mk] link +x in world = ({R[0][0]:+.2f},{R[0][1]:+.2f},{R[0][2]:+.2f})", flush=True)
print(f"[mk] link +y in world = ({R[1][0]:+.2f},{R[1][1]:+.2f},{R[1][2]:+.2f})", flush=True)
print(f"[mk] link +z in world = ({R[2][0]:+.2f},{R[2][1]:+.2f},{R[2][2]:+.2f})", flush=True)

cam = tuple(float(v) for v in args.cam.split(","))
markers = {  # name: (link_point, (r,g,b), radius)  — top face = link -z (= world UP)
    "origin": ((0, 0, 0), (1, 1, 1), 0.008),     # small white = gripper_link origin (reference)
    "cam":    (cam, (1, 0, 0), 0.016),           # big red = candidate camera mount
}
for nm, (lp, rgb, rad) in markers.items():
    wp = g_world.Transform(Gf.Vec3d(*lp))
    path = f"/World/marker_{nm}"
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.GetRadiusAttr().Set(rad)
    sph.AddTranslateOp().Set(Gf.Vec3d(wp[0], wp[1], wp[2]))
    sph.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
    print(f"[mk] {nm:7s} link={tuple(round(v,3) for v in lp)} world=({wp[0]:+.3f},{wp[1]:+.3f},{wp[2]:+.3f})", flush=True)

for _ in range(3):   # let markers register in the render
    env.sim.render()

os.makedirs("rl_output/recalib_diag", exist_ok=True)
for nm in VIEWS:
    rgb = env.scene[f"insp_{nm}"].data.output.get("rgb")
    if rgb is None:
        print(f"[mk] {nm}: no rgb", flush=True); continue
    a = rgb[0, ..., :3].detach().cpu().numpy()
    a = a if a.dtype == np.uint8 else (a * 255).clip(0, 255).astype(np.uint8)
    p = f"rl_output/recalib_diag/marker_{nm}_{args.tag}.png"
    imageio.imwrite(p, a); print(f"[mk] saved {p}", flush=True)
print("[mk] DONE", flush=True)
try:
    env.close(); app.close()
except Exception:
    pass

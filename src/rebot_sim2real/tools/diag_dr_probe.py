"""Probe visual-DR prerequisites: (1) does TiledCamera semantic_segmentation work + tag robot/cube?
(2) does greenscreen composite render? (3) what are the UsdLux light prim paths? (4) table entity?

  PYTHONPATH=src python -m rebot_sim2real.tools.diag_dr_probe
"""
import os
import torch
import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import configure_visual, composite_bg, build_fg_ids, downsample, WRIST_SIZE, FRONT_SIZE

DEVICE = "cuda:0"; N = 4; TARGET = "cube"

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=True)
print(f"[probe] scene entities: {list(env_cfg.scene.__dict__.get('_target_cfgs', {}).keys()) if hasattr(env_cfg.scene,'_target_cfgs') else 'n/a'}", flush=True)
print(f"[probe] has 'table' attr on scene cfg? {hasattr(env_cfg.scene, 'table')}", flush=True)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})
env.reset()
for _ in range(6):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

front = env.scene["front"]; wrist = env.scene["wrist"]
print(f"[probe] front.data.output keys = {list(front.data.output.keys())}", flush=True)
print(f"[probe] front.data.info keys   = {list((front.data.info or {}).keys()) if hasattr(front.data,'info') else 'NO .info'}", flush=True)
seg = front.data.output.get("semantic_segmentation")
print(f"[probe] front seg = {None if seg is None else (tuple(seg.shape), seg.dtype)}", flush=True)
info = None
try:
    info = front.data.info["semantic_segmentation"]
    # info may be per-camera or a dict; print idToLabels
    id2l = info.get("idToLabels") if isinstance(info, dict) else None
    print(f"[probe] front idToLabels = {id2l}", flush=True)
except Exception as e:
    print(f"[probe] front info access err: {e}", flush=True)
print(f"[probe] build_fg_ids(front) = {build_fg_ids(info)}", flush=True)

# greenscreen composite test (random solid bg) + save
os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
if seg is not None:
    bg = (torch.rand(N, 1, 1, 3, device=DEVICE) * 255).expand(N, 128, 128, 3)
    comp = composite_bg(front.data.output["rgb"], seg, info, bg)
    imageio.imwrite("rl_output/visual_diag/dr_front_raw.png", front.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8"))
    imageio.imwrite("rl_output/visual_diag/dr_front_greenscreen.png", comp[0].cpu().numpy())
    print(f"[probe] composite range raw[{int(front.data.output['rgb'][0,...,:3].min())},{int(front.data.output['rgb'][0,...,:3].max())}] "
          f"comp[{int(comp[0].min())},{int(comp[0].max())}] -> saved dr_front_{{raw,greenscreen}}.png", flush=True)

# UsdLux light prim paths (for the lighting DR event)
try:
    import omni.usd
    from pxr import UsdLux
    stage = omni.usd.get_context().get_stage()
    lights = [str(p.GetPath()) for p in stage.Traverse()
              if p.GetTypeName() in ("DistantLight", "DomeLight", "SphereLight", "RectLight", "CylinderLight", "DiskLight")]
    print(f"[probe] UsdLux light prims ({len(lights)}): {lights[:8]}", flush=True)
except Exception as e:
    print(f"[probe] light traverse err: {e}", flush=True)

print("[probe] DONE", flush=True)
env.close(); app.close()

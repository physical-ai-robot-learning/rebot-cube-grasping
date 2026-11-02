"""Spawn several FRONT cameras (each with a baked look-at offset) in one boot, save 128/64 for each.

set_world_poses_from_view does NOT stick on a scene Camera (the offset is re-applied every step),
but the OffsetCfg pos+rot IS honored at spawn. So we bake the look-at rotation into each camera's
OffsetCfg (convention="opengl") and compare candidate poses side by side.

  PYTHONPATH=src python -m rebot_sim2real.tools.diag_front_sweep
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "enable_cameras": True}).app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import configure_visual

DEVICE = "cuda:0"
N = 4
TARGET = "cube"

# candidate (eye, target) in env frame
CANDS = {
    "A_mod":   ((0.25, 0.12, 0.30), (0.26, -0.33, 0.06)),
    "B_close": ((0.25, 0.05, 0.20), (0.26, -0.35, 0.05)),
    "C_3q":    ((0.55, 0.10, 0.32), (0.22, -0.33, 0.06)),
    "D_high":  ((0.25, 0.30, 0.45), (0.27, -0.30, 0.06)),
}


def lookat_quat(eye, target):
    e = torch.tensor([list(eye)], dtype=torch.float32)
    t = torch.tensor([list(target)], dtype=torch.float32)
    R = create_rotation_matrix_from_view(e, t, "Z", device="cpu")
    return tuple(quat_from_matrix(R)[0].tolist())


env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=TARGET, img_render=128)

# add one baked-offset front camera per candidate
_spawn = sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
for name, (eye, tgt) in CANDS.items():
    setattr(env_cfg.scene, f"f_{name}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/f_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=lookat_quat(eye, tgt), convention="opengl"),
        data_types=["rgb"], spawn=_spawn, width=128, height=128, update_period=0))

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})

obs_dict, _ = env.reset()
for _ in range(6):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

eo = env.scene.env_origins
cube_local = (env.scene[TARGET].data.root_pos_w - eo)[0].cpu().numpy().round(3)
print(f"[sweep] env0 cube local = {cube_local}", flush=True)

os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
for name in CANDS:
    cam = env.scene[f"f_{name}"]
    r0 = cam.data.output["rgb"][0]
    print(f"[sweep] {name}: pos_w={cam.data.pos_w[0].cpu().numpy().round(2)} range[{int(r0[...,:3].min())},{int(r0[...,:3].max())}]", flush=True)
    imageio.imwrite(f"rl_output/visual_diag/sweep_{name}_128.png", r0[..., :3].cpu().numpy().astype("uint8"))
    x = r0[..., :3].permute(2, 0, 1).float().unsqueeze(0)
    x = F.interpolate(x, size=(64, 64), mode="area")[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    imageio.imwrite(f"rl_output/visual_diag/sweep_{name}_64.png", np.repeat(np.repeat(x, 2, 0), 2, 1))

print("[sweep] saved sweep_* to rl_output/visual_diag/", flush=True)
env.close(); app.close()

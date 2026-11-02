"""Phase 1 verification: visual env wiring (state 21D + wrist 16x16 + front 64x64).

  PYTHONPATH=src python -m rebot_sim2real.tools.diag_visual_obs
"""
import os
import numpy as np
import torch

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
from rebot_sim2real.env.visual_obs import (
    configure_visual, downsample, STATE_DIM, WRIST_SIZE, FRONT_SIZE)

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
env_cfg = configure_visual(env_cfg, target_cube=TARGET)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})

print(f"[diag] active policy obs terms: {list(env.observation_manager.active_terms['policy'])}", flush=True)
obs_dict, _ = env.reset()
state = obs_dict["policy"]
for _ in range(6):
    env.step(torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE))

wrist = env.scene["wrist"].data.output["rgb"]
front = env.scene["front"].data.output["rgb"]
w = downsample(wrist, WRIST_SIZE)
f = downsample(front, FRONT_SIZE)
print(f"[diag] state = {tuple(state.shape)} (expect (N,{STATE_DIM}))", flush=True)
print(f"[diag] wrist render={tuple(wrist.shape)} -> {tuple(w.shape)}  range[{int(wrist[...,:3].min())},{int(wrist[...,:3].max())}]", flush=True)
print(f"[diag] front render={tuple(front.shape)} -> {tuple(f.shape)}  range[{int(front[...,:3].min())},{int(front[...,:3].max())}]", flush=True)

os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
# native 128 renders (for the USER to judge framing vs the real cameras)
imageio.imwrite("rl_output/visual_diag/native_front128.png", front[0, ..., :3].cpu().numpy().astype("uint8"))
imageio.imwrite("rl_output/visual_diag/native_wrist128.png", wrist[0, ..., :3].cpu().numpy().astype("uint8"))
# downsampled (what the policy actually consumes), upscaled for viewing
imageio.imwrite("rl_output/visual_diag/final_wrist16.png", np.repeat(np.repeat(w[0].cpu().numpy(), 8, 0), 8, 1))
imageio.imwrite("rl_output/visual_diag/final_front64.png", np.repeat(np.repeat(f[0].cpu().numpy(), 2, 0), 2, 1))
ok = state.shape[1] == STATE_DIM and tuple(w.shape[1:]) == (WRIST_SIZE, WRIST_SIZE, 3) and tuple(f.shape[1:]) == (FRONT_SIZE, FRONT_SIZE, 3)
print("[diag] OK -- Phase 1 verified" if ok else "[diag] FAIL", flush=True)
env.close(); app.close()

"""Render WRIST camera candidates AT THE MOMENT A CUBE IS GRASPED (not at rest), so we can pick the
mount that actually shows the held cube between the jaws.

Runs the state-only teacher to grasp+lift a cube, then for the grasping env renders several candidate
wrist cameras (gripper_link children, each looking at the TCP = grasp center from a different local
offset). Saves rl_output/visual_diag/grasp_<name>_128.png + grasp_front_128.png.
  PYTHONPATH=src python -m rebot_sim2real.tools.diag_grasp_wrist
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
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix, quat_mul
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import configure_visual
from rebot_sim2real.student.visual_nets import StateOnlyActor
from rebot_sim2real.env.maniskill_rewards import _is_grasped

DEVICE = "cuda:0"
N = 16
TARGET = "cube"
TCP_LOCAL = torch.tensor([-0.086, 0.0, 0.103], device=DEVICE)
TCP_T = tuple(float(v) for v in TCP_LOCAL.cpu().numpy())

# chosen mount = d_negx8 (eye = TCP_local + (-0.08,0,0.04), look at TCP). It showed the held cube
# centered between the jaws but UPSIDE-DOWN -> compare original vs 180deg-roll-flipped.
DNEG8 = (-0.08, 0.0, 0.04)
CANDS = {"dneg8": False, "dneg8_flip": True}
ROLL180 = torch.tensor([0.0, 0.0, 0.0, 1.0])   # 180deg about the camera optical axis (w,x,y,z)


def quat_local(eye_l, flip):
    e = torch.tensor([list(eye_l)], dtype=torch.float32)
    t = TCP_LOCAL.unsqueeze(0).cpu()
    q = quat_from_matrix(create_rotation_matrix_from_view(e, t, "Z", device="cpu"))[0]
    if flip:
        q = quat_mul(q.unsqueeze(0), ROLL180.unsqueeze(0))[0]
    return tuple(float(v) for v in q.tolist())


env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True)  # 43D obs + front cam

_spawn = sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.004, 50.0), lock_camera=True)
eye_l = (TCP_T[0] + DNEG8[0], TCP_T[1] + DNEG8[1], TCP_T[2] + DNEG8[2])
offs = {}
for name, flip in CANDS.items():
    rot = quat_local(eye_l, flip)
    offs[name] = rot
    setattr(env_cfg.scene, f"g_{name}", TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/g_" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=eye_l, rot=rot, convention="opengl"),
        data_types=["rgb"], spawn=_spawn, width=128, height=128, update_period=0))
print(f"[gw] eye_local={tuple(round(v,4) for v in eye_l)}", flush=True)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, {})

teacher = StateOnlyActor(43, env.action_manager.total_action_dim).to(DEVICE)
ck = torch.load("rl_output/rl_c51_smooth/c51_final.pt", map_location=DEVICE)
teacher.load_state_dict(ck["actor"]); teacher.eval()
print("[gw] teacher loaded", flush=True)

obs_dict, _ = env.reset()
grasp_env = -1
for step in range(250):
    obs = torch.nan_to_num(obs_dict["policy"])
    with torch.no_grad():
        act = teacher.act_deterministic(obs)
    obs_dict, _, term, trunc, _ = env.step(act)
    g = _is_grasped(env, SceneEntityCfg(TARGET), min_force=0.3)
    cube_z = env.scene[TARGET].data.root_pos_w[:, 2]
    lifted = cube_z > 0.15   # clearly off the table (rest 0.06) -> a real held grasp
    hit = (g & lifted)
    if hit.any():
        grasp_env = int(torch.nonzero(hit).flatten()[0].item())
        # let the grasp stabilize a few more steps so the cube is held high between the jaws
        for _ in range(6):
            obs = torch.nan_to_num(obs_dict["policy"])
            with torch.no_grad():
                act = teacher.act_deterministic(obs)
            obs_dict, _, _, _, _ = env.step(act)
        cz = float(env.scene[TARGET].data.root_pos_w[grasp_env, 2])
        print(f"[gw] grasp+lift at step {step}, env {grasp_env}, cube_z={cz:.3f}", flush=True)
        break

if grasp_env < 0:
    print("[gw] NO clear grasp (cube_z>0.15) detected in 250 steps", flush=True)
    grasp_env = 0

os.makedirs("rl_output/visual_diag", exist_ok=True)
import imageio
for name in CANDS:
    r = env.scene[f"g_{name}"].data.output["rgb"][grasp_env]
    imageio.imwrite(f"rl_output/visual_diag/grasp_{name}_128.png", r[..., :3].cpu().numpy().astype("uint8"))
    print(f"[gw] {name}: range[{int(r[...,:3].min())},{int(r[...,:3].max())}] rot={tuple(round(v,5) for v in offs[name])}", flush=True)
rf = env.scene["front"].data.output["rgb"][grasp_env]
imageio.imwrite("rl_output/visual_diag/grasp_front_128.png", rf[..., :3].cpu().numpy().astype("uint8"))
print("[gw] saved grasp_* to rl_output/visual_diag/", flush=True)
env.close(); app.close()

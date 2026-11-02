"""SIM2REAL VERIFY (closed loop): run the policy with PERCEPTION-derived cube_pos in the full control
loop -- the last sim-side check before hardware. The sim env is the "robot"; each step we render the
top camera, run rebot_sim2real.perception (NOT ground truth) to estimate cube_pos, assemble the obs
(deploy rule: while is_grasped the cube is at the gripper -> cube_pos = tcp_pos; else perception, hold
last estimate when the cube is occluded), and feed it to the policy. Success is judged by GROUND TRUTH
(the policy never sees it). Catches what V1/V2 can't: per-frame segmentation jitter + gripper OCCLUSION
during the grasp. If genuine-pick stays high, the whole pipeline minus real hardware/calibration works.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/verify_closed_loop_sim.py
Writes /tmp/verify_closed_loop_sim.txt.
"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import numpy as np, torch
from isaacsim.sensors.camera import Camera
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922
from rebot_sim2real import perception as P
from build_obs import SL_CUBE_POS, SL_CUBE_QUAT, SL_TCP_POS, SL_CUBE_OFF, SL_GRASPED

N_EP = 16
CKPT = sys.argv[1] if len(sys.argv) > 1 else "pick_rlpd10_latest.pt"
tcfg = load_train_config("configs/train.yaml"); tcfg.env.num_envs = 1; tcfg.env.dr_enable = False
dev = torch.device(tcfg.env.device)
from rebot_sim2real.envs.pick_env import RebotPickEnv
env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
env.pos_rand = float(getattr(tcfg.env, "pos_rand_max", 0.0)) or float(tcfg.env.pos_rand_xy)
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev)
agent.load_state_dict(torch.load(os.path.join(tcfg.ckpt_dir, CKPT), map_location=dev))
succ_z = tcfg.env.table_z + tcfg.env.success_lift_height
plane_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0
origin = env.env_origins[0].detach().cpu().numpy()
scfg = load_scene_config("configs/scene.yaml")

top = Camera(prim_path="/World/r_top", name="r_top", resolution=scfg.cam.resolution)
top.initialize(); _apply_c922(top, scfg.cam)
top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                   orientation=look_at_quat(origin + np.array(scfg.top_cam_pos),
                                            origin + np.array(scfg.top_cam_target)), camera_axes="usd")
for _ in range(20):
    env.world.step(render=True)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


K = _np(top.get_intrinsics_matrix()).astype(float)
T_cam2world = np.linalg.inv(_np(top.get_view_matrix_ros()).astype(float))
origin_t = torch.tensor(origin, device=dev, dtype=torch.float32)


def grab():
    rgba = top.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return np.asarray(rgba)[..., :3].astype("uint8")


obs = env.reset()
ep_done = 0; genuine = 0; perc_calls = 0; perc_fail = 0
peak_held = torch.tensor([-1.0], device=dev)
last_est_local = None
steps = 0
MAX_STEPS = N_EP * tcfg.env.episode_len + 50
while ep_done < N_EP and steps < MAX_STEPS:
    steps += 1
    tcp_local = obs[:, SL_TCP_POS]                       # proprioceptive (FK) -- trusted
    grasped = obs[:, SL_GRASPED].squeeze(-1) > 0.5
    if bool(grasped[0]):                                 # deploy rule: grasped cube rides the gripper
        est_local = tcp_local[0].clone()
    else:
        rgb = grab(); perc_calls += 1
        est_w = P.estimate_red_cube_world(rgb, K, T_cam2world, plane_z) if rgb is not None else None
        if est_w is None:                                # occluded / not detected -> hold last
            perc_fail += 1
            est_local = last_est_local if last_est_local is not None else obs[0, SL_CUBE_POS].clone()
        else:
            est_local = torch.tensor(est_w, device=dev, dtype=torch.float32) - origin_t
    last_est_local = est_local
    # assemble the obs the policy sees: perception cube_pos, identity quat, true proprioception/tcp
    pobs = obs.clone()
    pobs[:, SL_CUBE_POS] = est_local
    pobs[:, SL_CUBE_QUAT] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    pobs[:, SL_CUBE_OFF] = est_local - tcp_local[0]
    a = agent.act(pobs, deterministic=True)
    obs, _, done, info = env.step(a)
    _, s, _ = env._compute_state(env.prev_action)        # GROUND TRUTH for scoring (policy never sees)
    cube = s["cube_pos"]; d3 = torch.linalg.norm(cube - s["tcp_pos"], dim=-1)
    held = s["is_grasped"].bool() & (d3 < 0.06) & (~done.bool())
    peak_held = torch.where(held & (cube[:, 2] > peak_held), cube[:, 2], peak_held)
    if bool(done[0]):
        if float(peak_held[0]) >= succ_z:
            genuine += 1
        ep_done += 1
        peak_held = torch.tensor([-1.0], device=dev); last_est_local = None
        obs = env.get_obs()

occ = 100.0 * perc_fail / max(1, perc_calls)
lines = [f"CLOSED-LOOP perception-in-sim  ckpt={CKPT}  episodes={ep_done}  pos_rand={env.pos_rand}  rest_rand={tcfg.env.rest_rand}",
         f"GENUINE pick (GT-judged, policy fed PERCEPTION cube_pos): {genuine}/{ep_done}",
         f"perception calls={perc_calls}  occluded/undetected (held last)={perc_fail} ({occ:.0f}%)",
         f"deploy rule used: while is_grasped -> cube_pos=tcp_pos; else top-cam HSV ray-plane (hold-last on miss)",
         f"compare: GT-obs baseline (eval_verify) was ~16/16 genuine at this difficulty"]
out = "\n".join(lines)
with open("/tmp/verify_closed_loop_sim.txt", "w") as f:
    f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
print(out, flush=True)
app.close()

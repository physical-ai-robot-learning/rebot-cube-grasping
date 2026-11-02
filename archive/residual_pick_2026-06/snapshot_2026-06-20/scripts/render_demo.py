"""Render the scripted demo expert (center grasp) as a 3-view GIF — to SHOW what the demos look like.

Camera-enabled (like monitor.py), 1 env, cube under the gripper at the J1=0 grasp config; runs the
same settle->close->lift the demo collector uses. Saves runs/demo_render.gif + a final PNG.
"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml"); ap.add_argument("--j1", type=float, default=0.0)
ap.add_argument("--fps", type=int, default=15); args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import imageio.v2 as imageio
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922, make_wrist_camera, make_quarter_camera
from rebot_sim2real.grasp_traj import (
    APPROACH as APPROACH_Q, GRASP as GRASP_Q, LIFT as LIFT_Q,
    APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS,
)


def grab(cam):
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return np.asarray(rgba)[..., :3].astype("uint8")


def main():
    tcfg = load_train_config(args.config); tcfg.env.num_envs = 1
    tcfg.env.pos_rand_xy = 0.0; tcfg.env.dr_enable = False    # cube at table center -> proven 100% grasp
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage()
    origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"
    if getattr(scfg, "green_bg_enabled", False):          # green-screen background (match monitor.py)
        from isaacsim.core.api.objects import VisualCuboid
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/demo_greenfloor", name="demo_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], dtype=float),
                     color=np.array(scfg.green_bg_color, dtype=float))
        if getattr(scfg, "green_bg_dome", False):
            from pxr import UsdLux, Gf
            dome = UsdLux.DomeLight.Define(stage, "/World/demo_greendome")
            dome.CreateColorAttr(Gf.Vec3f(*[float(x) for x in scfg.green_bg_color]))
            dome.CreateIntensityAttr(300.0)
    top = Camera(prim_path="/World/r_top", name="r_top", resolution=scfg.cam.resolution); top.initialize(); _apply_c922(top, scfg.cam)
    top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                       orientation=look_at_quat(origin + np.array(scfg.top_cam_pos), origin + np.array(scfg.top_cam_target)), camera_axes="usd")
    quarter = make_quarter_camera(scfg, origin=origin, prim_path="/World/r_quarter")
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    cams = {"quarter": quarter, "top": top, "wrist": wrist}

    arm_idx, grip_idx = env.arm_idx, env.grip_idx
    # eye-in-hand top-down pick: lifted approach (wrist cam on table) -> untilt -> descend (open
    # fingers straddle) -> close (grip) -> lift. Canonical poses in rebot_sim2real.grasp_traj
    # (joint1 overridden by --j1 for this single-env render).
    def _q(pose):
        return torch.tensor([[args.j1] + list(pose[1:])], device=dev)
    APPROACH = _q(APPROACH_Q)
    UNTILT = APPROACH                            # deprecated (28-deg approach needs no untilt)
    GRASP = _q(GRASP_Q)
    LIFT = _q(LIFT_Q)
    GO = tcfg.env.grip_open_qpos
    env.reset()
    q = env.robots.get_joint_positions().clone(); q[:, arm_idx] = APPROACH; q[:, grip_idx] = GO
    env.robots.set_joint_positions(q); env.arm_target[:] = APPROACH; env.grip_target[:] = GO
    env.robots.set_joint_position_targets(env._full_targets())
    for _ in range(20):                          # settle at the lifted approach pose
        env.world.step(render=True)

    N_APP, N_UNT, N_DESC, N_CLOSE, N_LIFT = APPROACH_STEPS, 0, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS
    frames = []
    grasp_frame = None                           # frame captured at the end of descent (the straddle moment)
    min_tip = 1e9                                # lowest fingertip z over the trajectory (table-pen check)
    max_cube_z = 0.0; succeeded = False
    for t in range(N_APP + N_UNT + N_DESC + N_CLOSE + N_LIFT):
        cur = env.robots.get_joint_positions()[:, arm_idx]
        if t < N_APP:                            # approach: hold lifted, wrist cam on table
            tgt, closing = APPROACH, False
        elif t < N_APP + N_UNT:                  # untilt the wrist (J4->0, fingers point down)
            tgt, closing = UNTILT, False
        elif t < N_APP + N_UNT + N_DESC:         # descend (open fingers straddle the cube)
            tgt, closing = GRASP, False
        elif t < N_APP + N_UNT + N_DESC + N_CLOSE:   # close -> grip
            tgt, closing = GRASP, True
        else:                                    # lift
            tgt, closing = LIFT, True
        a_arm = ((tgt - cur) / tcfg.env.arm_step).clamp(-1, 1)
        a_grip = torch.ones(1, 1, device=dev) if closing else -torch.ones(1, 1, device=dev)
        _, _, done, info = env.step(torch.cat([a_arm, a_grip], -1))
        _, s, _ = env._compute_state(env.prev_action)
        min_tip = min(min_tip, float(s["tcp_pos"][0, 2]) - tcfg.env.finger_tip_drop)
        max_cube_z = max(max_cube_z, float(s["cube_pos"][0, 2]))
        f = [grab(cams[k]) for k in ("quarter", "top", "wrist")]
        if not any(x is None for x in f):
            fr = np.concatenate(f, axis=1)
            frames.append(fr)
            if t == N_APP + N_DESC:              # descent just finished, fingers around the cube
                grasp_frame = fr
        if bool(done[0]):                        # stop at success (or fall/timeout) -> clean clip, no post-reset frames
            succeeded = float(info["ep_success"][0]) > 0.5
            break
    else:
        succeeded = False
    cube_z = max_cube_z
    pen = max(0.0, tcfg.env.table_z - min_tip)   # how far the fingertip dipped below the table top
    if frames:
        clip = np.stack(frames)[:, ::2, ::2, :]
        imageio.mimsave(os.path.join(tcfg.log_dir, "demo_render.gif"), list(clip), format="GIF", loop=0)
        imageio.imwrite(os.path.join(tcfg.log_dir, "demo_render.png"), clip[-1])
        if grasp_frame is not None:
            imageio.imwrite(os.path.join(tcfg.log_dir, "demo_render_grasp.png"), grasp_frame[::2, ::2, :])
    with open("/tmp/render_demo_result.txt", "w") as fh:
        fh.write(f"j1={args.j1} SUCCESS={succeeded} peak_cube_z={cube_z:.3f} "
                 f"min_tip_z={min_tip:.3f} table_z={tcfg.env.table_z} tip_penetration={pen:.3f} "
                 f"frames={len(frames)}\n"); fh.flush(); os.fsync(fh.fileno())
    print(f"[render] SUCCESS={succeeded} peak_cube_z={cube_z:.3f} min_tip_z={min_tip:.3f} -> runs/demo_render.gif", flush=True)
    app.close()


if __name__ == "__main__":
    main()

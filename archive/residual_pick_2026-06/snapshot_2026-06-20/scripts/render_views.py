"""Render the CURRENT pick scene (training env at the rest pose, cube at the spawn center) from the
three cameras -> runs/view_top.png, view_quarter.png, view_wrist.png + a combined view_all.png.
Use to eyeball the scene/table/camera geometry after a config change (e.g. real table 80x120,
top-cam 120cm from base). Writes a geometry summary to /tmp/render_views.txt (fastShutdown eats stdout).

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/render_views.py
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--settle", type=int, default=40)
ap.add_argument("--arm", default=None,
                help="override arm pose for the render: 'zero' (all joints 0) or 6 comma rad values; "
                     "default = the config rest pose. Use to match the REAL arm pose (e.g. --arm zero).")
ap.add_argument("--aim-y", type=float, default=None,
                help="override the top-cam aim Y (env frame). cam is at y~+0.56; LARGER aim_y (toward the "
                     "cam) = steeper DOWN-tilt; default = workspace center (~-0.20).")
ap.add_argument("--cube-y", type=float, default=0.06,
                help="cube env Y for the render, default = the real cube's measured spot (+0.06, sim2real viz "
                     "alignment). NOTE: forward of the trained workspace; the TRAINING cube spawns via EnvConfig.")
ap.add_argument("--wrist-aim", default=None, help="override wrist_aim_local 'x,y,z' (gripper_link frame) to tune the wrist look dir")
ap.add_argument("--wrist-mount", default=None, help="override wrist_mount_offset 'x,y,z' (gripper_link frame) to move the wrist lens")
args = ap.parse_args()

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


def grab(cam):
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return np.asarray(rgba)[..., :3].astype("uint8")


def main():
    tcfg = load_train_config(args.config)
    tcfg.env.num_envs = 1
    tcfg.env.pos_rand_xy = 0.0          # cube at the exact spawn center (table_center)
    tcfg.env.dr_enable = False          # clean scene (no obs/tcp jitter affects render anyway)
    tcfg.env.rest_rand = 0.0            # exact rest pose
    tcfg.env.yaw_rand = 0.0
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward)
    env.set_render(True)
    stage = get_current_stage()
    origin = env.env_origins[0].detach().cpu().numpy()

    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"
    if getattr(scfg, "green_bg_enabled", False):
        from isaacsim.core.api.objects import VisualCuboid
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/v_greenfloor", name="v_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], dtype=float),
                     color=np.array(scfg.green_bg_color, dtype=float))
        if getattr(scfg, "green_bg_dome", False):
            from pxr import UsdLux, Gf
            dome = UsdLux.DomeLight.Define(stage, "/World/v_greendome")
            dome.CreateColorAttr(Gf.Vec3f(*[float(x) for x in scfg.green_bg_color]))
            dome.CreateIntensityAttr(300.0)

    # aim the top cam at the ACTUAL (EnvConfig) workspace center where the cube spawns -- scfg.top_cam_target
    # uses the drifted SceneConfig workspace (-0.30); the trained/real cube center is the env workspace (-0.20).
    e0 = tcfg.env
    aim_y = args.aim_y if args.aim_y is not None else scfg.top_cam_aim_y   # saved, real-photo-aligned aim
    ws_aim = np.array([0.5 * (e0.workspace_x[0] + e0.workspace_x[1]), aim_y, e0.table_z + 0.03])
    print(f"[render_views] top_cam aim_y={aim_y:.3f} (LARGER=closer to cam=steeper down-tilt)", flush=True)
    top = Camera(prim_path="/World/v_top", name="v_top", resolution=scfg.cam.resolution)
    top.initialize(); _apply_c922(top, scfg.cam)
    top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                       orientation=look_at_quat(origin + np.array(scfg.top_cam_pos), origin + ws_aim),
                       camera_axes="usd")
    quarter = make_quarter_camera(scfg, origin=origin, prim_path="/World/v_quarter")
    if args.wrist_aim is not None:
        scfg.wrist_aim_local = tuple(float(v) for v in args.wrist_aim.split(","))
    if args.wrist_mount is not None:
        scfg.wrist_mount_offset = tuple(float(v) for v in args.wrist_mount.split(","))
    print(f"[render_views] wrist mount={tuple(round(v,4) for v in scfg.wrist_mount_offset)} "
          f"aim={tuple(round(v,4) for v in scfg.wrist_aim_local)}", flush=True)
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    cams = {"top": top, "quarter": quarter, "wrist": wrist}

    env.reset()
    for _ in range(args.settle):
        env.world.step(render=True)

    if args.arm is not None:                # override the arm pose to match the REAL rig (e.g. zero pose)
        q = (torch.zeros(6) if args.arm == "zero"
             else torch.tensor([float(x) for x in args.arm.split(",")], dtype=torch.float32))
        qpos = env.robots.get_joint_positions().clone()
        qpos[0, env.arm_idx] = q.to(qpos.device)
        env.robots.set_joint_positions(qpos)
        env.arm_target[0] = q.to(env.arm_target.device)
        env.robots.set_joint_position_targets(env._full_targets())
        for _ in range(args.settle):
            env.world.step(render=True)
        print(f"[render_views] arm overridden to {args.arm} -> {np.round(q.numpy(),3).tolist()}", flush=True)

    if args.cube_y is not None:             # move the cube to match the REAL placement (+y toward the camera)
        cx = 0.5 * (e0.workspace_x[0] + e0.workspace_x[1]); cz = e0.table_z + e0.cube_size / 2.0
        cpos = torch.tensor(origin + np.array([cx, args.cube_y, cz]), dtype=torch.float32,
                            device=env.device).unsqueeze(0)
        idx = torch.tensor([0], device=env.device)
        env.cubes.set_world_poses(positions=cpos, indices=idx)
        env.cubes.set_velocities(torch.zeros(1, 6, device=env.device), indices=idx)
        for _ in range(args.settle):
            env.world.step(render=True)
        print(f"[render_views] cube moved to env y={args.cube_y}", flush=True)

    _, st, _ = env._compute_state(env.prev_action)
    tcp = st["tcp_pos"][0].detach().cpu().numpy()
    cube = st["cube_pos"][0].detach().cpu().numpy()
    eff_line = "obs_torque=off"
    if getattr(env.cfg, "obs_torque", False):
        eff = env._measured_efforts()[0].detach().cpu().numpy()
        arm_e = eff[env.arm_idx.cpu().numpy()]; grip_e = eff[env.grip_idx.cpu().numpy()].mean()
        eff_line = (f"measured arm efforts(N*m)={np.round(arm_e,2).tolist()} grip={grip_e:.2f} "
                    f"-> scaled(x{env.cfg.torque_obs_scale})={np.round(np.append(arm_e,grip_e)*env.cfg.torque_obs_scale,2).tolist()}")

    os.makedirs(tcfg.log_dir, exist_ok=True)
    imgs = {}
    for k, cam in cams.items():
        im = grab(cam)
        if im is not None:
            imageio.imwrite(os.path.join(tcfg.log_dir, f"view_{k}.png"), im)
            imgs[k] = im
    if all(k in imgs for k in ("top", "quarter", "wrist")):
        h = min(imgs[k].shape[0] for k in imgs)
        combo = np.concatenate([imgs["quarter"][:h], imgs["top"][:h], imgs["wrist"][:h]], axis=1)
        imageio.imwrite(os.path.join(tcfg.log_dir, "view_all.png"), combo)

    e = tcfg.env
    lines = [
        f"base_pos={e.base_pos}  table_z={e.table_z}",
        f"table_size(x,y,thk)={e.table_size}  table_center(x,y)={e.table_center}",
        f"workspace_x={e.workspace_x}  workspace_y={e.workspace_y}",
        f"rest_arm={e.rest_arm}",
        f"rest tcp (env frame) = {np.round(tcp,3).tolist()}",
        f"cube spawn (env frame) = {np.round(cube,3).tolist()}",
        f"top_cam pos={np.round(np.array(scfg.top_cam_pos),3).tolist()} aim={np.round(np.array(scfg.top_cam_target),3).tolist()}",
        f"top_cam<->base dist(y) = {abs(scfg.top_cam_pos[1]-e.base_pos[1]):.3f} m",
        eff_line,
        f"saved imgs: {sorted(imgs.keys())}",
    ]
    with open("/tmp/render_views.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n"); fh.flush(); os.fsync(fh.fileno())
    print("[render_views]\n" + "\n".join(lines), flush=True)
    app.close()


if __name__ == "__main__":
    main()

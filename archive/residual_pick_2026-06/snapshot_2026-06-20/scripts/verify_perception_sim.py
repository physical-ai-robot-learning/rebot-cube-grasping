"""SIM2REAL VERIFY V2: how accurate is the red-cube perception (rebot_sim2real.perception) on rendered
images? Renders the fixed TOP camera with the cube teleported to a grid of KNOWN world positions, runs
the RGB-only ray-plane estimator, and reports the xy error vs ground truth. The camera intrinsics +
world pose are taken from the sim camera (exact), so this isolates the PERCEPTION-ALGORITHM error
(segmentation + ray-plane), separate from real-world calibration error. If error <~20mm, V1 says the
policy tolerates it -> perception path is good to deploy (after hand-eye calibration on the real rig).

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/verify_perception_sim.py
Writes /tmp/verify_perception_sim.txt.
"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from isaacsim.sensors.camera import Camera
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922
from rebot_sim2real import perception as P


def _np(x):
    """to host numpy, whether x is a torch tensor (maybe cuda), np array, or list."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def grab(cam):
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return _np(rgba)[..., :3].astype("uint8")


def main():
    tcfg = load_train_config("configs/train.yaml"); tcfg.env.num_envs = 1
    tcfg.env.dr_enable = False; tcfg.env.pos_rand_xy = 0.0
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config("configs/scene.yaml")
    plane_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0
    tc = tcfg.env.table_center

    top = Camera(prim_path="/World/r_top", name="r_top", resolution=scfg.cam.resolution)
    top.initialize(); _apply_c922(top, scfg.cam)
    top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                       orientation=look_at_quat(origin + np.array(scfg.top_cam_pos),
                                                origin + np.array(scfg.top_cam_target)), camera_axes="usd")
    for _ in range(30):
        env.world.step(render=True)
    # intrinsics + extrinsics from the sim camera (exact in sim; from calibration on the real rig)
    try:
        K = _np(top.get_intrinsics_matrix()).astype(float)
    except Exception:
        c = scfg.cam
        K = P.K_from_c922(c.focal_length_mm, c.horizontal_aperture_mm, c.vertical_aperture_mm,
                           c.resolution[0], c.resolution[1])
    # extrinsic in ROS convention (unambiguous): world->cam = view_matrix_ros; cam->world = its inverse
    view_ros = _np(top.get_view_matrix_ros()).astype(float)
    T_cam2world = np.linalg.inv(view_ros)
    print(f"[v2] K_fx={K[0,0]:.1f} fy={K[1,1]:.1f}  cam origin(world)={T_cam2world[:3,3]}  plane_z={plane_z}", flush=True)

    offsets = [(dx, dy) for dx in (-0.08, -0.04, 0.0, 0.04, 0.08) for dy in (-0.08, -0.04, 0.0, 0.04, 0.08)]
    rows = []
    for dx, dy in offsets:
        cube_w = np.array([origin[0] + tc[0] + dx, origin[1] + tc[1] + dy, origin[2] + plane_z])
        pos = env.cubes.get_world_poses()[0].clone()
        pos[0] = torch.tensor(cube_w, device=dev, dtype=pos.dtype)
        env.cubes.set_world_poses(positions=pos[:1])
        for _ in range(6):
            env.world.step(render=True)
        gt = env.cubes.get_world_poses()[0][0].detach().cpu().numpy()    # actual world cube pos
        rgb = grab(top)
        if rgb is None:
            rows.append((dx, dy, None, None)); continue
        est = P.estimate_red_cube_world(rgb, K, T_cam2world, plane_z)
        if est is None:
            rows.append((dx, dy, None, None)); continue
        err_xy = float(np.linalg.norm(est[:2] - gt[:2]))
        rows.append((dx, dy, est, err_xy))

    errs = [r[3] for r in rows if r[3] is not None]
    det = sum(1 for r in rows if r[3] is not None)
    lines = [f"PERCEPTION SIM-RENDER ACCURACY (top cam, RGB-only ray-plane). cells={len(offsets)} detected={det}",
             f"plane_z={plane_z:.3f}  K_fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}"]
    if errs:
        e = np.array(errs)
        lines.append(f"xy error vs ground-truth [mm]: mean={e.mean()*1000:.1f}  med={np.median(e)*1000:.1f}  max={e.max()*1000:.1f}")
        lines.append(f"V1 tolerance was ~20mm genuine-hold -> {'PASS' if e.mean()<0.02 else 'CHECK (mean>20mm)'}")
    else:
        lines.append("NO detections -- check red_mask thresholds / camera framing")
    lines.append("")
    for dx, dy, est, err in rows:
        es = "none" if est is None else f"({est[0]:.3f},{est[1]:.3f})"
        em = "  -" if err is None else f"{err*1000:5.1f}mm"
        lines.append(f"  off=({dx:+.2f},{dy:+.2f})  est={es}  err={em}")
    out = "\n".join(lines)
    with open("/tmp/verify_perception_sim.txt", "w") as f:
        f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
    print(out, flush=True)
    app.close()


if __name__ == "__main__":
    main()

"""SIM wrist-cam IBVS feasibility: drive the scripted approach->grasp->lift (1 env, clean, cube at
center) and at each step render the EYE-IN-HAND wrist cam + detect the red-cube HSV centroid. Validates
in sim (ground truth) that: (1) the cube stays in the wrist FOV through the approach, (2) the centroid
converges to a consistent "grasp-ready" target pixel, (3) when occlusion (red_area drop) begins.

Outputs: runs/wrist_grasp_ready.png (annotated frame at the grasp moment = the TARGET PIXEL), a few
phase frames, and a per-step centroid trace -> /tmp/wrist_servo_sim.txt.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/wrist_servo_sim.py
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--j1", type=float, default=0.0)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import imageio.v2 as imageio
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import make_wrist_camera
from rebot_sim2real.envs.pick_env import project_cube_to_wrist
from rebot_sim2real import perception as P
from rebot_sim2real.grasp_traj import (APPROACH as APPROACH_Q, GRASP as GRASP_Q, LIFT as LIFT_Q,
                                        APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS)


def _np(x):                                                   # Isaac getters can return CUDA tensors
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def grab(cam):
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return np.asarray(rgba)[..., :3].astype("uint8")          # RGB


def centroid(rgb):
    m = P.red_mask(rgb); c = P._largest_blob_centroid(m)
    area = int((m > 0).sum())
    bbc = None                                              # bbox center (OD-style; closer to geom center?)
    ys, xs = np.nonzero(m)
    if xs.size:
        bbc = np.array([(float(xs.min()) + float(xs.max())) / 2.0,
                        (float(ys.min()) + float(ys.max())) / 2.0])
    return c, area, m, bbc


def main():
    tcfg = load_train_config(args.config)
    tcfg.env.num_envs = 1; tcfg.env.pos_rand_xy = 0.0; tcfg.env.dr_enable = False
    tcfg.env.rest_rand = 0.0; tcfg.env.yaw_rand = 0.0
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage()
    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    H, W = scfg.cam.resolution[1], scfg.cam.resolution[0]
    cam = scfg.cam
    K = P.K_from_c922(cam.focal_length_mm, cam.horizontal_aperture_mm, cam.vertical_aperture_mm, W, H)
    origin = env.env_origins[0].cpu().numpy()
    plane_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0     # cube-center height (env frame)
    mount = torch.tensor(scfg.wrist_mount_offset, device=dev, dtype=torch.float32)
    aim = torch.tensor(scfg.wrist_aim_local, device=dev, dtype=torch.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    proj_cmp = []                                            # (mask_centroid, projected_center) convention check
    proj_cmp_bbox = []                                       # (bbox_center, projected_center) -- OD-style test

    arm_idx, grip_idx = env.arm_idx, env.grip_idx
    _q = lambda pose: torch.tensor([[args.j1] + list(pose[1:])], device=dev)
    APPROACH, GRASP, LIFT = _q(APPROACH_Q), _q(GRASP_Q), _q(LIFT_Q)
    GO = tcfg.env.grip_open_qpos
    env.reset()
    q = env.robots.get_joint_positions().clone(); q[:, arm_idx] = APPROACH; q[:, grip_idx] = GO
    env.robots.set_joint_positions(q); env.arm_target[:] = APPROACH; env.grip_target[:] = GO
    env.robots.set_joint_position_targets(env._full_targets())
    for _ in range(20):
        env.world.step(render=True)

    b1, b2, b3 = APPROACH_STEPS, APPROACH_STEPS + DESCEND_STEPS, APPROACH_STEPS + DESCEND_STEPS + CLOSE_STEPS
    total = b3 + LIFT_STEPS
    trace = []; key_frames = {}
    grasp_ready_uv = None
    for t in range(total):
        cur = env.robots.get_joint_positions()[:, arm_idx]
        if t < b1:      tgt, closing, phase = APPROACH, False, "approach"
        elif t < b2:    tgt, closing, phase = GRASP, False, "descend"
        elif t < b3:    tgt, closing, phase = GRASP, True, "close"
        else:           tgt, closing, phase = LIFT, True, "lift"
        a_arm = ((tgt - cur) / tcfg.env.arm_step).clamp(-1, 1)
        a_grip = torch.ones(1, 1, device=dev) if closing else -torch.ones(1, 1, device=dev)
        env.step(torch.cat([a_arm, a_grip], -1))
        _, st, _ = env._compute_state(env.prev_action)
        tcp = st["tcp_pos"][0].cpu().numpy(); cube = st["cube_pos"][0].cpu().numpy()
        rgb = grab(wrist)
        c, area, m, bbc = (None, 0, None, None) if rgb is None else centroid(rgb)
        uv = (round(float(c[0])), round(float(c[1]))) if c is not None else None
        # wrist-cam cube localization: back-project centroid -> ray-plane via the EXACT wrist cam pose
        # (get_view_matrix_ros), compare est world-xy to ground truth. Proves wrist vision can localize.
        est_err = None
        if c is not None and area > 50:
            try:
                Tcw = np.linalg.inv(_np(wrist.get_view_matrix_ros()).astype(float))
                est = P.estimate_red_cube_world(rgb, K, Tcw, plane_z + float(origin[2]))
                if est is not None:
                    est_err = round(float(np.linalg.norm((np.asarray(est) - origin)[:2] - cube[:2])), 3)
            except Exception as ex:
                if t == 0:
                    print(f"[wrist_servo_sim] localize err: {type(ex).__name__}: {ex}", flush=True)
        # analytical projection (NO render) of the cube center -> wrist pixel; compare to rendered centroid
        glp_w, glq = env.tcp_link.get_world_poses()
        glp = glp_w[:1].float() - torch.tensor(origin, device=dev, dtype=torch.float32)
        pu, pv, pdepth = project_cube_to_wrist(st["cube_pos"][:1].float(), glp, glq[:1].float(),
                                               mount, aim, fx, fy, cx, cy)
        proj_uv = (round(float(pu[0])), round(float(pv[0]))) if float(pdepth[0]) > 0 else None
        d3 = float(np.linalg.norm(cube - tcp))
        if uv is not None and proj_uv is not None:
            proj_cmp.append((np.array(uv), np.array(proj_uv), d3))
        if bbc is not None and proj_uv is not None:
            proj_cmp_bbox.append((bbc, np.array(proj_uv), d3))
        trace.append((t, phase, round(d3, 3), uv, area, est_err, proj_uv))
        # grasp-ready = the LAST descend step (cube at the jaw, fingers still open -> fully visible)
        if t == b2 - 1 and rgb is not None:
            grasp_ready_uv = uv
            ann = rgb[..., ::-1].copy()                       # to BGR for cv2 draw
            import cv2
            if m is not None:
                ann[m > 0] = (0, 255, 0)
            if uv is not None:
                cv2.drawMarker(ann, uv, (255, 0, 0), cv2.MARKER_CROSS, 40, 3)
            cv2.circle(ann, (W // 2, H // 2), 6, (0, 0, 255), 2)
            os.makedirs(tcfg.log_dir, exist_ok=True)
            cv2.imwrite(os.path.join(tcfg.log_dir, "wrist_grasp_ready.png"), ann)
        if phase in ("approach", "close") and phase not in key_frames and rgb is not None:
            key_frames[phase] = rgb
            imageio.imwrite(os.path.join(tcfg.log_dir, f"wrist_{phase}.png"), rgb)

    # per-phase centroid summary + occlusion onset (red_area collapse during close)
    errs = [r[5] for r in trace if r[5] is not None]
    mean_err = round(float(np.mean(errs)), 4) if errs else None
    def _cmp(pairs, far_only=False):
        d = np.array([float(np.linalg.norm(a - b)) for a, b, dd in pairs if (not far_only or dd > 0.05)])
        return f"mean={d.mean():.0f}px(min{d.min():.0f},max{d.max():.0f},n{len(d)})" if len(d) else "n/a"
    if proj_cmp:
        proj_line = ("PROJECTION CHECK (projected_center vs detector pixel):\n"
                     f"  ALL steps : mask-centroid {_cmp(proj_cmp)} | bbox-center {_cmp(proj_cmp_bbox)}\n"
                     f"  APPROACH  : mask-centroid {_cmp(proj_cmp, True)} | bbox-center {_cmp(proj_cmp_bbox, True)}")
    else:
        proj_line = "PROJECTION CHECK: no overlapping detections"
    lines = [f"image {W}x{H}  center=({W//2},{H//2})",
             f"GRASP-READY target pixel (cube centroid at jaw, fingers open) = {grasp_ready_uv}",
             f"wrist-cam cube-xy localization error: mean={mean_err} m over {len(errs)} visible steps",
             proj_line,
             "per-step trace (t, phase, dist3, rendered_uv, area, est_err, projected_uv):"]
    for row in trace:
        lines.append(f"  t={row[0]:3d} {row[1]:8s} dist={row[2]:.3f} uv={row[3]} area={row[4]} "
                     f"err={row[5]} proj={row[6]}")
    with open("/tmp/wrist_servo_sim.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n"); fh.flush(); os.fsync(fh.fileno())
    print("\n".join(lines[:5]), flush=True)
    seen = sum(1 for r in trace if r[4] > 50)
    print(f"[wrist_servo_sim] cube detected (area>50) in {seen}/{len(trace)} steps; "
          f"grasp_ready_uv={grasp_ready_uv}; mean wrist-localize err={mean_err}m; trace -> /tmp/wrist_servo_sim.txt", flush=True)
    app.close()


if __name__ == "__main__":
    main()

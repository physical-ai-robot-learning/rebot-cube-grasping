"""STEP 1 of the sim-trained cube detector: generate a labeled TOP-CAM dataset with VISUAL DR.

Renders the top camera with the red cube at random table positions, random arm poses, non-red
distractor cubes, and randomized lighting -> AUTO-LABELS the cube pixel (u,v) by projecting its
known world pose (FREE labels, no manual annotation). A small CNN+FPN trained on this (step 2)
replaces the brittle HSV detector with a clutter/lighting-robust (u,v) detector; the policy obs is
unchanged (same (u,v) token). v1 DR = lighting + clutter positions + arm/cube poses; color/texture
DR is a later pass.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/gen_perception_data.py --n 4000 --out data/percep_top.npz
    # quick test:  ... --n 8 --out /tmp/percep_test.npz
"""
import os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--n", type=int, default=4000, help="number of labeled samples to generate")
ap.add_argument("--out", default="data/percep_top.npz")
ap.add_argument("--save-w", type=int, default=320, help="saved image width (16:9 -> h=180)")
ap.add_argument("--n-distractors", type=int, default=4)
ap.add_argument("--settle", type=int, default=3, help="render steps per sample after re-posing")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
tcfg.env.num_envs = 1
tcfg.env.dr_enable = False          # our OWN visual DR below; no sim-dynamics DR needed for renders
tcfg.env.obs_wrist = True           # ensure the camera/render pipeline is enabled
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import numpy as np
import cv2
import torch
from pxr import UsdLux, Gf
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api.objects import VisualCuboid
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922
from rebot_sim2real.envs.pick_env import RebotPickEnv
from rebot_sim2real import perception as P


def project_uv(pt, cam, aim, fx, fy, cx, cy):
    """world point -> (u, v, depth) for a USD camera at `cam` looking at `aim` (optical = local -Z, up = +Z)."""
    fwd = aim - cam; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    z = -fwd
    x = np.cross(np.array([0.0, 0.0, 1.0]), z); x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    rel = pt - cam
    xc, yc, zc = float(rel @ x), float(rel @ y), float(rel @ z)
    depth = -zc
    if depth <= 1e-6:
        return None
    return cx + fx * (xc / depth), cy + fy * (yc / depth), depth


def main():
    tc = tcfg
    env = RebotPickEnv(tc.env, tc.reward); env.set_render(True)
    stage = get_current_stage()
    origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config(args.scene_config)

    # top camera at the aligned pose
    top = Camera(prim_path="/World/g_top", name="g_top", resolution=scfg.cam.resolution)
    top.initialize(); _apply_c922(top, scfg.cam)
    cam_pos = origin + np.array(scfg.top_cam_pos)
    cam_aim = origin + np.array(scfg.top_cam_target)
    top.set_world_pose(position=cam_pos, orientation=look_at_quat(cam_pos, cam_aim), camera_axes="usd")

    # GREEN-SCREEN background to MATCH the real rig (real deploy uses a green screen, per the captures);
    # same green floor + green dome as scripts/render_views.py so the detector trains on the real bg.
    dome = UsdLux.DomeLight.Define(stage, "/World/g_dome")
    dome.CreateIntensityAttr(300.0)
    if getattr(scfg, "green_bg_enabled", True):
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/g_greenfloor", name="g_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], float),
                     color=np.array(scfg.green_bg_color, float))
        dome.CreateColorAttr().Set(Gf.Vec3f(*[float(x) for x in scfg.green_bg_color]))

    # distractor cubes: fixed DIVERSE non-red colors, repositioned per sample
    palette = [(0.15, 0.35, 0.85), (0.20, 0.70, 0.30), (0.60, 0.25, 0.75),
               (0.85, 0.80, 0.15), (0.20, 0.65, 0.75), (0.50, 0.50, 0.55)]
    distr = []
    for i in range(args.n_distractors):
        col = palette[i % len(palette)]
        d = VisualCuboid(prim_path=f"/World/g_distr_{i}", name=f"g_distr_{i}",
                         position=np.array([0.0, 0.0, -1.0]),
                         scale=np.array([tc.env.cube_size] * 3), color=np.array(col, float))
        distr.append(d)

    W, H = scfg.cam.resolution
    fx = scfg.cam.focal_length_mm * W / scfg.cam.horizontal_aperture_mm
    fy = scfg.cam.focal_length_mm * H / scfg.cam.vertical_aperture_mm
    cx, cy = W / 2.0, H / 2.0
    plane_z = tc.env.table_z + tc.env.cube_size / 2.0
    wx, wy = tc.env.workspace_x, tc.env.workspace_y
    lo = np.array(tc.env.arm_limit_lo, float); hi = np.array(tc.env.arm_limit_hi, float)
    sw = args.save_w; sh = int(round(sw * H / W)); scale = sw / float(W)
    rng = np.random.default_rng(0)

    env.reset()
    for _ in range(40):                     # initial settle so the first render is valid
        env.world.step(render=True)

    imgs, uvs, viss = [], [], []
    n_ok = 0
    for n in range(args.n):
        # cube: random table position (a bit beyond the workspace so the detector sees the full range)
        cxw = rng.uniform(wx[0] - 0.05, wx[1] + 0.05)
        cyw = rng.uniform(wy[0] - 0.05, wy[1] + 0.15)
        cube_world = origin + np.array([cxw, cyw, plane_z])
        env.cubes.set_world_poses(positions=torch.tensor(cube_world, dtype=torch.float32).unsqueeze(0),
                                  indices=torch.tensor([0], device=env.device))
        # arm: random pose within soft limits
        q = lo + rng.random(6) * (hi - lo)
        qpos = env.robots.get_joint_positions().clone()
        qpos[0, env.arm_idx] = torch.tensor(q, dtype=torch.float32, device=qpos.device)
        env.robots.set_joint_positions(qpos)
        env.robots.set_joint_position_targets(env._full_targets())
        # distractors: random table positions
        for d in distr:
            dx = rng.uniform(wx[0] - 0.10, wx[1] + 0.10); dy = rng.uniform(wy[0] - 0.10, wy[1] + 0.20)
            d.set_world_pose(position=origin + np.array([dx, dy, plane_z]))
        dome.GetIntensityAttr().Set(float(rng.uniform(150.0, 600.0)))    # lighting DR
        # re-pin the cube last (so arm-teleport physics doesn't nudge it before the render)
        env.cubes.set_world_poses(positions=torch.tensor(cube_world, dtype=torch.float32).unsqueeze(0),
                                  indices=torch.tensor([0], device=env.device))
        for _ in range(args.settle):
            env.world.step(render=True)

        rgba = top.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            continue
        img = np.asarray(rgba)[..., :3].astype(np.uint8)
        # LABEL = the rendered cube's pixel. In sim the cube is the ONLY red (distractors are non-red),
        # so the HSV red-blob centroid is the cube's true (u,v) -- render-aligned + occlusion-aware
        # (vs analytic projection, which needs an exact camera convention). Scale full-res -> saved res.
        c = P._largest_blob_centroid(P.red_mask(img.astype(np.float32)))
        if c is not None and c[2] >= 10:
            u, v, visible = c[0] * scale, c[1] * scale, 1.0
        else:
            u = v = -1.0; visible = 0.0    # cube occluded / out of frame
        imgs.append(cv2.resize(img, (sw, sh)))
        uvs.append([u, v]); viss.append(visible)
        n_ok += 1
        if (n + 1) % 200 == 0:
            print(f"[gen] {n+1}/{args.n}  visible_frac={np.mean(viss):.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=np.asarray(imgs, np.uint8), uv=np.asarray(uvs, np.float32),
                        visible=np.asarray(viss, np.float32), res=np.array([sw, sh]))
    msg = f"[gen] saved {n_ok} samples -> {args.out}  (img {sw}x{sh}, visible_frac={np.mean(viss):.2f})"
    print(msg, flush=True)
    with open("/tmp/gen_perception.txt", "w") as fh:
        fh.write(msg + "\n"); fh.flush(); os.fsync(fh.fileno())
    app.close()


if __name__ == "__main__":
    main()

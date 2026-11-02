"""WRIST-CAM labeled dataset for the cube detector (sibling of gen_perception_data.py which does TOP).

The wrist cam is eye-in-hand -> it only sees the cube near the approach/grasp poses, so we pose the arm
at the IK grasp-table configs (lerp hover_q<->grasp_q + jitter) for random reachable cube positions, add
distractor cubes + lighting DR + the green-screen bg, render the WRIST cam, and AUTO-LABEL the cube (u,v)
by the red-blob centroid (the cube is the only red). Combine with percep_top.npz -> train ONE detector
that handles both cameras (deploy/cube_detector_io.py = "one detector, two consumers").

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/gen_perception_data_wrist.py --n 4000 --out data/percep_wrist.npz
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train_v6.yaml")   # v6 oriented IK -> the detector's training
#   viewpoints MATCH the v6 rollout (top-down cam, jaw-aligned J6 roll for rotated cubes); no grasp table.
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--out", default="data/percep_wrist.npz")
ap.add_argument("--save-w", type=int, default=320)
ap.add_argument("--n-distractors", type=int, default=6)   # >=3 RED (palette leads red) -> clutter robustness
ap.add_argument("--settle", type=int, default=3)
ap.add_argument("--gt-label", action="store_true",
                help="label with the GEOMETRIC GT (u,v) from env._a_wrist (project_cube_to_wrist on the TRUE "
                     "cube = the policy's own analytic obs) instead of the HSV blob centroid -> exact, "
                     "noise-free labels. The wrist-assist ablation showed +25pt headroom from a better wrist CNN.")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
tcfg.env.num_envs = 1
tcfg.env.dr_enable = False
tcfg.env.obs_wrist = True
tcfg.env.obs_analytic_uv = False     # RENDER the wrist cam (we need real frames)
tcfg.env.obs_top_uv = False
tcfg.env.obs_wrist_cnn = False       # we GENERATE the detector's training data -> don't run a detector here
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

import math
import numpy as np
import cv2
import torch
from pxr import UsdLux, Gf
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api.objects import VisualCuboid
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.envs.pick_env import RebotPickEnv, project_cube_to_wrist
from rebot_sim2real import perception as P


def _yaw_quat(theta):
    """yaw θ (rad) about world z -> wxyz quaternion [4] (np.float32)."""
    return np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)], dtype=np.float32)


def _image_yaw_gt(env, cube_local, theta_w):
    """In-image cube-face angle (rad) GT: project the cube center AND a small offset along the cube's world
    face direction into the WRIST cam (same projection as env._a_wrist), the image vector between them is the
    cube's edge direction as it APPEARS to the wrist (= the jaw↔cube misalignment the policy nulls). Folds
    mod 90° at train time via (cos4θ,sin4θ)."""
    dev = env.device
    face = torch.tensor([math.cos(theta_w), math.sin(theta_w), 0.0], device=dev)
    C = torch.tensor(cube_local, dtype=torch.float32, device=dev)
    Pt = C + 0.02 * face                                         # offset point along the cube +x face dir
    wp, wq = env.tcp_link.get_world_poses()
    glink = (wp[:1] - env.env_origins[:1])
    fx, fy, cx, cy = env._wrist_K
    uC, vC, _ = project_cube_to_wrist(C[None], glink, wq[:1], env._wrist_mount, env._wrist_aim, fx, fy, cx, cy)
    uP, vP, _ = project_cube_to_wrist(Pt[None], glink, wq[:1], env._wrist_mount, env._wrist_aim, fx, fy, cx, cy)
    return math.atan2(float(vP - vC), float(uP - uC))


def main():
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage()
    origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config(args.scene_config)
    assert env.wrist_cams and env.wrist_cams[0] is not None, "no wrist cam (need obs_wrist + not analytic)"
    wcam = env.wrist_cams[0]
    AIDX = env.arm_idx.tolist()

    plane_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0 + 0.002
    assert env.base is not None, "need residual_base=rule (oriented IK) to pose the arm"
    dev = env.device
    # POOL of v6 oriented poses (PROVEN viewpoints, visible~0.30): batch-solve the oriented hover/grasp configs
    # (look-at cam -> cube reliably IN-frame; jaw-aligned J6 for rotated cubes). Per sample we lerp hover<->grasp
    # + jitter. (A bare straight-down aim mis-frames the cube via the analytic-FK<->render mismatch + parallax ->
    # visible~0; the detector is orientation-agnostic, so the proven look-at viewpoints are what we keep.)
    wx0, wy0 = tcfg.env.workspace_x, tcfg.env.workspace_y
    M = int(min(args.n, 1500))
    rng0 = np.random.default_rng(0)
    pcx = torch.tensor(rng0.uniform(wx0[0], wx0[1], M), dtype=torch.float32, device=dev)
    pcy = torch.tensor(rng0.uniform(wy0[0], wy0[1], M), dtype=torch.float32, device=dev)
    pyaw = torch.tensor(rng0.uniform(-math.pi, math.pi, M), dtype=torch.float32, device=dev)
    pseed = torch.tensor([tcfg.env.rest_arm], dtype=torch.float32, device=dev).expand(M, 6)
    p_hover_q, p_grasp_q, _ = env.base.oriented_pick_poses(
        torch.stack([pcx, pcy], dim=-1), pyaw, pseed,
        tcfg.env.grasp_tilt_deg, tcfg.env.grasp_tilt_dir, iters=150)
    p_hover_q = p_hover_q.cpu().numpy(); p_grasp_q = p_grasp_q.cpu().numpy()
    pcx_n, pcy_n, pyaw_n = pcx.cpu().numpy(), pcy.cpu().numpy(), pyaw.cpu().numpy()
    # STRAIGHT-DOWN pool (DEPLOY viewpoint): TCP over the cube at a random descent height, cam DOWN; settle-hold
    # frames the cube (adv eval: 250/250 in-frame, detvis 95%). The v1 detector was look-at-ONLY -> degraded on
    # straight-down (depth 8->23mm, px 32->55). Mixing this in closes the deploy viewpoint gap. Small xy offset
    # spreads the cube across the frame; jaw-align J6 for the rotated-cube yaw.
    cube_top_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0
    z_hover = float(getattr(tcfg.env, "z_hover", 0.15))
    sd_z = rng0.uniform(cube_top_z + 0.04, cube_top_z + z_hover, M)
    sd_ang = rng0.uniform(-math.pi, math.pi, M); sd_mag = rng0.uniform(0.0, 0.3, M) * (sd_z - cube_top_z)
    sd_tgt = torch.tensor(np.stack([pcx_n + sd_mag * np.cos(sd_ang), pcy_n + sd_mag * np.sin(sd_ang), sd_z], -1),
                          dtype=torch.float32, device=dev)
    sd_down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(M, 3)
    p_sd_q = env.base.ik_oriented(pseed, sd_tgt, sd_down, iters=150, jaw_align=True, cube_yaw=pyaw).cpu().numpy()

    # green-screen bg + dome (match the real rig / the top dataset)
    dome = UsdLux.DomeLight.Define(stage, "/World/gw_dome"); dome.CreateIntensityAttr(300.0)
    if getattr(scfg, "green_bg_enabled", True):
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/gw_greenfloor", name="gw_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], float),
                     color=np.array(scfg.green_bg_color, float))
        # NOTE: light the cube with a WHITE dome (NOT green) -> the red cube renders BRIGHT (deploy lab light);
        # a green-tinted dome made the red cube dark -> red_mask caught ~16px -> the GT occlusion gate failed
        # (visible~0.01) even with the cube in-frame. The green is the FLOOR/bg only.
        dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    # palette LEADS with RED clutter: the deploy wrist OOD is RED background objects, but the old datagen used
    # only non-red distractors -> the detector never learned to REJECT red clutter (it keyed on color alone).
    # Red distractors of VARIED SIZE (distinguishable from the fixed-size cube) teach "the cube is the right-size
    # red blob on the table", and feed the depth head (apparent size -> distance).
    palette = [(0.80, 0.12, 0.12), (0.88, 0.18, 0.10), (0.75, 0.10, 0.20),    # RED clutter (deploy OOD)
               (0.15, 0.35, 0.85), (0.20, 0.70, 0.30), (0.60, 0.25, 0.75), (0.85, 0.80, 0.15)]
    distr, distr_sz = [], []
    for i in range(args.n_distractors):
        sz = float(tcfg.env.cube_size) * float(rng0.uniform(0.5, 2.2))        # varied size -> not the cube
        distr_sz.append(sz)
        d = VisualCuboid(prim_path=f"/World/gw_distr_{i}", name=f"gw_distr_{i}",
                         position=np.array([0.0, 0.0, -1.0]), scale=np.array([sz] * 3),
                         color=np.array(palette[i % len(palette)], float))
        distr.append(d)

    sw = args.save_w; sh = int(round(sw * tcfg.env.wrist_cam_res[1] / tcfg.env.wrist_cam_res[0]))
    scale = sw / float(tcfg.env.wrist_cam_res[0])
    rng = np.random.default_rng(0)
    wx, wy = tcfg.env.workspace_x, tcfg.env.workspace_y
    tc = tcfg.env.table_center

    env.reset()
    for _ in range(40):
        env.world.step(render=True)

    imgs, uvs, viss, yaws, depths = [], [], [], [], []
    n_ok = 0
    n_inframe = 0   # DIAG: env-analytic aw[2]>0.5 (cube geometrically in the wrist frame), before the red gate
    for n in range(args.n):
        j = int(rng.integers(M))                              # a pooled cube + its STRAIGHT-DOWN descent pose
        cube_world = origin + np.array([pcx_n[j], pcy_n[j], plane_z])
        theta_w = float(pyaw_n[j])                            # the cube yaw the pooled pose was solved for
        cquat = torch.tensor(_yaw_quat(theta_w), device=env.device).unsqueeze(0)
        cpos = torch.tensor(cube_world, dtype=torch.float32).unsqueeze(0)
        env.cubes.set_world_poses(positions=cpos, orientations=cquat,
                                  indices=torch.tensor([0], device=env.device))
        # arm viewpoint: 60% STRAIGHT-DOWN (deploy deep_reach view) + 40% look-at hover<->grasp lerp (clean
        # localization). The straight-down half closes the adv-eval gap (depth 23mm, px 55 on straight-down).
        if rng.random() < 0.5:
            q = p_sd_q[j] + (rng.random(6) * 2 - 1) * 0.02
        else:
            a = rng.uniform(0.0, 0.45)                          # hover->mid-descent (in-frame; grasp pose mis-frames)
            q = p_hover_q[j] * (1 - a) + p_grasp_q[j] * a + (rng.random(6) * 2 - 1) * 0.03
        qpos = env.robots.get_joint_positions().clone()
        qpos[0, AIDX] = torch.tensor(q, dtype=torch.float32, device=qpos.device)
        qpos[0, env.grip_idx] = tcfg.env.grip_open_qpos         # open jaws -> minimal cube occlusion in the wrist
        env.robots.set_joint_positions(qpos)
        env.robots.set_joint_position_targets(qpos)             # HOLD the posed q during settle (PD-drift to rest
        #   was pulling the arm off the pose over the 24 physics steps -> cube out of frame, inframe_frac~0.13)
        for d in distr:                                       # red(lead)+nonred clutter, PERIPHERAL only. NEAR-cube
            dx = rng.uniform(wx[0] - 0.10, wx[1] + 0.10)      #   identical-red clutter confused the heatmap/vis
            dy = rng.uniform(wy[0] - 0.10, wy[1] + 0.20)      #   labels (which red is THE cube?) -> v2 recall/px
            d.set_world_pose(position=origin + np.array([dx, dy, plane_z]))   # collapse. Deploy clutter = top-gated.
        dome.GetIntensityAttr().Set(float(rng.uniform(150.0, 600.0)))
        env.cubes.set_world_poses(positions=cpos, orientations=cquat,
                                  indices=torch.tensor([0], device=env.device))    # re-pin after arm teleport
        for _ in range(args.settle):
            env.world.step(render=True)

        rgba = wcam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            continue
        img = np.asarray(rgba)[..., :3].astype(np.uint8)
        env.get_obs()                                            # sets _a_wrist (GT u,v) + _wrist_K (yaw projection)
        cube_local = cube_world - origin
        if args.gt_label:
            # GEOMETRIC GT position (exact projection of the TRUE cube = env._a_wrist) + an OCCLUSION GATE.
            # P0 fix (lift-hold/detector diagnosis): the bare projection marked visible=1 whenever the cube was
            # geometrically in-frame, IGNORING jaw/arm occlusion -> "visible" labels landed on blocked/empty
            # pixels = untrainable supervision (wrist detector stuck ~66px, worse than the analytic token).
            # Keep the EXACT projected (u,v) for position, but require REAL red pixels near it (the cube actually
            # shows in the render) for visible=1 -> mirrors the top-cam labeling, removes the dominant error.
            aw = env._a_wrist[0].detach().cpu().numpy()          # [un, vn, vis] in [-1,1]
            n_inframe += int(aw[2] > 0.5)                         # DIAG
            Wc, Hc = float(tcfg.env.wrist_cam_res[0]), float(tcfg.env.wrist_cam_res[1])
            if aw[2] > 0.5:
                u_full = aw[0] * (Wc / 2.0) + Wc / 2.0
                v_full = aw[1] * (Hc / 2.0) + Hc / 2.0
                mask = P.red_mask(img.astype(np.float32))        # red near the PROJECTED cube center = the cube
                Himg, Wimg = mask.shape[:2]                       # (red distractors live elsewhere -> gate holds)
                r = max(4, int(0.03 * Wimg))                     # ~3% window around the projected center
                x0, x1 = max(0, int(u_full) - r), min(Wimg, int(u_full) + r)
                y0, y1 = max(0, int(v_full) - r), min(Himg, int(v_full) + r)
                seen = float(mask[y0:y1, x0:x1].sum()) >= 6.0    # cube actually visible (not jaw/arm-occluded)
                if seen:
                    u = u_full * scale; v = v_full * scale; visible = 1.0
                else:
                    u = v = -1.0; visible = 0.0                  # in-frame but occluded -> NOT visible (key fix)
            else:
                u = v = -1.0; visible = 0.0
        else:
            c = P._largest_blob_centroid(P.red_mask(img.astype(np.float32)))
            if c is not None and c[2] >= 10:
                u, v, visible = c[0] * scale, c[1] * scale, 1.0
            else:
                u = v = -1.0; visible = 0.0
        # YAW GT: in-image cube-face angle (rad), geometric (exact). 0 when not visible (not supervised).
        theta_img = _image_yaw_gt(env, cube_local, theta_w) if visible > 0.5 else 0.0
        # DEPTH GT (m): true cube distance from the wrist lens (project_cube_to_wrist 3rd return) = the
        # FK-INDEPENDENT Z cue (apparent size -> distance). 0 when not visible (not supervised).
        if visible > 0.5:
            Cc = torch.tensor(cube_local, dtype=torch.float32, device=dev)[None]
            wp2, wq2 = env.tcp_link.get_world_poses()
            glink2 = (wp2[:1] - env.env_origins[:1])
            fxw, fyw, cxw, cyw = env._wrist_K
            _, _, dC = project_cube_to_wrist(Cc, glink2, wq2[:1], env._wrist_mount, env._wrist_aim,
                                             fxw, fyw, cxw, cyw)
            depth_label = float(dC[0]) if hasattr(dC, "__len__") else float(dC)
        else:
            depth_label = 0.0
        imgs.append(cv2.resize(img, (sw, sh))); uvs.append([u, v]); viss.append(visible)
        yaws.append(theta_img); depths.append(depth_label)
        n_ok += 1
        if (n + 1) % 200 == 0:
            print(f"[gen-wrist] {n+1}/{args.n}  visible_frac={np.mean(viss):.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=np.asarray(imgs, np.uint8), uv=np.asarray(uvs, np.float32),
                        visible=np.asarray(viss, np.float32), yaw=np.asarray(yaws, np.float32),
                        depth=np.asarray(depths, np.float32), res=np.array([sw, sh]))
    _vd = np.asarray(depths, np.float32)[np.asarray(viss) > 0.5]
    _dmsg = f" depth[med={np.median(_vd):.3f} min={_vd.min():.3f} max={_vd.max():.3f}]m" if _vd.size else ""
    msg = (f"[gen-wrist] saved {n_ok} -> {args.out} (img {sw}x{sh}, visible_frac={np.mean(viss):.2f}, "
           f"inframe_frac={n_inframe/max(1,n_ok):.2f}){_dmsg}")
    print(msg, flush=True)
    with open("/tmp/gen_perception_wrist.txt", "w") as fh:
        fh.write(msg + "\n"); fh.flush(); os.fsync(fh.fileno())
    app.close()


if __name__ == "__main__":
    main()

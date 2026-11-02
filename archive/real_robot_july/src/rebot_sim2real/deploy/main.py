"""reBot real-robot deployment main program (promoted version of the old debug_deepreach.py).

The setup/build block (593-982) of the old debug_deepreach.py `main()` was moved here: argument parsing → camera resolution →
config·base(FK/IK)·HW-limit intersection → front camera calibration → detectors (OWLv2/HSV; CNN dropped) → AsyncPerception
thread → robot_io / rest setup → dashboard → shaper·Blackboard·(opt-in) IO thread → RobotIoFacade →
ControlLoop assembly → loop.run().

The dashboard is embedded (HTTP thread inside the main process; main_dashboard module). Run: `python deploy/main.py …`.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from rebot_sim2real.rbot_calculator.base_controller import RuleBaseController
from rebot_sim2real.rbot_calculator.config import load_train_config

from . import kinematics
from .blackboard import Blackboard
from .cli import load as load_config
from .control.control_loop import ControlLoop
from .control.robot_io_facade import RobotIoFacade
from .io.io_thread import RobotIoThread
from .io.robot_io import (DamiaoRobotIO, HW_LIMIT_DEG, JAW_FWD, JointSlew, LEROBOT_ARM,
                          OFFSET_RAD, SIGN)
from .main_dashboard import DebugDashboard
from .vision import openvocab, perception_thread
from .vision.perception_thread import (AsyncPerception, _resolve_cam_index,
                                       wrist_uvn_ov, wrist_uvn_redmask)

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"


def _build_wrist_uvn(wrist_ov):
    """Builds the wrist (u,v) detection callback (passed to AsyncPerception). (input: wrist_ov or None → output: rgb→(un,vn,vis,depth,cos4,sin4))
    Box-gated HSV if OWLv2 (--wrist-openvocab) is present, otherwise HSV redmask. CNN dropped, so depth/yaw(cos4,sin4)=0
    → Align falls back to seg edge-voting yaw. deep_reach_step consumes only (un,vn,vis)."""
    def wrist_uvn(rgb):
        if rgb is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if wrist_ov is not None:                    # OWLv2 box-gated HSV wrist (u,v); no depth/yaw
            un, vn, vis = wrist_uvn_ov(rgb, wrist_ov)
            return un, vn, vis, 0.0, 0.0, 0.0
        un, vn, vis = wrist_uvn_redmask(rgb)        # HSV redmask fallback
        return un, vn, vis, 0.0, 0.0, 0.0
    return wrist_uvn


def _apply_hw_limit_intersection(base):
    """Intersects the base IK limits (cfg.arm_limit_*) with the motor HW clamp (robot_io.HW_LIMIT_DEG).
    IK proposes aim/roll the HW cannot reach → the motor np.clip silently truncates it, causing silent mis-aim/mis-roll (especially at the J1-yaw edge,
    J6 roll >±90). With the intersection, IK never proposes unreachable joints and the resid test in _danger correctly
    catches over-extension. (old debug_deepreach 611-625)"""
    for j, lerobot_joint in enumerate(LEROBOT_ARM):
        lo_deg, hi_deg = HW_LIMIT_DEG[lerobot_joint]
        s1 = float(SIGN[j]) * np.deg2rad(lo_deg) + float(OFFSET_RAD[j])   # sim_rad mapping
        s2 = float(SIGN[j]) * np.deg2rad(hi_deg) + float(OFFSET_RAD[j])
        base.lo[j] = max(float(base.lo[j]), min(s1, s2))
        base.hi[j] = min(float(base.hi[j]), max(s1, s2))
    print(f"[deploy] base IK limits ∩ HW clamp -> lo°={np.round(np.degrees(base.lo.cpu().numpy())).astype(int).tolist()} "
          f"hi°={np.round(np.degrees(base.hi.cpu().numpy())).astype(int).tolist()} (J4 lo→-80, J6→±90 = no silent motor clamp)",
          flush=True)


def _apply_wrist_calib(base, path):
    """OPT-IN wrist camera calibration override (deploy only; sim/config unchanged). base.wrist_mount/aim[+focal] + WRIST_*_A signs.
    (old 629-642)"""
    calib = np.load(path)
    mount = torch.tensor(calib["wrist_mount_offset"], dtype=torch.float32, device=base.device)
    aim = torch.tensor(calib["wrist_aim_local"], dtype=torch.float32, device=base.device)
    base.wrist_mount = mount
    base.wrist_aim = aim
    base.cam_ax_local = (aim - mount) / (aim - mount).norm()   # read by _cam_axis/_project_wrist_unvn — keep in sync
    if bool(calib.get("fit_focal", False)):
        scale = float(calib["focal_scale"])
    else:
        scale = 1.0
    if scale != 1.0:
        fx, fy, ux, uy = base.wrist_K
        base.wrist_K = (fx * scale, fy * scale, ux, uy)
    kinematics.set_wrist_calib(float(calib["sign_u"]), 0.0, float(calib["sign_v"]), 0.0)   # physical fit → affine = pure sign
    print(f"[deploy] ⭐ WRIST CALIB applied from {path}: mount={[round(v,4) for v in mount.tolist()]} "
          f"aim={[round(v,4) for v in aim.tolist()]} focal×{scale:.3f}.", flush=True)


def _lift_rest_pose(base, rest_np, rest_lift):
    """Solves via IK for the joints of the REST hover pose raised by rest_lift (m) along TCP +z (margin before descent). (old 756-766)"""
    q = torch.tensor(rest_np, dtype=torch.float32).unsqueeze(0)
    target = base.fk_world(q)[0].clone()
    target[2] += float(rest_lift)
    down = torch.tensor([[0.0, 0.0, -1.0]])
    for _ in range(24):                                # iterate clamped IK → reach the full rest-lift
        q = base._ik_step6(q, target.unsqueeze(0), down)
        if float(base.fk_world(q)[0][2]) >= float(target[2]) - 0.004:
            break
    lifted = q[0].numpy().astype(float)
    print(f"[deploy] REST lifted +{rest_lift * 100:.0f}cm -> hover TCP z {float(base.fk_world(q)[0][2]):.3f}", flush=True)
    return lifted


def _build_openvocab(dcfg):
    """With --front-openvocab / --wrist-openvocab, builds the OWLv2 ROI providers (front_ov, wrist_ov). (old 705-728)
    (output: (front_ov, wrist_ov) — each an OpenVocabROI or None). Both streams share one shared model."""
    if not (dcfg.front_openvocab or dcfg.wrist_openvocab):
        return None, None
    if dcfg.detector_device is not None:               # OWLv2 device: flag first, otherwise cuda-if-available
        ov_dev = dcfg.detector_device
    elif torch.cuda.is_available():
        ov_dev = "cuda"
    else:
        ov_dev = "cpu"
    model_name = dcfg.front_ov_model or openvocab.DEFAULT_MODEL
    ov_model = openvocab.OpenVocabModel(model=model_name, device=ov_dev, size=dcfg.front_ov_size,
                                        do_compile=dcfg.front_ov_compile)
    print(f"[deploy] open-vocab OWLv2 model on {ov_dev} ({ov_model.name}); query LIVE-editable via dashboard box.", flush=True)
    front_ov = None
    wrist_ov = None
    if dcfg.front_openvocab:
        front_ov = openvocab.OpenVocabROI(ov_model, score_thresh=float(dcfg.front_ov_thresh),
                                          period=float(dcfg.front_ov_period), name="front")
        print(f"[deploy] FRONT open-vocab ON (thresh={dcfg.front_ov_thresh}, re-detect≥{dcfg.front_ov_period}s) "
              "-> dynamic ROI, HSV centroid inside box.", flush=True)
    if dcfg.wrist_openvocab:
        wrist_ov = openvocab.OpenVocabROI(ov_model, score_thresh=float(dcfg.front_ov_thresh),
                                          period=float(dcfg.front_ov_period), name="wrist")
        print("[deploy] WRIST open-vocab ON -> wrist (u,v) = OWLv2-box-gated HSV (no depth/yaw; Align uses seg yaw).", flush=True)
    return front_ov, wrist_ov


def main(argv=None):
    """Deployment program entry point: build all collaborators → ControlLoop.run(). (input: argv or None=sys.argv)"""
    dcfg = load_config(argv)
    perception_thread.USE_MJPG = bool(dcfg.mjpg)        # MJPG transfer (both cameras) → USB2 bandwidth ~10x↓ → ~30fps

    # ---- Camera resolution (udev symlink → V4L2 index). FRONT required, WRIST optional (front-only if absent). ----
    front_idx = _resolve_cam_index(dcfg.front_dev)
    if str(dcfg.wrist_dev).isdigit() or os.path.exists(dcfg.wrist_dev):
        wrist_idx = _resolve_cam_index(dcfg.wrist_dev)
    else:
        wrist_idx = None
    if wrist_idx is None:
        print(f"[deploy] WARNING: wrist cam '{dcfg.wrist_dev}' NOT found -> FRONT-ONLY (wrist_uv=0). "
              "Reconnect + restart to enable it.", flush=True)
    print(f"[deploy] cameras: front='{dcfg.front_dev}'->/dev/video{front_idx}  "
          f"wrist={('/dev/video' + str(wrist_idx)) if wrist_idx is not None else 'MISSING (front-only)'}", flush=True)

    # ---- config + base (FK/DLS-Jacobian, same as sim) + HW-limit intersection + pedestal/step + wrist-calib override ----
    tc = load_train_config(dcfg.config)
    cfg = tc.env
    cfg.num_envs = 1
    device = torch.device("cpu")
    base = RuleBaseController(cfg, device)
    _apply_hw_limit_intersection(base)
    base.base_pos[2] = float(dcfg.base_z)              # PEDESTAL: real J1 base z → FK matches reality
    base.arm_step = float(dcfg.rule_step)              # per-command joint clamp = speed knob
    print(f"[deploy] base_pos z={dcfg.base_z:+.3f} (config {cfg.base_pos[2]:+.3f}) · rule_step={dcfg.rule_step:.3f}", flush=True)
    if dcfg.wrist_calib:
        _apply_wrist_calib(base, dcfg.wrist_calib)

    # ---- front camera calibration + cube plane z + auto-z derived values ----
    calib = np.load(dcfg.calib)
    K = calib["K"].astype(float)
    T_cam2world = calib["T_cam2world"].astype(float)
    if "dist" in calib:
        dist = calib["dist"].astype(float).ravel()    # real lens distortion (checkerboard) → undistort detection pixels
    else:
        dist = None
    if dist is not None and np.any(dist != 0.0):
        print(f"[deploy] front-cam undistort ON (dist={np.round(dist, 5).tolist()})", flush=True)
    else:
        print("[deploy] front-cam dist=0/absent (approx intrinsics) -> XY has position-dependent bias.", flush=True)
    plane_z = cfg.table_z + cfg.cube_size / 2.0

    # ---- Detectors: OWLv2 open-vocab (opt-in) + HSV. (all CNN detectors dropped) ----
    front_ov, wrist_ov = _build_openvocab(dcfg)

    # ---- robot_io + REST (optional rest-lift) + initial lift-up ----
    if dcfg.arm_vlim > 0:
        arm_vlim = dcfg.arm_vlim
    else:
        arm_vlim = None
    io = DamiaoRobotIO(cfg, URDF, grip_force_ratio=dcfg.grip_force, arm_vlim_rad_s=arm_vlim)
    io.connect()
    io.base_pos[2] = float(dcfg.base_z)                # CRITICAL: both FK chains (base_ctrl·robot_io) at the same base_link z
    print(f"[deploy] gripper FORCE_POS ratio={dcfg.grip_force:.2f} (clamp-force cap; robot_io hard-clamps ≤0.18).", flush=True)
    rest_np = np.array(cfg.rest_arm, dtype=float)
    if dcfg.rest_lift > 0.0:
        rest_np = _lift_rest_pose(base, rest_np, dcfg.rest_lift)
    io.initial_liftup(rest_np, cfg.grip_open_qpos)     # scripted → REST hover, gripper open

    # ---- Dashboard (embedded HTTP thread) ----
    dash = DebugDashboard(dcfg.port)
    dash.start()

    # ---- AsyncPerception thread (always on): camera grab + detection + front-Z + annotation at camera rate ----
    if dcfg.front_roi:
        front_roi = tuple(float(x) for x in dcfg.front_roi.split(","))
        assert len(front_roi) == 4, "--front-roi must be 'y0,y1,x0,x1'"
        print(f"[deploy] HSV front WORKSPACE ROI={front_roi} · red s_min={dcfg.front_red_smin}", flush=True)
    else:
        front_roi = None
    wrist_uvn = _build_wrist_uvn(wrist_ov)
    pt = AsyncPerception(front_idx, wrist_idx, dcfg.front_focus, dcfg.wrist_focus, wrist_uvn,
                         K, T_cam2world, plane_z, float(dcfg.top_bias), dash, dist=dist, hz=30.0,
                         front_exposure=dcfg.front_exposure, wrist_exposure=dcfg.wrist_exposure,
                         x_corr=float(dcfg.x_corr),
                         front_roi=front_roi, front_smin=float(dcfg.front_red_smin),
                         front_ov=front_ov, wrist_ov=wrist_ov)
    pt.start_running()
    print(f"[deploy] ASYNC perception ON: cameras in a background thread; control loop @ {dcfg.control_hz:g}Hz.", flush=True)

    # ---- Motion shaper + Blackboard + (opt-in) IO thread + RobotIoFacade ----
    dt = 1.0 / float(dcfg.control_hz)
    shaper = JointSlew(rest_np, dt, v_max=float(dcfg.vmax), a_max=float(dcfg.amax))   # initialized at REST (after liftup)
    use_shaper = (dcfg.arm_vlim <= 0)                  # with motor-vlim glide (default) the motor smooths → send IK target directly
    bb = Blackboard()
    io_tj = None
    if dcfg.io_thread:
        io_shaper = JointSlew(rest_np, dt, v_max=float(dcfg.vmax), a_max=float(dcfg.amax))   # the thread's own shaper
        io_tj = RobotIoThread(io, bb, io_shaper, use_shaper=use_shaper, cmd_lpf_tau=float(dcfg.cmd_lpf_tau),
                              action_chunk=int(dcfg.action_chunk), chunk_decay=float(dcfg.chunk_decay),
                              rest_np=rest_np, grip_open=cfg.grip_open_qpos, hz=float(dcfg.control_hz))
        io_tj.start()
        import time
        deadline = time.time() + 3.0                   # wait for the first proprio publish (before the loop reads)
        while bb.proprio.get("arm_qpos") is None and time.time() < deadline:
            time.sleep(0.02)
        print("[deploy] ⭐ --io-thread ON: RobotIoThread owns the bus; control reads/writes the blackboard.", flush=True)
    facade = RobotIoFacade(io, bb, io_tj, shaper, use_shaper=use_shaper, cmd_lpf_tau=float(dcfg.cmd_lpf_tau),
                           chunk_n=int(dcfg.action_chunk), chunk_decay=float(dcfg.chunk_decay), rest_np=rest_np)

    # ---- Control loop assembly + run ----
    loop = ControlLoop(cfg, dcfg, base, facade, dash, pt, bb, K=K, T_cam2world=T_cam2world, dist=dist,
                       plane_z=plane_z, front_ov=front_ov, wrist_ov=wrist_ov, rest_np=rest_np,
                       jaw_fwd_fallback=JAW_FWD)
    loop.run()


if __name__ == "__main__":
    main()

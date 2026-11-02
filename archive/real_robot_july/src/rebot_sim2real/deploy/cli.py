"""Deployment (deploy) runtime argument definitions + typed config object.

The ~200 argparse calls at the top of the old debug_deepreach.py `main()` were split into this module.
  - build_parser()  : builds the argparse.ArgumentParser (single source of all defaults and help text).
  - DeployConfig    : immutable dataclass holding the parse result (field name = argparse dest).
                      Defaults are not duplicated here; they live only in argparse and are filled via from_namespace.
                      → only the field name exists in two places (parser/dataclass), and a mismatch surfaces immediately
                        as a TypeError at construction time in load().
  - load(argv)      : parse → DeployConfig.

⚠️ The CNN detectors (sd3 descent · seg-align · front CNN) were dropped in this refactor (OpenVocab finalized).
   Therefore the old --wrist-detector / --align-seg / --align-detector / --front-detector /
   --lookat-ibvs / --lookat-vn / --yaw-src flags were removed. wrist (u,v)=OWLv2 or HSV redmask,
   yaw fixed to edge-voting (length-weighted circular mean of the seg mask's boundary edges).

The CLI --help text is the operator interface and is kept concise in English; the meaning of each value (unit/role/input→effect)
lives in the comments on the DeployConfig fields.

⭐ Defaults = this robot's HW-tuned working config (docs/31 launch values baked in). So `python deploy/main.py` alone
   brings up the working setup. Detection/modes (front/wrist-openvocab·track·io-thread·autograsp-yaw) default ON (disable with --no-…),
   motion (vmax5·amax8·cmd-lpf-tau0.15·action-chunk20·chunk-decay0.9·lookat-j5-rate0.05)·grip-force0.05·
   safety-reach-resid0.15·front-roi are working values too. ⚠️ calib-dependent defaults (grasp-zgap-io·z-reach-slope/-ref·x-corr)
   are tied to the current robot calib, so update these defaults when recalibrating (fk_verify_z refit).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


# ---------------------------------------------------------------------------------------------------
# argparse parser: single source of defaults and help text for all deployment flags.
# ---------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the deployment argument parser. (input: none → output: fully configured ArgumentParser)"""
    ap = argparse.ArgumentParser(description="reBot real-robot deployment control program (OpenVocab perception + rule-based deep-reach grasp)")

    # ---- Runtime / connection ----
    ap.add_argument("--config", default="configs/deploy_env.yaml",
                    help="Training config YAML (source of mechanism/camera/joint parameters).")
    ap.add_argument("--calib", default="deploy/calib.npz",
                    help="front camera intrinsic/extrinsic parameters npz (K, T_cam2world, dist).")
    ap.add_argument("--port", type=int, default=8899, help="Dashboard HTTP port.")
    ap.add_argument("--control-hz", type=float, default=100.0, dest="control_hz",
                    help="Control loop Hz. Cameras/detection always run in a separate perception thread (~30Hz); the control loop runs independently at this Hz"
                         "(CAN read+write ~2.3ms → headroom for 100Hz) → denser setpoints.")
    ap.add_argument("--front-dev", default="/dev/video_front", dest="front_dev",
                    help="front (fixed) camera: integer index or /dev/video* path (default udev symlink).")
    ap.add_argument("--wrist-dev", default="/dev/video_wrist", dest="wrist_dev",
                    help="wrist (eye-in-hand) camera: integer index or /dev/video* path. Swap them if the two panels are reversed.")

    # ---- Z depth / FK correction / grasp depth ----
    ap.add_argument("--base-z", type=float, default=0.010, dest="base_z",
                    help="World z of URDF base_link (FK root) = base_pos.z. Measured J1 axis +0.095 → base_pos.z=+0.010.")
    ap.add_argument("--grasp-zgap-io", type=float, default=-0.04, dest="grasp_zgap_io",
                    help="Target z_gap_io at grasp-ready (exact height with FK_Z_CORR correction) = auto-z setpoint. 0≈cube center, "
                         "lower = deeper. Gate=|z_gap_io - this value|<0.015. Default -0.04 (2026-07-23 HW: firm grasp depth at the working "
                         "radial; real-robot tuned value baked in. Distance-independence requires a reach-slope refit).")
    ap.add_argument("--z-bias", type=float, default=0.05, dest="z_bias",
                    help="Descent depth below the cube center in BASE_CTRL space (auto-z initial value; fixed value with --no-auto-z). HW ~0.05.")
    ap.add_argument("--no-auto-z", action="store_true", dest="no_auto_z",
                    help="Disable auto-z (live-measure the base-FK vs robot_io z difference → feedforward) → use a fixed --z-bias.")
    ap.add_argument("--z-reach-slope", type=float, default=0.0, dest="z_reach_slope",
                    help="Distance-dependent FK-z correction (m per m radial). Subtracts slope*(radial - --z-reach-ref) from tcp_io.z "
                         "so z_gap_io is a consistent real height at every distance. Default 0 (refit after the 2026-07-23 joint_map fix: "
                         "the old +0.0676 was stale on top of the identity-offset; after correction the working-area residual is a constant -1.5cm→absorbed into FK_Z_CORR +0.015, "
                         "slope kept at 0 due to inconsistent near-radial data). Refit with src/tools/fk_verify_z.py after recalibration. 0=OFF.")
    ap.add_argument("--z-reach-ref", type=float, default=0.486, dest="z_reach_ref",
                    help="Radial distance (m) from the J1 axis where the reach correction is 0. Default 0.486=fit mean radial.")

    # ---- FK-touch verification (semi-automatic table-touch) ----
    ap.add_argument("--fk-touch-step", type=float, default=0.002, dest="fk_touch_step",
                    help="'FK-touch' button: straight-descent step per tick (m). Smaller = gentler contact.")
    ap.add_argument("--fk-touch-dtau", type=float, default=2.0, dest="fk_touch_dtau",
                    help="FK-touch contact threshold: |J2/J3 torque - hover baseline| (N*m). Above free-descent drift, below the contact spike.")
    ap.add_argument("--fk-touch-max", type=float, default=0.30, dest="fk_touch_max",
                    help="FK-touch safety backstop: max descent (m) at which to abort if there is no contact. Must exceed the hover height (rest~25cm).")

    # ---- Gripper ----
    ap.add_argument("--grip-force", type=float, default=0.05, dest="grip_force",
                    help="Gripper FORCE_POS torque/current ratio [0,1] = clamp-force upper bound. ⚠️robot_io hard-clamps to 0.18"
                         "(broken twice by over-torque). Default 0.05 (working value).")
    ap.add_argument("--grip-torque-target", type=float, default=0.8, dest="grip_torque_target",
                    help="Closed-loop grasp target: 'close' gently closes until the measured torque (grip_tau) reaches this value, then holds. "
                         "Default 0.8 (firm-gentle; above the is_grasped threshold 0.5, well below the ~1.86 that broke it; the servo clamps to 1.2).")

    # ---- Cameras / transport ----
    ap.add_argument("--mjpg", action=argparse.BooleanOptionalAction, default=True, dest="mjpg",
                    help="Open both cameras as MJPG (compressed). YUYV 720p x2 saturates USB2 (~10fps); MJPG is ~10x smaller → ~30fps "
                         "(K/intrinsics unchanged). Default ON; --no-mjpg=raw YUYV.")
    ap.add_argument("--wrist-focus", type=float, default=None, dest="wrist_focus",
                    help="Fix wrist C922 focus (AF always OFF; focus breathing→focal change→(u,v)→world xy drift). "
                         "None=AF off + fix the starting value.")
    ap.add_argument("--front-exposure", type=float, default=None, dest="front_exposure",
                    help="front camera manual exposure (auto whites out this rig). ~39-156, 78 typical. None=auto.")
    ap.add_argument("--wrist-exposure", type=float, default=None, dest="wrist_exposure",
                    help="wrist camera manual exposure (same white-out risk). None=auto.")
    ap.add_argument("--front-focus", type=float, default=None, dest="front_focus",
                    help="Fix front camera focus (AF always OFF; keeps the calibrated K valid). None=AF off + fix the starting value.")

    # ---- Perception: OpenVocab (OWLv2) + HSV (CNN detectors dropped) ----
    ap.add_argument("--front-roi", default="0.4,1.0,0.0,1.0", dest="front_roi",
                    help="front HSV cube workspace limit 'y0,y1,x0,x1' (fractions). Ignore red outside the box (excludes hands/skin/background). "
                         "e.g. '0.4,1.0,0.0,1.0'=drop the top 40%%. OFF=full frame.")
    ap.add_argument("--front-red-smin", type=float, default=0.45, dest="front_red_smin",
                    help="front cube HSV red saturation lower bound (default 0.45). Raise (0.6) to exclude low-saturation skin.")
    ap.add_argument("--front-openvocab", action=argparse.BooleanOptionalAction, default=True, dest="front_openvocab",
                    help="front: use OWLv2 open-vocab as a dynamic ROI (box→saturation centroid inside the box=mm). zero-shot (transfers to the real robot), "
                         "excludes red distractors. OFF=HSV baseline.")
    ap.add_argument("--front-ov-model", default=None, dest="front_ov_model",
                    help="OWLv2 checkpoint (default google/owlv2-base-patch16-ensemble).")
    ap.add_argument("--front-ov-thresh", type=float, default=0.25, dest="front_ov_thresh",
                    help="OWLv2 score threshold (default 0.25; real cube 0.5-0.7, distractors <0.1).")
    ap.add_argument("--front-ov-period", type=float, default=0.5, dest="front_ov_period",
                    help="Minimum OWLv2 re-detection interval (s, default 0.5≈2Hz). In between, HSV runs every frame inside the cached box.")
    ap.add_argument("--front-ov-size", type=int, default=None, dest="front_ov_size",
                    help="OWLv2 square input px (default=model 960). Smaller = faster forward (only ~10%% gain; owlvit-p32 recommended). "
                         "Smaller degrades detection of distant cubes.")
    ap.add_argument("--front-ov-compile", action="store_true", dest="front_ov_compile",
                    help="torch.compile the open-vocab model (shared by front+wrist). owlvit-p32 forward 5ms; first-infer compile ~30s.")
    ap.add_argument("--wrist-openvocab", action=argparse.BooleanOptionalAction, default=True, dest="wrist_openvocab",
                    help="Also use OWLv2 box-gated HSV centroid for wrist (u,v). Shares the front OWLv2 model + --front-ov-thresh/period.")
    ap.add_argument("--detector-device", default=None, dest="detector_device",
                    help="OWLv2 open-vocab model device ('cuda' default, 'cpu' if absent). GPU=prevents async-perception STALE fail-stop.")
    ap.add_argument("--top-bias", type=float, default=0.0, dest="top_bias",
                    help="LEGACY front-cam oblique cube-xy shift (m); absorbed by the real checkerboard intrinsics→default 0. "
                         "Pushes the captured xy along the calibrated cam→cube direction.")
    ap.add_argument("--x-corr", type=float, default=0.0235, dest="x_corr",
                    help="Constant world-X re-centering (m) = extrinsic-x residual left after the real intrinsics calibration. Measured on this rig (8pt grid): "
                         "front cube-x flat -2.35cm→+0.0235 re-center (residual ~3.4mm). Re-measure if the extrinsics change.")

    # ---- wrist camera calibration (calib_wrist sweep → fit_wrist_calib.py) ----
    ap.add_argument("--wrist-calib", default=None, dest="wrist_calib",
                    help="OPT-IN wrist camera calibration npz (fit_wrist_calib.py). Overrides base.wrist_mount/aim[+focal]. "
                         "OFF by default (working baseline untouched).")
    ap.add_argument("--calib-offset", type=float, default=0.06, dest="calib_offset",
                    help="calib_wrist: lateral TCP offset (m) from the cube at the look-at hover → optical axis ~15deg tilt (avoids gimbal).")
    ap.add_argument("--calib-j4", type=float, default=0.12, dest="calib_j4",
                    help="calib_wrist: J4 (wrist pitch) sweep half-amplitude (rad). Moves the cube vertically in view.")
    ap.add_argument("--calib-j1", type=float, default=0.10, dest="calib_j1",
                    help="calib_wrist: J1 (base yaw) sweep half-amplitude (rad). Key excitation for identifying extrinsics + u-sign (keep small).")

    # ---- Control / motion smoothing ----
    ap.add_argument("--rule-step", type=float, default=0.45, dest="rule_step",
                    help="deep_reach IK per-command joint clamp (rad). With --arm-vlim ON, used to keep the IK target ahead of the motor glide.")
    ap.add_argument("--vmax", type=float, default=5.0, dest="vmax",
                    help="JointSlew acceleration-limited shaper max joint velocity (rad/s). Keep >= rule_step*rate (IK has speed authority).")
    ap.add_argument("--amax", type=float, default=8.0, dest="amax",
                    help="JointSlew max joint acceleration (rad/s^2, only when --arm-vlim 0). Lower = smoother start/stop.")
    ap.add_argument("--arm-vlim", type=float, default=0.0, dest="arm_vlim",
                    help="ARM speed = motor POS_VEL speed limit (rad/s). Driver hardcodes vlim=32 (snap→jerky); with this value "
                         "the motor glides between setpoints. Keep rule_step*rate > this value. 0=OFF (soft JointSlew shaper).")
    ap.add_argument("--cmd-lpf-tau", type=float, default=0.15, dest="cmd_lpf_tau",
                    help="First-order LPF time constant (s) for joint setpoints, ahead of JointSlew. rate-correct. Removes high-frequency jitter/jerk. "
                         "0=OFF. look-at uses 0.12-0.20.")
    ap.add_argument("--action-chunk", type=int, default=20, dest="action_chunk",
                    help="action chunking (temporal ensemble): command = exponentially weighted mean of the last N IK targets. 1=OFF. 3-6 recommended.")
    ap.add_argument("--chunk-decay", type=float, default=0.9, dest="chunk_decay",
                    help="--action-chunk exponential weighting of older targets (0<d<=1; latest=1, age-k=d^k). Smaller = more emphasis on the latest. Default 0.9.")
    ap.add_argument("--io-thread", action=argparse.BooleanOptionalAction, default=True, dest="io_thread",
                    help="Run Robot-IO in a separate thread (RobotIoThread): control reads proprio and writes raw commands to the Blackboard, "
                         "the IO thread handles shaper/LPF/chunk + command + grip force-servo + scripted moves. OFF=synchronous single loop.")

    # ---- Grasp offset / target correction ----
    ap.add_argument("--jaw-fwd", type=float, default=0.01, dest="jaw_fwd",
                    help="JAW-FORWARD correction (m): the real jaw closes this far ahead of the base-FK finger-midpoint along the gripper-forward axis (rotates with J1) "
                         "→ shift the deep-reach XY target back along that axis. Working value 0.01. (not robot_io.JAW_FWD=0.06)")
    ap.add_argument("--lookat-jaw-fwd", type=float, default=None, dest="lookat_jaw_fwd",
                    help="approach grasp offset (m, gripper-forward axis). Aims the kinematic TCP this far ahead of the cube. "
                         "None→robot_io.JAW_FWD(0.06). approach has top_bias=0, so it carries the whole offset.")
    ap.add_argument("--cube-x-off", type=float, default=0.0, dest="cube_x_off",
                    help="World x offset (m) added to the detected cube target. Cancels systematic XY bias (both deep_reach servo + grasp gate).")
    ap.add_argument("--cube-y-off", type=float, default=0.0, dest="cube_y_off",
                    help="World y offset (m) added to the detected cube target. Use a negative value to pull back if the TCP lands at cube +y.")

    # ---- Lift / rest ----
    ap.add_argument("--lift-step", type=float, default=0.30, dest="lift_step",
                    help="Total TCP rise height per 'Lift +z' click (m, default 0.30=clears the table). Clamped to LIFT_Z_MAX. Speed is --lift-rate.")
    ap.add_argument("--lift-rate", type=float, default=0.01, dest="lift_rate",
                    help="'Lift +z' smooth rise speed: the TCP-z target rises this much per tick toward --lift-step, then holds. "
                         "Independent of tracking/descend --vmax/amax. Default 0.01 (~0.3m/s@30Hz).")
    ap.add_argument("--rest-lift", type=float, default=0.11, dest="rest_lift",
                    help="Raise the REST hover pose this much (m, IK) along TCP +z above cfg.rest_arm. Default 0.11 (more margin). 0=config as-is.")

    # ---- Track / Follow ----
    ap.add_argument("--track", action=argparse.BooleanOptionalAction, default=True, dest="track",
                    help="Live cube tracking: cube_world is updated every frame from the fixed front camera (no re-Capture needed). "
                         "DEADBAND (--track-deadband) ignores jitter. OFF=one-shot Capture baseline.")
    ap.add_argument("--track-deadband", type=float, default=0.015, dest="track_deadband",
                    help="--track: the live front estimate must move more than this (m) from cube_world to update. Excludes front jitter. Default 0.015.")
    ap.add_argument("--follow-z", type=float, default=0.16, dest="follow_z",
                    help="'Follow' hover height: this far (m) above the cube, camera straight-down, tracks cube_world. No descent/look-at. Default 0.16.")
    ap.add_argument("--follow-deadband", type=float, default=0.010, dest="follow_deadband",
                    help="'Follow': the (offset) cube xy must move more than this (m) to retarget. Prevents chatter. Default 0.010.")
    ap.add_argument("--follow-iters", type=int, default=40, dest="follow_iters",
                    help="'Follow': ik_oriented iterations/loop toward the hover pose (the shaper limits the actual speed). Default 40.")
    ap.add_argument("--follow-low", type=float, default=0.07, dest="follow_low",
                    help="'Follow' descent floor: when the cube is STABLE, descend straight from the +follow-z hover down to this height (m, auto-z surface) "
                         "above the cube. If it moves, rise to hover and re-track. Default 0.07.")
    ap.add_argument("--follow-descend-step", type=float, default=0.004, dest="follow_descend_step",
                    help="'Follow': hover→follow-low descent ramp when the cube is STABLE (m/tick). Extra smoothing by shaper/LPF. Default 0.004.")
    ap.add_argument("--follow-move", type=float, default=0.04, dest="follow_move",
                    help="'Follow' RISE threshold (m): only real cube movement (xy>this value) rises to hover and re-approaches. deadband..move only "
                         "re-tracks xy during descent (so front jitter does not cause rise/descend cycles). Must exceed front jitter. Default 0.04.")
    ap.add_argument("--follow-wrist-center", action=argparse.BooleanOptionalAction, default=False, dest="follow_wrist_center",
                    help="'Follow' (opt-in): keep the cube near the wrist center with the wrist camera while STABLE/descending (front-only descent drifts out of the wrist "
                         "FoV). The camera stays straight-down. deadband+low gain+hold-on-loss. Default OFF.")
    ap.add_argument("--follow-center-margin", type=float, default=0.08, dest="follow_center_margin",
                    help="--follow-wrist-center: wrist center margin (normalized [0..1]). Inside it = aligned → no correction (deadband against chatter).")
    ap.add_argument("--follow-center-gain", type=float, default=0.2, dest="follow_center_gain",
                    help="--follow-wrist-center: proportional gain/tick pulling the follow target toward the wrist-seen cube xy (capped 1.5cm/tick). Default 0.2.")
    ap.add_argument("--follow-cam-fwd", type=float, default=0.03, dest="follow_cam_fwd",
                    help="'Follow' deterministic centering: the wrist camera looks ~22deg ahead of the TCP + lens offset, so the cube is biased toward the bottom → "
                         "shift hover this much (m) along the gripper-forward axis (J1 rotation) to v-center. No feedback, so no oscillation. Default 0.03.")
    ap.add_argument("--follow-cam-lat", type=float, default=-0.02, dest="follow_cam_lat",
                    help="'Follow' centering: shift hover this much (m) along the gripper-lateral axis (perpendicular to forward, J1 rotation) to u-center. Default -0.02.")
    ap.add_argument("--follow-auto-center", action=argparse.BooleanOptionalAction, default=True, dest="follow_auto_center",
                    help="'Follow' (opt-in): closed-loop auto-center — nudge the cam-fwd/lat offset every tick so the wrist cube (u,v)→(0,0). "
                         "Stable Newton step via the HW-measured decoupled Jacobian (dvn/dfwd=-19, dun/dlat=-9.5). deadband+5mm/tick cap. Default ON.")
    ap.add_argument("--follow-auto-gain", type=float, default=0.1, dest="follow_auto_gain",
                    help="--follow-auto-center loop gain (fraction of the Newton step per tick; <1 is stable). Default 0.1.")
    ap.add_argument("--follow-cam-max", type=float, default=0.07, dest="follow_cam_max",
                    help="ANTI-WINDUP: auto-center cam-offset (fwd/lat) cap (m). At far reach IK cannot deliver the TCP shift→offset grows without bound"
                         "→windup oscillation. Prevented by clamping (near converges well inside this, so the margin stays tight). Default 0.07.")
    ap.add_argument("--follow-center-vn", type=float, default=0.44, dest="follow_center_vn",
                    help="--follow-auto-center target wrist v (not dead center at 0). + puts the cube slightly below the wrist center (toward the jaw) = grasp framing. Default 0.44.")
    ap.add_argument("--follow-center-un", type=float, default=-0.03, dest="follow_center_un",
                    help="--follow-auto-center target wrist u. Live-tune with lat+/-. Default -0.03 (2026-07-22 one dashboard lat- click baked in).")
    ap.add_argument("--follow-lookat", action=argparse.BooleanOptionalAction, default=False, dest="follow_lookat",
                    help="'Follow' (opt-in): GIMBAL look-at — HOLD the TCP at hover and tilt only the wrist (J4/J5 null-space IK) to "
                         "aim the camera at the cube. IMAGE-mode aim (measured (u,v)→framing target, cancels extrinsic error). J6 frozen. Default OFF.")

    # ---- Look-at approach ('approach' command) ----
    ap.add_argument("--lookat-hover", type=float, default=0.12, dest="lookat_hover",
                    help="approach: descent start height (m above the cube surface). Default 0.12.")
    ap.add_argument("--lookat-approach-stop", type=float, default=0.05, dest="lookat_approach_stop",
                    help="approach: descent floor = stop hover height (m above the cube); after the look-at hold, hand off to Align/Final/close. Default 0.05. "
                         "offline: look-at is accurate at >=5cm (aim 0deg), aim is extreme at ~3cm (~15deg).")
    ap.add_argument("--lookat-descend-step", type=float, default=0.003, dest="lookat_descend_step",
                    help="approach: max TCP z decrease per tick (m), scaled by the xy-alignment ramp (off=0, aligned=full). Default 0.003.")
    ap.add_argument("--lookat-xy-slow", type=float, default=0.06, dest="lookat_xy_slow",
                    help="approach: xy error (m) at which the descent speed becomes 0 (aligned→this value, linear 1→0). Replaces a hard gate. Default 0.06.")
    ap.add_argument("--lookat-descend-uv", type=float, default=0.35, dest="lookat_descend_uv",
                    help="WRIST-ACQUIRE descent gate (with --lookat-ibvs-xy): HOLD z until the wrist sees the cube and roughly centers it "
                         "(|(un,vn)|<this value). Default 0.35.")
    ap.add_argument("--lookat-deadband", type=float, default=0.004, dest="lookat_deadband",
                    help="approach: xy retarget deadband (m). Live front tracking, excludes only sub-4mm px jitter. Default 0.004.")
    ap.add_argument("--lookat-fuse", type=float, default=0.2, dest="lookat_fuse",
                    help="approach: wrist-(u,v)→world xy refine fuse alpha [0..1] (0=front only). Slower = more stable. Default 0.2.")
    ap.add_argument("--lookat-iters", type=int, default=80, dest="lookat_iters",
                    help="approach: IK iterations/loop (gimbal look-at needs more than straight-down; warm-start every tick). Default 80.")
    ap.add_argument("--lookat-aim-gain", type=float, default=3.0, dest="lookat_aim_gain",
                    help="approach GIMBAL aim gain: cam-aim step scale (larger = faster LOCK on the cube). Feedforward IK, so no oscillation. Default 3.0.")
    ap.add_argument("--lookat-astep", type=float, default=0.07, dest="lookat_astep",
                    help="approach GIMBAL aim step cap (rad/inner-iter). Default 0.07 (aim converges with the gain).")
    ap.add_argument("--lookat-straight", action=argparse.BooleanOptionalAction, default=False, dest="lookat_straight",
                    help="approach mode. Default (gimbal-safe): J4+J5 aim with J6 frozen + J5 rate-cap (no whip). "
                         "--lookat-straight=straight-down mode (point down + xy tracking + J5&J6 hold, ~21deg off-axis).")
    ap.add_argument("--lookat-freeze-j6", action=argparse.BooleanOptionalAction, default=True, dest="lookat_freeze_j6",
                    help="gimbal: freeze J6 (wrist-roll) inside IK → aim with J4+J5 only. Removed because J6 whips (88deg swing). Default ON.")
    ap.add_argument("--lookat-j5-rate", type=float, default=0.05, dest="lookat_j5_rate",
                    help="gimbal: OPT-IN J5 max change per tick (rad). Default 0=OFF (with J6 frozen J5 is already gentle; a cap hurts aim). "
                         "Only ~0.15 if the initial J5 swing is fast.")
    ap.add_argument("--lookat-reapproach-dist", type=float, default=0.12, dest="lookat_reapproach_dist",
                    help="approach safety: if the cube deviates more than this (m) from the TCP xy during descent, rise to REST and re-approach from above. Default 0.12. 0=disabled.")
    ap.add_argument("--lookat-grasp", action=argparse.BooleanOptionalAction, default=False, dest="lookat_grasp",
                    help="approach: extend the look-at descent all the way to grasp (floor=surf+--lookat-grasp-clear). OFF=hover-only handoff. "
                         "When ON, use with --lookat-approach-stop 0.")
    ap.add_argument("--lookat-grasp-clear", type=float, default=0.009, dest="lookat_grasp_clear",
                    help="--lookat-grasp: descent floor (m) above the auto-z surface. 0.009=deep_reach_step default grasp_clear (same z as RUN/Final).")
    ap.add_argument("--lookat-taper", type=float, default=0.03, dest="lookat_taper",
                    help="--lookat-grasp: below this dz (m) above the surface, TAPER gimbal→straight-down (the camera cannot aim in the last cm). Default 0.03.")
    ap.add_argument("--lookat-descend-aim-gain", type=float, default=1.5, dest="lookat_descend_aim_gain",
                    help="--lookat-grasp damping: schedule the aim gain near the cube (hover's --lookat-aim-gain→this value at the surface). Reduces J4+J5 hunting.")
    ap.add_argument("--lookat-descend-astep", type=float, default=0.04, dest="lookat_descend_astep",
                    help="--lookat-grasp damping: schedule the aim step-cap near the cube (--lookat-astep→this value). Default 0.04.")
    ap.add_argument("--lookat-ibvs-xy", action=argparse.BooleanOptionalAction, default=False, dest="lookat_ibvs_xy",
                    help="approach XY: correct the front (top) xy with calib-tolerant wrist IBVS (_wrist_ibvs_dxy: measured wrist px vs the cube reference under the "
                         "FK-TCP, same camera pose→cancels extrinsic error; 55mm→12mm). OFF=_wrist_uv_to_world fuse (calib-biased).")
    ap.add_argument("--lookat-ibvs-gain", type=float, default=0.4, dest="lookat_ibvs_gain",
                    help="--lookat-ibvs-xy: IBVS Newton-step gain. Default 0.4 (slow=stable).")
    ap.add_argument("--lookat-ibvs-step", type=float, default=0.01, dest="lookat_ibvs_step",
                    help="--lookat-ibvs-xy: max wrist-IBVS xy correction per tick (m) = fine-align speed. Smaller = gentler. Default 0.01.")
    ap.add_argument("--lookat-reacquire", type=float, default=0.15, dest="lookat_reacquire",
                    help="COARSE↔FINE handoff: after the wrist gets a FINE lock, ignore front (coarse, ~10cm off). front must move more than this (m) from the wrist lock "
                         "to trigger a coarse re-approach. Must exceed the normal front error (~0.10). Default 0.15.")
    ap.add_argument("--lookat-image-aim", action=argparse.BooleanOptionalAction, default=True, dest="lookat_image_aim",
                    help="approach look-at aim source. ON (default)=IMAGE-based: aim at the measured cube bearing (measured (u,v)→center; relative rotation→"
                         "cancels extrinsic/front error). --no-lookat-image-aim=old WORLD aim. WORLD aim fallback if wvis<0.5.")
    ap.add_argument("--lookat-image-iters", type=int, default=5, dest="lookat_image_iters",
                    help="--lookat-image-aim IK iterations/tick (small; online loop, HW latency-safe).")
    ap.add_argument("--lookat-image-gain", type=float, default=0.2, dest="lookat_image_gain",
                    help="--lookat-image-aim aim gain (small=partial aim + re-measure. HW: 0.4 pushes the TCP, 0.2 is safe).")
    ap.add_argument("--lookat-image-astep", type=float, default=0.02, dest="lookat_image_astep",
                    help="--lookat-image-aim aim step-cap rad/iter (small=anti-overshoot).")
    ap.add_argument("--lookat-image-gate", type=float, default=0.6, dest="lookat_image_gate",
                    help="--lookat-image-aim NEAR-gate: image-aim only when measured |(un,vn)|<this value (roughly framed). Above it, WORLD aim "
                         "brings it into the frame first. Default 0.6.")
    ap.add_argument("--lookat-image-xygate", type=float, default=0.04, dest="lookat_image_xygate",
                    help="--lookat-image-aim: image-aim only when the cube xy distance is within this value (m) (already above). While moving, WORLD aim handles position first. Default 0.04.")
    ap.add_argument("--lookat-autograsp", action=argparse.BooleanOptionalAction, default=False, dest="lookat_autograsp",
                    help="--lookat-grasp: auto-close the gripper (force-servo) at the grasp floor (exact zgap_io gate). OFF=jaws-open hold after descent.")
    ap.add_argument("--lookat-grasp-resid", type=float, default=None, dest="lookat_grasp_resid",
                    help="--lookat-grasp: _danger resid_max override (m) during the look-at descent. None=--safety-reach-resid as-is.")

    # ---- Safety supervisor ----
    ap.add_argument("--safety-auto-rest", action=argparse.BooleanOptionalAction, default=True, dest="safety_auto_rest",
                    help="Safety (approach): kinematically check the commanded pose every tick (IK cannot REACH xy / TCP below the table) → on violation auto-LIFT to REST "
                         "+ HALT. Default ON. Disable with --no-safety-auto-rest.")
    ap.add_argument("--safety-reach-resid", type=float, default=0.15, dest="safety_reach_resid",
                    help="Safety: trip if the commanded pose misses the xy target by more than this (m) (cube out of reach/over-stretch). Normal<1cm. Default 0.15.")

    # ---- Align (auto-center + J6 yaw) ----
    ap.add_argument("--align-vn-target", type=float, default=0.2, dest="align_vn_target",
                    help="'Align' auto-center target: cube TOP-edge normalized y in the wrist frame (-1=top..+1=bottom). Default +0.2 (bottom ~40%% band=grasp framing).")
    ap.add_argument("--align-vn-band", type=float, default=0.10, dest="align_vn_band",
                    help="'Align' auto-center tolerance: if |top_edge_vn - target|<this value, stop the J4 servo and roll J6. Larger = looser.")
    ap.add_argument("--align-j4-gain", type=float, default=-0.6, dest="align_j4_gain",
                    help="'Align' auto-center J4 servo gain (rad per vn-error). Default -0.6 (measured rig direction). Flip the sign if it servos the wrong way.")

    # ---- Auto-grasp ----
    ap.add_argument("--autograsp-yaw", action=argparse.BooleanOptionalAction, default=True, dest="autograsp_yaw",
                    help="OPT-IN: align (roll) J6 to the cube face during the AUTO-GRASP descent. OFF=sacred straight-down (J6 hold). "
                         "ON: after the xy commit, the first confident wrist yaw (latch, else edge-voted seg) sets the J6 target, then hold+freeze. "
                         "'Align flip' flips the roll direction. Requires a seg yaw source (default edge-voting).")
    ap.add_argument("--autograsp-commit-zgap", type=float, default=0.015, dest="autograsp_commit_zgap",
                    help="AUTO-GRASP descent closed-loop→commit switch z_gap_io margin (m). Above this (+cube visible), auto-center "
                         "cancels gravity droop in real time (open-loop straight descent causes +Y drift, docs/32-33); once below this, "
                         "FREEZE the last centered grasp xy, then a pure vertical plunge (safe through the cube-occlusion zone). Default 0.015 (≈last 1.5cm).")
    ap.add_argument("--autograsp-descend-step", type=float, default=0.0015, dest="autograsp_descend_step",
                    help="AUTO-GRASP descent ramp (m/tick), separate from the Follow descent (--follow-descend-step). Slower = the 30Hz auto-center "
                         "cancels droop (+Y) during descent over more frames (docs/33: 0.004 is ~5 frames = 3-4x short). Default 0.0015 (≈2.6x slower).")

    return ap


# ---------------------------------------------------------------------------------------------------
# DeployConfig: immutable dataclass holding the parsed config.
#   Field name = argparse dest. Defaults are not kept here, only in argparse (avoids duplication).
#   Filled via from_namespace(); if the parser and fields disagree, it surfaces immediately as a TypeError at that point.
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeployConfig:
    # --- Runtime / connection ---
    config: str            # training config YAML path (source of mechanism·camera·joint parameters)
    calib: str             # front camera calibration npz path (K·T_cam2world·dist)
    port: int              # dashboard HTTP port
    control_hz: float      # control loop Hz (perception always runs in a separate thread ~30Hz)
    front_dev: str         # front camera device (index or /dev/video* path)
    wrist_dev: str         # wrist camera device (index or /dev/video* path)

    # --- Z depth / FK correction / grasp depth ---
    base_z: float          # URDF base_link world z (FK root)
    grasp_zgap_io: float   # grasp-ready target z_gap_io (auto-z setpoint; 0≈cube center)
    z_bias: float          # descent depth below the cube center (BASE_CTRL space; auto-z initial/fixed value)
    no_auto_z: bool        # True=disable auto-z and use a fixed z_bias
    z_reach_slope: float   # distance-dependent FK-z correction slope (m per m radial; 0=OFF)
    z_reach_ref: float     # radial distance (m) from the J1 axis where the reach correction is 0

    # --- FK-touch verification ---
    fk_touch_step: float   # FK-touch straight-descent step per tick (m)
    fk_touch_dtau: float   # FK-touch contact threshold torque (N*m)
    fk_touch_max: float    # FK-touch safety backstop max descent (m)

    # --- Gripper ---
    grip_force: float          # FORCE_POS torque/current ratio upper bound [0,1] (robot_io hard-clamps to 0.18)
    grip_torque_target: float  # closed-loop grasp target torque (the servo closes up to this value and holds)

    # --- Cameras / transport ---
    mjpg: bool                 # True=MJPG-compressed transfer for both cameras (~30fps)
    wrist_focus: float | None  # fixed wrist focus value (None=fix the starting value)
    front_exposure: float | None  # front manual exposure (None=auto)
    wrist_exposure: float | None  # wrist manual exposure (None=auto)
    front_focus: float | None  # fixed front focus value (None=fix the starting value)

    # --- Perception: OpenVocab + HSV ---
    front_roi: str | None      # front HSV workspace limit 'y0,y1,x0,x1' (fractions; None=full)
    front_red_smin: float      # front cube HSV red saturation lower bound
    front_openvocab: bool      # use front OWLv2 open-vocab ROI
    front_ov_model: str | None # OWLv2 checkpoint (None=default owlv2-base-patch16-ensemble)
    front_ov_thresh: float     # OWLv2 score threshold
    front_ov_period: float     # minimum OWLv2 re-detection interval (s)
    front_ov_size: int | None  # OWLv2 square input px (None=model default)
    front_ov_compile: bool     # torch.compile the OWLv2 model
    wrist_openvocab: bool      # wrist (u,v) also uses the OWLv2 box-gated HSV centroid
    detector_device: str | None  # OWLv2 device (None=cuda if available else cpu)
    top_bias: float            # LEGACY front-cam oblique cube-xy shift (m; default 0)
    x_corr: float              # constant world-X re-centering (m; corrects the front extrinsic-x residual)

    # --- wrist camera calibration ---
    wrist_calib: str | None    # wrist calibration npz override (None=baseline)
    calib_offset: float        # calib_wrist look-at hover lateral offset (m)
    calib_j4: float            # calib_wrist J4 sweep half-amplitude (rad)
    calib_j1: float            # calib_wrist J1 sweep half-amplitude (rad)

    # --- Control / motion smoothing ---
    rule_step: float           # deep_reach IK per-command joint clamp (rad)
    vmax: float                # JointSlew max joint velocity (rad/s)
    amax: float                # JointSlew max joint acceleration (rad/s^2; only when arm_vlim is 0)
    arm_vlim: float            # motor POS_VEL speed limit (rad/s; 0=soft shaper)
    cmd_lpf_tau: float         # first-order LPF time constant for joint setpoints (s; 0=OFF)
    action_chunk: int          # action chunking window size N (1=OFF)
    chunk_decay: float         # action chunking exponential weighting (0<d<=1)
    io_thread: bool            # use a separate Robot-IO thread

    # --- Grasp offset / target correction ---
    jaw_fwd: float             # JAW-FORWARD correction (m; gripper-forward axis, deep-reach path)
    lookat_jaw_fwd: float | None  # approach grasp offset (m; None→robot_io.JAW_FWD)
    cube_x_off: float          # detected cube target world x offset (m; cancels systematic bias)
    cube_y_off: float          # detected cube target world y offset (m)

    # --- Lift / rest ---
    lift_step: float           # total TCP rise height per 'Lift +z' click (m)
    lift_rate: float           # 'Lift +z' rise speed (m/tick)
    rest_lift: float           # TCP +z (m) raising the REST hover above cfg.rest_arm

    # --- Track / Follow ---
    track: bool                # live cube tracking (front updated every frame)
    track_deadband: float      # --track update deadband (m)
    follow_z: float            # 'Follow' hover height (m above the cube)
    follow_deadband: float     # 'Follow' retarget deadband (m)
    follow_iters: int          # 'Follow' ik_oriented iterations/loop
    follow_low: float          # 'Follow' descent floor (m above the cube)
    follow_descend_step: float # 'Follow' descent ramp (m/tick)
    follow_move: float         # 'Follow' RISE threshold (m; real cube movement)
    follow_wrist_center: bool  # 'Follow' wrist centering (opt-in)
    follow_center_margin: float  # follow-wrist-center margin (normalized [0..1])
    follow_center_gain: float  # follow-wrist-center proportional gain/tick
    follow_cam_fwd: float      # 'Follow' deterministic centering fwd offset (m)
    follow_cam_lat: float      # 'Follow' deterministic centering lat offset (m)
    follow_auto_center: bool   # 'Follow' closed-loop auto-center (measured Jacobian)
    follow_auto_gain: float    # follow-auto-center loop gain
    follow_cam_max: float      # follow-auto-center anti-windup cap (m)
    follow_center_vn: float    # follow-auto-center target wrist v (grasp framing)
    follow_center_un: float    # follow-auto-center target wrist u
    follow_lookat: bool        # 'Follow' gimbal look-at(opt-in)

    # --- Look-at approach ---
    lookat_hover: float        # approach descent start height (m above the cube)
    lookat_approach_stop: float  # approach stop hover height (m above the cube)
    lookat_descend_step: float # approach max z decrease per tick (m)
    lookat_xy_slow: float      # xy error (m) at which the approach descent becomes 0
    lookat_descend_uv: float   # WRIST-ACQUIRE descent gate |(un,vn)|
    lookat_deadband: float     # approach xy retarget deadband (m)
    lookat_fuse: float         # approach wrist-(u,v)→world fuse alpha
    lookat_iters: int          # approach IK iterations/loop
    lookat_aim_gain: float     # approach GIMBAL aim gain
    lookat_astep: float        # approach GIMBAL aim step-cap(rad)
    lookat_straight: bool      # approach straight-down mode (default gimbal-safe)
    lookat_freeze_j6: bool     # freeze J6 inside the gimbal IK
    lookat_j5_rate: float      # gimbal J5 max change per tick (rad; 0=OFF)
    lookat_reapproach_dist: float  # approach re-approach safety distance (m; 0=disabled)
    lookat_grasp: bool         # extend the approach look-at descent all the way to grasp
    lookat_grasp_clear: float  # lookat-grasp descent floor (m above the surface)
    lookat_taper: float        # lookat-grasp gimbal→straight-down taper dz(m)
    lookat_descend_aim_gain: float  # lookat-grasp near-cube aim gain schedule
    lookat_descend_astep: float  # lookat-grasp near-cube aim step-cap schedule
    lookat_ibvs_xy: bool       # correct approach XY with calib-tolerant wrist IBVS
    lookat_ibvs_gain: float    # lookat-ibvs-xy Newton-step gain
    lookat_ibvs_step: float    # lookat-ibvs-xy max xy correction per tick (m)
    lookat_reacquire: float    # COARSE↔FINE handoff front re-approach distance (m)
    lookat_image_aim: bool     # approach look-at IMAGE-based aim (default ON)
    lookat_image_iters: int    # lookat-image-aim IK iterations/tick
    lookat_image_gain: float   # lookat-image-aim aim gain
    lookat_image_astep: float  # lookat-image-aim aim step-cap(rad)
    lookat_image_gate: float   # lookat-image-aim NEAR-gate |(un,vn)|
    lookat_image_xygate: float # lookat-image-aim xy-gate(m)
    lookat_autograsp: bool     # auto-close at the lookat-grasp grasp floor
    lookat_grasp_resid: float | None  # lookat-grasp _danger resid override(m; None=safety-reach-resid)

    # --- Safety supervisor ---
    safety_auto_rest: bool     # on violation, REST auto-LIFT+HALT
    safety_reach_resid: float  # xy miss trip threshold (m)

    # --- Align ---
    align_vn_target: float     # Align auto-center target top-edge vn
    align_vn_band: float       # Align auto-center tolerance
    align_j4_gain: float       # Align auto-center J4 servo gain

    # --- Auto-grasp ---
    autograsp_yaw: bool        # align J6 to the cube face during AUTO-GRASP descent (opt-in)
    autograsp_commit_zgap: float  # descent closed-loop→commit (freeze) switch z_gap_io margin (m)
    autograsp_descend_step: float # AUTO-GRASP descent ramp (m/tick, separate from the Follow descent; slower = more servo headroom)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "DeployConfig":
        """argparse.Namespace → DeployConfig. (input: parsed ns → output: immutable config)
        Dataclass field names must correspond exactly to ns attributes. A mismatch raises TypeError here."""
        field_names = [f.name for f in fields(cls)]
        kwargs = {}
        for name in field_names:
            kwargs[name] = getattr(ns, name)
        return cls(**kwargs)


def load(argv=None) -> DeployConfig:
    """Parses CLI arguments and returns a DeployConfig. (input: argv list or None=sys.argv → output: DeployConfig)"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return DeployConfig.from_namespace(args)

"""Deep-Reach debugger (REACH -> DESCEND, gripper OPEN throughout) — brings the TCP down to the
cube surface via top+wrist closed-loop. Extends debug_reach: instead of stopping at hover, descends via base.deep_reach_step (straight-down reach +
wrist (u,v)->world fusion). The gripper is always OPEN (grasp is the next stage). Each Step = 1 deep_reach servo step.

XY correction = WRIST cam (world-unprojection fusion of base.deep_reach_step with ibvs=False; IBVS was DROPPED after evaluation).
Z = 3-source fusion (median): (1) front-gripper geometry, (2) wrist depth head [display only], (3) base-z-corrected FK.

What the dashboard (:8899) shows:
  cube_top(x,y) | TCP(base_ctrl world) vs TCP(robot_io FK) | wrist(u,v,vis) | xy_err | z_gap(tcp-plane) | at_cube
  + robust Z readout: z_gap_front | z_gap_fk | z_gap_med(fused) | z_wrist_depth(display only) | at_cube_robust
  + LIVE CAMERA panels: FRONT(cube/gripper/FK-box overlay), WRIST(CNN (u,v) + center target)  ← verify detections by eye
  + VERIFY panel: in order cam-id -> cube -> wrist CNN -> pedestal z_bias≈0 -> RUN latch -> front-Z -> grasp.

** MOVES THE ARM on 'Step'/'Jog'/'RUN' only. E-STOP ready. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/debug_deepreach.py --port 8899 \
        --config configs/train_v6a_yaw.yaml --base-z 0.02
Dashboard: http://localhost:8899  (remote: ssh -L 8899:localhost:8899)
"""
from __future__ import annotations
import os, re, sys, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.base_controller import RuleBaseController
from rebot_sim2real import perception as P
from robot_io import DamiaoRobotIO
from debug_dashboard import DebugDashboard
import front_gripper_z                      # FK-INDEPENDENT front-cam gripper Z (the robust-hybrid Z source)

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
# Camera identity comes from the persistent udev symlinks (/dev/video_front, /dev/video_wrist) so a kernel
# re-enumeration can't silently SWAP front<->wrist (raw indices 0/2 are not stable). Override with the CLI.
FRONT_DEV, WRIST_DEV = "/dev/video_front", "/dev/video_wrist"


def _resolve_cam_index(dev):
    """Accept an int, a numeric string, or a /dev/video* path (incl. a udev symlink) -> V4L2 index int.
    Resolving the symlink to its videoN index lets us open with the V4L2 backend by integer (most robust;
    cv2.VideoCapture('<path>') can pick the wrong backend for device nodes)."""
    if isinstance(dev, int):
        return dev
    s = str(dev)
    if s.isdigit():
        return int(s)
    real = os.path.realpath(s)                       # /dev/video_wrist -> /dev/video2
    m = re.search(r"(\d+)\s*$", real)
    if not m:
        raise SystemExit(f"[deep-reach] cannot resolve camera device '{dev}' -> '{real}' (no trailing index)")
    return int(m.group(1))


def _read_rgb(idx, focus=None, cache={}):
    import cv2
    if idx not in cache:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)    # explicit V4L2 backend (integer index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3); cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        # DISABLE autofocus (C922 ships AF ON). AF refocusing as the cube distance changes during the descent
        # changes the effective focal length (focus breathing) -> the FIXED wrist_K (u,v)->world unprojection
        # xy DRIFTS. Locking AF makes the intrinsics STABLE (and AF hunting won't blur the detection mid-motion).
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if focus is not None:                        # lock to a FIXED focus (cv2 ~0=far..255=near); match wrist K calib
            cap.set(cv2.CAP_PROP_FOCUS, float(focus))
        for _ in range(10):
            cap.read()
        cache[idx] = cap
    ok, fr = cache[idx].read()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def wrist_uvn_redmask(rgb):
    """FALLBACK wrist (u,v): red centroid -> normalized (un, vn, vis) ∈ [-1,1]. Brittle HSV; used only if the
    trained wrist CNN ckpt is missing. wrist input to deep_reach_step (same convention as the CNN path)."""
    if rgb is None:
        return 0.0, 0.0, 0.0
    H, W = rgb.shape[:2]
    c = P._largest_blob_centroid(P.red_mask(rgb.astype(np.float32)))
    if c is None or c[2] < 8:
        return 0.0, 0.0, 0.0
    return (c[0] - W / 2.0) / (W / 2.0), (c[1] - H / 2.0) / (H / 2.0), 1.0


# ---- overlay helpers: draw the live detections on the camera frames so they can be EYE-verified ----
def _encode(img_bgr):
    """resize-for-transport + JPEG. img_bgr is annotated BGR (cv2 native). -> bytes (or None)."""
    import cv2
    H, W = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (512, max(1, int(512 * H / W))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def annotate_wrist(rgb, un, vn, vis, depth, is_cnn):
    """WRIST eye-in-hand frame: image-center crosshair (= the alignment target for unproj/IBVS) + the
    detected cube (u,v). Green=CNN, yellow=red-mask fallback. So you SEE whether the detector locks the cube."""
    import cv2
    if rgb is None:
        return None
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    cv2.drawMarker(img, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 26, 1)   # target = centered
    tag = "CNN" if is_cnn else "redmask"
    if vis > 0.5:
        u = int((un + 1.0) / 2.0 * W); v = int((vn + 1.0) / 2.0 * H)
        col = (0, 255, 0) if is_cnn else (0, 255, 255)
        cv2.circle(img, (u, v), 13, col, 2); cv2.circle(img, (u, v), 2, col, -1)
        txt = f"{tag} vis=1" + (f"  depth={depth:.2f}m" if (is_cnn and depth > 0) else "")
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    else:
        cv2.putText(img, f"{tag}  NO DETECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return _encode(img)


def correct_top_xy(est, T, bias):
    """Correct the front-cam ray-plane cube xy for the OBLIQUE-CAMERA bias. estimate_red_cube_world assumes the
    red-blob centroid sits at the cube CENTER (z=plane_z), but at the camera's angle the visible (camera-facing)
    faces pull the centroid toward the cam, AND a height mismatch projects ~1:1 into xy along the line of sight
    -> the estimate lands FORWARD (toward the cam). Fix: shift the estimate AWAY from the cam along the HORIZONTAL
    cam->cube direction (taken from the calib, so it's camera-angle aware) by `bias` (m). bias also absorbs the
    fixed jaw-forward offset, so the JAWS (not just the TCP) land on the cube. Returns a NEW xyz (z unchanged)."""
    if est is None or bias == 0.0:
        return est
    cam_xy = np.asarray(T, float)[:2, 3]                 # front-cam world xy (T_cam2world translation)
    d = np.asarray(est[:2], float) - cam_xy              # horizontal cam -> cube (points AWAY from the cam)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return est
    out = np.array(est, float)
    out[:2] = out[:2] + bias * (d / n)                   # push AWAY from the cam -> cancels the toward-cam bias
    return out


def annotate_front(rgb, cube_world, uv_grip, fk_pred_uv, K, T, plane_z, box=140):
    """FRONT fixed frame: cube xy marker (orange, projected captured world) + gripper frame-diff detection
    (green) + the FK-prior search box (grey). So you SEE whether front cube-detect + gripper-Z inputs are sane."""
    import cv2
    if rgb is None:
        return None
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if cube_world is not None:
        uvp = front_gripper_z.project_world_to_front([cube_world[0], cube_world[1], plane_z], K, T)
        if uvp is not None:
            cv2.drawMarker(img, (int(uvp[0]), int(uvp[1])), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
            cv2.putText(img, "cube xy", (int(uvp[0]) + 10, int(uvp[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if fk_pred_uv is not None:
        gx, gy = int(fk_pred_uv[0]), int(fk_pred_uv[1])
        cv2.rectangle(img, (gx - box, gy - box), (gx + box, gy + box), (120, 120, 120), 1)   # diff search window
        cv2.drawMarker(img, (gx, gy), (180, 180, 180), cv2.MARKER_CROSS, 16, 1)              # FK gripper prior
    if uv_grip is not None:
        cv2.circle(img, (int(uv_grip[0]), int(uv_grip[1])), 9, (0, 255, 0), -1)
        cv2.putText(img, "gripper", (int(uv_grip[0]) + 10, int(uv_grip[1]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return _encode(img)


# ---- VERIFY SEQUENCE (deep-reach hardware bring-up, gripper OPEN until the last GRASP step) ----
# Each step's gate is auto-evaluated against the live snapshot dict; the dashboard shows PASS/…/manual and
# the operator advances with 'Verify ▶ Next'. The ORDER encodes the dependency chain (handoff docs/11 §5):
#   cam identity -> cube detect -> wrist CNN -> DESCEND to grasp-z -> RUN descend-latch -> front-gripper Z -> grasp.
GRASP_CLEAR = 0.005   # mirrors deep_reach_step's grasp_clear default; descend stops grasp_clear above cube_top_z
Z_BIAS_MAX = 0.10     # safety clamp on the (auto or manual) descend depth so a bad reading can't over-plunge
FK_BIAS_EMA = 0.35    # smoothing for the measured base-FK-vs-robot_io z discrepancy (auto-z feedforward; higher
#                     = less lag so the descend doesn't stop "too high" while the EMA is still catching up)


def _g(s, *path, default=None):
    cur = s
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_verify_seq():
    """VERIFY ④ gates on the ACCURATE z_gap_io ≈ goal (grasp_zgap_io, read live from the snapshot). HARDWARE
    FINDING (2026-06-25): the gripper is at the cube (cube-center plane_z=0.019) when z_gap_io ≈ 0 (operator
    measured 0.006), while the base_ctrl 'z gap (tcp-surface)' reads ≈ -0.034 there (base-FK underreads ~40-50mm,
    a -0.015 = below-table reading = proof of the FK error). AUTO-Z closes the descend on z_gap_io (measures the
    base-FK-vs-robot_io discrepancy live and feeds it forward) so the gripper lands at the cube at ANY position
    with NO manual z-bias tuning. The base_ctrl -0.034 is the FK shadow — ignored."""
    return [
        dict(name="① CAM IDENTITY",
             instr="Check the two video panels: WRIST=wrist (moves along when the gripper is jogged), FRONT=fixed overhead. If swapped, "
                    "restart with --front-dev/--wrist-dev. Once confirmed, Next.",
             gate=lambda s: None),                                  # manual (eyeball the two panels)
        dict(name="② CUBE DETECT (front HSV)",
             instr="Place the red cube, keep the arm far away. 'Capture cube'. If cube_top(x,y) is inside the workspace + the orange marker in the FRONT panel "
                    "lands on the cube, OK -> Next.",
             gate=lambda s: bool(s.get("captured"))),
        dict(name="③ WRIST CNN (u,v)",
             instr="Make the wrist look at the cube (a few Steps if needed). Green dot in the WRIST panel on the cube + vis=1 (CNN working). "
                    "If the dot is off, it is a CNN label/lighting problem. OK -> Next.",
             gate=lambda s: (_g(s, "wrist_uv", default=[0, 0, 0]) or [0, 0, 0])[2] > 0.5),
        dict(name="④ DESCEND to cube (z_gap_io → goal)",
             instr="Descend with Step/RUN. **AUTO-Z** automatically corrects the base-FK under-reading via robot_io and brings z_gap_io (the accurate height) "
                    "down to the goal (default 0 = cube center 0.019) — position-independent, no manual z-bias adjustment needed. Ignore the −0.034 in base_ctrl 'z gap "
                    "(tcp-surface)'; it is an FK shadow. Fine-tune the goal with ± (down = deeper grasp).",
             gate=lambda s: bool(s.get("at_cube")) and
                            abs(_g(s, "gates", "z_gap_io", default=9.0) - _g(s, "gates", "grasp_zgap_io", default=0.0)) < 0.015),
        dict(name="⑤ REACH/DESCEND (RUN, latch)",
             instr="Reset arm (far away)->Capture->RUN. Descends continuously without 5cm oscillation (limit cycle) to 'reached' (z_gap_io≈goal "
                    "+ xy aligned to the jaw-fwd-corrected grasp target). If it oscillates, it is a descend-latch problem.",
             gate=lambda s: bool(s.get("reached"))),
        dict(name="⑥ FRONT-GRIPPER Z",
             instr="Check that during descent z_gap_front gets a numeric value (gripper frame-diff detection = green dot in the FRONT panel), and that z_gap_med "
                    "matches z_gap_io within ~1-2cm. If not detected, tune diff_thresh/box.",
             gate=lambda s: _g(s, "gates", "z_gap_front") is not None),
        dict(name="⑦ GRASP (manual: open->close->lift)",
             instr="After reaching grasp-z, 'Grip close'. grip_tau rises + grasped=true. If the cube comes up with the "
                    "gripper when repeating 'Lift +z', success.",
             gate=lambda s: bool(_g(s, "gates", "grasped"))),
        dict(name="⑧ DONE",
             instr="deep-reach + grasp verification complete. ABORT for e-stop. Next: port the v6b residual policy (autonomous full pick).",
             gate=lambda s: None),
    ]


VERIFY_SEQ = build_verify_seq()   # gate reads the live goal (grasp_zgap_io) from the snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_v6a_yaw.yaml")
    ap.add_argument("--calib", default="deploy/calib.npz")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--front-dev", default=FRONT_DEV, dest="front_dev",
                    help="front/fixed cam: int index, or a /dev/video* path (default udev symlink /dev/video_front).")
    ap.add_argument("--wrist-dev", default=WRIST_DEV, dest="wrist_dev",
                    help="wrist/eye-in-hand cam: int index, or a /dev/video* path (default /dev/video_wrist). "
                         "Swap these if the VERIFY ① cam-identity check shows them reversed.")
    ap.add_argument("--base-z", type=float, default=0.010, dest="base_z",
                    help="world-frame-N z of the URDF base_link (FK root) = base_pos.z. NOT the J1 axis: URDF "
                         "joint1 origin is +0.08465 above base_link, so base_pos.z = J1_axis_z - 0.08465. MEASURED "
                         "J1 axis (motor-box top) = +0.095 over the table -> base_pos.z = +0.010 (prev +0.02 put J1 "
                         "at +0.105, 1cm high). NOTE: this only sets the BASE term; the larger ~9cm FK underread is "
                         "the URDF kinematic chain (band-aided by robot_io.FK_Z_CORR + --grasp-zgap-io), unaffected.")
    ap.add_argument("--grasp-zgap-io", type=float, default=0.001, dest="grasp_zgap_io",
                    help="GOAL robot_io z_gap_io (the ACCURATE FK_Z_CORR-corrected height) at grasp-ready = the "
                         "VERIFY ④ target AND the auto-z setpoint. HW: gripper at the cube-center is z_gap_io ≈ 0; "
                         "default +0.001 stops the TCP ~1mm HIGHER (avoid bottoming on the cube/table). Lower (e.g. "
                         "-0.01) for a deeper grasp, raise for higher. Gate = |z_gap_io - this| < 0.015.")
    ap.add_argument("--z-bias", type=float, default=0.05, dest="z_bias",
                    help="descend depth past cube-center in BASE_CTRL space, used as the AUTO-Z initial value and as "
                         "the FIXED depth when --no-auto-z. HW ballpark ~0.05 (compensates the base-FK underread).")
    ap.add_argument("--no-auto-z", action="store_true", dest="no_auto_z",
                    help="DISABLE auto-z (the closed loop that measures the base-FK-vs-robot_io z discrepancy live "
                         "and feeds it forward so z_gap_io lands on the goal at ANY position with NO manual tuning). "
                         "With this flag the descend uses the FIXED --z-bias (live ± buttons) instead.")
    ap.add_argument("--grip-force", type=float, default=0.3, dest="grip_force",
                    help="gripper FORCE_POS torque/current ratio [0,1]. Driver default 0.1 (10%%) is too weak to "
                         "clamp a cube (grip_tau stays ~air 0.07); 0.3 squeezes firmly. Raise toward ~0.5 if a cube "
                         "still slips (watch grip_tau -> ~1.x); too high can crush/strain — NOT a calibration fix.")
    ap.add_argument("--wrist-detector", default="ckpt/cube_detector_wrist_sd3.pt", dest="wrist_detector",
                    help="trained wrist CNN ckpt for the (u,v[,depth]) detection (clutter/lighting robust drop-in "
                         "for the brittle red-mask). Falls back to red-mask if this file is missing.")
    ap.add_argument("--rule-step", type=float, default=0.12, dest="rule_step",
                    help="per-command joint clamp (rad) for the deep_reach IK = the MOTOR VELOCITY knob. Joint "
                         "vel ≈ rule_step * rate rad/s (0.12*10Hz=1.2). RAISE for a faster approach (e.g. 0.20). "
                         "Used by both single Step and the continuous RUN (one-shot) mode.")
    ap.add_argument("--smooth", type=float, default=0.8, dest="smooth",
                    help="motion SMOOTHING: EMA factor applied to the SENT arm command (q_send += smooth*(target-"
                         "q_send)) so the servo gets an eased reference instead of stepped targets. 1.0 = OFF "
                         "(instant), lower = smoother but slower (effective step ≈ smooth*rule_step — raise "
                         "--rule-step to compensate). Grip is never smoothed. Freeze/scripted moves snap (no lag).")
    ap.add_argument("--wrist-focus", type=float, default=None, dest="wrist_focus",
                    help="LOCK the wrist C922 focus (cv2 CAP_PROP_FOCUS, ~0=far..255=near). AF is ALWAYS disabled "
                         "(focus breathing changes the focal length -> the fixed-K wrist (u,v)->world xy drifts as "
                         "the gripper descends). None = AF off + focus frozen at the startup value. For metric xy, "
                         "set a value matched to the descent range and (re)calibrate wrist_K AT that focus. NOTE: "
                         "C922 min-focus is ~10cm so the cube blurs when very close (<~8cm) regardless.")
    ap.add_argument("--front-focus", type=float, default=None, dest="front_focus",
                    help="LOCK the front cam focus (AF always disabled). Front sits at a fixed distance so AF would "
                         "settle, but locking keeps the calibrated K valid. None = AF off + frozen at startup.")
    ap.add_argument("--top-bias", type=float, default=0.035, dest="top_bias",
                    help="front-cam cube-xy OBLIQUE-ANGLE correction (m): the ray-plane estimate lands FORWARD "
                         "(toward the cam) because the visible camera-facing faces pull the red centroid + the "
                         "height mismatch projects ~1:1 into xy. This shifts the captured cube xy AWAY from the cam "
                         "along the calib cam->cube direction (cam-angle aware), also absorbing the jaw-forward "
                         "offset. HW: ~0.035 lands the jaws on the cube. Set 0 to see the raw estimate.")
    ap.add_argument("--jaw-fwd", type=float, default=0.06, dest="jaw_fwd",
                    help="JAW-FORWARD correction (m): the real gripper jaws close ~this far FORWARD (+y, camera-"
                         "side) of the base-FK TCP, so the deep-reach XY target is shifted BACK (-y) by this so the "
                         "JAWS land on the cube (deploy analog of robot_io.JAW_FWD=0.06). If the gripper still lands "
                         "in front of the cube, raise it; behind, lower it (negative flips the side).")
    ap.add_argument("--approach-z", type=float, default=0.3, dest="approach_z",
                    help="TCP z (table-rel) for the '↑ Approach' button: a rule-based (DLS-IK, cam straight down) "
                         "move to a high hover over the cube (or workspace center) = a clean, repeatable approach "
                         "START pose. From there use Step/RUN to deep-reach down.")
    args = ap.parse_args()

    # ---- resolve cameras from the udev symlinks (stable identity) -> V4L2 indices ----
    # FRONT is required (cube xy + gripper Z). WRIST is OPTIONAL: if its device is missing, run FRONT-ONLY
    # (wrist_uv=0, no wrist fusion) instead of crashing at startup — the front-cam top-bias reach still works.
    front_idx = _resolve_cam_index(args.front_dev)
    wrist_idx = (_resolve_cam_index(args.wrist_dev)
                 if (str(args.wrist_dev).isdigit() or os.path.exists(args.wrist_dev)) else None)
    if wrist_idx is None:
        print(f"[deep-reach] WARNING: wrist cam '{args.wrist_dev}' NOT found -> FRONT-ONLY (wrist_uv=0, no wrist "
              "fusion; front-cam reach still works). Reconnect the wrist cam + restart to enable it.", flush=True)
    print(f"[deep-reach] cameras: front='{args.front_dev}'->/dev/video{front_idx}  "
          f"wrist={('/dev/video' + str(wrist_idx)) if wrist_idx is not None else 'MISSING (front-only)'}", flush=True)

    tc = load_train_config(args.config); cfg = tc.env
    cfg.num_envs = 1
    dev = torch.device("cpu")
    base = RuleBaseController(cfg, dev)            # same reach/deep_reach as sim (FK/DLS-Jacobian)
    base.base_pos[2] = float(args.base_z)         # PEDESTAL fix: real J1 base z (table-rel) -> FK matches reality
    base.arm_step = float(args.rule_step)         # per-command joint clamp = velocity knob (Step + RUN one-shot)
    print(f"[deep-reach] base_pos z={args.base_z:+.3f} (config {cfg.base_pos[2]:+.3f}; pedestal) · rule_step={args.rule_step:.3f}", flush=True)
    cal = np.load(args.calib); K = cal["K"].astype(float); T_cam2world = cal["T_cam2world"].astype(float)
    plane_z = cfg.table_z + cfg.cube_size / 2.0
    # sim2real FK z-gap: base-FK reads the gripper ~z_bias LOW vs the real robot -> lower the deep_reach
    # descend target (base.cube_top_z) by z_bias so the REAL gripper reaches the surface. Tune live.
    # HARDWARE: the gripper is at the cube when the ACCURATE z_gap_io ≈ grasp_zgap_io (≈0). base_ctrl z_gap reads
    # ~-0.034 there (FK underread, ignored). AUTO-Z (default) measures the base-FK-vs-robot_io z discrepancy live
    # and feeds it forward so z_gap_io lands on the goal at ANY cube position with NO manual z-bias tuning.
    grasp_zgap_io = float(args.grasp_zgap_io)
    auto_z = not args.no_auto_z
    z_bias = float(args.z_bias)                    # auto-z initial value / fixed depth when --no-auto-z
    base.cube_top_z = plane_z - z_bias
    verify_seq = build_verify_seq()
    fk_bias_ema = None                             # EMA of (robot_io z - base_ctrl z) = the base-FK underread
    print(f"[deep-reach] auto_z={auto_z} (closed loop on z_gap_io -> goal {grasp_zgap_io:+.3f}); "
          f"z_bias init={z_bias:.3f}. VERIFY ④ PASSes when z_gap_io ≈ {grasp_zgap_io:+.3f} "
          f"(= TCP at cube-center 0.019; base_ctrl 'z gap (tcp-surface)' ~-0.034 is the FK shadow).", flush=True)

    # WRIST (u,v[,depth]) detector: trained CNN (robust) with red-mask FALLBACK if the ckpt is missing.
    # Returns (un, vn, vis, depth) — depth is the lens->cube distance (meters) from the detector's depth
    # head; vis<thresh -> all zeros (sentinel). deep_reach_step gets (un,vn,vis) exactly as before.
    wrist_det = None
    if os.path.exists(args.wrist_detector):
        from cube_detector_io import CubeDetectorIO
        wrist_det = CubeDetectorIO(ckpt=args.wrist_detector, device="cpu")
        print(f"[deep-reach] wrist detector = CNN {args.wrist_detector}", flush=True)
    else:
        print(f"[deep-reach] WARNING wrist detector ckpt '{args.wrist_detector}' missing -> red-mask FALLBACK "
              "(brittle; no depth head)", flush=True)

    def wrist_uvn_depth(rgb):
        """(un, vn, vis, depth_m) from the SAME frame used for the overlay. CNN path when available (depth from
        the detector's depth head); else red-mask fallback (depth=0). Feeds (un,vn,vis) to deep_reach_step."""
        if rgb is None:
            return 0.0, 0.0, 0.0, 0.0
        if wrist_det is None:
            un, vn, vis = wrist_uvn_redmask(rgb)
            return un, vn, vis, 0.0
        return wrist_det.wrist_uvn_depth(rgb)

    io = DamiaoRobotIO(cfg, URDF, grip_force_ratio=args.grip_force); io.connect()
    print(f"[deep-reach] gripper FORCE_POS ratio = {args.grip_force:.2f} (driver default 0.1 is too weak to clamp).", flush=True)
    io.base_pos[2] = float(args.base_z)              # CRITICAL: robot_io defaults to cfg.base_pos.z (−0.03) while
    #   base_controller uses --base-z -> a ~5cm inter-FK base MISMATCH that pollutes auto-z fk_bias (=tcp_io−tcp_w)
    #   AND z_gap_io (the goal metric). Force BOTH FK chains to the SAME measured base_link z so fk_bias/z_gap_io
    #   reflect ONLY the real kinematic residual, not a base-frame disagreement.
    print(f"[deep-reach] base_pos.z={args.base_z:+.3f} applied to BOTH base_controller + robot_io "
          f"(config default was {cfg.base_pos[2]:+.3f}); J1 axis ≈ base_z+0.0847 = {args.base_z + 0.08465:+.3f}.", flush=True)
    rest_np = np.array(cfg.rest_arm, dtype=float)
    io.initial_liftup(rest_np, cfg.grip_open_qpos)   # scripted -> REST hover, gripper open
    dash = DebugDashboard(args.port); dash.start()

    # ---- motion SMOOTHING: route every servo command through send_arm, which EMAs the SENT joints toward the
    #      target so the real arm gets an eased reference (less jerk). snap=True freezes/hands-off without lag. ----
    smooth = float(args.smooth)
    q_send = rest_np.copy()                          # smoothed arm command state (init = REST after liftup)
    jaw_fwd = float(args.jaw_fwd)                     # jaw-forward XY target correction (real jaws close +y of TCP)
    top_bias = float(args.top_bias)                   # front-cam oblique-angle cube-xy correction (cam-away shift)
    cube_raw = None                                   # last RAW front-cam estimate (pre-correction), for display
    cube_dxy = [0.0, 0.0]                              # LIVE task-space xy nudge on the grasp target (per-position trim)
    approach_z = float(args.approach_z)
    ws_cx = 0.5 * (cfg.workspace_x[0] + cfg.workspace_x[1])   # workspace center (approach target when no cube)
    ws_cy = 0.5 * (cfg.workspace_y[0] + cfg.workspace_y[1])

    def send_arm(q_target, grip, snap=False):
        """SMOOTHED arm command: EMA the sent joints toward q_target (smooth>=1 -> OFF); snap=True sets them
        directly (freeze / scripted handoff -> no lag). Grip is NOT smoothed (needs a crisp close)."""
        qt = np.asarray(q_target, dtype=float).reshape(-1)
        if snap or smooth >= 0.999:
            q_send[:] = qt
        else:
            q_send[:] = q_send + smooth * (qt - q_send)
        io.command(q_send.copy(), float(grip))

    cube_world = None
    dt = 1.0 / args.rate
    step = 0
    grip_cmd = cfg.grip_open_qpos      # persistent grip state (open by default; close/open buttons set it)
    gripping = False                   # when closed: re-command the grip every idle loop -> SUSTAINED force
    grip_force_qpos = max(0.0, cfg.grip_close_qpos - 0.005)   # overdrive PAST the close floor -> harder grip
    # IBVS is DEPRECATED: the finalized robust-hybrid uses the WRIST world-unprojection (deep_reach_step
    # ibvs=False) for XY. The toggle/code is kept but stays OFF/unused — do NOT enable it.
    ibvs_on = False
    auto_running = False               # RUN (one-shot): loop deep_reach_step EVERY iteration (no per-Step click)
    #   until at_cube; the motor velocity = rule_step * rate. E-STOP(abort) + Stop/Hold halt it.
    approach_hi = False                # '↑ Approach' (one-shot): rule IK to TCP z=approach_z over the cube, then hold
    prev_front = None                  # PREVIOUS front frame for the gripper frame-differencing (front Z source)
    verify_idx = 0                     # VERIFY sequence cursor (advanced by 'Verify ▶ Next')
    print(f"[deep-reach] dashboard up. z_bias={z_bias:.3f} (descend target = surface - z_bias; tune live with "
          "the z-bias +/- buttons). Follow the VERIFY panel: ① cam identity -> ② Capture cube -> ③ wrist CNN "
          "-> ④ Step/RUN to at_cube (gripper OPEN; grasp is ⑦).", flush=True)
    try:
        while True:
            t0 = time.time()
            pr = io.read_proprio()
            arm_q = torch.tensor(pr.arm_qpos, dtype=torch.float32).unsqueeze(0)   # [1,6]
            tcp_b = base.fk(arm_q)[0]                                             # BASE frame (for deep_reach internal use)
            tcp_w_t = base.fk_world(arm_q)[0]                                     # WORLD frame N (base-z corrected)
            tcp_w = tcp_w_t.tolist()
            tcp_io = list(map(float, io.fk_tcp(pr.arm_qpos)))

            # ---- AUTO-Z: cancel the (position-varying) base-FK underread using robot_io (accurate) as truth.
            #      fk_bias = robot_io_z - base_ctrl_z (≈ +0.04-0.05); feed it forward into the descend depth so
            #      z_gap_io lands on grasp_zgap_io at ANY cube position -> NO manual z-bias tuning. EMA + clamp.
            fk_bias = float(tcp_io[2] - tcp_w[2])
            fk_bias_ema = fk_bias if fk_bias_ema is None else (1 - FK_BIAS_EMA) * fk_bias_ema + FK_BIAS_EMA * fk_bias
            zgap_io_val = float(tcp_io[2] - plane_z)     # ACCURATE height above the cube-center (robot_io FK)
            if auto_z:
                z_bias = min(Z_BIAS_MAX, max(0.0, GRASP_CLEAR + fk_bias_ema - grasp_zgap_io))
                base.cube_top_z = plane_z - z_bias           # re-target the descend each loop (auto-compensated)

            # ---- read the WRIST frame ONCE, detect on it (CNN or red-mask), then overlay the SAME frame ----
            wrist_rgb = (_read_rgb(wrist_idx, focus=args.wrist_focus)   # AF locked -> stable wrist_K for unprojection
                         if wrist_idx is not None else None)            # front-only when the wrist cam is absent
            wun, wvn, wvis, wdepth = wrist_uvn_depth(wrist_rgb)
            # DIAGNOSTIC: wrist eye-in-hand world-xy (≈ angle-INVARIANT when descending straight-down over the
            # cube) vs the front-cam ray-plane xy. The front ray-plane is ~1:1 sensitive to the red-centroid's
            # effective HEIGHT x cam angle, so it biases (forward) when the blob is dominated by the camera-facing
            # face. Compare wrist_xy vs cube_top to SEE the front-cam bias + whether the wrist correction engages.
            wrist_xy = (base._wrist_uv_to_world(arm_q, torch.tensor([wun]), torch.tensor([wvn]), plane_z)[0].tolist()
                        if wvis > 0.5 else None)
            # ---- read the CURRENT front frame EVERY loop (was capture-only) -> front-gripper Z source ----
            cur_front = _read_rgb(front_idx, focus=args.front_focus)   # AF locked -> stable front K for ray-plane

            # ---- 3-SOURCE Z fusion (median): (1) front-gripper geometry, (2) wrist depth [display only,
            #      NOT fused yet], (3) base-z-corrected FK. Each is a TCP "gap" above the cube surface. ----
            tcp_world_np = np.asarray(tcp_w, float)              # base.fk_world(arm_q)[0] (base-z corrected FK TCP)
            gap_fk = float(tcp_world_np[2] - plane_z)            # (3) FK gap
            gap_front = None                                     # (1) front-gripper geometry gap (None if no detect)
            fk_pred_uv = None                                    # FK gripper prior pixel (for overlay + diff window)
            uv_grip = None                                       # detected gripper pixel (frame-diff)
            z_wrist_depth = round(float(wdepth), 3) if wvis > 0.5 else None  # (2) wrist depth (lens->cube, display)
            if cube_world is not None and prev_front is not None and cur_front is not None:
                fk_pred_uv = front_gripper_z.project_world_to_front(tcp_world_np, K, T_cam2world)
                uv_grip = front_gripper_z.detect_gripper_uv(prev_front, cur_front, fk_pred_uv=fk_pred_uv)
                if uv_grip is not None:
                    res = front_gripper_z.gripper_z_from_ray(uv_grip, cube_world[:2], K, T_cam2world, plane_z)
                    if res is not None:
                        _gz, gap_front, _t = res                 # (gripper_z, gap, t)
            # fuse: median of the AVAILABLE NUMERIC gaps among {gap_front, gap_fk}. wrist depth is shown but
            # NOT fused into the gap yet (lens->cube distance != TCP gap; needs a TCP<-lens transform). TODO.
            avail = [g for g in (gap_front, gap_fk) if g is not None]
            gap_med = float(np.median(avail)) if avail else None
            at_cube_robust = bool(gap_med is not None and gap_med < 0.01)         # robust "reached surface" gate

            # ---- push the annotated camera frames so the detections are EYE-verifiable in the browser ----
            dash.update_frame("wrist", annotate_wrist(wrist_rgb, wun, wvn, wvis, wdepth, wrist_det is not None))
            dash.update_frame("front", annotate_front(cur_front, cube_world, uv_grip, fk_pred_uv,
                                                       K, T_cam2world, plane_z))

            snap = dict(phase="DEEP-REACH", step=step, grip_q=float(np.mean(pr.grip_qpos)),
                        joints={"q": list(map(float, pr.arm_qpos)),
                                "tau": list(map(float, getattr(pr, "arm_torque", [0] * 6)))},
                        tcp=tcp_w, tcp_io=tcp_io, wrist_uv=[round(wun, 2), round(wvn, 2), wvis],
                        gates={"z_bias": round(z_bias, 3),                       # descend depth (auto or fixed)
                               "auto_z": auto_z,                                 # closed loop on z_gap_io?
                               "fk_bias": round(fk_bias_ema, 3),                 # robot_io_z - base_ctrl_z (FK underread)
                               "grasp_zgap_io": round(grasp_zgap_io, 3),         # GOAL z_gap_io (VERIFY ④ + auto-z setpoint)
                               "cube_raw": cube_raw,                             # RAW front estimate (pre oblique-angle correction)
                               "cube_dxy": [round(cube_dxy[0], 3), round(cube_dxy[1], 3)],  # live task-space target nudge
                               "wrist_xy": [round(wrist_xy[0], 3), round(wrist_xy[1], 3)] if wrist_xy else None,  # wrist unproj (angle-free near cube)
                               "wf_dxy": (round(float(np.linalg.norm(np.array(wrist_xy) - np.array(cube_world[:2]))), 3)
                                          if (wrist_xy and cube_world is not None) else None),  # wrist↔front xy gap = the front bias
                               "z_gap_io": round(tcp_io[2] - plane_z, 3),        # robot_io FK height above surface
                               "z_gap_front": round(gap_front, 3) if gap_front is not None else None,  # (1) front
                               "z_gap_fk": round(gap_fk, 3),                     # (3) base-z-corrected FK gap
                               "z_gap_med": round(gap_med, 3) if gap_med is not None else None,  # fused (median)
                               "z_wrist_depth": z_wrist_depth,                   # (2) wrist depth (display only)
                               "at_cube_robust": at_cube_robust,                 # robust gap<1cm (readout; not driving)
                               "grip_tau": round(float(getattr(pr, "grip_torque", 0.0)), 3),  # gripper motor torque
                               "grasped": bool(io.is_grasped(pr.grip_qpos))})    # torque>0.5 -> holding the cube
            q_cmd = None
            at_cube = torch.zeros(1, dtype=torch.bool)
            reached_run = False                          # deploy "reached the cube" = ACCURATE z + xy on grasp target
            if cube_world is not None:
                cube_tx = cube_world[0] + cube_dxy[0]                      # detected cube + LIVE task-space nudge
                cube_ty = cube_world[1] + cube_dxy[1]
                cw_xy = torch.tensor([[cube_tx, cube_ty]], dtype=torch.float32)
                # JAW-FWD: the real jaws close ~jaw_fwd FORWARD (+y) of the base-FK TCP, so aim the TCP target
                # jaw_fwd BACK (-y) -> the JAWS land on the cube (deploy analog of robot_io.JAW_FWD).
                jaw_off = torch.tensor([[0.0, -jaw_fwd]], dtype=torch.float32)
                q_cmd, at_cube = base.deep_reach_step(
                    arm_q, cw_xy, torch.tensor([wun]), torch.tensor([wvn]), torch.tensor([wvis]),
                    ibvs=ibvs_on, tcp_xy_offset=jaw_off)
                gtx, gty = cube_tx, cube_ty - jaw_fwd                      # nudged + jaw-fwd corrected grasp target xy
                xy_err_val = float(np.linalg.norm(np.array(tcp_w[:2]) - np.array([gtx, gty])))
                # 'reached' = ACCURATE z_gap_io at goal + xy on the grasp target. NOT base_ctrl at_cube, which
                # fires too high while the auto-z EMA is still catching up to the descent (the "at_cube but too
                # high" symptom). RUN stops on this; VERIFY ⑤ gates on it.
                reached_run = abs(zgap_io_val - grasp_zgap_io) < 0.015 and xy_err_val < base.reach_eps
                snap.update(cube_top=[round(cube_world[0], 3), round(cube_world[1], 3)],
                            grasp_xy=[round(gtx, 3), round(gty, 3)],       # jaw-fwd corrected target (where TCP aims)
                            xy_err=round(xy_err_val, 3),                   # tcp ↔ GRASP target (jaw-fwd corrected)
                            z_gap=round(float(tcp_w[2] - plane_z), 3),     # base_ctrl tcp - cube center (FK shadow)
                            at_cube=bool(at_cube[0]),                      # base_ctrl gate (readout)
                            reached=reached_run)                           # accurate deploy reached (drives RUN stop)
                snap["gates"]["ibvs"] = ibvs_on

            # ---- VERIFY: evaluate the current step's gate against this snapshot, expose to the dashboard ----
            snap["captured"] = cube_world is not None
            vs = verify_seq[verify_idx]
            try:
                gp = vs["gate"](snap)
            except Exception:
                gp = None
            snap["verify"] = dict(idx=verify_idx, total=len(verify_seq), name=vs["name"],
                                  instr=vs["instr"], gate=("manual" if gp is None else bool(gp)))
            dash.update(**snap)

            c = dash.poll_cmd()
            if c:
                a, ex = c
                if a == "abort":
                    dash.update(msg="ABORT -> estop"); io.estop(); break
                elif a == "capture":
                    # capture sets cube_world (front-cam ray-plane), then corrects the OBLIQUE-CAM xy bias
                    # (shift AWAY from the cam by top_bias). Reuse THIS loop's front frame.
                    if cur_front is None:                    # GUARD: a dead/missing front cam must not crash capture
                        dash.update(msg="capture FAILED: front cam gave NO frame — check the device / --front-dev "
                                        "(V4L2 'No such device' = wrong or metadata-only node, or unplugged).")
                    else:
                        est = P.estimate_red_cube_world(cur_front, K, T_cam2world, plane_z)
                        cube_raw = [round(float(est[0]), 3), round(float(est[1]), 3)] if est is not None else None
                        est = correct_top_xy(est, T_cam2world, top_bias)
                        cube_world = list(map(float, est)) if est is not None else None
                        dash.update(msg=f"captured cube={np.round(cube_world, 3) if cube_world else None} "
                                        f"(raw {cube_raw}, top_bias {top_bias:+.3f} cam-away)")
                elif a == "reset":                           # return to REST hover (gripper open); clear modes/latch
                    auto_running = False; approach_hi = False; gripping = False
                    grip_cmd = cfg.grip_open_qpos
                    io.initial_liftup(rest_np, cfg.grip_open_qpos)   # scripted lift-clear -> REST hover, open (BLOCKS ~2s)
                    base.reset()                              # clear BOTH latches (_is_latched + _descend_latched)
                    q_send[:] = rest_np                       # resync the smoothing state to the achieved REST pose
                    dash.update(msg="RESET ↩ -> REST hover (gripper open; reach + descend latches + modes cleared)")
                elif a == "zero":                            # park at the URDF home ZERO (q=0); gripper open
                    auto_running = False; approach_hi = False; gripping = False
                    grip_cmd = cfg.grip_open_qpos
                    io.comeback_to_zero(cfg.grip_open_qpos)   # scripted: open + lift-clear via APPROACH -> ZERO (BLOCKS ~2.4s)
                    base.reset()                              # clear BOTH latches
                    q_send[:] = 0.0                           # ZERO = all joints 0 -> resync smoothing state
                    dash.update(msg="ZERO ⊙ -> URDF home q=0 (gripper open; latches + modes cleared)")
                elif a == "approach":                        # rule IK to a high TCP z over the cube = approach START
                    approach_hi = True; auto_running = False
                    base.reset()                              # clear stale latches so the post-approach RUN re-reaches
                    q_send[:] = pr.arm_qpos                    # resync smoothing to the measured pose before the move
                    dash.update(msg=f"APPROACH ↑ -> rule IK to TCP z={approach_z:.2f} over cube (cam down); "
                                    "Stop/Hold halts. Then RUN to deep-reach down.")
                elif a == "step" and q_cmd is not None:
                    send_arm(q_cmd[0].numpy(), grip_cmd, snap=True)  # one deep_reach servo step (CRISP, full step)
                    step += 1
                elif a == "jog":
                    q = pr.arm_qpos.copy(); q[ex["j"]] += ex["d"] * 0.02
                    send_arm(q, grip_cmd, snap=True)          # jog (CRISP full nudge); up after a close = lift test
                elif a == "hold":
                    auto_running = False; approach_hi = False  # Hold HALTS RUN and the approach move
                    send_arm(pr.arm_qpos, grip_cmd, snap=True)
                elif a == "close":                           # grasp test: drive grip PAST the close floor (overdrive)
                    auto_running = False; approach_hi = False  # SAFETY: closing HALTS any reach/descent (gripper must
                    #   stay OPEN through reach) so a close mid-RUN can't keep driving the CLOSED gripper down.
                    grip_cmd = grip_force_qpos               # PERSISTENT + SUSTAINED (re-commanded each idle loop)
                    gripping = True
                    send_arm(pr.arm_qpos, grip_cmd, snap=True)  # freeze arm, motor presses cube harder -> torque rises
                    dash.update(msg=f"grip CLOSE(force) -> {grip_cmd:.3f} sustained (watch grip_tau / grasped)")
                elif a == "open":
                    grip_cmd = cfg.grip_open_qpos
                    gripping = False
                    send_arm(pr.arm_qpos, grip_cmd, snap=True)
                    dash.update(msg=f"grip OPEN -> {grip_cmd:.3f}")
                elif a in ("grip_tight", "grip_loose"):      # LIVE grip-force dial: command PAST the close floor to
                    #   clamp the cube harder (lower qpos -> more +deg). Watch grip_tau rise toward the cube-grip
                    #   level (~1.86). If grip_q won't drop below grip_close_qpos, the closed-cal clamps it.
                    d = -0.005 if a == "grip_tight" else 0.005
                    grip_cmd = float(np.clip(grip_cmd + d, cfg.grip_close_qpos - 0.02, cfg.grip_open_qpos))
                    gripping = grip_cmd < 0.5 * (cfg.grip_open_qpos + cfg.grip_close_qpos)   # sustain only when closing
                    send_arm(pr.arm_qpos, grip_cmd, snap=True)
                    try:
                        deg = io._grip_qpos_to_deg(grip_cmd)
                    except Exception:
                        deg = float("nan")
                    dash.update(msg=f"grip {'TIGHTER' if d < 0 else 'LOOSER'} -> qpos {grip_cmd:.3f} ({deg:+.0f}deg); "
                                    "watch grip_tau (cube grip ~1.86) + grasped. If grip_q won't go below "
                                    f"{cfg.grip_close_qpos:.3f}, the closed-cal (grip_closed_deg) clamps it -> recalibrate.")
                elif a == "lift":                            # raise TCP +z (cam-down), holding grip = LIFT test
                    tgt = base.fk_world(arm_q)[0].clone(); tgt[2] += 0.05
                    down_v = torch.tensor([[0.0, 0.0, -1.0]])
                    q_lift = base._ik_step6(arm_q, tgt.unsqueeze(0), down_v)
                    send_arm(q_lift[0].numpy(), grip_cmd, snap=True)  # CRISP full step; grip stays (closed after grasp)
                    dash.update(msg=f"LIFT +z (grip={grip_cmd:.3f}); click repeatedly to keep rising")
                elif a in ("zbias_up", "zbias_dn"):
                    deeper = (a == "zbias_up")               # '+' button = go DEEPER (matches the old z_bias intent)
                    if auto_z:                               # auto-z: ± nudges the GOAL z_gap_io (deeper = lower goal)
                        grasp_zgap_io += (-0.005 if deeper else 0.005)
                        dash.update(msg=f"auto-z goal z_gap_io={grasp_zgap_io:+.3f} ({'deeper' if deeper else 'shallower'})")
                    else:                                    # manual: ± nudges the fixed descend depth
                        z_bias = min(Z_BIAS_MAX, max(0.0, z_bias + (0.01 if deeper else -0.01)))
                        base.cube_top_z = plane_z - z_bias
                        dash.update(msg=f"z_bias={z_bias:.3f} (manual; descend = surface - z_bias)")
                elif a == "ibvs":
                    ibvs_on = not ibvs_on                    # toggle wrist closed-loop XY (sag-tolerant)
                    dash.update(msg=f"IBVS {'ON' if ibvs_on else 'OFF'} (wrist closed-loop XY; for off-center)")
                elif a in ("cube_xp", "cube_xm", "cube_yp", "cube_ym", "cube_z0"):
                    if a == "cube_z0":
                        cube_dxy = [0.0, 0.0]
                    else:
                        i = 0 if a in ("cube_xp", "cube_xm") else 1
                        cube_dxy[i] += 0.01 if a in ("cube_xp", "cube_yp") else -0.01
                    dash.update(msg=f"cube target nudge dxy={[round(v,3) for v in cube_dxy]} (task-space; per-position trim)")
                elif a == "next":                            # VERIFY: advance to the next bring-up step (wraps)
                    verify_idx = (verify_idx + 1) % len(verify_seq)
                    dash.update(msg=f"VERIFY -> {verify_idx + 1}/{len(verify_seq)}: {verify_seq[verify_idx]['name']}")
                elif a == "run":                             # ONE-SHOT: auto deep_reach every loop until reached
                    auto_running = (cube_world is not None)
                    if auto_running:
                        q_send[:] = pr.arm_qpos               # resync smoothing to the measured pose before the run
                    dash.update(msg="RUN ▶ auto deep_reach until reached (z_gap_io≈goal; Stop/Hold/ABORT halts)"
                                if auto_running else "RUN ignored: Capture cube first")
                elif a == "stop":
                    auto_running = False; approach_hi = False
                    send_arm(pr.arm_qpos, grip_cmd, snap=True)   # freeze at the current pose
                    dash.update(msg="STOP ■ auto/approach halted, holding pose")
            elif approach_hi:
                # rule-based DLS-IK to a high TCP hover (cam straight down) over the cube (or workspace center) =
                # a clean, repeatable approach START at TCP z=approach_z. One-shot: holds when reached.
                tgt_xy = cube_world[:2] if cube_world is not None else [ws_cx, ws_cy]
                target = torch.tensor([[float(tgt_xy[0]), float(tgt_xy[1]), approach_z]], dtype=torch.float32)
                down = torch.tensor([[0.0, 0.0, -1.0]])
                q_appr = base._ik_step6(arm_q, target, down)
                send_arm(q_appr[0].numpy(), grip_cmd)        # smoothed rule-IK approach
                step += 1
                if abs(tcp_io[2] - approach_z) < 0.02 and \
                        float(np.linalg.norm(np.array(tcp_w[:2]) - np.array(tgt_xy, float))) < 0.03:
                    approach_hi = False                       # gate on robot_io z (accurate), not base_ctrl tcp_w
                    dash.update(msg=f"APPROACH  at TCP z≈{approach_z:.2f} over cube. RUN to deep-reach down.")
            elif auto_running and q_cmd is not None:
                send_arm(q_cmd[0].numpy(), grip_cmd)         # ONE-SHOT RUN: deep_reach servo EVERY loop (smoothed)
                step += 1
                if reached_run:                              # ACCURATE reached (z_gap_io≈goal + xy), not base at_cube
                    auto_running = False
                    dash.update(msg=f"RUN  reached cube (step {step}, z_gap_io≈goal). gripper OPEN — Grip close to grasp.")
            elif gripping:
                send_arm(pr.arm_qpos, grip_cmd, snap=True)   # no button this loop -> SUSTAIN the close (hold pose)
            if cur_front is not None:
                prev_front = cur_front                        # carry THIS frame -> next loop's diff "previous"
            time.sleep(max(0.0, dt - (time.time() - t0)))
    finally:
        io.estop(); dash.stop()
        print("[deep-reach] done (torque off).", flush=True)


if __name__ == "__main__":
    main()

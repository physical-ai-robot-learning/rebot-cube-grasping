"""Vision thread for the deploy control stack (3-thread refactor Step 2, docs/28).

The AsyncPerception camera/detection thread + its detector/annotation helpers, moved VERBATIM out of
deploy/debug_deepreach.py so the vision stage is ONE file you can breakpoint in isolation (PyCharm).
Interface unchanged (set_ctx / snapshot / request_capture / clear_cube / start_running / stop) -- the
control loop drives it exactly as before; the Blackboard rewiring happens when the IO/control threads
split (Steps 3-4).

Owns ONLY cameras + detectors + pure geometry (perception P / front_gripper_z / annotate_*). Never
touches io (CAN bus = control's) or base (IK = control's) -> no lock vs the control loop's motor I/O.
"""
from __future__ import annotations
import os, re, time, threading
import numpy as np

# This file already lives inside the rebot_sim2real package (deploy/vision/), so the old sys.path manipulation is unnecessary.
# perception (HSV, ray-plane) is imported by package absolute path; front_gripper_z (front-cam gripper Z) is a sibling module in vision/, so it is imported relatively.
from rebot_sim2real import perception as P                # HSV+ray-plane cube world-coordinate estimate (shared with deploy)
from . import front_gripper_z                             # front-camera frame-difference based gripper Z (same vision/ package)


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


USE_MJPG = True   # both cams DO support MJPG (v4l2 fmt [1]); set False for raw YUYV. Toggled by debug_deepreach --mjpg.


def _read_rgb(idx, focus=None, exposure=None, cache={}):
    import cv2
    if idx not in cache:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)    # explicit V4L2 backend (integer index)
        # ⭐ MJPG (compressed) transport: BOTH cams support MJPG (v4l2 fmt [1], corrected 2026-07-20 — the old
        # "YUYV-only" note was WRONG). Uncompressed YUYV 720p x2 saturates USB2.0 (480M) -> ~10fps each (the rd_w/rd_f
        # bottleneck); MJPG is ~10x smaller on the wire -> both cams fit USB2 at ~30fps. SAME 1280x720 resolution ->
        # intrinsics/K UNCHANGED (only the transport format); OpenCV decodes MJPG->BGR (~1-3ms). FOURCC BEFORE res.
        if USE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        if exposure is not None:                     # MANUAL exposure — the C922 Aperture-Priority auto blows
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   #   this rig's frame to pure white (red/cube washed out)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
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
    trained wrist CNN ckpt is missing. Wrist input for deep_reach_step (same convention as the CNN path)."""
    if rgb is None:
        return 0.0, 0.0, 0.0
    H, W = rgb.shape[:2]
    c = P._largest_blob_centroid(P.red_mask(rgb.astype(np.float32)))
    if c is None or c[2] < 8:
        return 0.0, 0.0, 0.0
    return (c[0] - W / 2.0) / (W / 2.0), (c[1] - H / 2.0) / (H / 2.0), 1.0


def _recenter_frac_roi(roi, c_px, hw):
    """Move a fractional ROI (y0,y1,x0,x1) so it re-centers on the pixel centroid c_px=(u,v), KEEPING its size,
    clamped to [0,1]. The detect-then-track loop calls this each frame so the (cheap HSV) search box FOLLOWS the
    cube between the (slow ~0.35s) OWLv2 re-anchors -> per-frame LIVE tracking of a moving cube."""
    H, W = hw
    y0, y1, x0, x1 = roi
    dy, dx = (y1 - y0), (x1 - x0)
    cy, cx = float(c_px[1]) / H, float(c_px[0]) / W
    ny0 = min(max(cy - dy / 2.0, 0.0), max(0.0, 1.0 - dy))
    nx0 = min(max(cx - dx / 2.0, 0.0), max(0.0, 1.0 - dx))
    return (ny0, ny0 + dy, nx0, nx0 + dx)


def _sat_blob_centroid(rgb, roi, s_min=0.40, v_min=0.25):
    """COLOR-AGNOSTIC dominant-blob centroid (u,v,area) inside `roi` (fractional y0,y1,x0,x1): the most saturated
    region — the queried object WHATEVER its colour vs the low-saturation table. This is why the open-vocab path
    uses THIS instead of the red-only mask: OWLv2 already gates the box to the queried object, so a plain
    saturation threshold inside that box localizes it regardless of colour — a 'purple cube' query gives the
    PURPLE cube's centroid, not a hardcoded red one (the earlier bug). Saturated colours only: a white/grey object
    would not separate from the table (fall back to a colour-specific mask for those). None if too small."""
    if roi is not None:
        # CROP FIRST, then HSV only the small box. Was: full-720p rgb_to_hsv EVERY call (~75ms) even for a tiny
        # tracked box -> the front/wrist per-frame bottleneck. Cropping first makes it ~(box/frame) cheaper.
        H, W = rgb.shape[:2]
        y0, y1, x0, x1 = roi
        iy0, iy1, ix0, ix1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        sub = rgb[iy0:iy1, ix0:ix1]
        if sub.size == 0:
            return None
        hsv = P.rgb_to_hsv(sub.astype(np.float32))
        m = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
        ys, xs = np.nonzero(m)
        if xs.size < 25:                     # blob too small / absent in the box
            return None
        # simple centroid: the OWLv2 box already gates to the object, so inside it a plain mean is accurate (and
        # exact vs the frame — no binning quantization) + faster than _largest_blob_centroid. Offset to full-frame px.
        return (float(xs.mean()) + ix0, float(ys.mean()) + iy0, int(xs.size))
    hsv = P.rgb_to_hsv(rgb.astype(np.float32))
    m = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
    return P._largest_blob_centroid(m)


def wrist_uvn_ov(rgb, det):
    """OPT-IN wrist (u,v) via OWLv2 open-vocab box -> COLOUR-AGNOSTIC saturated-blob centroid INSIDE the box (same
    box-gate idea as the front cam, and colour-agnostic so it follows WHATEVER the query selects, not just red).
    Same normalized (un,vn,vis) ∈ [-1,1] convention as wrist_uvn_redmask. NO depth / NO CNN yaw (caller returns 0
    -> Align falls back to the seg minArea yaw). Strictly opt-in (--wrist-openvocab); sd3 CNN descend untouched."""
    if rgb is None:
        return 0.0, 0.0, 0.0
    H, W = rgb.shape[:2]
    for roi, _sc in det.rois(rgb):                     # best-score box first; first box with a saturated blob wins
        c = _sat_blob_centroid(rgb, roi)
        if c is not None:
            return (c[0] - W / 2.0) / (W / 2.0), (c[1] - H / 2.0) / (H / 2.0), 1.0
    return 0.0, 0.0, 0.0


def _edge_vote_yaw(bin_img, min_conf=0.55, min_side_px=4.0):
    """Robust in-image yaw (mod-90deg) of a square top face by EDGE-DIRECTION voting — the "extract the cube's
    sides" idea. Take the cube boundary, split it into straight SIDES (approxPolyDP), and return the
    LENGTH-WEIGHTED circular mean of the side orientations folded mod-90 (a square's 4 sides fall on ONE mod-90
    direction). EVERY side votes, so — unlike cv2.minAreaRect, which is pinned by just 2 extreme points and
    collapses toward 0/90deg under mask noise / a square window / one occluded corner — a noisy or oblique side
    averages out. If too many sides disagree the resultant magnitude (conf) drops, and the caller keeps the
    minAreaRect angle (= no regression).
    Returns (cos4, sin4, ang_deg in [-45,45], conf in [0,1]) or None (degenerate / conf < min_conf)."""
    import cv2
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = float(cv2.arcLength(c, True))
    if peri < 24.0 or cv2.contourArea(c) < 30.0:
        return None
    approx = cv2.approxPolyDP(c, 0.03 * peri, True).reshape(-1, 2).astype(np.float64)
    if approx.shape[0] < 3:                                     # under-resolved polygon -> raw contour walk
        approx = c.reshape(-1, 2).astype(np.float64)
    # ---- Symbol map (×4 fold + length-weighted circular mean) ----
    #   @th (θ)        = in-image direction angle of one side. A square's 4 sides are 90° apart, so folding by 4θ (mod-90)
    #                    makes the four sides overlap in one direction -> multiplying the angle by 4 is that fold (multiplicity ×4).
    #   @acc_r, @acc_i = Σ L·cos(4θ), Σ L·sin(4θ)  (resultant vector weighted by side length L = length-weighted resultant)
    #   @conf          = |resultant|/ΣL  (1.0 = all sides agree on one mod-90 direction; low = noise/disagreement)
    acc_r = acc_i = tot = 0.0
    n = approx.shape[0]
    for i in range(n):
        dx, dy = approx[(i + 1) % n] - approx[i]                # closed polygon: edge i -> i+1 (wraps)
        L = float(np.hypot(dx, dy))
        if L < min_side_px:                                     # skip corner-rounding micro-segments
            continue
        th = np.arctan2(dy, dx)
        acc_r += L * np.cos(4.0 * th); acc_i += L * np.sin(4.0 * th); tot += L   # @acc_r/@acc_i = Σ L·cos/sin(4θ)
    mag = float(np.hypot(acc_r, acc_i))                         # |resultant|
    if tot < 1.0 or mag < 1e-6:
        return None
    conf = mag / tot                                            # |resultant|/ΣL: 1.0 = every side agrees on one mod-90 direction
    if conf < min_conf:
        return None
    cos4, sin4 = acc_r / mag, acc_i / mag
    ang_deg = float(np.degrees(np.arctan2(acc_i, acc_r) / 4.0))
    return float(cos4), float(sin4), ang_deg, float(conf)


def _seg8_from_bin(bin_img):
    """seg8 (un,vn,vis,cos4,sin4,ang_deg,box,top_vn) from a binary cube mask (uint8 HxW, full frame):
      un,vn     normalized centroid in [-1,1] (minAreaRect center = unbiased for a square).
      vis       1.0 if enough cube pixels, else 0.0.
      cos4,sin4 = (cos4theta, sin4theta): theta = EDGE-VOTED face yaw (mod-90); minAreaRect-angle fallback when
                  the sides don't agree. Feeds the J6 image-yaw null (deploy latch / Align J6 roll). (0,0) if vis=0.
      ang_deg   the reported yaw in degrees (display); NaN when vis=0.
      box       4x2 int overlay corners, oriented AT the reported yaw (so the magenta box matches the number).
      top_vn    cube TOP-edge y, normalized [-1(top)..+1(bottom)]."""
    import cv2
    H, W = bin_img.shape[:2]
    ys, xs = np.nonzero(bin_img)
    if xs.size < 15:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), None, 0.0
    rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
    (rcx, rcy), (rw, rh), mar_ang = rect
    ev = _edge_vote_yaw(bin_img)
    if ev is not None:
        cos4, sin4, ang_deg, _conf = ev                         # robust edge-voted yaw (extract the square's sides)
    else:
        ang_deg = float(mar_ang); theta = np.deg2rad(ang_deg)   # fallback = old minAreaRect angle (no regression)
        cos4, sin4 = float(np.cos(4.0 * theta)), float(np.sin(4.0 * theta))
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), ang_deg)).astype(np.int32)   # overlay box AT the reported yaw
    un = (float(rcx) - W / 2.0) / (W / 2.0)
    vn = (float(rcy) - H / 2.0) / (H / 2.0)
    top_vn = (float(box[:, 1].min()) - H / 2.0) / (H / 2.0)
    return un, vn, 1.0, cos4, sin4, ang_deg, box, top_vn


def _clean_blob(b):
    """Mask cleanup (docs/33): morphological OPEN removes specular/noise fragments + keeps only the largest connected component so
    edge-voting focuses on a single cube outline (mitigates fragments that appear and vanish frame to frame making the mod-90 yaw
    jump around over time — docs/33 seg-yaw instability). If degenerate (all erased / too small), keep the original."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    o = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((o > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return b
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == biggest).astype(np.uint8) * 255
    if int((keep > 0).sum()) < 15:
        return b
    return keep


def _cube_bin_redmask(rgb, near_frac=0.22):
    """Full-frame binary cube mask (uint8 0/255) from the HSV red mask, WINDOWED to the dominant red blob (a
    blob-scaled window isolates the cube's top face from far red clutter — the same windowing wrist_seg used).
    None if no blob. (legacy red-cube path.)"""
    H, W = rgb.shape[:2]
    mask = P.red_mask(rgb.astype(np.float32))
    c = P._largest_blob_centroid(mask)                          # dominant-blob SEED (u,v,area); rejects stray red
    if c is None or c[2] < 8:
        return None
    cu, cvv, area = c
    ys, xs = np.nonzero(mask)
    r = float(np.clip(1.3 * np.sqrt(max(area, 1.0)), 40.0, near_frac * max(W, H)))
    near = (np.abs(xs - cu) <= r) & (np.abs(ys - cvv) <= r)
    if int(near.sum()) < 15:                                    # degenerate window -> keep all red pixels
        near = np.ones(xs.shape, bool)
    b = np.zeros((H, W), np.uint8)
    b[ys[near], xs[near]] = 255
    return _clean_blob(b)


def _cube_bin_ov(rgb, wrist_ov, s_min=0.40, v_min=0.25):
    """Full-frame binary cube mask, COLOUR-AGNOSTIC: the OWLv2 box gates the region, a saturation threshold
    inside it localizes the queried object WHATEVER its colour (red/green/...). None if no box / blob too small.
    Mirrors _sat_blob_centroid's gate, so the Align yaw is fit on the SAME object the descend (u,v) follows."""
    if wrist_ov is None:
        return None
    H, W = rgb.shape[:2]
    for roi, _sc in wrist_ov.rois(rgb):                         # best-score box first
        y0, y1, x0, x1 = roi
        iy0, iy1, ix0, ix1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        sub = rgb[iy0:iy1, ix0:ix1]
        if sub.size == 0:
            continue
        hsv = P.rgb_to_hsv(sub.astype(np.float32))
        m = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
        if int(m.sum()) < 25:
            continue
        b = np.zeros((H, W), np.uint8)
        b[iy0:iy1, ix0:ix1][m] = 255
        return _clean_blob(b)
    return None


def wrist_seg(rgb, near_frac=0.22):
    """WRIST red-cube SEGMENTATION -> seg8 (un,vn,vis,cos4,sin4,ang_deg,box,top_vn) for J4/J5 centering + J6 align.
    HSV red mask windowed to the dominant blob (isolates the top face), then _seg8_from_bin (EDGE-VOTED yaw).
    Red-mask based (no detector dependency). The colour-agnostic OWLv2-box path lives in _align_seg."""
    if rgb is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), None, 0.0
    b = _cube_bin_redmask(rgb, near_frac)
    if b is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), None, 0.0
    return _seg8_from_bin(b)


def _align_seg(rgb, wrist_ov=None):
    """WRIST Align seg -> (seg8, mask). Cube MASK priority (all fed to the SAME edge-voted yaw):
      1) OWLv2-box COLOUR-AGNOSTIC saturated blob (--wrist-openvocab) — follows the queried object of ANY colour
         (fixes the red-mask-only yaw that went blind on e.g. a green cube);
      2) HSV red mask (baseline). mask (for the dashboard fill) is always None here (no CNN seg head).
    ONLY drives the Align box/yaw; the descend (u,v) stays on the primary wrist detector -> descend unchanged."""
    if rgb is None:
        return wrist_seg(rgb), None
    b = _cube_bin_ov(rgb, wrist_ov)                             # colour-agnostic OV-box mask (red/green/...)
    if b is not None:
        return _seg8_from_bin(b), None
    return wrist_seg(rgb), None                                 # red-mask baseline


# ---- overlay helpers: draw the live detections on the camera frames so they can be EYE-verified ----
def _encode(img_bgr):
    """resize-for-transport + JPEG. img_bgr is annotated BGR (cv2 native). -> bytes (or None)."""
    import cv2
    H, W = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (512, max(1, int(512 * H / W))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def annotate_wrist(rgb, un, vn, vis, seg=None, mask=None, ov_box=None):
    """WRIST eye-in-hand frame: image-center crosshair (= the alignment/centering target) + the detected cube
    (u,v) (yellow) + the SEGMENTATION oriented box (magenta) with its in-image yaw.
    mask (if provided) is drawn as a translucent green fill so you SEE the actual segmentation.
    seg = (box_px, ang_deg, seg_un, seg_vn, seg_vis), or None.
    ov_box = ((x0,y0,x1,y1),score) draws the OWLv2 open-vocab gate (cyan) = the region HSV is restricted to."""
    import cv2
    if rgb is None:
        return None
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    if ov_box is not None:                                                            # OWLv2 open-vocab gate (cyan)
        (bx0, by0, bx1, by1), _sc = ov_box
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), (255, 255, 0), 2)
        cv2.putText(img, f"OV {_sc:.2f}", (int(bx0), int(by0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    if mask is not None:                                                              # translucent CNN seg mask
        m = cv2.resize(np.asarray(mask, np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        if m.any():
            ov = img.copy(); ov[m > 0] = (0, 210, 0)
            cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    cv2.drawMarker(img, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 26, 1)   # target = centered
    # ---- segmentation: oriented rect (magenta) + centroid + in-image yaw (the J6-align signal) ----
    if seg is not None and seg[0] is not None and seg[4] > 0.5:
        box, ang_deg, sun, svn = seg[0], seg[1], seg[2], seg[3]
        cv2.drawContours(img, [box.reshape(-1, 1, 2)], 0, (255, 0, 255), 2)           # segmented cube outline
        su = int((sun + 1.0) / 2.0 * W); sv = int((svn + 1.0) / 2.0 * H)
        cv2.circle(img, (su, sv), 3, (255, 0, 255), -1)
        cv2.putText(img, f"seg yaw={ang_deg:+.0f}deg", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    tag = "redmask"                                                                   # the (u,v) source is now always red-mask/OV
    if vis > 0.5:
        u = int((un + 1.0) / 2.0 * W); v = int((vn + 1.0) / 2.0 * H)
        col = (0, 255, 255)                                                           # yellow = red-mask/OV detection
        cv2.circle(img, (u, v), 13, col, 2); cv2.circle(img, (u, v), 2, col, -1)
        cv2.putText(img, f"{tag} vis=1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    else:
        cv2.putText(img, f"{tag}  NO DETECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return _encode(img)


def correct_top_xy(est, T, bias, x_corr=0.0):
    """Correct the front-cam ray-plane cube xy. Two independent terms:
    (1) `bias` (m): OBLIQUE-CAMERA shift AWAY from the cam along the horizontal cam->cube direction (legacy;
        used with the pre-calibration approx intrinsics). With a real checkerboard calib this is ~0.
    (2) `x_corr` (m): a CONSTANT world-X offset = the residual extrinsic-x error left after the real intrinsic
        calib. MEASURED on this rig (8-pt grid, 2026-07-07): the front cube-x was a flat -2.35cm (no scale, no
        depth-dependence) -> +0.0235 recenters it (fit residual RMS ~3.4mm). Applied EVEN when bias==0.
    Returns a NEW xyz (z unchanged)."""
    if est is None:
        return est
    out = np.array(est, float)
    if bias != 0.0:
        cam_xy = np.asarray(T, float)[:2, 3]             # front-cam world xy (T_cam2world translation)
        d = np.asarray(est[:2], float) - cam_xy          # horizontal cam -> cube (points AWAY from the cam)
        n = float(np.linalg.norm(d))
        if n >= 1e-9:
            out[:2] = out[:2] + bias * (d / n)           # push AWAY from the cam -> cancels the toward-cam bias
    out[0] += x_corr                                     # constant world-X recentering (residual extrinsic-x)
    return out


def annotate_front(rgb, cube_world, uv_grip, fk_pred_uv, K, T, plane_z, box=140, ov_box=None):
    """FRONT fixed frame: cube xy marker (orange, projected captured world) + gripper frame-diff detection
    (green) + the FK-prior search box (grey). So you SEE whether front cube-detect + gripper-Z inputs are sane.
    ov_box=((x0,y0,x1,y1),score) draws the OWLv2 open-vocab gate (cyan) = the region HSV is restricted to."""
    import cv2
    if rgb is None:
        return None
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ov_box is not None:                                    # OWLv2 open-vocab gate (cyan) -> the dynamic HSV ROI
        (bx0, by0, bx1, by1), _sc = ov_box
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), (255, 255, 0), 2)
        cv2.putText(img, f"OV {_sc:.2f}", (int(bx0), int(by0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
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


# ---- deploy bring-up dependency chain (the button flow encodes it):
#   cam identity -> Capture (cube detect) -> RUN (descend-latch to grasp-z, front-gripper Z) -> Align (center + J6)
#   -> Final (descend) -> Grip close (force-servo) -> Lift.


class AsyncPerception(threading.Thread):
    """Decoupled camera/detection thread. Grabs front+wrist, runs the wrist detector, does the front-gripper-Z
    frame-diff + annotates the dashboard frames, and (on request) the front-cam cube capture — ALL at the camera's
    own rate. Publishes the latest results for the control loop to read non-blockingly via snapshot().

    Owns ONLY the cameras + detector + pure geometry helpers (front_gripper_z / perception P / annotate_*). Never
    touches `io` (CAN bus = control's) or `base` (IK/latches = control's) -> the cameras (USB video) and motors
    (CAN) are separate buses, so this needs NO lock against the control loop's motor I/O. The control loop publishes
    the current FK TCP via set_ctx() (front-Z search prior); this thread publishes detections back."""

    def __init__(self, front_idx, wrist_idx, front_focus, wrist_focus, wrist_uvn_depth_fn,
                 K, T, plane_z, top_bias, dash, dist=None, hz=30.0, front_exposure=None, wrist_exposure=None,
                 x_corr=0.0, front_roi=None, front_smin=0.45,
                 front_ov=None, wrist_ov=None):
        super().__init__(daemon=True, name="perception")
        self._front_idx, self._wrist_idx = front_idx, wrist_idx
        self._front_focus, self._wrist_focus = front_focus, wrist_focus
        self._front_exposure, self._wrist_exposure = front_exposure, wrist_exposure   # manual exposures (auto blows to white)
        self._wrist_uvn_fn = wrist_uvn_depth_fn
        self._front_ov = front_ov                             # OPT-IN OWLv2 open-vocab ROI provider (box->HSV-in-box); None = off
        self._wrist_ov = wrist_ov                             # OPT-IN OWLv2 wrist ROI (only for the overlay; (u,v) via the closure)
        self._track_roi = None                                # detect-then-track: per-frame HSV search box (follows the cube)
        self._ov_redetect_stamp = -1                          # last-seen OWLv2 re-detect stamp -> re-anchor _track_roi on change
        self._ov_overlay_box = None                           # ((x0,y0,x1,y1),score) actual tracked box, for the dashboard overlay
        self._front_roi = front_roi                           # HSV workspace ROI (y0,y1,x0,x1 frac) -> reject off-table red
        self._front_smin = float(front_smin)                  # HSV red saturation floor (raise to reject skin)
        self._K, self._T, self._plane_z, self._top_bias, self._dash = K, T, plane_z, top_bias, dash
        self._dist = dist                                # real lens distortion -> undistort the capture pixel
        self._x_corr = x_corr                            # constant world-X recentering (residual extrinsic-x)
        self._period = 1.0 / hz
        self._lock = threading.Lock()
        self._running = False
        self._tcp = None                     # control-published FK TCP (front-Z prior)
        self._cube = None; self._cube_raw = None   # captured cube world xy(z) — OWNED here (front cam)
        self._cap_req = False
        self._need_align = False             # control-gated: run the heavy align_seg (720p, ~89ms) ONLY while Aligning
        self._snap = {"wun": 0.0, "wvn": 0.0, "wvis": 0.0, "gap_front": None,
                      "has_front": False, "hz": 0.0, "cube_live": None,   # live front est (--track)
                      "scos4": 0.0, "ssin4": 0.0, "sang": float("nan"), "sun": 0.0, "svn": 0.0, "svis": 0.0,
                      "stopvn": 0.0,   # seg cube TOP-edge normalized y (align auto-center target)
                      "ts": time.time(), "dead": False}   # ts=freshness (control-loop staleness fail-stop); dead=thread died

    def set_ctx(self, tcp_world_np, need_align=False):
        with self._lock:
            self._tcp = None if tcp_world_np is None else np.asarray(tcp_world_np, float).copy()
            self._need_align = bool(need_align)   # control tells the thread whether to run the heavy align_seg

    def request_capture(self):
        with self._lock:
            self._cap_req = True

    def clear_cube(self):
        with self._lock:
            self._cube = None; self._cube_raw = None

    def snapshot(self):
        """(latest-detection dict, cube_world, cube_raw) — non-blocking; cube_world is the capture result or None."""
        with self._lock:
            return dict(self._snap), (list(self._cube) if self._cube else None), \
                   (list(self._cube_raw) if self._cube_raw else None)

    def start_running(self):
        self._running = True
        self.start()

    def stop(self):
        self._running = False
        if self.is_alive():
            self.join(timeout=1.0)

    def run(self):
        # Wrap the whole loop: a throw in a camera read / detector / geometry must NOT silently kill this daemon
        # and leave the control loop reading a FROZEN snapshot (it would keep servoing to the stale target). On
        # ANY exception -> flag dead + stop; the control loop's staleness/dead check then fail-stops the arm.
        prev_front = None
        t_prev = time.perf_counter()
        try:
            while self._running:
                t0 = time.perf_counter()
                with self._lock:
                    tcp = None if self._tcp is None else self._tcp.copy()
                    cube = list(self._cube) if self._cube else None
                    cap = self._cap_req
                    need_align = self._need_align
                wrist_rgb = (_read_rgb(self._wrist_idx, focus=self._wrist_focus, exposure=self._wrist_exposure)
                             if self._wrist_idx is not None else None)
                _tm_rw = time.perf_counter()                               # TIMING: wrist read
                wun, wvn, wvis = self._wrist_uvn_fn(wrist_rgb)[:3]         # (un,vn,vis); depth/cos4/sin4 (=0, CNN dropped) unused
                _tm_uvn = time.perf_counter()                              # TIMING: wrist detect (OWLv2/redmask)
                if need_align:                                            # GATE: align_seg (720p, ~89ms) ONLY while Aligning
                    (sun, svn, svis, scos4, ssin4, sang, sbox, stopvn), seg_mask = _align_seg(wrist_rgb, self._wrist_ov)   # OV-box colour-agnostic > red-mask
                else:                                                     # not aligning -> skip the heavy seg (sentinel; Align consumes these only when active)
                    (sun, svn, svis, scos4, ssin4, sang, sbox, stopvn), seg_mask = (0.0, 0.0, 0.0, 1.0, 0.0, float("nan"), None, 0.0), None
                _tm_align = time.perf_counter()                            # TIMING: align_seg
                cur_front = _read_rgb(self._front_idx, focus=self._front_focus, exposure=self._front_exposure)
                _tm_rf = time.perf_counter()                              # TIMING: front read
                # ---- front-cam cube xy (ray-plane + oblique correction). Computed EVERY frame -> a LIVE estimate
                #      (est_live) for --track: the FIXED front cam gives a STABLE target (no eye-in-hand feedback),
                #      so following it can't oscillate. The Capture button additionally FREEZES it into self._cube
                #      (the baseline one-shot path, used when --track is off). ----
                est_live = None; raw_xy = None
                if cur_front is not None:
                    if self._front_ov is not None:           # OPEN-VOCAB detect-then-TRACK: OWLv2 gives a periodic ANCHOR box;
                        cube_world_est = None                #   between re-detects the box FOLLOWS the (COLOUR-AGNOSTIC) blob EVERY frame
                        rois = self._front_ov.rois(cur_front)   # (re-runs OWLv2 only every --front-ov-period; else cached)
                        box_score = rois[0][1] if rois else 0.0
                        if self._front_ov.stamp != self._ov_redetect_stamp:   # OWLv2 just re-ran -> (re)anchor the tracking box
                            self._ov_redetect_stamp = self._front_ov.stamp
                            if rois:
                                self._track_roi = rois[0][0]               # anchor to the top OWLv2 box (drift/occlusion reset)
                        if self._track_roi is not None:                    # per-frame: COLOUR-AGNOSTIC blob in the tracked box
                            centroid_px = _sat_blob_centroid(cur_front, self._track_roi, self._front_smin)   # any queried colour, not just red
                            if centroid_px is not None:
                                cube_world_est = P.cube_world_from_pixel(centroid_px[0], centroid_px[1], self._K, self._T, self._plane_z, self._dist)
                                self._track_roi = _recenter_frac_roi(self._track_roi, centroid_px, cur_front.shape[:2])  # FOLLOW
                        if cube_world_est is None:                         # blob dropout in the tracked box (occlusion/noise):
                            # fall back to an OWLv2 box for THIS frame's estimate but KEEP self._track_roi — a SINGLE
                            # dropout must NOT collapse the follow back to the (up-to-a-period-stale) OWLv2 anchor and
                            # lose a moving cube; the blob is retried in the SAME box next frame, and a new stamp re-anchors it.
                            for _roi, fallback_score in rois:
                                centroid_px = _sat_blob_centroid(cur_front, _roi, self._front_smin)
                                if centroid_px is not None:
                                    cube_world_est = P.cube_world_from_pixel(centroid_px[0], centroid_px[1], self._K, self._T, self._plane_z, self._dist)
                                    box_score = fallback_score; break
                        if cube_world_est is None and self._front_roi is not None:   # graceful fallback: static workspace ROI (colour-agnostic)
                            centroid_px = _sat_blob_centroid(cur_front, self._front_roi, self._front_smin)
                            if centroid_px is not None:
                                cube_world_est = P.cube_world_from_pixel(centroid_px[0], centroid_px[1], self._K, self._T, self._plane_z, self._dist)
                        # overlay = the ACTUAL tracked box (px), not the stale OWLv2 anchor, so you SEE it follow the cube
                        if self._track_roi is not None:
                            _H, _W = cur_front.shape[:2]; _r = self._track_roi
                            self._ov_overlay_box = ((_r[2] * _W, _r[0] * _H, _r[3] * _W, _r[1] * _H), box_score)
                        else:
                            self._ov_overlay_box = None
                        oblique_bias = self._top_bias        # HSV centroid -> same oblique shift as the HSV path
                    else:
                        cube_world_est = P.estimate_red_cube_world(cur_front, self._K, self._T, self._plane_z, self._dist,
                                                                   roi=self._front_roi, s_min=self._front_smin)  # HSV + workspace ROI/skin-reject
                        oblique_bias = self._top_bias
                    if cube_world_est is not None:
                        raw_xy = [round(float(cube_world_est[0]), 3), round(float(cube_world_est[1]), 3)]
                        est_live = list(map(float, correct_top_xy(cube_world_est, self._T, oblique_bias, self._x_corr)))
                    if cap:                                    # Capture -> FREEZE the current live estimate
                        cube = list(est_live) if est_live is not None else None
                        with self._lock:
                            self._cube, self._cube_raw, self._cap_req = cube, raw_xy, False
                cube_disp = cube if cube is not None else est_live   # overlay/gripper-Z target: capture else live
                _tm_front = time.perf_counter()                           # TIMING: front detect (OWLv2 rois + HSV centroid)
                # ---- front-gripper-Z (frame-diff; needs the control-published TCP prior + prev/cur front) ----
                gap_front = None; uv_grip = None; fk_pred_uv = None
                if cube_disp is not None and prev_front is not None and cur_front is not None and tcp is not None:
                    fk_pred_uv = front_gripper_z.project_world_to_front(tcp, self._K, self._T)
                    uv_grip = front_gripper_z.detect_gripper_uv(prev_front, cur_front, fk_pred_uv=fk_pred_uv)
                    if uv_grip is not None:
                        res = front_gripper_z.gripper_z_from_ray(uv_grip, cube_disp[:2], self._K, self._T, self._plane_z)
                        if res is not None:
                            _gz, gap_front, _t = res
                _tm_gz = time.perf_counter()                             # TIMING: front-gripper-Z frame-diff
                # ---- overlays -> dashboard frames (thread-safe: dash guards with its own lock) ----
                self._dash.update_frame("wrist", annotate_wrist(wrist_rgb, wun, wvn, wvis,
                                                                seg=(sbox, sang, sun, svn, svis), mask=seg_mask,
                                                                ov_box=(self._wrist_ov.last_box_px if self._wrist_ov is not None else None)))
                self._dash.update_frame("front", annotate_front(cur_front, cube_disp, uv_grip, fk_pred_uv,
                                                                self._K, self._T, self._plane_z,
                                                                ov_box=(self._ov_overlay_box if self._front_ov is not None else None)))
                _tm_annot = time.perf_counter()                          # TIMING: annotate x2 (720p overlays)
                hz = 1.0 / max(1e-3, t0 - t_prev); t_prev = t0
                _ov = self._front_ov or self._wrist_ov       # active open-vocab (front primary) -> live query/score readout
                _ovq = ", ".join(_ov.queries) if _ov is not None else None
                _ovs = round(_ov.last_box_px[1], 2) if (_ov is not None and _ov.last_box_px is not None) else None
                # BOTH cams share the ONE query but detect INDEPENDENTLY -> surface the WRIST's own query + live score
                # so the dashboard confirms the wrist cam really is on the same object (not just the front).
                _ovqw = ", ".join(self._wrist_ov.queries) if self._wrist_ov is not None else None
                _ovsw = (round(self._wrist_ov.last_box_px[1], 2)
                         if (self._wrist_ov is not None and self._wrist_ov.last_box_px is not None) else None)
                with self._lock:
                    self._snap = {"wun": wun, "wvn": wvn, "wvis": wvis, "gap_front": gap_front,
                                  "has_front": cur_front is not None, "cube_live": est_live,
                                  "scos4": scos4, "ssin4": ssin4, "sang": sang, "sun": sun, "svn": svn, "svis": svis,
                                  "stopvn": stopvn, "ov_q": _ovq, "ov_score": _ovs, "ov_q_wrist": _ovqw, "ov_score_wrist": _ovsw,
                                  "hz": round(hz, 1), "ts": time.time(), "dead": False,
                                  "tm": {"rd_w": round((_tm_rw - t0) * 1e3), "det_w": round((_tm_uvn - _tm_rw) * 1e3),
                                         "align": round((_tm_align - _tm_uvn) * 1e3), "rd_f": round((_tm_rf - _tm_align) * 1e3),
                                         "front": round((_tm_front - _tm_rf) * 1e3), "gz": round((_tm_gz - _tm_front) * 1e3),
                                         "annot": round((_tm_annot - _tm_gz) * 1e3)}}   # TIMING: per-stage ms -> /state.gates.tm
                prev_front = cur_front
                time.sleep(max(0.0, self._period - (time.perf_counter() - t0)))
        except Exception:
            import traceback
            traceback.print_exc()
            with self._lock:
                self._snap = dict(self._snap); self._snap["dead"] = True   # control loop fail-stops on this
        finally:
            self._running = False


# (Removed the VERIFY bring-up checklist/sequence — the live status table + button messages cover it.)

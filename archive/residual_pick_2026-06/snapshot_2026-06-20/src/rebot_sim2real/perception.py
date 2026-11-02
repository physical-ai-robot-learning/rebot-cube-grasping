"""Red-cube perception for sim2real deployment.

Estimates the red cube's 3-D position from a single RGB image, WITHOUT a depth sensor: the cube
rests on a known table plane (z = table_z + cube_size/2), so the cube-center pixel's camera ray is
intersected with that plane. This is the value the state policy needs (obs cube_pos); the rest of the
obs is proprioception (FK). cube_quat is ~identity (training used yaw_rand=0), so orientation is not
estimated here.

Pure numpy (no cv2 dependency). Same code runs on sim renders (V2 verification) and real camera frames
(deployment) — only K (intrinsics) and the camera world pose differ (sim: known; real: calibrated).

Conventions (Isaac USD camera): optical axis = camera-local -Z, image +x = local +x (right),
image +y = local +y (up) but pixel rows increase DOWNWARD, so row v maps to local +y as -(v-cy).
"""
from __future__ import annotations

import numpy as np


def rgb_to_hsv(rgb):
    """rgb uint8/float [H,W,3] in 0..255 (or 0..1) -> hsv float, h in [0,360), s,v in [0,1]."""
    a = rgb.astype(float)
    if a.max() > 1.0:
        a = a / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    mask = d > 1e-6
    # per-channel hue
    rc = ((mx - r) / np.where(mask, d, 1))
    gc = ((mx - g) / np.where(mask, d, 1))
    bc = ((mx - b) / np.where(mask, d, 1))
    h = np.where((mx == r) & mask, (bc - gc), h)
    h = np.where((mx == g) & mask, 2.0 + (rc - bc), h)
    h = np.where((mx == b) & mask, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0 * 360.0
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1), 0.0)
    return np.stack([h, s, mx], axis=-1)


def red_mask(rgb, s_min=0.45, v_min=0.25, h_lo=345.0, h_hi=15.0):
    """Boolean mask of saturated red pixels (hue wraps around 0/360)."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_h = (h >= h_lo) | (h <= h_hi)
    return red_h & (s >= s_min) & (v >= v_min)


def _largest_blob_centroid(mask, bins=24, win=2):
    """(u,v,area) pixel centroid of the DOMINANT red cluster; None if too small.

    Was the mean of ALL red pixels — but on a real cluttered scene a few scattered spurious red pixels
    (curtain edges, reflections, a stray red object high in frame) drag that mean off the cube; when the
    cube ray then grazes the table plane it casts to a far garbage world point (the deploy y=-2.07 abort).
    Robust version: bin red pixels into a coarse `bins`x`bins` grid, find the densest cell (the cube is a
    compact blob), and average only the pixels within +/-`win` cells of that peak. For a single clean blob
    (sim renders) this equals the old global centroid, so the sim/verify paths are unchanged."""
    ys, xs = np.nonzero(mask)
    n = xs.size
    if n < 15:                             # too few red pixels -> not detected
        return None
    H, W = mask.shape[-2:]
    bx = np.minimum(xs * bins // W, bins - 1)
    by = np.minimum(ys * bins // H, bins - 1)
    counts = np.bincount(by * bins + bx, minlength=bins * bins)
    pj, pi = divmod(int(counts.argmax()), bins)        # densest cell: row (y), col (x)
    sel = (np.abs(by - pj) <= win) & (np.abs(bx - pi) <= win)
    if int(sel.sum()) < 15:                            # degenerate peak -> fall back to all red
        sel = np.ones(n, dtype=bool)
    return float(xs[sel].mean()), float(ys[sel].mean()), int(sel.sum())


def estimate_red_cube_world(rgb, K, T_cam2world, plane_z):
    """Returns (x,y,z) WORLD position of the red cube center, or None if not detected.

    K: 3x3 pinhole intrinsics. T_cam2world: 4x4 camera->world extrinsic in the ROS optical convention
    (camera +Z forward, +X right, +Y down) -- in sim use inv(camera.get_view_matrix_ros()); on the real
    rig use the hand-eye-calibrated transform. plane_z = table_z + cube_size/2 (the cube sits on the
    table). Method: red-blob centroid pixel -> ROS camera ray -> intersect the table plane."""
    m = red_mask(rgb)
    c = _largest_blob_centroid(m)
    if c is None:
        return None
    u, v, _ = c
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])     # ROS optical: +Z fwd, +X right, +Y down
    T = np.asarray(T_cam2world, dtype=float)
    R = T[:3, :3]; origin = T[:3, 3]
    d = R @ d_cam
    d = d / (np.linalg.norm(d) + 1e-12)
    if abs(d[2]) < 1e-6:
        return None                        # ray parallel to the plane
    t = (plane_z - origin[2]) / d[2]
    if t <= 0:
        return None                        # plane behind the camera
    p = origin + t * d
    return np.array([p[0], p[1], plane_z], dtype=float)


def K_from_c922(focal_mm, h_aperture_mm, v_aperture_mm, width, height):
    """Pinhole K from the C922 focal/aperture model (mirrors the sim intrinsics)."""
    fx = focal_mm * width / h_aperture_mm
    fy = focal_mm * height / v_aperture_mm
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]], dtype=float)

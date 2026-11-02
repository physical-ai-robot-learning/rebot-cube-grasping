"""FK-INDEPENDENT gripper Z from the FRONT cam (the IBVS-hybrid's Z source; the wrist gives XY, the front
gives Z). The arm's FK/gravity-sag makes "where is the gripper?" wrong, but a FIXED calibrated camera that
SEES the gripper measures it directly. A single point can't give 3D, so we use the one extra fact we have:
after wrist-IBVS the gripper xy ≈ the cube xy (known from the same front cam, ray-plane). Intersect the
front-cam gripper RAY with the vertical line through the cube xy -> gripper z -> gap = gripper_z - plane_z.

Gripper pixel = FRAME DIFFERENCING (the gripper is the only thing that MOVES between control steps; the cube
and background are static) -> no new detector. Restrict to a box around the FK-predicted gripper pixel
(a sagged-but-useful prior) to reject other motion.

Geometry convention mirrors perception.estimate_red_cube_world (ROS optical: +Z fwd, +X right, +Y down;
d_cam = K^-1 [u,v,1], d_world = R @ d_cam, point = C + t d).  CPU self-test at the bottom (no Isaac/robot):
    python deploy/front_gripper_z.py
"""
from __future__ import annotations
import numpy as np


def _ray(uv, K, T_cam2world):
    """Front-cam pixel (u,v) -> world ray (origin C, unit dir d)."""
    Kinv = np.linalg.inv(np.asarray(K, float))
    d_cam = Kinv @ np.array([uv[0], uv[1], 1.0])
    T = np.asarray(T_cam2world, float)
    C = T[:3, 3]
    d = T[:3, :3] @ d_cam
    d = d / (np.linalg.norm(d) + 1e-12)
    return C, d


def gripper_z_from_ray(uv_gripper, cube_xy, K, T_cam2world, plane_z):
    """Front-cam gripper pixel + known cube xy (gripper is ~above it after IBVS) -> (gripper_z, gap, t).
    t* minimizes |C_xy + t d_xy - cube_xy|^2 (the ray's closest approach to the cube's vertical line);
    gripper_z = C_z + t* d_z; gap = gripper_z - plane_z (TCP height above the cube surface)."""
    C, d = _ray(uv_gripper, K, T_cam2world)
    dxy2 = float(d[0] ** 2 + d[1] ** 2)
    if dxy2 < 1e-9:                                   # ray nearly vertical in image-plane -> xy uninformative
        return None
    t = float((np.asarray(cube_xy, float) - C[:2]) @ d[:2] / dxy2)
    if t <= 0:
        return None
    gripper_z = float(C[2] + t * d[2])
    return gripper_z, gripper_z - float(plane_z), t


def detect_gripper_uv(prev_rgb, cur_rgb, fk_pred_uv=None, box=140, diff_thresh=18, min_px=40):
    """Gripper pixel by FRAME DIFFERENCING (gripper moves, scene static). Returns (u,v) or None.
    fk_pred_uv: project the (sagged) FK gripper into the front cam -> search a `box`-px window around it to
    reject unrelated motion. The gripper TIP = the LOWEST (max-v) cluster of motion (the descending jaws)."""
    import cv2
    a = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    b = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    m = (np.abs(b - a) > diff_thresh)
    if fk_pred_uv is not None:                        # gate motion to a window around the FK prior
        H, W = m.shape
        gx, gy = int(fk_pred_uv[0]), int(fk_pred_uv[1])
        mask = np.zeros_like(m)
        x0, x1 = max(0, gx - box), min(W, gx + box)
        y0, y1 = max(0, gy - box), min(H, gy + box)
        mask[y0:y1, x0:x1] = True
        m = m & mask
    ys, xs = np.nonzero(m)
    if xs.size < min_px:
        return None
    vmax = ys.max()                                   # lowest motion row = the descending gripper tip
    sel = ys >= vmax - 12                              # the bottom ~12px band of the moving gripper
    return float(xs[sel].mean()), float(ys[sel].mean())


def project_world_to_front(p_world, K, T_cam2world):
    """World point -> front-cam pixel (u,v). For the FK prior + the self-test."""
    T = np.asarray(T_cam2world, float)
    R, C = T[:3, :3], T[:3, 3]
    pc = R.T @ (np.asarray(p_world, float) - C)        # world -> cam (ROS optical)
    if pc[2] <= 1e-6:
        return None
    uvw = np.asarray(K, float) @ (pc / pc[2])
    return float(uvw[0]), float(uvw[1])


if __name__ == "__main__":
    # CPU self-test: synthesize a calibrated front cam, place a gripper at a known world point, project it,
    # then recover its z from the ray + cube xy. Confirms the geometry is exact (no Isaac/robot).
    K = np.array([[906.8, 0, 640.0], [0, 906.8, 360.0], [0, 0, 1.0]])
    T = np.array([[-0.9999, -0.0131, -0.0048, 0.0081],     # the deploy calib (deploy/calib.npz)
                  [-0.0040, 0.5994, -0.8004, 1.0086],
                  [0.0134, -0.8003, -0.5994, 0.5377],
                  [0, 0, 0, 1.0]])
    plane_z = 0.019
    print("err over a descent (gripper directly above the cube xy):")
    for cube_xy in ([0.0, 0.45], [0.18, 0.50], [-0.20, 0.50]):
        worst = 0.0
        for gz in np.linspace(plane_z, plane_z + 0.20, 9):
            gp = np.array([cube_xy[0], cube_xy[1], gz])    # gripper exactly above the cube at height gz
            uv = project_world_to_front(gp, K, T)
            out = gripper_z_from_ray(uv, cube_xy, K, T, plane_z)
            worst = max(worst, abs(out[0] - gz))
        print(f"  cube {cube_xy}: worst |recovered_z - true_z| = {worst*1000:.2f} mm")
    # robustness: a 1cm xy mis-alignment (IBVS residual) -> how much z error?
    cube_xy = [0.0, 0.45]; gz = plane_z + 0.10
    uv = project_world_to_front(np.array([cube_xy[0] + 0.01, cube_xy[1], gz]), K, T)  # gripper 1cm off in x
    out = gripper_z_from_ray(uv, cube_xy, K, T, plane_z)
    print(f"  +1cm xy misalign -> z err {abs(out[0]-gz)*1000:.1f} mm (IBVS residual sensitivity)")

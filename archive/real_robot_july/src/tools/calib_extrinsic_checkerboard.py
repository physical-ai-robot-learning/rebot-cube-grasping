#!/usr/bin/env python3
"""ACCURATE front-cam EXTRINSIC from a CHECKERBOARD laid FLAT on the table (replaces the 10-red-cube
calib_extrinsic_known.py). Anchored to the J1 base axis; needs the REAL checkerboard intrinsics first
(deploy/cam_front_intrinsics.npz, approx=False).

WHY THIS BEATS THE CUBES
  calib_extrinsic_known.py fits the pose to 10 RED-BLOB CENTROIDS of cubes on the table. Two problems:
    (1) a red blob centroid is NOT the projection of any single fixed 3D point -- the visible face mix
        (top vs near side) changes across the FOV, so the "target" wanders position-dependently, and
    (2) each cube's (dx,dy) is a separate hand tape-measurement.
  Measured floor of that data: ~12.8px homography residual = ~12.5mm mean / 20mm max WORLD error, and it
  is INVARIANT to the assumed cube z (a single plane can't see z), so "use z=0.038" does not help -- it
  only slides the reported camera height up. A checkerboard gives ~54 SUB-PIXEL corners (each a true
  geometric point) in ONE frame -> the fit residual collapses to <1px (~0.4mm). The dominant error then
  becomes how well the BOARD is anchored to the base frame (see PRECISION below).

ANCHORING (the part that now matters most)
  Lay the board FLAT on the table, roughly aligned to the base x/y axes. Do NOT trust an eyeballed yaw:
  TAPE-MEASURE the (dx,dy) from the J1 axis of >=2 detected corners that are FAR APART (ideally the two
  opposite DIAGONAL corners, ~236mm apart on a 9x6/25mm board). This script fits the board's rigid pose
  (yaw + origin) from those anchors -- so the yaw comes from the long tape BASELINE, not the eye -- then
  places all 54 corners by the PRINTED grid (square size is exact) and solvePnP -> T_cam2world.

  Corner indexing matches cv2.findChessboardCorners == the intrinsic script: local coords are
    objp[:, :2] = mgrid[0:cols, 0:rows].T.reshape(-1,2) * square, z=0.
  Anchor a corner by its (col,row) index (0..cols-1, 0..rows-1); corner (0,0) is the origin corner.
  You must know which physical corner is index (0,0): it is the one findChessboardCorners calls first.
  Easiest: put a tiny mark on that corner and confirm from the saved overlay (drawChessboardCorners
  colors corner 0 red). The two recommended anchors are (0,0) and (cols-1,rows-1).

BOARD THICKNESS: corners sit at z = table_z + board_thickness. This is NOT negligible: because the cube
  plane (z=0.019) differs from the board plane, an IGNORED thickness leaks ~0.9mm world error per mm
  (measured). Include it: --board-thickness-mm (paper on glass ~0.3mm; foam-core ~4-5mm).

PRECISION NEEDED FOR <5mm WORLD (measured on this rig's geometry, ~1.2/0.9 px/mm at the table):
  * board ORIGIN error  -> 1.00 : 1 into world  -> keep <=3-5mm.
  * board YAW error     -> ~6mm per degree at the far workspace edge (0.4m lever) -> keep <=0.5-0.7deg.
    Defining yaw from two corners 236mm apart, each tape-measured to +-2mm -> yaw ~0.5deg -> OK. This is
    why you measure two FAR corners rather than eyeball the board rotation.
  * board THICKNESS ignored -> ~0.9mm per mm -> measure it.
  * corner subpix + intrinsics -> ~0.4mm (negligible).

RUN (after calib_intrinsics_checkerboard.py):
  cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python src/tools/calib_extrinsic_checkerboard.py \
      --cols 9 --rows 6 --square-mm 25 --board-thickness-mm 0.5
"""
import argparse
import os
import re
import numpy as np
import cv2
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rebot_sim2real.rbot_calculator.config import load_train_config


def _resolve_dev(dev):
    s = str(dev)
    if s.isdigit():
        return int(s)
    real = os.path.realpath(s)
    m = re.search(r"(\d+)\s*$", real)
    if not m:
        raise SystemExit(f"[Ecb] cannot resolve camera device '{dev}' -> '{real}'")
    return int(m.group(1))


def _grab(dev, focus, exposure):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # MUST match intrinsic + deploy focus, else effective K differs
    if focus is not None:
        cap.set(cv2.CAP_PROP_FOCUS, float(focus))
    if exposure is not None:                    # C922 auto-exposure blows this rig to white; manual if needed
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    fr = None
    for _ in range(12):
        ok, f = cap.read()
        if ok:
            fr = f
    cap.release()
    return fr


def fit_board_pose(anchors_local, anchors_world_xy):
    """Rigid 2D (yaw + translation, scale FIXED=1 since the grid is metric) local->world from >=2 anchors.
    Least-squares (Umeyama, no scaling) so 3-4 tape points average out per-point tape error."""
    L = np.asarray(anchors_local, float)[:, :2]
    W = np.asarray(anchors_world_xy, float)
    lc, wc = L.mean(0), W.mean(0)
    H = (L - lc).T @ (W - wc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R2 = Vt.T @ D @ U.T                          # 2x2 rotation local->world
    t2 = wc - R2 @ lc
    yaw = np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
    # anchor fit residual (how self-consistent the tape measurements are)
    pred = (R2 @ L.T).T + t2
    res_mm = np.linalg.norm(pred - W, axis=1) * 1000
    return R2, t2, yaw, res_mm


def main():
    ap = argparse.ArgumentParser(description="front-cam checkerboard-on-table extrinsic")
    ap.add_argument("--front-dev", default="/dev/video_front")
    ap.add_argument("--cols", type=int, default=9, help="INNER corners along the LONG side")
    ap.add_argument("--rows", type=int, default=6, help="INNER corners along the SHORT side")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--board-thickness-mm", type=float, default=0.5,
                    help="corner height above the table = table_z + this")
    ap.add_argument("--focus", type=float, default=None, help="lock focus (MATCH deploy --front-focus)")
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure if auto blows to white")
    ap.add_argument("--from-image", default=None, help="detect from a saved frame instead of the camera")
    ap.add_argument("--save-dir", default="deploy/calib_front_imgs")
    ap.add_argument("--out", default="deploy/calib.npz")
    args = ap.parse_args()

    intr = np.load("deploy/cam_front_intrinsics.npz")
    K = intr["K"].astype(float)
    dist = intr["dist"].astype(float).ravel()
    if bool(intr["approx"]) if "approx" in intr.files else True:
        raise SystemExit("[Ecb] intrinsics are APPROXIMATE -> run calib_intrinsics_checkerboard.py first.")

    cfg = load_train_config("configs/train.yaml").env
    base = np.array(cfg.base_pos, float)
    board_z = cfg.table_z + args.board_thickness_mm / 1000.0
    cube_z = cfg.table_z + cfg.cube_size / 2.0

    # local (board-frame) corner coords, SAME ordering as findChessboardCorners / the intrinsic script
    cols, rows, sq = args.cols, args.rows, args.square_mm / 1000.0
    local = np.zeros((cols * rows, 3), np.float64)
    local[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    # --- detect the 54 sub-pixel corners in ONE flat-on-table frame ---
    if args.from_image:
        fr = cv2.imread(args.from_image)
        if fr is None:
            raise SystemExit(f"[Ecb] cannot read {args.from_image}")
    else:
        fr = _grab(_resolve_dev(args.front_dev), args.focus, args.exposure)
        if fr is None:
            raise SystemExit("[Ecb] no frame from camera")
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        raise SystemExit("[Ecb] board not found (whole board flat in view? glare? add light).")
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    img2d = corners.reshape(-1, 2).astype(np.float64)
    os.makedirs(args.save_dir, exist_ok=True)
    ov = fr.copy(); cv2.drawChessboardCorners(ov, (cols, rows), corners, ok)
    cv2.imwrite(os.path.join(args.save_dir, "extrinsic_board.png"), ov)
    print(f"[Ecb] detected {len(img2d)} corners; overlay -> {args.save_dir}/extrinsic_board.png "
          "(corner (0,0) is drawn RED -- confirm it is the corner you tape-measured).")

    # --- ANCHOR: tape-measure >=2 FAR-APART corners' (dx,dy) cm from the J1 axis ---
    print(f"[Ecb] base(J1)={np.round(base,3).tolist()}  +x=robot RIGHT, +y=toward cam.")
    print("[Ecb] enter anchors 'col row dx dy' (corner index, cm from J1). Recommend the two DIAGONAL")
    print(f"[Ecb] corners: (0 0 ..) and ({cols-1} {rows-1} ..). Blank line = done (>=2, farther apart = better).")
    a_local, a_world = [], []
    while True:
        s = input(f"[Ecb] anchor #{len(a_local)+1} 'col row dx dy' (or ENTER=done): ").strip()
        if not s:
            break
        try:
            c, r, dx, dy = s.replace(",", " ").split()[:4]
            c, r = int(c), int(r)
            idx = r * cols + c                    # mgrid[0:cols,0:rows].T.reshape order: flat = row*cols + col
            if not (0 <= c < cols and 0 <= r < rows):
                print(f"   col in 0..{cols-1}, row in 0..{rows-1}"); continue
            a_local.append(local[idx])
            a_world.append([base[0] + float(dx) / 100.0, base[1] + float(dy) / 100.0])
            print(f"   corner({c},{r}) local={np.round(local[idx,:2],3).tolist()} "
                  f"world_xy={np.round(a_world[-1],3).tolist()}")
        except Exception:  # noqa: BLE001
            print("   bad input -- e.g. '0 0 -10 30'")
    if len(a_local) < 2:
        raise SystemExit("[Ecb] need >=2 anchors to fix origin+yaw.")

    R2, t2, yaw, ares = fit_board_pose(a_local, a_world)
    print(f"[Ecb] board yaw vs base x-axis = {yaw:+.2f}deg   anchor fit residual mm = {np.round(ares,1).tolist()}")
    if ares.max() > 4:
        print("[Ecb] ** anchor residual >4mm: a tape point is off, or the wrong corner index -> re-measure.")
    # place all 54 corners in world
    obj3d = np.zeros_like(local)
    obj3d[:, :2] = (R2 @ local[:, :2].T).T + t2
    obj3d[:, 2] = board_z

    # --- solvePnP (fixed real K+dist; 54 points, planar but ITERATIVE=homography-init is stable) ---
    ok, rvec, tvec = cv2.solvePnP(obj3d, img2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    T_wc = np.eye(4); T_wc[:3, :3] = R; T_wc[:3, 3] = tvec.ravel()
    T_cam2world = np.linalg.inv(T_wc)
    proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, dist)
    per = np.linalg.norm(proj.reshape(-1, 2) - img2d, axis=1)
    org = T_cam2world[:3, 3]
    print(f"\n[Ecb] reproj mean={per.mean():.2f}px max={per.max():.2f}px  (target <1px; a lone big one = a")
    print(f"[Ecb]   spurious corner -> check the overlay). cam origin (env)={np.round(org,3).tolist()}")
    print(f"[Ecb]   -> {(org[2]-cfg.table_z)*100:.1f} cm above table (physical ~57cm; note a FLAT board only")
    print(f"[Ecb]   weakly constrains camera HEIGHT -- trust reproj + the cube cross-check below, not height).")

    # cross-check: unproject each corner to the CUBE plane and to the board plane; report world spread
    und = cv2.undistortPoints(img2d.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = (T_cam2world[:3, :3] @ np.hstack([und, np.ones((len(und), 1))]).T).T
    tt = (board_z - org[2]) / rays[:, 2]
    wp = org + tt[:, None] * rays
    cross_mm = np.linalg.norm(wp[:, :2] - obj3d[:, :2], axis=1) * 1000
    print(f"[Ecb] board-plane unproject residual: mean={cross_mm.mean():.2f}mm max={cross_mm.max():.2f}mm")

    # GUARD: coplanar solvePnP has a 2-fold (above/below-plane) ambiguity, and inconsistent hand anchors can
    # force a physically-impossible pose. A LOW reproj does NOT validate it (both branches reproject well).
    # If the camera lands at/below the table, the pose is WRONG (mirror / bad anchors) -> refuse to save.
    if org[2] <= board_z + 0.05:
        raise SystemExit(
            f"[Ecb] ** camera solved BELOW/at the table (z={org[2]:.3f} <= {board_z:.3f}) = a WRONG pose "
            "(coplanar mirror or inconsistent anchors). NOT saving. Fixes: measure >=3 well-spread corners "
            "(kills the 2-anchor yaw ambiguity), re-check which physical corner is (0,0) from the overlay, "
            "and TILT the board or raise it on a known-height block to break the flat-plane ambiguity.")
    np.savez(args.out, K=K, T_cam2world=T_cam2world, dist=dist,
             reproj_px=float(per.mean()), board_yaw_deg=float(yaw),
             board_z=float(board_z), anchor_res_mm=ares)
    print(f"[Ecb] saved {args.out}. Verify with diag_front / debug_deepreach; then place a real cube at a")
    print(f"[Ecb] KNOWN (dx,dy) and confirm perception reports it within ~5mm (cube center z={cube_z:.3f}).")


if __name__ == "__main__":
    main()

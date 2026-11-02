#!/usr/bin/env python3
"""Real front-cam INTRINSIC calibration (true K + Brown-Conrady distortion) from a printed CHECKERBOARD.

Replaces the APPROXIMATE spec-derived deploy/cam_front_intrinsics.npz (fx guessed from the C922 FOV,
principal point pinned to the exact image center, dist=0, approx=True) which is the ROOT CAUSE of the
deploy cube-XY position-dependent bias: the un-modeled focal/distortion error gets absorbed into the
extrinsic solvePnP (~12.6px reproj) and shows up as a monotone-in-x world-XY swing (the manual nudges).

PROCEDURE:
  1. Print a checkerboard on FLAT rigid backing. Default 9x6 INNER corners, 25mm squares (set via CLI).
     ("inner corners" = one fewer than squares each way; a 10x7-square board = 9x6 inner corners.)
  2. Hold it at MANY poses: tilt/roll it, and cover the whole frame — ESPECIALLY the lower-center region
     where the cube sits, and vary depth (hold it near ~1m, the cube's distance). ~15-25 good views.
  3. Calibrate -> writes deploy/cam_front_intrinsics.npz (K, dist, image_size, rms, approx=False).
  4. RE-SOLVE the extrinsic: run src/tools/calib_extrinsic_known.py (it now loads the real K+dist), then
     verify with diag_front / debug_deepreach. Aim: RMS < 0.5px, extrinsic reproj 12.6px -> ~1-2px.

CRITICAL: calibrate at the SAME resolution (1280x720) and FOCUS deploy uses. AF is forced OFF here
(focus breathing changes the focal length -> unstable K). If deploy runs --front-focus X, pass --focus X;
otherwise leave default (AF off, frozen at startup) and use the same in deploy.

MODES:
  interactive (default): press ENTER to grab a frame; corners are detected + the frame kept & saved to
                         --save-dir; type 'done' (>= --min-frames good frames) to calibrate.
  --from-dir DIR       : (re)calibrate from already-captured images (no camera needed).

RUN: cd ~/workspace/rebot_sim2real && \
     .venv-deploy/bin/python src/tools/calib_intrinsics_checkerboard.py --cols 9 --rows 6 --square-mm 25
"""
import argparse
import glob
import os
import re
import numpy as np
import cv2


def _resolve_dev(dev):
    """int / numeric string / /dev/video* (incl. udev symlink) -> V4L2 integer index."""
    s = str(dev)
    if s.isdigit():
        return int(s)
    real = os.path.realpath(s)
    m = re.search(r"(\d+)\s*$", real)
    if not m:
        raise SystemExit(f"[cb] cannot resolve camera device '{dev}' -> '{real}'")
    return int(m.group(1))


def _open_front(dev, focus):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # AF OFF: focus breathing changes the focal length -> unstable K
    if focus is not None:
        cap.set(cv2.CAP_PROP_FOCUS, float(focus))
    for _ in range(10):
        cap.read()
    return cap


def main():
    ap = argparse.ArgumentParser(description="front-cam checkerboard intrinsic calibration")
    ap.add_argument("--front-dev", default="/dev/video_front", help="camera device (udev symlink / index)")
    ap.add_argument("--cols", type=int, default=9, help="INNER corners along the LONG side")
    ap.add_argument("--rows", type=int, default=6, help="INNER corners along the SHORT side")
    ap.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size (mm)")
    ap.add_argument("--focus", type=float, default=None,
                    help="lock focus (MATCH deploy --front-focus); None = AF off, frozen at startup")
    ap.add_argument("--min-frames", type=int, default=12, help="minimum good views before calibrating")
    ap.add_argument("--save-dir", default="deploy/calib_front_imgs", help="where grabbed frames are saved")
    ap.add_argument("--from-dir", default=None, help="(re)calibrate from existing images (no camera)")
    ap.add_argument("--out", default="deploy/cam_front_intrinsics.npz")
    args = ap.parse_args()

    pat = (args.cols, args.rows)                  # (corners_x, corners_y) as findChessboardCorners expects
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= (args.square_mm / 1000.0)             # metric board coords (z=0 plane)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    objpoints, imgpoints = [], []
    img_size = [None]                             # (w,h); set on the first successful detection

    def try_frame(gray):
        ok, corners = cv2.findChessboardCorners(
            gray, pat, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            return False
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        img_size[0] = (gray.shape[1], gray.shape[0])   # (w,h)
        return True

    if args.from_dir:
        files = sorted(glob.glob(os.path.join(args.from_dir, "*.png")) +
                       glob.glob(os.path.join(args.from_dir, "*.jpg")))
        if not files:
            raise SystemExit(f"[cb] no .png/.jpg in {args.from_dir}")
        print(f"[cb] calibrating from {len(files)} images in {args.from_dir}")
        for f in files:
            img = cv2.imread(f)
            if img is None:
                print(f"   unreadable {os.path.basename(f)}"); continue
            good = try_frame(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            print(f"   {'OK ' if good else 'no-corners'} {os.path.basename(f)}  ({len(objpoints)} good)")
    else:
        dev = _resolve_dev(args.front_dev)
        os.makedirs(args.save_dir, exist_ok=True)
        cap = _open_front(dev, args.focus)
        print(f"[cb] board {args.cols}x{args.rows} inner corners @ {args.square_mm}mm; focus={args.focus} (AF off), 1280x720.")
        print("[cb] hold the board at varied tilts + across the frame (esp. LOWER-CENTER + ~1m depth).")
        print("[cb] ENTER = grab a view; 'done' = calibrate.")
        saved = 0
        try:
            while True:
                s = input(f"[cb] #{len(objpoints) + 1}  ENTER=grab / done: ").strip().lower()
                if s in ("done", "d"):
                    break
                ok, fr = cap.read()
                if not ok or fr is None:
                    print("   no frame from camera (device? --front-dev?)"); continue
                good = try_frame(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
                if good:
                    p = os.path.join(args.save_dir, f"cb_{saved:03d}.png"); cv2.imwrite(p, fr); saved += 1
                    print(f"   OK corners found -> {p}   ({len(objpoints)} good so far)")
                else:
                    print("   NO corners (whole board in view? reduce glare / add light / hold flatter)")
        finally:
            cap.release()

    if len(objpoints) < args.min_frames:
        raise SystemExit(f"[cb] only {len(objpoints)} good views; need >= {args.min_frames} (varied poses). "
                         "Re-run (interactive) or add images to --from-dir.")
    print(f"[cb] calibrating from {len(objpoints)} views (img_size={img_size[0]}) ...")
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size[0], None, None)
    w, h = img_size[0]
    print(f"\n[cb] RMS reproj = {rms:.3f} px   (target < 0.5; > ~1 => add varied views / better corners)")
    print(f"[cb] K =\n{np.round(K, 2)}")
    print(f"[cb] dist (k1,k2,p1,p2,k3) = {np.round(dist.ravel(), 5).tolist()}")
    print(f"[cb] principal point ({K[0, 2]:.1f},{K[1, 2]:.1f})  vs image center ({w / 2:.0f},{h / 2:.0f})  "
          f"| fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} (approx spec was 906.8)")
    np.savez(args.out, K=K.astype(float), dist=dist.ravel().astype(float),
             image_size=np.array([w, h]), rms=float(rms), approx=False)
    print(f"[cb] saved {args.out} (approx=False).")
    print("[cb] NEXT: re-run src/tools/calib_extrinsic_known.py (now uses the real K+dist) -> then "
          "diag_front / debug_deepreach. Expect the extrinsic reproj to drop 12.6px -> ~1-2px and the "
          "per-position cube-XY nudges to shrink toward zero.")


if __name__ == "__main__":
    main()

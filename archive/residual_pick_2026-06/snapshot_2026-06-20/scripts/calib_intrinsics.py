"""Calibrate camera INTRINSICS (K + distortion) from a checkerboard, headless. No arm motion.

Hold a checkerboard in front of the FRONT camera and slowly move/tilt it to cover the whole frame +
several angles. The script grabs frames, auto-detects the inner-corner grid (tries a few common
sizes), collects N well-spread views, then runs cv2.calibrateCamera. K/distortion are independent of
the square's physical size, so only the grid (inner corners) matters. Saves deploy/cam_front_intrinsics.npz.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/calib_intrinsics.py --device 2 --n 20 --seconds 75
"""
import argparse, time, os, sys
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=2)              # /dev/video2 = front cam
ap.add_argument("--grids", default="9x6,8x6,7x6,10x7,6x9,6x8")  # candidate inner-corner cols x rows
ap.add_argument("--n", type=int, default=20)                  # target good views
ap.add_argument("--seconds", type=float, default=75)
ap.add_argument("--min-gap", type=float, default=0.6, help="min seconds between accepted views")
ap.add_argument("--out", default="deploy/cam_front_intrinsics.npz")
args = ap.parse_args()

grids = [tuple(int(x) for x in g.lower().split("x")) for g in args.grids.split(",")]
cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp_cache = {}


def objp_for(g):
    if g not in objp_cache:
        o = np.zeros((g[0] * g[1], 3), np.float32)
        o[:, :2] = np.mgrid[0:g[0], 0:g[1]].T.reshape(-1, 2)   # square size = 1 (K is scale-invariant)
        objp_cache[g] = o
    return objp_cache[g]


print(f"[calib] device={args.device}  hold a checkerboard, move/tilt it across the frame. "
      f"target {args.n} views in {args.seconds:.0f}s. detecting grids {grids}", flush=True)
obj_pts, img_pts = [], []
locked = None; t0 = time.time(); last = 0.0; shape = None; frames = 0
while time.time() - t0 < args.seconds and len(img_pts) < args.n:
    ok, frame = cap.read()
    if not ok:
        continue
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); shape = gray.shape[::-1]
    cand = [locked] if locked else grids
    for g in cand:
        found, corners = cv2.findChessboardCorners(
            gray, g, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if found:
            if time.time() - last < args.min_gap:
                break
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            img_pts.append(corners); obj_pts.append(objp_for(g)); locked = g; last = time.time()
            print(f"  view {len(img_pts)}/{args.n}  grid={g[0]}x{g[1]}", flush=True)
            break
cap.release()

if len(img_pts) < 6:
    print(f"[calib] FAIL: only {len(img_pts)} views (need >=6). Got {frames} frames. "
          f"Check the board is fully visible / lit, or pass the right --grids.", flush=True)
    sys.exit(1)

rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, shape, None, None)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez(args.out, K=K, dist=dist, image_size=np.array(shape), rms=rms, grid=np.array(locked))
print(f"[calib] DONE  views={len(img_pts)}  reproj RMS={rms:.3f}px (good < ~0.5)")
print(f"  K=\n{K}\n  dist={dist.ravel()}\n  saved -> {args.out}", flush=True)

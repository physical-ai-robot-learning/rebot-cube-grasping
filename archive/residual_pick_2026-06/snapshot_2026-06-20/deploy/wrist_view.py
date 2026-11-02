"""Grab a frame from a camera, run the red-cube HSV centroid, save raw + annotated PNGs.
CAMERA-READ ONLY (no arm motion) -> safe to run anytime. Used to (1) identify which C922 is the
wrist vs front cam, (2) check the red cube is detected, (3) READ the centroid pixel to later define
the IBVS "grasp-ready target pixel".

    .venv-deploy/bin/python deploy/wrist_view.py --device 0 --tag wrist
    .venv-deploy/bin/python deploy/wrist_view.py --device 2 --tag front
"""
import sys, argparse
import numpy as np, cv2
sys.path.insert(0, "src")
from rebot_sim2real import perception as P

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)        # /dev/video0 = the non-perception C922 (likely wrist)
ap.add_argument("--tag", default="cam")                 # label for the output filenames
ap.add_argument("--warmup", type=int, default=30)       # frames to let auto-exposure settle (C922 is slow)
ap.add_argument("--exposure", type=int, default=0)      # >0 = manual exposure (V4L2 units); 0 = auto
args = ap.parse_args()

cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # 3 = auto (V4L2/UVC); adapt to room lighting
cap.set(cv2.CAP_PROP_AUTO_WB, 1)
if args.exposure > 0:                      # manual override (0 = leave auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
frame = None
for _ in range(args.warmup):
    ok, f = cap.read()
    if ok:
        frame = f
cap.release()
if frame is None:
    print(f"[wrist_view] device {args.device}: NO FRAME (camera busy / wrong index?)"); sys.exit(1)
brightness = float(frame.mean())           # 0..255; <~25 = too dark to detect anything

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mask = P.red_mask(rgb)
c = P._largest_blob_centroid(mask)
h, w = mask.shape
area = int(mask.sum() // 255) if mask.dtype == np.uint8 else int(mask.sum())

ann = frame.copy()
ann[mask > 0] = (0, 255, 0)                              # paint detected red region GREEN for visibility
if c is not None:
    cv2.circle(ann, (int(c[0]), int(c[1])), 12, (255, 0, 0), 3)
    cv2.line(ann, (int(c[0]) - 20, int(c[1])), (int(c[0]) + 20, int(c[1])), (255, 0, 0), 2)
    cv2.line(ann, (int(c[0]), int(c[1]) - 20), (int(c[0]), int(c[1]) + 20), (255, 0, 0), 2)
cv2.circle(ann, (w // 2, h // 2), 6, (0, 0, 255), 2)     # image center (red) for reference

raw_p = f"deploy/view_{args.tag}_raw.png"; ann_p = f"deploy/view_{args.tag}_ann.png"
cv2.imwrite(raw_p, frame); cv2.imwrite(ann_p, ann)
det = f"centroid=({c[0]:.0f},{c[1]:.0f})" if c is not None else "centroid=NONE"
dark = "  ** TOO DARK (turn on the workspace light)" if brightness < 25 else ""
print(f"[wrist_view] device={args.device} tag={args.tag} {w}x{h}  brightness={brightness:.1f}/255{dark}")
print(f"  red_area={area}px  {det}  img_center=({w//2},{h//2})")
print(f"  saved {raw_p} , {ann_p}")

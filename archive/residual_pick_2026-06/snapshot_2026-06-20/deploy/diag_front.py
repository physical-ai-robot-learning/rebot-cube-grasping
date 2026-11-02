"""CAMERA-ONLY diagnostic (NO robot, NO motion): grab the front cam, run the red detector, and
compare the current GLOBAL-centroid (mean of ALL red pixels -> hijacked by scattered spurious red)
against a robust LARGEST-CLUSTER centroid. Prints image coords + the world point each produces, and
flags grazing rays (detection high in frame -> ray ~parallel to table -> far/garbage intersection,
the y=-2.x signature). Saves deploy/diag_front.png annotated.

    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/diag_front.py
"""
from __future__ import annotations
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real import perception as P

FRONT_DEV = 2


def grab(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3); cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    for _ in range(12):
        cap.read()
    ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def cluster_centroid(mask):
    """Largest connected red component centroid via cv2 (deploy has cv2). Robust to scattered red."""
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    if stats[i, cv2.CC_STAT_AREA] < 15:
        return None
    return float(cent[i, 0]), float(cent[i, 1]), int(stats[i, cv2.CC_STAT_AREA])


def world_from_px(u, v, K, T, plane_z):
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R = T[:3, :3]; o = T[:3, 3]
    d = R @ d_cam; d = d / (np.linalg.norm(d) + 1e-12)
    if abs(d[2]) < 1e-6:
        return None, d
    t = (plane_z - o[2]) / d[2]
    if t <= 0:
        return None, d
    return o + t * d, d


def main():
    cfg = load_train_config("configs/train.yaml").env
    cal = np.load("deploy/calib.npz"); K = cal["K"].astype(float); T = cal["T_cam2world"].astype(float)
    plane_z = cfg.table_z + cfg.cube_size / 2.0
    wx, wy = cfg.workspace_x, cfg.workspace_y
    print(f"[diag] workspace_x={wx} workspace_y={wy} plane_z={plane_z:.3f}")

    rgb = grab(FRONT_DEV)
    if rgb is None:
        print("[diag] FRONT cam grab FAILED (busy? unplugged?)"); return
    H, W = rgb.shape[:2]
    m = P.red_mask(rgb.astype(np.float32))
    nred = int(m.sum())
    print(f"[diag] front {W}x{H}  red_pixels={nred}")

    glob = P._largest_blob_centroid(m)           # current method: GLOBAL centroid of all red
    clus = cluster_centroid(m)                    # robust: largest connected component
    bgr = rgb[..., ::-1].copy(); bgr[m > 0] = (0, 255, 0)
    cv2.circle(bgr, (W // 2, H // 2), 6, (0, 0, 255), 2)
    cx_w, cy_w = float(np.mean(wx)), float(np.mean(wy))     # workspace center (policy sweet spot)
    p_best = None
    for tag, c, col in [("GLOBAL", glob, (255, 0, 0)), ("CLUSTER", clus, (0, 165, 255))]:
        if c is None:
            print(f"[diag] {tag:8s}: no detection"); continue
        u, v, area = c
        p, d = world_from_px(u, v, K, T, plane_z)
        graze = abs(d[2]) < 0.20             # ray within ~11.5deg of the table plane -> unreliable
        inws = p is not None and (wx[0] < p[0] < wx[1] and wy[0] < p[1] < wy[1])     # STRICT (training dist)
        vfrac = v / H
        cv2.drawMarker(bgr, (int(u), int(v)), col, cv2.MARKER_CROSS, 36, 3)
        print(f"[diag] {tag:8s}: px=({u:.0f},{v:.0f}) v_frac={vfrac:.2f} area={area} "
              f"ray_dz={d[2]:+.3f}{' GRAZING' if graze else ''} -> world={None if p is None else np.round(p,3).tolist()}"
              f"  {'IN-WORKSPACE' if inws else 'OUT'}")
        if p is not None and not graze:
            p_best = p
    cv2.imwrite("deploy/diag_front.png", bgr)
    print("[diag] saved deploy/diag_front.png (green=red-mask, blue=GLOBAL centroid, orange=CLUSTER, red=img center)")
    if p_best is not None:                                  # placement guide -> nudge toward workspace center
        dx, dy = cx_w - p_best[0], cy_w - p_best[1]
        base_dir = "TOWARD the robot base" if dy < 0 else "AWAY from the robot base (forward)"
        side_dir = "robot's LEFT" if dx < 0 else "robot's RIGHT"
        print(f"[diag] target (workspace center) = [{cx_w:.3f}, {cy_w:.3f}].  cube is OFF by "
              f"dx={dx:+.3f} dy={dy:+.3f} m")
        print(f"[diag]  -> move cube {abs(dy)*100:.0f}cm {base_dir}"
              f"{'' if abs(dx)<0.03 else f' and {abs(dx)*100:.0f}cm to the '+side_dir}, then re-run.")
        if wy[0] < p_best[1] < wy[1] and wx[0] < p_best[0] < wx[1]:
            print("[diag]  *** cube IS in the training workspace -> ready to run_pick. ***")


if __name__ == "__main__":
    main()

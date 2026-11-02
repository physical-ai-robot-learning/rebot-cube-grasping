"""FK-FREE front-cam extrinsic, anchored to the J1 axis (the robot base).

WHY: the old deploy/calib_extrinsic.py solved T_cam2world from FK-derived 3D points (the cube held in
the gripper -> obj3d = io.fk_tcp), so the arm's multi-joint FK error propagated straight into the camera
pose. Physical check exposed it: the top cam is ~57 cm above the table but the old calib placed it at
~41 cm (and the perceived cube sat ~8 cm too far from the base). This recalibrates WITHOUT the arm:
place the cube BY HAND at several positions measured from the J1 axis, capture the front-cam red pixel,
and solvePnP against those KNOWN env-frame positions. Tape-measure error (~cm) << accumulated FK error.

ENV FRAME (matches configs/train.yaml; the policy + perception live here):
  origin = world; robot base (J1 axis) = base_pos. On the table, looking from BEHIND the base toward
  the camera:  +x = robot's RIGHT,  +y = toward the TOP/FRONT camera (the base->cam depth),  z = up.
  So a cube placed dx cm to the robot's RIGHT and dy cm toward the CAMERA from the J1 axis is at
    env = base_pos + (dx/100, dy/100, 0),  z = table_z + cube_size/2  (the cube CENTER height).
  (A cube in the training workspace sits at dy ~ +30..+44 cm, dx ~ -20..+20 cm.)

CAMERA-ONLY — no arm motion (park the arm clear of the cam view). Front cam = /dev/video2. RUN YOURSELF:
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_extrinsic_known.py
Collect >= 6 placements that VARY BOTH dx AND dy (non-collinear) and span the table.
"""
import os, sys
import numpy as np, cv2

sys.path.insert(0, "src")
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real import perception as P

FRONT_DEV = 2
cfg = load_train_config("configs/train.yaml").env
base = np.array(cfg.base_pos, float)
cube_z = cfg.table_z + cfg.cube_size / 2.0
K = (np.load("deploy/cam_front_intrinsics.npz")["K"] if os.path.exists("deploy/cam_front_intrinsics.npz")
     else np.load("deploy/calib.npz")["K"]).astype(float)
dist = np.zeros(5)


def red_pixel():
    cap = cv2.VideoCapture(FRONT_DEV)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3); cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    fr = None
    for _ in range(12):
        ok, f = cap.read()
        if ok:
            fr = f
    cap.release()
    if fr is None:
        return None
    c = P._largest_blob_centroid(P.red_mask(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)))
    return None if c is None else (float(c[0]), float(c[1]), int(c[2]))


def main():
    print(f"[Ek] base(J1 axis)={np.round(base,3).tolist()}  cube_center_z={cube_z:.3f}")
    print("[Ek] +x = robot RIGHT, +y = toward the camera. Enter cube offsets in CM from the J1 axis.")
    obj3d, img2d = [], []
    while True:
        s = input(f"\n[Ek] #{len(obj3d)+1} place cube, type 'dx dy' cm (or 'done'): ").strip()
        if s.lower() in ("done", "d", ""):
            break
        try:
            dx, dy = [float(v) / 100.0 for v in s.replace(",", " ").split()[:2]]
        except Exception:  # noqa: BLE001
            print("   bad input — e.g. '12 35'"); continue
        px = red_pixel()
        if px is None:
            print("   no cube detected (reposition / lighting / arm occluding?)"); continue
        env = [base[0] + dx, base[1] + dy, cube_z]
        obj3d.append(env); img2d.append([px[0], px[1]])
        print(f"   ok #{len(obj3d)}: env={np.round(env,3).tolist()} px=({px[0]:.0f},{px[1]:.0f}) area={px[2]}")

    obj3d = np.array(obj3d, np.float64); img2d = np.array(img2d, np.float64)
    np.savez("deploy/extrinsic_known_raw.npz", obj3d=obj3d, img2d=img2d, K=K)
    if len(obj3d) < 6:
        raise SystemExit(f"[Ek] only {len(obj3d)} points; need >=6 well-spread (vary dx AND dy).")
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj3d, img2d, K, dist, reprojectionError=6.0,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    inl = inl.ravel() if inl is not None else np.arange(len(obj3d))
    if len(inl) >= 6:
        ok, rvec, tvec = cv2.solvePnP(obj3d[inl], img2d[inl], K, dist, rvec, tvec, useExtrinsicGuess=True)
    R, _ = cv2.Rodrigues(rvec)
    T_wc = np.eye(4); T_wc[:3, :3] = R; T_wc[:3, 3] = tvec.ravel()
    T_cam2world = np.linalg.inv(T_wc)
    proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, dist)
    per = np.linalg.norm(proj.reshape(-1, 2) - img2d, axis=1)
    org = T_cam2world[:3, 3]
    print(f"\n[Ek] inliers {len(inl)}/{len(obj3d)}  reproj={per[inl].mean():.1f}px (good < ~3)")
    print(f"[Ek] cam origin (env) = {np.round(org,3).tolist()}")
    print(f"[Ek]   -> {(org[2]-cfg.table_z)*100:.0f} cm above the table  (your physical measurement = ~57 cm)")
    if abs((org[2] - cfg.table_z) - 0.57) > 0.05:
        print("[Ek]   ** still off from 57 cm -> add more / better-spread points, check dx/dy signs & axes")
    np.savez("deploy/calib.npz", K=K, T_cam2world=T_cam2world, dist=dist, reproj_px=float(per[inl].mean()))
    print("[Ek] saved deploy/calib.npz (K + FK-free T_cam2world). Verify: .venv-deploy/bin/python deploy/diag_front.py")


if __name__ == "__main__":
    main()

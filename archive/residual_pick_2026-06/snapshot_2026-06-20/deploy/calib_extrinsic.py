"""Hand-eye (eye-to-hand) extrinsic via the robot: hold the red cube in the gripper, command the arm
to ~12 poses, at each read FK (env-frame cube center) + the front-cam red pixel, then solvePnP -> the
camera->env-frame transform T_cam2world. Saves deploy/calib.npz (K + T_cam2world) used by perception.

Uses the VALIDATED joint map (deploy/joint_map.npz). ** MOVES THE ARM (slow ramps). Cube AFFIXED in
the gripper (grip + tape so it can't slip). Keep a hand on the E-STOP. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_extrinsic.py
"""
import sys, time
import numpy as np, cv2
sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real import perception as P
from robot_io import DamiaoRobotIO

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
DT = 1 / 15.0
cfg = load_train_config("configs/train.yaml").env
rest = np.array(cfg.rest_arm, float)
K = np.load("deploy/cam_front_intrinsics.npz")["K"].astype(float)
dist = np.zeros(5)

# ~18 poses: vary J1(yaw) x J2(height/reach) x J4(tilt/depth) for a non-coplanar 3D spread
POSES = []
for j1 in (-0.30, 0.0, 0.30):
    for j2 in (-1.45, -1.65, -1.85):
        for j4 in (0.40, 0.75):
            q = rest.copy(); q[0] = j1; q[1] = j2; q[3] = j4; POSES.append(q)

io = DamiaoRobotIO(cfg, URDF)
io.connect()
cap = cv2.VideoCapture(2); cap.set(3, 1280); cap.set(4, 720)


def red_pixel():
    for _ in range(8):
        ok, fr = cap.read()
    if not ok:
        return None
    m = P.red_mask(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); c = P._largest_blob_centroid(m)
    return None if c is None else (c[0], c[1])


def ramp_to(q, n=40):
    cur = io.read_proprio().arm_qpos
    for k in range(n + 1):
        io.command(cur + (q - cur) * (k / n), cfg.grip_close_qpos); time.sleep(DT)
    for _ in range(8):
        io.command(q, cfg.grip_close_qpos); time.sleep(DT)


try:
    input("\n[E] Put the red cube in the gripper and AFFIX it (grip + tape). Then ENTER (gripper closes)...")
    for _ in range(15):
        io.command(rest, cfg.grip_close_qpos); time.sleep(DT)
    input("[E] Will move through ~12 poses (slow). Cube must stay visible to the front cam. E-STOP ready. ENTER...")
    obj3d, img2d = [], []
    for i, q in enumerate(POSES):
        ramp_to(q)
        tcp = io.fk_tcp(io.read_proprio().arm_qpos)      # env-frame cube center (held at grasp center)
        px = red_pixel()
        if px is None:
            print(f"   pose {i}: cube NOT detected -> skip"); continue
        obj3d.append(tcp); img2d.append(px)
        print(f"   pose {i}: env3d={np.round(tcp,3)}  pixel=({px[0]:.0f},{px[1]:.0f})")
    ramp_to(rest)                                        # back to rest
    obj3d = np.array(obj3d, np.float64); img2d = np.array(img2d, np.float64)
    np.savez("deploy/extrinsic_raw.npz", obj3d=obj3d, img2d=img2d, K=K)   # for diagnosis
    print(f"\n[E] collected {len(obj3d)} correspondences (raw saved -> deploy/extrinsic_raw.npz)")
    if len(obj3d) < 6:
        raise SystemExit("[E] <6 detections -> reposition camera/cube and retry")
    # RANSAC -> robust to slipped/occluded poses; then refine on inliers
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj3d, img2d, K, dist, reprojectionError=8.0,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
    inl = inliers.ravel() if inliers is not None else np.arange(len(obj3d))
    if len(inl) >= 6:
        ok, rvec, tvec = cv2.solvePnP(obj3d[inl], img2d[inl], K, dist, rvec, tvec, useExtrinsicGuess=True)
    Rwc, _ = cv2.Rodrigues(rvec)                          # world(env)->cam
    T_wc = np.eye(4); T_wc[:3, :3] = Rwc; T_wc[:3, 3] = tvec.ravel()
    T_cam2world = np.linalg.inv(T_wc)
    proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, dist)
    per = np.linalg.norm(proj.reshape(-1, 2) - img2d, axis=1)
    err = float(per[inl].mean())
    print(f"[E] inliers {len(inl)}/{len(obj3d)}  inlier reproj err = {err:.1f}px (good < ~3)")
    print("[E] per-pose reproj (px):", np.round(per, 1))
    print("    (large ones = slipped/occluded poses -> excluded by RANSAC)")
    org = T_cam2world[:3, 3]
    print(f"[E] cam origin in env frame = {np.round(org,3)}")
    sane = (org[2] > 0.1) and (org[1] > -0.6)             # cam must be ABOVE the table, FRONT of the base
    print("[E] sanity:", "OK (cam above table, front side)" if sane else
          "** SUSPECT (cam below table / behind base -> bad solve; cube likely slipped, re-tape & retry)")
    np.savez("deploy/calib.npz", K=K, T_cam2world=T_cam2world, dist=dist, reproj_px=err)
    print("[E] saved -> deploy/calib.npz  (K + T_cam2world)")
finally:
    cap.release(); io.estop()
print("[E] done.")

"""Reconcile the LeRobot joint convention (DEGREES, sit-down zero) with the sim/URDF convention
(RADIANS, used by the trained policy). REST-ANCHORED:

  you HAND-POSE the arm to the sim rest_arm physical pose (gripper hovering ~7cm above the cube,
  tilted ~28deg down), we read LeRobot there = L_rest, and define
      sim_q  = rest_arm + SIGN * deg2rad(lerobot_deg - L_rest)
      cmd_deg = L_rest + SIGN * rad2deg(sim_target - rest_arm)
  (SIGN defaults to +1 per vendor consistency; Phase B verifies it.)

Also reads the gripper open/closed endpoints. Saves deploy/joint_map.npz (loaded by DamiaoRobotIO).

** INTERACTIVE and MOVES THE ARM in Phase B (small, ramped). Keep a hand on the e-stop. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_convention.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from rebot_sim2real.rl.config import load_train_config
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import LEROBOT_ARM, JOINT_DIR

cfg = load_train_config("configs/train.yaml").env
REST = np.array(cfg.rest_arm, float)                       # sim rad (the pre-grasp hover)
GRASP = np.array([0.0, -1.85, -1.00, 0.40, 0.0, 0.0])      # sim rad (at the cube) — descent target for the check
HZ = 15.0; DT = 1.0 / HZ
robot = SeeedB601DMFollower(SeeedB601DMFollowerConfig(port="/dev/ttyACM0", id="follower1", can_adapter="damiao"))


def read():
    o = robot.get_observation()
    return np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float), float(o["gripper.pos"])


robot.connect(calibrate=False)
try:
    # ---------- Phase A: anchor + gripper endpoints (TORQUE OFF, hand-pose) ----------
    robot.disable_torque()
    input("\n[A1] Torque OFF. By HAND, pose the arm so the gripper HOVERS ~7cm above the cube, tilted "
          "~28deg down (match the sim rest_arm). Hold it there, then press ENTER...")
    Ls = []
    for _ in range(10):
        a, _ = read(); Ls.append(a); time.sleep(0.05)
    L_rest = np.mean(Ls, 0)
    print("   L_rest (deg):", np.round(L_rest, 2))
    input("[A2] Hand-CLOSE the gripper fully, then ENTER..."); _, gc = read()
    input("[A2] Hand-OPEN  the gripper fully, then ENTER...");  _, go = read()
    print(f"   gripper: closed={gc:.1f}deg  open={go:.1f}deg")

    SIGN = np.ones(6)
    os.makedirs("deploy", exist_ok=True)
    np.savez("deploy/joint_map.npz", L_rest_deg=L_rest, rest_arm_rad=REST, sign=SIGN,
             grip_closed_deg=gc, grip_open_deg=go, lerobot_arm=np.array(LEROBOT_ARM))
    print("   saved -> deploy/joint_map.npz")

    def cmd_sim(q, grip_deg):
        deg = L_rest + SIGN * np.rad2deg(q - REST)                 # raw motor-deg targets
        act = {f"{j}.pos": float(deg[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
        act["gripper.pos"] = float(grip_deg / JOINT_DIR["gripper"])  # /direction: counteract send_action's *direction
        robot.send_action(act)

    # ---------- Phase B: verify by commanding (TORQUE ON, small ramps) ----------
    robot.configure()                                       # re-enable torque/modes
    input("\n[B1] Torque ON. Will HOLD the rest pose (commands L_rest = no motion). Hand on E-STOP. ENTER...")
    for _ in range(20):
        cmd_sim(REST, go); time.sleep(DT)
    input("[B2] Small LIFT test: shoulder_lift +0.15rad (the arm should rise a little). E-STOP ready. ENTER...")
    q = REST.copy();
    for k in range(16):
        q[1] = REST[1] + 0.15 * (k / 15); cmd_sim(q, go); time.sleep(DT)
    time.sleep(0.4)
    for k in range(16):
        q[1] = REST[1] + 0.15 * (1 - k / 15); cmd_sim(q, go); time.sleep(DT)   # back to rest
    if not input("[B2] Did the arm RISE (not drop/swing)? (y/n): ").strip().lower().startswith("y"):
        raise SystemExit("[calib] shoulder_lift direction wrong -> set SIGN[1]=-1 and rerun (inspect other joints too).")
    input("[B3] Full check: SLOWLY ramp rest->GRASP (gripper should DESCEND toward the cube). E-STOP ready. ENTER...")
    N = 60
    for k in range(N + 1):
        cmd_sim(REST + (GRASP - REST) * (k / N), go); time.sleep(DT)
    time.sleep(0.5); a, _ = read(); print("   at GRASP, LeRobot read (deg):", np.round(a, 1))
    ok = input("[B3] Did the gripper descend to ~the cube? (y/n): ").strip().lower().startswith("y")
    input("[B3] Returning to rest. ENTER...")
    for k in range(N + 1):
        cmd_sim(GRASP + (REST - GRASP) * (k / N), go); time.sleep(DT)
    print("\n[calib] RESULT:", "MAP OK -> deploy/joint_map.npz is good." if ok else
          "MAP OFF -> note which joint moved wrong; flip its SIGN in joint_map and rerun B.")
finally:
    robot.disconnect()                                      # torque off (safe)
print("[calib] done.")

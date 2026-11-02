"""Calibrate the gripper open/closed motor endpoints by COMMANDING (the gripper motor isn't hand-
backdrivable, so the hand-pose method failed -> 0/0). Holds the arm at its current pose, ramps the
gripper to each extreme (gentle FORCE_POS), reads the achieved motor deg, you label open/closed.
Updates deploy/joint_map.npz (keeps the validated arm map).

** MOVES THE GRIPPER. Keep fingers clear. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_gripper.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import LEROBOT_ARM, JOINT_DIR

DT = 1 / 15.0
jm = dict(np.load("deploy/joint_map.npz", allow_pickle=True))   # keep arm map, update gripper
robot = SeeedB601DMFollower(SeeedB601DMFollowerConfig(port="/dev/ttyACM0", id="follower1", can_adapter="damiao"))
robot.connect(calibrate=False)
try:
    robot.configure()                                  # torque on
    o = robot.get_observation()
    arm0 = np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float)   # raw motor deg -> hold

    def cmd(grip_input):
        act = {f"{j}.pos": float(arm0[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
        act["gripper.pos"] = float(grip_input)         # raw send-input; send_action does *(-6)
        robot.send_action(act)

    def ramp_and_read(target_input, label):
        cur = robot.get_observation()["gripper.pos"] / JOINT_DIR["gripper"]   # current input-equiv
        N = 25
        for k in range(N + 1):
            cmd(cur + (target_input - cur) * (k / N)); time.sleep(DT)
        for _ in range(10):
            cmd(target_input); time.sleep(DT)
        pos = robot.get_observation()["gripper.pos"]   # raw motor deg
        print(f"   [{label}] gripper input={target_input:.1f} -> motor pos={pos:.1f}deg")
        return pos

    input("\n[G] Will gently move the gripper to EXTREME A (input 0). Keep fingers clear. ENTER...")
    posA = ramp_and_read(0.0, "A")
    la = input("    Gripper now OPEN or CLOSED? (o/c): ").strip().lower()
    input("[G] Now EXTREME B (input 45). ENTER...")
    posB = ramp_and_read(45.0, "B")
    lb = input("    Gripper now OPEN or CLOSED? (o/c): ").strip().lower()

    open_deg = posA if la.startswith("o") else posB
    closed_deg = posA if la.startswith("c") else posB
    if la[:1] == lb[:1]:
        raise SystemExit("[G] both labels same -> retry (gripper may not have moved / both extremes look alike)")
    jm["grip_open_deg"] = np.array(open_deg); jm["grip_closed_deg"] = np.array(closed_deg)
    np.savez("deploy/joint_map.npz", **jm)
    print(f"\n[G] saved gripper endpoints: closed={closed_deg:.1f}deg  open={open_deg:.1f}deg -> joint_map.npz")
finally:
    robot.disconnect()
print("[G] done.")

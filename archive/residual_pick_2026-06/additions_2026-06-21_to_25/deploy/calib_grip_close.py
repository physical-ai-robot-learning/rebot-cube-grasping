"""Recalibrate the gripper CLOSED endpoint (grip_closed_deg) in joint_map.npz.

The old closed was set to input 0 (~0deg) — but that is NOT the firm close; the jaws still had room
(grip_tau stayed at air ~0.07 even with a cube). Result: commanding grip_close never builds clamping
force. This HOLDS the arm and gently sweeps the gripper toward MORE-closed (negative input -> +motor deg
via JOINT_DIR -6), reading motor deg + torque. The FIRM close = the first deg where torque spikes (jaws
hit each other / a hard stop). Saves that as grip_closed_deg (keeps the arm map + grip_open_deg). If the
torque NEVER rises (deg stuck near 0), the lerobot driver clamps the gripper at its calibrated closed ->
the lerobot gripper calibration itself must be redone (not just joint_map).

** ARM IS HELD (no arm motion); only the gripper moves, gentle FORCE_POS. NO cube, fingers clear.
   E-STOP ready. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_grip_close.py
"""
import sys, time
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import LEROBOT_ARM, JOINT_DIR

DT = 1.0 / 15.0
STRAIN = 0.5          # torque magnitude that marks the firm/hard close (same threshold as is_grasped)
SWEEP = [0, -3, -6, -9, -12, -15, -18, -21, -24]   # raw inputs toward MORE closed (input*-6 = +motor deg)

jm = dict(np.load("deploy/joint_map.npz", allow_pickle=True))
old_closed, old_open = float(jm["grip_closed_deg"]), float(jm["grip_open_deg"])
print(f"[gc] current calib: grip_closed_deg={old_closed:.1f}  grip_open_deg={old_open:.1f}", flush=True)

_c = SeeedB601DMFollowerConfig(port="/dev/ttyACM0", id="follower1", can_adapter="damiao")
# UNLOCK the close direction: the gripper CLOSES at +motor-deg, but the driver's default gripper joint_limit
# max=0 clipped EVERY closing command at 0deg (both reachable extremes 0/-250 were OPEN) -> jaws never closed.
# Raise the max so the negative-input sweep below can actually drive the jaws shut.
_c.joint_limits["gripper"] = (_c.joint_limits["gripper"][0], 90.0)
robot = SeeedB601DMFollower(_c)
robot.connect(calibrate=False)
try:
    robot.configure()                                   # torque on (hold)
    o = robot.get_observation()
    arm0 = np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float)   # raw arm degs -> hold the arm here

    def cmd(grip_input):
        act = {f"{j}.pos": float(arm0[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
        act["gripper.pos"] = float(grip_input)          # raw input; send_action multiplies by JOINT_DIR(-6)
        robot.send_action(act)

    def goto(grip_input, n=15):
        cur = robot.get_observation()["gripper.pos"] / JOINT_DIR["gripper"]   # current input-equiv
        for k in range(n + 1):
            cmd(cur + (grip_input - cur) * (k / n)); time.sleep(DT)
        for _ in range(6):
            cmd(grip_input); time.sleep(DT)
        oo = robot.get_observation()
        return float(oo["gripper.pos"]), abs(float(oo.get("gripper.torque", 0.0)))   # raw motor deg, torque

    input("\n[gc] NO cube, fingers clear. Will sweep the gripper CLOSED until it firmly clamps. ENTER...")
    print("  input  ->  motor_deg   torque")
    closed_deg, prev_deg = None, None
    for gi in SWEEP:
        deg, tau = goto(gi)
        moved = "" if prev_deg is None else f" (Δ{deg - prev_deg:+.0f})"
        flag = "  <-- FIRM CLOSE" if tau > STRAIN else ""
        print(f"   {gi:4.0f}   ->  {deg:8.1f}{moved}   {tau:.2f}{flag}", flush=True)
        if tau > STRAIN:
            closed_deg = deg; break
        prev_deg = deg
    goto(0)                                             # back off to the loose close (safe, not straining)

    if closed_deg is None:
        print("\n[gc] torque NEVER spiked -> the gripper did not reach a firm close in range. If motor_deg also "
              "stayed ~flat, the lerobot driver clamps the gripper at its calibrated closed -> redo the LEROBOT "
              "gripper calibration (not just joint_map). Else widen SWEEP.")
    else:
        jm["grip_closed_deg"] = np.array(closed_deg)
        np.savez("deploy/joint_map.npz", **jm)
        print(f"\n[gc] SAVED grip_closed_deg={closed_deg:.1f}deg (was {old_closed:.1f}; open stays {old_open:.1f}) "
              "-> joint_map.npz. Restart the deploy: Grip close should now drive into the firm close, so a cube "
              "between the jaws builds grip_tau (air~0.07 vs cube~1.86) and grasped flips true.")
finally:
    robot.disconnect()
print("[gc] done (torque off).", flush=True)

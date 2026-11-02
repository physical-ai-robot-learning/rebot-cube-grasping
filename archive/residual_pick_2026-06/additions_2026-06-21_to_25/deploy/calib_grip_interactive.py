"""Interactive GRIPPER-ONLY calibration (the arm map is NOT touched). JOG the gripper by eye to the CLOSED
and OPEN positions you want, mark each, and save grip_closed_deg / grip_open_deg to joint_map.npz.

Why: the gripper CLOSES at +motor-deg, but the driver's default gripper joint-limit (max 0deg) clipped every
closing command -> the jaws never closed. This unlocks the close direction (limit -> +120) and lets you set
the real closed/open endpoints visually (the torque-sweep couldn't auto-find it under the gentle calib force).

** ARM IS HELD (no arm motion); only the gripper moves, gentle FORCE_POS. NO cube, fingers clear, E-STOP ready.
   RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/calib_grip_interactive.py
"""
import sys, time
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import LEROBOT_ARM, JOINT_DIR

DT = 1.0 / 15.0
jm = dict(np.load("deploy/joint_map.npz", allow_pickle=True))
print(f"[gi] current joint_map: grip_closed_deg={float(jm['grip_closed_deg']):.1f}  "
      f"grip_open_deg={float(jm['grip_open_deg']):.1f}", flush=True)

_c = SeeedB601DMFollowerConfig(port="/dev/ttyACM0", id="follower1", can_adapter="damiao")
_c.joint_limits["gripper"] = (-270.0, 120.0)        # UNLOCK the close direction (+deg); default max 0 clipped it
robot = SeeedB601DMFollower(_c)
robot.connect(calibrate=False)
try:
    robot.configure()                               # torque on (hold the arm)
    o = robot.get_observation()
    arm0 = np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float)   # raw arm degs -> hold the arm here
    gi = float(robot.get_observation()["gripper.pos"]) / JOINT_DIR["gripper"]   # current gripper input-equiv

    def send(g, ramp_from=None):
        act = {f"{j}.pos": float(arm0[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
        steps = 1 if ramp_from is None else 10
        for k in range(1, steps + 1):
            g_k = g if ramp_from is None else ramp_from + (g - ramp_from) * (k / steps)
            act["gripper.pos"] = float(g_k); robot.send_action(act); time.sleep(DT)

    def deg_tau():
        oo = robot.get_observation()
        return float(oo["gripper.pos"]), abs(float(oo.get("gripper.torque", 0.0)))

    print("\n[gi] JOG the gripper. Type a delta then ENTER:  NEGATIVE = CLOSE more (e.g. -3),  POSITIVE = OPEN more.")
    print("     'c' = mark current as CLOSED,  'o' = mark current as OPEN,  'q' = save & quit.\n", flush=True)
    closed_deg = open_deg = None
    while True:
        d, t = deg_tau()
        print(f"  gripper motor={d:+7.1f}deg  torque={t:.2f}   [marked: closed={closed_deg} open={open_deg}]", flush=True)
        s = input("  > ").strip().lower()
        if s == "q":
            break
        elif s == "c":
            closed_deg = deg_tau()[0]; print(f"    -> CLOSED marked at {closed_deg:+.1f}deg")
        elif s == "o":
            open_deg = deg_tau()[0]; print(f"    -> OPEN marked at {open_deg:+.1f}deg")
        else:
            try:
                gi = float(np.clip(gi + float(s), -45.0, 40.0))   # input range -> motor +120(close).. -240(open)
            except ValueError:
                print("    ? type a number, or c/o/q"); continue
            send(gi, ramp_from=robot.get_observation()["gripper.pos"] / JOINT_DIR["gripper"])

    if closed_deg is not None and open_deg is not None:
        jm["grip_closed_deg"] = np.array(closed_deg)
        jm["grip_open_deg"] = np.array(open_deg)
        np.savez("deploy/joint_map.npz", **jm)
        print(f"\n[gi] SAVED grip_closed_deg={closed_deg:+.1f}  grip_open_deg={open_deg:+.1f} -> joint_map.npz. "
              "Restart deploy with --grip-force 0.3: Grip close drives to your CLOSED deg and a cube builds grip_tau.")
    else:
        print("\n[gi] NOT saved (mark BOTH 'c' and 'o' first).")
finally:
    robot.disconnect()
print("[gi] done (torque off).", flush=True)

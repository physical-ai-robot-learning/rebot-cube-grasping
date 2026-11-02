"""Gripper-only diagnostic: HOLD the arm, sweep the gripper command closed->open (raw lerobot inputs)
and read the motor deg + TORQUE at each level. Reveals where the gripper STRAINS against its limit
(the "over-open"): torque stays ~0 while it moves freely, then spikes when it hits a hard stop. The
sane OPEN = the largest input with LOW torque. Use it to fix grip_open_deg in joint_map.npz if the
calibrated open (-251 deg) over-drives into the stop.

** ARM IS HELD (no arm motion); only the gripper moves, gentle FORCE_POS. E-stop ready. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/grip_check.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from rebot_sim2real.rl.config import load_train_config
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import LEROBOT_ARM, JOINT_DIR

DT = 1.0 / 15.0
cfg = load_train_config("configs/train.yaml").env
jm = dict(np.load("deploy/joint_map.npz", allow_pickle=True))
print(f"[grip] current calib: grip_closed_deg={float(jm['grip_closed_deg']):.1f}  "
      f"grip_open_deg={float(jm['grip_open_deg']):.1f}  (JOINT_DIR gripper={JOINT_DIR['gripper']})")
robot = SeeedB601DMFollower(SeeedB601DMFollowerConfig(port="/dev/ttyACM0", id="follower1", can_adapter="damiao"))
robot.connect(calibrate=False)
try:
    robot.configure()                                   # torque on (hold)
    o = robot.get_observation()
    arm0 = np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float)   # raw arm degs -> hold the arm here

    def cmd(grip_input):
        act = {f"{j}.pos": float(arm0[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
        act["gripper.pos"] = float(grip_input)          # raw input; send_action multiplies by JOINT_DIR(-6)
        robot.send_action(act)

    def goto(grip_input, n=18):
        cur = robot.get_observation()["gripper.pos"] / JOINT_DIR["gripper"]    # current input-equiv
        for k in range(n + 1):
            cmd(cur + (grip_input - cur) * (k / n)); time.sleep(DT)
        for _ in range(8):
            cmd(grip_input); time.sleep(DT)
        oo = robot.get_observation()
        return oo["gripper.pos"], abs(float(oo.get("gripper.torque", 0.0)))

    input("\n[grip] ARM HELD, gripper will sweep CLOSED->OPEN. Keep fingers clear. E-stop ready. ENTER...")
    print("  input  ->  motor_deg    torque")
    rows = []
    for gi in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:   # 0=closed-ish ... 45=calibrated-open
        deg, tau = goto(gi)
        rows.append((gi, deg, tau))
        flag = "   <-- STRAIN (hard stop)" if tau > 0.5 else ""
        print(f"   {gi:4.0f}   ->  {deg:8.1f}     {tau:.2f}{flag}", flush=True)
    goto(20)                                            # leave it mid-open (safe, not straining)
    free = [r for r in rows if r[2] <= 0.5]
    if free:
        gi_o, deg_o, _ = max(free, key=lambda r: r[0])  # largest non-straining input
        print(f"\n[grip] sane OPEN ~ input {gi_o:.0f} -> motor {deg_o:.1f} deg (largest with torque<=0.5)")
        print(f"[grip] calibrated grip_open_deg={float(jm['grip_open_deg']):.1f}. If that strains above, "
              f"set grip_open_deg={deg_o:.1f} (re-test). closed stays {float(jm['grip_closed_deg']):.1f}.")
    else:
        print("\n[grip] gripper strained at ALL inputs -> mechanical/calibration issue (check the gripper).")
finally:
    robot.disconnect()
print("[grip] done (torque off).")

"""Diagnose the real grasp signal: the sim is_grasped (finger-gap ~0.046) false-fires on air-close.
Measure the gripper GAP and TORQUE for (1) closing on AIR vs (2) closing on the CUBE, so we can set a
real is_grasped threshold (torque is usually the clean discriminator). Holds the arm; only the gripper
moves (gentle FORCE_POS). RUN YOURSELF:
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/diag_grasp.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from rebot_sim2real.rl.config import load_train_config
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from robot_io import DamiaoRobotIO, LEROBOT_ARM, JOINT_DIR

cfg = load_train_config("configs/train.yaml").env
io = DamiaoRobotIO(cfg, "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf")
io.connect()


def hold_and_grip(grip_qpos, n=25):
    """Hold the arm at its current pose, ramp the gripper to grip_qpos (sim m)."""
    arm = io.read_proprio().arm_qpos
    for _ in range(n):
        io.command(arm, grip_qpos); time.sleep(1 / 15)


def read_grip():
    o = io.robot.get_observation()
    deg = o["gripper.pos"]; tau = o.get("gripper.torque", 0.0)
    qpos = io._grip_deg_to_qpos(deg); gap = 2 * qpos
    return deg, gap, tau


try:
    input("\n[diag] EMPTY gripper. ENTER -> close on AIR...")
    hold_and_grip(cfg.grip_close_qpos)
    da, ga, ta = read_grip(); print(f"   AIR  : motor={da:.1f}deg  gap(sim)={ga:.3f}  torque={ta:.2f}")
    input("[diag] Open + place the CUBE between the fingers. ENTER -> open...")
    hold_and_grip(cfg.grip_open_qpos)
    input("[diag] Cube positioned between fingers? ENTER -> close on CUBE...")
    hold_and_grip(cfg.grip_close_qpos)
    dc, gc, tc = read_grip(); print(f"   CUBE : motor={dc:.1f}deg  gap(sim)={gc:.3f}  torque={tc:.2f}")
    print("\n[diag] is_grasped should fire for CUBE not AIR. Discriminator:")
    print(f"   gap:    air={ga:.3f} vs cube={gc:.3f}   (sim threshold grip_grasp_gap={cfg.grip_grasp_gap}+/-{cfg.grip_gap_tol})")
    print(f"   torque: air={ta:.2f} vs cube={tc:.2f}   (torque is usually the clean signal)")
finally:
    io.estop()
print("[diag] done.")

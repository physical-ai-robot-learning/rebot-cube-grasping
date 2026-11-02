"""ARM system-ID (deploy/hardware): measure the real robot so the sim DR is set from MEASURED values
instead of guesses. Targets the gaps seen on deploy:
  1. LOOP/ACTUATION LATENCY   -> grounds dr_action_latency_max (the bobbing was a latency signature)
  2. PER-JOINT STEP RESPONSE  -> dead-time, rise-time, STEADY-STATE TRACKING ERROR, overshoot.
     (deploy showed cmd-vs-measured tcp gap trk~0.06 = real motors softer/laggier than sim) ->
     grounds dr_gain_scale / sim drive stiffness.
  3. GRAVITY-HOLD TORQUE       -> magnitude per joint -> grounds torque_obs_scale IF obs_torque is re-enabled.

Small steps from REST, one joint at a time, others HELD. ** MOVES THE ARM. E-stop ready. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/sysid_arm.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "deploy")
from rebot_sim2real.rl.config import load_train_config
from robot_io import DamiaoRobotIO

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
DT = 1.0 / 15.0
STEP_RAD = 0.10           # ~5.7deg test step (small + safe)
HOLD_S = 1.6              # observe window per step
PROBE_JOINTS = (0, 1, 2)  # J1 (yaw) + J2/J3 (reach joints that under-tracked in deploy)


def main():
    cfg = load_train_config("configs/train.yaml").env
    rest = np.array(cfg.rest_arm, float)
    lo, hi = np.array(cfg.arm_limit_lo, float), np.array(cfg.arm_limit_hi, float)
    io = DamiaoRobotIO(cfg, URDF); io.connect()
    try:
        input("\n[sysid] arm -> REST, then small one-joint steps (others held). E-stop ready. ENTER...")
        io.move_to(rest, cfg.grip_open_qpos, duration_s=3.0); time.sleep(0.5)

        # 1) LOOP READ LATENCY (read_proprio round-trip over CAN)
        ts = []
        for _ in range(40):
            t0 = time.time(); io.read_proprio(); ts.append((time.time() - t0) * 1000)
        ts = np.array(ts)
        print(f"\n[sysid] read_proprio: mean={ts.mean():.1f}ms p95={np.percentile(ts,95):.1f}ms "
              f"(policy loop dt={DT*1000:.0f}ms)")

        # 2) GRAVITY-HOLD TORQUE at rest
        pr = io.read_proprio()
        tau = np.abs(np.asarray(pr.arm_torque, float))
        print(f"[sysid] rest |arm_torque| (Nm) per joint: {np.round(tau,3).tolist()}")
        print(f"[sysid]   max|arm_torque|={tau.max():.3f}  grip_torque={pr.grip_torque:.3f}")

        # 3) PER-JOINT STEP RESPONSE
        print(f"\n[sysid] step response (Δ={STEP_RAD:.2f} rad / {np.degrees(STEP_RAD):.1f}°):")
        deads, t63s = [], []
        for j in PROBE_JOINTS:
            base = io.read_proprio().arm_qpos.copy()
            target = base.copy()
            target[j] = float(np.clip(base[j] + STEP_RAD, lo[j] + 0.05, hi[j] - 0.05))
            applied = target[j] - base[j]
            if abs(applied) < 1e-3:
                print(f"[sysid] J{j+1}: skipped (at limit)"); continue
            traj = []; t_cmd = time.time()
            io.command(target, cfg.grip_open_qpos)
            for _ in range(int(HOLD_S / DT)):
                t0 = time.time()
                q = io.read_proprio().arm_qpos[j]
                traj.append((time.time() - t_cmd, q))
                io.command(target, cfg.grip_open_qpos)              # hold the target
                time.sleep(max(0.0, DT - (time.time() - t0)))
            traj = np.array(traj); tt, qq = traj[:, 0], traj[:, 1]
            d = qq - base[j]                                        # signed displacement
            prog = d / applied                                     # fraction of commanded step (1.0 = perfect)
            reached = d[-1]; sse = applied - reached               # steady-state tracking error (rad)
            mv = np.where(prog > 0.1)[0]; dead = tt[mv[0]] if len(mv) else float("nan")
            r6 = np.where(prog >= 0.63)[0]; t63 = tt[r6[0]] if len(r6) else float("nan")
            overshoot = max(float(prog.max()) - 1.0, 0.0) * abs(applied)
            deads.append(dead); t63s.append(t63)
            print(f"  J{j+1}: reached={reached:+.3f}/{applied:+.3f} rad  sse={np.degrees(sse):+.2f}° "
                  f"({prog[-1]*100:.0f}% of cmd)  dead={dead*1000:.0f}ms  t63={t63*1000:.0f}ms  "
                  f"overshoot={np.degrees(overshoot):.1f}°")
            io.move_to(base, cfg.grip_open_qpos, duration_s=1.5); time.sleep(0.3)

        # SUGGESTIONS -> DR grounding
        dead_steps = np.nanmax(deads) / DT if deads else float("nan")
        print("\n[sysid] DR grounding hints:")
        print(f"  - dr_action_latency_max ~ ceil(max_dead/dt) = ceil({np.nanmax(deads)*1000:.0f}ms/{DT*1000:.0f}ms) "
              f"= {int(np.ceil(dead_steps)) if dead_steps==dead_steps else '?'} step(s).")
        print( "  - tracking: large sse / slow t63 => real motor softer & laggier than sim. Lower sim arm kp "
               "or WIDEN dr_gain_scale so the trained policy is robust to it (asymmetric DR).")
        print(f"  - torque_obs_scale: if re-enabling obs_torque, pick scale so max|arm_torque|*scale ~ O(1) "
              f"(here ~{1.0/max(tau.max(),1e-6):.3f}).")
        io.move_to(rest, cfg.grip_open_qpos, duration_s=2.0)
    finally:
        io.estop()
    print("[sysid] done (torque off).")


if __name__ == "__main__":
    main()

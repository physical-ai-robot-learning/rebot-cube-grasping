"""Motor latency / tracking probe for the reBot B601-DM (deploy venv).

Answers "how do I CHECK motor latency, and what's the achievable control rate?" for the Damiao
arm (which the lerobot driver commands in POS_VEL mode with a HARDCODED vlim=32 rad/s -> the motor
SNAPS to each setpoint; smoothness is set entirely by the SETPOINT STREAM we feed + the loop rate).

Modes:
  comm   (SAFE, no motion)  : time read_proprio() and a hold-command (re-send the MEASURED pose ->
                              zero net motion). Reports per-call ms + the implied max control rate.
  step   (MOTION, needs --go): command a small step on ONE joint, poll pos vs time -> motor step
                              response (time-to-63% = servo time-constant, settle time, overshoot).
  track  (MOTION, needs --go): command a slow sinusoid on ONE joint at --rate, log commanded vs
                              measured each tick -> phase lag (ms) by cross-correlation = the
                              operational tracking latency that makes a moving-target descend lag.

Examples:
  .venv-deploy/bin/python src/tools/measure_latency.py --mode comm --n 300
  .venv-deploy/bin/python src/tools/measure_latency.py --mode step  --joint 0 --deg 5  --go
  .venv-deploy/bin/python src/tools/measure_latency.py --mode track --joint 0 --amp 8 --period 4 --rate 30 --go
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rebot_sim2real.rbot_calculator.config import load_train_config
from rebot_sim2real.deploy.io.robot_io import DamiaoRobotIO, JointSlew

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
JN = ["J1", "J2", "J3", "J4", "J5", "J6"]


def _stats(xs_ms):
    a = np.asarray(xs_ms, float)
    return (f"mean={a.mean():6.2f}  median={np.median(a):6.2f}  p95={np.percentile(a,95):6.2f}  "
            f"max={a.max():6.2f}  min={a.min():6.2f}  (ms, n={len(a)})")


def comm(io, n):
    """Read-only-ish: time read_proprio + a HOLD command (re-send measured pose = no motion)."""
    rd, wr = [], []
    pr = io.read_proprio()
    for _ in range(n):
        t = time.perf_counter(); pr = io.read_proprio(); rd.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); io.command(pr.arm_qpos, io.cfg.grip_open_qpos); wr.append((time.perf_counter() - t) * 1e3)
    tot = np.asarray(rd) + np.asarray(wr)
    print(f"\nread_proprio : {_stats(rd)}")
    print(f"command(hold): {_stats(wr)}")
    print(f"read+write   : {_stats(tot)}")
    print(f"\n=> a control tick needs read+write+compute. read+write median={np.median(tot):.1f}ms "
          f"-> bus-limited max rate ~ {1000.0/np.percentile(tot,95):.0f} Hz (p95). The loop sleeps the "
          f"rest up to 1/--rate, so you can raise --rate until read+write+IK ~ the period.")


def step(io, j, deg, go, dur=2.5):
    pr = io.read_proprio(); q0 = pr.arm_qpos.copy()
    tgt = q0.copy(); tgt[j] = q0[j] + np.deg2rad(deg)
    print(f"\nSTEP {JN[j]}: {np.rad2deg(q0[j]):+.2f} -> {np.rad2deg(tgt[j]):+.2f} deg (Δ{deg:+.1f}).")
    if not go:
        print("DRY: pass --go to actually command the motion."); return
    ts, ys = [], []
    t0 = time.perf_counter(); io.command(tgt, io.cfg.grip_open_qpos)
    while time.perf_counter() - t0 < dur:
        ts.append(time.perf_counter() - t0); ys.append(io.read_proprio().arm_qpos[j])
    ts, ys = np.asarray(ts), np.rad2deg(np.asarray(ys))
    y0, yf = ys[0], np.median(ys[-20:]); rng = yf - y0
    def t_at(frac):
        thr = y0 + frac * rng
        idx = np.where((ys - y0) * np.sign(rng) >= abs(frac * rng))[0]
        return ts[idx[0]] * 1e3 if len(idx) else float("nan")
    overshoot = (ys.max() - yf) if rng > 0 else (yf - ys.min())
    print(f"  reached {yf:+.2f} deg (target {np.rad2deg(tgt[j]):+.2f}, err {yf-np.rad2deg(tgt[j]):+.2f}).")
    print(f"  t@63% = {t_at(0.63):.0f} ms (servo time-constant)   t@95% = {t_at(0.95):.0f} ms   "
          f"overshoot = {overshoot:+.2f} deg   samples={len(ts)} (poll {len(ts)/dur:.0f} Hz)")
    io.command(q0, io.cfg.grip_open_qpos)   # return to start


def track(io, j, amp, period, rate, go, cycles=3, shaper=False, amax=12.0, vmax=6.5):
    pr = io.read_proprio(); q0 = pr.arm_qpos.copy()
    dt = 1.0 / rate; n = int(cycles * period / dt)
    sh = JointSlew(q0, dt, v_max=vmax, a_max=amax) if shaper else None
    tag = f"SHAPER(amax={amax},vmax={vmax})" if shaper else ("GLIDE vlim=%.0f" % io.arm_vlim if io.arm_vlim else "raw POS_VEL vlim=32")
    print(f"\nTRACK {JN[j]}: sin amp={amp}deg period={period}s @ {rate}Hz, {cycles} cycles ({n} ticks) [{tag}].")
    if not go:
        print("DRY: pass --go to actually command the motion."); return
    cmd, meas = [], []
    t0 = time.perf_counter()
    for k in range(n):
        t = k * dt
        q = q0.copy(); q[j] = q0[j] + np.deg2rad(amp) * np.sin(2 * np.pi * t / period)
        q_out = sh.step(q, dt) if sh is not None else q      # optionally route through the SW shaper (its lag)
        io.command(q_out, io.cfg.grip_open_qpos)
        meas.append(io.read_proprio().arm_qpos[j]); cmd.append(q[j])   # cmd = INTENDED target -> lag = total D
        time.sleep(max(0.0, (k + 1) * dt - (time.perf_counter() - t0)))
    cmd = np.rad2deg(np.asarray(cmd)) - np.rad2deg(q0[j]); meas = np.rad2deg(np.asarray(meas)) - np.rad2deg(q0[j])
    # phase lag = argmax cross-correlation of meas vs cmd, in samples -> ms
    c = np.correlate(meas - meas.mean(), cmd - cmd.mean(), mode="full")
    lag = (np.argmax(c) - (len(cmd) - 1)) * dt
    amp_ratio = (meas.max() - meas.min()) / max(1e-6, (cmd.max() - cmd.min()))
    print(f"  phase lag = {lag*1e3:+.0f} ms ({lag/period*360:+.1f} deg of the cycle)   "
          f"amplitude tracking = {amp_ratio*100:.0f}% of commanded")
    io.command(q0, io.cfg.grip_open_qpos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deploy_env.yaml")
    ap.add_argument("--mode", choices=["comm", "step", "track"], default="comm")
    ap.add_argument("--joint", type=int, default=0, help="0..5 = J1..J6")
    ap.add_argument("--deg", type=float, default=5.0, help="step size (step mode)")
    ap.add_argument("--amp", type=float, default=8.0, help="sine amplitude deg (track mode)")
    ap.add_argument("--period", type=float, default=4.0, help="sine period s (track mode)")
    ap.add_argument("--rate", type=float, default=30.0, help="command rate Hz (track mode)")
    ap.add_argument("--n", type=int, default=300, help="iters (comm mode)")
    ap.add_argument("--arm-vlim", type=float, default=0.0, dest="arm_vlim",
                    help="POS_VEL glide vlim rad/s (0 = driver default 32). Set e.g. 8 to compare MOTOR-GLIDE vs "
                         "the software shaper: raw POS_VEL should already be low-lag (the motor is fast).")
    ap.add_argument("--shaper", action="store_true",
                    help="track: route commands through the JointSlew SOFTWARE shaper (--amax/--vmax) -> measures "
                         "the SHAPER's added phase lag (the current descend/look-at path). Off = raw POS_VEL/glide.")
    ap.add_argument("--amax", type=float, default=12.0, help="--shaper accel limit rad/s^2 (the smoothness/lag dial)")
    ap.add_argument("--vmax", type=float, default=6.5, help="--shaper velocity cap rad/s")
    ap.add_argument("--go", action="store_true", help="ACTUALLY command motion (step/track)")
    args = ap.parse_args()

    cfg = load_train_config(args.config).env
    cfg.num_envs = 1
    io = DamiaoRobotIO(cfg, URDF, arm_vlim_rad_s=(args.arm_vlim if args.arm_vlim > 0 else None)); io.connect()
    print(f"[latency] connected. mode={args.mode}  arm={'GLIDE vlim=%.0f' % args.arm_vlim if args.arm_vlim > 0 else 'raw POS_VEL vlim=32'}  shaper={args.shaper}")
    try:
        if args.mode == "comm":
            comm(io, args.n)
        elif args.mode == "step":
            step(io, args.joint, args.deg, args.go)
        else:
            track(io, args.joint, args.amp, args.period, args.rate, args.go,
                  shaper=args.shaper, amax=args.amax, vmax=args.vmax)
    finally:
        io.estop()
        print("[latency] done (torque off).")


if __name__ == "__main__":
    main()

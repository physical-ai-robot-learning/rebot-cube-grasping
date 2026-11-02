"""FROM-SCRATCH gripper calibration — ONE guided, safe flow that supersedes the four scattered scripts
(calib_gripper / calib_grip_interactive / calib_grip_close / grip_check). Holds the arm, moves ONLY the
gripper via gentle FORCE_POS, and calibrates in three stages:

  ① OPEN   endpoint  (NO cube): sweep toward open, sane OPEN = largest travel with LOW torque (before the
                                open-side hard stop). -> grip_open_deg
  ② CLOSED endpoint  (NO cube): sweep toward closed until the torque SPIKES (jaws self-contact / hard stop).
                                -> grip_closed_deg   (the firm close, so a cube later builds grip_tau)
  ③ FORCE  RATIO     (WITH cube): hold the jaws CLOSED on the cube and RAISE the FORCE_POS ratio until the
                                stall torque hits the gentle target. A rigid cube makes torque a vertical
                                wall in position, so grip force = the current cap (ratio), not depth. Prints
                                the recommended --grip-force (ratio) + --grip-torque-target.

Stages ① ② write grip_closed_deg / grip_open_deg to deploy/joint_map.npz (the arm map is untouched).
Stage ③ writes nothing; it PRINTS the recommended --grip-force / --grip-torque-target.

⚠️ SAFETY (this gripper BROKE TWICE from over-torque):
  * FORCE_POS ratio is HARD-CLAMPED to GRIP_FORCE_RATIO_MAX (0.18); default 0.10.
  * Every sweep aborts + backs off if torque exceeds ABORT_TAU (1.2, well below the ~1.86 crush).
  * Gentle ramps, small steps, always backs off to a safe non-straining pose after each stage + on exit.
  * Fingers clear for ① ②; a firm cube in the jaws for ③ (never sweep force onto nothing).

** MOVES THE GRIPPER (arm HELD). E-STOP READY. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python src/tools/calib_grip.py
    (options: --port /dev/ttyACM0  --ratio 0.10  --skip-open --skip-closed --skip-force)
"""
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
from rebot_sim2real.deploy.io.robot_io import LEROBOT_ARM, JOINT_DIR, GRIP_FORCE_RATIO_MAX, GRIP_TORQUE_THRESH

DT = 1.0 / 15.0
STRAIN = float(GRIP_TORQUE_THRESH)   # torque marking a firm close / hard stop (== is_grasped threshold, 0.5)
ABORT_TAU = 1.2                      # global over-torque abort (<< the ~1.86 that broke the gripper)
HOLD_TARGET = 0.8                    # the gentle grip we want stage ③ to confirm reachable (servo target)
JMAP = "deploy/joint_map.npz"

# raw gripper INPUT sweeps (send_action multiplies input by JOINT_DIR["gripper"] = -6 -> motor deg).
# negative input -> +motor deg -> CLOSING; positive input -> OPENING.
CLOSE_SWEEP = [0, -1, -2, -3, -4, -5, -6, -7, -8]             # toward firm self-close (stage ②); 1-step ≈6 motor-deg (self-contact wall is STEEP: torque 0.13→1.4 within ~18 motor-deg, so a coarse -3 step overshot STRAIN→ABORT)
CLOSE_ON_CUBE = -6                                            # input commanding the jaws CLOSED past a ~38mm cube (stall)
# OPEN sweep (stage ①) is built from --open-max / --open-step so the calibrated open isn't driven too wide (A3
# also opens to --open-max before you insert the cube). Stage ③ is a RATIO sweep: a rigid cube makes torque a
# near-vertical WALL in position (grip_tau jumps ~0.1->1.3 within ~3 motor-deg), so grip force is set by the
# FORCE_POS current cap = ratio, NOT by depth. Sweep the ratio (jaws held closed) for the gentle hold torque.
# (raw input: + = OPENING, - = CLOSING.)
RATIO_SWEEP = [r for r in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12] if r <= GRIP_FORCE_RATIO_MAX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="follower1")
    ap.add_argument("--ratio", type=float, default=0.10,
                    help="FORCE_POS torque/current ratio used for ALL calibration motion. HARD-CLAMPED to "
                         f"GRIP_FORCE_RATIO_MAX={GRIP_FORCE_RATIO_MAX}. Stage ③ characterizes force AT this ratio; "
                         "if it can't build a hold, re-run with a slightly higher --ratio (0.10->0.12->0.15).")
    ap.add_argument("--open-max", type=int, default=30,
                    help="OPEN sweep ceiling (raw input; higher = wider gap). Default 30 (was 45 = over-open). A1 "
                         "sweeps 0..open-max, and A3 opens to this before you insert the cube. WIDEN if the cube "
                         "won't fit between the jaws; LOWER if they open too wide.")
    ap.add_argument("--open-step", type=int, default=5, help="OPEN sweep step (raw input).")
    ap.add_argument("--close-max", type=int, default=14,
                    help="CLOSED sweep depth (|raw input|; higher = closes further). Default 14 (was a FIXED 8; a "
                         "shifted gripper zero — e.g. after motor/reassembly work — can put the firm-close CONTACT "
                         "past input -8 so torque 'never spikes'. The ABORT_TAU guard stops the instant it contacts, "
                         "so a deeper sweep is safe). If it STILL never spikes at a large --close-max, the jaws are "
                         "mechanically not meeting (damaged/loose finger) -> physical repair, not calibration.")
    ap.add_argument("--close-step", type=int, default=1, help="CLOSED sweep step (raw input).")
    ap.add_argument("--close-on-cube", type=int, default=None,
                    help="stage ③: input commanding the jaws closed on the cube. Default = the stage-② firm-close "
                         "input (else -6). Set deeper (e.g. -10) if a shifted zero means -6 doesn't reach the cube.")
    ap.add_argument("--skip-open", action="store_true")
    ap.add_argument("--skip-closed", action="store_true")
    ap.add_argument("--skip-force", action="store_true")
    args = ap.parse_args()

    ratio = float(np.clip(args.ratio, 0.0, GRIP_FORCE_RATIO_MAX))
    if args.ratio > GRIP_FORCE_RATIO_MAX:
        print(f"[calib] ⚠️ --ratio {args.ratio:.2f} > safety cap {GRIP_FORCE_RATIO_MAX:.2f} -> CLAMPED to {ratio:.2f}")

    jm = dict(np.load(JMAP, allow_pickle=True))
    old_c = float(jm.get("grip_closed_deg", np.array(0.0)))
    old_o = float(jm.get("grip_open_deg", np.array(-90.0)))
    print(f"[calib] joint_map now: grip_closed_deg={old_c:.1f}  grip_open_deg={old_o:.1f}  ratio={ratio:.2f}")

    # Build the follower at OUR ratio + UNLOCK the close direction (driver default gripper max=0 clips closing).
    cfg = SeeedB601DMFollowerConfig(port=args.port, id=args.id, can_adapter="damiao",
                                    force_pos_torque_ration=ratio)
    cfg.joint_limits["gripper"] = (cfg.joint_limits["gripper"][0], 120.0)
    robot = SeeedB601DMFollower(cfg)
    robot.connect(calibrate=False)
    try:
        robot.configure()                                        # torque on -> holds the arm
        o = robot.get_observation()
        arm0 = np.array([o[f"{j}.pos"] for j in LEROBOT_ARM], float)   # freeze the arm at the present pose

        def cmd(grip_input):
            act = {f"{j}.pos": float(arm0[i] / JOINT_DIR[j]) for i, j in enumerate(LEROBOT_ARM)}
            act["gripper.pos"] = float(grip_input)               # raw input; send_action does *JOINT_DIR(-6)
            robot.send_action(act)

        def read(n=4):
            deg = tau = 0.0
            for _ in range(n):
                oo = robot.get_observation(); time.sleep(DT)
                deg += float(oo["gripper.pos"]); tau += abs(float(oo.get("gripper.torque", 0.0)))
            return deg / n, tau / n                              # averaged motor deg, torque (noise down)

        def goto(grip_input, n=15, guard=True):
            """Ramp gently to grip_input, dwell, and return (motor_deg, torque). When guard, ABORTS (raises)
            if torque crosses ABORT_TAU mid-ramp so a bad CLOSING sweep can't crush the gripper. Release /
            back-off moves pass guard=False: torque is momentarily high while relieving off a loaded cube, so
            guarding there would spuriously abort the very move that RELIEVES it."""
            cur = robot.get_observation()["gripper.pos"] / JOINT_DIR["gripper"]
            for k in range(n + 1):
                cmd(cur + (grip_input - cur) * (k / n)); time.sleep(DT)
                if guard:
                    t = abs(float(robot.get_observation().get("gripper.torque", 0.0)))
                    if t > ABORT_TAU:
                        cmd(0.0)                                 # instant relief toward loose
                        raise RuntimeError(f"ABORT: torque {t:.2f} > {ABORT_TAU} while ramping to input {grip_input}")
            for _ in range(4):
                cmd(grip_input); time.sleep(DT)
            return read()

        def sweep(inputs, stop_when, label):
            """Run a sweep, printing each level; stop at the first row where stop_when(tau) is True.
            Returns (motor_deg, tau) at the stop row, or None if never triggered. Backs off to input 0."""
            print(f"  input  ->  motor_deg   torque   [{label}]")
            hit, prev = None, None
            for gi in inputs:
                deg, tau = goto(gi)
                dd = "" if prev is None else f" (Δ{deg - prev:+.0f})"
                flag = "  <--" if stop_when(tau) else ""
                print(f"   {gi:4.0f}   ->  {deg:8.1f}{dd}   {tau:.2f}{flag}", flush=True)
                if stop_when(tau):
                    hit = (gi, deg, tau); break
                prev = deg
            return hit

        # ─────────────────────────── ① OPEN endpoint ───────────────────────────
        if not args.skip_open:
            input(f"\n[①/OPEN] NO cube, fingers clear. Sweep toward OPEN (0..{args.open_max}) for the non-straining "
                  "max. ENTER (s=skip)…")
            print("  input  ->  motor_deg   torque   [OPEN sweep]")
            rows = []
            for gi in range(0, args.open_max + 1, args.open_step):
                deg, tau = goto(gi)
                rows.append((gi, deg, tau))
                print(f"   {gi:4.0f}   ->  {deg:8.1f}     {tau:.2f}{'   <-- STRAIN' if tau > STRAIN else ''}", flush=True)
            goto(min(20, args.open_max), guard=False)            # leave mid-open (safe; relieving move)
            free = [r for r in rows if r[2] <= STRAIN]
            if free:
                gi_o, deg_o, _ = max(free, key=lambda r: r[0])   # largest non-straining input
                jm["grip_open_deg"] = np.array(deg_o)
                np.savez(JMAP, **jm)
                print(f"[①] SAVED grip_open_deg={deg_o:.1f} (input {gi_o:.0f}; was {old_o:.1f}) -> {JMAP}")
            else:
                print("[①] gripper strained at ALL inputs -> mechanical/lerobot-calib issue; open NOT saved.")

        # ─────────────────────────── ② CLOSED endpoint ─────────────────────────
        close_input = None                                       # stage-② firm-close INPUT (feeds stage ③ default)
        close_sweep = [0] + [-k for k in range(args.close_step, args.close_max + 1, args.close_step)]  # [0,-1,..,-close_max]
        if not args.skip_closed:
            input(f"\n[②/CLOSED] NO cube, fingers clear. Sweep toward CLOSED (0..-{args.close_max}) to the FIRM "
                  "self-close. ENTER (s=skip)…")
            hit = sweep(close_sweep, lambda t: t > STRAIN, "CLOSED sweep; stop at firm close")
            goto(0, guard=False)                                 # back off to the loose close (relieving move)
            if hit is not None:
                close_input = hit[0]
                jm["grip_closed_deg"] = np.array(hit[1])
                np.savez(JMAP, **jm)
                print(f"[②] SAVED grip_closed_deg={hit[1]:.1f} (input {hit[0]:.0f}, torque {hit[2]:.2f}; was {old_c:.1f}) -> {JMAP}")
            else:
                print(f"[②] torque NEVER spiked through input -{args.close_max} (motor deg kept climbing without a "
                      "self-contact wall) -> the jaws are NOT meeting. Retry with a larger --close-max; if it STILL "
                      "never spikes, the gripper is mechanically not closing (damaged/loose finger or the LEROBOT "
                      "gripper calibration is off) -> physical check, not just joint_map.")

        # ─────────────────────────── ③ FORCE profile (RATIO sweep) ──────────────
        # Rigid cube -> torque is a vertical wall in position, so grip force is the FORCE_POS current cap (ratio).
        # Hold the jaws CLOSED on the cube and raise the ratio until the stall torque reaches the gentle target;
        # that ratio is the --grip-force to deploy (the servo then just holds against that cap).
        if not args.skip_force:
            # close-on-cube input: CLI override > stage-② firm-close input > default -6 (a shifted zero can put the
            # cube contact past -6, so prefer the measured close input).
            close_on_cube = args.close_on_cube if args.close_on_cube is not None else (close_input if close_input is not None else CLOSE_ON_CUBE)
            goto(args.open_max, guard=False)                     # open so the cube can be inserted
            input(f"\n[③/FORCE] Jaws OPEN (input {args.open_max}). Place a FIRM CUBE between them, then it closes "
                  f"(input {close_on_cube}) and RAISES the FORCE_POS ratio until grip_tau reaches the gentle target "
                  f"{HOLD_TARGET}. ENTER (s=skip)…")
            robot.config.force_pos_torque_ration = RATIO_SWEEP[0]   # gentlest ratio for the first contact
            goto(close_on_cube, guard=True)                      # close onto the cube (gentle)
            print(f"  ratio  ->  grip_tau   [stall on cube; jaws commanded closed @ input {close_on_cube}]")
            prof, rec_ratio, aborted = [], None, False
            for r in RATIO_SWEEP:
                robot.config.force_pos_torque_ration = r         # live ratio (config is a plain dataclass)
                tau = 0.0
                try:
                    for _ in range(8):                           # re-command closed + let torque settle at this ratio
                        cmd(close_on_cube); time.sleep(DT)
                        t = abs(float(robot.get_observation().get("gripper.torque", 0.0)))
                        if t > ABORT_TAU:
                            cmd(0.0); raise RuntimeError(f"ABORT: torque {t:.2f} > {ABORT_TAU} at ratio {r:.2f}")
                    _, tau = read()
                except RuntimeError as e:
                    print(f"   {e}  -> ceiling hit; stop."); aborted = True; break
                prof.append((r, tau))
                mark = "  <-- HOLD_TARGET" if tau >= HOLD_TARGET else ("  <-- GRASP" if tau > STRAIN else "")
                print(f"   {r:.2f}   ->  {tau:.2f}{mark}", flush=True)
                if tau >= HOLD_TARGET:                            # gentle firm clamp reached -> enough
                    rec_ratio = r; break
            robot.config.force_pos_torque_ration = RATIO_SWEEP[0]   # restore gentle before releasing
            goto(0, guard=False)                                 # release the cube
            if rec_ratio is not None:
                print(f"\n[③] ratio {rec_ratio:.2f} clamps this cube at grip_tau≈{prof[-1][1]:.2f} (≈target {HOLD_TARGET}).\n"
                      f"     RECOMMEND:  --grip-force {rec_ratio:.2f}  --grip-torque-target {HOLD_TARGET:.1f}\n"
                      "     (force = RATIO: deploy commands the jaws closed and this current cap holds the clamp; the "
                      "servo just holds against it. Rigid cube -> force is NOT set by depth.)")
            elif aborted and prof:
                print(f"\n[③] hit the {ABORT_TAU} ceiling; last clean = ratio {prof[-1][0]:.2f} -> grip_tau≈{prof[-1][1]:.2f}. "
                      f"Deploy with --grip-force {prof[-1][0]:.2f} (a touch below the ceiling).")
            elif prof:
                pr, pt = max(prof, key=lambda x: x[1])
                print(f"\n[③] swept to ratio {pr:.2f} -> peak grip_tau≈{pt:.2f} (< target {HOLD_TARGET}). Extend RATIO_SWEEP "
                      f"upward (hard cap {GRIP_FORCE_RATIO_MAX}) or accept --grip-force {pr:.2f}.")
            else:
                print("\n[③] no reading -> check the cube placement / that the jaws actually closed on it.")

        print(f"\n[calib] DONE. joint_map: grip_closed_deg={float(jm.get('grip_closed_deg', old_c)):.1f}  "
              f"grip_open_deg={float(jm.get('grip_open_deg', old_o)):.1f}")
    except Exception as e:
        print(f"\n[calib] STOPPED: {e}")
    finally:
        try:
            cmd(0.0); time.sleep(0.2)     # leave the gripper at the loose close (not straining)
        except Exception:
            pass
        robot.disconnect()
    print("[calib] disconnected (torque off).", flush=True)


if __name__ == "__main__":
    main()

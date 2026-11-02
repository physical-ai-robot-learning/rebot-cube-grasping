#!/usr/bin/env python3
"""FK/IK Z-verification DRIVER — orchestrates the semi-auto table-touch sweep + fits the reach-dependent Z error.

Symptom: the AUTO-GRASP / RUN descend stops at the wrong HEIGHT depending on how far the cube is from the arm
base — cube CLOSE => over-descends (too low), cube FAR => under-descends (too high).

Why it's FK: the gate fires on z_gap_io = tcp_io.z - plane_z where tcp_io.z = robot_io Pinocchio FK + a CONSTANT
FK_Z_CORR. A constant can't cancel a reach-DEPENDENT FK z error (J2/J3 zero-offset or link mismatch): right at one
distance, high when closer, low when farther = exactly the symptom.

HOW THIS TOOL WORKS (drives the deploy's semi-auto 'FK-touch' button; deploy must be up on :8899):
  For each distance you position the arm at, this sends /cmd?a=fktouch -> the ARM auto-descends straight down until
  the J2/J3 torque rises (table contact), records z_gap_io at that FIXED true height, and lifts back. z_gap_io read
  at the table should be CONSTANT across distances; its SLOPE vs radial IS the reach-dependent FK bias -> the fix.
  (The physical touch + torque-contact + lift are owned by the deploy; this only triggers + reads /state.)

USAGE (deploy up):  python3 src/tools/fk_verify_z.py   (stdlib only)
  1. JOG / Follow the arm to a HOVER over a CLEAR table spot (NO cube under the gripper — a consistent bare-table
     reference) at some distance.
  2. Press ENTER -> the arm auto-touches, records, lifts.  Repeat at >=3 distances: NEAR, MID, FAR.
  3. 'f' = fit now, 'q' = quit (auto-fits). The fit prints the exact --z-reach-slope / --z-reach-ref to relaunch with.
⚠ The arm MOVES on ENTER (descends to the table). Clear space below + WATCH it; press Stop on the dashboard to halt.
"""
import json, math, time, urllib.request, urllib.parse

BASE = "http://localhost:8899"
BASE_XY = (0.0, 0.06)          # cfg.base_pos xy = the J1 (base yaw) axis; radial = |tcp_io_xy - base_xy|
PLANE_Z = 0.019                # table_z(0) + cube_size(0.038)/2 -> cube CENTER world z (z_gap reference)
TRUE_TOUCH_GAP = -PLANE_Z      # tip at table_z(0) => ideal z_gap_io at touch (slope vs radial is what matters)


def _state():
    try:
        return json.load(urllib.request.urlopen(BASE + "/state", timeout=2))
    except Exception as e:
        print(f"  [/state unavailable: {e}]  (deploy up on :8899?)"); return None


def _cmd(action):
    try:
        urllib.request.urlopen(BASE + "/cmd?a=" + urllib.parse.quote(action), timeout=2).read()
        return True
    except Exception as e:
        print(f"  [/cmd {action} failed: {e}]"); return False


def _samples():
    d = _state()
    return (d.get("gates", {}).get("fk_touch_samples", []) if d else []) or []


def drive_touch(timeout=25.0):
    """Trigger one auto table-touch and wait for the deploy to record it (or report a no-contact abort)."""
    d = _state()
    if d is None:
        return False
    n0 = len(d.get("gates", {}).get("fk_touch_samples", []))
    if d.get("gates", {}).get("fk_touch") is not None:
        print("  a touch is already in progress — wait / press Stop."); return False
    if not _cmd("fktouch"):
        return False
    t_end = time.time() + timeout
    started = False
    while time.time() < t_end:
        time.sleep(0.25)
        g = (_state() or {}).get("gates", {})
        ph, samps = g.get("fk_touch"), g.get("fk_touch_samples", []) or []
        if ph is not None:
            started = True
        if len(samps) > n0:                                   # a new touch sample landed
            r, io, fb = samps[-1]
            print(f"  ✓ TOUCH #{len(samps)}  radial={r:.3f}m  z_gap_io={io:+.4f}  (err vs ideal {io - TRUE_TOUCH_GAP:+.4f})")
            return True
        if started and ph is None:                            # returned to idle WITHOUT a new sample = no-contact abort
            print("  ✗ no contact recorded (deploy aborted). lower --fk-touch-dtau / raise --fk-touch-max, "
                  "or check the arm actually reached the table."); return False
    print("  ✗ timed out waiting for the touch to complete."); return False


def _linfit(xs, ys):
    # OLS (least squares) fit  y = intercept + slope·x :  slope = (n·Σxy − Σx·Σy)/(n·Σxx − (Σx)²).
    n = len(xs); sum_x = sum(xs); sum_y = sum(ys); sum_xx = sum(x * x for x in xs); sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None, sum(ys) / n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope, (sum_y - slope * sum_x) / n


def fit():
    s = _samples()
    if len(s) < 2:
        print(f"  need >=2 touch samples at DIFFERENT distances (have {len(s)})"); return
    xs = [r[0] for r in s]; io = [r[1] for r in s]
    print("\n  ── TABLE-TOUCH FIT (z_gap_io read with the tip on the table) ──")
    for r in s:
        print(f"     radial={r[0]:.3f}m  z_gap_io={r[1]:+.4f}  (fk_bias={r[2]:+.4f})")
    b, a = _linfit(xs, io)
    if b is None:
        print("     all samples at one radial — vary the distance for a slope"); return
    ref = sum(xs) / len(xs)
    print(f"     z_gap_io_touch ≈ {a:+.4f} {b:+.4f}*radial     (slope = {b * 1000:+.1f} mm per metre of reach)")
    if abs(b) > 0.03:
        print("     ⚠ REACH-DEPENDENT confirmed: a CONSTANT FK_Z_CORR can NOT fix this (that's the bug).")
    else:
        print(f"     slope small ({b * 1000:+.1f} mm/m) — reach dependence minor; residual is ~constant (tune --grasp-zgap-io).")
    print("\n  ── FIX: add these flags to debug_deepreach.py (corrects tcp_io.z by -slope*(radial-ref)) ──")
    print(f"       --z-reach-slope {b:+.4f}  --z-reach-ref {ref:.3f}")
    print(f"     Then re-run this sweep: z_gap_io at touch should be ~CONSTANT ({a + b * ref:+.4f}) at every distance.")
    print("     The remaining constant is absorbed by --grasp-zgap-io (you already tune it), so re-check depth after.")


def main():
    print(__doc__)
    d = _state()
    if d is None:
        print("deploy not reachable on :8899 — relaunch debug_deepreach.py first."); return
    have = len(_samples())
    if have:
        if input(f"{have} touch sample(s) already on the deploy. CLEAR for a fresh sweep? [Y/n] ").strip().lower() != "n":
            _cmd("fktouch_clear"); time.sleep(0.3); print("  cleared.")
    print("\n  ENTER=auto-touch this spot · 'f'=fit · 'q'=quit(+fit)\n")
    while True:
        try:
            c = input("fk-verify> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); fit(); break
        if c in ("q", "quit"):
            fit(); break
        elif c in ("f", "fit"):
            fit()
        else:
            drive_touch()


if __name__ == "__main__":
    main()

"""Dashboard command routing.

Split out the command if/elif chain (1259-1556) of the old debug_deepreach.py `main()`. Receives the (action, extra)
given by dash.poll_cmd(), changes ControlState/modes, and applies it to the robot/dashboard.

⚠️ CNN dropped: removed align's IBVS state reset (ibvs_*) (was only for the sd4 CNN bearing servo).
⚠️ sync dropped: removed capture's inline front-estimate branch (the perception thread always exists → request_capture).
Style: control flow/logic uses explicit if/else (avoid ternaries); simple value-guard ternaries are allowed.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..kinematics import _project_wrist_unvn
from .constants import LIFT_Z_MAX

# Auto-mode/stop commands — these cancel any in-progress one-shot sequence (autograsp/fk_touch) and lift.
_MODE_OR_HALT = ("stop", "reset", "zero", "run", "follow", "approach", "align",
                 "close", "open", "lift", "final", "abort")


def handle(loop, cmd) -> bool:
    """Handles one dashboard command. (input: loop=ControlLoop, cmd=(action, extra) → output: True ends the control loop (abort))

    As in the monolith, ticks that received a command skip mode execution (caller: if command: handle() else: step_modes()).
    """
    action, extra = cmd
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    cfg = loop.cfg
    pr = tick.pr

    # ---- Shared guard: manual mode/stop commands cancel one-shot sequences ----
    if action in _MODE_OR_HALT:
        st.autograsp.phase = None                     # cancel an in-progress AUTO-GRASP sequence
        st.fk_touch.phase = None                      # cancel an in-progress FK-touch descent
        if action != "lift":                          # 'lift' STACKs one step; anything else ends lift
            st.lift.active = False

    if action == "abort":
        dash.update(msg="ABORT -> estop")
        facade.estop()
        return True                                   # end the control loop

    if action == "capture":
        # front camera cube capture. The perception thread owns the front frame → on request the cube is picked up in the next snapshot.
        st.yaw_latch.weight = 0.0                     # new cube → discard stale yaw latch
        if st.track.enabled:                          # --track: Capture re-anchors live front tracking (fresh)
            st.track.anchor = None
            dash.update(msg="CAPTURE (track) -> re-anchoring the live front track")
        else:
            loop.perception.request_capture()
            dash.update(msg="capture requested (async perception) — cube appears once the thread detects it")

    elif action == "reset":                           # return to REST hover (gripper open); clear modes/latches
        st.run_active = False
        st.grip.servoing = False
        st.align.active = False
        st.follow.active = False
        st.approach.active = False
        st.yaw_latch.weight = 0.0
        st.grip.command = cfg.grip_open_qpos
        facade.run_scripted("initial_liftup", cfg.grip_open_qpos)   # scripted lift-clear → REST hover, open
        base.reset()                                  # clear both reach + descend latches
        facade.reset_shaper(loop.rest_np)             # re-sync the shaper to the reached REST pose
        dash.update(msg="RESET ↩ -> REST hover (gripper open; reach + descend latches + modes clear)")

    elif action == "zero":                            # park at URDF home ZERO (q=0); gripper open
        st.run_active = False
        st.grip.servoing = False
        st.align.active = False
        st.follow.active = False
        st.approach.active = False
        st.yaw_latch.weight = 0.0
        st.grip.command = cfg.grip_open_qpos
        facade.run_scripted("comeback_to_zero", cfg.grip_open_qpos)   # scripted: open + lift-clear → ZERO
        base.reset()
        facade.reset_shaper(np.zeros(6))
        dash.update(msg="ZERO ⊙ -> URDF home q=0 (gripper open; latch + modes clear)")

    elif action == "ovquery":                         # dashboard text box → set the open-vocab query live
        _handle_ovquery(loop, extra)

    elif action == "jog":
        st.grip.servoing = False
        st.lift.active = False                        # jog ends lift-hold; after grasp, jog moves the arm (grip_cmd kept)
        q = pr.arm_qpos.copy()
        q[extra["j"]] += extra["d"] * 0.02
        facade.send_arm(q, st.grip.command, loop.loop_dt, snap=True)   # CRISP full nudge; after close, a lift test

    elif action == "close":                           # grasp: closed-loop force-servo up to grip_torque_target
        st.run_active = False                         # SAFETY: close stops reach/align/follow/approach
        st.align.active = False
        st.follow.active = False
        st.approach.active = False
        st.grip.command = st.grip.mid                 # start just open; the servo closes gently
        st.grip.servoing = True                       # idle loop force-servos → auto-stop at target
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
        dash.update(msg=f"grip CLOSE -> force-servo to grip_tau≈{st.grip.torque_target:.2f} "
                        "(creeps + auto-stops; watch grip_tau / grasped)")

    elif action == "open":
        st.grip.command = cfg.grip_open_qpos
        st.grip.servoing = False
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
        dash.update(msg=f"grip OPEN -> {st.grip.command:.3f}")

    elif action in ("grip_tight", "grip_loose"):      # live dial: nudge the force-servo target torque (not raw position)
        if action == "grip_tight":
            delta = 0.1                               # tighter
        else:
            delta = -0.1                              # looser
        st.grip.torque_target = float(np.clip(st.grip.torque_target + delta, 0.3, 1.2))   # crush ~1.86 << 1.2
        st.grip.servoing = True                       # re-arm the servo so it re-converges to the new target
        if delta > 0:
            _dir = "TIGHTER"
        else:
            _dir = "SOFTER"
        dash.update(msg=f"grip {_dir} -> target grip_tau≈{st.grip.torque_target:.2f} "
                        "(servo re-converging; crush was ~1.86, capped 1.2). Raise only if the cube SLIPS.")

    elif action == "lift":                            # TCP +z rise (cam-down), keep grip = LIFT
        # GENTLE LIFT mode: ramp the TCP-z target at --lift-rate/tick up to +lift_step, then hold. Re-click=STACK (up-only, capped at LIFT_Z_MAX).
        st.run_active = False
        st.follow.active = False
        st.align.active = False
        st.approach.active = False
        st.grip.servoing = False                      # so the force-servo idle branch cannot re-freeze the arm (grip stays CLOSED via grip_cmd)
        cur_z = float(base.fk_world(tick.arm_q)[0][2])
        if st.lift.active and st.lift.target_z is not None:
            base_z = st.lift.target_z                 # STACK on top of the previous target
        else:
            base_z = cur_z
        st.lift.target_z = float(min(max(cur_z, base_z + float(loop.dcfg.lift_step)), LIFT_Z_MAX))
        if not st.lift.active:
            st.lift.cur_z = cur_z                     # the ramp starts here
            st.lift.xy = (float(tick.tcp_w[0]), float(tick.tcp_w[1]))   # rise straight up above the current spot
        st.lift.active = True
        dash.update(msg=f"LIFT ↑ ramp to z={st.lift.target_z:.3f} @ {loop.dcfg.lift_rate*100:.1f}cm/tick "
                        f"(gentle; cap {LIFT_Z_MAX:.2f}; grip={st.grip.command:.3f}); click to keep rising")

    elif action == "align":                           # start LATCHED auto-center-then-J6 mode (jaws-OPEN, before grasp)
        st.run_active = False
        st.grip.servoing = False
        st.follow.active = False
        st.approach.active = False
        st.align.active = True
        st.align.dwell = 0
        dash.update(msg=f"ALIGN ▶ auto-center J4 -> cube top-edge to vn≈{st.align.vn_target:+.2f} "
                        "(bottom-40%), then roll J6 to null seg yaw. 'Align flip' if J6 wrong way.")

    elif action == "align_flip":                      # invert the rig-dependent J6-roll direction (without restart)
        st.align.jaw_sign = -st.align.jaw_sign
        dash.update(msg=f"ALIGN J6 sign -> {st.align.jaw_sign:+.0f}. Press 'Align' again to re-run with this dir.")

    elif action == "autograsp":                       # ONE-SHOT: ⊙centered xy COMMIT → straight descent → Grip
        if not st.follow.active or tick.cube_world is None or not st.follow.auto_center_latched:
            dash.update(msg="AUTO-GRASP needs FOLLOW + ⊙CENTERED first (the centered pose IS the optimal grasp xy; "
                            "it gets FROZEN, then a straight-down descent). Wait for ⊙CENTERED.")
        else:
            st.autograsp.j6_target = None             # fresh: re-latch yaw + re-freeze xy + reset dwell
            st.autograsp.committed_xy = None
            st.autograsp.ready_t0 = None
            st.autograsp.align_yaw_cos = 0.0          # fresh: reset the seg-yaw EMA accumulator (docs/33 conservative gate)
            st.autograsp.align_yaw_sin = 0.0
            st.autograsp.align_yaw_n = 0
            st.autograsp.align_yaw_ts = None
            if loop.dcfg.autograsp_yaw:
                st.follow.auto_center_latched = False  # align RE-centers simultaneously with the J6 roll
                st.autograsp.phase = "align"          # yaw-align FIRST(hover) → descend → grip
                dash.update(msg="AUTO-GRASP ▶ xy COMMITTED (⊙centered) → YAW-ALIGN J6 at hover → straight-down "
                                "descend → grip. Stop/ABORT halts.")
            else:
                st.autograsp.phase = "descend"        # off → descend immediately
                dash.update(msg="AUTO-GRASP ▶ xy COMMITTED (⊙centered) → straight-down descend → grip at grasp "
                                "depth (J6 held). Stop/ABORT halts.")
            st.autograsp.phase_t0 = time.time()

    elif action == "fktouch":                         # SEMI-AUTO TABLE-TOUCH (FK-Z verification)
        if st.fk_touch.phase is not None:
            dash.update(msg="FK-TOUCH already running (Stop to cancel).")
        elif pr is None or getattr(pr, "arm_torque", None) is None:
            dash.update(msg="FK-TOUCH needs proprio torque (arm_torque) — not available.")
        else:
            st.run_active = False
            st.follow.active = False
            st.align.active = False
            st.approach.active = False
            st.grip.servoing = False                  # exclusive mode
            st.autograsp.phase = None
            st.fk_touch.xy = (float(tick.tcp_w[0]), float(tick.tcp_w[1]))   # fix the current tcp xy, straight descent
            st.fk_touch.z_start = float(tick.tcp_w[2])
            st.fk_touch.z_cur = st.fk_touch.z_start
            st.fk_touch.tau_baseline = [float(x) for x in pr.arm_torque]    # hover baseline (reset to steady-descent at ARM_DZ)
            st.fk_touch.armed = False                 # arm contact detection only after descending ARM_DZ
            st.fk_touch.phase = "descend"
            facade.reset_shaper(pr.arm_qpos)
            radial = np.hypot(st.fk_touch.xy[0] - float(cfg.base_pos[0]),
                              st.fk_touch.xy[1] - float(cfg.base_pos[1]))
            dash.update(msg=f"FK-TOUCH ▶ descending to table contact (dtau>{loop.dcfg.fk_touch_dtau} on J2/J3); "
                            f"records z_gap_io + lifts. radial={radial:.3f}m. Stop halts.")

    elif action == "fktouch_clear":                   # FK-Z verification: discard accumulated touch samples (fresh sweep)
        st.fk_touch.samples.clear()
        dash.update(msg="FK-TOUCH samples cleared — fresh sweep. Position + FK-touch at each distance.")

    elif action == "final":                           # descend from align-hover to grasp (deep_reach auto-z), keeping J6 alignment
        if tick.cube_world is not None:
            st.run_active = True
            st.align.active = False
            st.follow.active = False
            st.approach.active = False
            facade.reset_shaper(pr.arm_qpos)          # re-sync the shaper to the aligned pose
            dash.update(msg="FINAL ↓ descend to grasp (deep_reach auto-z; J6 align held). Then Grip close.")
        else:
            dash.update(msg="FINAL ignored: Capture cube first")

    elif action == "run":                             # ONE-SHOT: auto deep_reach every loop until reached
        st.run_active = (tick.cube_world is not None)
        st.align.active = False
        st.follow.active = False
        st.approach.active = False                     # RUN stops align/follow/approach (fresh approach)
        st.yaw_latch.weight = 0.0                       # fresh descent → re-latch the mid-range yaw
        if st.run_active:
            facade.reset_shaper(pr.arm_qpos)           # re-sync the shaper to the measured pose before run
            dash.update(msg="RUN ▶ auto deep_reach until reached (z_gap_io≈goal; Stop/Hold/ABORT halts)")
        else:
            dash.update(msg="RUN ignored: Capture cube first")

    elif action == "follow":                          # HOVER-FOLLOW (opt-in): TCP tracks cube_world from --follow-z above
        st.follow.active = (tick.cube_world is not None)
        st.run_active = False
        st.align.active = False
        st.grip.servoing = False
        st.approach.active = False
        st.follow.target_xy = None                    # fresh: start at hover, descend once stable
        st.follow.front_ref_xy = None
        st.follow.height_above_cube = None
        if st.follow.active:
            facade.reset_shaper(pr.arm_qpos)
            dash.update(msg=f"FOLLOW ▶ track cube; DESCEND to +{loop.dcfg.follow_low*100:.0f}cm when stable, RISE to "
                            f"+{loop.dcfg.follow_z*100:.0f}cm + re-track when it moves. Stop/RUN halts.")
        else:
            dash.update(msg="FOLLOW ignored: Capture cube first")

    elif action == "followlookat":                    # live toggle: GIMBAL look-at ↔ straight-down auto-center
        st.follow.lookat = not st.follow.lookat
        st.follow.lookat_last_q = None                # discard the held gimbal pose on mode switch (fresh aim)
        if st.follow.lookat:
            dash.update(msg="FOLLOW mode -> GIMBAL LOOK-AT (cam tilts to aim at cube, TCP held)")
        else:
            dash.update(msg="FOLLOW mode -> STRAIGHT-DOWN auto-center (TCP moves)")

    elif action == "followcam":                       # live tuning of Follow cube centering
        _d = float(extra.get("d", 1))
        _ax = extra.get("ax")
        if loop.dcfg.follow_auto_center:              # auto-center ON: buttons tune the target (u,v)
            if _ax == "lat":
                st.follow.center_un_target += 0.03 * _d
            else:
                st.follow.center_vn_target += 0.03 * _d
            st.follow.auto_center_latched = False     # re-servo to the new target
            dash.update(msg=f"AUTO-CENTER target -> cube@wrist goal (un {st.follow.center_un_target:+.2f}, "
                            f"vn {st.follow.center_vn_target:+.2f})")
        else:                                         # auto-center OFF: buttons tune the fixed offset directly
            if _ax == "lat":
                st.follow.cam_lat += 0.005 * _d
            else:
                st.follow.cam_fwd += 0.005 * _d
            dash.update(msg=f"FOLLOW cam-offset -> fwd {st.follow.cam_fwd*100:+.1f}cm / lat {st.follow.cam_lat*100:+.1f}cm")

    elif action == "graspz":                          # live tuning of grasp DEPTH (target z_gap_io) (used by AUTO-GRASP + RUN)
        _d = float(extra.get("d", 1))
        loop.grasp_zgap_io = float(np.clip(loop.grasp_zgap_io - 0.005 * _d, -0.06, 0.04))
        dash.update(msg=f"GRASP depth -> goal z_gap_io {loop.grasp_zgap_io:+.3f}m "
                        f"(closes at z_gap_io <= {loop.grasp_zgap_io + 0.012:+.3f}m; lower=deeper). auto-z retargets live.")

    elif action == "approach":                        # LOOK-AT APPROACH (opt-in): continuous descent to hover while the wrist aims at the cube
        st.approach.active = (tick.cube_world is not None)
        st.run_active = False
        st.align.active = False
        st.grip.servoing = False
        st.follow.active = False
        st.approach.target_xy = None                  # fresh: rise to hover, then descend
        st.approach.height = None
        st.approach.risen = False
        st.approach.wrist_lock_xy = None
        if st.approach.active:
            facade.reset_shaper(pr.arm_qpos)
            dash.update(msg=f"APPROACH ▶ smooth look-at descend to +{loop.dcfg.lookat_approach_stop*100:.0f}cm HOVER "
                            "(wrist aims at cube throughout; tracks moves). Then Align/Final/close. Stop/RUN halts.")
        else:
            dash.update(msg="APPROACH ignored: Capture cube first")

    elif action == "stop":
        st.run_active = False
        st.align.active = False
        st.follow.active = False
        st.approach.active = False
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)   # freeze at the current pose
        dash.update(msg="STOP ■ auto/align/follow halted, holding pose")

    elif action == "calib_wrist":                     # WRIST-cam CALIB sweep (redesigned, identifiable)
        _cmd_calib_wrist(loop, extra)

    return False


def _handle_ovquery(loop, extra):
    """Sets the open-vocab query live (shared by front+wrist). If the object changes: during approach, go to REST then re-approach; otherwise re-acquire."""
    st = loop.state
    dash = loop.dash
    qtext = (extra.get("q") or "").strip()
    if not qtext:
        dash.update(msg="OV query empty — type e.g. 'red cube' or 'red cube, red block'")
        return
    if loop.front_ov is None and loop.wrist_ov is None:
        dash.update(msg="OV query ignored: enable --front-openvocab / --wrist-openvocab first")
        return
    ref_ov = loop.front_ov or loop.wrist_ov
    old_q = ref_ov.queries                            # detect a real OBJECT change (normalized comparison)
    n_set = 0
    for ov in (loop.front_ov, loop.wrist_ov):
        if ov is not None and ov.set_queries(qtext):
            n_set += 1
    cams_list = []
    if loop.front_ov is not None:
        cams_list.append("front")
    if loop.wrist_ov is not None:
        cams_list.append("wrist")
    cams = "+".join(cams_list)
    changed = (n_set > 0 and (loop.front_ov or loop.wrist_ov).queries != old_q)
    if changed and st.approach.active:
        # OBJECT changed during approach → LIFT to REST + re-approach the NEW object from above (a low lateral chase to a distant new object is risky)
        loop.return_to_rest(f"OV query → '{qtext}': OBJECT CHANGED → lifted to REST, RE-APPROACHING it.")
        if not st.track.enabled:
            loop.perception.request_capture()         # non-track: re-capture the NEW object (track re-acquires live)
    elif changed:
        # not approaching: re-acquire so a later Approach targets the NEW object (otherwise non-track one-shot keeps the stale cube)
        if st.track.enabled:
            st.track.anchor = None
        else:
            loop.perception.request_capture()
        dash.update(msg=f"OV query → '{qtext}' set on {cams} — object changed, re-acquiring; press Approach.")
    else:
        dash.update(msg=f"OV query → '{qtext}' set on {cams} (re-detect next frame).")


def _cmd_calib_wrist(loop, extra):
    """WRIST-cam calibration sweep (redesigned, identifiable): diagonal J1+J4 sweep at an OFFSET LOOK-AT hover (cam ~15° off vertical),
    APPENDing each pose's observed (u,v) vs cube_world to wrist_calib_data.npz. >=6 positions → fit_wrist_calib.py recovers the physical
    mount/aim[/focal]. BLOCKS ~9s. ⚠️ The arm swings laterally a few cm — do it from a clear hover and watch the first run."""
    st = loop.state
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    cfg = loop.cfg
    dcfg = loop.dcfg
    tick = loop.tick
    cube_world = tick.cube_world
    plane_z = loop.plane_z
    grip_cmd = st.grip.command

    if cube_world is None:
        dash.update(msg="CALIB ignored: Capture the cube first (front cam), at a safe hover")
        return
    st.run_active = False
    st.align.active = False
    cw3 = [float(cube_world[0]), float(cube_world[1]), float(plane_z)]
    cpath = "deploy/wrist_calib_data.npz"
    cols = ["q0", "q1", "q2", "q3", "q4", "q5", "cx", "cy", "cz", "mun", "mvn", "mdepth",
            "sun", "svn", "svis", "wun", "wvn", "wvis", "base_z"]
    prev = None                                       # existing data (only when the schema matches this redesign)
    if os.path.exists(cpath):
        loaded = np.load(cpath, allow_pickle=True)
        if list(loaded["cols"]) == cols:
            prev = loaded["rows"]
        else:
            dash.update(msg="CALIB: existing npz has the OLD schema (non-identifiable) — starting FRESH. "
                            "(rm deploy/wrist_calib_data.npz to silence.)")
            time.sleep(1.5)
    # position index → rotate the lateral offset direction (pose diversity across cube positions)
    if prev is not None:
        pos_idx = len({(round(float(r[6]), 2), round(float(r[7]), 2)) for r in prev})
    else:
        pos_idx = 0
    phi = 2.0 * np.pi * pos_idx / 6.0
    off = (float(dcfg.calib_offset) * np.cos(phi), float(dcfg.calib_offset) * np.sin(phi))
    cube_w = torch.tensor([[cw3[0], cw3[1], cw3[2]]], dtype=torch.float32, device=base.device)
    # OFFSET LOOK-AT hover: TCP lateral offset, ~14cm above, AIM the cam at the cube → tilted from vertical
    target = torch.tensor([[cw3[0] + off[0], cw3[1] + off[1], float(plane_z + 0.14)]],
                          dtype=torch.float32, device=base.device)
    down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=base.device)
    seed = torch.tensor(facade.read_proprio().arm_qpos, dtype=torch.float32, device=base.device).view(1, 6)
    q_hover = base.ik_oriented(seed, target, down, iters=80, look_at=cube_w)[0].cpu().numpy().astype(float)
    facade.send_arm(q_hover, grip_cmd, loop.loop_dt, snap=True)
    dash.update(msg=f"CALIB pos#{pos_idx}: moving to tilted look-at hover (off {off[0]:+.02f},{off[1]:+.02f})…")
    time.sleep(2.0)
    q0 = facade.read_proprio().arm_qpos.copy()
    rows = []
    for t in np.linspace(-1.0, 1.0, 11):              # diagonal J1+J4 sweep (both scaled by t)
        q = q0.copy()
        q[3] = float(np.clip(q0[3] + float(dcfg.calib_j4) * t, float(base.lo[3]), float(base.hi[3])))
        q[0] = float(np.clip(q0[0] + float(dcfg.calib_j1) * t, float(base.lo[0]), float(base.hi[0])))
        facade.send_arm(q, grip_cmd, loop.loop_dt, snap=True)   # SNAP: reach the commanded pose (crawls in shaper mode)
        time.sleep(0.6)                               # settle + fresh perception frame
        prc = facade.read_proprio()
        snap_perc, _, _ = loop.perception.snapshot()
        mun, mvn, mdepth, _vis = _project_wrist_unvn(base, prc.arm_qpos, cw3, calib=False)   # RAW(ref)
        rows.append([*[float(x) for x in prc.arm_qpos], *cw3, mun, mvn, mdepth,
                     float(snap_perc["sun"]), float(snap_perc["svn"]), float(snap_perc["svis"]),
                     float(snap_perc["wun"]), float(snap_perc["wvn"]), float(snap_perc["wvis"]), float(dcfg.base_z)])
        dash.update(msg=f"CALIB pos#{pos_idx}: sweep {len(rows)}/11 "
                        f"(wrist_vis {snap_perc['wvis']:.0f} wvn {snap_perc['wvn']:+.2f})")
    facade.send_arm(q_hover, grip_cmd, loop.loop_dt, snap=True)   # return to hover
    new = np.array(rows, dtype=np.float32)
    if prev is not None:
        new = np.concatenate([prev, new], axis=0)
    np.savez(cpath, rows=new, cols=np.array(cols))
    npos = len({(round(float(r[6]), 2), round(float(r[7]), 2)) for r in new})
    nvis = int(sum(1 for r in new if r[17] > 0.5))    # r[17]=wvis visible count
    dash.update(msg=f"CALIB pos#{pos_idx} +{len(rows)} -> {len(new)} pts / {npos} positions "
                    f"({nvis} wrist-vis). MOVE the cube to a NEW spot + Capture + re-CALIB (need >=6 "
                    "positions); delete wrist_calib_data.npz to reset.")

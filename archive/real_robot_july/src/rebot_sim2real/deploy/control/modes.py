"""FSM mode steps + AUTO-GRASP sequencer.

Split out the AUTO-GRASP sequencer (1205-1257) and the mode elif chain (1557-2069) of the old debug_deepreach.py `main()`.
Each function takes the ControlLoop (loop) and operates via loop.state / loop.tick / loop.facade / loop.base / loop.dash.
Logic is preserved line-for-line; the only changes are (a) variable access (state fields) (b) CNN branch removal (c) explicit if/else.

⚠️ CNN dropped: Align's IBVS branch (--lookat-ibvs, sd4 CNN bearing servo) removed entirely; yaw priority=latch>seg (cnn removed).
⚠️ sync dropped + prev_front dead: simplified the pt-None branches of follow/approach.
⚠️ Naming: the monolith's follow-wrist-center `step` (shadowing the tick counter) is now `center_step`.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..kinematics import _danger, _gimbal_lookat_ik
from .constants import (AG_ALIGN_DWELL, ALIGN_DWELL_MAX, AUTOGRASP_ALIGN_TIMEOUT,
                        FK_TOUCH_ARM_DZ, FOLLOW_DUN_DLAT, FOLLOW_DVN_DFWD, LOOKAT_RISE_TOL,
                        YAW_ACC_BLEND, YAW_ACC_GIVEUP_N, YAW_ACC_MIN_N, YAW_CONC_MIN)


def _fit_reach(samples):
    """FK-touch reach fit. samples=[(radial, z_gap_io, ...)].
    z_gap_io = intercept + slope·radial OLS (least squares) fit: slope = (n·Σxy − Σx·Σy)/(n·Σxx − (Σx)²).
    (output: (slope, intercept, ref_radial) or None (<2pt/degenerate))"""
    if len(samples) < 2:
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope, (sum_y - slope * sum_x) / n, sum_x / n


# =================================================================================================
# AUTO-GRASP sequencer (runs every tick after the preamble, before the command poll). Phase transitions/abort only; the actual descent motion is in _step_follow.
# =================================================================================================
def step_autograsp_sequencer(loop):
    """AUTO-GRASP one-shot sequencer: align (hover J6 yaw + re-center, dwell-stable) → descend (exact z_gap_io gate) → grasp (force-servo)."""
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    ag = st.autograsp
    pr = tick.pr
    if ag.phase is None:
        return

    # abort guard: stale / over-plunge / 30s timeout → cancel the sequence + hold
    over_plunge = tick.zgap_io_val < loop.grasp_zgap_io - 0.03
    timed_out = ag.phase_t0 is not None and time.time() - ag.phase_t0 > 30.0
    if tick.perc_stale or timed_out or over_plunge:
        ag.phase = None
        st.follow.active = False
        st.grip.servoing = False
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
        dash.update(msg="AUTO-GRASP aborted (stale / over-plunge / timeout) — holding.")
        return

    if ag.phase == "align":
        # YAW-ALIGN FIRST (at hover; best framing/yaw). J6 acquisition + roll happen here (follow step); no re-acquisition during descent.
        if not st.follow.active:                       # Follow stopped → abort, no grip
            ag.phase = None
            dash.update(msg="AUTO-GRASP stopped: Follow halted during yaw-align. Holding.")
            return
        # enter descend = STABLE convergence (yaw reached AND re-centered AND J5 unsaturated) for AG_ALIGN_DWELL continuously. Timeout without convergence → ABORT (no descending over a bad alignment).
        j6_ready = ag.j6_target is not None and abs(float(pr.arm_qpos[5]) - ag.j6_target) < 0.02
        ctr_ready = st.follow.auto_center_latched or not loop.dcfg.follow_auto_center
        j5_ok = (float(base.hi[4]) - abs(float(pr.arm_qpos[4]))) > 0.02   # >~1.1° margin from the ±90 HW bound
        ready_now = j6_ready and ctr_ready and j5_ok
        if ready_now:
            if ag.ready_t0 is None:
                ag.ready_t0 = time.time()              # dwell start
        else:
            ag.ready_t0 = None                         # reset
        stable = ag.ready_t0 is not None and (time.time() - ag.ready_t0) >= AG_ALIGN_DWELL
        align_timeout = ag.phase_t0 is not None and time.time() - ag.phase_t0 > AUTOGRASP_ALIGN_TIMEOUT
        if stable:
            ag.phase = "descend"
            ag.phase_t0 = time.time()                  # yaw aligned + re-centered (stable) → vertical descent
            dash.update(msg="AUTO-GRASP: jaw YAW ALIGNED + ⊙re-CENTERED (stable) → straight-down descend → grip.")
        elif align_timeout:
            ag.phase = None                            # no convergence → ABORT (no descent)
            facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
            yaw_mk = "✓" if j6_ready else "✗"
            ctr_mk = "✓" if ctr_ready else "✗"
            j5_mk = "✓" if j5_ok else "✗sat"
            dash.update(msg=f"AUTO-GRASP: align did NOT stabilize in {AUTOGRASP_ALIGN_TIMEOUT:.0f}s "
                            f"(yaw{yaw_mk} ctr{ctr_mk} J5{j5_mk}) — ABORTED, holding. Re-center + AUTO-GRASP again.")

    elif ag.phase == "descend":
        # XY+J6 already COMMITTED. The only close gate = exact grasp DEPTH (z_gap_io) — no re-framing/yaw re-acquisition (cube is under the jaw).
        if not st.follow.active:                       # Follow stopped → abort, no grip
            ag.phase = None
            dash.update(msg="AUTO-GRASP stopped: Follow halted before grasp depth. Holding.")
        elif tick.zgap_io_val <= loop.grasp_zgap_io + 0.012:   # grasp depth → CLOSE(committed xy)
            ag.phase = "grasp"
            st.follow.active = False
            st.grip.servoing = True
            st.grip.command = st.grip.mid
            ag.settle_count = 0
            facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
            ag.phase_t0 = time.time()
            dash.update(msg="AUTO-GRASP: at grasp depth → GRIP close (force-servo)")

    elif ag.phase == "grasp":                          # force-servo closing; done when grasped + dwell
        if facade.is_grasped(pr.grip_qpos):
            ag.settle_count += 1
        if ag.settle_count > 15 or (ag.phase_t0 is not None and time.time() - ag.phase_t0 > 5.0):
            grasped_ok = facade.is_grasped(pr.grip_qpos)
            ag.phase = None
            if grasped_ok:
                dash.update(msg="AUTO-GRASP ✓ GRASPED — holding. Lift to raise.")
            else:
                dash.update(msg="AUTO-GRASP ✓ closed (no grasp signal — check contact).")


# =================================================================================================
# Mode dispatch (same order and mutual exclusion as the monolith elif chain). Called only on ticks without a command.
# =================================================================================================
def step(loop):
    """Runs the one active mode (priority RUN → Align → Follow → Approach → Lift → FK-touch → Grip)."""
    st = loop.state
    tick = loop.tick
    if st.run_active and tick.q_cmd is not None:
        _step_run(loop)
    elif st.align.active:
        _step_align(loop)
    elif st.follow.active:
        _step_follow(loop)
    elif st.approach.active:
        _step_approach(loop)
    elif st.lift.active:
        _step_lift(loop)
    elif st.fk_touch.phase is not None:
        _step_fk_touch(loop)
    elif st.grip.servoing:
        _step_grip(loop)


def _step_run(loop):
    """ONE-SHOT RUN: deep_reach servo every loop (smoothed); stop on exact reach (reached_run)."""
    st = loop.state
    tick = loop.tick
    loop.facade.send_arm(tick.q_cmd[0].numpy(), st.grip.command, loop.loop_dt)
    st.step += 1
    if tick.reached_run:                               # exact reach (z_gap_io≈goal + xy), not base at_cube
        st.run_active = False
        loop.dash.update(msg=f"RUN ✓ reached cube (step {st.step}, z_gap_io≈goal). gripper OPEN — Align, then grasp.")


def _step_align(loop):
    """LATCHED auto-center-then-J6: frame the cube TOP-edge with J4 (align_vn_target) → J6 roll by seg yaw. Requires the cube to be visible."""
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    pr = tick.pr
    al = st.align
    if tick.svis <= 0.5:
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)   # cube not visible → hold
        dash.update(msg="ALIGN …: cube not wrist-visible (svis<0.5) — holding; RUN to the cube first")
        return
    err = al.vn_target - tick.stopvn                   # target the top-edge
    al.dwell += 1
    if abs(err) > al.vn_band and al.dwell < ALIGN_DWELL_MAX:   # still centering
        q = pr.arm_qpos.copy()
        dj4 = float(np.clip(al.j4_gain * err, -0.05, 0.05))   # gentle per-loop pitch step
        j4_new = float(np.clip(q[3] + dj4, float(base.lo[3]), float(base.hi[3])))
        if abs(j4_new - q[3]) < 1e-5:                  # J4 pinned at its limit, not yet framed
            al.active = False
            dash.update(msg="ALIGN ✗ J4 hit its limit before framing — NEGATE --align-j4-gain, re-Align")
        else:
            q[3] = j4_new
            facade.send_arm(q, st.grip.command, loop.loop_dt)   # shaped J4 servo
            dash.update(msg=f"ALIGN …centering: top-edge vn={tick.stopvn:+.2f} → {al.vn_target:+.2f} "
                            f"(err {err:+.2f}); J4 {np.degrees(dj4):+.1f}°")
    else:                                              # framed (or dwell timeout) → J6 roll, done
        # yaw priority: (1) LATCHED mid-descend yaw (most trusted), (2) live seg edge-voting. (CNN head dropped)
        yl = st.yaw_latch
        if yl.weight > 0.3:                            # frozen mid-range(RUN latch)
            theta = float(np.arctan2(yl.sin4, yl.cos4) / 4.0)
            ysrc = "latch"
        else:
            theta = float(np.arctan2(tick.ssin4, tick.scos4) / 4.0)
            ysrc = "seg-live"
        q = pr.arm_qpos.copy()
        q[5] = float(np.clip(q[5] - al.jaw_sign * theta, float(base.lo[5]), float(base.hi[5])))
        facade.send_arm(q, st.grip.command, loop.loop_dt, snap=True)
        al.active = False
        if abs(err) <= al.vn_band:
            framed_mk = "✓"
        else:
            framed_mk = "⚠ frame-timeout"
        dash.update(msg=f"ALIGN {framed_mk} (top-edge vn={tick.stopvn:+.2f}) + "
                        f"J6 {np.degrees(-al.jaw_sign * theta):+.0f}° [{ysrc} yaw θ={np.degrees(theta):+.0f}°]. "
                        "'Final' to descend; wrong J6 way → 'Align flip' + re-Align.")


def _step_follow(loop):
    """HOVER-FOLLOW: TCP tracks straight-down at --follow-z above cube_world. Stable → descend to --follow-low, moving → rise and re-track.
    Keeps the cube at the wrist center via auto-center (measured Jacobian) or gimbal look-at. The AUTO-GRASP descent motion also rides on this."""
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    dcfg = loop.dcfg
    pr = tick.pr
    fol = st.follow
    ag = st.autograsp
    cube_world = tick.cube_world
    if cube_world is None:
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
        dash.update(msg="FOLLOW …: no cube (Capture, or run with --track). Holding.")
        return

    cxy = np.array([cube_world[0] + dcfg.cube_x_off, cube_world[1] + dcfg.cube_y_off])
    surf_z = float(base.cube_top_z)                    # auto-z (FK-corrected) surface → +Ncm is the REAL height
    cen_e = None                                       # wrist centroid center offset (set below while centering)
    # ⭐ Descent closed-loop gate (docs/32-33): during descend, while the cube is visible and before the final commit, keep running auto-center
    # to visually cancel gravity droop in real time. An open-loop straight descent (even though IK holds xy perfectly) exposed the real arm
    # reaching down-forward and leaking toward +Y from J2/J3 gravity droop (the jaw closed on air next to the cube). Below the commit,
    # FREEZE the last centered grasp xy (IK block below), then a pure vertical plunge = safe through the cube-occlusion zone.
    ag_descend_servo = (ag.phase == "descend" and bool(dcfg.follow_auto_center) and tick.wvis > 0.5
                        and float(tick.zgap_io_val) > loop.grasp_zgap_io + float(dcfg.autograsp_commit_zgap))
    # the RISE decision reads only the FRONT est (front_ref_xy) — separate from the wrist-owned target_xy (so front-vs-wrist bias does not become a fake 'move')
    if fol.front_ref_xy is None:
        dmove = 0.0
    else:
        dmove = float(np.linalg.norm(cxy - fol.front_ref_xy))
    # real relocation → RISE. Suppressed during AUTO-GRASP (descent committed; low-z front jitter must not bounce back to hover).
    moved = (ag.phase is None) and (fol.front_ref_xy is None or dmove > float(dcfg.follow_move))
    if moved:
        fol.front_ref_xy = cxy                         # moved far → re-anchor front, RISE from above + re-track
        fol.target_xy = cxy                            # reset the setpoint to the coarse front (wrist refines again)
        fol.height_above_cube = float(dcfg.follow_z)
        fol.auto_center_latched = False                # relocate → re-center at the new spot (discard latch)
        fol.lookat_last_q = None                       # discard held gimbal pose → re-aim at the new spot
    else:                                              # stable-ish: descend to --follow-low
        if dmove > float(dcfg.follow_deadband):        # small front drift → re-anchor the RISE ref (ignore jitter)
            fol.front_ref_xy = cxy
        # AUTO-CENTER (closed loop): nudge the cam-fwd/lat offset so the wrist cube (u,v)→0. Only on FRESH perception frames (ts-gate; prevents 30Hz over-correction).
        ac_ts = tick.perc.get("ts")
        if (dcfg.follow_auto_center and not fol.lookat and (ag.phase in (None, "align") or ag_descend_servo)
                and tick.wvis > 0.5 and ac_ts != fol.auto_center_ts):
            # RATE-INDEPENDENT gain: the integration step scales with the perception PERIOD (constant integration rate per second whether 10 or 30Hz).
            if fol.auto_center_ts is not None and ac_ts is not None:
                dtp_raw = ac_ts - fol.auto_center_ts
            else:
                dtp_raw = 0.1
            dtp = float(np.clip(dtp_raw, 0.02, 0.3))
            fol.auto_center_ts = ac_ts
            gain = float(dcfg.follow_auto_gain) * (dtp / 0.1)
            margin = float(dcfg.follow_center_margin)
            err_u = float(tick.wun) - fol.center_un_target   # error from the target (not 0,0)
            err_v = float(tick.wvn) - fol.center_vn_target
            err_mag = max(abs(err_u), abs(err_v))
            if err_mag < margin:                       # HYSTERESIS: target reached → latch, stop
                fol.auto_center_latched = True
            elif err_mag > 2.0 * margin:               # well off (real movement) → resume
                fol.auto_center_latched = False
            # In the descent closed loop (docs/33) bypass the latch = continuous tracking. The hover latch hysteresis prevents chatter, but
            # during descent droop keeps growing, so leaving err between margin~2*margin (≈4~8mm) leaves that much residual miss →
            # during ag_descend_servo always servo when err>margin, tightening to <margin right up to the grasp.
            if (not fol.auto_center_latched) or ag_descend_servo:   # servo when not centered (+ descent closed loop is continuous)
                if abs(err_v) > margin:                # dvn/dfwd=-19/m → Δfwd drives wvn to target
                    fol.cam_fwd += float(np.clip(-gain * err_v / FOLLOW_DVN_DFWD, -0.004, 0.004))
                if abs(err_u) > margin:                # dun/dlat=-9.5/m → Δlat drives wun to target
                    fol.cam_lat += float(np.clip(-gain * err_u / FOLLOW_DUN_DLAT, -0.004, 0.004))
                # ANTI-WINDUP: at far reach IK cannot deliver the shift → offset grows without bound (windup). Cap (near converges well inside).
                cam_max = float(dcfg.follow_cam_max)
                fol.cam_fwd = float(np.clip(fol.cam_fwd, -cam_max, cam_max))
                fol.cam_lat = float(np.clip(fol.cam_lat, -cam_max, cam_max))
        # xy FINE-tracking: when the cube is visible the WRIST owns it (straight-down cam = angle-invariant, more accurate than front near the cube). FREEZE during AUTO-GRASP.
        if dcfg.follow_wrist_center and ag.phase is None and tick.wvis > 0.5 and tick.wrist_xy is not None:
            cen_e = max(abs(float(tick.wun)), abs(float(tick.wvn)))   # centroid offset from the wrist center [0..1]
            if cen_e > float(dcfg.follow_center_margin):              # outside margin → re-center nudge
                # pixel offset → 0. The reference is the point the wrist cam CENTER (0,0) actually sees — not the TCP (lens is off wrist_mount + optical axis ~22° tilt).
                wrist_center = base._wrist_uv_to_world(tick.arm_q, torch.zeros(1), torch.zeros(1),
                                                       loop.plane_z)[0].cpu().numpy()
                center_step = float(dcfg.follow_center_gain) * (np.array(tick.wrist_xy) - wrist_center)   # centering direction
                step_norm = float(np.linalg.norm(center_step))
                if step_norm > 0.015:                  # cap step size at 1.5cm/tick (outlier-safe)
                    center_step = center_step * (0.015 / step_norm)
                fol.target_xy = fol.target_xy + center_step
            # inside margin → aligned: hold xy + keep descending (deadband that kills chatter)
        elif dmove > float(dcfg.follow_deadband) and ag.phase is None:
            fol.target_xy = cxy                        # no wrist (target_xy kept by hold-on-loss) → front drift track
        if fol.height_above_cube is None:              # gently ramp (shaper/LPF/chunk smooth the actual motion)
            fol.height_above_cube = float(dcfg.follow_z)
        if ag.phase == "align":
            pass                                       # YAW-ALIGN phase: HOLD hover height while rolling J6 (descent is started by the sequencer after align completes)
        elif ag.phase == "descend":
            # AUTO-GRASP descent: SETTLE-GATED (docs/33). Lowering with follow-step (0.004) in a 100Hz loop covers 5cm in ~0.15s → 30Hz
            # auto-center gets only ~5 frames, so droop (+Y) correction is 3-4x short (exceeds the τ≈1s servo bandwidth). So z is lowered (dedicated slow step) only while
            # the wrist is centered; outside the center margin z is HELD, waiting until the servo re-centers →
            # descent speed self-limits to the servo bandwidth. At commit (ag_descend_servo=False, final plunge) descend without the gate.
            descend_ok = True
            if ag_descend_servo and tick.wvis > 0.5:
                err_c = max(abs(float(tick.wun) - fol.center_un_target),
                            abs(float(tick.wvn) - fol.center_vn_target))
                if err_c > float(dcfg.follow_center_margin):   # not centered → stop z, wait for the servo
                    descend_ok = False
            if descend_ok:
                fol.height_above_cube = max(-0.05, fol.height_above_cube - float(dcfg.autograsp_descend_step))
        else:                                          # normal Follow: descend to the --follow-low hover
            fol.height_above_cube = max(float(dcfg.follow_low),
                                        fol.height_above_cube - float(dcfg.follow_descend_step))

    seed = torch.tensor(pr.arm_qpos, dtype=torch.float32, device=base.device).view(1, 6)
    lk_tag = ""
    framed = (tick.wvis > 0.5
              and float(np.hypot(float(tick.wun) - fol.center_un_target, float(tick.wvn) - fol.center_vn_target))
              < float(dcfg.lookat_image_gate))
    if fol.lookat and framed:
        # GIMBAL LOOK-AT: HOLD the TCP at hover (no cam-offset), tilt only the wrist to aim the cam at the cube. IMAGE-mode (cancels errors).
        # ts-GATE: re-aim only on FRESH frames (prevents re-applying stale ones); HOLD last_q in between.
        lk_ts = tick.perc.get("ts")
        if lk_ts != fol.lookat_ts or fol.lookat_last_q is None:
            fol.lookat_ts = lk_ts
            tgt = torch.tensor([[fol.target_xy[0], fol.target_xy[1], surf_z + fol.height_above_cube]],
                               dtype=torch.float32, device=base.device)
            look = torch.tensor([[fol.target_xy[0], fol.target_xy[1], surf_z]],
                                dtype=torch.float32, device=base.device)   # cube world (aim ref)
            meas = torch.tensor([[float(tick.wun) - fol.center_un_target, float(tick.wvn) - fol.center_vn_target]],
                                dtype=torch.float32, device=base.device)   # (u,v)→framing target
            if dcfg.lookat_freeze_j6:
                freeze = (5,)
            else:
                freeze = ()
            qf = _gimbal_lookat_ik(base, seed, tgt, look, iters=int(dcfg.lookat_image_iters),
                                   astep=float(dcfg.lookat_image_astep), aim_gain=float(dcfg.lookat_image_gain),
                                   freeze=freeze, meas_uv=meas)[0].cpu().numpy()
            if float(dcfg.lookat_j5_rate) > 0:         # J5 per-tick rate cap → smooth pan, no whip
                r = float(dcfg.lookat_j5_rate)
                qf[4] = float(np.clip(qf[4], float(pr.arm_qpos[4]) - r, float(pr.arm_qpos[4]) + r))
            fol.lookat_last_q = qf
        else:
            qf = fol.lookat_last_q                     # STALE frame → HOLD the last gimbal pose (no re-aim)
        lk_tag = " · look-at⊙"
    else:
        # STRAIGHT-DOWN + auto-center: shift hover along the gripper axis (J1 rotation) to bring the cube to the wrist center.
        ghat = base.fwd_axis_xy(tick.arm_q)[0].cpu().numpy()          # gripper-forward unit xy (J1 rotation)
        ctgt = (np.array(fol.target_xy) + fol.cam_fwd * ghat
                + fol.cam_lat * np.array([-ghat[1], ghat[0]]))        # + lateral (perpendicular to forward)
        if ag.phase == "descend":
            if ag_descend_servo:
                # descent closed loop: keep it uncommitted (ctgt=live target_xy+cam offset; the auto-center above cancels droop).
                # explicit None → even if the servo briefly turns off/on from a momentary occlusion, no stale freeze remains and it is fresh at the real commit.
                ag.committed_xy = None
            else:
                # COMMIT (final plunge): ctgt recomputes the LIVE ghat every tick → when the arm reconfigures the cam-offset rotates → TCP creeps forward.
                # So FREEZE the 'last centered' grasp xy at commit time and hold → xy fixed, only z decreases.
                if ag.committed_xy is None:
                    ag.committed_xy = ctgt.copy()
                ctgt = ag.committed_xy
        tgt = torch.tensor([[ctgt[0], ctgt[1], surf_z + fol.height_above_cube]],
                           dtype=torch.float32, device=base.device)
        down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=base.device)
        qf = base.ik_oriented(seed, tgt, down, iters=int(dcfg.follow_iters))[0].cpu().numpy()
        qf[5] = float(pr.arm_qpos[5])                  # HOLD J6 (roll): straight-down leaves J6 FREE → ik whips → freeze the current angle
        if dcfg.autograsp_yaw and ag.phase in ("align", "descend"):
            # align the jaw to the cube FACE. The J6 target is acquired at HOVER (align) from the first confident yaw (latch>edge-voted seg), then HELD.
            if ag.phase == "align" and ag.j6_target is None:
                # ⭐ CONSERVATIVE YAW GATE (docs/33): live seg edge-voting yaw jumps around the mod-90 range from frame to frame;
                # latching on the first confident frame grabs a wrong yaw (≤45°) and fails. Accumulate seg yaw as a fresh-frame EMA
                # and rotate J6 only when temporally stable (EMA vector magnitude=concentration ≥ YAW_CONC_MIN); if instability
                # persists until GIVEUP, grasp with J6=0 (no-roll) (for a near-square that is safer than a wrong yaw).
                yl = st.yaw_latch
                theta = None                                               # trustworthy yaw (rad) or None
                no_roll = False                                            # True → commit to J6=current (no rotation)
                if yl.weight > 0.3 and (yl.cos4 * yl.cos4 + yl.sin4 * yl.sin4) >= YAW_CONC_MIN * YAW_CONC_MIN:
                    theta = float(np.arctan2(yl.sin4, yl.cos4) / 4.0)      # RUN-descent latch (averaged, stable) → trusted
                elif tick.svis > 0.5 and (tick.scos4 * tick.scos4 + tick.ssin4 * tick.ssin4) > 0.5:
                    yaw_fr_ts = tick.perc.get("ts")
                    if yaw_fr_ts != ag.align_yaw_ts:                       # accumulate only on fresh perception frames (30Hz gate)
                        ag.align_yaw_ts = yaw_fr_ts
                        b = YAW_ACC_BLEND
                        ag.align_yaw_cos = (1.0 - b) * ag.align_yaw_cos + b * float(tick.scos4)
                        ag.align_yaw_sin = (1.0 - b) * ag.align_yaw_sin + b * float(tick.ssin4)
                        ag.align_yaw_n += 1
                    conc = float(np.hypot(ag.align_yaw_cos, ag.align_yaw_sin))   # temporal concentration [0..1]
                    if ag.align_yaw_n >= YAW_ACC_MIN_N and conc >= YAW_CONC_MIN:
                        theta = float(np.arctan2(ag.align_yaw_sin, ag.align_yaw_cos) / 4.0)   # EMA direction (noise averaged)
                    elif ag.align_yaw_n >= YAW_ACC_GIVEUP_N:
                        no_roll = True                                     # instability persists → no-roll (J6=current), just grasp
                if theta is not None:
                    ag.j6_target = float(np.clip(float(pr.arm_qpos[5]) - st.align.jaw_sign * theta,
                                                 float(base.lo[5]), float(base.hi[5])))
                elif no_roll:
                    ag.j6_target = float(pr.arm_qpos[5])                   # J6=current (no rotation)
            if ag.j6_target is not None:
                qf[5] = ag.j6_target                   # HOLD the aligned jaw yaw after the (align) roll
    facade.send_arm(qf, st.grip.command, loop.loop_dt)   # SHAPED (not snap) → smooth rate-limited tracking

    # ---- Dashboard message (many value-guard ternaries; display only) ----
    if ag.phase == "align":
        if ag.j6_target is not None:
            _yawmk = "✓"
        else:
            _yawmk = "…"
        if fol.auto_center_latched:
            _ctrmk = "✓"
        else:
            _ctrmk = "…"
        phase_tag = f"AG:align(yaw{_yawmk}/ctr{_ctrmk})"
    elif ag.phase == "descend":
        phase_tag = "AG:descend↓"
    elif moved:
        phase_tag = "RISE"
    elif fol.height_above_cube <= float(dcfg.follow_low) + 1e-4:
        phase_tag = "AT-LOW"
    else:
        phase_tag = "DESCEND"
    cen_tag = ""
    if dcfg.follow_wrist_center and not moved:
        if cen_e is not None:
            if cen_e <= float(dcfg.follow_center_margin):
                cen_tag = " · wrist CENTERED"
            else:
                cen_tag = f" · wrist re-center e={cen_e:.2f}"
        else:
            cen_tag = " · wrist·(no cube)"
    off_tag = ""
    if fol.cam_fwd or fol.cam_lat:
        off_tag = f" · cam-off fwd{fol.cam_fwd*100:+.0f}/lat{fol.cam_lat*100:+.0f}cm"
    tgt_tag = ""
    if dcfg.follow_auto_center and (fol.center_un_target or fol.center_vn_target):
        tgt_tag = f"→({fol.center_un_target:+.2f},{fol.center_vn_target:+.2f})"
    if tick.wvis > 0.5:
        if fol.auto_center_latched:
            centered_tag = " ⊙CENTERED"
        else:
            centered_tag = ""
        uv_tag = f" · cube@wrist({tick.wun:+.2f},{tick.wvn:+.2f}){tgt_tag}{centered_tag}"
    else:
        uv_tag = " · wrist·(no cube)"
    dash.update(msg=f"FOLLOW ▶ track ({fol.target_xy[0]:+.3f},{fol.target_xy[1]:+.3f}) "
                    f"+{fol.height_above_cube*100:.0f}cm [{phase_tag}]{lk_tag}{off_tag}{uv_tag}{cen_tag}")


def _step_approach(loop):
    """LOOK-AT APPROACH (gimbal): continuous descent to hover (--lookat-approach-stop) while the wrist cam aims at the cube.
    coarse (front)→fine (wrist lock) handoff, xy-alignment ramp descent, safety danger→REST. With --lookat-grasp, all the way to grasp."""
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    dcfg = loop.dcfg
    pr = tick.pr
    ap = st.approach
    # live per-frame front est (--track), otherwise the captured cube_world. approach uses the live one directly (bypasses the 1.5cm anchor deadband).
    if st.track.enabled:
        live = tick.perc.get("cube_live")
    else:
        live = None
    if live is not None:
        base_xy = live
    else:
        base_xy = tick.cube_world
    if base_xy is None:
        facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
        dash.update(msg="APPROACH …: no cube (Capture, or run with --track). Holding.")
        return

    # COARSE→FINE handoff (front=coarse, WRIST=fine; so they do not fight each other).
    cxy_front = np.array([base_xy[0] + dcfg.cube_x_off, base_xy[1] + dcfg.cube_y_off])
    if ap.wrist_lock_xy is None:
        if ap.target_xy is None or np.linalg.norm(cxy_front - ap.target_xy) > dcfg.lookat_deadband:
            ap.target_xy = cxy_front                   # coarse: per-frame front follow(deadband)
        cxy = np.array(ap.target_xy, float)
    else:
        cxy = np.array(ap.wrist_lock_xy, float)        # fine: the wrist owns the target
        ap.target_xy = np.array(ap.wrist_lock_xy, float)
        if np.linalg.norm(cxy_front - ap.wrist_lock_xy) > float(dcfg.lookat_reacquire):
            ap.wrist_lock_xy = None                    # relocation → re-coarse
            ap.target_xy = cxy_front
            cxy = cxy_front
    seed = torch.tensor(pr.arm_qpos, dtype=torch.float32, device=base.device).view(1, 6)
    if tick.wvis > 0.5:
        if dcfg.lookat_ibvs_xy:
            # CALIB-TOLERANT IBVS xy: measured wrist px vs the cube reference under the FK-TCP (same cam pose → cancels extrinsic error; 55mm→12mm).
            un = torch.tensor([float(tick.wun)], dtype=torch.float32, device=base.device)
            vn = torch.tensor([float(tick.wvn)], dtype=torch.float32, device=base.device)
            vis = torch.tensor([float(tick.wvis)], dtype=torch.float32, device=base.device)
            cxy_t = torch.tensor([[float(cxy[0]), float(cxy[1])]], dtype=torch.float32, device=base.device)
            dxy = base._wrist_ibvs_dxy(seed, cxy_t, un, vn, vis, gain=float(dcfg.lookat_ibvs_gain))[0].cpu().numpy()
            if float(np.linalg.norm(dxy)) > 0.0:       # 0 = not visible/clutter-gate → keep coarse
                ibvs_step = float(dcfg.lookat_ibvs_step)
                cand = np.array(tick.tcp_w[:2], float) + np.clip(dxy, -ibvs_step, ibvs_step)   # FINE step
                if np.linalg.norm(cand - cxy) < 0.12:  # sanity vs the current target (reject rogue blobs)
                    cxy = cand
        else:
            wxy = base._wrist_uv_to_world(seed,
                                          torch.tensor([float(tick.wun)], dtype=torch.float32, device=base.device),
                                          torch.tensor([float(tick.wvn)], dtype=torch.float32, device=base.device),
                                          float(base.cube_top_z))[0].cpu().numpy()
            if np.linalg.norm(wxy - cxy) < 0.12:       # agrees with front → slow fuse (reject FK-corrupt unproj)
                cxy = (1.0 - float(dcfg.lookat_fuse)) * cxy + float(dcfg.lookat_fuse) * wxy
        ap.wrist_lock_xy = np.array(cxy, float)        # PERSIST the wrist-refined xy (survives wrist drop-out)
    elif ap.wrist_lock_xy is not None:
        cxy = np.array(ap.wrist_lock_xy, float)        # HOLD-ON-LOSS: no snap-back to the coarse front

    surf_z = float(base.cube_top_z)                    # auto-z descent surface (same target as RUN)
    hover_abs = surf_z + float(dcfg.lookat_hover)
    if dcfg.lookat_grasp:
        floor_abs = surf_z + float(dcfg.lookat_grasp_clear)   # all the way to grasp
    else:
        floor_abs = surf_z + float(dcfg.lookat_approach_stop)  # +5cm HOVER endpoint (default)
    if ap.height is None:
        ap.height = hover_abs                          # start at hover → the arm first rises ~10cm
    # EXPLICIT grasp offset (io-INDEPENDENT): aim the kinematic TCP lookat_jaw_fwd ahead (+fwd) (fwd_axis rotates with J1).
    ghat = np.array(base.fwd_axis_xy(tick.arm_q)[0].cpu().tolist(), float)
    cxy_cmd = np.array(cxy, float) + loop.lookat_jaw_fwd * ghat   # grasp-position IK target
    # GATE metric = REAL grasp alignment (io grasp vs cube). When the io alignment becomes small (real jaw above the cube) the gimbal (null-space) engages.
    xy_off = float(np.linalg.norm(np.array(tick.tcp_io[:2]) - cxy))
    # SAFETY (deviation): cube moved far while physically LOW → forbid a low lateral lunge → REST then re-approach from above.
    #   MEASURED tcp_w[2] gate (not commanded): physically high after lift → DISARM (prevents an infinite loop). -0.02=2cm hysteresis.
    if (float(dcfg.lookat_reapproach_dist) > 0 and xy_off > float(dcfg.lookat_reapproach_dist)
            and (float(tick.tcp_w[2]) - surf_z) < (float(dcfg.lookat_hover) - 0.02)):
        loop.return_to_rest(f"APPROACH ⚠ cube deviated {xy_off*100:.0f}cm while low → LIFTED to REST, "
                            "re-approaching from the top.")
        return                                         # (monolith continue; return_to_rest blocks ~2s so skipping the sleep is meaningless)
    align = float(np.clip(1.0 - xy_off / max(float(dcfg.lookat_xy_slow), 1e-6), 0.0, 1.0))
    # WRIST-ACQUIRE descent gate (--lookat-ibvs-xy): HOLD z (align→0) until the wrist acquires the cube (wvis>0.5) + roughly centers it.
    if dcfg.lookat_ibvs_xy:
        if tick.wvis > 0.5:
            wuv_mag = float(np.hypot(float(tick.wun), float(tick.wvn)))
        else:
            wuv_mag = 9.9
        if not (tick.wvis > 0.5 and wuv_mag < float(dcfg.lookat_descend_uv)):
            align = 0.0                                # hover dwell; IBVS refines xy every tick
    # rise-first: keep commanding hover and HOLD until the arm physically rises, then descend.
    if not ap.risen:
        ap.risen = (float(tick.tcp_w[2]) >= hover_abs - LOOKAT_RISE_TOL)
        ap.height = hover_abs                          # keep commanding hover → rise, no descent yet
    else:
        ap.height = max(floor_abs, ap.height - float(dcfg.lookat_descend_step) * align)   # continuous, monotone

    tgt = torch.tensor([[float(cxy_cmd[0]), float(cxy_cmd[1]), ap.height]], dtype=torch.float32, device=base.device)
    look = torch.tensor([[float(cxy[0]), float(cxy[1]), surf_z]], dtype=torch.float32, device=base.device)   # VIEW aims at the real cube
    down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=base.device)
    dz = ap.height - surf_z
    # AIM only when the arm is above the cube (small xy_off). Aiming while moving (large xy_off) pushes the TCP away → straight-down first.
    aligned = xy_off < float(dcfg.lookat_image_xygate)
    do_gimbal = (not dcfg.lookat_straight) and (dz > float(dcfg.lookat_taper)) and aligned
    if do_gimbal:                                      # gimbal-SAFE: J4+J5 aim(J6 frozen in-IK), J5 rate-cap
        # IMAGE-aim (default): measured (u,v)→center (cancels extrinsic/front error). Only when the arm is above the cube + roughly framed + visible.
        uv_mag = float(np.hypot(float(tick.wun), float(tick.wvn)))
        if (dcfg.lookat_image_aim and tick.wvis > 0.5 and xy_off < float(dcfg.lookat_image_xygate)
                and uv_mag < float(dcfg.lookat_image_gate)):
            meas = torch.tensor([[float(tick.wun), float(tick.wvn)]], dtype=torch.float32, device=base.device)
        else:
            meas = None
        if meas is not None:
            # IMAGE-aim = online closed loop → SMALL steps (large steps DIVERGE).
            aim_iters = int(dcfg.lookat_image_iters)
            aim_gain = float(dcfg.lookat_image_gain)
            aim_step = float(dcfg.lookat_image_astep)
        else:
            # WORLD-aim = feedforward → converges in one tick; gain scheduled DOWN near the cube (damping).
            w = float(np.clip(dz / max(float(dcfg.lookat_hover), 1e-6), 0.0, 1.0))   # hover=1, surface=0
            aim_gain = w * float(dcfg.lookat_aim_gain) + (1.0 - w) * float(dcfg.lookat_descend_aim_gain)
            aim_step = w * float(dcfg.lookat_astep) + (1.0 - w) * float(dcfg.lookat_descend_astep)
            aim_iters = int(dcfg.lookat_iters)
        if dcfg.lookat_freeze_j6:
            freeze = (5,)
        else:
            freeze = ()
        qf = _gimbal_lookat_ik(base, seed, tgt, look, iters=aim_iters, astep=aim_step, aim_gain=aim_gain,
                               freeze=freeze, meas_uv=meas)[0].cpu().numpy()
        if float(dcfg.lookat_j5_rate) > 0:             # J5 per-tick rate cap
            r = float(dcfg.lookat_j5_rate)
            qf[4] = float(np.clip(qf[4], float(pr.arm_qpos[4]) - r, float(pr.arm_qpos[4]) + r))
    else:                                              # STRAIGHT-DOWN + xy tracking + J5,J6 HOLD
        qf = base.ik_oriented(seed, tgt, down, iters=int(dcfg.lookat_iters))[0].cpu().numpy()
        qf[4] = float(pr.arm_qpos[4])                  # HOLD J5(wrist-yaw) → no whip
        qf[5] = float(pr.arm_qpos[5])                  # HOLD J6(wrist-roll) → no spin
    if dcfg.safety_auto_rest:                          # KINEMATIC danger supervisor (surf_z=base FK frame)
        if dcfg.lookat_grasp_resid is not None:
            resid = float(dcfg.lookat_grasp_resid)
        else:
            resid = float(dcfg.safety_reach_resid)
        danger_reason = _danger(base, qf, cxy_cmd, surf_z, resid_max=resid)
        if danger_reason is not None:
            ap.danger_count += 1                       # DEBOUNCE single-tick IK transients
        else:
            ap.danger_count = 0
        if ap.danger_count >= 3:                       # persistent danger → auto-rest + HALT
            ap.danger_count = 0
            loop.return_to_rest(f"⚠ SAFETY: {danger_reason} → auto-returned to REST + HALTED. "
                                "Re-Capture/Approach to retry.", halt=True)
            return
    facade.send_arm(qf, st.grip.command, loop.loop_dt)   # SHAPED(rate-limited)
    at_floor = ap.height <= floor_abs + 1e-3
    if tick.wvis > 0.5:
        wr = "wrist✓"
    else:
        wr = "wrist·"
    if do_gimbal:
        mode = "gimbal J4+J5"
    elif dcfg.lookat_grasp:
        mode = "straight↓ taper"
    else:
        mode = "straight↓ yaw-fixed"
    if dcfg.lookat_grasp:
        # grasp handoff: 'at grasp' is the exact robot_io height gate (same as reached_run).
        at_grasp = at_floor and xy_off < 0.02 and abs(tick.zgap_io_val - loop.grasp_zgap_io) < 0.015
        if at_grasp and dcfg.lookat_autograsp:
            st.grip.command = st.grip.mid              # mirror the 'close' handler → force-servo
            st.grip.servoing = True
            ap.active = False
            dash.update(msg=f"APPROACH ✓ AT GRASP (zgap_io={tick.zgap_io_val:+.3f}) → AUTO Grip close (force-servo)")
        else:
            if at_grasp:
                tail = " · AT GRASP → click Grip"
            elif ap.risen:
                tail = " · descending to grasp"
            else:
                tail = " · rising to hover"
            dash.update(msg=f"APPROACH ▶ ({cxy[0]:+.3f},{cxy[1]:+.3f}) dz={ap.height - surf_z:+.3f}m "
                            f"[{mode} {wr}] xy_off={xy_off * 1000:.0f}mm{tail}")
    else:
        if at_floor:
            hover_tail = " · HOVER-READY → Align/Final/close"
        else:
            hover_tail = ""
        dash.update(msg=f"APPROACH ▶ ({cxy[0]:+.3f},{cxy[1]:+.3f}) dz={ap.height - surf_z:+.3f}m "
                        f"[{mode} {wr}] xy_off={xy_off * 1000:.0f}mm desc={align * 100:.0f}%{hover_tail}")


def _step_lift(loop):
    """GENTLE LIFT: ramp the TCP-z target at --lift-rate/tick up to target_z (dedicated speed), straight up over lift_xy. Hold at the top."""
    st = loop.state
    tick = loop.tick
    facade = loop.facade
    base = loop.base
    pr = tick.pr
    lf = st.lift
    lf.cur_z = min(lf.target_z, lf.cur_z + float(loop.dcfg.lift_rate))
    seed = torch.tensor(pr.arm_qpos, dtype=torch.float32, device=base.device).view(1, 6)
    tgt = torch.tensor([[lf.xy[0], lf.xy[1], lf.cur_z]], dtype=torch.float32, device=base.device)
    down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=base.device)
    qf = base.ik_oriented(seed, tgt, down, iters=int(loop.dcfg.follow_iters))[0].cpu().numpy()
    qf[5] = float(pr.arm_qpos[5])                      # no J6 roll during lift
    facade.send_arm(qf, st.grip.command, loop.loop_dt)   # SHAPED; grip stays CLOSED (cube held)
    if lf.cur_z >= lf.target_z - 1e-4:
        at_mk = "✓ holding"
    else:
        at_mk = "↑"
    loop.dash.update(msg=f"LIFT {at_mk} z={float(tick.tcp_w[2]):.3f}→{lf.target_z:.3f} "
                         f"(@{loop.dcfg.lift_rate*100:.1f}cm/tick; grip={st.grip.command:.3f}); "
                         "click to keep rising, Rest/Stop to end")


def _step_fk_touch(loop):
    """SEMI-AUTO TABLE-TOUCH (FK-Z verification): straight descent → record z_gap_io at the J2/J3 torque rise (table contact) → rise. reach-slope fit."""
    st = loop.state
    tick = loop.tick
    dash = loop.dash
    facade = loop.facade
    base = loop.base
    dcfg = loop.dcfg
    pr = tick.pr
    ft = st.fk_touch
    seed = torch.tensor(pr.arm_qpos, dtype=torch.float32, device=base.device).view(1, 6)
    down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=base.device)
    if ft.phase == "descend":
        dz = ft.z_start - ft.z_cur                     # descent commanded so far
        if not ft.armed and dz >= FK_TOUCH_ARM_DZ:     # after the accel/settle transient → re-baseline with steady-descent torque, arm contact detection
            ft.tau_baseline = [float(x) for x in pr.arm_torque]
            ft.armed = True
        if ft.armed:
            dtau = max(abs(float(pr.arm_torque[1]) - ft.tau_baseline[1]),
                       abs(float(pr.arm_torque[2]) - ft.tau_baseline[2]))   # J2/J3 rise vs the steady baseline
        else:
            dtau = 0.0
        if ft.armed and dtau > float(dcfg.fk_touch_dtau):   # ---- CONTACT: freeze, record, lift ----
            radial = float(np.hypot(tick.tcp_io[0] - float(loop.cfg.base_pos[0]),
                                    tick.tcp_io[1] - float(loop.cfg.base_pos[1])))
            ft.samples.append((radial, tick.zgap_io_val, float(tick.tcp_io[2] - tick.tcp_w[2])))
            facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
            ft.phase = "lift"
            msg = (f"FK-TOUCH ✓ #{len(ft.samples)} radial={radial:.3f}m z_gap_io={tick.zgap_io_val:+.4f} "
                   f"(dtau={dtau:.2f}). ")
            fit = _fit_reach(ft.samples)
            if fit:
                slope_b, _intercept, ref = fit
                msg += f"FIT: slope={slope_b*1000:+.0f}mm/m → --z-reach-slope {slope_b:+.4f} --z-reach-ref {ref:.3f}"
            else:
                msg += "need ≥2 distances for a slope."
            print("[fk-touch] " + msg, flush=True)
            dash.update(msg=msg)
        elif dz > float(dcfg.fk_touch_max):            # ---- SAFETY: no contact → abort, lift ----
            facade.send_arm(pr.arm_qpos, st.grip.command, loop.loop_dt, snap=True)
            ft.phase = "lift"
            dash.update(msg=f"FK-TOUCH: no contact within {dcfg.fk_touch_max * 100:.0f}cm — lifting. "
                            "lower --fk-touch-dtau or raise --fk-touch-max?")
        else:                                          # ---- keep descending straight ----
            ft.z_cur -= float(dcfg.fk_touch_step)
            tgt = torch.tensor([[ft.xy[0], ft.xy[1], ft.z_cur]], dtype=torch.float32, device=base.device)
            qf = base.ik_oriented(seed, tgt, down, iters=int(dcfg.follow_iters))[0].cpu().numpy()
            qf[5] = float(pr.arm_qpos[5])
            facade.send_arm(qf, st.grip.command, loop.loop_dt)   # J6 hold, shaped straight descent
            if ft.armed:
                status = f"dtau={dtau:.2f}/{dcfg.fk_touch_dtau}"
            else:
                status = f"arming @{FK_TOUCH_ARM_DZ*100:.0f}cm…"
            dash.update(msg=f"FK-TOUCH ↓ Δ{dz * 100:.1f}cm  {status}  z_gap_io={tick.zgap_io_val:+.4f}. Stop halts.")
    elif ft.phase == "lift":                           # return to hover, then done
        if float(tick.tcp_w[2]) < ft.z_start - 0.005:
            tgt = torch.tensor([[ft.xy[0], ft.xy[1], ft.z_start]], dtype=torch.float32, device=base.device)
            qf = base.ik_oriented(seed, tgt, down, iters=int(dcfg.follow_iters))[0].cpu().numpy()
            qf[5] = float(pr.arm_qpos[5])
            facade.send_arm(qf, st.grip.command, loop.loop_dt)
        else:
            n_samples = len(ft.samples)
            ft.phase = None
            if n_samples < 3:
                hint = " (≥3 for a fit)"
            else:
                hint = "; then relaunch with the printed --z-reach flags"
            dash.update(msg=f"FK-TOUCH lifted — {n_samples} sample(s). Move the arm to another DISTANCE + press "
                            f"FK-touch again{hint}.")


def _step_grip(loop):
    """CLOSED-LOOP force-servo: gently close until grip_tau reaches the target, then HOLD (arm frozen)."""
    st = loop.state
    tick = loop.tick
    pr = tick.pr
    st.grip.command = loop.facade.grip_force_servo(pr.arm_qpos, st.grip.command, float(pr.grip_torque),
                                                   st.grip.torque_target, loop.loop_dt)

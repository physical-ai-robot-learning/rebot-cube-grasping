"""Control loop orchestrator.

Owns the while loop (984-2074) of the old debug_deepreach.py `main()`. By design this class is kept "thin":
it holds the collaborators (base/facade/dash/perception/bb) + config-derived constants + ControlState, and only the per-tick preamble and
loop driving live here. Command routing and per-mode step bodies are split into commands.py / modes.py (avoids a god-class).

⚠️ Always async perception (sync fallback dropped). ⚠️ CNN detector dropped → wrist (u,v)=OWLv2/redmask, yaw=seg.
Collaborator/geometry/detector construction (setup) is done by main.py (P3); this class receives what is already built.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .constants import FK_BIAS_TAU, GRASP_CLEAR, YAW_LATCH_Z, Z_BIAS_MAX
from .context import ControlState
from .telemetry import build_snap
from .tick import Tick


class ControlLoop:
    def __init__(self, cfg, deploy_cfg, base, facade, dash, perception, bb, *,
                 K, T_cam2world, dist, plane_z, front_ov, wrist_ov, rest_np, jaw_fwd_fallback):
        """Inputs (all built and passed in by main.py):
          cfg          : training env config (mechanism/camera parameters; base_pos·table_z·grip_* etc.)
          deploy_cfg   : DeployConfig(CLI)
          base         : RuleBaseController (FK/DLS-Jacobian, deep_reach_step)
          facade       : RobotIoFacade (proprio read / arm command / scripted move / estop)
          dash         : DebugDashboard (telemetry + commands + camera frames)
          perception   : AsyncPerception thread (always present)
          bb           : Blackboard (for reading io-thread telemetry)
          K/T_cam2world/dist : front camera intrinsics/extrinsics/distortion
          plane_z      : cube center plane z (table_z + cube_size/2)
          front_ov/wrist_ov  : OWLv2 ROI providers (for live query editing / modes) or None
          rest_np      : REST joint array
          jaw_fwd_fallback : default used when approach lookat_jaw_fwd is None (= robot_io.JAW_FWD)
        """
        self.cfg = cfg
        self.dcfg = deploy_cfg
        self.base = base
        self.facade = facade
        self.dash = dash
        self.perception = perception
        self.bb = bb
        self.K = K
        self.T_cam2world = T_cam2world
        self.dist = dist
        self.plane_z = float(plane_z)
        self.front_ov = front_ov
        self.wrist_ov = wrist_ov
        self.rest_np = np.asarray(rest_np, dtype=float)

        # ---- Timing (always async → the control loop runs at control_hz) ----
        self.dt = 1.0 / float(deploy_cfg.control_hz)
        self.loop_dt = self.dt                 # measured loop period (updated every tick by preamble) → shaper rate-correct
        self.prev_loop_t = None

        # ---- Grasp offset / front-xy correction (setup 820-824) ----
        self.jaw_fwd = float(deploy_cfg.jaw_fwd)
        if deploy_cfg.lookat_jaw_fwd is not None:
            self.lookat_jaw_fwd = float(deploy_cfg.lookat_jaw_fwd)
        else:
            self.lookat_jaw_fwd = float(jaw_fwd_fallback)     # None → robot_io.JAW_FWD(0.06)
        self.top_bias = float(deploy_cfg.top_bias)
        self.x_corr = float(deploy_cfg.x_corr)

        # ---- auto-z / z correction state (setup 657-661) ----
        self.grasp_zgap_io = float(deploy_cfg.grasp_zgap_io)   # target z_gap_io (auto-z setpoint; changed live by the graspz command)
        self.auto_z = not deploy_cfg.no_auto_z
        self.z_bias = float(deploy_cfg.z_bias)                 # auto-z initial value / fixed depth with --no-auto-z
        self.fk_bias_ema = None                                # (robot_io z - base_ctrl z) EMA = base-FK underread
        self.base.cube_top_z = self.plane_z - self.z_bias

        # ---- FSM state init (config-derived initial values are set here; context.py defines shape only) ----
        self.state = ControlState()
        st = self.state
        st.track.enabled = bool(deploy_cfg.track)
        st.track.deadband = float(deploy_cfg.track_deadband)
        st.grip.command = cfg.grip_open_qpos                                    # default open
        st.grip.torque_target = float(np.clip(deploy_cfg.grip_torque_target, 0.3, 1.2))
        st.grip.mid = 0.5 * (cfg.grip_open_qpos + cfg.grip_close_qpos)          # force-servo start point
        st.follow.cam_fwd = float(deploy_cfg.follow_cam_fwd)
        st.follow.cam_lat = float(deploy_cfg.follow_cam_lat)
        st.follow.center_un_target = float(deploy_cfg.follow_center_un)
        st.follow.center_vn_target = float(deploy_cfg.follow_center_vn)
        st.follow.lookat = bool(deploy_cfg.follow_lookat)
        st.align.vn_target = float(deploy_cfg.align_vn_target)
        st.align.vn_band = float(deploy_cfg.align_vn_band)
        st.align.j4_gain = float(deploy_cfg.align_j4_gain)

        self.tick = None                                       # latest Tick (set by preamble)

    # ------------------------------------------------------------------------------------------------
    # Per-tick preamble: proprio → FK → z correction → perception → yaw-latch → diagnostics/3-Z → deep_reach_step → telemetry.
    #   Updates ControlState / auto-z state, fills and returns Tick (None if perception has not published).
    # ------------------------------------------------------------------------------------------------
    def preamble(self):
        """Performs this tick's observation/computation and returns Tick. (output: Tick or None=proprio not published)"""
        st = self.state
        now = time.time()
        # measured loop period (previous tick): the shaper uses real elapsed time, so speed is correct even if --control-hz is not sustained. Clamp first-tick/pause outliers.
        if self.prev_loop_t is None:
            self.loop_dt = self.dt
        else:
            self.loop_dt = float(np.clip(now - self.prev_loop_t, 0.5 * self.dt, 5.0 * self.dt))
        self.prev_loop_t = now

        pr = self.facade.read_proprio()
        if pr is None:                                  # io-thread: proprio not published (or thread dead) → caller retries
            return None
        arm_q = torch.tensor(pr.arm_qpos, dtype=torch.float32).unsqueeze(0)   # [1,6]
        tcp_b = self.base.fk(arm_q)[0]                                        # BASE frame (internal to deep_reach)
        tcp_w_t = self.base.fk_world(arm_q)[0]                               # WORLD frame N (base-z corrected)
        tcp_w = tcp_w_t.tolist()
        tcp_io = self.facade.tcp_io(pr)
        if self.dcfg.z_reach_slope != 0.0:
            # distance-dependent FK-z correction (fk_verify_z.py table-touch fit): robot_io FK z drifts with reach →
            # a constant FK_Z_CORR cannot capture it (over-descends near, under-descends far). Subtracting the fit slope from tcp_io.z here makes
            # z_gap_io (grasp gate)·fk_bias·auto-z all a consistent real height at every distance. slope 0 = no-op.
            radial = float(np.hypot(tcp_io[0] - float(self.cfg.base_pos[0]),
                                    tcp_io[1] - float(self.cfg.base_pos[1])))
            tcp_io[2] -= float(self.dcfg.z_reach_slope) * (radial - float(self.dcfg.z_reach_ref))

        # ---- AUTO-Z: cancel the (position-varying) base-FK underread using robot_io (accurate) as truth ----
        fk_bias = float(tcp_io[2] - tcp_w[2])
        alpha = 1.0 - np.exp(-self.loop_dt / FK_BIAS_TAU)     # rate-independent EMA (encoder noise does not leak through at 100Hz async)
        if self.fk_bias_ema is None:
            self.fk_bias_ema = fk_bias
        else:
            self.fk_bias_ema = (1 - alpha) * self.fk_bias_ema + alpha * fk_bias
        zgap_io_val = float(tcp_io[2] - self.plane_z)         # exact height (robot_io FK)
        if self.auto_z:
            # z_bias = GRASP_CLEAR + fk_bias - grasp_zgap_io (so that the exact z_gap_io lands on the grasp target at base-FK cube_top_z).
            # Lower bound is -Z_BIAS_MAX: the old max(0.0,…) assumed fk_bias>0 (robot_io higher than base_ctrl), but
            # after the FK_Z_CORR refit (2026-07-22, robot_io made accurate) fk_bias can become negative, so z_bias must be negative too (raising cube_top_z
            # above plane_z to cancel base_ctrl's over-read). With the negative clamp, z_bias was stuck at 0 and Follow hover sat ~4cm
            # too low (near the table). Negative z_bias = higher target = opposite of over-plunge, so it is safe; the upper Z_BIAS_MAX still prevents over-plunge.
            self.z_bias = min(Z_BIAS_MAX, max(-Z_BIAS_MAX, GRASP_CLEAR + self.fk_bias_ema - self.grasp_zgap_io))
            self.base.cube_top_z = self.plane_z - self.z_bias  # reset the descent target every loop (auto correction)

        # ---- PERCEPTION (always an async thread): publish FK TCP (front-Z prior) + gated align_seg, read snapshot ----
        tcp_world_np = np.asarray(tcp_w, float)
        # align_seg (heavy) is enabled only in Align, RUN (the seg yaw-latch consumes it), and AUTO-GRASP. (yaw_src is always seg → simplified to run_active)
        need_align = st.align.active or st.run_active or (st.autograsp.phase is not None)
        self.perception.set_ctx(tcp_world_np, need_align=need_align)
        perc, cube_world, cube_raw = self.perception.snapshot()   # cube_world = owned by the thread (capture result)
        if st.track.enabled:
            # live tracking: cube_world = live front estimate that passed the deadband (no re-Capture needed, follows a moving cube).
            cube_live = perc.get("cube_live")
            if cube_live is not None:
                if st.track.anchor is None:
                    st.track.anchor = list(cube_live)             # auto-acquire (no Capture needed)
                elif ((cube_live[0] - st.track.anchor[0]) ** 2
                      + (cube_live[1] - st.track.anchor[1]) ** 2) ** 0.5 > st.track.deadband:
                    st.track.anchor = list(cube_live)             # real movement → follow; small jitter → ignore
            if st.track.anchor is not None:
                cube_world = list(st.track.anchor)
        # FAIL-STOP: if the perception thread is dead/STALE, forbid autonomous servoing on a frozen target → stop RUN/approach + freeze
        perc_age = time.time() - perc.get("ts", 0.0)
        perc_stale = (not self.perception.is_alive()) or bool(perc.get("dead")) or perc_age > 2.0
        if perc_stale:
            if st.run_active or st.align.active or st.approach.active:
                st.run_active = False
                st.align.active = False
                st.approach.active = False
                self.facade.send_arm(pr.arm_qpos, st.grip.command, self.loop_dt, snap=True)   # freeze at the current pose
            self.dash.update(msg=f"PERCEPTION STALE ({perc_age:.1f}s old) -> auto/align/approach halted. "
                                 "Recovers on its own (startup's first frame ~3.5s; steady ~30Hz).")
            st.perception_was_stale = True
        elif st.perception_was_stale:                         # recovered → clear the lingering STALE msg
            self.dash.update(msg=f"perception OK ({perc.get('hz', 0):.0f}Hz) — recovered.")
            st.perception_was_stale = False

        # perception fields (CNN dropped: wcos4/wsin4/wdepth are not read)
        wun, wvn, wvis = perc["wun"], perc["wvn"], perc["wvis"]               # wrist (u,v) = OWLv2/redmask
        scos4, ssin4, sang = perc["scos4"], perc["ssin4"], perc["sang"]      # seg yaw (edge-voting/redmask)
        sun, svn, svis = perc["sun"], perc["svn"], perc["svis"]              # seg centroid (Align centering)
        stopvn = perc["stopvn"]                                              # seg cube TOP-edge y (Align auto-center)
        gap_front = perc["gap_front"]
        has_front = perc["has_front"]

        # ---- YAW LATCH: freeze the wrist yaw while RUN descends through the trusted z-band (for close Align). seg edge-voting only.
        #   Accumulate 1 step per NEW perception frame (not per control tick; prevents the async 100Hz loop from saturating ~30Hz frames 4x = ts-gate).
        yaw_ts = perc.get("ts")
        yl = st.yaw_latch
        if (st.run_active and yl.weight < 1.0
                and (YAW_LATCH_Z[0] <= zgap_io_val <= YAW_LATCH_Z[1]) and yaw_ts != yl.ts):
            if svis > 0.5:
                yl.ts = yaw_ts                                # consume this perception frame
                if yl.weight <= 0.0:
                    yl.cos4, yl.sin4 = scos4, ssin4           # seed (avoids EMA-warmup bias)
                else:
                    blend = 0.3
                    yl.cos4 = (1 - blend) * yl.cos4 + blend * scos4
                    yl.sin4 = (1 - blend) * yl.sin4 + blend * ssin4
                yl.weight = min(1.0, yl.weight + 0.34)        # FULL after ~3 in-band frames → freeze

        # diagnostic wrist eye-in-hand world-xy (angle-invariant above the cube) vs front ray-plane xy → check front bias
        wrist_xy = (self.base._wrist_uv_to_world(arm_q, torch.tensor([wun]), torch.tensor([wvn]),
                                                 self.plane_z)[0].tolist() if wvis > 0.5 else None)
        # 3-SOURCE Z fuse: median of the usable gaps {front-gripper, base-z FK} (wrist depth was display-only → dropped).
        gap_fk = float(tcp_world_np[2] - self.plane_z)
        available_gaps = [g for g in (gap_front, gap_fk) if g is not None]
        gap_med = float(np.median(available_gaps)) if available_gaps else None
        at_cube_robust = bool(gap_med is not None and gap_med < 0.01)

        tick = Tick(loop_dt=self.loop_dt, pr=pr, arm_q=arm_q, tcp_b=tcp_b, tcp_w=tcp_w, tcp_w_t=tcp_w_t,
                    tcp_io=tcp_io, zgap_io_val=zgap_io_val, perc=perc, perc_stale=perc_stale,
                    cube_world=cube_world, cube_raw=cube_raw,
                    wun=wun, wvn=wvn, wvis=wvis, scos4=scos4, ssin4=ssin4, sang=sang, sun=sun, svn=svn, svis=svis,
                    stopvn=stopvn, gap_front=gap_front, has_front=has_front, wrist_xy=wrist_xy,
                    gap_fk=gap_fk, gap_med=gap_med, at_cube_robust=at_cube_robust)

        # ---- deep_reach_step: if there is a cube, XY servo command + grasp target/reach gate ----
        if cube_world is not None:
            cube_tx = cube_world[0] + self.dcfg.cube_x_off    # detected cube + world XY bias offset (both default 0)
            cube_ty = cube_world[1] + self.dcfg.cube_y_off
            cw_xy = torch.tensor([[cube_tx, cube_ty]], dtype=torch.float32)
            # JAW-FWD: the real jaw closes ~jaw_fwd ahead of the base-FK finger-midpoint along the gripper-forward axis (J1 rotation) →
            #   deep_reach_step shifts the TCP target back along that rotating axis. The grasp gate uses the same axis (base.fwd_axis_xy).
            ghat = self.base.fwd_axis_xy(arm_q)[0].cpu().tolist()   # world gripper-forward unit xy (JSON-safe floats)
            # RUN descends straight with J6 FREE; jaw yaw is aligned in the POST-RUN 'Align' (J4 nudge + J6 roll) (aligning during descent is unstable on HW).
            q_cmd, at_cube = self.base.deep_reach_step(
                arm_q, cw_xy, torch.tensor([wun]), torch.tensor([wvn]), torch.tensor([wvis]),
                jaw_fwd=self.jaw_fwd)
            gtx = cube_tx + self.jaw_fwd * ghat[0]            # jaw-fwd (rotating) grasp target
            gty = cube_ty + self.jaw_fwd * ghat[1]
            xy_err_val = float(np.linalg.norm(np.array(tcp_w[:2]) - np.array([gtx, gty])))
            # 'reached' = exact z_gap_io (target) + xy on the grasp target. Not base_ctrl at_cube (fires too high before the auto-z EMA catches up).
            reached_run = abs(zgap_io_val - self.grasp_zgap_io) < 0.015 and xy_err_val < self.base.reach_eps
            tick.cube_tx, tick.cube_ty, tick.ghat = cube_tx, cube_ty, ghat
            tick.q_cmd, tick.at_cube = q_cmd, bool(at_cube[0])
            tick.gtx, tick.gty, tick.xy_err_val, tick.reached_run = gtx, gty, xy_err_val, reached_run

        tick.snap = build_snap(self, tick)
        self.dash.update(**tick.snap)
        self.tick = tick
        return tick

    def return_to_rest(self, msg, halt=False):
        """Shared 'LIFT to REST + reinitialize approach' recovery. (input: msg dashboard message, halt=True stops approach)
        The three triggers OV object-change / cube deviation / kinematic danger all use this one place.
        ⚠️ facade.run_scripted('initial_liftup') is script-blocking ~2s (same as the Reset button) — STOP/ABORT/
        perc-stale cannot be serviced meanwhile, but it is allowed since it moves to a safe pose. With halt=False approach stays on so
        the next tick descends fresh from above."""
        st = self.state
        self.facade.run_scripted("initial_liftup", st.grip.command)   # → REST hover, gripper unchanged (blocking; io-thread=io_task)
        self.base.reset()                                             # clear reach + descend latches
        self.facade.reset_shaper(self.rest_np)                        # re-sync the shaper to the reached REST pose
        st.approach.target_xy = None
        st.approach.height = None
        st.approach.wrist_lock_xy = None                              # descend fresh from above on the next approach tick
        st.approach.risen = False                                     # rise to hover again before descending
        if st.track.enabled:
            st.track.anchor = None                                   # re-acquire the cube live from above (discard the old anchor)
        if halt:
            st.approach.active = False
        self.dash.update(msg=msg)

    def run(self):
        """Control loop body. Per tick: preamble (observe/compute) → AUTO-GRASP sequencer → route the command if any (skip modes that tick),
        otherwise mode step → sleep. On exit, clean up facade.shutdown (motor estop)/perception/dash."""
        from . import commands, modes                  # deferred import (readability; not circular — modes/commands do not import control_loop)
        print(f"[deploy] control loop up @ {1.0/self.dt:.0f}Hz. z_bias={self.z_bias:.3f} (auto-z {self.auto_z}). "
              "GRASP FLOW: Capture → RUN/Follow → (AUTO-GRASP or Align→Final) → Grip → Lift.", flush=True)
        try:
            while True:
                tick = self.preamble()
                if tick is None:                        # proprio not published (io-thread starting/dead) → retry shortly
                    time.sleep(0.005)
                    continue
                modes.step_autograsp_sequencer(self)    # phase transitions/abort (before the command poll, same order as the monolith)
                cmd = self.dash.poll_cmd()
                if cmd:                                 # ticks that received a command skip mode execution (preserves the monolith elif structure)
                    if commands.handle(self, cmd):      # abort → True → exit loop
                        break
                else:
                    modes.step(self)
                # wait only dt from this tick's start (recorded by preamble as self.prev_loop_t) → consistent with the loop_dt basis
                time.sleep(max(0.0, self.dt - (time.time() - self.prev_loop_t)))
        finally:
            self.facade.shutdown()                      # motor estop FIRST (so safety does not wait on the camera join)
            self.perception.stop()
            self.dash.stop()
            print("[deploy] done (torque off).", flush=True)

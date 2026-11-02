"""Dashboard telemetry (snap dict) builder.

The inline snap dict (1127-1198) of the old debug_deepreach.py was split into one function (separating control logic from dashboard
format concerns → easier review). Called once after the preamble has filled Tick, to build the complete dict
(merges the monolith's two-stage build-then-update into one stage; the final dict is identical).

⚠️ Keys removed because the CNN was dropped: cnn_yaw_deg (CNN yaw head), yaw_src (always seg), z_wrist_depth (CNN depth).
   wrist yaw = seg (edge-voting) only.
"""
from __future__ import annotations

import time

import numpy as np


def build_snap(loop, tick) -> dict:
    """Builds the dashboard telemetry dict from the control loop (loop) and this tick (tick).
    (input: loop=ControlLoop, tick=Tick → output: dict to pass to dash.update(**snap))"""
    state = loop.state
    pr = tick.pr
    perc = tick.perc or {}

    seg_visible = tick.svis > 0.5
    yaw_latch = state.yaw_latch

    gates = {
        "z_bias": round(loop.z_bias, 3),                              # descent depth (auto or fixed)
        "align_top_vn": (round(tick.stopvn, 2) if seg_visible else None),      # cube TOP-edge y (Align)
        "align_top_vn_goal": round(state.align.vn_target, 2),         # Align framing target (~bottom 40%)
        # J6-align yaw: seg (edge-voting) and latch (freeze) — trust by comparing with the visible cube rotation
        "seg_yaw_deg": (round(float(np.degrees(np.arctan2(tick.ssin4, tick.scos4) / 4.0)), 1)
                        if seg_visible else None),
        "latch_yaw_deg": (round(float(np.degrees(np.arctan2(yaw_latch.sin4, yaw_latch.cos4) / 4.0)), 1)
                          if yaw_latch.weight > 0.3 else None),        # FROZEN mid-descend yaw (used by Align)
        "auto_z": loop.auto_z,                                        # z_gap_io closed loop?
        "fk_bias": round(loop.fk_bias_ema, 3) if loop.fk_bias_ema is not None else None,  # robot_io_z - base_ctrl_z
        "grasp_zgap_io": round(loop.grasp_zgap_io, 3),                # target z_gap_io (VERIFY + auto-z setpoint)
        "cube_raw": tick.cube_raw,                                    # RAW front estimate (before correction)
        "wrist_xy": ([round(tick.wrist_xy[0], 3), round(tick.wrist_xy[1], 3)] if tick.wrist_xy else None),
        "wf_dxy": (round(float(np.linalg.norm(np.array(tick.wrist_xy) - np.array(tick.cube_world[:2]))), 3)
                   if (tick.wrist_xy and tick.cube_world is not None) else None),  # wrist↔front xy gap = front bias
        "z_gap_io": round(tick.tcp_io[2] - loop.plane_z, 3),          # robot_io FK height above the cube
        "z_gap_front": round(tick.gap_front, 3) if tick.gap_front is not None else None,   # (1) front
        "z_gap_fk": round(tick.gap_fk, 3),                            # (3) base-z corrected FK gap
        "z_gap_med": round(tick.gap_med, 3) if tick.gap_med is not None else None,   # fused (median)
        "at_cube_robust": tick.at_cube_robust,                        # robust gap<1cm (display only; does not drive)
        "grip_tau": round(float(getattr(pr, "grip_torque", 0.0)), 3),  # gripper motor torque
        "grip_tau_target": (round(state.grip.torque_target, 2) if state.grip.servoing else None),  # force-servo setpoint
        "grasped": loop.facade.is_grasped(pr.grip_qpos),              # torque>0.5 → cube grasped
        "wrist_yaw_deg": (round(tick.sang, 1) if seg_visible else None),   # seg in-image yaw (J6-align signal)
        "wrist_seg_uv": ([round(tick.sun, 2), round(tick.svn, 2)] if seg_visible else None),  # seg centroid (centering)
        "perc_hz": round(float(perc.get("hz", 0.0)), 1),              # perception thread rate = real health
        "perc_age_ms": round((time.time() - perc.get("ts", 0.0)) * 1e3),  # snapshot freshness (>2000ms=STALE)
        "ov_q": perc.get("ov_q"),                                     # active open-vocab query (edited from the dashboard)
        "ov_score": perc.get("ov_score"),                             # best front OV box score (None=not detected)
        "ov_q_wrist": perc.get("ov_q_wrist"),                         # wrist cam's own OV query
        "ov_score_wrist": perc.get("ov_score_wrist"),                 # wrist cam's own best box score
        "fk_touch": state.fk_touch.phase,                             # FK-Z verification phase (None/descend/lift)
        "fk_touch_samples": [[round(r, 4), round(io_z, 4), round(fb, 4)]
                             for (r, io_z, fb) in state.fk_touch.samples],  # record of each table touch
        "io": (loop.bb.debug.get("io") if loop.facade.uses_io_thread else None),  # RobotIoThread bus_hz/last_seq
        "tm": perc.get("tm"),                                         # per-stage perception ms (30Hz diagnostics)
    }

    snap = {
        "phase": "DEEP-REACH",
        "step": state.step,
        "grip_q": float(np.mean(pr.grip_qpos)),
        "joints": {"q": list(map(float, pr.arm_qpos)),
                   "tau": list(map(float, getattr(pr, "arm_torque", [0] * 6)))},
        "tcp": tick.tcp_w,
        "tcp_io": tick.tcp_io,
        "wrist_uv": [round(tick.wun, 2), round(tick.wvn, 2), tick.wvis],
        "gates": gates,
    }

    if tick.cube_world is not None:
        # ticks with a deep_reach_step result: add grasp target/error/reach gate (the monolith's snap.update).
        snap["cube_top"] = [round(tick.cube_world[0], 3), round(tick.cube_world[1], 3)]
        snap["grasp_xy"] = [round(tick.gtx, 3), round(tick.gty, 3)]           # jaw-fwd corrected target (what the TCP aims at)
        snap["xy_err"] = round(tick.xy_err_val, 3)                            # tcp ↔ grasp target
        snap["z_gap"] = round(float(tick.tcp_w[2] - loop.plane_z), 3)         # base_ctrl tcp - cube center (FK shadow)
        snap["at_cube"] = bool(tick.at_cube)                                  # base_ctrl gate (display)
        snap["reached"] = tick.reached_run                                    # exact reach (drives RUN stop)

    return snap

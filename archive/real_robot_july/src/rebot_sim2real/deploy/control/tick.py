"""Tick: data bag of values the control loop recomputes on every iteration (tick).

preamble() reads proprio·FK·perception to fill this object; command handlers/mode steps read it.
Distinct from ControlState (persistent state): Tick is "computed results valid only for this tick" (recreated every tick).

⚠️ Because the CNN was dropped, wcos4/wsin4 (CNN yaw head) and wdepth/z_wrist_depth (CNN depth) are not stored.
   wrist yaw is seg (scos4/ssin4/sang) only; wrist (u,v)=OWLv2/redmask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Tick:
    loop_dt: float = 0.0            # measured previous loop period (s) — shaper speed is rate-correct
    pr: Any = None                  # proprio (arm/grip qpos·qvel·torque; includes tcp_io/is_grasped in io-thread mode)
    arm_q: Any = None               # current arm joints [1,6] tensor
    tcp_b: Any = None               # BASE-frame TCP [3] tensor (internal to deep_reach)
    tcp_w: Any = None               # WORLD-frame N TCP [3] list (base-z corrected base_ctrl FK)
    tcp_w_t: Any = None             # tensor version of the above
    tcp_io: Any = None              # robot_io FK grasp point [3] list (reach-slope correction applied)
    zgap_io_val: float = 0.0        # height above the cube center per robot_io FK = tcp_io.z - plane_z (exact height)

    perc: Any = None                # perception thread snapshot dict (wun/wvn/... + ts/hz/dead/tm)
    perc_stale: bool = False        # perception thread dead/STALE (>2s) — read by the autograsp sequencer for the abort decision
    cube_world: Any = None          # grasp-target cube world (x,y[,z]) or None (capture or track anchor)
    cube_raw: Any = None            # RAW front estimate (before correction) — for dashboard display

    wun: float = 0.0                # wrist (u,v) normalized coordinates + visibility (OWLv2/redmask)
    wvn: float = 0.0
    wvis: float = 0.0
    scos4: float = 0.0              # seg yaw vector (cos4θ,sin4θ) + in-image angle (edge-voting/redmask)
    ssin4: float = 0.0
    sang: float = float("nan")
    sun: float = 0.0                # seg centroid (u,v) + visibility (Align centering)
    svn: float = 0.0
    svis: float = 0.0
    stopvn: float = 0.0             # seg cube TOP-edge normalized y (Align auto-center target)
    gap_front: Any = None           # front-gripper geometric Z gap (m) or None
    has_front: bool = False         # front frame presence sentinel (the frame is owned by the thread)

    wrist_xy: Any = None            # wrist (u,v)→world unprojected xy (angle-invariant diagnostic above the cube)
    gap_fk: float = 0.0             # base-z corrected FK gap = tcp_w.z - plane_z
    gap_med: Any = None             # median of the usable {front, fk} gaps (3-source Z fuse)
    at_cube_robust: bool = False    # gap_med < 1cm (robust reach gate; display only, does not drive)

    q_cmd: Any = None               # command joints [1,6] produced by deep_reach_step, or None (no cube)
    at_cube: bool = False           # base_ctrl deep_reach gate (display only)
    reached_run: bool = False       # exact reach (z_gap_io target + xy on grasp target) → drives RUN stop
    cube_tx: Any = None             # detected cube + world xy offset (deep_reach servo target)
    cube_ty: Any = None
    ghat: Any = None                # world gripper-forward unit xy (rotates with J1; jaw-fwd direction)
    gtx: Any = None                 # jaw-fwd corrected grasp target xy
    gty: Any = None
    xy_err_val: Any = None          # tcp ↔ grasp target xy distance (m)

    snap: Any = None                # dashboard telemetry dict (build_snap result)

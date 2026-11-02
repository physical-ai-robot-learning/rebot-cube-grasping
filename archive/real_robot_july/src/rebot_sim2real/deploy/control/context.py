"""Control loop FSM state (collection of data-only dataclasses).

The FSM state scattered across ~70 `nonlocal` locals inside the old debug_deepreach.py `main()`
is grouped into small per-concern dataclasses (item 12: cohesive pieces instead of one giant state object).
Each mode (Follow/Align/Approach/AutoGrasp/FkTouch/Lift) actually has its own independent lifecycle and state,
so the split is justified.

⚠️ Most defaults here are "reset/neutral values" (None/False/0). Initial values that come from config (cfg / DeployConfig)
   (e.g. grip_torque_target, follow_cam_fwd, align_vn_target, track_db) are set during the control setup stage
   — this file only defines the "shape" of the state.
⚠️ Because the CNN detector was dropped, these old states were removed: yaw_src (always 'seg' edge-voting), lookat_vn / ibvs_pj4 /
   ibvs_pwvn / ibvs_wf / ibvs_J (all were only for --lookat-ibvs = sd4 CNN bearing servo).

--- old monolith variables → new field mapping (P2.3 porting reference) ---
  cube_world→tick.cube_world(per tick, not persisted) · track_on→track.enabled · track_anchor→track.anchor · track_db→track.deadband
  grip_cmd→grip.command · gripping→grip.servoing · grip_torque_target→grip.torque_target · grip_mid→grip.mid
  following→follow.active · follow_tgt→follow.target_xy · follow_front→follow.front_ref_xy · follow_z_cur→follow.height_above_cube
  follow_cam_fwd/lat→follow.cam_fwd/cam_lat · last_ac_ts→follow.auto_center_ts · ac_centered→follow.auto_center_latched
  ac_un_tgt/ac_vn_tgt→follow.center_un_target/center_vn_target · follow_lookat→follow.lookat · last_lk_ts→follow.lookat_ts
  last_lookat_qf→follow.lookat_last_q
  aligning→align.active · align_vn_target/band/j4_gain→align.vn_target/vn_band/j4_gain · align_dwell→align.dwell · jaw_sign→align.jaw_sign
  yaw_lat_c/s/w→yaw_latch.cos4/sin4/weight · last_yaw_ts→yaw_latch.ts
  auto_running→run_active(ControlState) · autograsp→autograsp.phase · ag_t0→autograsp.phase_t0 · ag_dwell→autograsp.settle_count
  ag_yaw_q6→autograsp.j6_target · ag_ready_t0→autograsp.ready_t0 · ag_xy→autograsp.committed_xy
  approaching→approach.active · approach_tgt→approach.target_xy · approach_z→approach.height · wrist_lock→approach.wrist_lock_xy
  approach_risen→approach.risen · danger_count→approach.danger_count
  fk_touch→fk_touch.phase · fk_touch_xy→fk_touch.xy · fk_touch_z0→fk_touch.z_start · fk_touch_z_cur→fk_touch.z_cur
  fk_touch_tau0→fk_touch.tau_baseline · fk_touch_armed→fk_touch.armed · fk_touch_samples→fk_touch.samples
  lifting→lift.active · lift_target_z→lift.target_z · lift_cur_z→lift.cur_z · lift_xy→lift.xy
  step→ControlState.step · perc_was_stale→ControlState.perception_was_stale (prev_front removed: sync-only dead)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GripState:
    """Gripper state. The close/open buttons set command; 'close' is a closed-loop force-servo (gently closes to torque_target and holds)."""
    command: float = 0.0        # current gripper setpoint (qpos). initial=cfg.grip_open_qpos
    servoing: bool = False      # True=closed; one force-servo step per idle loop (creep→hold)
    torque_target: float = 0.8  # force-servo target torque (clamped 0.3~1.2). initial=DeployConfig.grip_torque_target
    mid: float = 0.0            # force-servo start point (opening just before contact). initial=0.5*(open+close)


@dataclass
class TrackState:
    """Front-camera live cube tracking (deadband). The grasp-target cube coordinates themselves flow per tick via tick.cube_world
    (from the perception snapshot or anchor) — not persisted here (same as the monolith async behavior)."""
    enabled: bool = False       # live front tracking on (deadband) / off=one-shot Capture. initial=DeployConfig.track
    anchor: Any = None          # live cube xy that passed the deadband (updated only on real >deadband movement). consumed as tick.cube_world
    deadband: float = 0.015     # live tracking update threshold (m). initial=DeployConfig.track_deadband


@dataclass
class FollowState:
    """'Follow' hover-track mode: straight-down position servo + wrist closed-loop auto-center (measured Jacobian) or gimbal look-at."""
    active: bool = False                 # Follow mode on
    target_xy: Any = None                # tracked xy that passed the deadband = actual setpoint (wrist-owned while centering)
    front_ref_xy: Any = None             # front-glued xy reference used only for the RISE decision (separate from the wrist-owned target → so front-vs-wrist bias does not cause a fake rise)
    height_above_cube: Any = None        # live descent height: when STABLE, ramp hover(follow_z)→follow_low; rise to hover on movement
    cam_fwd: float = 0.03                # deterministic centering offset (gripper-forward axis). initial=DeployConfig.follow_cam_fwd, live-tuned
    cam_lat: float = -0.02               # deterministic centering offset (gripper-lateral axis). initial=DeployConfig.follow_cam_lat
    auto_center_ts: Any = None           # auto-center: consumed perception ts (updated only on FRESH frames → prevents 30Hz over-correction)
    auto_center_latched: bool = False    # auto-center hysteresis latch: stop once centered (|uv|<margin), do not resume until real drift (>2*margin)
    center_un_target: float = 0.0        # auto-center target u. initial=DeployConfig.follow_center_un
    center_vn_target: float = 0.44       # auto-center target v (>0 = cube slightly below toward the jaw = grasp framing). initial=DeployConfig.follow_center_vn
    lookat: bool = False                 # gimbal look-at lock (live toggle). initial=DeployConfig.follow_lookat
    lookat_ts: Any = None                # look-at: ts-gate the gimbal aim to FRESH frames (prevents re-applying stale ones)
    lookat_last_q: Any = None            # look-at: last IK q held between frames


@dataclass
class AlignState:
    """POST-RUN 'Align': latched auto-center (J4)-then-J6-roll. Frames the cube TOP-edge with J4, then J6 roll by seg yaw."""
    active: bool = False        # Align mode on
    vn_target: float = 0.2      # target cube TOP-edge normalized v (-1=top..+1=bottom). initial=DeployConfig.align_vn_target
    vn_band: float = 0.10       # in-band tolerance → J4 stop, J6 roll. initial=DeployConfig.align_vn_band
    j4_gain: float = -0.6       # J4 servo gain (rad per vn-error). initial=DeployConfig.align_j4_gain
    dwell: int = 0              # J4-servo loop counter (forced J6 roll after ALIGN_DWELL_MAX, framing stall guard)
    jaw_sign: float = -1.0      # rig-dependent J6-roll direction (live-inverted via 'Align flip'). ⭐2026-07-23 HW-verified: -1 is
    #                             correct (with the opposite sign the residual = 2·theta mod90 → only 45° cubes align by chance, others are off, docs/33)


@dataclass
class YawLatchState:
    """During RUN descent, latch/freeze the wrist yaw in the trustworthy z band (YAW_LATCH_Z) → consumed at the close Align pose.
    The yaw source is always seg edge-voting (CNN yaw dropped)."""
    cos4: float = 0.0    # latched yaw vector cos(4θ)
    sin4: float = 0.0    # latched yaw vector sin(4θ)
    weight: float = 0.0  # latch weight (0=empty → Align falls back to live seg)
    ts: Any = None       # perception ts of the last latch accumulation (async 30Hz gate)


@dataclass
class RunAutoGraspState:
    """AUTO-GRASP one-shot sequencer: align (J6 yaw at hover + simultaneous auto-center) → descend (pure straight descent) → grasp (force-servo close)."""
    phase: Any = None        # None | 'align' | 'descend' | 'grasp'
    phase_t0: Any = None     # start time of the current phase (timeout decision)
    settle_count: int = 0    # grasp settle counter
    j6_target: Any = None    # --autograsp-yaw: J6 target latched once in align from the temporally stable seg yaw (unstable→J6=0)
    ready_t0: Any = None     # align: time when the (yaw+center+J5-ok) ready condition started → dwell-stable gate
    committed_xy: Any = None # grasp xy FROZEN at descend tick1 → pure vertical descent (no fwd creep)
    align_yaw_cos: float = 0.0   # align seg-yaw EMA (cos4θ) — temporal stability gate (docs/33)
    align_yaw_sin: float = 0.0   # align seg-yaw EMA(sin4θ)
    align_yaw_n: int = 0         # number of accumulated fresh seg frames
    align_yaw_ts: Any = None     # last accumulated perception ts (30Hz fresh-frame gate)


@dataclass
class ApproachState:
    """'approach' LOOK-AT mode: descend while aiming with a feedforward wrist look-at."""
    active: bool = False       # approach mode on
    target_xy: Any = None      # cube xy that passed the deadband (front + wrist-refine)
    height: Any = None         # live descent height
    wrist_lock_xy: Any = None  # HOLD-ON-LOSS: last wrist-refined cube xy (no snap-back to the coarse front even if the wrist loses it)
    risen: bool = False        # rise-first: descend only after physically finishing the rise to hover
    danger_count: int = 0      # DEBOUNCE of consecutive kinematic-danger ticks (no abort from a single-tick IK transient)


@dataclass
class FkTouchState:
    """Semi-automatic TABLE-TOUCH FK-Z verification: straight descent → J2/J3 torque contact → record z_gap_io → rise. For the reach-slope fit."""
    phase: Any = None          # None | 'descend' | 'lift'
    xy: Any = None             # touch point xy (fixed for vertical descent)
    z_start: Any = None        # TCP z at descent start
    z_cur: Any = None          # current descent target z
    tau_baseline: Any = None   # reference torque for contact detection (re-baselined after descending FK_TOUCH_ARM_DZ)
    armed: bool = False        # contact detection armed only after passing FK_TOUCH_ARM_DZ (prevents descent-start transient false-fires)
    samples: list = field(default_factory=list)  # [(radial, z_gap_io, fk_bias)] recorded at each contact → reach-slope fit


@dataclass
class LiftState:
    """'Lift +z' = gentle MODE: ramp TCP-z at lift_rate up to target_z, then hold (vertical rise)."""
    active: bool = False    # Lift mode on
    target_z: Any = None    # target TCP z (capped at LIFT_Z_MAX)
    cur_z: Any = None        # z target being ramped
    xy: Any = None          # fixed xy (straight up)


@dataclass
class ControlState:
    """Top-level collection of all FSM mode states. The control loop/command handlers/mode steps share this one instance.
    Simple flags such as run_active (RUN) live here directly; compound modes are delegated to sub-dataclasses.
    Modes are kept mutually exclusive by command handlers turning each other off (same as the monolith)."""
    step: int = 0                       # control loop tick counter
    run_active: bool = False            # RUN (one-shot): deep_reach_step servo every tick until at_cube
    perception_was_stale: bool = False  # track perception STALE state → clear msg on recovery
    # (prev_front removed: it was sync-only — in async-only mode the front frame-difference Z is owned by the perception thread itself)

    grip: GripState = field(default_factory=GripState)
    track: TrackState = field(default_factory=TrackState)
    follow: FollowState = field(default_factory=FollowState)
    align: AlignState = field(default_factory=AlignState)
    yaw_latch: YawLatchState = field(default_factory=YawLatchState)
    autograsp: RunAutoGraspState = field(default_factory=RunAutoGraspState)
    approach: ApproachState = field(default_factory=ApproachState)
    fk_touch: FkTouchState = field(default_factory=FkTouchState)
    lift: LiftState = field(default_factory=LiftState)

# Handoff — deploy grasp bring-up: gripper force-safety + Align auto-center + wrist yaw-latch (2026-07-09, session 7)

> Follows docs/16 (session6, perception-topwrist-coop). This session **built up the full grasp flow with the real-robot deploy tool (`deploy/debug_deepreach.py`)**: prevent gripper over-torque breakage → closed-loop force-servo → from-scratch calibration → **Align auto-center (J4) + wrist yaw-latch (J6)** → speed/Lift tuning → dashboard/code refactoring. Confirmed on the real robot working through Align (user: "it works well"). Handoff.
>
> **One line**: Got past the two walls of real-robot grasping — (1) the gripper broke twice from over-torque → hard cap 0.18 + `grip_force_servo` (auto-stop at target torque); (2) at the close top-down pose the wrist yaw (both seg minArea and CNN) collapses → **latch and freeze the yaw at the reliable mid-range (RUN descent z∈[0.05,0.20]) and consume it in the close Align**. Final flow: **Capture → RUN → Align (J4 center + J6 yaw-latch) → Final → Grip close (force-servo) → Lift**.

---

## 0. What is in place now
- **Branch `perception-topwrist-coop`**. Most of this deploy work is committed (user pushed). **Warning: only `deploy/debug_deepreach.py` is uncommitted (+71/−18 = latest yaw-latch, auto-center Align, CNN-yaw) → must be committed.** The other deploy files (`robot_io.py`, `calib_grip.py`, `debug_dashboard.py`, `cube_detector_io.py`) are clean (committed). Lineage: `16a2659` (align-latch/CNN-yaw-head/force-servo, previous session), `b79104d` (grip-force safety).
- **Unrelated uncommitted (do not touch)**: `src/` (pick_env/perception/perception_fusion/base_controller/config, test), `scripts/train.py`, `deploy/deep_reach_world.py` (+docs/17), `configs/train_v6c_hover07e.yaml` — these belong to the separate perception-topwrist-coop / v6c / deep_reach_world workstreams.
- **Real-robot verification status**: cameras, Capture, RUN (descent, speed), **J4 auto-center, J6 yaw-latch alignment = confirmed working**. The full Final→Grip close (force-servo)→Lift cycle is not verified (next step).

## 1. Gripper over-torque (broke twice) — hard cap + force-servo
- grip = **FORCE_POS** (position + current/torque cap). `robot.motors["gripper"].send_force_pos(pos,vel,ratio)`, the arm uses `send_pos_vel`. `grip_tau≈1.86` is the stall torque **read** when crushing (not a setpoint).
- **Broke twice**: lowered from 0.3 → 0.15, then raised above 0.15 again and it broke. → `robot_io.GRIP_FORCE_RATIO_MAX=0.18` **hard clamp** (whatever anyone passes is clamped + warned), default `grip_force_ratio=0.10`.
- **Rigid cube finding**: torque is a **vertical wall with respect to position** (grip_tau 0.13→1.35 within ~3° of motor travel). I.e. force is determined by **ratio**, not position → `calib_grip.py` A3 is a ratio↔stall sweep.
- **FIX = `robot_io.grip_force_servo(grip_cmd, grip_torque, target)`**: close gently until the measured torque reaches target (default 0.8, MAX 1.2 ≪1.86) and **stop and hold**. debug `close` = start servo, `grip_tight/loose` = target torque ±0.1. Triple safety (servo stop + position floor + ratio hard cap). Verified offline.

## 2. `deploy/calib_grip.py` (new, from-scratch consolidation)
Merges the old 4 (`calib_gripper/interactive/close`, `grip_check`) into one (→`_archive/deploy/`). 3 stages: **(1) OPEN** (no cube, max without strain = sane open) **(2) CLOSED** (no cube, torque spike = firm close) → joint_map, **(3) FORCE (cube)**: live sweep of `config.force_pos_torque_ration` (dataclass) → recommends the ratio that gives the target stall. `--open-max` (30; fixed a bug where A3 could not open if the blob was cut off, so A3 opens first), `--force-step`, ABORT_TAU 1.2, per-stage backoff.

## 3. (key) Align — auto-center (J4) + wrist yaw-latch (J6)  [core of this session]
**Problem**: at the close top-down alignment pose the wrist yaw **collapses for both** — (a) seg `cv2.minAreaRect`: the cube exceeds the near window (fixed 0.22·1280≈282px) up close → fits the window → **θ≈0/90°** (misjudges a rotated cube as "aligned"); (b) CNN: close/blur is OOD so vis<thresh, not detected. → J6 rotates 0°, "not aligned at all" (user: actually ~25° CW was needed).
**Design (multi-angle workflow analysis)**: give up on reading up close; **freeze it in from the reliable mid-range.**
- **J4 auto-center** (joint space): servo the cube top edge to `--align-vn-target 0.2` (bottom ~40% framing). `--align-j4-gain **−0.6**` (direction measured on the rig; flip sign if reversed), dwell timeout (80 loops) guarantees the J6 stage. top edge = min-y of the `wrist_seg` box (`stopvn`, async/sync wiring).
- **yaw-latch**: during RUN descent at `z_gap_io∈[0.05,0.20]` (YAW_LATCH_Z), the yaw vector (cos4θ,sin4θ) is **seeded from the first → EMA**, and once `yaw_lat_w` reaches 1.0 it **freezes** (prevents contamination from the FINAL re-descent). Align J6 consumes in order **latch → live CNN → live seg**. Source = `--yaw-src` (**default seg** = ckpt-independent, mid-range minArea is trustworthy; `cnn`+sd4 is opt-in). Changed the `wrist_seg` near window to **scale with blob size** (`1.3·√area`) to isolate the top face at mid-range.
- **Final**: resume deep_reach (auto-z) to descend to the grasp, keeping J6 alignment (base_j6_servo off and wrist_cos4 not passed, so J6 is fixed during descent = latch stays valid).
- Dashboard gates: `align_top_vn/goal`, `cnn_yaw_deg`/`seg_yaw_deg`/`latch_yaw_deg` side by side (for comparison with the real rotation).
- **Passed adversarial review** (0 correctness bugs); 2 items reinforced: FINAL contamination and capture-stale. The mod-90 fold is correct since the cube is square (the problem was measurement).

## 4. Speed · Lift · refactoring
- **Speed**: with `--arm-vlim 0`, JointSlew (vmax cruise cap / amax accel / zero-overshoot clamp) dominates. Bottleneck = vmax. **Defaults set to the tuned set**: async **ON**, vmax **6.5**, amax **12**, rule-step **0.45**, arm-vlim **0**, grip-force **0.07**, grip-torque-target **0.8**. Running without flags uses these settings.
- **Lift +z**: repeat `_ik_step6` (cap 16) to reach `--lift-step` (**0.15**) per click, snap. "Crisp".
- **Refactoring**: **removed** IBVS, Step, Hold, Approach, the Verify sequence, cube-nudge, z-bias buttons (debug −120 lines, dashboard −44). Buttons = GRASP FLOW (Capture/RUN/Align/Align flip/Final/Grip/Lift) + grip force + STOP/Rest/Zero/Abort + jog. Buttons↔handlers match 15:15.

## 5. Remaining (next session)
1. ~~Commit `deploy/debug_deepreach.py`~~ **DONE (session 8, review+fix+commit)**: adversarial review of the whole uncommitted diff (multi-pass workflow, 7 groups × 2-lens rebuttal verification) → fixed 4 substantive items, then 2 commits along the docs/18 boundary. **A** (this commit) gate the debug_deepreach yaw-latch **on the perception frame ts** (the 100Hz per-tick latch froze within ~30ms < 1 frame → what was a single-frame value becomes a ~3-frame average; same pattern as the `est.update`/`fk_bias` EMA). In commit-2 (perception/deep_reach_world): **B** port the yaw-latch to deep_reach_world + `--yaw-src` default `seg` (matches the HW-verified version, making the docstring's "IDENTICAL" claim true), **C** `conf_floor` descent gate (if track_conf≤floor, hold instead of blind descent and auto-resume on reacquire, Capture seed preserved), **D** 2 regression tests for CubeTracker EMA-warmup (first-obs SNAP). Verification: py_compile + module import + `tests/test_cube_tracker.py` 22 PASS.
2. **Real-robot verification of the full Final → Grip close → Lift cycle** (Align works). For grip, check grasp force at `--grip-force 0.07` (if weak, grip_tight / recalibrate the ratio with `calib_grip.py --skip-open --skip-closed`).
3. yaw: check whether the default seg-latch is sufficient. If not, `--yaw-src cnn --wrist-detector ckpt/cube_detector_wrist_sd4.pt` (sd4 = roll-corrected yaw head, yaw_err 2.4° per analysis; do not use sd3 = biased).
4. Port force-servo into run_pick.py autonomous grasping (currently blind close; protected by the 0.18 hard cap).

## 6. Files · commands · knobs
- **Files**: `deploy/debug_deepreach.py` (tool + control loop), `deploy/robot_io.py` (force-servo/hard cap), `deploy/calib_grip.py`, `deploy/debug_dashboard.py`, `deploy/cube_detector_io.py` (CNN yaw 6-tuple).
- **Run**: `.venv-deploy/bin/python deploy/debug_deepreach.py` (defaults = tuned set).
- **Verification flow**: Capture→RUN (check `latch_yaw_deg` fills with the real rotation during descent)→Align (J4 center + J6 = latch yaw; if reversed, `Align flip`)→Final→Grip close (grip_tau→0.8)→Lift.
- **Main knobs**: `--yaw-src` (seg/cnn), `--align-vn-target/-band`, `--align-j4-gain` (−0.6), `--grip-force/-torque-target`, `--vmax/-amax/-rule-step`, `--lift-step`, `--yaw-latch` (the z band is YAW_LATCH_Z in code).

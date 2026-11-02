# Deploy Bring-up handoff — real-robot reach→grasp success (2026-06-25 session 3b)

> After docs/11 (robust-hybrid architecture). **Verified on the real robot, using the debug_deepreach debugger, deep-reach arrival and grasping the cube.**
> **One line: deep-reach settles onto the cube on the real robot and the gripper bites the cube. The root bug in each of the 3 control axes (Z/XY/Grip) was caught — Z=base_pos mismatch between the two FK chains (5cm), Grip=closing direction (+deg) blocked by the driver limit + force 10%, XY=oblique-cam bias. What remains is XY position dependence (being nudged manually) and ⑦ lift · commit.**

---

## 0. What works now (verified on the real robot)
**reach (top-cam→IK descent) → xy centering → grasp (gripper clamp)** works on the real hardware.
- Z: correct without hardcoding (grasp-zgap-io 0).
- Grip: clamps the cube after gripper recalibration (grip_tau rises).
- XY: top-bias + **manual cube-nudge** (still has to be adjusted per position — the only non-robust point).
- Not done: final confirmation of ⑦ lift, XY automation, commit.

## 1. Run command (working config)
```bash
# Turn off the current dashboard first (the calib scripts occupy the same /dev/ttyACM0)
.venv-deploy/bin/python deploy/debug_deepreach.py --port 8899 --config configs/train_v6a_yaw.yaml \
    --base-z 0.010 --top-bias 0.035 --jaw-fwd 0.01 --grip-force 0.3
#  Applied by default: --grasp-zgap-io 0, --smooth 0.8, --wrist/front-dev=udev symlinks, AF off, limit+120
# Flow: ① cam identity → ② Capture → (if xy misses) cube y±/x± nudge → RUN (descend) → Grip close → Lift +z
```
Dashboard http://localhost:8899 (remote `ssh -L 8899:localhost:8899`). The arm moves only when Step/RUN/Approach/jog is pressed. ABORT=estop.

## 2. The 3 control axes — root causes and fixes

| Axis | Symptom | Root cause | Fix |
|---|---|---|---|
| **Z** | stops 5cm above the cube; manual grasp-zgap-io −0.05 needed | **`--base-z` changed only base_controller while robot_io used the cfg (−0.03) → a 5cm mismatch between the two FK chains** contaminated auto-z `fk_bias` + `z_gap_io` | Apply the measured value to **both FKs** via `io.base_pos[2]=args.base_z` → **grasp-zgap-io 0** (fk_bias 0.04→0.09 = the removed mismatch). auto-z closed loop + reached based on z_gap_io |
| **XY** | the gripper settles **in front (+y)** of the cube | the front-cam ray-plane at an oblique angle (~45°) → **visible-face centroid height × line of sight → +y bias**. + additionally **front X is position-dependent** (+3~4cm bias toward −x at −x off-center) | `top-bias 0.035` (inverse correction along the calib cam→cube direction) + `jaw-fwd 0.01` (jaw geometry −1cm y) + **manual cube-nudge** (per-position X residual). **Root fix not done** |
| **Grip** | the gripper **never closes** | ① closing = **+motor-deg**, but the driver `joint_limits[gripper]=(−270,0)` **clips** +deg commands at 0° (both reachable extremes 0°/−250° are OPEN) ② `force_pos_torque_ration=0.1` (10%) too weak | robot_io: limit max 0→**+120** (`grip_close_limit_deg`) + force **0.3** (`grip_force_ratio`/`--grip-force`). **Recalibrate only the gripper** (`calib_grip_interactive`) → **grip_closed_deg=+120°, grip_open_deg=−129°** |

### Important facts
- **`base_pos` = world position of the URDF `base_link` (FK root), ≠ J1 axis** (the URDF joint1 origin is +0.08465 above). Measured J1 axis +0.095 → base_pos.z **+0.010**. (The old note "measurement reference = J1 axis" is wrong.)
- **Do not run the full lerobot `calibrate()`** — it re-zeroes all motors (`set_zero_position`) → **breaks the verified arm joint_map**. Only the gripper, via `calib_grip_interactive`.
- **Camera indices get re-enumerated** (video0↔2↔3 change) → always open via **udev symlinks** (`/dev/video_front`,`/dev/video_wrist`). **Turn off AF** on the C922 (focus breathing drifts the fixed-K wrist back-projection). Fix it with `--wrist-focus`.
- **It was not that robot_io was accurate only in Z** — robot_io z also read ~5cm low at the reach pose because of the base_pos mismatch, and after making it consistent, z_gap_io≈0 is a true settle.

## 3. New tools / dashboard features
- `deploy/calib_grip_interactive.py` (key) — **gripper-only interactive calibration**: jog (negative=close/+deg, positive=open) and mark closed/open by eye, saving to joint_map (arm unchanged). Unlocks closing with limit +120.
- `deploy/calib_grip_close.py` — closing-direction torque sweep diagnosis (this is what revealed the +deg clip). Auto-detect fails because of the weak force → use interactive.
- Dashboard (debug_dashboard.py): **live camera panels** (FRONT cube/gripper, WRIST CNN overlay) + **VERIFY sequence panel** + buttons: `↩Rest`/`⊙Zero` (URDF home q=0)/`↑Approach` (z=0.3 rule IK)/`cube xy nudge ±`/`grip−(tight)/+(loose)`/`z-bias ±`.
- debug_deepreach features: **auto-z** (fk_bias feed-forward closed-loop z), **smoothing** (EMA, single-shot buttons snap), **wrist made optional** (FRONT-ONLY if absent), **reached** = based on z_gap_io.

## 4. Commit status
- Committed: **`8a088cb`** (base_pos consistency + camera/auto-z/perception), **`ffacc17`** (zero/cube-nudge/wrist-optional/capture-guard). branch `residual-rl-v6-phase1-2a`.
- **Uncommitted (working — commit recommended)**: `deploy/debug_deepreach.py`, `debug_dashboard.py` (grip dial), `robot_io.py` (force/limit/readback-clamp relaxed), **`joint_map.npz`** (gripper recalibrated +120/−129), new `calib_grip_close.py`·`calib_grip_interactive.py`.
- gitignore/artifact: `*.png`, `*.npz` (except joint_map), `data/`, `viz_out/`.

## 5. Remaining work (priority order)
1. **⑦ Lift +z** — if the cube is lifted and comes up, **real-robot full pick reach→grasp→lift verification is complete**.
2. **Commit** — save the §4 uncommitted items (gripper fix + joint_map).
3. **XY root fix** (the only non-robust part): front-cam X position bias (~+3cm). Two paths — **(a) wrist angle-invariant xy** (directly from above; CNN straight-down recall 6% backlog → red-mask) or **(b) front lens distortion/geometry correction** (fit with dxy-vs-position data). dxy data: x−0.073→[+0.03,0], x−0.167→[+0.04,+0.01], x+0.035→[0,+0.01].
4. **v6b policy full deploy** — port deep_reach base + residual + this stack (base-z/top-bias/jaw-fwd/grip cal) into run_pick (the current run_pick is an old version that does not support obs41).
5. Fix the wrist CNN straight-down recall labels (backlog).

## 6. Key knob summary
| knob | value | meaning |
|---|---|---|
| `--base-z` | 0.010 | URDF base_link world z (J1=+0.0847 above). Applied to both FKs |
| `--grasp-zgap-io` | 0.0 | z target (robot_io z_gap_io). With the base made consistent, 0 means settled |
| `--top-bias` | 0.035 | front cube-xy oblique correction (cam-away) |
| `--jaw-fwd` | 0.01 | TCP −1cm y grasp offset (jaw geometry) |
| `--grip-force` | 0.3 | gripper FORCE_POS torque ratio (driver default 0.1 = weak) |
| `grip_close_limit_deg` | 120 | unlock gripper closing (+deg) (driver default 0 = clip) |
| joint_map grip_closed/open_deg | +120 / −129 | recalibrated (closed=+120°, open less wide) |
| `--smooth` | 0.8 | motion EMA |


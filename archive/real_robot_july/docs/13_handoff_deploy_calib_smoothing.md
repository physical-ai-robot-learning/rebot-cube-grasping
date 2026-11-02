# Deploy Handoff — front-cam CALIBRATION established + motion SMOOTHING (2026-07-07 session 4)

> After docs/12 (reach→grasp bring-up). **This session: (1) found the real root cause of the front-cam XY bias (fake intrinsics → real checkerboard calib), (2) made the descent motion smooth (camera-control decoupling + several judder/spike fixes).**
>
> **One line**: front-cam cube XY became accurate to **~5mm at all positions** (Y = real intrinsics, X = constant −2.35cm correction), and the descent goes down **smoothly without jolts/spikes**. MIT smoothing was dropped (POS_VEL is enough); instead perception was split into a thread so control runs at 100Hz. What remains is **continuous real-robot full pick (reach→grasp→lift) verification**.

---

## 0. What works now

- **Front-cam cube XY accurate** (with `--top-bias 0`): **within ~5mm** of ground truth at several positions. Position-dependent bias gone.
- **Smooth descent**: control at 100Hz with `--async-perception` + several fixes remove jolts/spikes.
- reach→descent→grasp (manual) already worked in docs/12. This time **precision + smoothness** were added on top.
- Not done: **continuous full pick (reach→grasp→lift) real-robot verification**, v6b policy port.

## 1. Run command (working config)

```bash
cd ~/workspace/rebot_sim2real
.venv-deploy/bin/python deploy/debug_deepreach.py --port 8899 --config configs/train_v6a_yaw.yaml \
    --base-z 0.010 --jaw-fwd 0.01 --grip-force 0.3 \
    --front-exposure 78 --wrist-exposure 78 \
    --async-perception --arm-vlim 0 --vmax 2.5 --amax 6
#  --top-bias / --x-corr defaults (0 / 0.0235) are correct, so no need to pass them.
#  Dashboard http://localhost:8899 (remote ssh -L 8899:localhost:8899). The arm moves only on button press. ABORT=e-stop.
```

## 2. CALIBRATION — front cam (the core of this session)

**The root cause of the XY bias = the front cam was never properly calibrated** (old hypotheses like top_bias/jaw-frame were all supporting players). Confirmed with a quantitative workflow:
`cam_front_intrinsics.npz` was **fake** — `fx=fy=906.8` (estimated from the C922 spec), principal point = image center, `dist=0`, `approx=True`. Measured: **fy=859.7 (5% lower than assumed), cy=374 (14px up from 360), real distortion** → this was the main culprit of the position-dependent bias (especially Y/depth).

### 2a. Intrinsic (checkerboard) — done
- Pattern: `deploy/make_checkerboard_pdf.py` → A4 9×6 inner corners, 25mm vector PDF (**print at 100%**, measure squares with a ruler).
- calib: `deploy/calib_intrinsics_checkerboard.py --cols 9 --rows 6 --square-mm 25 [--from-dir DIR]`
  - Interactive: ENTER=capture (varied tilts/positions, especially bottom-center + ~1m depth, 15~25 shots), `done`=calibrate.
  - Achieved **RMS 0.408px** → `cam_front_intrinsics.npz` (real K+dist, `approx=False`, **committed**).
  - `perception.estimate_red_cube_world(..., dist=)` does `undistortPoints` before ray-cast (unchanged for sim/legacy when dist=None/0).

### 2b. Extrinsic (camera pose) — using the cube method
- **Current calib.npz = cube extrinsic** (`calib_extrinsic_known.py`, cam +0.609 normal). Re-solved with the real K.
- Warning: checkerboard-on-table (`calib_extrinsic_checkerboard.py`) was also built, but **the 2-point hand anchor is fragile**: the planar solvePnP two-fold ambiguity flips the camera below the table (pose is garbage even at reproj 0.29px). **Added a guard that refuses to save below-table poses**. To use it, the planar degeneracy must be broken with **3+ points + a tilted board** (not done). **For now the cube method is the right one.**
- Warning: **Trap**: the 10-point coplanar cube reproj (17px) **measures hand-placement noise and is not an acceptance metric**. Judge by intrinsic RMS + an **independent known-position world check**.

### 2c. XY bias residual → constant X correction
- After real intrinsics: **Y bias gone, only X remains**. 8-point grid (2 depths × 4 X) measurement → **constant −2.35cm offset** (scale 1.008≈1, no depth dependence = not yaw/fx).
- Recenter with **`--x-corr` default +0.0235** (m) → residual **RMS 3.4mm / max 5.5mm**. X-only term in `correct_top_xy` (Y/Z unchanged, applied even with `--top-bias 0`). `--top-bias` default 0.035→0 (real intrinsics absorb the oblique effect).
- Warning: **x_corr is tied to the current extrinsic** — if the extrinsic is recalibrated, re-measure the 8 points and update `--x-corr`.

### 2d. Exposure (required!)
- C922 **auto exposure (Aperture Priority) burns the image to white under this lighting** → 0 red detections. **Manual exposure `--front-exposure 78`/`--wrist-exposure 78`** (39~156 OK). The calib scripts are also manual (`FRONT_EXPOSURE` env / `--exposure`).
- **AF always off** + focus must be **identical** across intrinsic/extrinsic/deploy (fixed cam, so it holds unless powered off). Can be specified with `--focus`.

### 2e. Recalibration order (if needed)
1. Print checkerboard at 100% + measure → `calib_intrinsics_checkerboard.py` (target RMS<0.5px).
2. `calib_extrinsic_known.py` (cube ≥6 points, wide dx/dy; real K/dist auto-loaded).
3. Re-measure the X residual with the 8-point grid → update `--x-corr`.
4. Verify: cube at a known position → XY ~5mm in `debug_deepreach --top-bias 0`.

## 3. SMOOTHING — descent motion

**MIT mode was implemented and then dropped** (`05b2900` revert): POS_VEL glide is enough. **Root problem = control was tied to the ~10Hz camera loop** (measured: CAN read+write 2.3ms = ~400Hz headroom, not the bottleneck).

- **`--async-perception`**: split the camera (grab+CNN+front-Z+overlay+capture) into a background thread → the control loop (proprio→IK→send) runs at **`--control-hz` (default 100)**. Camera (USB) and motors (CAN) are on different buses, so no lock needed. Default OFF (sync fallback). Fixes after adversarial review: **fail-stop** (perception dead/stale>0.5s → auto-halt+freeze), focus argument swap, estop order.
- **Speed/smoothness knobs**:
  - `--arm-vlim 0` → enables the soft JointSlew shaper. **`--amax` = acceleration = smoothness dial** (lower is smoother, start at 6). `--vmax` = cruise speed. **`--vmax` is effective only with `--arm-vlim 0`** (ignored with the default 2.5 glide).
  - Slow = `--arm-vlim 1.5` (low-speed motor glide).
- **Descent judder fix**: the auto-z z-target EMA was rate-dependent (tuned at 10Hz), so noise leaked through at 100Hz → **rate-independent time-constant EMA (`FK_BIAS_TAU=0.4s`)**.
- **TCP spike during descent fix**: the `deep_reach_step` descent latch released at `gross_miss (=reach_eps×4=12cm = same as the wrist gate)` and jumped back to hover → **threshold separated to 18cm + 4-frame debounce** (base_controller; no-op in sim with the analytic wrist = policy unchanged).
- **Raising control_hz does not help spikes/discontinuities** — it only densifies smooth segments. Keep 100.

## 4. Commits (branch residual-rl-v6-phase1-2a, 12 after `65778d3`, not pushed)

`c9c1de0` action shaping · `657fdee`+`c9df753` MIT (→`05b2900` **revert**) · `0035b4e` async-perception · `0171ad4` real intrinsic calib+undistort · `651ed38` checkerboard PDF · `1296efb`/`b259325` extrinsic AF/exposure · `b07167c` checkerboard extrinsic · `ed2e3bd` below-table guard · `6987e5e` --front/wrist-exposure · `bcfc7ba` **--x-corr** · `f771eb5` auto-z rate-EMA · `993da4a` **descend-latch debounce**.
- gitignore: `calib.npz`/`extrinsic_known_raw.npz`/`calib_front_imgs/`/`*.png` ignored; **`cam_front_intrinsics.npz` is tracked** (real calib).

## 5. Remaining work (priority order)
1. **Continuous real-robot full pick verification**: reach→descent→**grasp (step 7: Grip close, grip_tau↑/grasped)**→**Lift +z**, check whether the cube comes up. XY (±5mm) + Z (front-gripper) + gripper are all handled, so continuous success is now the gate.
2. **wrist CNN**: straight-down recall ~6% (label backlog). Currently irrelevant to deploy progress thanks to the red-mask fallback + front XY as primary, but retrain to raise wrist fusion reliability.
3. **Extrinsic improvement (optional)**: if X drifts again, re-solve with 15~20 cube points (wide X) or a **tilted checkerboard**.
4. **v6b policy port**: `run_pick` (old version without obs41 support) → deep_reach base + residual + this stack (exposure/x_corr/base-z).

## 6. Knob summary
| knob | value | meaning |
|---|---|---|
| `--front-exposure`/`--wrist-exposure` | 78 | **Manual exposure** (auto washes out to white). Required |
| `--x-corr` | 0.0235 | Constant world-X recenter (residual extrinsic-x). Update on recalibration |
| `--top-bias` | 0 | Legacy oblique shim (unneeded with real intrinsics) |
| `--base-z` | 0.010 | URDF base_link world z |
| `--jaw-fwd` | 0.01 | jaw-forward grasp offset |
| `--grip-force` | 0.3 | gripper FORCE_POS torque ratio |
| `--async-perception` | (flag) | split camera thread → control at 100Hz (smoothness) |
| `--control-hz` | 100 | control rate (irrelevant to spikes/judder) |
| `--arm-vlim` | 0 (+shaper) / 1.5 (low-speed glide) | 0 = soft shaper (--vmax/--amax), >0 = motor glide speed |
| `--amax` | 6 | shaper acceleration = smoothness dial (lower is smoother) |
| `--vmax` | 2.5 | shaper cruise speed (only with arm-vlim 0) |

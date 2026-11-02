# Handoff — grasp+Lift verified and committed + wrist look-at R&D (option 3) started (2026-07-13~14, session 12~13)

> Follows docs/21 (session 11, J4 reassembly recalibration + FK_Z_CORR bug found, re-verification pending). This session:
> **(A)** FK_Z_CORR 0.03 re-verification passed → real grasp+Lift succeeded (reassembly verification done) → **Lift force-servo-freeze bug found and fixed** → **3 commits pushed**.
> **(B)** Tried to stabilize wrist "look-at" framing → all 3 approaches failed; established that **the root cause = front-cam XY estimation error + uncalibrated wrist camera**.
> **(C)** Started option 3 (making look-at work properly) R&D → **built the wrist physical-calibration infrastructure offline** (fit script + calib_wrist append). HW collection, fit and verification are for the next session.
>
> **One line**: grasp+Lift = committed/pushed (works). Look-at is an unsolved research problem — root cause confirmed as XY estimation / wrist calibration; only the C infrastructure was built offline (uncommitted).

---

## 0. (key) Wrist camera real calibration (C) — protocol REDESIGN (2026-07-14, verified) + HW procedure
> **The original §0 plan from session 12→13 (J4-only sweep, cam pointing straight down) turned out to be NON-IDENTIFIABLE** (offline verification; scratchpad/wrist_calib_*.py and a multi-pass adversarial review converged on the same root causes). Even starting from the true values, the fit drifts 2~46° in axis; multiple positions (fixed clearance) are also invalid. Root causes = ① the `x=up×z` projection frame is **gimbal-singular** when the cam is vertical ② J4-only small excursion = no signal ③ in shaper mode a ±0.10 sweep **collapses to an actual ±0.034** ④ optimizer (Adam 3k) not converged ⑤ the fit used base_z (−0.03) vs recorded (+0.01), a **40mm frame mismatch** (→ 26° error) ⑥ focal/sign had **no consumption path**. **All 6 fixed + re-verified (uncommitted).**

**Redesigned protocol (synthetic identifiability + end-to-end verification: axis recovered to 0.05°, focal 190/193, sign correct):**
- calib_wrist = **offset look-at hover** (TCP offset 6cm sideways from the cube + `look_at=cube` → cam ~15° off-vertical = escapes the gimbal singularity) + **J1+J4 diagonal sweep** (J1 = base yaw is the key excitation) + `snap` (fixes the sweep collapse) + base_z recorded. The offset direction rotates automatically per position.
- **≥6 DISTINCT cube positions** required (a ring / few positions gives large variance → not trustworthy). Focal is **optional** (axis is recovered from the extrinsic alone; focal error is absorbed by the --lookat-vn trim).
- Applying it = **deploy-only override** (`--wrist-calib`, same pattern as `--base-z`): shared config/sim untouched. After the physical fit, sign is reduced to a pure ±1.

```bash
# 0) Run the baseline (calib does not need --lookat-ibvs). Leave --base-z at the usual value (default 0.010) → recorded in the npz.
cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/debug_deepreach.py \
  --wrist-detector ckpt/cube_detector_wrist_sd4.pt --grip-force 0.05 --grasp-zgap-io 0.03
# 1) Start clean (an old-schema npz is automatically treated as fresh, but explicit deletion is recommended)
rm -f deploy/wrist_calib_data.npz
# 2) For each of ≥6 cube positions (spread over the workspace: center/left/right/near/far/extra):
#    move the cube → curl .../cmd?a=capture → curl .../cmd?a=calib_wrist
#    Warning: watch the FIRST run: offset (6cm) + J1 (±0.10) + J4 (±0.12) swings the gripper sideways by several cm (TCP stays ~14cm up). Stop/estop immediately if needed.
#    The dashboard shows "pos#N ... / N positions". Tuning: --calib-offset/--calib-j1/--calib-j4
# 3) Physical fit (offline; base_z is loaded automatically from the npz). Default = estimate focal, --no-fit-focal = extrinsic only (more robust)
.venv-deploy/bin/python deploy/fit_wrist_calib.py --target cnn      # must not print the "UNDER-IDENTIFIED risk" warning (≥6 pos, J1 span≥0.15)
# 4) Review the output (RMS down, axis moved, sign u+1/v-1) → apply on rerun with --wrist-calib (do not edit config):
.venv-deploy/bin/python deploy/debug_deepreach.py --wrist-calib deploy/wrist_calib_fit.npz \
  --wrist-detector ckpt/cube_detector_wrist_sd4.pt --grip-force 0.05 --grasp-zgap-io 0.03
# 5) Re-verify (u,v) agreement on HW (model projection tracks the detector throughout the J4 sweep) → then B (continuous look-at)
```
**Caution**: if the fit prints `Warning: UNDER-IDENTIFIED risk` (positions < 6 or J1 span < 0.15), **do not apply it**; collect more positions. `--wrist-calib` is opt-in, so it has no effect on the baseline sd3 descent (but when enabled, the baseline also uses the same override cam → check (u,v) agreement).

---

## 1. Git / committed vs uncommitted state (branch `perception-topwrist-coop`)

**Committed + pushed (HW verified, session 12):**
```
a6c1f09 docs: session12 handoff update — FK_Z_CORR verified + lift bugfix
bd94fb5 fix(deploy): lift force-servo-freeze fix + cube world-offset (+ S10 track/align-seg/dashboard)
bfe1ea5 fix(deploy): FK_Z_CORR 0.09→0.03 (HW-measured) + J4-reassembly recalib
```

**Uncommitted (opt-in, no effect on baseline, recommended to commit after HW verification):**
```
 M deploy/debug_deepreach.py   ← all look-at work: feedforward (discarded) → IBVS Align (--lookat-ibvs/--lookat-vn) +
                                  calib_wrist command (append) + _project_wrist_unvn/_wrist_lookat_j4 helpers + WRIST_VN constant
 ?? deploy/fit_wrist_calib.py  ← wrist physical-parameter fit (new, verified offline)
 (deploy/wrist_calib_data.npz  = 1-position example data; rm and recollect next session)
 (deploy/wrist_calib_fit.npz   = example fit output)
 M uv.lock / ?? deepreach_sac_world.py / docs/20 = other workstream, unrelated (leave as is)
```

---

## 2. (A) grasp+Lift verified (the real goal of the reassembly verification achieved)
- **FK_Z_CORR 0.03 re-verification passed**: after RUN `fk_bias=0.03` (normalized from 0.09, auto_z contamination resolved), `tcp_io.z≈0.05` hover (does not hit the table), `z_gap_io≈goal`.
- **Real grasp+Lift succeeded**: Capture→RUN→(jog descend)→Grip (grasped=True, grip_tau 0.93)→**Lift (tcp_io_z 0.024→0.323, cube lifted off the table)**→put down.
- **Lift force-servo-freeze bug fixed**: the idle `elif gripping:` re-pinned the arm every tick with `send_arm(pr.arm_qpos,snap)` → after the grasp, Lift could not raise the arm. **FIX = `gripping=False` in the lift/jog handlers** (grip stays closed). Exposed by the first complete run of the full grip+lift cycle.

---

## 3. (B) look-at framing — all 3 approaches failed, root cause established (do not retry)
Goal: stabilize the wrist cam looking at the cube (framing via J4). **All failed**:
1. **seg servo** (existing Align, fixed gain −0.6): overshoot (gain does not fit the pose).
2. **Geometric feedforward**: depends on the **uncalibrated wrist extrinsic (~150mm)** → drives J4 the wrong way and snags. + **v-sign flip** (legacy look-at v=+ vs trained detector render v=−). A 1-pose affine correction failed to generalize.
3. **IBVS** (bearing, extrinsic-robust): sd4 wvn has the **opposite v-sign from seg (+6.2 vs −4.69)**; **∂wvn/∂J4 ranges 0~+6 depending on the pose** (authority ≈0 off-center; rotation + parallax of the J4-offset lens cancel out); false convergence due to noise.

**Root cause (pointed out by the user, correct)**: **front-cam cube XY estimation error** is the real blocker — look-at is downstream (it aims at a wrong cube_world). Evidence: after RUN "reached", the cube is at u=0.73 in the wrist view (the wrist is not above the cube). On a flat table the planar assumption is valid, so the error = **oblique red-centroid bias** (the 45° cam sees the side faces). **Two-cam stereo (Feature C) exists but is worse on a plane** (needs wrist calibration; the code's own conclusion, perception_fusion.py:166). **Monocular depth AI is unsuitable** (precision, OOD). **Z = front-gripper-Z is robust.**

**Key point**: the baseline (sd3 unproj descent) already grasps (§2). Look-at is a refinement on top of it.

---

## 4. (C) Option 3 R&D — wrist real-calibration infrastructure (offline build done)
Plan: **C (wrist calibration) → B (control) → A (detector)**. C is the foundation (accurate model → pose-dependent authority in B is resolved geometrically with `ik_oriented(look_at=cube_world)`; multiple joints aim the optical axis). This build (offline, py_compile OK):
- **`deploy/fit_wrist_calib.py`** (new): fits the wrist physical parameters (mount_offset + aim_local + focal scale) by torch-autograd reprojection onto the detector (u,v). **Reuses the existing deploy projection convention** (results plug directly into cfg) + auto-detects u/v signs. 1-position verification: RMS 0.225→0.004 + UNDER-IDENTIFIED warning works.
- **`calib_wrist` command**: **appends** per cube position (multi-position accumulation).
- **B (next)**: call `base.ik_oriented(look_at=cube_world)` every tick (already exists; needs C). **A (parallel/offline)**: retrain sd4 straight-down (low-noise center).

---

## 5. Robot / infrastructure state
- Robot: **stopped, torque-off** (`done (torque off)` confirmed). Safe.
- Dashboard live state = `curl localhost:8899/state` (when the process is rerun). Two FKs: `tcp_io` (actual) vs `tcp` (base_ctrl shadow).
- Warning: **"PERCEPTION DEAD/STALE" is a false positive from a startup latch** — do not chase it by restarting. Confirm liveness by the `/frame` md5 updating or by `capture` filling cube_raw (4 useless restarts this time because of this).
- Warning: **risk of over-descent on RUN with a far cube (y≳0.6)** (does not reach, keeps descending) → place near the center + guard tcp_io_z during descent.

---

## 6. Next-session actions
1. **C: wrist real calibration** (§0 procedure) — multi-position data → fit → apply to cfg → verify (u,v) agreement on HW.
2. **B: continuous look-at** = wire `ik_oriented(look_at=cube_world)` (after C).
3. (parallel) **A: retrain sd4 straight-down** (cluster GPU).
4. **Alternative**: instead of look-at, directly improve **front-cam XY accuracy** (cube center / ground-contact point detection) — leverage on the root XY cause. Or **accept the working baseline grasp**.

# Handoff — (key) AUTO-GRASP full-cycle HW success: joint_map identity-fallback root cause found (P4 adversarial review) + reach-Z refit + closed-loop descent (+Y drift) + Align yaw gate · sign bug fix (2026-07-23, session 22)

> Continuation of docs/32 (S21: debug_deepreach→src/deploy refactor P0–P3 + reach-Z refit). Per the user's instruction, **restarted deploy to verify the full grasp cycle on HW**; while doing so, traced the cause that had never been caught all the way down — the P4 adversarial review caught **the root cause (joint_map not loaded)** — and then resolved reach-Z, +Y drift and Align yaw all on the real robot. **Branch `deploy-refactor-main`. 4 commits this session (`6730a29`..`f497b42`), all local and unpushed (86 cumulative since main).**
>
> **One line**: ① P4 (17 tools deploy/→src/tools/) + **11-dimension adversarial review (0 regressions)** found a **pre-existing root bug** — `robot_io` was moved to src/ in P1 (`dbe34de`) but `joint_map.npz` stayed in deploy/, so **since P1 the runtime has silently fallen back to the identity joint convention (J2/J3/J4 offsets off by −18/−14/+16°)**. This is the root cause of the reach-Z drift, +Y drift and base↔io FK mismatch. ② Read-path fix → base↔io FK z mismatch **−62mm→0**. ③ reach-Z refit on the corrected FK (FK_Z_CORR −0.063→+0.015, slope 0). ④ **Closed-loop settle-gated descent** removes the +Y droop drift. ⑤ **Align yaw**: conservative time-stability gate (rejects jumpy seg → J6=0 fallback) + **jaw_sign sign-bug fix** (masked by the 45° diamond: residual = 2·theta mod 90). **Result: grasp succeeded after XY·Z·yaw alignment on a cube at an arbitrary angle (grip_tau ~0.9, grasped=true).**

---

## 0. Git / robot state
- Branch `deploy-refactor-main`. 4 commits this session (all unpushed): `6730a29` (P4 tool move) · `ca06141` (joint_map fix + reach-Z refit) · `28c5c31` (closed-loop settle-gated descent) · `f497b42` (AUTO-GRASP yaw gate + sign + depth). docs/20-33 untracked + scripts/submit_train_detector.sh are unrelated to the refactor.
- Robot: HW tuning by repeatedly restarting deploy (port 8899). `.venv-deploy/bin/python deploy/main.py`. Warning: **changes to code/CLI defaults take effect only after a restart.** Live-tuned graspz · jaw_sign values reset on restart, so the working values were baked in (below).
- Warning: during the session I wrote **scripts/probes/verify_descend_ikfk.py** (descent IK/FK verification), but `scripts/probes` is gitignored, so it is untracked (kept locally).

## 1. (key) joint_map identity fallback = root cause (ca06141)
**Discovery**: during the P4 adversarial review (§6). `robot_io.py:214` reads `joint_map.npz` from `dirname(__file__)/joint_map.npz`, but **P1 (`dbe34de`) moved robot_io from `deploy/` to `src/rebot_sim2real/deploy/io/` while leaving the data file in deploy/**, so the dirname-relative read points to an empty location → **since P1 every startup falls back to identity SIGN=[1..]/OFFSET=[0..]** (the warning text "no deploy/joint_map.npz" is misleading). The actual rest-anchored OFFSET = **J2 −18.3° J3 −14.2° J4 +15.6°** → every FK-based position was contaminated.
- **This is the true identity of the "J2/J3 motor-zero drift" suspected in docs/32 §4.** Root cause of the reach-Z "9.3cm constant over-read", the +Y descent drift and the 5.7cm Y mismatch between base↔io FK.
- **Fix**: read `deploy/joint_map.npz` relative to the repo root. **HW confirmation**: `loaded joint_map.npz ... OFFSET=[-0.002 -0.319 -0.248 0.272 -0.025 -0.006]`; **base↔io FK z mismatch −62mm→0** (proves the Z root cause is resolved). Warning: a ~6cm Y mismatch remains (not the offset; difference in TCP-frame definitions between the two FKs = separate issue).

## 2. reach-Z refit (ca06141)
Table-touch sweep with `src/tools/fk_verify_z.py` on the corrected FK. At working radial ~0.45, z_gap_io = **−0.034±0.002** (ideal −0.019) = over-read dropped sharply from **9.3cm → 1.5cm** (root cause = offset, confirmed).
- `FK_Z_CORR −0.063→+0.015` (robot_io, refit after removing the stale band-aid), CLI `--z-reach-slope 0.0676→0` (the old value was stale, fit on top of identity; **slope kept at 0 because of inconsistent near-radial data**).
- Warning: **fktouch false contacts**: at FAR (radial≥0.52), near the reach limit, the arm cannot reach the table and gravity / IK-saturation torque triggers false positives (the user discarded runs 8-13); near is noisy too. **The correct answer for a precise distance-independent slope is a refit after improving fktouch (saturation gate + torque step) — not started.**

## 3. Closed-loop settle-gated descent (28c5c31)
**Symptom (before the fix)**: the AUTO-GRASP straight descent was open-loop base-FK, so the real arm **drifted in +Y** (physically confirmed with wrist (u,v): wvn 0.44→0.92 during descent, tcp_y +23mm) → the jaw closed on nothing beside the cube. Offline IK/FK (probe) confirmed the model holds xy perfectly → real-robot FK mismatch (joint_map) + gravity droop.
- **Fix (modes._step_follow)**: keep auto-center running during descend (`ag_descend_servo`) to re-center in real time from the wrist + bypass the latch (continuous) + **settle-gate** (if outside the center margin, HOLD z → descent speed self-limits to the servo bandwidth). Only the last `--autograsp-commit-zgap` (0.015) freezes + plunges vertically. Dedicated slow `--autograsp-descend-step` (0.0015).
- **HW**: wvn held at 0.49 during descent (no runaway), tcp_y fixed, grasp succeeded. (The joint_map fix is the root fix, so this is supplementary, but kept for robustness.)

## 4. (key) AUTO-GRASP yaw: gate + sign bug (f497b42)
**Symptom**: Align intermittently over-rotated by 45° → failure. Logs: the wrist seg edge-voting yaw **jumps around the mod-90 range from frame to frame** (−30↔+43), and align **latched j6_target on the first confident frame** → grabbed a wrong value. A square can be grasped within ±~30° even with J6=0, so **a wrong yaw is more dangerous than no roll**.
- **Conservative gate** (constants/context/commands/modes): accumulate the align seg yaw as an **EMA over fresh frames** (J6 held = fixed frame); when the EMA concentration (vector magnitude) ≥ **YAW_CONC_MIN 0.85** and **MIN_N 45 (~1.5s)** is exceeded, commit to the EMA direction (noise average); if not stable by **GIVEUP 80 (~2.7s)**, **J6=0 no-roll fallback** (instead of abort). The long window distinguishes rejecting near-aligned jumpiness (conc ~0.5) from passing a real rotation (45° diamond conc ~1.0). Timeout 6→12s. + Mask cleanup `_clean_blob` (open + largest component) to clean the seg input.
- **(key) jaw_sign sign bug** (context, default +1→**−1**): this was the ~20° that remained even after the gate gave a "stable alignment". Derived on HW: rotating J6 by +32° rotates image yaw by −32° (1:1). To align (image→0) requires Δ=+theta, but the code rotated by Δ=−theta, so **residual = 2·theta mod 90**. → **Only 45° cubes happened to align (2·45≡0)**; all others were off. **After flipping the sign, every angle aligns correctly.** (It looked like an "Align margin" problem but was a direction bug.) Verified live with 'Align flip', then baked.
- **Baked**: `jaw_sign −1`, `--grasp-zgap-io 0.0→−0.04` (firm-grasp depth at the working radial; prevents reset on restart).

## 5. P4: tool move + adversarial review (6730a29)
17 executable tools in `deploy/` (calib_* · diag_* · fit_* · fk_verify_z · measure_latency · sysid_arm · debug_reach/grasp · damiao_params · make_checkerboard_pdf · wrist_view) → **`src/tools/`** (git mv). Import fixes (`from robot_io`→`rebot_sim2real.deploy.io.robot_io` · `debug_dashboard`→`.main_dashboard` · sys.path file-relative). **Remaining in deploy/**: 8 shims + main.py + debug_deepreach.py + **cube_detector_io · build_obs** (old policy helpers, P5). CWD-relative calib data paths (deploy/calib.npz · joint_map.npz etc.) kept. **11-dimension adversarial review workflow (import/syspath/sed/runtime/data-path/completeness, find→verify): 0 CONFIRMED regressions** — and this review found the §1 root bug.

## 6. Gotchas
- (key) **joint_map/calib data lives in the repo's `deploy/`** (the runtime `robot_io` reads it relative to the repo root). If runtime modules are moved again, recheck this path.
- (key) **jaw_sign −1**: rig orientation. Testing with a 45° cube looks aligned even with the wrong sign → **use an oblique (~30°) cube to determine the sign.**
- (key) **grasp-zgap-io −0.04 · FK_Z_CORR +0.015 · z-reach-slope 0** = depend on the current calib (working radial). Refit after recalibration / distance change.
- (key) Yaw accumulation is valid only when it happens **in a fixed frame while j6_target=None (J6 held)** (once J6 starts rotating, image yaw follows J6). Rotate after commit.
- (key) **Near-square yaw makes seg unstable → J6=0**, **a real rotation is stable → rotate**. The gate handles both.
- git commit -m gets corrupted by backticks → use `-F file`. When one file such as cli.py spans two commits, split hunks with `git add -p` (patch) (used this session).

## 7. Open / next (priority)
1. **Distance-independent reach-slope Z**: the current grasp-zgap-io −0.04 is for the working radial only. Start a **grasp-based refit** (collect (radial, z_gap_io) at firm grasp at each distance → slope) — currently only 1 point at FAR 0.565 (z_gap_io −0.013). Better than fktouch since there are no false contacts. (Or table-touch after improving the fktouch saturation gate.)
2. **Reproducibility sweep**: check grasp success rate over several cube angles (0/15/30/45°) and distances (NEAR/MID/FAR). Monitor v3 (`scratchpad/grasp_monitor3.py`→grasp_cycle4.log) logs seg_yaw/J6/z_gap_io.
3. **Wrist yaw source quality**: seg edge-voting is biased/unstable (worked around by the gate). For a more accurate yaw, strengthen top-face isolation in the mask or use a separate detector. The current gate is sufficient in practice.
4. **Push the 17 commits** + clean up untracked docs/20-33 + deploy-refactor-main P5 (original debug_deepreach + shims → _archive).
5. Room for fine-tuning the ~5° yaw residual (wrist view changed after the J4 correction).

## Run (next session)
```
# Normal grasp (all fixes/calib/jaw_sign/depth baked):
.venv-deploy/bin/python deploy/main.py
#   Zero → place cube and Capture → Follow ⊙CENTERED → AUTO-GRASP (arbitrary angle)
# Live diagnostics: curl -s localhost:8899/state ; wrist cam: curl localhost:8899/frame?n=wrist -o w.jpg
# reach-slope refit data: after grasping at several distances, fit (radial, z_gap_io@grasped) from the scratchpad/grasp_monitor3.py log
```
Details: docs/32.

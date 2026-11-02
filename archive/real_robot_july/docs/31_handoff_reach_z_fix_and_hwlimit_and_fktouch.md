# Handoff — (key) Reach-dependent FK-Z error quantified and corrected (TABLE-TOUCH) + HW-limit silent-clamp bug fix + AUTO-GRASP align→descend + edge-voting yaw + FK-touch verification tool + anti-windup/lift-rate (2026-07-21, session 20)

> Continuation of docs/30 (S19: Follow auto-center + MJPG 30Hz + AUTO-GRASP v4). Start = restart from handoff → iterative live deploy tuning (user-driven) → **quantifying and correcting the reach-dependent Z bug with TABLE-TOUCH** is the headline. **Branch `perception-topwrist-coop`. 3 commits this session (all PUSHED): `cd395d8`, `0ffeb79`, `078b418` (revise).**
>
> **One line**: (1) (key) **Reach-dependent Z solved** (user observation "overshoots the descent when near, undershoots when far" = robot_io FK z depends on arm REACH, cannot be fixed with the constant FK_Z_CORR) → **TABLE-TOUCH verification** (fk_verify_z.py drives a semi-automatic FK-touch button, records z_gap_io at the table via J2/J3 torque contact) 3 sweeps 20pt fit **slope −209mm/m**, workspace Z deviation **87mm → ~6mm**, `--z-reach-slope/-ref` **defaults −0.209/0.475 baked in**. (2) (key) **HW-limit silent-clamp bug fix** (base IK limit J6 ±180 > motor HW clamp ±90 → at the J1-yaw edge J6 silently mis-rolls ~15°, not caught by `_danger` → `base.lo/hi ∩ HW_LIMIT_DEG`). (3) **AUTO-GRASP = commit → align (J6 yaw roll at hover + simultaneous auto-center, dwell convergence, timeout → abort) → pure-vertical descend → grip**. (4) **Edge-voting Align yaw** (replaces minAreaRect 0/90° collapse) + color-agnostic OV mask. (5) auto-center **anti-windup** (far oscillation) + gentle **--lift-rate**. (6) **GDino spike evaluated, then reverted** (OWLv2 better at color discrimination). (7) **Adversarial design-verification workflow** (caught, with HIGH from 5 lenses, that the original look-at absolute-solve design was a regression).

---

## 0. Git / robot state
- Branch `perception-topwrist-coop`. **3 commits this session, PUSHED**: `cd395d8` (edge-yaw + initial AUTO-GRASP-yaw + graspz + per-cam OV readout) → `0ffeb79` (reach-Z + HW-limit + FK-touch + anti-windup + lift-rate) → `078b418` (revise, user).
- **Uncommitted = cross-session leftovers only** (unrelated to this session, separate cleanup): `calib_grip.py`·`measure_latency.py`·`joint_map.npz`·`scripts/train_cube_detector.py`·`src/perception*.py`·`uv.lock` + untracked `damiao_params.py`·`deepreach_sac_world.py`·`fit_wrist_calib*.py`·`gen_perception_data_*`. **docs/20–31 untracked** (handoffs, need a separate commit).
- **Robot**: tuning with repeated deploy restarts (port 8899). `.venv-deploy`. Camera symlinks `/dev/video_front`·`/dev/video_wrist` (udev re-enumeration absorbed automatically). Warning: a deploy launched as a detached background job exits intermittently → the user restarts it directly from the terminal (stable). Warning: GPU shared with Ray training jobs (~5–8GB) — coexisting with deploy is OK.

## 1. (key) Reach-dependent FK-Z error quantified and corrected (headline, `0ffeb79`)
- **Symptom** (user): AUTO-GRASP/RUN descent **goes lower than intended when the cube is near (over), not low enough when far**. Root cause = the descend gate stops at `z_gap_io = tcp_io.z − plane_z`, but `tcp_io.z` (robot_io Pinocchio FK + **constant** `FK_Z_CORR=0.03`) is **biased depending on arm REACH (radial distance)** → a constant correction is right at only one distance. `fk_bias` (= tcp_io − tcp_w) is +0.030 (constant) across all samples, so the only difference between the two FK chains is a constant = the reach error is rooted in the **shared FK (URDF/joints)**.
- **Verification = TABLE-TOUCH** (absolute reference, user's choice): the arm **descends straight onto an empty table (or a consistent reference surface) → J2/J3 torque contact → records z_gap_io at that fixed height → rises**. If z_gap_io **varies with distance** across several distances, that slope = the reach bias.
- **Result**: combined fit over 3 sweeps, 20pt: **raw slope −209 mm/m** (per-sweep −169/−211/−247, ±40mm/m = touch noise floor), ref 0.475. Workspace (0.29–0.72m) z_gap_io deviation **87mm → residual ~6mm**. Sign check: near reads z_gap_io high (over-descent) / far reads low (under) = matches the symptom.
- **FIX**: `--z-reach-slope`/`--z-reach-ref` → `tcp_io.z −= slope*(radial−ref)` (right after the read; z_gap_io, fk_bias, gates and auto-z are all corrected consistently). radial = `|tcp_io_xy − cfg.base_pos_xy(0,0.06)|`. **Defaults baked −0.209/0.475** (depends on this robot's current calib — **must rerun fk_verify_z.py after recalibration**). `--grasp-zgap-io` absorbs the constant offset (currently −0.014).
- Warning: **FK-touch trap**: at far distances (reach limit, ~0.68m+) the arm **cannot reach the table and the straining torque is recorded as a false contact** (spike samples) → valid **only at distances where the arm actually reaches the table**. Drop far-unreachable outliers when fitting.

## 2. (key) HW-limit silent-clamp bug fix (`0ffeb79`, adversarial review HIGH)
- The base IK limits (`cfg.arm_limit_*`: J4 ±90, **J6 ±180**, J2 −180) are **wider** than the motor HW clamp (`robot_io.HW_LIMIT_DEG`: J4 −80/+90, **J6 ±90**, J2 −170). IK proposes aim/roll not reachable by HW → `robot_io.command` np.clip silently truncates → **at the J1-yaw edge jaw-align requires J6 > ±90 → ~15° silent mis-roll**. `_danger` does not check joint limits + roll does not move the TCP, so the resid test structurally cannot catch it.
- **FIX** (`debug_deepreach.py`, right after `RuleBaseController` is constructed): `base.lo/hi ∩ HW_LIMIT_DEG` (mapping sim_rad = SIGN*deg2rad + OFFSET) → J4 lo → −80°, J6 → ±90°, J2 → −170°. Startup log `base IK limits ∩ HW clamp -> lo°/hi°`. Verified offline (confirmed the actual config has J6 ±180).

## 3. AUTO-GRASP align→descend + edge-voting yaw (`cd395d8` + `0ffeb79`)
- **Sequence** (`--autograsp-yaw`): commit (⊙centered) → **`align`** (at hover, roll J6 to the edge-voted yaw + **simultaneous auto-center** to re-center against roll disturbance) → **`descend`** (xy + J6 **frozen**, pure-vertical) → **grip**. Yaw is acquired **only at hover** (best framing), not reacquired during descent.
- **Stronger convergence**: entering descend = yaw reached AND centered AND **J5 not saturated**, stable for **consecutive `AG_ALIGN_DWELL` (0.4s)**. **timeout (6s) → abort-hold, not descend** (no descending over a bad alignment).
- **pure-vertical**: the committed grasp xy is **frozen at descend tick 1** (previously the live ghat was recomputed every tick → TCP crept forward due to cam-offset rotation). Geometry: the image rotation center of the J6 roll ≈ (0,0.44) = projection of the approach axis, so the cube is near the rotation center → no target-rotation correction needed; simply re-enabling the existing (0,0.44) auto-center solves it.
- **edge-voting yaw** (`vision_thread._edge_vote_yaw`): length-weighted mod-90 circular mean over cube boundary sides → replaces minAreaRect (2 extreme points → **0/90° collapse** under noise/occlusion/square window). minAreaRect fallback (when sides disagree). Synthetic smoke: under noise, edge 22° vs minAreaRect 0°. **Color-agnostic OV mask** (`_cube_bin_ov`: OWLv2 box + saturation blob) → independent of query color such as green/red (no longer red-mask only). Priority CNN-seg > OV box > red.
- **graspz live knob** (added after `0cb5326`, `cd395d8`): tune `--grasp-zgap-io` via dashboard grasp deeper/higher without restarting. **per-cam OV readout** (`ov_q_wrist`/`ov_score_wrist`): queries were already set for both front and wrist, but only front was displayed → exposes the wrist's independent score.

## 4. FK-touch verification tool (`0ffeb79`)
- **Semi-automatic FK-touch mode** (`--fk-touch-*`, dashboard button): straight descent from hover → J2/J3 torque contact → record z_gap_io → rise. Warning: **contact detection is ARMED only after descending 2.5cm (`FK_TOUCH_ARM_DZ`)** + **re-baselined** with the steady-descent torque at that point (fix for the descent-start acceleration transient exceeding dtau relative to the static hover baseline on the first tick and causing a **false contact at hover**). `--fk-touch-dtau` (2.0, contact threshold) `--fk-touch-max` (0.30, so it reaches even from the 25cm rest hover) `--fk-touch-step` (0.002).
- **`fk_verify_z.py` driver**: sends `/cmd?a=fktouch` → reads `fk_touch_samples` from `/state` → repeats → reach slope fit + prints `--z-reach-slope/-ref`. `fktouch_clear` resets the sweep. /state exposes `fk_touch` (phase) + `fk_touch_samples`.

## 5. auto-center anti-windup + gentle lift (`0ffeb79`)
- **anti-windup** (`--follow-cam-max` 7cm): at far reach the IK cannot deliver the cam-offset shift → the loop grows the offset without bound → **windup oscillation**. Clamping the cam-offset to ±7cm → far is stable (best-effort) **while near is not loosened** (near converges well inside this → `--follow-center-margin` stays tight = near precision). User confirmed: globally relaxing gain/margin sacrifices near → anti-windup is the right answer.
- **gentle lift** (`--lift-rate` 0.01 m/tick): 'Lift +z' as a **mode** — ramps the TCP-z target at a dedicated rate, then holds. Independent of the tracking/descent `--vmax/--amax`, not a snap (shaped). Total rise of 0.3m kept, re-click = stacks.

## 6. GDino spike (reverted) + adversarial design-verification workflow
- Spiked a **GDino** (Grounding DINO-tiny) open-vocab backend behind the OpenVocabModel interface (`--ov-backend gdino`). Offline: infer ~179ms (2x OWLv2's 92ms), color discrimination clear only with competing queries (even a single "red cube" boxed green at 0.58). **User on the real robot: lower accuracy → REVERTED** (`front_openvocab.py` git checkout, debug_deepreach flags/branches removed). OWLv2 kept.
- **Adversarial workflow** (multiple review passes, 1.57M tokens): attacked the "measured look-at → absolute back-solve → straighten" Align redesign with 5 lenses. 5-lens HIGH convergence = **the absolute back-solve reintroduces the unidentifiable wrist extrinsic = regression** (gimbal look-at only aims the cam, it does not move the jaw). → Design discarded, pivoted to keeping the calib-tolerant servo (auto-center) + same-pose J6 roll. **The workflow blocked a regression before it was implemented.**

## 7. Gotchas
- (key) **reach-Z defaults −0.209/0.475 depend on this robot's current calib** — after recalibration (joint zero / FK_Z_CORR change) **must refit** with `fk_verify_z.py`.
- (key) **FK-touch only at distances where the arm actually reaches the table** — at the reach limit (~0.68m+) straining becomes a false contact. Drop far outliers.
- (key) **FK-touch contact detection is armed after a 2.5cm descent** (the descent-start acceleration transient false-fires at hover). dtau is relative to the steady-descent re-baseline.
- (key) **After the HW-limit intersection** `_danger` legitimately catches over-extension and the J6 silent mis-roll is gone.
- (key) **pure-vertical descent**: freeze the committed xy at descend tick 1 (no live ghat recomputation = forward creep).
- (key) **auto-center far oscillation = windup from IK delivery failure** → anti-windup cam-offset cap (near margin stays tight).
- **edge-voting yaw**: if sides disagree (oblique/noise), minAreaRect fallback. Color-agnostic = OV box saturation blob.
- **FK-touch samples live in process memory** — lost when deploy exits (save the fit/records before restarting). The driver output is the backup.
- **GDino reverted** — `front_openvocab.py` is committed OWLv2-only. Retrying requires reimplementing the spike.

## 8. Open / next
1. **Remaining HW verification**: reach-Z is verified. **Simultaneous align-centering / anti-windup / lift-rate / fktouch full cycle** are at the bring-up stage — confirm with repeated real-robot runs.
2. **AUTO-GRASP full cycle**: whether align (yaw + center stable) → pure-vertical descend → grip lands, aligns and grasps correctly at each distance (whether the landing height is distance-independent after the reach-Z correction).
3. **Near auto-center precision**: if still loose, tighten with `--follow-center-margin 0.08` (anti-windup covers far).
4. **Commit docs/20–31**: separate cleanup of the untracked handoffs.
5. **Cross-session leftovers** (§0): clean up calib_grip/perception*/damiao_params/fit_wrist_calib*.
6. Root lever (still): front-cam XY calib / wrist extrinsic (unidentifiable).

## Run (next session, reach-Z defaults baked → no flags needed)
```
.venv-deploy/bin/python deploy/debug_deepreach.py --detector-device cuda \
  --front-openvocab --wrist-openvocab --front-roi 0.4,1.0,0.0,1.0 --track \
  --grip-force 0.05 --grip-torque-target 0.8 --vmax 5 --amax 8 \
  --x-corr 0.0235 --safety-reach-resid 0.15 --rate 30 --front-ov-period 0.5 \
  --cmd-lpf-tau 0.15 --action-chunk 20 --chunk-decay 0.9 \
  --follow-low 0.07 --follow-descend-step 0.004 --follow-move 0.04 --io-thread --lookat-j5-rate 0.05 \
  --grasp-zgap-io -0.014 --autograsp-yaw
```
Flow: Capture → Follow (⊙CENTERED) → **AUTO-GRASP ▶▶** (align: J6 yaw + simultaneous centering, dwell convergence → pure-vertical descend → grip). reach-Z corrected automatically. FK-Z re-verification = `python3 deploy/fk_verify_z.py` (empty table, ENTER at 3 distances → fit). Live = `curl localhost:8899/state`. Details: docs/30.

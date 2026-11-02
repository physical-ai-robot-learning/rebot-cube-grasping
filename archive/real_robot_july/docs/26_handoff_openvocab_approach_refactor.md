# Handoff — (key) OpenVocab detection (front/wrist) + full redesign of look-at approach (gimbal-safe) + kinematic safety stop + adversarial review ×2 and project cleanup (2026-07-16, session 16)

> Continuation of docs/25 (session 15: gimbal look-at RnD, perception-stale fix, gripper recalibration). **A long conversational session, collaborating in front of the HW.** This session was mostly code (offline verification done, **not verified on HW**) + refactoring, bug fixes, and project cleanup based on adversarial review.
>
> **One line**: introduced OWLv2 open-vocab as a "dynamic ROI provider" (front+wrist, color-agnostic detect-then-track, live dashboard queries) + redesigned look-at approach to be gimbal-safe (J6 IK-freeze) + kinematic `_danger` safety stop. Two adversarial reviews (deploy diff 26 findings + whole project with multiple review passes) led to 8 bug fixes and 5 dead modules archived. 5 commits (**not pushed**), waiting for HW verification.

---

## 0. Git / robot state
- Branch `perception-topwrist-coop`. **5 commits this session, not pushed**:
  - `5dde977` OWLv2 open-vocab detector module (`deploy/front_openvocab.py` new)
  - `110b118` OV front/wrist wiring + detect-then-track + approach/gimbal/safety (`debug_deepreach.py`+`debug_dashboard.py`)
  - `e681efe` cleanup: 5 dead modules to `_archive/` + removed make_top_camera
  - `346b4be` fix(deploy): robot_io velocity-obs bug (step-0 spike + dt)
  - `f8bc6e5` fix(rl): pick_env prev_action/act_hist reset leak + was_grasped dead
- **Uncommitted (bug fixes applied, mixed with previous-session work → uncommitted)**: `deploy/cube_detector_io.py` (multi-class decode fix), `src/.../perception_cnn.py` (heatmap convention fix), `deploy/deepreach_sac_world.py` (untracked, dead-cam fail-stop). Warning: **committing these 3 files will bundle previous-session work too** (non-interactive, so hunks cannot be split) → recommended to commit in a perception/detector/sac-world context.
- Other uncommitted items (perception.py roi/s_min, joint_map, calib_grip, measure_latency, damiao_params, fit_wrist_calib, docs 20/22-25, gen_*, submit_*, etc.) = previous sessions, unchanged.
- Warning: **transformers 5.13.1 is installed in `.venv-deploy`** (for OWLv2). Not imported when unused (lazy) → no effect on baseline. Not recorded in uv.lock/pyproject → recommended to add to deploy deps when committing.

## 1. (key) OpenVocab detection (headline) — `deploy/front_openvocab.py` + debug_deepreach wiring
**Concept**: OWLv2 ("red cube", etc.) **boxes** the object zero-shot (zero labels, no sim2real gap) (where to look) → **HSV centroid** only inside that box (exactly where, mm). Distractors (tape/hands) are outside the box and thus excluded. **Bypasses the problem of the front 5k CNN not transferring to the real robot** (HSV transfers; OV acts as the gate).
- **Structure**: one shared `OpenVocabModel` (a single heavy ViT) + per-stream `OpenVocabROI` (period cache 0.35 s + thread-safe query). front and wrist share it (one set of weights); wrist reuses `--front-ov-thresh/period`.
- (key) **detect-then-track**: OWLv2 re-detects only every 0.35 s (periodic anchor, `stamp`); in between, the **box is recentered every frame by the HSV centroid** (`_recenter_frac_roi`) → front est is **fully per-frame live** (offline: moving cube tracked at 0.9 px vs static box 0→14 px sawtooth). On a single dropout, the tracking box is kept (prevents loss).
- (key) **Color-agnostic centroid** (`_sat_blob_centroid`, session 16 bug fix): a **saturation-based** blob inside the box (table = low saturation vs object = high saturation) → the center of the object **of any color**. Fixes the hard-coded red_mask (bug where a purple cube was treated as red). Verified: no regression on red + blue recolor gives the same position.
- **Live dashboard query**: `OpenVocab query:` input box → front and wrist re-detect immediately (version guard makes it take effect at once). When the object changes → lift to rest, then re-approach the new object.
- **flags**: `--front-openvocab`/`--wrist-openvocab` (opt-in), `--front-ov-model/thresh(0.25)/period(0.35)`. Warning: wrist OV **replaces** the sd3 descent (u,v) (no depth/CNN yaw, Align = seg yaw) → off by default since sd3 is untouchable.

## 2. (key) Full redesign of look-at APPROACH (gimbal-safe)
Redesigned the docs/25 approach based on HW feedback (J5/J6 whipping around dangerously, descending all the way to the cube). **Derived through verification that refuted several attempts:**
- **Stop at +N cm hover** (`--lookat-approach-stop`, default **0.05** = 5 cm) — does not go all the way to the cube; hands off to Align/Final/close. Offline sweep: at ≥5 cm, look-at aim 0° and position accurate; at 3 cm, aim 15° + 7 mm undershoot (offset-cam geometry).
- **Smooth continuous descent**: xy alignment is also a proportional ramp (no gate/steps, monotonic), directly tracks the live front est (deadband 4 mm).
- (key) **gimbal-safe aim** (default): this rig has an **offset-mounted camera** → aiming needs J4, J5, and J6 together (the real whip = **J6 roll 87° swing**). **Freeze J6 inside the IK** (`_gimbal_lookat_ik freeze=(5,)`) → aim with J4+J5 (whip removed, aim~26° = within FOV). `--lookat-freeze-j6` (default on), `--lookat-aim-gain` (3.0), `--lookat-astep` (0.07), `--lookat-j5-rate` (opt-in, default 0 = off; verified that the cap worsens aim).
  - Warning: **verification lesson**: freezing J5 → aim breaks at 63° (J5 = pan axis); freezing J6 **outside** the IK also breaks at 81° → J6 must be removed **inside** the IK for J4+J5 to aim. J5 is required for aiming and cannot be frozen.
- **`--lookat-straight`** (alternative): J5 and J6 fully fixed, straight-down + xy tracking (aim 21°, zero wrist motion). Safest when active aiming is not needed.
- (key) **taper removed**: because of approach_stop, `--lookat-taper` was fully redundant with `--lookat-straight` → deleted, `do_gimbal = not lookat_straight`.
- **flags**: `--lookat-{hover(0.12)/approach-stop(0.05)/descend-step/xy-slow(0.06)/deadband(0.004)/fuse/iters(80)/aim-gain/astep/freeze-j6/j5-rate/straight}`.

## 3. Two-layer safety (`_danger` new)
- **Re-approach** (deviation): if the cube moves away during descent (judged by measured tcp_w[2], prevents infinite loops) (`--lookat-reapproach-dist` 0.12) → lift to rest, then **retry**.
- (key) **Danger supervisor** (`_danger`, `--safety-auto-rest` default on): every tick **judges danger via FK/IK** — (1) IK fails to reach xy (residual>0.05 = out-of-reach/over-extension) (2) TCP below the descend surface (penetration). On violation, lift to rest + **stop** (3-tick debounce). Warning: key lesson: joint-limit checks are **meaningless on a clamped qf** (gimbal J5~90° = limit, so always a false positive) → removed, replaced by residual. The floor is **based on base-FK surf_z** (table_z=0 gives false positives due to the z_bias frame mismatch).
- 3 lift-to-rest blocks (object change/deviation/danger) → unified into the `_return_to_rest(msg, halt)` helper. Warning: io.initial_liftup is **blocking ~2 s** (STOP/ESTOP not serviced, same as Reset) — acceptable since it is moving to a safe pose.

## 4. Adversarial review ×2 + project cleanup + 8 bugs
- **Review A** (deploy diff, 8 components→verify): 26 confirmed → applied (TOCTOU race, stamp-on-success, tracker collapse, dead args, `_return_to_rest` unification, etc.). Commit `110b118`.
- **Review B** (whole project, multiple review passes, UNCERTAIN 0): dead modules + bugs.
  - **archive→`_archive/`** (commit `e681efe`): `scene/builder.py` (orphan), `deploy/calib_extrinsic.py` (superseded by _known), `scripts/{collect_vla_dataset,wrist_ibvs_sim,wrist_servo_sim}.py`. Removed `make_top_camera`.
  - **8 bugs fixed (all verified)**: robot_io step-0 qvel spike + dt mis-scaling (`346b4be`) · pick_env prev_action/act_hist reset leak + was_grasped dead (`f8bc6e5`) · **cube_detector_io multi-class decode ×2** (`_pick_cls`, uncommitted) · **perception_cnn heatmap convention half-stride** (uncommitted, no effect on existing ckpts) · **deepreach_sac_world dead-cam fail-stop** (uncommitted).

## 5. Deploy status summary
```
front(fixed)+wrist(eye-in-hand) → AsyncPerception(30Hz)
  front: HSV / CNN / (key) OV detect-then-track → cube_world(live, --track)
  wrist: sd3 CNN descent (untouchable) / OV(opt-in)
control(100Hz CAN): Capture→RUN(descent latch,sd3)→Align→Final→Grip→Lift + Follow + (key) Approach(gimbal-safe) + safety(stale/re-approach/_danger)
motors: POS_VEL(vlim32)+JointSlew shaper. KP_APR=54(RAM)/config A(flash)
```
| Component | HW status |
|---|---|
| sd3 descent, RUN, gripper force-servo | HW verified (earlier), kept untouchable |
| OpenVocab front/wrist | offline only, waiting for HW |
| approach (gimbal-safe/straight/+5cm/re-approach/danger) | offline only, waiting for HW |

**Run**: `.venv-deploy/bin/python deploy/debug_deepreach.py --detector-device cuda --front-openvocab --wrist-openvocab --front-roi 0.4,1.0,0.0,1.0 --track --grip-force 0.05 --grip-torque-target 0.8 --vmax 12 --amax 30`

## 6. Key constraints / unresolved levers
1. (key) **Motor latency KP_APR 54 (~700 ms)** = fundamental limit on tracking speed and look-at depth. **config A** (fast / ~1° tremor at rest) vs **54** (stable / slow). **MIT mode is the proper solution** (the lerobot damiao driver already implements `_mit_control_batch` — dynamic kp + velocity FF + independent kd for fast+damped; the KP_APR flash dilemma disappears). **Not migrated** (needs robot_io wiring + re-verification + the gripper must keep the force path).
2. **Offset-mounted wrist camera** → active aiming needs large motion of J4/J5/J6 together = inherent whip. straight-down (no motion) or gimbal-safe (J6-freeze, aim 26°).
3. **Minimizing POS_VEL latency**: vlim is already max (not the bottleneck); the real caps are **shaper `--vmax`/`--amax`** (currently set low) + the KP_APR floor.

## 7. Next actions (priority)
1. (key) **OpenVocab + approach HW verification** — front/wrist cyan box following, color-agnostic (purple/blue) tracking, gimbal-safe vs straight descent stability, +5 cm stop, re-approach and `_danger` behavior.
2. **Push the 5 commits** (if HW is satisfactory).
3. **3 uncommitted files** (cube_detector/perception_cnn/sac_world) commit in a perception context.
4. (optional, tracking-performance lever) **restore KP_APR config A** or **migrate to MIT**.
5. Record transformers in deploy deps.

## 8. Gotchas
- **3 cases where verification refuted an approach** (paid off this session): J5-freeze / J6-outside-freeze break aim (offset-cam), J5 rate-cap worsens aim, the danger joint-limit check false-positives on a clamped qf. **Keep the habit of offline `scratchpad/test_*.py` before HW.**
- **wrist OV and approach are opt-in; sd3 descent is untouchable** — without --wrist-openvocab in the command, it is sd3.
- **OV color-agnostic**: saturation rather than red_mask → white/grey objects cannot be distinguished from the table (colored objects only).
- **Trap with the 3 uncommitted files**: bug fixes applied but not committed — when using these paths (multi-class detector/sac-world) on HW, the fixed versions live in the working tree (be careful with checkouts).
- **KP_APR 54 in RAM / config A in flash** — after reboot it is config A.
- compile OK (all files), OV/approach/danger verified offline, **not verified on HW**.

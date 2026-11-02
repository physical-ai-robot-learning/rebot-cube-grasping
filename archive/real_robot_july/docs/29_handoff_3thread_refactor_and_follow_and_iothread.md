# Handoff — (key) 3-THREAD + Blackboard refactor (docs/28) Step 0→3b implemented + Follow descend-FSM + LPF/Action-Chunking smoothing + arm-oscillation root cause = wrist-IBVS/gimbal limit cycle (perception is the real blocker) (2026-07-20, session 18)

> Continuation of docs/27 (S17: grasp-point FK/IK). **Very long session.** Start = adversarial review of the uncommitted docs/27 diff → HW re-verification of the oscillation → smoothing → **re-architecture of the deploy control stack into 3 threads + Blackboard (user instruction)**. **12 commits this session (all unpushed), branch `perception-topwrist-coop`.**
>
> **One line**: (1) The arm-oscillation root cause is not the io-xy_off metric but the **`--lookat-ibvs-xy` wrist fusion + gimbal-gate limit cycle** (= reproduces docs/19 S10); the real blocker is **front-cam perception (~10cm error, frame edge, wrist false-detect)** — not control. (2) Per user instruction, **refactored the monolithic debug_deepreach (2140 lines) into 3 threads + Blackboard** (docs/28 design) → **Steps 0-3b implemented and committed** (blackboard/kinematics/vision_thread/io_thread + `--io-thread` opt-in wiring, **OFF = byte-identical**). (3) **Follow redesign** (descend to 7cm when stable, rise and re-track on movement, J6 fixed) HW-verified. (4) LPF + Action Chunking smoothing.

---

## 0. Git / robot state
- Branch `perception-topwrist-coop`. **12 commits this session (`db545fe`..`ecd5e85`), 17 unpushed in total.** Order:
  `db545fe` (front_openvocab OV offline + fp16) → `bb6f28b` (cube_detector_io multi-class + crash + no_grad) → `68b2417` (approach look-at grasp-point + smoothing stack) → `cc2a938` (dashboard broken-pipe guard) → `bdce1fb` (**Step0-1** blackboard + kinematics) → `61de291` (kinematics wiring + approach COARSE→FINE) → `b0ac0dd` (**Step2** vision_thread) → `a4f9ec5` (**Follow** descend FSM + J6) → `485b08a` (**Step3a** io_thread module) → `ddfa6ba` (**Step3b** --io-thread wiring) → `d4afa70` (gates.io telemetry) → `ecd5e85` (Follow RISE = --follow-move, jitter fix).
- **Uncommitted = cross-session leftovers only** (same C/D as docs/27 §0): `calib_grip/measure_latency/train_cube_detector/perception*/joint_map/uv.lock` + untracked `damiao_params/deepreach_sac_world/fit_wrist_calib/gen_perception_data_*/submit_train_detector.sh` + **docs/20-28 untracked** (handoffs + design doc, not committed — needs separate cleanup). All deploy code is committed.
- **Robot**: camera symlinks `/dev/video_front`→video2, `/dev/video_wrist`→video0. `.venv-deploy`. **At session end the debug_deepreach process was down** (port 8899 not open — presumably Ctrl-C). Warning: GPU sharing: if `rebot_eval` training occupies the GPU, OWLv2/perception slows down; run `nvidia-smi` before deploy.

## 1. (key) 3-THREAD + BLACKBOARD refactor (headline, docs/28 design)
**User instruction**: monolithic `debug_deepreach.py` (2140 lines, everything entangled, impossible to debug) → split into Vision / Robot-IO / IK·FK+Control threads + Blackboard, with PyCharm step debugging + logging. **Design = `docs/28_deploy_3thread_refactor_design.md`** (based on a multi-pass parallel mapping of the current code). User decisions: **Control rate 30Hz** (matched to perception), single coarse lock, Vision kept single-threaded (GPU sharing).
- **[done Step0] `deploy/blackboard.py`**: thread-safe Blackboard. Sections proprio/perception/perception_ctx/command/mode/debug, single RLock, **deepcopy snapshot** (safe for nested tm dict). Self-test passes. Commit `bdce1fb`.
- **[done Step1] `deploy/kinematics.py`**: verbatim extraction of `_project_wrist_unvn`/`_wrist_lookat_j4`/`_gimbal_lookat_ik`/`_danger` + `WRIST_*` calib + **`set_wrist_calib()`** (replaces the calib handler's cross-module `global`). `bdce1fb` + `61de291`.
- **[done Step2] `deploy/vision_thread.py`**: verbatim extraction of AsyncPerception + camera/detection/annotate helpers (~490 lines), `set_ctx/snapshot` interface kept. 9 import-backs. Commit `b0ac0dd`. **HW-confirmed** (Follow works correctly on the refactored code = extraction preserves behavior). Warning: bug lesson: missing `re` import → passed py_compile/import but runtime NameError (inside a function). **Every later extraction requires a synthetic-input smoke test** (an import test does not catch it).
- **[done Step3a] `deploy/io_thread.py` RobotIoThread**: single-bus read-then-write. Owns shaper/LPF/chunk + grip force-servo + scripted io_task + estop. **mock-bus offline smoke passed** (proprio publish, shaped cmd @100Hz, force-servo, io_task, estop). Commit `485b08a`. **Verified first in an unwired state**.
- **[done Step3b] `--io-thread` opt-in wiring** (commit `ddfa6ba`): control-loop IO gated by flag. **Default OFF = byte-identical** (every branch is `io_tj is not None`; if None, fallback → RUN/Follow unchanged). ON: spawns RobotIoThread (it has its own shaper, so control shaper.reset is harmless), `read_pr()` = bb.proprio, `send_arm` → bb.command (raw), scripted → `run_scripted` (io_task + block), force-servo → grip_mode, estop → bb.mode. `d4afa70` = gates.io (bus_hz) telemetry. **HW: Follow behavior confirmed with --io-thread** (proprio goes through bb = proves IO-thread publishing). Warning: ON-path HW-pending: rate coupling (qvel finite-diff, auto-z EMA tau, docs/28 §6 Step5) + proprio-dead fail-stop gap.
- **[ ] Step4** control_thread.py + run_deploy.py (FSM extraction, sole ownership of base, single writer for cube_top_z). **[ ] Step5** retune rate constants (qvel, EMA tau — the rate changes when io-thread is ON).

## 2. Follow redesign (HW-verified) — commits `a4f9ec5` + `ecd5e85`
User request: when Follow sees the cube, descend to 7cm; when the position changes, rise and re-track. **straight-down + stable front track = path to reach 7cm without look-at oscillation** (RUN/sd3 untouched).
- **FSM**: STABLE (movement < `--follow-deadband` 1cm) → descend from the `--follow-z` (16cm) hover down to `--follow-low` (7cm) at `--follow-descend-step` (0.004/tick). MOVED → rise and re-track. **Referenced to the auto-z surface (`base.cube_top_z`)**, so +Ncm is the real height (old follow used raw plane_z).
- **J6 fixed**: straight-down leaves J6 (roll) as a free gimbal → ik_oriented shakes it → freeze `qf[5]=current J6`.
- (key) **`ecd5e85` RISE jitter fix**: in the initial implementation the RISE trigger was the tracking deadband (1cm), so **front jitter was falsely detected as "movement" → RISE↔descend cycle** ("goes to Rest, then back to Follow"). → **RISE separated with `--follow-move` (4cm)**: jitter/drift (deadband..move) only re-tracks xy and keeps descending; only real movement (≥ move) triggers RISE. **Warning: not HW-reconfirmed** (process went down after the fix). There are signs that the faster loop of --io-thread checks the move gate more often and amplified it.

## 3. Motion smoothing — LPF + Action Chunking (commit `68b2417`, opt-in)
In `send_arm`, order chunk → LPF → existing JointSlew accel-shaper:
- `--cmd-lpf-tau` (1st-order LPF, rate-correct EMA, removes high-frequency jitter/jerk) · `--action-chunk N` + `--chunk-decay d` (exp-weighted temporal ensemble, spreads target jumps). Reset on snap/large jump.
- Warning: **chunk-decay interaction**: even with N=20, decay 0.5 is exponentially weighted so effectively ~5 taps. **To make 20 count, raise decay** (0.9 = 20 taps contribute). Settings used by the user: `--cmd-lpf-tau 0.15 --action-chunk 20 --chunk-decay 0.9`. `--vmax 5 --amax 8` (old 12/30 was "too fast and too strong" → oscillation).

## 4. (key) HW findings — arm-oscillation root cause + real blocker = perception (S17b/c)
- **Arm-oscillation root cause is not the io-xy_off metric**: the io-xy_off fix is valid (on a stable target xy_off 7-9mm, tcp sub-mm). The oscillation is the **`--lookat-ibvs-xy` wrist fusion + gimbal-gate limit cycle** (gimbal → wrist sees → IBVS pulls cxy ±10cm → moves → loses it → returns). = **reproduces docs/19 S10** ("wrist fusion + look-at diverges, sd3 straight descent is stable").
- (key) **The real blocker = front-cam perception (not control)**: the cube is **detected at the edge** of the front frame → oblique ray → world XY **~10cm inaccurate + fluctuating** (during the session x 0.08→0.21→-0.14, y 0.35→0.675). The wrist also **false-detects** on the real robot (redmask picks a central feature, not the actual cube). → Any control is meaningless on top of inaccurate input. **Next lever = front calib / detection reliability** (edge distortion, extrinsic, wrist shape-detect); the refactor improves debuggability, it is not an accuracy fix.
- **COARSE→FINE handoff** (commit `61de291`, approach): front coarse → once the wrist acquires, the wrist owns the target and slowly fine-aligns (`--lookat-ibvs-step` 1cm/tick); front re-coarses only beyond `--lookat-reacquire` (15cm). + **hold-on-loss** (no snap-back to front when the wrist loses it) + **WRIST-ACQUIRE descend gate** (`--lookat-descend-uv`: hover dwell before wrist centering). All compile-validated; approach not completed on HW due to perception noise.

## 5. Adversarial-review pre-fixes (docs/27 uncommitted diff, before HW) — commit `61de291` etc.
multi-pass workflow. 2 HIGH pre-blocked: **need_align gating regression** (§4 align_seg skipped in RUN → default seg yaw-latch starved → Align yaw collapse) → `need_align=aligning or (auto_running and yaw_src=="seg")`. **lookat_jaw_fwd default footgun** (None → jaw_fwd 0.01, approach needs JAW_FWD 0.06) → fallback to JAW_FWD. + L2010 stale comment. + restored cube_detector_io no_grad (§4 perf).

## 6. Open / next (priority)
1. (key) **HW reconfirmation of the Follow cycle** (`ecd5e85` --follow-move fix): rerun → Capture → Follow; check it settles at 7cm without a RISE↔descend cycle. **If it still cycles, A/B without `--io-thread`** → isolate io-thread rate coupling (cycles with / stable without = Step5 retune confirmed). Command below.
2. **--io-thread RUN grasp A/B**: whether RUN descend accuracy (`zgap_io≈0`) and motion are equivalent with/without io-thread → full Step3 verification. If worse, Step5 rate retune (EMA tau/qvel).
3. **Step4/5**: control_thread.py + run_deploy.py (FSM extraction), rate constant retune.
4. **Perception root cause** (the real lever): front-cam edge distortion/extrinsic + wrist shape-detect reliability. Can be worked around by placing the cube centrally (front calib verified region ~11mm).
5. **Commit docs**: docs/20-29 untracked → commit. Clean up cross-session leftovers (C/D).

## 7. Gotchas
- (key) **--io-thread OFF = byte-identical** (io_tj None fallback). ON is opt-in and HW-pending. If problems occur, removing the flag gives `a4f9ec5` behavior = safe.
- (key) **Extraction verification**: py_compile/import does not catch undefined names inside functions (e.g. `re`) → **synthetic-input smoke test required** (already established with the 9 vision helpers and the io mock-bus).
- (key) **Arm oscillation = wrist fusion / gimbal limit cycle** (S10), not io-xy_off. Stable descent = straight-down front-target ≈ RUN (sd3).
- (key) **Real blocker = perception** (front 10cm, wrist false-detect), not control. The refactor is for debuggability.
- **Follow RISE uses `--follow-move` (4cm)**, not the tracking deadband (1cm) (prevents jitter cycles). Referenced to the auto-z surface.
- **chunk-decay governs action-chunk N** (at 0.5, N is meaningless; at 0.9, 20 taps are alive).
- STALE fail-stop does not touch Follow (only auto/align/approach). The Follow cycle is RISE jitter.
- kinematics WRIST_* can only be overridden via `set_wrist_calib()` (cross-module global does not work).

## Run (next session §6-1 reconfirmation = first command)
```
.venv-deploy/bin/python deploy/debug_deepreach.py --detector-device cuda \
  --front-openvocab --wrist-openvocab --front-roi 0.4,1.0,0.0,1.0 --track \
  --grip-force 0.05 --grip-torque-target 0.8 --vmax 5 --amax 8 \
  --x-corr 0.0235 --safety-reach-resid 0.15 --rate 30 --front-ov-period 0.5 \
  --cmd-lpf-tau 0.15 --action-chunk 20 --chunk-decay 0.9 \
  --follow-low 0.07 --follow-descend-step 0.004 --follow-move 0.04 --io-thread
```
Capture → Follow: confirm it settles at 7cm without a RISE↔descend cycle. If it cycles, A/B without `--io-thread`. Live = `curl localhost:8899/state`. Details: docs/28.

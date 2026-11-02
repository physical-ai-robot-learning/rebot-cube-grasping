# Handoff — (key) debug_deepreach.py (2086-line monolith) → structured `deploy/main.py` full refactor (P0–P3 done, real-robot startup confirmed) + reach-Z refit (FK_Z_CORR −0.063) + auto-z negative z_bias fix (2026-07-22, session 21)

> Continuation of docs/31 (S20: reach-Z TABLE-TOUCH, HW-limit, AUTO-GRASP). Per user instruction, **promoted the deploy control stack from a debug tool to the main program + thorough restructuring**. **Branch `deploy-refactor-main` (branched from perception-topwrist-coop). 13 commits this session, all local and unpushed.**
>
> **One line**: (1) Decomposed the 2086-line `main()` monolith into the `src/rebot_sim2real/deploy/` package (cli · control/ · vision/ · io/ · kinematics · main_dashboard) — **all CNN detectors dropped (OpenVocab settled), always async, all rule-based features preserved**. Strangler (shims) keep the old program working without interruption. (2) P2 adversarial review (11-dimension find→verify) **drift 0**. (3) The new `deploy/main.py` **confirmed to start end-to-end on the real robot** (io.connect, cameras 30Hz, OWLv2, io-thread, control loop, dashboard). (4) The **reach-Z calib drift** that surfaced during startup (unrelated to the refactor; A/B shows the old program is the same) refit with a clean 4pt table-touch (slope +0.0676 / ref 0.486 + FK_Z_CORR 0.03 → −0.063 + grasp-zgap-io 0). (5) Making FK_Z_CORR negative flipped the sign of fk_bias and triggered an **auto-z z_bias clamp bug (assumes ≥ 0)** (Follow hover ~4cm low) → fixed by allowing negative z_bias.

---

## 0. Git / robot state
- Branch `deploy-refactor-main`. **13 commits, all unpushed** (`a0100a4`..`5d45ddb`). No refactor-related uncommitted changes (leftovers = docs/20-32 untracked + scripts/submit_train_detector.sh, unrelated to the refactor).
- Robot: HW tuning with repeated deploy restarts (port 8899). `.venv-deploy` (torch 2.10+cu128, cv2, transformers/OWLv2, DM SDK). **System python3 has no torch → always use `.venv-deploy/bin/python`.** Camera symlinks `/dev/video_front`→video3, `/dev/video_wrist`→video1. GPU shared with Ray jobs (~3.5GB headroom, tight for OWLv2).
- Warning: **fk_touch_samples live in process memory** — lost on deploy restart. Warning: **code/cli default changes take effect only after restart** (no hot-reload in the long-running process).

## 1. (key) Refactor stage details (P0–P3)
Design: decompose `debug_deepreach.py main()` (argparse 99-592 + setup 593-982 + nested closures + control loop 984-2074, FSM) by concern. **Strangler**: module-alias shims left at the old deploy/ paths (`sys.modules[__name__]=_mod`, preserving USE_MJPG/WRIST_* global mutation) so the **old program and tools keep working without interruption**. Offline smoke test at each stage (real base (CPU) + mock; docs/28 lesson = an import test does not catch runtime NameError).

- **P0** `a0100a4`: dead/RL → `_archive`. `deepreach_sac_world`·`run_pick`·`dry_run` (SAC)·`deep_reach_world` (rule-only predecessor sibling). The main program has no SAC (deepreach is in name only).
- **P1** `dbe34de`: 8 runtime modules → `src/rebot_sim2real/deploy/{,io/,vision/}` + 8 shims. `robot_io·io_thread → io/`; `front_gripper_z·front_openvocab (→openvocab)·vision_thread (→perception_thread) → vision/`; `blackboard·kinematics·debug_dashboard (→main_dashboard) → root`. Removed perception_thread sys.path manipulation + `from . import front_gripper_z`. **No logic change** (CNN removal happens in new P2 code).
- **P2.1/2.2/2.3a** `4bbff70`: `cli.py` (argparse → build_parser() + frozen `DeployConfig` (fields = dest, defaults only in argparse, from_namespace)) — **7 CNN flags dropped** (wrist-detector·align-seg·align-detector·front-detector·lookat-ibvs·lookat-vn·yaw-src). `control/context.py` (~70 FSM state variables → dataclasses Grip/Track/Follow/Align/YawLatch/RunAutoGrasp/Approach/FkTouch/Lift + `ControlState`; **old variable → new field crib** docstring). `control/robot_io_facade.py` (merges the 4 closures send_arm/read_pr/run_scripted/estop, io-thread/sync branching in one place). Also resolved the `--help` %-format bug.
- **P2.3b** `bec0fb4`: `ControlLoop.__init__` (config-derived state init) + `preamble()` (984-1199 line-for-line, **always async, CNN-free**: proprio → FK → z-reach-slope → auto-z → perception/track/stale → seg yaw-latch → wrist-xy/3-Z → deep_reach_step). `tick.py` (per-tick computed values) · `telemetry.py` (build_snap) · `constants.py`. **Dropped**: wcos4/wsin4 · wdepth · cnn_yaw_deg/yaw_src · sync path → cli also removes --rate/--async-perception.
- **P2.3c-①** `5a4319f`: `commands.py` (command if/elif 1259-1556 → handle(loop,cmd), abort=True) + `ControlLoop.return_to_rest` + `facade.reset_shaper`. Removed the capture sync branch and the align IBVS reset; calib_wrist goes through the facade (avoids the io-thread hazard).
- **P2.3c-②** `a31b1b8`: `modes.py` (autograsp sequencer 1205-1257 + `step(loop)` dispatch + 7 steps run/align/follow/approach/lift/fk_touch/grip) + `run()` wiring (preamble → sequencer → command ? handle : step → sleep, finally = facade.shutdown motors first → perception → dash). Align IBVS branch removed entirely, yaw = latch > seg, follow `step` → `center_step` (shadowing), return_to_rest(); continue → return, removed prev_front (dead), facade.grip_force_servo/shutdown, tick.perc_stale. **Package runnable.**
- **P2 review hygiene** `02ed0cc`: removed unused `TrackState.cube_world`.
- **P3** `f29d62a`: `src/.../deploy/main.py` = setup/build wiring (port of 593-982: config · base · HW-limit intersection · wrist-calib · calib · plane_z · OWLv2/HSV detectors (no CNN) · rest-lift IK · io.connect · dash · AsyncPerception (wrist_det/align_det/front_det=None) · shaper · bb · io_tj · RobotIoFacade · ControlLoop.run). `deploy/main.py` = thin launcher (src path + call). wrist (u,v) callback = OWLv2/redmask (depth/yaw=0 → Align uses seg edge-voting).
- **cli defaults baked** `f83f6ed`: the working flags from docs/31 made cli defaults → **bare `.venv-deploy/bin/python deploy/main.py` gives the working settings**. Detection/modes (front/wrist-openvocab · track · io-thread · autograsp-yaw) = BooleanOptionalAction default ON (disable with --no-…), motion (vmax5/amax8/cmd-lpf0.15/chunk20-0.9/j5-rate0.05) · grip-force0.05 · safety-resid0.15 · front-roi. Warning: calib-dependent values (grasp-zgap-io · z-reach-slope/-ref · x-corr) are also baked → update on recalibration.

## 2. P2 adversarial review
11-dimension find→verify workflow . **CONFIRMED drift 0.** The finders correctly classified every candidate diff as "intended change / pre-existing (e.g. stale help that was also in the original) / harmless". Warning: if the workflow returns an empty result, confirm the actual work via the sizes of journal.jsonl + agent-*.jsonl.

## 3. (key) P3-HW startup (new main.py confirmed on the real robot)
Started on the real robot with `.venv-deploy/bin/python deploy/main.py …` → **io.connect OK (proprio flowing, io_bus 98.7Hz), cameras perc_hz 29.9, OWLv2 loaded (ov_q set), io-thread, control loop DEEP-REACH, dashboard :8899** all confirmed live. **The refactored stack works end-to-end.** The offline integration smoke (HW classes + run mock, real calib.npz) also passes.

## 4. reach-Z refit + auto-z negative z_bias (HW calib, unrelated to the refactor)
- **Symptom**: in Follow/grasp the arm crashed into the table (z_gap_io read +5cm). **A/B reproduced the same with the old debug_deepreach = not a refactor regression** (robot_io.py shared + z path byte-identical). "The Zero pose is accurate" = joint zero trusted; the error is reach-dependent FK.
- **Clean 4pt table-touch** (`fk_verify_z.py`, started with `--z-reach-slope 0` → recorded = raw): radial 0.381/0.464/0.533/0.565 → z_gap_io 0.0676/0.0727/0.0762/0.0809. **Linear fit slope +0.0676/m, ref 0.486, residual ±1.3mm.** (The old −0.209 had the opposite sign and 3x magnitude = stale.)
- **Applied** (`c7a0490`): cli `--z-reach-slope 0.0676 --z-reach-ref 0.486`. Remaining **constant ~9.3cm over-read** (beyond the grasp-zgap-io range) → robot_io **`FK_Z_CORR 0.03→−0.063`** (shared module = also applies to the old program). `--grasp-zgap-io 0`. **Real-robot check**: with the new calib, at radial 0.455 and TCP +3cm, z_gap_io +0.009 (≈ actual +0.011) = **~2mm accurate, +9cm crash resolved.**
- Warning: **auto-z negative z_bias fix** (`5d45ddb`): making FK_Z_CORR negative **flipped fk_bias negative (−0.061)** → auto-z `z_bias=max(0.0,GRASP_CLEAR+fk_bias−grasp_zgap_io)` clamps z_bias to 0 (assumes positive) → cube_top_z does not rise → because base_ctrl (still over-reads by ~4cm) **Follow hover is ~4cm low (+2.5cm)**. RUN/AUTO-GRASP are unaffected since they use the z_gap_io gate. **Fix = allow a lower bound of −Z_BIAS_MAX** (negative z_bias = higher target = the safe opposite of over-plunge; upper Z_BIAS_MAX kept). Verification: fk_bias −0.061 → z_bias −0.056 → Follow z_gap_io ≈ +0.065 (normal). **Warning: not HW-verified (need to confirm Follow hover ~7cm after restart).**
- Suspected root cause: 9cm constant + accurate Zero = **J2/J3 motor-zero drift**. FK_Z_CORR −0.063 is a band-aid; recommend recalibration (`calib_convention.py`) followed by a fk_verify_z refit.
- (key) **Calib-dependent values baked** (update all on recalibration): `FK_Z_CORR=−0.063` (robot_io.py), cli `--z-reach-slope 0.0676 --z-reach-ref 0.486 --grasp-zgap-io 0`. + user tuning baked: `--follow-center-margin 0.08` (a55f189), `--follow-center-un −0.03` (3890f79, one click of dashboard lat-).

## 5. New structure (`src/rebot_sim2real/deploy/`)
cli.py · blackboard.py · kinematics.py · main.py · main_dashboard.py · io/{robot_io,io_thread} · vision/{perception_thread,openvocab,front_gripper_z} · control/{control_loop,context,tick,telemetry,constants,commands,modes,robot_io_facade}. Entry = `deploy/main.py` (launcher) + `deploy/main_dashboard.py` (the dashboard is embedded = run by main as a thread). The old `deploy/` keeps 8 shims + tools + the original debug_deepreach.py (kept as A/B reference). The core engine `RuleBaseController` (src/rl, shared with training) was not moved.

## 6. Gotchas
- (key) **`.venv-deploy/bin/python deploy/main.py`** (system python3 has no torch). Bare invocation gives the working settings.
- (key) **Code/cli changes take effect only after restart**. fk_touch_samples are lost on restart.
- (key) **Calib-dependent defaults** (FK_Z_CORR · z-reach · grasp-zgap-io · x-corr) are tied to the current robot calib — update on recalibration.
- (key) **FK-touch only at distances where the tip actually reaches the table** (at the reach limit ~0.6m+ straining = false contact; check visually and drop).
- (key) **RUN/AUTO-GRASP = z_gap_io gate (accurate)**; **Follow hover = fixed base_ctrl** (via the auto-z z_bias) — different z paths.
- **Backticks in git commit -m = bash command substitution contamination** → use `-F file`.
- Turning shims OFF / returning to the old program = just import the old path (the shim is an alias). Shims removed in P5.

## 7. Open / next (priority)
1. (key) **Restart → confirm Follow hover ~7cm** (HW verification of the auto-z negative z_bias fix; z_gap_io ≈ +0.06~0.07).
2. (key) **Full grasp-cycle HW verification** (Capture → Follow ⊙CENTERED → AUTO-GRASP: close at z_gap_io → 0, grasped, Lift). = **gate for completing P3-HW.** Make sure the cube is well lit for ov_score 0.5+.
3. **P4**: move tools (calib_* · diag_* · fit_* · fk_verify_z · sysid · measure_latency · debug_reach/grasp · build_obs etc.) → `src/tools/` + fix imports, archive the dead ones. (Works now via shims, not urgent.)
4. **P5**: original `debug_deepreach.py` + `debug_dashboard` (shim) + 8 shims → `_archive`. **After full grasp-cycle HW verification** (keep the old program as fallback until then).
5. **Push the 13 commits** + clean up untracked docs/20-32.
6. Root lever (still): check J2/J3 motor zero (root cause behind the reach-Z band-aid), front-cam XY calib.

## Run (next session)
```
# normal grasping (all fixes/calib baked):
.venv-deploy/bin/python deploy/main.py
# reach-Z re-verification/refit sweep (fixed hover without descent):
.venv-deploy/bin/python deploy/main.py --follow-z 0.16 --follow-low 0.16 --z-reach-slope 0
#   then python3 deploy/fk_verify_z.py (separate terminal; same /cmd·/state protocol)
# live diagnostics: curl -s localhost:8899/state
# A/B reference (old program, shim): .venv-deploy/bin/python deploy/debug_deepreach.py --rate 30 …
```
Details: docs/31.

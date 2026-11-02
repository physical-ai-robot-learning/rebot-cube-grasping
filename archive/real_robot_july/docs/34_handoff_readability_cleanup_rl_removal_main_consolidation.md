# Handoff — (key) readability cleanup (variable names · formula comments · dead CNN removal) + full removal of RL/sim code (project = FK/IK + FSM only) + branch / main consolidation · artifact purge (2026-07-23, session 23)

> Continuation of docs/33 (AUTO-GRASP full-cycle HW success: joint_map root cause · reach-Z · closed-loop descent · yaw gate/sign). After docs/33, in the same session, did **quality cleanup + RL removal + repo consolidation**. **All work pushed. `main` itself is now the clean FK/IK state (= the former deploy-refactor-main).**
>
> **One line**: ① readability audit of the deploy package (not adversarial, a careful review) → renamed abbreviated variable names · added formula `@var→meaning` comments · **fully removed dead CNN wiring** (verified the refactor changes no behavior). ② Per the user's instruction, **fully deleted the RL/sim training stack** (SAC · train · envs · scene · viz · perception_cnn/fusion · sim scripts · RL configs · sim helpers = 64 files) — the project is now **FK/IK + FSM (RuleBaseController + deploy control loop)** only. KEEP set verified by the deploy transitive closure. ③ Deleted 3 old experiment branches (local+remote), **fast-forwarded `main` to deploy-refactor-main** (main ⊆ deploy-refactor-main, 0 loss) → main = remote = clean, ④ purged ~6.6GB of on-disk RL artifacts.

---

## 0. Git / repo state ((key) changed a lot)
- **Branches (all pushed, local=remote)**: `main` (= `48d24da`, clean FK/IK) · `deploy-refactor-main` (**same SHA** as main, duplicate) · `perception-topwrist-coop` (kept). **Deleted**: residual-rl-v6-phase1-2a · v5-analytic-obs · v6-RLPD (local+remote).
- **main = deploy-refactor-main = `48d24da`**. Current checkout = main. `git status` clean, in sync with `origin`.
- **Repo directory 28G→22G** (artifact purge). Most of the remaining 22G = Python venv (`.venv-deploy` torch+cu128, required for deploy) + **`.git` 3.5G** (history bloated by large past commits — slimming needs filter-repo + force-push, not started) + `_archive` 696M.
- Running is unchanged: `.venv-deploy/bin/python deploy/main.py` (bare = working settings).

## 1. Readability cleanup (`554aa26`, verified no functional change)
After docs/33, a careful audit of the whole deploy package → confirmed the debt is concentrated in the "moved verbatim" `perception_thread.py` and geometric formulas. The new `control/` code is already exemplary (per-mode dataclasses · role comments · input→output docstrings).
- **Variable renames**: perception_thread AsyncPerception attrs (`_fi→_front_idx` · `_pz→_plane_z` · `_xc→_x_corr` · `_uvn→_wrist_uvn_fn` · `_ov_ovl→_ov_overlay_box`…) · run-loop temporaries (`_e→cube_world_est` · `_c→centroid_px` · `_sc→box_score` · `_bias→oblique_bias` · `_raw→raw_xy`) · kinematics (`Jp→J_pos` · `Ja→J_aim` · `Nproj→null_space`…) · OLS accumulators (`sx→sum_x` · `b→slope`) · front_gripper_z (`a/b/m→prev_gray/cur_gray/motion_mask`) · openvocab (`_m→_model`…).
- **Formula `@var→meaning` comments** (in Korean, in the style of the existing `_project_wrist_unvn`): `_gimbal_lookat_ik` (damped-pinv + null-space) · `_edge_vote_yaw` (×4 folding · resultant = confidence) · OLS normal equations (2 files) · `_quat_wxyz_to_R`.
- **Dead CNN wiring removed** (main hard-codes align_det/front_det=None, wrist callback cos4/sin4/depth=0, 0 consumers of wcos4/wsin4/wdepth → verified unreachable): `_seg_from_cnn_mask` · `_align_seg` CNN branch + align_det param · front_det branch · publishing of wcos4/wsin4/wdepth · annotate_wrist is_cnn/depth · unused torch · dead `wrist_det` param · stale dashboard keys (track_conf/searching/cnn_yaw_deg/yaw_src) · `--jaw-gain` · 2 CLI help entries (safety-reach-resid 0.15 · chunk-decay 0.9) · stale io_thread docstring. **Verification**: py_compile · deploy.main import · snapshot key contract intact.

## 2. Full removal of RL/sim code (`48d24da`) — project = FK/IK + FSM
User instruction ("we go with the current FK/IK, FSM-based way of operating, not RL"). KEEP/DELETE was statically verified via the **deploy transitive dependency closure (including relative imports)**.
- **Deleted (64 files, git rm)**: `rl/{sac,train,rewards,factory,nstep_buffer,resfit,demo}` · `envs/` (pick_env RebotPickEnv) · `scene/` (Isaac cameras) · `viz/` (training dashboard) · `perception_cnn.py` · `perception_fusion.py` · `deploy/{cube_detector_io,build_obs}.py` (old policy obs helpers) · `scripts/` (all Isaac/RL — train · eval_* · gen_perception_data_* · render_* · sweep_* · verify_perception_* · collect_demos · viz_rollout · monitor · train_cube_detector · wrist_env_check · wrist_render_perf · submit_train*.sh) · `configs/` (everything except train_v6a_yaw.yaml) · `tests/` (nstep_buffer · cube_tracker).
- **KEEP (closure verified)**: all of `deploy/` · `src/tools/` · `rl/{base_controller (FK + FD-Jacobian DLS IK engine),config}` · `constants.py` (config depends on `from ..constants`) · `grasp_traj.py` · `perception.py` (HSV + ray-plane) · `configs/train_v6a_yaw.yaml` (deploy loads it as EnvConfig) · `assets/robot_description_gripper/*.urdf` (Pinocchio FK) · `scripts/calib_intrinsics.py` (pure cv2 deploy intrinsics tool). `rl/__init__` cleaned up to a package marker (RL lazy exports removed).
- **Verification**: deploy.main import OK · train_v6a_yaw.yaml load OK · RuleBaseController construction OK (base_pos correct) · tracked LIVE code imports 0 deleted modules (gitignored `scripts/probes/` is local scratch, so irrelevant). `_archive/` is a graveyard, so dangling references are harmless = kept.
- Warning: **earlier notes about the RL training stack, cluster Ray/MLflow training, the staged phase-conditioned policy, the RLPD2 run and residual RL v6 status now refer to deleted code = historical.**

## 3. Branches + main consolidation + artifact purge
- **Branch deletion** (user instruction, local+remote): residual-rl-v6-phase1-2a (merged into deploy-refactor-main = 0 loss) · v5-analytic-obs · v6-RLPD (local only). KEEP = main · deploy-refactor-main · perception-topwrist-coop.
- **main consolidation**: confirmed main ⊆ deploy-refactor-main (0 unique commits, merge-base = old main HEAD) → `git merge --ff-only` fast-forward → main = `48d24da`. origin/main also updated (verified with ls-remote). **main is now the canonical clean state.**
- **Artifact purge** (rm after confirming 0 tracked, ~6.6GB): `data` (3.8G) · `runs` (1.6G) · `viz_out` (365M) · `demos` (237M) · `ckpt` (159M). `_archive` (696M) not selected = kept.

## 4. Current repo shape (FK/IK + FSM only)
```
src/rebot_sim2real/ = constants.py · grasp_traj.py · perception.py · deploy/ · rl/{base_controller,config,__init__}
configs/ = train_v6a_yaw.yaml    scripts/ = calib_intrinsics.py    src/tools/ = calib_*·diag_*·fit_*·fk_verify_z·measure_latency·sysid_arm·debug_reach/grasp·damiao_params·make_checkerboard_pdf·wrist_view
assets/robot_description_gripper/*.urdf    deploy/ = main.py (launcher) + no main_dashboard (embedded) → deploy/ holds calib data + main.py
_archive/ = graveyard (old debug_deepreach · shims · RL entry points)
```
Deploy engine: `deploy/main.py` (launcher) → `src/rebot_sim2real/deploy/main.py` (setup/build) → ControlLoop (control/) + AsyncPerception (vision/) + RobotIoThread (io/) + Blackboard. FK/IK = `rl/base_controller.RuleBaseController`.

## 5. Gotchas
- (key) The `rl/` directory now holds only the **FK/IK engine** (base_controller + config) — the name is kept for historical reasons (rename not started).
- (key) `configs/train_v6a_yaw.yaml` = deploy's EnvConfig source (base_pos · joint limits · cube/table · reach). The name says "train" but it was kept for deploy only (rename candidate).
- (key) `main == deploy-refactor-main` (same SHA). deploy-refactor-main is now a duplicate — can be deleted (not done).
- (key) The gitignored `scripts/probes/*` that reference deleted RL code are scratch that needs Isaac, so they do not run in this repo (irrelevant).
- Calib-dependent baked values (FK_Z_CORR +0.015 · z-reach-slope 0 · grasp-zgap-io −0.04 · jaw_sign −1) — update on recalibration.

## 6. Open / next (priority)
1. (key) **Distance-independent reach-slope Z**: grasp-zgap-io −0.04 is for the working radial (~0.45) only. **Grasp-based refit** (collect (radial, z_gap_io) at firm grasp at several distances → slope). Currently 1 point at FAR 0.565.
2. **AUTO-GRASP reproducibility sweep**: success rate per cube angle (0/15/30/45°) and distance (NEAR/MID/FAR).
3. (cleanup options) rename `rl/` → an appropriate name · rename train_v6a_yaw.yaml → a deploy config · delete the deploy-refactor-main branch · slim the `.git` history (filter-repo) · clean up `_archive/`.
4. Yaw seg bias (worked around by the gate; a more accurate source is optional) · front-cam XY calib (remaining root issue).

## Run (next session)
```
# Normal grasp (all fixes/calib/jaw_sign/depth baked, main branch):
.venv-deploy/bin/python deploy/main.py
#   Zero → place cube and Capture → Follow ⊙CENTERED → AUTO-GRASP (arbitrary angle)
# Live: curl -s localhost:8899/state ; wrist cam: curl localhost:8899/frame?n=wrist -o w.jpg
# reach-slope refit: after grasping at several distances, linear fit of (radial, z_gap_io@grasped) → --z-reach-slope/-ref
```
Details: docs/33.

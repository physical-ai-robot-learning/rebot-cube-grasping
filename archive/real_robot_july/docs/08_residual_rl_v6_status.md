# Residual-RL v6 — Work Status Handoff (2026-06-23)

> This document is a handoff for picking the work up where it left off. Spec: `docs/06_resfit_residual_rl_plan.md`.
> One-line summary: **The residual-RL implementation from docs/06 (SAC+n-step5+UTD4+asymmetric critic+RLPD+rule reach+wrist-CNN IBVS) is
> finished and passed static/unit/smoke verification. The only current blocker = full-workspace demo grasp success rate 49% (>50% needed before training).**

## 0. What to do right now (starting point for the next session)
1. Check `/tmp/collect_v6.log` + `/tmp/diag_demos.pt` — diagnostic demos (6 rounds, friction 5.0, `--out /tmp/diag_demos.pt`) were running in the background (the process may still be alive; check with `pgrep -af collect_demos`).
2. Raise the demo grasp success rate to **>50% (ideally >70%)**, then re-collect into `demos/pick_v6_demos.pt`. (That file is currently the 49% oriented set.)
3. Then launch **v6 training + tensorboard (6006) + monitor dashboard (8890)**.

## 1. Diagnosis of demos at 47–49% (key blocker) — **friction is not the lever; grasp orientation is the cause**
IK-scripted demo SR with full workspace (±0.35) + high hover (TCP ~25cm) + oriented 25° grasp tilt:
- friction 2.0: **49.5%** (190/384), failures = {fallen 16, other 178}.
- friction **5.0**: **47.4%** (91/192), failures = {fallen 11, **other_no_grasp 51 (27%)**, **other_grasp_failed 39 (20%)**}.
- → **friction 2.0→5.0 does not move SR** = grip force/slip is not the main culprit. `grasped_ever ~73%` but success ~47%.

Breakdown: `no_grasp 27%` (the jaw **never bites the cube even once** = air-close/miss) + `grasp_failed 20%` (grasped but dropped before the 10-step hold) + `fallen 6%`. Reference: the previous **centered + hand-tuned GRASP_Q was 93%**.

**Key hypothesis (verify first in the next session):** 6-DOF oriented IK only controls position (3) + **cam axis direction (2)** = 5-DOF, and **rotation about the cam axis (= gripper/jaw yaw, 1-DOF) is uncontrolled**. The cube is axis-aligned (yaw_rand=0) and the jaw must close on flat faces, but with J1 unlocked the IK null-space leaves the jaw yaw arbitrary → **the jaw closes on an edge/diagonal instead of the cube faces** → air-close (no_grasp) + unstable grasp (grasp_failed). The hand-tuned GRASP_Q (J1=0) had the jaw aligned to the faces and got 93%.

**Next fix candidates (priority order):**
1. **Add jaw yaw control**: extend oriented IK so the cam axis + **jaw-closing axis is parallel to the cube faces** (full orientation IK, or match jaw yaw to cube yaw with J6). Add a second-axis (jaw axis) alignment term to `_ik_step6`.
2. `grasp_tilt_deg` sweep (0/10/15/25) — 25° may be too large (check whether smaller/0 improves jaw seating).
3. Measure a grasp success heatmap with `scripts/eval_grid100.py`/`sweep_success_surface.py` (weak spots by position/angle).
4. Quick diagnosis: `--num-envs 32 --rounds 6 --out /tmp/diag.pt` to check SR + the `other` split (the current code prints it).

## 2. Implementation done (verified) — what is where
**RL core (torch only, verified without Isaac):**
- `rl/nstep_buffer.py`: **unified canonical 8-tuple** `(obs, priv, act, ret, next_obs, next_priv, disc, done_n)`. Timeout-aware n-step (`gamma**n_eff`, `obs_n` frozen at boundaries, `terminal`≠`done`). Shared `rlpd_sample()`. With `priv_dim=0 & terminal=done & n=1` it is exactly identical to legacy.
- `rl/sac.py`: `ReplayBuffer` also canonical 8-tuple, `Critic(critic_obs_dim,...)` (asymmetric), `SAC(...,critic_obs_dim)`, `update` uses `critic_obs=cat([obs,priv])`/actor=obs.
- `rl/resfit.py`: asymmetric critic + `rlpd_sample` (fixed demo/online arity mismatch bug).
- `rl/factory.py` (new): `make_agent(algo,...)` + `make_buffer(...)`. `rl/__init__.py`: lazy export.
- **Unit tests `tests/test_nstep_buffer.py` 5/5 PASS** (timeout-bootstrap survival, obs_n freeze, n=1 equivalence, legacy equivalence).

**env (`envs/pick_env.py`):**
- `RuleBaseController` hook; **integral residual** `arm_target=clip(q_base+res_accum)`, `res_accum += g*a*residual_arm_step` + anti-windup (joint clip). (The absolute form could only deviate ±step from the frozen hover, so grasping was impossible → changed to integral form.)
- g-ramp (`g_reach_overlay`→`residual_gain` over `g_ramp_steps`), **safety shield** (`fk_world` prediction → if tip_z<table_z-term or tcp_y<y_back_term, revert to the previous target), **sparse anneal** (`shaping_coef`, driven by train.py).
- Detection top-(u,v)→world reach (`_top_uv_to_world`, `_cube_est`, last-valid on dropout), **priv exposure** (`get_priv`, cube_pos+quat+rel=10D), `info["terminal"]/["next_obs_true"]/["next_priv_true"]/["is_grasped"]/["grasp_lift"]`.
- **obs_wrist_cnn**: render + batched CNN (`_wrist_cnn_tokens`) + HSV fallback, per-camera render gate (wrist is rendered even when top is analytic).
- Note: `base.reset(env_ids)` + `_res_accum`/`_reach_steps` reset live in env.reset (prevents latch leak).

**rule controller (`rl/base_controller.py`):**
- `ik_to` (position IK), `fk_world`, **oriented IK**: `_fk_full` (tcp + gripper rotation), `_cam_axis`, `_ik_step6` (6-DOF DLS: position + wrist-cam axis), `ik_oriented`. `reach_orient` branch in `compute_q_base` → **keeps the cam optical axis straight DOWN (⊥ table) at hover**. FK verification: hover cam tilt 0–6°, grasp exactly 25°.

**train/deploy:**
- `rl/train.py`: `make_agent`/`make_buffer`, priv + terminal + `next_obs_true` wiring, sparse anneal, **curriculum clean-eval trigger** (`curriculum_use_eval`).
- `rl/rewards.py`: `shaping_coef` (multiplies shaping only; safety/success kept).
- `scripts/train.py`: cnn-wrist included in `_need_cams`. `scripts/monitor.py`: `make_agent`. `deploy/run_pick.py`: wrist CNN (`CubeDetectorIO`) + **actor-only load** + `make_agent` (compatible with asymmetric/resfit ckpts).
- `scripts/collect_demos.py`: **per-cube IK demos** (oriented: hover cam-down, grasp 25° tilt toward cube), saves priv/terminal, n-step fold possible, per-round progress log + 2-way 'other' split diagnosis, full workspace ±0.35.
- `rl/demo.py`: priv + terminal save/load + n-step fold.

## 3. Key design decisions (all per user instruction)
- **Frame**: got ff58a03 via `git pull` — table_z=0 (ground -0.02), base z=-0.03, table_center (0,0.43), workspace re-centered. Base + table moved together by -0.04, so relative geometry is preserved.
- **obs_wrist_cnn=true**: top=analytic (fast), wrist=render+CNN (`cube_detector_wrist_gt.pt`, consistent with deploy).
- **J1 unlocked ±90° (±1.57)**: J1 aims azimuth; backward tips are guarded by the task-space `y_back` shield (the conservative J1 box was the cause of the dead zone).
- **High hover TCP ~25cm (`z_hover`=0.23)**: runway for the wrist cam to do IBVS centering during the long top-down descent.
- **oriented reach**: cam ⊥ table at hover, gripper tilted 25° toward the table at grasp (`grasp_tilt_deg`/`dir`).
- **workspace ±0.35, fully random spawn (curriculum off), episode_len 220**.
- Flags: `asymmetric_critic=true, safety_shield=true, reach_orient=true, bootstrap_truncation=true, curriculum_use_eval=true` (moot since spawn is fixed).

**Perception→reach→grasp pipeline**: TOP(u,v)→ray∩plane→cube(x,y)→RULE (J1 aim + DLS) reaches to the cam-down hover + latch → RESIDUAL does wrist(u,v) IBVS to descend → tilt 25° and grasp → lift. TOP=coarse reach, WRIST=fine alignment.

## 4. Commands
```bash
# Demo collection (residual IK, full workspace). tail -f /tmp/collect_v6.log for per-round SR in real time.
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/collect_demos.py \
    --config configs/train_v6.yaml --num-envs 32 --rounds 12 --spawn-rad 0.35 --out demos/pick_v6_demos.pt
# n-step unit test (no Isaac needed)
uv run --no-sync python tests/test_nstep_buffer.py
# smoke (4env/300step, full path)
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py --config configs/train_v6.yaml --smoke-test --exp v6_smoke
# main training
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py --config configs/train_v6.yaml --exp pick_v6_sac_nstep
# Ports (bring up together with training):
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync tensorboard --logdir runs --port 6006
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/monitor.py --config configs/train_v6.yaml --exp pick_v6_sac_nstep --http-port 8890
# SSH forwarding: ssh -L 8890:localhost:8890 -L 6006:localhost:6006 <workstation>@<cluster-node>  (or LAN <lan-ip>)
```

## 5. Verification status
- All changed files `py_compile` OK. v6 config loads OK (no unknown keys).
- n-step unit tests 5/5 PASS.
- smoke (4env/300step) end-to-end PASS (env init, residual, asymmetric priv obs_dim39/critic49, obs_wrist_cnn render+CNN, n-step, eval, ckpt; no crash).
- IK reachability probe: hover (TCP0.25) + grasp reach everywhere in ±0.35. oriented-IK FK probe: hover cam tilt 0–6°, grasp 25°.
- **Not run**: main training, UTD/n ablation, real robot. (Proceed after the demo SR is resolved.)

## 6. git / uncommitted
- branch `main` @ `ff58a03`. **Uncommitted**: 14 modified files (below) + new `rl/factory.py`, `tests/`, `docs/08_...md`. (Not committed until the user instructs a commit.)
- Modified: configs/train_v6.yaml, deploy/run_pick.py, scripts/{collect_demos,monitor,train}.py, src/rebot_sim2real/envs/pick_env.py, src/rebot_sim2real/rl/{__init__,base_controller,config,demo,nstep_buffer,resfit,rewards,sac,train}.py.

## 7. Risks/caveats
- Keep smoke tests to a minimum.
- Demo obs_dim=39 (actor)/critic_obs_dim=49 (=39+priv10). With `asymmetric_critic=true`, demos must include priv (re-collect).
- oriented reach runs 6-DOF DLS every step during the reach phase — if fps is slow, add an optimization that skips IK for latched envs.
- Training uses analytic top + CNN wrist; deploy uses CNN for both. Risk of analytic clean-overfit → cross-eval needed.

---
## 9. AUTO-MODE RUN (2026-06-24) — Phase 1 progress (autonomous run while the user was asleep)

User decision: **A (stack validation) → B (extremes fix) → retrain → Phase 2 (yaw-robust)**, auto mode. Key design settled = **validate A cleanly first (yaw_rand=0, analytic wrist, obs_dim 39), then stack B (wrist-CNN yaw) on top.**

### Phase 1 settled findings
- **grasp tilt = 6° is optimal** (thorough sweep: 6/9/12/15/30/45/60 × pullback × slanted-approach, all of them). 60° (=approach 120° from top-normal, jaw-strong = tilt-about-jaw) **physically collapses to 8%** — a parallel jaw at 30°-above-horizontal cannot wrap the side faces of a cube on the table. **Measured forward offset = 0mm** (probe_tcp_v6) → demo grasps are already centered; the forward grasping seen was the initial policy.
- **Reward rebalance (escape from a shaping local optimum)**: the first training run only hovered and avoided grasping (grasp dropped 14→8%, succ ~2%; clean-eval confirmed it is not DR). FIX (train_v6a.reward): success_bonus 80→300, fall_penalty 5→1, w_grasp 0.5→3, hover terms (w_reach/xy_align/vertical/wrist_center/hold) reduced 3–6x, **keep w_lift=25 + 1.5M anneal** (shaping_coef also multiplies r_lift/r_grasp, so accelerating the anneal is counterproductive).
- **1-A validation OK**: 150k demos + rebalance → online succ 23%, grasp 30–62%, eval succ up to 25% (clear learning vs the previous failed run at grasp ~10%/succ ~2%).
- **1-B extremes fix (SR by position)**: cause of the rear (y0.53) dead zone + corners (±0.30 at 31%) = **the cam-down hover at z_hover 0.23 (TCP 25cm) cannot reach the extremes**. FIX: **z_hover 0.23→0.15** (train_v6a + train_v6) → back 29→95%, corner 31→93%, overall 58→89% (extreme_fix_sweep). ow=0.6 kept. Measured on demo re-collection: SR 65.8→80.6%, no_grasp 238→34.

### Bugs/blockers found
- **wrist-CNN gate bug (fixed)**: with `obs_analytic_uv=True`, `_wrist_obs` took the analytic shortcut and masked the CNN → v6 only loaded the wrist CNN and never used it. FIX: skip the analytic shortcut when `obs_wrist_cnn` (pick_env).
- **CNN-wrist render scaling (prerequisite for Phase 2)**: rendering 64 wrist cams at once → RTX descriptor exhaustion (404x "Unable to allocate descriptor sets"). Phase 2 needs fewer num_envs.

### New/changed files (all uncommitted)
- base_controller: jaw-yaw IK (J6) + `_snap_jaw_target`/`oriented_pick_poses` (+approach_pullback) + `rule_arm_step`.
- perception_cnn: yaw head (cos4θ,sin4θ) + `angle_from_yaw2`. train_cube_detector: yaw loss. gen_perception_data_wrist: oriented viewpoints + image-yaw GT (dataset data/percep_wrist_yaw.npz, 5000 samples, generated).
- pick_env: wrist-CNN gate fix + yaw token (obs_wrist_yaw, obs_dim+2) + gate. config: rule_arm_step/reach_jaw_weight/grasp_approach_pullback/obs_wrist_yaw, grasp_tilt 6, z_hover 0.15, yaw_rand 0.785 (v6)/0 (v6a), reward rebalance (v6a).
- configs/train_v6a.yaml (new, A stack-validation variant). scripts/probes/{sweep_grasp,grasp_geom_sweep,demo_region_sr,probe_tcp_v6,extreme_fix_sweep,eval_clean}.py.

### Current state / next
- **Retraining in progress** (train_v6a, z_hover=0.15, 230k high-quality demos) — monitoring eval. If it climbs well, Phase 1 is complete.
- **Phase 2 (yaw-robust)**: train the detector yaw head (dataset ready) + CNN render scaling (fewer envs) + yaw token (code done) + yaw_rand → re-collect → train. train_v6.yaml = B config.
- watch: monitor :8890, tensorboard :6006, MLflow <cluster-node>:5000.

### Phase 2 progress (yaw-robust) — detector blocker + analytic-yaw workaround
- **Phase 1 result: policy eval 94% / grasp 94% / fall 0%** (train_v6a, z_hover=0.15, 230k demos, step ~9000). Backup: `ckpt/pick_v6a_phase1_94pct.pt`. Stack + reward + geometry all verified.
- **CNN wrist detector blocker**: retrained the detector to learn the yaw head → **px_err 66–93px (localization failed)**, yaw_err ≈ random. Cause isolation: not the yaw loss (94px with w_yaw=0 too), not z_hover (93px at 0.15 too), **the OLD "working" dataset also gives 66px with my code** → (u,v) detection in the close-up wrist view is inherently coarse (low visible_frac 0.27–0.33 = camera mount parallax; at static poses the cube is out of FOV), and the yaw head needs precise localization so it cannot be learned. **CNN-yaw is a perception task that needs human review.**
- **Workaround = Phase 2a (analytic yaw)**: consistent with A using the analytic wrist (u,v), validate yaw-robust **learning** with an **analytic image-yaw token** (geometrically project cube center + a face-direction point into the wrist → (cos4θ,sin4θ), exact). Detector decoupled. CNN is separate deploy perception. Demo jaw alignment uses the privileged world yaw (unchanged).
- **CNN-wrist render scaling** (64-cam descriptor) also unresolved — the analytic-yaw path needs no rendering, so this blocker is also bypassed.

### Phase 2a (analytic yaw) → fundamental wrist VISIBILITY blocker (autonomous-run STOP point)
- Implemented analytic image-yaw token (pick_env `_a_wrist` +cos4θ,sin4θ, obs_wrist_yaw gate, obs_dim 41) + test gate (probe_analytic_yaw).
- **Finding: the wrist camera cannot see the cube in the v6 oriented pose (cam-down).** vis=0 at both hover (15cm) and grasp (0cm). Raw projection: **un≈0, vn≈-1.6** (horizontally centered, vertically above/outside the frame). The cube is in front (depth>0), but because of **cam mount offset parallax + straight-down aim** it lands above the screen.
- Implication: **all wrist-based perception ((u,v), yaw, CNN, analytic) is ineffective.** That is why A reached 94% without the wrist, using top+state. The datagen visible_frac 0.27–0.33 was only incidental visibility at lerp/jitter intermediate poses.
- **Recommendation (needs user decision):**
  1. **TOP-cam yaw (priority 1)**: overhead, so the cube top-face yaw is always visible. Add a cube world-yaw (cos4θ,sin4θ) token to `_a_top` → the actor combines it with its own joint state (J1/J6) to learn jaw alignment. Bypasses wrist visibility. Cleanest.
  2. Modify reach_orient/wrist_aim_local so the wrist cam **aims** at the cube (cam axis toward the cube instead of straight down) — but this affects Phase 1 (verified) + sim2real alignment, needs re-verification.
- **Autonomous-run conclusion**: Phase 1 (= 94% working policy) is complete as the key deliverable. Phase 2 (yaw-robust) stopped at the wrist visibility design issue — recommend resuming with priority 1 above (top-cam yaw). The code change (analytic yaw in _a_wrist) defaults to obs_wrist_yaw=false, so it has no effect on Phase 1 (safe).

### Phase 2a RESULT — yaw-robust VALIDATED (look-at + analytic yaw)
- **FIX (user's idea): LOOK-AT reach** (`reach_look_at`): aim the wrist cam AT the cube (not straight-down) so it CENTERS in the wrist (J4/J5 pitch; J1+top fix xy). Probe: vis 0→1, un≈0 vn≈0 across center/front/±x (back y0.53 vis=1 vn=0.86, reach limit). Implemented: `_fk_full` returns cam_pos, `ik_oriented(look_at=)` recomputes the cam-aim each iter, `oriented_pick_poses` hover + `compute_q_base` use it.
- **Cam rigidity (user check)**: VERIFIED rigid — Δcam_z·Δtcp_z same sign (no opposite-motion bug). Found base_controller cam_pos ~150mm off the USD wrist cam (URDF gripper_link FK vs USD calib gap, mostly z); look-at works anyway (z-robust) + the YAW TOKEN uses the env's correct cam (`_a_wrist`), so it's accurate. (Refinement: recalibrate `_Tgj` to the USD for precise look-at; not blocking.)
- **Analytic yaw token**: pick_env `_a_wrist` projects cube-center + a face-offset point into the wrist → (cos4θ,sin4θ) image-frame face angle (obs_wrist_yaw, obs_dim 41). Exact (no CNN) → validates yaw-robust LEARNING; CNN is the deploy sensor.
- **TRAIN pick_v6a_yaw (yaw_rand=0.785, 123k demos@64.5%): GRASP consistently 88-94% on ROTATED cubes, eval success best 94% (bounces 31-94%).** => the residual learns to roll J6 and align the jaw to rotated faces FROM the yaw token. **Yaw-robust GRASP works.** The success bounce = lift-HOLD variance on rotated grasps (fall 6-62%, grip slip), NOT jaw alignment (grasp stable ~90%). Backup `ckpt/pick_v6a_yaw_phase2a_94pct.pt`.
- **Remaining refinements**: (1) rotated-cube lift-hold (grip force/friction → reduce the fall-rate bounce); (2) 150mm cam calib for precise look-at; (3) CNN-wrist detector for DEPLOY (analytic≈render but the wrist detector localization was 66-93px — needs the look-at viewpoints + better data); (4) commit (all uncommitted).

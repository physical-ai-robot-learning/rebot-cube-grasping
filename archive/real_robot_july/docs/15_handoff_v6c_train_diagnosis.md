# Handoff — v6c vision-token training: the friction/geometry diagnosis + clean hover-base (2026-07-08, session 5b)

> Follows docs/14 (v6c vision-token retrain prep). This session: v6c was **launched → training did not improve → root cause identified by ablation (cube-table friction) → several control-geometry defects fixed (misaligned grasp, ramming into the table) → finally simplified to "learning, not gates" (clean hover-base)**. The clean hover-base run is currently baking. **Handoff.**
>
> **One line**: The root cause of the v6c success plateau was not the CNN but **the realistic cube-table friction I introduced (2.75→0.5), which let an untrained policy push the cube out of the workspace**. Reverting to sticky (2.75) let analytic reach 0.722. The user's geometry observations (misaligned grasp, ramming the table) are **transient behaviors of an untrained policy**, so instead of gates they are solved **by learning** → **only reach is rule-based; align→descend→grasp→lift are all residual (clean hover-base)**.

---

## 0. What is running now
- **Ray job `rebot-pick_v6c_vision` (rtx5080a) RUNNING** — **clean hover-base**: `residual_base: rule` (rule only reaches up to the pre-grasp hover) + the residual **learns all of align (J4/J5/J6)→descend→grasp→lift**. No gates. + sticky cube-table (μ 2.75) + grip μ 5.0 (silicone proxy) + CNN sd4 tokens (obs_wrist_cnn) + gentle-lift + mass/velocity DR + arm-limit (⊆ real HW) + **analytic 0.722 warm-start** (`--init-ckpt ckpt/pick_v6c_analytic_phase1_0p72_best.pt`, absolute path).
- sim video: `scripts/monitor.py --config configs/train_v6c_vision.yaml --exp pick_v6c_vision --http-port 8890` (separate process, 8890). Curves: MLflow `http://<cluster-node>:5000` exp=pick_v6c_vision.
- **All in sim. Not verified on the real robot.**

## 1. (key) This session's core finding (why success did not rise)
1. **Root cause = cube-table friction** (the 3-material "realistic" change I added in SESSION 5c). v6b used the legacy single μ=5.0 (cube+gripper), so the **effective cube-table ≈ avg(5.0, table default 0.5) = 2.75 (sticky)**, and it reached 0.969 + transferred to the real robot. When I dropped it to the realistic wood-steel 0.5, **an untrained policy that bumps the cube slides it out of the workspace → fall=1.0 → zero learning signal**. **Isolated by analytic ablation** (same point in time: realistic 0.5 = succ0/fall1.0, sticky 2.75 = succ0.72). **CNN-in-loop, gentle-lift, velocity/mass DR, arm-limit are all innocent** (proven by reaching 0.722 with sticky).
2. **Realistic 0.5 failed to recover even with warm-start** → abandoned. Sticky 2.75 adopted (learnable + history of real-robot transfer). **"A good grasp policy grasps the cube cleanly regardless of friction and transfers"** — the evidence is that v6a/v6b trained with sticky and worked on the real robot.
3. **Grip μ=5.0 is a rigid-contact proxy for silicone pad compliance** (realistic silicone-wood ~1.0 slips in this sim; history: μ2.0 slipped / 5.0 held). Warning: the real gripper is **fragile (broken twice)** → deploy grip force FORCE_POS **0.10** (robot_io hard cap 0.18). I.e. the real robot grasps with low force + silicone compliance, and the sim stands in with high μ = sim2real consistency.

## 2. User geometry observations → gate attempts → simplified to learning (lesson)
The user saw two defects in the monitor → I reactively stacked gates → **counterproductive** → ended in simplification:
- **(1) Misaligned (45° corner) grasp**: grip-align gate (no close before alignment) + **valid-grasp hold-latch** (a misaligned close does not latch and can reopen — this is **kept**, a good fix). Corner closes were already invalid via on_cube (finger_gap ≈ edge).
- **(2) Gripper rams the table while aligning**: 2cm align-hover two-stage descent + base-J6 servo. → base-J6 had a **double-control conflict with residual-J6 → grasp 0.5→0.2**. Even with base-J6 removed, the align-hover **dwell forced a misaligned descent → fall 0.9**.
- (key) **Conclusion (re-evaluated after FREEZE)**: analytic Phase1 (0.722) was learned **without any gate/hover** → misalignment and table ramming are **transient behaviors of an untrained policy that disappear once learned**. The gates were over-intervention. → **Remove all gates; only reach is rule, and the residual learns all of align→descend→grasp→lift** (user's final instruction). Warning: **Lesson: do not hastily change the architecture based on one monitor frame / one eval** — v6c was restarted ~12 times this session, resetting the warm-start each time, so convergence was never observed.

## 3. Current architecture (clean hover-base)
```
Rule(compute_q_base): reach → pre-grasp hover(z_hover 0.15, cam straight-down). Only up to here.
Residual SAC(full authority after hover latch): ALIGN(J4/J5 view + J6 jaw) → DESCEND(-Z) → GRASP(close) → LIFT.
  - gripper: forced OPEN outside gate_near(6cm); inside, the policy closes. Only valid grasps hold-latch (misaligned close recoverable).
  - obs 41: arm qpos/qvel/grip/dist-to-rest + top(u,v)analytic + wrist(u,v,vis,cos4θ,sin4θ) CNN sd4.
  - reward: sparse+anneal, success_bonus 300, w_grasp 3, w_lift 25, gentle-lift(w_lift_jerk 0.2), verticality/wrist-center pre-grasp.
```
An architecture diagram (earlier deep_reach version) was published separately (Warning: it is based on deep_reach, so it differs from the current hover-base; here the residual does the descend).

## 4. Next steps (priority)
1. **Observe clean hover-base convergence** (slow, 16 envs CNN). MLflow/monitor. **Judge from the trend over many evals, not one or two.** By A/B history hover-base ~0.75 (lower/slower than deep_reach 0.969), but better than gated deep_reach (0.125).
   - **If it rises well** (success 0.5+) → deploy port: `run_pick` obs41 + **residual_base=rule** + sd4 detector + deploy stack (base-z/exposure/x_corr/async/grip-force 0.10). The residual does the descend too, so the deploy must match.
   - **If it does not rise** (hover-base cannot learn the residual descent): (a) lower z_hover from 0.15 to ~0.08 to shorten the residual descent, (b) revert to deep_reach-base (base descends, no gates) — but leave table ramming to learning, (c) revisit warm-start (0.722 is deep_reach so only partially transfers to hover-base; cold or from a hover-base ckpt).
2. **If the CNN tokens really are the cap** (vision cannot match analytic 0.72): improve CNN straight-down/2cm-view recall (datagen occlusion labels, backlog), or the view reward term `w_wrist_view`.
3. **Commit**: much of this session's code is uncommitted → committed together with this handoff (§5 below). Deploy port after convergence is confirmed.

## 5. Commit state
Warning: **Uncommitted (on disk only)** — whether to commit is pending user confirmation. Note: the sim-training changes below + **the user's deploy-calib refactor** (4 calib scripts deleted + new `calib_grip.py` + README/docs12/joint_map.npz edits) are **mixed in one working tree**. Consider separating them when committing.
This session's sim changes (SESSION 5c~5l): `pick_env.py` (3-material friction/grip-align gate/valid-grasp latch/gentle-lift wiring/velocity DR/CNN log) · `base_controller.py` (view-align gate/align-hover two-stage descent+dwell/base-J6 toggle/jaw_aligned) · `config.py` (friction/gate/DR/align-hover/gentle-lift fields) · `rewards.py` (gentle-lift p_lift_jerk) · `train.py` (wrist_cnn_ckpt REBOT_DATA_ROOT path fix) · `configs/train_v6c_vision.yaml` (clean hover-base + sticky friction + new knobs) · `configs/train_v6c_analytic.yaml`·`train_v6c_analytic_p2.yaml` (diagnosis/curriculum ablation) · `configs/train_v6b_deepreach.yaml` (view_align_gate false) · deploy grip-force (robot_io/debug_deepreach/calib_grip_interactive). ckpt/data/png gitignored. Previous commits `b79104d`·`c911d2a` (SESSION 5d/5e).
Backup ckpts: `ckpt/pick_v6c_analytic_phase1_0p72_best.pt` (sticky analytic 0.722, warm-start source), `ckpt/pick_v6c_vision_warmstart_0p35_best.pt` (first CNN-token policy 0.35).

## 6. Operational traps (avoid repeating)
- **Ray submission_id collision**: after `job stop`→`job delete`, even if `job status` shows "failed" for 30-60s+, deletion is still in progress → a **retry loop (delete→30s→submit, up to 6 times)** is needed. Too hasty gives "already exists".
- **monitor**: do not put `pkill -f "scripts/monitor.py"` (matches its own command line → kills itself) or a trailing `&` (conflicts with background launching) in the command. Clean up the port with `lsof -ti:8890 | xargs -r kill -9`.
- **--init-ckpt must be an absolute path** (train.py checks os.path.exists directly; if missing it silently falls back to cold). Because of the Ray ephemeral CWD.
- **wrist_cnn_ckpt is also resolved as absolute/REBOT_DATA_ROOT** (fixed in train.py) — otherwise the Ray worker falls back to HSV (no crash, v6c neutralized).
- Initialize buffers after setting self.num_envs.

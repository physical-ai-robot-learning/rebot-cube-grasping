# Handoff — grasp-proxy A+B fix (RL) + Top+Wrist seg/tracking perception (2026-07-09, session 6)

> Follows docs/15 (v6c friction/geometry diagnosis + clean hover-base). This session: **(1) RL — the user caught a fake grasp in the monitor → the grasp check was a proxy and gave false positives; fixing it with A+B lifted training from a plateau (succ 0) into the target band; (2) on a new perception branch `perception-topwrist-coop`, implemented and verified a Top+Wrist seg+tracking cooperation architecture (A fusion+memory, B seg detector+per-cam tracker+graded-conf).** Handoff.
>
> **One line**: The root of the fake-grasp plateau was that `is_grasped` is a **proxy** of near + finger-gap width + verticality, so it false-fires when the jaws close on air behind the cube + `r_grasp` rewarded that → the cold policy parked on fake grasps. Fixing it with **A (enclosure gate) + B (couple r_grasp to grasp_lift)** took eval succ from 0 → peak 1.0. Then, to reduce real-robot knock-downs, a **Top+Wrist cooperative perception** branch was created and A/B implemented (only real-robot wiring remains).

---

## 0. What is in place now (all committed+pushed; training runs stopped)
- **RL (branch `residual-rl-v6-phase1-2a`)**: A+B grasp-fix + hover07 config tuning → `pick_v6c_hover07d` eval (30-60g deploy band) **peak 1.0 / best-saved**. **Stopped** (GPU used for perception). best.pt = `ckpt/deploy_candidate_v6c_hover07d.pt` = **deploy policy candidate**. Commits `52073f4` (A+B), `16a2659` (deploy align-latch — from the previous session).
- **Perception (branch `perception-topwrist-coop`, off residual-rl-v6-phase1-2a)**: A/B implemented. Commits `4ed49b4` (A), `3f514d6` (B seg detector), `8bef23a` (B4+B5-core).
- **GPU currently idle** (hover07d stopped). Isaac local (rtx5080a), Ray cluster.

## 1. (key) RL core finding — fake-grasp proxy (user diagnosis)
From monitor images the user caught that **the gripper had not enclosed the cube but closed on air behind it** → asked why that counts as a grasp. Code check:
- `is_grasped = near & closing & on_cube & vertical_ok` — a **proxy** (TCP within 3.5cm of the cube + finger gap within the cube-width band 30-62mm + vertical). **It does not verify that the cube is enclosed between the two jaws** → false positive when the cube props one finger at ~46mm from behind the jaw.
- `r_grasp = w_grasp * is_grasped` → rewards fake grasps → the cold policy parks by "nuzzling the cube" (eval grasp 0.6 / success 0).
- **The real metric success = grasp_lift & (cube_z ≥ table+7cm)** cannot be fooled → success 0 was honest.

**FIX = A+B (verified, `52073f4`)**:
- **A enclosure** (`pick_env.py` is_grasped): a grasp requires the cube to lie **on the segment between the two fingertips (lpw/rpw)** — perpendicular distance from the finger-to-finger axis `perp < grasp_enclose_perp(0.025)` + within the finger span. Closes behind/beside are rejected. `is_grasped &= enclosed`.
- **B reward** (`rewards.py`): couple `r_grasp` to **`grasp_lift` (real enclosed+lift)** instead of the proxy → no reward for closing on air.
- Result: grasp ≈ success now track together, eval succ 0→0.28→peak 1.0.

## 2. RL config tuning (`configs/train_v6c_hover07.yaml`, in the order the user instructed)
`pick_v6c_hover07` → `07b` (A+B) → `07c` (eval 30-60g) → `07d` (32env + friction 2.0). Final:
- **z_hover 0.07** (shortened from 0.15, shorter residual descent), **residual_base=rule** (only reach is rule; residual does align→descend→grasp→lift), **COLD** start.
- **eval_holdout 30-60g** (the old 50-100g is heavier than the real object and misleading → bracket around the real ~40g; console `[eval]` is in-dist 20-50g, holdout is a separate MLflow metric).
- **num_envs 32** (fps 60→~120 = 2×; **monitor off** = RTX descriptor headroom; 64 crashes from descriptor exhaustion), **cube-table friction 2.75→2.0** (toward the real object).
- Train succ is structurally lower than eval due to exploration noise + DR (e.g. train 0.6 / eval 0.85 is normal). **Warm-start-from-latest does not work** (the empty SAC buffer/warmup shakes the loaded weights).

## 3. Perception architecture (`perception-topwrist-coop`) — A/B implemented
Goal: reduce real-robot approach knock-downs. The perception module outputs a **fused cube WORLD pose + per-cam vis + memory** and feeds the RULE (`_cube_est`)/deploy — **the policy only sees raw tokens, so no retraining is needed**. coord-source pluggable (ray-plane now / stereo=C).
- **A `CubeTracker`** (`src/rebot_sim2real/perception_fusion.py`, `4ed49b4`): 2-cam confidence-weighted fusion + pairwise disagreement cross-check + occlusion MEMORY (first-snap/EMA/hold+decay+staleness). Wired into sim `pick_env._cube_est` (replacing the old top-only + 1-frame last-valid). **Ready to accept graded conf** (currently binary vis).
- **B seg detector** (`3f514d6`): opt-in **seg mask head** (P2→/4) on `CubeDetector` + `seg_summary` (centroid + peak-prob graded conf + coverage). The datagen saves seg masks (a seg annotator newly added for top). Training: **wrist px7.88/yaw2.3°/depth12.1mm/seg-IoU0.935, top px16.7/seg-IoU0.936** → `ckpt/cube_detector_{wrist,top}_seg.pt`.
- **B4 `CamTracker`** (`8bef23a`): per-cam SORT-lite (EMA + outlier gate + occlusion bridge). Multi-object ByteTrack = extension slot.
- **B5-core `CubeDetectorIO.cube_seg`** (`8bef23a`): seg head → (un,vn,**graded conf**=vis×seg-peak, depth, cos4/sin4, mask). Offline verification (real wrist data): centroid ~11px, conf **0.96 visible / 0.03 occluded** (graded works).
- Unit tests `tests/test_cube_tracker.py` 8/8 (CubeTracker 5 + CamTracker 3).

## 4. Next steps (after this session = "B↔A integration + C")
1. **B↔A integration**: bind `cube_seg` (B) → `CamTracker` (B4) → `CubeTracker` (A) into one perception path (reusable module + offline tests). Make graded conf actually drive A's fusion/cross-check.
2. **B5 deploy wiring (ON-ROBOT)**: connect the above path to the `deploy/debug_deepreach.py` AsyncPerception → publish a fused, tracked, occlusion-surviving cube pose (replacing one-shot capture). **Live verification in front of the robot** (avoid writing large amounts blind).
3. **C stereo source**: top-ray × wrist-ray triangulation as an extra CubeTracker source (off-plane/lifted cube/variable height). Drop-in interface. On-table, ray-plane is more accurate; this is for extension.
4. (Optional) Put the seg detector into the sim RL loop (obs_wrist_cnn→seg) and retrain the policy — closes the policy's sim2real detector gap (not needed for rule/deploy).

## 5. Commit state — all committed+pushed
- `residual-rl-v6-phase1-2a`: `52073f4` (A+B grasp fix + hover07 config), `16a2659` (deploy align-latch, previous).
- `perception-topwrist-coop`: `4ed49b4` (A CubeTracker), `3f514d6` (B seg detector), `8bef23a` (B4 CamTracker + B5 cube_seg), this handoff.
- gitignore: ckpt/*.pt, data/*.npz (seg data and detectors local only).

## 6. Operational traps (avoid repeating)
- **EMA warmup**: using a fixed decay 0.999 on short training (~1.2k step) leaves the EMA ~30% contaminated by the random init (0.999^1200≈0.30) → the first seg training failed with px66/IoU0.007. A `min(decay,(step+1)/(step+10))` ramp is required.
- **fake-grasp proxy**: a geometric grasp check must verify **enclosure (between the two fingers)**. Otherwise closes on air give false positives.
- **MLflow narrow-grep misjudgment**: do not judge whether a run is alive by the number of metric-history points → check with `ray job logs`/latest.pt mtime (this time a run that was alive and improving was misjudged as dead).
- **num_envs vs descriptor**: CNN-in-loop wrist rendering is one camera per env → 64 crashes. 32+monitor risks descriptor exhaustion → at 32, turn monitor off.
- **Deploy wiring is hardware-verified** step by step (deploy bring-up pattern).

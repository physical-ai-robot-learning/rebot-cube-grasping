# Handoff — Vision-token residual RL retrain (v6c) + Tier B wrist perception (2026-07-07, session 5)

> Follows docs/13 (deploy calib + smoothing). **This session: (1) confirmed real-robot full pick (reach→grasp→lift) works, (2) deploy J6 align servo + jog bug fix, (3) rebuilt Tier B wrist perception (Isaac seg-GT, projection root cause + fix, datagen rewrite, sd4 detector retrain = yaw 2.4°/vis 99.5%), (4) prepared the v6c vision-token retrain (config + blockers + strategy).**
>
> **One line**: The wrist cam now has a precise "eye" (sd4, yaw 2.4°); next the policy is retrained with **vision tokens (CNN-in-loop)** so the residual SAC learns **J4/J5 view alignment → J6 jaw alignment → approach → grasp**. **Launch the retrain next.**

---

## 0. Current state (what works and what is ready)

- Done: **Real-robot full pick works**: reach→grasp→lift on the real robot lifts the cube (docs/13 stack + this session).
- Done: **Deploy J6 align servo built**: `debug_deepreach.py` `Align J6`/`Align flip` buttons + red-mask seg overlay (magenta box + yaw). **HW sign/convergence verification not done** (task #7). Jog bug (direction always +1) fixed = `encodeURIComponent`.
- Done: **Tier B wrist perception complete**: `ckpt/cube_detector_wrist_sd4.pt` — **yaw_err 2.4°, vis_acc 99.5%, px 11.9, depth 15mm** (sim val). Fixes the old 6% recall + minAreaRect contamination problem.
- Done: **`project_cube_to_wrist` roll-fix** + analytic-mode bake (`_bake_wrist_cam_local`) — analytic training tokens are also render-accurate (verified).
- Next: **v6c vision-token retrain is ready** (`configs/train_v6c_vision.yaml`) — **only the launch remains** (§3 below).
- Paused: the preliminary v6c (analytic token) training was **stopped** (user instruction — restart fresh with vision tokens).

## 1. Target architecture (confirmed by user)

**Rule base (deep_reach) + Residual SAC**, with the residual learning **everything**:
```
J4/J5 view alignment (make the cube appear as a clean square → only a small yaw turn needed) → J6 jaw alignment → approach (descend) → grasp+lift
```
- **Approach (reach + descend)** = deep_reach rule base.
- **View alignment (J4/J5) + jaw alignment (J6) + grasp + lift + fine correction** = residual SAC.
- **Use vision tokens**: the policy trains on the **actual sd4 CNN output** (rendered wrist) rather than analytic tokens → faithful sim2real.
- Geometric rationale (verified with CPU-FK this session): **J6 roll = 0mm TCP motion + jaw yaw ~1:1** (decoupled = stable), **J4/J5 pitch = 28-43mm TCP motion + yaw ~0** (for view aiming, cannot align the jaw). So the split **jaw=J6, view=J4/J5** is correct for both accuracy and stability.

## 2. (key) How J4/J5 view alignment is learned (core mechanism)

**Switching to `obs_wrist_cnn=true` (vision tokens) makes view alignment EMERGENT**: the CNN tokens (vis/yaw) are only high-confidence when the cube is seen as a clean square → J6 alignment and grasp succeed → reward. Analytic tokens "always see the cube", so that incentive did not exist (view alignment was unnecessary). → **Vision tokens = the learning signal for view alignment.**

**(Optional, recommended) Accelerate/guarantee with an explicit view reward term** — add and verify in the retrain session (unverified, so this time only config, no reward code added):
- In the `pick_env` reward, a pre-grasp-stage-gated term like `r_view = w_wrist_view * wrist_vis * exp(-k*|wrist_un,wrist_vn|)` (+ the more the cube is visible in the wrist cam and the closer to center). `wrist_vis`, `wrist_un/vn` already exist from `_a_wrist`/CNN tokens.
- Add `w_wrist_view` to the config (default 0=off) → turn on small (e.g. 0.5~2) and tune. **Run first with it off (emergent), and turn it on if view alignment is weak.**

## 3. How to run the retrain

**config: `configs/train_v6c_vision.yaml`** (copy of train_v6b_deepreach + vision-token changes). Already set:
- `obs_wrist_cnn: true` (wrist render + sd4 CNN), `wrist_cnn_ckpt: ckpt/cube_detector_wrist_sd4.pt` (has yaw head), `obs_analytic_uv: true` (top stays analytic), `obs_wrist: true`, `obs_wrist_yaw: true` → obs_dim **41** unchanged.
- `num_envs: 16` Warning: **key point**: at 64 envs the CNN-wrist render causes **RTX descriptor-set exhaustion** ("Unable to allocate descriptor sets" x404, 5080). Start at 16; raise if it works, drop to 8 if it does not.
- `residual_base: deep_reach`, `yaw_rand: 0.785`, `base_pos z +0.02`, `demo_frac 0`.

**launch** (Ray, rtx5080a = this box; after confirming the GPU is free):
```bash
cd ~/workspace/rebot_sim2real
EXP=pick_v6c_vision STEPS=10000000 ./scripts/submit_train.sh --config configs/train_v6c_vision.yaml
# status: ray job status rebot-pick_v6c_vision  |  logs: ray job logs rebot-pick_v6c_vision -f
# check log: "[env] obs_wrist_cnn: loaded ...sd4.pt"  (if missing = HSV fallback = wrong ckpt path)
```
Warning: **Before launch, check the GPU is free**: `nvidia-smi`. (This session's preliminary job was stopped.)
Warning: **CNN-in-loop is slow** (16 envs + render + CNN). Much slower than analytic (64 envs, no render) — be patient. If it dies from descriptor exhaustion, lower num_envs.

## 4. Training visualization (sim rollout video)

Metric curves = **MLflow** `http://<cluster-node>:5000` exp=`pick_v6c_vision` (success/grasp/fall/loss). Ray dashboard `http://<cluster-node>:8265`.

**sim rollout video** = separate monitor process (trainer is headless):
```bash
cd ~/workspace/rebot_sim2real
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/monitor.py \
    --config configs/train_v6c_vision.yaml --exp pick_v6c_vision --monitor-every 30 --http-port 8890
# browser http://localhost:8890 (remote: ssh -L 8890). Polls ckpt/<exp>_latest.pt and renders a 3-view rollout.
```
- Warning: monitor is a **second, cameras-on Isaac process** (extra VRAM on the same GPU) — watch for VRAM contention with the retrain (16 envs).
- Warning: with a Ray job, the trainer's `<exp>_live.json` goes to an ephemeral dir, so **the metrics panel is empty** (camera views work since they read `$REPO/ckpt`). View metrics in MLflow. (With a local trainer both work.)
- Alternative: `scripts/render_policy.py --ckpt ckpt/pick_v6c_vision_latest.pt --episodes 3` → `runs/policy_rollout.gif` (one-shot).

## 5. Residual demo data? (user question)

**No demos are currently used** (`demo_frac 0`, `demo_path ""`). The deep_reach rule base bootstraps by reaching the cube ~90% of the time + the residual learns grasp+lift via the success bonus = **demo-free RLPD**. `data/demos/` has v6a demos, but **they are hover-base, so their action distribution differs from deep_reach-base and they don't fit** (config comment). **If training is slow, regenerate deep_reach-base demos** (`collect_demos`-style) and set `demo_path`/`demo_frac`. Training may be slow because of vision tokens, so demo bootstrapping could help — a follow-up lever.

## 6. Tier B wrist perception details (for reproduction/debugging)

- **Isaac seg-GT**: `add_labels(cube_prim, ["cube"])` + `wcam.add_semantic_segmentation_to_frame()` → `get_current_frame()["semantic_segmentation"]["data"]` (uint32 mask). The datagen (`gen_perception_data_wrist.py`) uses this for visibility (px count, occlusion-aware) + centroid labels → visible_frac **0.99** (previously 6%).
- **Projection root cause** (important): Isaac `Camera.get_image_coords_from_world_points`/`get_world_pose` are **STALE** (not synced with the articulation). `tcp_link.get_world_poses()` is live. `project_cube_to_wrist` recomputed the cam direction with a world-up look-at → **roll drift when the link rotates → biased yaw tokens** (v6b also affected). **Fix**: `_setup_wrist`/`_bake_wrist_cam_local` bake the cam prim's local xform (4x4), and `project_cube_to_wrist(cam_local_M=)` composes **live link pose ∘ static cam-local**. Verified ~1px in u + v-top against the seg mask. **However the absolute vertical position has a ~9px residual** (presumably the tcp_link RigidPrim frame ≠ the cam's parent prim) → **yaw is robust since it is a 2-point difference, but absolute corner GT is inaccurate** (hence the corner head was deferred and the yaw head adopted).
- **datagen J6 roll injection**: jaw_align=True poses always show the cube aligned, so yaw labels are ~0 (nothing learned) → inject random J6 offset ±0.785 → yaw labels span [-45,45].
- **Retrain**: `train_cube_detector.py --data data/percep_wrist_seg.npz --out ckpt/cube_detector_wrist_sd4.pt` (heatmap+vis+yaw+depth automatic). 40ep → **yaw 2.4°/vis 99.5%/px 11.9/depth 15mm**.
- The corner-keypoint head (3D "face geometry") is **deferred** (projection vertical residual unresolved). Resume after a depth annotator or the frame is pinned down.

## 7. Uncommitted → commit

Changes this session (regenerable ckpt/data are gitignored):
- `src/rebot_sim2real/envs/pick_env.py` — project_cube_to_wrist roll-fix + `_bake_wrist_cam_local` + analytic-mode wiring.
- `scripts/gen_perception_data_wrist.py` — seg-GT labels + J6 roll injection + roll-correct yaw/depth.
- `deploy/debug_deepreach.py` + `deploy/debug_dashboard.py` — J6 align servo + red-mask seg overlay + J5/J6 jog + jog encodeURIComponent bug fix.
- `configs/train_v6c_vision.yaml` (new) — vision-token retrain config.
- (gitignored, not committed) `ckpt/cube_detector_wrist_sd4.pt`, `data/percep_wrist_seg.npz`.

## 8. Remaining work (priority)
1. **Launch v6c vision-token retrain** (§3). Observe via MLflow/monitor. A/B against v6b once converged.
2. **(Recommended) Add + verify the view reward term** (§2) — if emergent view alignment is weak.
3. **Deploy port**: run_pick → obs41 + deep_reach base + sd4 detector (wrist_token) + deploy stack (base-z/exposure/x_corr/async). Real-robot align→grasp→lift.
4. **Deploy J6 align servo HW verification** (task #7) — sign/convergence with red-mask (bring-up before attaching sd4).
5. corner head / depth-annotator 3D (deferred); deep_reach-base demos (if training is slow).

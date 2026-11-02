# Handoff — hover07e (both-cams CNN fusion) DONE + SAC-policy deploy tool + real-robot bring-up BLOCKER (2026-07-13)

> Branch `perception-topwrist-coop`. This session: **(1)** hover07e training finished (rule reach + residual SAC driven by tracked perception from two seg-CNNs) → converged, **holdout ~0.89**, deploy_candidate saved and committed. **(2)** New real-robot deploy tool **`deploy/deepreach_sac_world.py`** (Rule+SAC+seg mask+dashboard buttons) + dashboard parameterization / seg-mask overlay. **(3)** (key) **Real-robot bring-up: GATE A (perception) passed, GATE B (RUN) failed — the arm soars toward the sky.** Diagnosis and safety guards done, root fix pending.
>
> **One line**: rule reach is correct (proven offline), and **the residual SAC policy pushes upward on OOD / non-transferred obs**, so res_accum accumulates up to the joint limits → soar. Next step: isolate with `--rule-only`, then log obs/actions with `--res-clamp` + `--log-policy` to narrow down the root cause (which obs component is OOD).

Warning: **concurrent edits**: during this session **a parallel work session was actively editing and committing on the same branch** (commits `8dfaad3`/`210b225`, docs/18·19, files `debug_deepreach.py`/`robot_io.py`/`calib_grip.py`/`joint_map.npz`). That is a separate deploy workstream ("wrist-centric incremental re-add + **HW motor-connect blocker**", docs/19). **My work and that work are mixed in the working tree** → commit selectively (see §6).

---

## 0. Current state
- **hover07e (RL)**: done — converged and committed (`27ae1c2`). Candidate = `ckpt/deploy_candidate_v6c_hover07e.pt`.
- **Perception B↔A+C**: done — committed (`4a59b85`, `c0278da`).
- **deepreach_sac_world.py (SAC deploy)**: written + offline verification done, **uncommitted (untracked)**. Bring-up GATE B failed (soar) → safety guards added.
- **debug_dashboard.py**: my parameterization (buttons/title/caption) mixed with the parallel session's JS split (ESSENTIALS/DETAILS), **uncommitted**.
- GPU idle (hover07e training stopped). Robot: this box (`<workstation>`), cameras `/dev/video_front→video0` · `/dev/video_wrist→video2`, DM USB `/dev/ttyACM0`.

## 1. hover07e — training results (commit `27ae1c2`)
- **Architecture** (confirmed by the user): `residual_base: rule` — the RULE does **only the reach to a 7cm hover above the cube**, and the **residual SAC learns all of Align (J4/J5 view + J6 jaw) → Descend → Grasp → Lift**. The reach target `_cube_est` is driven by **top+wrist seg-CNN detections (u,v + graded seg conf) → CamTracker → unproject/stereo → CubeTracker (fusion + occlusion memory)** (= trained with the same noisy perception as the real robot = sim2real fidelity). Config `configs/train_v6c_hover07e.yaml`, num_envs 16 (2 cams × 16 renders), COLD, obs_dim 41.
- **Results**: console in-dist eval **converged to 1.0** (plateau at step ~264k). But **the fair holdout (30-60g, 64 ep, CNN-in-loop) A/B gives the real deploy numbers**: `_3200000` **0.889** / latest 0.862 / best (19:00) 0.720 / _4000000 0.583 (dip). (key) **Lesson: console 1.0 is optimistic, holdout is the truth → always select candidates with a fair A/B.** Winner `_3200000` → `deploy_candidate_v6c_hover07e.pt`.
- Reproducing the A/B: `scripts/probes/eval_ab.py` hard-codes `enable_cameras=False` (analytic only) → hover07e **needs cameras ON**. This session selected with `scratchpad/eval_hover07e_select.py` (launch Isaac once + iterate over candidates, holdout, cameras ON).

## 2. Perception B↔A+C (committed)
- `perception_fusion.py`: `CubeTracker` (A) + `CamTracker` (B4) + **`TrackedCubeEstimator`** (B↔A facade: per-cam CamTracker→unproject→CubeTracker[+stereo]) + `triangulate_rays`/`triangulate_source` (C). Warning: **fixes for the 6 bugs from the adversarial review are included** (stereo `>=2` cross-check, top token not-visible → 0, **wrist CamTracker `max_stale=0`** [moving eye = no pixel bridging], deploy est.update ts-gate, stereo x_corr). Unit tests `tests/test_cube_tracker.py`.
- sim `pick_env`: with `cube_track_cnn_fusion` on → fuses the two CNN detections (new top-CNN path `_load_top_cnn`/`_cnn_detect`/`_top_cnn_tokens`). With the flag off, the existing analytic path is byte-identical.

## 3. Three deploy tools
- `debug_deepreach.py` — HW-verified baseline (one-shot capture, manual Align/Final). **Being edited by the parallel session.**
- `deploy/deep_reach_world.py` — drives the RULE deep_reach with the **tracked fused world pose** (manual buttons). Guide: **docs/17**.
- (key) `deploy/deepreach_sac_world.py` (**new, this session**) — **RULE base reach + trained SAC residual + two seg-CNN tokens/fusion + dashboard**. Control is a precise port of `pick_env._apply_action` (residual_base=rule) (q_base=compute_q_base → g-ramp → `res_accum += g·a·residual_arm_step` → arm_target=clip(q_base+res_accum) → safety shield → grip gate + hold-latch). obs = `build_obs` (keypoint 39 + wrist yaw 2 = 41): top_feat = front top-seg `cube_seg`, wrist_feat = wrist-seg `cube_seg` (u,v,vis,cos4,sin4). **The front cam is the real-robot counterpart of the sim top cam** (run_pick convention). Offline verification: obs_dim 41 / actor load / compute_q_base / build_obs[1,41] / agent.act[1,7] all passed.

## 4. Dashboard refactor (`debug_dashboard.py`, uncommitted)
- **Parameterization**: `DebugDashboard(port, buttons=, title=, front_caption=, wrist_caption=)` — placeholder substitution, **defaults = current (byte-identical for the RULE tool)**. The SAC tool shows only its own buttons (RUN/Stop/open/Reset/Zero/Abort + jog).
- **Segmentation mask overlay**: `deepreach_sac_world.annotate_seg` draws the CNN `cube_seg` mask on the frame as **translucent green** (front top-seg + wrist wrist-seg) + (u,v) + center crosshair. Status table trimmed (removed base tcp · torque · track_stale; merged grip / src conf).
- (The parallel session split the JS into two tiers, ESSENTIALS/DETAILS — it is in the working tree together with this.)

## 5. (key) Real-robot BRING-UP status + BLOCKER
Runbook: docs/17 + this session. **GATE A (perception) passed** — on the dashboard the seg mask sticks to the cube and cube_est/track_conf are normal. **GATE B (RUN) failed — the arm shoots upward and looks at the sky.**
- **Diagnosis (reproduced offline, no robot needed)**: `compute_q_base` (rule reach) is **correct** — REST (tcp z 0.128) → down-hover (0.089), wrist cam pointing down (cam_axis_z −1.0), 2-step latch. **The rule does not drive upward.** The obs layout exactly matches sim `_compute_state`, and proprio units follow the sim convention too (arm = HW-verified with the RULE tool, grip = `robot_io` L72/227 returns sim m). → **Cause = the residual SAC policy outputs actions that push upward on OOD / non-transferred obs** → `res_accum` accumulates up to the joint limits (J2/J3 hi=0) → soar. **This was the policy's first HW test, so the sim2real obs/dynamics gap surfaced.**
- **Safety guards added** (deepreach_sac_world): `--rule-only` (policy removed, rule reach only = safe isolation), `--res-clamp` (default 0.5 rad; clamps the residual to ±rad per joint to block the soar), `--log-policy` (logs action/residual/dist).
- **Next steps (exact order)**:
  1. **STEP 1 `--rule-only`**: RUN → the arm should **descend and stop (open)** at the 7cm hover above the cube. If yes, the rule is correct on HW and the residual is confirmed as the culprit. If not (still soars), it is an FK/base-z HW issue (differs from offline).
  2. **STEP 2 `--res-clamp 0.15 --log-policy`**: only a small residual (cannot soar) + capture the `[pol]` log (whether the `a=[...]` action is consistently one direction on J2/J3, direction of res accumulation, dist3/zgap_io).
  3. Use the logs to narrow down **which obs component is OOD** and fix the root cause. Candidates (unverified, in order): qvel scale (15Hz finite difference vs sim physics), tcp_obs frame, small token-normalization differences, or a reaction the policy only learned to work in sim. **Most decisive = compare the real-robot obs vector with the sim obs distribution** (add an obs dump to deploy → compare with obs from sim eval).
  - Warning: alternatives, if non-transfer of the residual to HW is the root cause → (a) retrain with stronger obs DR, (b) revert to a `deep_reach` base (RULE goes through descend) to reduce the residual's burden, (c) reinforce the grip with a run_pick-style commit-latch (force-close).

## 6. Uncommitted files (commit cleanup guide)
- **Mine**: `deploy/deepreach_sac_world.py` (untracked) + `deploy/debug_dashboard.py` (parameterization; but mixed with the parallel session's JS split) + this document. → **The two must be committed together to be self-consistent** (deepreach_sac_world uses `DebugDashboard(buttons=)`). Recommended to commit together once the parallel session's dashboard edits settle.
- **The parallel session's** (do not touch): `debug_deepreach.py`, `robot_io.py`, `calib_grip.py`, `joint_map.npz`, `uv.lock`.
- **Caution**: no `git add .` (mixed changes). Surgical adds only.

## 7. Run commands + operations
```
# STEP 1 (safe isolation — rule reach only):
.venv-deploy/bin/python deploy/deepreach_sac_world.py --ckpt ckpt/deploy_candidate_v6c_hover07e.pt \
    --front-exposure 78 --wrist-exposure 78 --rule-only
# STEP 2 (clamped residual + logging):
.venv-deploy/bin/python deploy/deepreach_sac_world.py --ckpt ckpt/deploy_candidate_v6c_hover07e.pt \
    --front-exposure 78 --wrist-exposure 78 --res-clamp 0.15 --log-policy
# Dashboard http://localhost:8899 (remote: ssh -L 8899:localhost:8899)
```
- Camera defaults are the `/dev/video_front` (→video0) · `/dev/video_wrist` (→video2) symlinks. Visually check that the FRONT/WRIST panels are not swapped.
- Auto exposure = white-out → `--front/wrist-exposure 78` is required.
- Grip `--grip-force 0.10` (robot_io HARD-CLAMP 0.18; **broke twice at 0.3 — never use**).
- Calib `deploy/calib.npz` (real checkerboard fx906/fy860, dist≠0, undistort ON). `--x-corr 0.0235`, `--base-z 0.010` (PEDESTAL).
- Just launching it makes `initial_liftup` raise the arm to REST → stand by at the E-STOP.

## 8. Related docs
- Docs: docs/17 (deep_reach_world), docs/16 (perception coop), parallel session's docs/18·19 (deploy grasp-align / wrist-centric — HW motor-connect blocker).

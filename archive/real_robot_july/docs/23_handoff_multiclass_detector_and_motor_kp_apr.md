# Handoff — Multiclass vision detector (front+wrist) + (key) root fix for Damiao motor latency (KP_APR) (2026-07-14, session 14)

> Follows docs/22 (sessions 12–13, grasp+Lift verification + wrist-calib redesign). This session ran **3 workstreams in parallel**:
> **(A)** wrist-calib redesign completed (§2, continuation of docs/22) · **(B)** multiclass shape-based detector for front+wrist built, trained, verified + deploy wiring · **(C)** hover-follow mode · **(D)** (key) **identified and fixed the root of the Damiao motor command-propagation latency (look-at instability amp-3) = position-loop gain KP_APR=54 (too low)**.
>
> **One line**: "detects anything red" → **multiclass seg detector (front cube 4.9 px / wrist 2.6 px, red-clutter robust)** sim verification done + deploy wiring (opt-in). And **motor latency 696 ms→67 ms (10×) solved by raising KP_APR** (POS_VEL kept, MIT/shaper unnecessary). **All uncommitted; sim/offline/partial-HW verification. Commit after full HW verification.**

---

## 0. (key) Top priority = finish Damiao KP_APR tuning (almost done; only per-joint + --store remain)

**Root cause (HW measurement)**: the dominant latency behind look-at/descent servo instability (amp-3 = latency feedback) = **motor POS_VEL step response 696 ms** (t@63%). Investigation results:
- vlim (32 vs 8) irrelevant, ACC/DEC/MAX_SPD irrelevant (2→80 no effect), shaper irrelevant → **all red herrings**.
- **Root cause = position-loop gain `KP_APR` (RID 27)=54 is too low.** Applied immediately via RAM write (unlike ACC, no reboot needed).
- Measurement: KP_APR 54→200→400→600 = step **696→205→95→67 ms**, track (2.5 Hz) amplitude **10→61→77%**. At 600, overshoot +0.07° (ringing onset). **J1 sweet spot ~400-500.**

**Remaining work (tool `deploy/damiao_params.py` is ready)**:
1. **Check wrist (J3/4/5)**: currently all 6 joints have KP_APR=600 (RAM). The wrist has low inertia, so oscillation risk → check overshoot/ringing with `measure_latency step --joint 3/4/5 --deg 5 --go`. If large, lower only that joint: `damiao_params --joint 4 --set-kp-apr 250`.
2. Finalize per joint (expected: shoulder 0,1,2 high ~400-500 · wrist 3,4,5 low ~200-300).
3. Warning: **Permanent storage**: currently in RAM, so **powering off reverts KP_APR to 54!** Store finalized values with `damiao_params --joint N --set-kp-apr <v> --store` (per joint or all joints).
4. **Zero code changes** — it is a motor register, so after --store deploy automatically uses the fast motors. **Downstream (look-at/hover-follow/descent tracking) all become stable and precise automatically** (D down 10× → stable loop-gain limit L*(D) up 10×).
5. Side note: the SW shaper (amax=12) is now a secondary limit in the very-fast regime — if needed, raise `--amax` or use `--arm-vlim` (bypasses the shaper; measure_latency already has A/B knobs).

**Diagnostic tools (new this session, HW)**:
- `deploy/damiao_params.py`: register READ (no-motion, safe) + `--set-kp-apr/--set-acc/--set-dec/--set-max-spd/--set-kp-asr [--joint N] [--store]`. READ is the default.
- `deploy/measure_latency.py` (extended): `--mode step/track/comm` + **`--arm-vlim` (POS_VEL glide) · `--shaper` (SW shaper A/B) · `--amax/--vmax`**. Warning: step/track `--go` = real arm motion.

Detailed analysis: amp-3 = latency × gain Nyquist; shaper is frequency-dependent [<1.5 Hz transparent, >2.5 Hz +133 ms]; MIT is available but unnecessary.

---

## 1. Git / commit state (branch perception-topwrist-coop, all uncommitted)
```
 M deploy/cube_detector_io.py        ← multiclass recognition + seg_cube_uv_full (cube center = seg centroid)
 M deploy/debug_dashboard.py         ← 'Follow' button
 M deploy/debug_deepreach.py         ← follow mode + --front-detector wiring + --wrist-calib (docs/22) + calib_wrist redesign
 M deploy/measure_latency.py         ← --arm-vlim/--shaper/--amax A/B
 M scripts/train_cube_detector.py    ← C-aware multiclass (single-class backward-compat)
 M src/rebot_sim2real/perception_cnn.py  ← CubeDetector(n_classes) opt-in (C=1 = identical to before)
?? deploy/damiao_params.py           ← (key) read/write motor registers such as KP_APR
?? deploy/fit_wrist_calib.py         ← wrist-calib fit (docs/22)
?? scripts/gen_perception_data_front.py       ← front multiclass datagen
?? scripts/gen_perception_data_wrist_multi.py ← wrist multiclass datagen
?? scripts/submit_train_detector.sh  ← detector Ray submit (rtx5080a)
```
**Commit criteria**: KP_APR after HW verification (per joint) + --store / detector after HW (u,v) alignment / deploy wiring and follow after HW behavior is confirmed. (All new items are opt-in, so no effect on baseline.)

---

## 2. (B) Multiclass shape-based detector — sim verification done, deploy wiring done
**Goal**: improve front-cam XY accuracy + fix "detects anything red" (HSV color-based) → the CNN outputs **the cube center directly via seg, based on shape/class** (independent of red clutter). User decision: **multiclass** (cube + other objects) · small scale first.

**Design/implementation (all verified)**:
- **Model** `perception_cnn.CubeDetector(n_classes=1 opt-in)`: hm/vis/seg with C channels, yaw/depth per class. **With C=1 the shapes are identical → no effect on the existing 13 ckpts (verified)**.
- **datagen** front (`gen_perception_data_front.py`) + wrist (`gen_perception_data_wrist_multi.py`): 4 classes = [redcube (target), ball, rod, greybox], ball/rod include red (forces shape learning). front = aligned to calib pose + **projected-center labels** (camera `get_view_matrix_ros` + intrinsics, convention auto-detected) · wrist = eye-in-hand, so `project_cube_to_wrist` (yaw/depth) + seg centroid, distractors placed in a ring around the cube. Per-class semantic-seg→classmap.
- **trainer** `train_cube_detector.py` is C-aware (multi/single auto). ckpt stores n_classes + classes.
- **Ray submit** `submit_train_detector.sh` (rtx5080a, absolute local data path, MLflow).

**Results (sim verification, small 1500)**:
- **front `ckpt/cube_detector_front.pt`**: SEG IoU 0.87-0.88 across all classes, **cube seg-centroid 4.9 px**, red-robust. (The heatmap lands 41 px from the cube / on background = unsuitable → **using seg is the right answer**.)
- **wrist `ckpt/cube_detector_wrist_multi.pt`**: cube seg IoU 0.898, **2.6 px**, yaw 2.7°, **distractor-leak 1/200** (no red confusion). ball/rod IoU 0.44-0.54 (irrelevant since they are distractors).

**Deploy wiring (opt-in)**:
- `cube_detector_io.CubeDetectorIO`: recognizes n_classes + **`seg_cube_uv_full(cls=0)`** = target cube seg centroid → full-res (u,v). Verified offline at 4.9 px.
- `debug_deepreach.py --front-detector ckpt/cube_detector_front.pt`: AsyncPerception uses CNN seg (class0) instead of HSV → `cube_world_from_pixel` → world. top_bias=0 (CNN outputs the center). Unset = HSV baseline unchanged.

**Remaining work**:
1. **Scale up**: `data/percep_front_5k.npz` (5000, generated) → retrain (`DATA=data/percep_front_5k.npz OUT=ckpt/cube_detector_front_5k.pt EXP=det_front_5k ./scripts/submit_train_detector.sh` — but if the Ray queue has the rtx5080a token held by other jobs, run locally directly: if the GPU has headroom, `.venv/bin/python scripts/train_cube_detector.py ...`). Scale up wrist too.
2. **HW (u,v) alignment verification**: with `--front-detector`, is real-robot cube detection more accurate + red-robust than HSV? **Whether the sim 4.9 px transfers to the real robot is the D bottleneck** — v1 only **aligns the camera pose**; the real intrinsic (fy≠fx, cy), distortion, and roll **warp (v2)** are not applied → if transfer error remains, add the warp.
3. vis_frac~0.97 (mostly visible) → strengthen not-visible (out of frame/occluded) in v2.

---

## 3. (C) hover-follow — built + offline verified, waiting for HW
`debug_deepreach.py` opt-in **`follow` mode** (dashboard 'Follow' button / curl `cmd?a=follow`): the TCP **tracks cube_world with a straight-down cam** at `--follow-z` (0.16 m) above it, NO descend/look-at = **stable position servo** (low parallax gain at rest height and constant speed). `--follow-deadband` (0.01) · `--follow-iters` (40). Combine with `--track` for a live cube. **Descent baseline unchanged**. Offline IK verification: 4 positions, TCP xy_err 0 mm · cam 0°. **Confirm truly stable tracking on HW after KP_APR tuning (§0)**.

---

## 4. (A) wrist-calib redesign (continuation of docs/22 §0, done)
Showed that the old §0 (J4-only vertical sweep) is non-identifiable → redesigned (offset look-at + J1 sweep + ≥6 positions) → synthetic + end-to-end verification (axis 0.05°) → implemented (`fit_wrist_calib.py` + calib_wrist redesign + `--wrist-calib` deploy override). HW multi-position collection→fit→apply not yet run.

---

## 5. Infra/state
- **Ray**: head `<cluster-node>:8265` (job API, RAY_ADDRESS=http), MLflow `:5000`. Training node = **rtx5080a = this box** (Isaac + .venv cu128). Warning: if Ray has the `rtx5080a` **resource token** held by other jobs (eval5080p/det_TP*), detector jobs stay PENDING → **if the GPU has headroom (10 GB), bypass by running locally** (the wrist run went this way this time). `ray status` does not work over http (it is for GCS); jobs use http.
- **Robot**: J1 moved through small angles several times for KP_APR tuning tests, then torque-off. Currently KP_APR=600 (all joints, RAM, **not stored → reverts to 54 on power cycle**).
- Background: 5k front datagen done (`data/percep_front_5k.npz`). Retraining not yet run.

## 6. Next session actions (priority)
1. **§0 finish KP_APR**: check wrist (J3/4/5) step → finalize per-joint values → **--store** (permanent). Then confirm the improvement with measure_latency track.
2. **(B) scale up front/wrist detector (5k) and retrain + HW (u,v) alignment** → if good, intrinsic/distortion warp (v2) as needed.
3. **(C) hover-follow HW test** (stability expected to improve after KP_APR).
4. **Commit**: starting with HW-verified parts (KP_APR tool, detector, deploy wiring, follow).

## 7. Gotchas
- **KP_APR is RAM-only**: before --store, a power reboot reverts it. ACC/DEC are the opposite: RAM writes have no effect (irrelevant to the POS_VEL loop; confirmed no effect even when stored).
- **Detector cube center = SEG, not heatmap** (in multiclass the heatmap lands 41 px from the cube / on background; seg is 4.9 px). Deploy uses `seg_cube_uv_full`.
- **The sim 4.9 px is sim val** — real-robot transfer not verified (only camera pose aligned). This is the detector's real bottleneck (D).
- When the Ray token is held → PENDING → run locally if the GPU has headroom.
- All new items are opt-in → baseline (sd3 descent grasp) untouched.

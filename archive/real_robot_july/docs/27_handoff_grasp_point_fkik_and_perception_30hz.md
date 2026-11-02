# Handoff — (key) GRASP-POINT FK/IK root redesign (6 cm = JAW_FWD identified) + perception 3.4→18.9 Hz (sat_blob crop, align gating) + KP_APR all-54 rollback + _pick_cls crash fix + approach image-aim/danger-gate (2026-07-16, session 17)

> Continuation of docs/26 (session 16: OpenVocab + gimbal look-at approach). **A very long real-robot session in front of the HW.** Repeated code edits + real-robot polling (`curl :8899/state`). **All uncommitted.** The final arm-oscillation fix (xy_off io restore) is **waiting for HW re-verification**.
>
> **One line**: identified the root cause of the "total failure" (approach could not grasp the cube) — **the 6 cm x-error is not an FK bug but the JAW_FWD grasp-contact offset** (the actual jaw grips 6 cm ahead of the kinematic TCP in the gripper-forward direction). Redesigned the approach IK target as `cube + jaw_fwd·fwd_axis` (the actual grasp point). At the same time, identified the perception bottleneck via timing → **sat_blob crop (104→2 ms) + align_seg gating** raised perc_hz 3.4→18.9 (remaining bottleneck = camera YUYV USB2.0 10 fps, not SW).

---

## 0. Git / robot state
- Branch `perception-topwrist-coop`. **0 commits this session, all uncommitted.** The 5 unpushed commits (docs/26) remain as they are.
- **Main changes this session**: `deploy/debug_deepreach.py` (+279 lines), `deploy/front_openvocab.py` (+31), `deploy/cube_detector_io.py` (+66, including the crash fix). Others (calib_grip/measure_latency/train_cube_detector/perception*/joint_map/uv.lock/untracked) = leftovers from previous sessions.
- **Robot HW state**: motor CAN connection OK (confirmed io.connect passes). Camera symlinks `/dev/video_front`→video2, `/dev/video_wrist`→video0 (YUYV-only, USB2.0 480M). `.venv-deploy` transformers 5.13.1.
- Warning: **GPU sharing**: when `train_sac_det_xattn.py` from `rebot_eval/.venv` occupies 11.5 GB+ of GPU at 99% util, OWLv2/perception slow down a lot. Killed it twice this session (with user consent). Check `nvidia-smi` before deploy benchmarks.

## 1. (key) GRASP-POINT FK/IK redesign (headline, waiting for HW re-verification)
**Root cause of the "total failure"**: the base_controller FK and the robot_io FK are **kinematically identical** (same URDF / base_pos (0,0.06,-0.03) / joints). The 6 cm-x / 3 cm-z difference comes from **two intentional grasp-contact corrections** in robot_io.fk_tcp:
- **`JAW_FWD=0.06`** (robot_io.py:68/280): the actual jaw grips 6 cm ahead / camera-side of the finger-origin midpoint in the gripper-forward direction (J1 yaw) (measured on-arm).
- **`FK_Z_CORR=0.03`** (robot_io.py:62/282): tip table z, HW-verified (2026-07-10).
- I.e. **base FK = raw kinematic TCP (finger midpoint), robot_io FK = actual grasp point**.

**Bug**: approach aligned the kinematic TCP to the cube → the actual jaw (6 cm ahead) misses + the gripper-mounted wrist cam sees the cube at the frame edge (uv 0.85). → a vicious cycle where IBVS/image-aim is rejected by the clutter/uvmag gate.

**Design A (implemented)**: approach IK target set to **`cxy_cmd = cube + lookat_jaw_fwd·fwd_axis(q)`** (sign **+**, the deep_reach_step:341 approach, independent of io). Places the kinematic TCP jaw_fwd ahead of the cube so the actual jaw lands on the cube. `--lookat-jaw-fwd` (default None→`--jaw-fwd` fallback; on HW **0.06** given explicitly = stopgap, no regression). Reuses `base.fwd_axis_xy(arm_q)` (base_controller.py:210). For z, auto-z (surf_z) already includes FK_Z_CORR → **xy only, do not double-count z**. `look` = actual cube (view), `tgt` = cxy_cmd (grasp), `_danger` = cxy_cmd (consistent with base-FK).
- Warning: removed the previous stopgap `fk_bias_xy=tcp_io-tcp_w` (depends on io.fk_tcp measurement) → explicit model. Algebraically identical (`cxy-fk_bias_xy = cube+JAW_FWD·fwd`), sign + verified (ghat rest=[0,-1], rotates with J1, unit).

**HW result (partial success + oscillation)**: with `--lookat-jaw-fwd 0.06`, **wrist_uv un 0.85→0.1 (horizontal grasp alignment succeeded!)** + **tcp_io (actual grasp) ≈ cube (1.4 cm)**. The fundamental direction is right. **But the arm oscillates** (front est fixed, yet tcp varies 3-5 cm, xy_off varies 40-71).
- **Oscillation cause identified**: changing `xy_off` to the base-FK residual (`|tcp_w-cxy_cmd|`) gave 40-71 mm > gate (0.04), so it was **always in straight-mode → straight IK oscillation** (cannot use the gimbal null-space stability). The io grasp was on the cube, but the metric could not read it.
- **Fix (not verified)**: reverted to `xy_off = |tcp_io[:2]-cxy|` (restores io grasp alignment, the metric that was stable in the earlier fk_bias_xy version). cxy_cmd (explicit grasp) kept. → if io≈cube, the gate passes → gimbal (stable) expected. **This is the first HW verification next session.**

## 2. KP_APR all-54 rollback (user instruction)
- After reboot, **config A was still alive in flash** (pan500/lift500/elbow400/**wrist_flex150**/yaw200/roll400 — the values finalized in docs/24). The "settings were lost" impression was wrong.
- User observed violent J4 oscillation (unworkable) → **per user instruction, all 6 joints rolled back to KP_APR=54 in RAM** (`damiao_params.py --set-kp-apr 54`, not stored = config A returns on reboot). Warning: docs/24 warned that all-54 was the worst at REST hover with J2/J3 12°+ (measured before reassembly), but the user proceeded. **Not saved to flash, so reboot = config A**. Need to re-check the oscillation state at 54 on HW.

## 3. approach look-at extensions (image-aim + descend-to-grasp + danger-gate)
Added on top of the docs/26 approach (all opt-in, sd3 descent untouchable):
- **image-based aim** (`--lookat-image-aim` default on): `_gimbal_lookat_ik(meas_uv=)` — rotates relative to the optical axis using the **measured (u,v) bearing** instead of the world cube → centers the cube in the frame regardless of extrinsic/front errors. Warning: **online divergence**: iters80/gain2.5/astep0.07 = |uv| 0.93→23 blow-up. **Since it is online closed-loop, keep per-tick steps small** (`--lookat-image-iters 5/gain 0.2/astep 0.02`, offline sim TCP drift ≤0.5 mm). + **proximity gate** (`--lookat-image-gate`, only when |uv|<gate; relaxed to 1.0 on HW when the cube is at the frame edge).
- **descend-to-grasp** (`--lookat-grasp`): floor=surf+`--lookat-grasp-clear` (0.009). grasp-close taper (straight-down at `--lookat-taper` 0.03). `--lookat-autograsp` (default off = hold at grasp floor, human presses Grip). #1 rise-first (`approach_risen`, rise to hover first).
- (key) **danger alignment-gate** (fix for 347 mm false positive): aim (world or image) during motion (when the arm is far from the cube) pushed the TCP sideways and blew up the `_danger` resid → **`do_gimbal = ... and (xy_off < image_xygate)`** (aim only after the arm is aligned above the cube; position-only straight-down while moving). `_danger` target=cxy_cmd.

## 4. (key) perception 3.4→18.9 Hz (bottleneck identified via timing)
Instrumented **per-stage ms** in `/state.gates.tm` (rd_w/det_w/align/rd_f/front/gz/annot). Measurement reversed the guesses:
- **The bottleneck was not annotate/gripper-Z/read but deep-learning forward + HSV** (initially front 120 / det_w 87 / align 89 ms; annot/gz/rd = 1/2/0).
- (key) **sat_blob crop** (`_sat_blob_centroid`, largest effect): it ran **rgb_to_hsv on the full 720p frame** (104 ms) even with an roi → **crop to the roi first** (HSV on the small box only, 2 ms = 48×). The centroid had a binning artifact (28 px off), so use a **simple center of mass** (accurate due to the box gate, offset restored). front 120→**1 ms**.
- **align_seg gating** (89→0): only runs during Align. Publish `AsyncPerception.set_ctx(tcp, need_align=aligning)` → `if need_align:` in run(). Skipped for approach/track/idle (lossless).
- **fp16** (OWLv2 forward 2×, `OpenVocabModel._half`), **local_files_only** (bypasses HF hub HEAD 504, Auto classes cache-only), **Auto classes** (auto-load owlvit/owlv2), **torch.compile option** (`--front-ov-compile`).
- Warning: **owlvit-p32 failed to detect on the real robot**: fast in benchmarks at 18 ms (owlv2 48 ms) but **cannot find the real red cube** (ov_score None, no overlay, HSV fallback heads straight for red). Only detects synthetic blobs. → **reverted to owlv2** (real-robot ov_score 0.64, detected). The owlvit-p32 code is kept (as an option).
- Warning: **owlv2 input size has no effect**: 960→480 improved forward by only 10% (the bottleneck is elsewhere, not patches). `--front-ov-size` code kept, unused.
- **Remaining bottleneck = camera read (rd_f 47-96 ms)**: YUYV 720p ×2 cams = exceeds USB2.0 (480M) bandwidth → ~10 fps each. **MJPG not supported (YUYV-only cameras)**. For 30 Hz: (a) front 640x480 + scale front K, or (b) USB3 port. Software is maxed out (18.9 Hz).

## 5. _pick_cls crash fix (uncommitted)
At `cube_detector_io.py:63`, `@torch.no_grad()` was **above** `@staticmethod`, so no_grad wrapped the staticmethod into a regular function → `self._pick_cls` became a bound method (self injected as hm) → the `hm[:,cls]` subscript failed. **Swapped the decorator order** (staticmethod outermost). Verified single/multi-class.

## 6. Unresolved / next (priority)
1. (key) **HW re-verification of arm oscillation** (§1 fix): after restoring io `xy_off=|tcp_io-cxy|`, does approach converge in gimbal (stable)? Confirm tcp oscillation disappears, xy_off→0 + wuv→0, and the actual jaw grips the cube. **Restart command below.**
2. **HW measurement of grasp-point jaw_fwd** (plan §2): with `--lookat-jaw-fwd 0 --no-lookat-image-aim`, grasp→close→measure the signed distance d along the fwd axis between the jaw contact line and the cube → `--lookat-jaw-fwd d`. Start from the default 0.06. Verify a single scalar by reproducing off-center (J1 yaw).
3. **30 Hz** (optional): front 640x480 + scale the front calib K (the K in cube_world_from_pixel is for 720p) or USB3. Skip if the current 18.9 Hz is sufficient.
4. **Commit**: crash fix (cube_detector) + perception optimization (sat_blob/align gating/fp16/Auto/timing) + approach grasp-point/image-aim/danger-gate. Beware of mixing with previous-session uncommitted changes (non-interactive, hunks cannot be split).

## 7. Gotchas
- (key) **6 cm is not an FK bug = JAW_FWD grasp offset**. The two FKs are kinematically identical. approach must target the grasp point (cube+jaw_fwd·fwd). Sign **+** (same as deep_reach_step).
- (key) **image-aim online divergence**: keep per-tick small (iters5/gain0.2/astep0.02) + proximity gate. Large gains blow up.
- (key) **The choice of xy_off metric determines oscillation**: the base-FK residual (|tcp_w-cxy_cmd|) is > gate by jaw_fwd·|fwd_io-ghat|, so always straight + oscillation. **io alignment (|tcp_io-cxy|) is stable** (enters gimbal null-space).
- **owlvit-p32 failed to detect on the real robot** (only synthetic OK) → keep owlv2. **Cameras are YUYV-only + USB2.0 = 10 fps** (no MJPG).
- **KP_APR config A survives in flash** (reboot ≠ lost). all-54 is in RAM (reboot = back to config A). J4 oscillation is either config A (150) residual or the J4 reassembly (docs/24 §17 non-monotonic resonance).
- **fk_bias_xy (io-base) stopgap removed** → explicit grasp model. io.fk_tcp is still used for z (auto-z) only.
- Warning: **large amount uncommitted**: debug_deepreach +279 lines. Commit after HW re-verification (§6-1). compile OK for all.

## Run (next session §6-1 re-verification = first command)
```
.venv-deploy/bin/python deploy/debug_deepreach.py --detector-device cuda \
  --front-openvocab --wrist-openvocab --front-roi 0.4,1.0,0.0,1.0 --track \
  --grip-force 0.05 --grip-torque-target 0.8 --vmax 12 --amax 30 \
  --lookat-hover 0.10 --lookat-approach-stop 0.10 --x-corr 0 \
  --safety-reach-resid 0.15 --rate 30 --front-ov-period 0.5 \
  --lookat-ibvs-xy --lookat-jaw-fwd 0.06 --lookat-image-gate 1.0
```
Live debug = `curl localhost:8899/state` (polling script = scratchpad/poll.py). Check: xy_off→0 (gimbal stable, no oscillation) · wuv→0 · the actual jaw grips the cube.

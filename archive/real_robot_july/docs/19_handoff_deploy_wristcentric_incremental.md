# Handoff — deploy `debug_deepreach` wrist-centric re-add (incremental) + HW motor-connect BLOCKER (2026-07-10, session 10)

> Follows docs/18 (session 8, deploy grasp bring-up). This session: the user asked to extend `debug_deepreach.py` to be **wrist-centric** (rest +z · fast lift · live tracking · wrist seg-mask · descend cam-on-cube · lost-search) → **everything was stacked at once and regressed on HW** (violent oscillation + forward Y-bias in descent) → **cause localized → reverted to the stable version and re-introducing one at a time** → during re-introduction verification **the HW motor CAN connection itself broke (physical)**. Handoff.
>
> **One line**: rest/lift committed on top of the stable baseline (`8dfaad3`, HW OK) + `--track` (live tracking) and `--align-seg` (seg-mask) **re-introduced as opt-in flags that do not touch the servo** (uncommitted, HW verification pending). Top priority now = recover from the **motor CAN connect failure (no response from the very first motor = physical)**.

---

## 0. Warning: top priority now = HW motor CAN connect failure (not code related)
```
motorbridge.errors.CallError: ensure_mode failed: register 10 write ack not received within 50ms
```
- `io.connect()` → lerobot `configure()` → fails at the mode-write (register 10) of the **first motor**. **With INFO logging on, there are 0 per-motor "ensure mode" success lines → from the first motor = the whole CAN chain is silent.** (Not one motor; the whole bus is unresponsive.)
- The driver `configure()` **already retries 10 times** (`num_retry=9`+sleep) before raising → **not transient**. All soft attempts exhausted:
  - Port OK (`/dev/ttyACM0` = `2e88:4603 HDSC` Damiao adapter, dialout OK).
  - **Bridge USB is stable** (dmesg: no re-enumeration/disconnect after the 11:33 enumerate). ttyACM0 mtime changes are traces of run attempts opening the port.
  - **Tried a soft USB reset of the bridge → still fails** → not bus-off.
- **Conclusion = physical/electrical**: (a) the arm CAN cable (bridge↔arm) or (b) **motor power not supplied** or (c) a connector loosened by vibration / motor fault.
- (key) **Most likely**: the earlier **violent oscillation ("thrashing") shook the CAN/power connector or caused a motor over-current fault**. (That oscillation was caused by my servo changes — beware of HW stress.)

### Recovery procedure (all physical — software will not fix it)
1. **Confirm the motor power supply is actually ON** (supply/motor LEDs, output voltage). No power gives exactly this symptom.
2. **Fully OFF → ≥30 seconds → ON** (discharges and clears latched faults that a short re-power does not release).
3. **Reseat connectors**: bridge↔arm CAN + each motor's CAN and power daisy chain (first motor silent = upstream/power).
4. Check the **motor fault LEDs** (blinking/red = that motor latched over-current/over-temperature).
5. Connection check (arm does not move, torque briefly):
```
.venv-deploy/bin/python -c "import logging,sys; logging.basicConfig(level=logging.INFO); sys.path.insert(0,'deploy'); sys.path.insert(0,'src'); from rebot_sim2real.rl.config import load_train_config; from robot_io import DamiaoRobotIO; cfg=load_train_config('configs/train_v6a_yaw.yaml').env; io=DamiaoRobotIO(cfg,'assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf'); io.connect(); print('=== CONNECT OK ==='); io.estop()"
```
- **Bridge soft USB reset recipe** (if bus-off is suspected, sudo): `for d in /sys/bus/usb/devices/*/; do [ "$(cat $d/idVendor 2>/dev/null)" = 2e88 ] && echo 0|sudo tee ${d}authorized && echo 1|sudo tee ${d}authorized; done` (bridge=`3-4`). No effect on physical problems.

---

## 1. Git / stash state (branch `perception-topwrist-coop`)
- **Committed (not pushed, push = user)**:
  - `c0278da` yaw-latch ts-gate + docs/18 (session 8, **HW-verified stable version**).
  - `4a59b85` deep_reach_world (perception-driven deploy) + perception-coop fusion.
  - `8dfaad3` **rest-lift 0.11 + fast-lift** on baseline — **confirmed on HW: no oscillation, no Y error** (this session's increment 1).
  - `27ae1c2` v6c hover07e config + train.py top_cnn_ckpt (v6c leftover, committed by the user).
- **Uncommitted (my work, HW verification pending — increments 2 and 3)**: `deploy/debug_deepreach.py` (+`--track`, `--align-seg`, mask overlay) + `deploy/debug_dashboard.py` (cleanup + mask). ~202 lines.
- **stash@{0}** = the reverted **session 9 full stack** (tracking/seg-mask/rest-lift/fast-lift/RUN-continuous/search/look-at + adversarial-review fixes). The oscillation/Y-bias regression. Recoverable for reference.
- Noise: `uv.lock` (unrelated edit), `deploy/deepreach_sac_world.py` (untracked, not created by me — needs checking).

---

## 2. (key) Regression + localization (core lesson)
In session 9 the requested features were **all stacked at once** → on HW **violent oscillation (left-right/front-back) + descent landing in front of the cube (Y-bias)**.
- **Cause = the changes that touched the servo path**: (1) item4 **replaced the wrist (u,v) source from the verified sd3 → seg ckpt** (suspected real-robot OOD) (2) live **fusion** (front+wrist) pulled the target toward the biased front (3) **#1 RUN continuous servo** (did not stop at reached → fine-tracking of noisy wrist) (4) look-at (diverged).
- **The baseline is accurate + stable** (user confirmed on HW): descent XY = **sd3 wrist correction (use_w near_top)** → no Y-bias. "The cube is in the wrist view but the arm goes in front" = my fusion pulling toward front, not a baseline problem.
- Oscillation also with `--no-track` (frozen one-shot) → not live tracking alone; #1 and seg-(u,v) are also causes.
- → **Revert (stash) + re-introduce one at a time with HW verification**. **stereo does not help the Y-bias** (it inherits the biased front-ray) — deferred.

---

## 3. Re-introduction roadmap + current position
```
[done]    1. rest-lift 0.11 + fast-lift          — commit 8dfaad3, HW OK (stable, Y normal)
[in prog] 2. --track (live tracking)             — implemented, offline-verified, HW pending (uncommitted)
[in prog] 3. --align-seg (seg-mask Align) + mask overlay + dashboard cleanup — implemented, verified, HW pending (uncommitted)
[ ]       4. lost-search (if lost, REST + J1 scan)   — in stash, servo-independent
[ ]       5. RUN continuous tracking (deadband) / cam-on-cube look-at — in stash, affects servo, risky
```
Each increment is an **opt-in flag that does not touch the servo (descend sd3)**; only those passing HW verification are committed. No flags = the committed stable baseline.

---

## 4. This session's implementation details (uncommitted increments 2 and 3)
- **`--track`** (increment 2): live tracking from the fixed **front cam** (deadband `--track-deadband` 1.5cm) → no re-Capture needed, follows a moving cube. **Fixed cam, so no eye-in-hand feedback → does not oscillate** (deadband unit-verified). Descent (u,v) stays sd3. Disarms at reached (tracks only during RUN). The thread publishes `cube_live` → the control loop uses it as cube_world via a deadband anchor. Capture = re-anchor.
- **`--align-seg`** (increment 3): Align box/yaw from the **CNN seg mask** (`--align-detector` default `cube_detector_wrist_seg.pt`) — robust to red clutter/side faces/specular. Being a **separate detector**, the descent (u,v) keeps the primary sd3 (avoids the item4 regression). Mask OOD/no detection → red-mask fallback. `_seg_from_cnn_mask` (upsample mask to full frame → minAreaRect, verified 30°→29.1°).
- **Mask overlay**: `_align_seg` returns (seg8, mask) → `annotate_wrist(..., mask=)` shows the actual segmentation as a **semi-transparent green fill** on the wrist frame.
- **Dashboard cleanup** (`debug_dashboard.py`): **only essentials always shown** (reached · z_gap_io · xy err · cube · track conf · SEARCH · yaw[src] · grasped · grip tau) + the rest in a **collapsible `▸ details`**. Zero information loss; shared dashboard, so it also applies to deep_reach_world.
- **Flags**: `--track`, `--track-deadband` (0.015), `--rest-lift` (0.11), `--lift-step` (0.30), `--align-seg`, `--align-detector`.
- Verification: py_compile · import · seg geometry · deadband unit · dashboard HTML build (all OK). **Not HW verified** (motor blocker).

---

## 5. Design principles (must hold when re-introducing)
- **Keep the verified sd3 for descent (u,v)** — the (u,v) from seg/wrist_seg is sim-val only and caused oscillation via real-robot OOD.
- **Live tracking uses the fixed front cam source + deadband** — chasing the moving wrist in closed loop oscillates.
- **Disarm at reached** (no continuous fine servo) — if needed, a "re-approach only on large moves" deadband.
- **seg-mask is Align-only** (separate detector); the descent servo is off-limits.
- **look-at has a divergence history** — if added, only in the descend-latched close range + optional.

## 6. Detector facts (sd3/sd4/seg)
- Same net (width32/stride8/320×180). The difference is training.
- **sd3** (default, HW-verified (u,v); yaw biased), **sd4** (roll-corrected yaw 2.4°), **wrist_seg/top_seg** (latest 7/9, seg head, sim-val only, HW-OOD risk). Deploy front = HSV red-mask (not CNN).

## 7. Next session actions
1. **Recover motor CAN** (§0 physical procedure) → CONNECT OK.
2. **HW-verify increment 2 `--track`** → commit if it passes → verify and commit **increment 3 `--align-seg`** → 4, 5.

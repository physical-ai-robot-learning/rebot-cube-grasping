# reBot LiftCube — SIM PROVIDED SPEC (answers to the sim-PC preparation checklist)

> ## UPDATE 2026-06-10 (v3) — VISUAL dual-128 policy (camera-only, recommended for sim2real)
> A **visual student** distilled (DAgger) from a **99.8%-success state-only teacher**, trained on a
> **fully-random cube workspace** (target appears left/center/right) with **visual domain randomization**
> (cluttered background + lighting + **camera FoV jitter** + **wrist roll/position jitter** + qpos noise).
> v3 real-robot fixes: **coupled elbow gate REMOVED** (was the cause of "wanders after align, never
> grasps") + **gripper open capped (stroke 0→0.05) & convention corrected** (qpos↑=OPEN; see §5 gripper).
> Runs from **CAMERAS + joint encoders only** — NO cube perception, NO gripper-link FK, NO grasp heuristic.
> - **Deploy file**: `rebot_deploy_visual.py` (Isaac-free, torch only). Checkpoint =
>   `checkpoints/dagger_dual128_final.pt` (= enc_w + enc_f + actor; sim success ~86-90%).
>   Backup = `checkpoints/dagger_dual128_last.pt` (last periodic ckpt, same arch).
> - **Inputs**: joint encoders (8, JOINT_ORDER) + **front camera RGB** + **wrist camera RGB**.
>   state(21) = qpos(8) + target_qpos(7) + dist_to_rest(6). No 43D privileged obs.
> - **CAMERA MOUNTS (must match sim or the policy fails):**
>   - **front** = fixed, **~90 cm FORWARD of the arm base, ~55 cm high, ~45° downward, HEAD-ON**
>     (centered on the arm centerline, aimed at the workspace). Pinhole ~focal 18. Downsamples to **128×128**.
>   - **wrist** = on the **camera bracket on TOP of the gripper**, looking **FORWARD+DOWN at the grasp /
>     workspace ahead** (upright: gripper fingers at the frame bottom, the grasped cube sits low-center).
>     Pinhole ~focal 12. Downsamples to **128×128**.  (NOT the old "between-the-jaws" mount.)
>   - give FULL-RES RGB (≥128×128); the policy area-downsamples internally. DR-trained → robust to
>     background/lighting/FoV, so just feed the raw camera image.
> - **ActionController is IDENTICAL** to the state-only deploy below (scale 0.05 + hard joint limits +
>   coupled elbow gate); the §"slow+constrained" motor velocity limits apply unchanged.
> - **Hardware interface** (`VisualRobotInterface`): `read_joint_pos()[8]`, `get_front_image()`,
>   `get_wrist_image()`, `send_joint_targets(arm6, gripper)`, `go_to_rest()`. Test: `--mock`.
> - Note: the visual policy SEES the cube, so sections §3–§4 (cube world frame / quat) are NOT needed
>   for it — only the camera mounts matter. Joint order / REST / gripper / motor limits below still apply.

> ## UPDATE 2026-06-09 — FINAL "slow + constrained" policy (sim2real-safe) [state-only path]
> The deployed `checkpoints/c51_policy.pt` = `c51_final.pt`, **train step 7.58M, 5236 successes,
> success_rate ~52%**. It was trained with HALVED speed + hard joint limits for safe sim2real.
> **`rebot_deploy.py` already encodes ALL of this** (ActionController) — just match the real MOTOR
> velocity limits below. Changes vs the older spec:
> - **arm action scale 0.1 → 0.05** rad/step (calmer; ~0.5 rad/s).
> - **HARD per-joint target limits** (ActionController.ARM_LO/HI): pan **±90°**; wrist_flex UP **−10°**
>   / DOWN +90°; wrist_yaw **±105°**; wrist_roll **±180°**; lift/elbow [−180°,0].
> - **Coupled elbow gate**: elbow may extend only once shoulder_pan ≥ ~88° (PAN_FORWARD=1.54); below
>   that the elbow is held in a compact tuck [−1.4, −0.7] (no straightening / backward extension).
> - **Real motor velocity limits** (set these on the robot to match sim): shoulder block (pan/lift/
>   elbow) **2.5 rad/s**, wrist block (flex/yaw/roll) **1.5 rad/s**, gripper **0.2 m/s**.
> - reward = 6 ungated terms (action-rate penalty removed; motion shaped purely by the hard limits).
> Everything else below (joint order, obs, REST, TCP, frame, gripper, cube) is unchanged.

**All constants/units/frames needed for real-robot deployment**, extracted from the training PC (Isaac). On the inference PC, use these values to finalize
the constants in `rebot_deploy.py` and the `RobotInterface` adapter (units/FK/frames).

> **Joint order verified empirically**: 3689 successes occurred, and success requires `reached_rest = ||arm_target[:6] − REST_QPOS|| < 0.2`.
> If the order were wrong, this value could never get below 0.2 and success would never have occurred → **order confirmed correct.**
> (It can also be printed with `dump_sim_spec.py`, but that was skipped because the GPU is busy training; the values below are correct.)

---

## 1. Joint order / observation / action / REST / TCP  (matches rebot_deploy.py constants)

**JOINT_ORDER (8, articulation order):**
```
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper, gripper_joint2]
```
- First 6 = arm (rad), 7th = gripper (m), 8th = gripper_joint2 (m, mirror of gripper).

**OBS (43 total, in exactly this order):**
| # | Item | dim | Content |
|---|---|---|---|
| 1 | qpos | 8 | positions of the 8 joints (encoder) |
| 2 | qvel | 8 | velocities of the 8 joints |
| 3 | target_qpos | 7 | **commanded delta-target** (arm6+gripper1) — not a sensor; tracked by ActionController |
| 4 | dist_to_rest | 6 | `arm_qpos[:6] − REST_QPOS` (vector) |
| 5 | obj_pos | 3 | target (red) cube position, **world frame** |
| 6 | obj_quat | 4 | cube orientation, **wxyz** |
| 7 | tcp_pos | 3 | `gripper_link_pos + R(gripper_link_quat)@(-0.086,0,0.103)` |
| 8 | tcp_to_obj | 3 | `obj_pos − tcp_pos` |
| 9 | is_grasped | 1 | 0/1 (heuristic on the real robot) |

**ACTION dim = 7** = arm_action(6, delta rad) + gripper_action(1, delta m). Output range [-1,1].

**REST joint_pos (rad / gripper m):**
```
shoulder_pan  0.00 | shoulder_lift -1.15 | elbow_flex -1.00 | wrist_flex 0.55
wrist_yaw     0.00 | wrist_roll     0.00 | gripper 0.02 | gripper_joint2 0.02
```
→ `REST_QPOS = [0, -1.15, -1.00, 0.55, 0, 0]` (6 arm joints).

**TCP offset = `(-0.086, 0.0, 0.103)`** (gripper_link local). Both ee_frames use the same offset → that point is the TCP.

---

## 2. URDF + meshes (for FK) — deliver these together

- **URDF**: `robot_description_gripper/rebot_prepared.urdf` (the one converted to the sim USD).
  The original is `robot_description_gripper/reBot_B601_DM_with_gripper.urdf`.
- **Meshes**: all of `robot_description_gripper/meshes/` (referenced by the URDF via relative paths). The whole folder is 27MB.
- The `gripper_link` link name **exists as-is** in the URDF (FK target). TCP = apply the offset above.
- Asset actually loaded by the sim: `LEISAAC_ASSETS_ROOT/robots/rebot_follower.usd`
  (`LEISAAC_ASSETS_ROOT = ~/workspace/rebot_isaac_rl/leisaac/assets`).
- → **Zip `robot_description_gripper/` and send it to the real-robot PC** (URDF+mesh together; load into an FK library such as yourdfpy/pytorch_kinematics).

---

## 3. SIM WORLD frame / scene layout  (z-up, meters)

- **Robot base spawn**: pos `(0.35, -0.64, 0.01)`, quat `(0.70711, 0, 0, 0.70711)` = **+90° yaw (z axis)**.
- **Target (red) cube "cube"**: base `(0.25, -0.30, 0.06)` + random `x,y ∈ ±0.10`, `yaw ∈ ±π`.
  → actual spawn: `x∈[0.15,0.35], y∈[-0.40,-0.20], z=0.06 (center), random yaw`.
- Distractors: purple base `(0.35,-0.30)`, blue base `(0.45,-0.30)` (not included in obs/reward).
- **Cube size 0.04 m (4cm)**, resting center z = 0.06 (table top ~0.04 + half 0.02). Table top z ≈ 0.04.
- **Yes, z-up.** obj_pos/tcp_pos are **global world absolute coordinates** (root_pos_w) — the frame in which the base sits at the pose above.
  → The cube coordinates captured by the camera must be **transformed into this world frame** before being put into obj_pos.

---

## 4. Cube orientation (quat) convention

- `obj_quat` = `root_quat_w`, **wxyz**.
- **Orientation is randomized**: yaw (z axis) `±π`. roll/pitch are fixed (cube lies flat on the table).
  → quat ≈ `(cos(yaw/2), 0, 0, sin(yaw/2))`.
- Because the cube is a symmetric cube there is yaw ambiguity, but the actual root_quat_w was fed into obs.
  On the real robot, either feed the **camera-estimated yaw** as wxyz, or, if precise estimation is hard, use a quat assuming it is upright;
  the impact is small (the policy mainly depends on the tcp_to_obj relative position).

---

## 5. Gripper spec

- **sim = PRISMATIC `0 ~ 0.0715 m`; real = ROTARY `0° ~ +320°`** — the kinematics differ, so raw values cannot be mapped 1:1.
  **You must map via the "mouth gap"** (both produce a physical gap).
- **Direction (confirmed by render + measurement, `diag_gripper_openclose`/`diag_gripper_stroke`):**
  **sim `qpos ↑ = open (jaws spread)`, `qpos ↓ = closed`** ↔ **real `angle ↑(→+320°) = closed`, `↓(→0°) = open`** (opposite direction).
  (The previous spec's "qpos>0.05=closed" **was backwards** — discarded.)
- **Mapping table (sim qpos ↔ mouth gap ↔ real motor angle; match via gap):**

  | sim qpos | mouth gap | state | → real motor angle |
  |---|---|---|---|
  | 0.000 | 0.0 cm | fully closed | **+320°** (fully closed) |
  | 0.030 | ~3.5 cm | **4cm cube grip** (light squeeze) | near +320° (blocked by cube) |
  | 0.035 | 4.0 cm | cube width | — |
  | **0.045** | **~5.5 cm** | **max opening in training** | **~+40°** (moderate open) |
  | (0.0715 | 10.4 cm | mechanical max = over-open | **0°** — **forbidden in training/deployment**) |

- **Opening range capped to `[0, 0.045]`** (training, feedback v2) — matches the real safe range (closed +320° ~ moderately open +40°).
  Over-opening (real 0° / sim q>0.045) is **excluded** from the policy distribution → the policy cannot command over-opening. REST=0.02m.
- Grasp: the 4cm cube is lightly squeezed at `q≈0.03` (mouth ~3.5cm) (= near real +320°, blocked by the cube). Closing further is blocked by the cube, so it does not close more.
- **`gripper_joint2` simply mirrors `gripper`** (the action drives both to the same target).
- Action: **delta-on-target**, gripper scale `0.0143 m/step`, arm scale `0.05 rad/step`, action=0=hold.
  The gripper target is clamped to **`[0, 0.045]`**. (ActionController handles this — just send its output to the motors.)
- Reference PD (for matching real-robot gains): stiffness 2000, damping 100, effort 200N, velocity 0.2 m/s, friction 2.0.

---

## 6. Checkpoint metadata

- Deployment default: **`deploy_package/checkpoints/c51_policy.pt`**
  = `c51_ckpt_8396800.pt`, **train step 8.40M, success ~3689, success_rate ~25% (rising)**.
- Algorithm: **distributional C51 SAC, state-based (obs 43), action 7 ∈ [-1,1]**. (Isaac not required; loads with torch alone)
- Training is scheduled to auto-stop when it reaches 5000 successes and save `c51_final.pt` → at that point, updating `c51_policy.pt` to the final version
  is recommended (higher success rate). The current version also has 25% success, so it can be tested right away.
- Identical file for traceability: `c51_step8396800.pt`.

---

### Inference PC summary checklist
1. `pip install -r requirements.txt` (torch, numpy) + an FK library (yourdfpy, etc.).
2. Take `robot_description_gripper/` (URDF+mesh) and compute the `gripper_link` world pose via FK.
3. Implement `RobotInterface`: 8-joint encoders (order above), cube pose (world frame, wxyz), gripper_link pose, is_grasped heuristic, motor commands (arm rad · gripper m, both fingers identical).
4. Verify frames/units/order 1:1 against the values above → `run(load_policy("checkpoints/c51_policy.pt"), ReBot())`.

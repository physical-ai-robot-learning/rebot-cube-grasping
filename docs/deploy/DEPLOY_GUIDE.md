# reBot LiftCube — Visual Policy Deployment Guide (detailed inference-side explanation)

Written 2026-06-12. Audience: whoever is responsible for real-robot (reBot B601-DM) deployment/inference.
This document explains **how the inference pipeline works** + **what must be supplied** + **what environment each checkpoint was trained in**.

---

## 0. Package contents

```
deploy_package/
├── rebot_deploy_visual.py     # visual policy inference (main)
├── rebot_deploy.py            # state-based policy inference (fallback, requires CV cube detection)
├── checkpoints/
│   ├── dagger_dual128_fresh.pt    # student A: FRESH training (step 345600)
│   ├── dagger_dual128_resume.pt   # student B: RESUME training (step 691200)  ← trained longer
│   ├── dagger_dual128_final.pt    # default = copy of resume
│   └── c51_policy.pt              # state-based teacher (for fallback)
├── DEPLOY_SPEC.md             # sim↔real PARITY spec (joint order/units/limits)
├── DEPLOY_GUIDE.md           # (this document)
└── README.md
```

### The two students (A/B) — what is the difference?
Both were **trained with the same environment/DR** (§4 below), and their inputs/architecture/inference code are **completely identical**. The only difference is **the training starting point**:
- **A (fresh)**: trained from scratch from random initialization.
- **B (resume)**: continued (warm-start) from the previous DR checkpoint (step 307200) → trained longer (691200).

→ **Alternate between the two on the real robot and compare them.** Just change `--checkpoint`. (B is trained longer, so it is expected to usually be more stable.)

---

## 1. One-line summary — inference needs only 2 cameras + joint encoders
The visual policy needs **no cube perception, FK, or grasp heuristic at all.** The cameras take over that role. What you must supply:
- **front camera image** (RGB)
- **wrist camera image** (RGB, eye-in-hand above the gripper)
- **8 joint encoders** (arm 6 + gripper 2)
The policy then outputs a **7D action** (arm 6 delta + gripper 1 delta) at 10Hz.

---

## 2. Inference pipeline (step by step)

```
[front 1920x1080]  ┐
[wrist 1920x1080]  ┤→ _ds(): CENTER-CROP 1:1 → area-resize 128x128  ──┐
                   │                                                   ├→ Actor.mean_action → a[7] (tanh, [-1,1])
[joint encoders 8] ┴→ build_state(): state21 (qpos8+target7+dist6) ───┘        │
                                                                                ▼
                                                              ActionController.step(a, pan_qpos)
                                                              = target += a*scale, per-joint clamp, elbow gate
                                                                                ▼
                                                              send_joint_targets(7)  →  motors
```

### 2-1. Camera preprocessing `_ds()` — MOST IMPORTANT (cause of the previous failure)
- The input is the real camera's **16:9 (1920×1080)**. The student was trained on **square 128×128**.
- You **must CENTER-CROP the center to 1:1 and then resize to 128** (`_ds()` does this automatically).
- Resizing 16:9 directly to a square gives **~1.78x horizontal compression (anamorphic)** → a distortion the CNN never saw during training = **OOD** → collapses to action≈0 or saturation. (This was the #1 cause of the previous deployment failure and is now fixed automatically in the code.)
- **Physical camera mount/zoom**: adjust so that the central 1:1 crop region covers the **sim FoV (front ~60° / wrist ~82°)**. That way the cube's share of the image matches training. (See DEPLOY_SPEC for front 90cm ahead/55cm high/45°down/head-on, and wrist mount above the gripper between-jaws.)

### 2-2. Encoder/actor
- `enc_w` (wrist 128) + `enc_f` (front 128): each a SQUINT-style CNN → features.
- `Actor.mean_action(wf, ff, state21)` → `tanh(mean)` = **7D action**, deterministic inference.
- Normalization: `img/255 - 0.5` inside the encoder (no separate standardization).

### 2-3. state21 (proprio) — `build_state(joint_pos, target_qpos7)`
```
state21 = qpos[8]  | target_qpos[7] | dist_to_rest[6]
          (encoders) (kept by control loop) (= arm_qpos[:6] - REST_QPOS)
```
- `qpos[8]` = JOINT_ORDER order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper, gripper_joint2]` (encoder raw→rad).
- `target_qpos[7]` = not a sensor. The command target (arm6+gripper1) that ActionController accumulates and maintains.
- `dist_to_rest[6]` = `arm_qpos[:6] - REST_QPOS`, `REST_QPOS=[0.0,-1.15,-1.00,0.55,0.0,0.0]`.
- **The visual policy does not put cube position/FK/grasp status into the state** (the cameras handle that).

### 2-4. ActionController (1:1 replica of the sim action term)
- 10Hz, arm delta scale **0.05** rad/step, gripper delta scale **0.0143** m/step, action=0=hold.
- Per-joint hard clamp `ARM_LO/HI` (real-robot safety + sim parity).
- **Coupled elbow gate**: the elbow is free only when shoulder_pan ≥ 1.54rad (~88°); otherwise tuck [-1.4,-0.7] (→ pass the measured pan to `ctrl.step(a, pan_qpos)`).
- Gripper clamp **[0.0, 0.05]** m (§3 gripper).

---

## 3. Gripper convention (be sure to check)
- **sim qpos convention (confirmed by measurement)**: **qpos↑ = OPEN (mouth spread), qpos↓ = CLOSED**. (The old "qpos>0.05=closed" was backwards.)
- Mouth gap: q0→0cm (fully closed), **q0.03→~3.5cm (4cm cube grasp)**, q0.05→~6.6cm (open). Deployment clamp `GRIP [0.0, 0.05]`.
- **The real gripper = rotary**: measured fully open 0.3rad (+17°), 4cm grasp 3.5rad (+201°), mechanical limit +320°. → sim (prismatic m) ↔ real (rad) **linear mapping via mouth gap** + over-close clamp (`GripperCalib`).
- Key point: **GRIP_LO must be 0.** Grasp force comes from pushing the closing target well below the cube contact point (~q0.035). Limit real-robot grasp torque with a **current/torque limit**, not with the position target.

---

## 4. Training domain (the scene the student "expects") — match the inference environment to this
The student was trained with **a sim matched to the actual inference environment (office) + domain randomization (DR)**. Keeping the inference environment close to the following gives good transfer:

| Element | Training setting (= real environment to match) |
|---|---|
| Work surface | **white glossy table** (including specular highlights) |
| Gripper | **black** |
| Robot arm | render color (not randomized) |
| Background | **office-style** (indoor photos) — trained to be robust to variations |
| Cubes | red = target (4cm), blue/purple = distractors (colors fixed since they are task cues) |
| Cameras | front head-on ~60°FoV / wrist above gripper ~82°FoV, square 128 |

Variations the student was additionally trained to be robust to (DR): **ISP color/exposure, specular glare, lighting, camera focal (±30%), image scale, wrist rotation/translation, proprio noise.** In other words, it tolerates somewhat different lighting, color tone, and background. **However, the aspect ratio (§2-1) and camera FoV/cube size have a narrow robust range, so it is better to match them via the mount.**

> Why this way? In the previous deployment, the visual policy collapsed due to **OOD on the real cameras** (glare, different background, aspect ratio). This package is the version that closes that gap with (1) the aspect-ratio center-crop fix + (2) the domain matching/stronger DR above.

---

## 5. How to run

```bash
cd deploy_package
# Student B (resume, trained longer, recommended default)
python rebot_deploy_visual.py --checkpoint checkpoints/dagger_dual128_resume.pt --device auto
# Student A (fresh) — for comparison
python rebot_deploy_visual.py --checkpoint checkpoints/dagger_dual128_fresh.pt   --device auto
```
- `--device auto` → automatically cuda | **mps (Apple Silicon)** | cpu. A Mac is sufficient (inference takes a few ms @10Hz; the Mac only does camera + motor I/O).
- Interface to implement (`VisualRobotInterface`): `read_joint_pos()→[8]`, `get_front_image()→HxWx3`, `get_wrist_image()→HxWx3`, `send_joint_targets(7)`. **No cube perception/FK/grasp detection needed.**

---

## 6. State-based fallback (`rebot_deploy.py` + `c51_policy.pt`)
If the visual policy fails, or for validation, this path **obtains the cube position with classical CV (HSV red detection) and runs the state-based teacher**.
- This path is camera-agnostic (43D state obs), so it is unaffected by camera OOD.
- However, CV/FK must supply the **43D obs**: `qpos8|qvel8|target7|dist6|obj_pos3|obj_quat4|tcp_pos3|tcp_to_obj3|is_grasped1`. The key items are `obj_pos` (HSV→deproject→sim world transform), `tcp_pos` (encoder FK, `robot_description_gripper.zip`), `is_grasped` (gripper qpos + motor current heuristic). For details see DEPLOY_SPEC §state.

---

## 7. Troubleshooting (OOD symptoms)
- **Actions are almost 0 (idle) or all +saturation, and slightly changing the input changes the output completely** → typical OOD. Check in this order:
  1. **Aspect ratio**: make sure you are not simply resizing 16:9 to a square (always center-crop 1:1 first — `_ds` does this, but do not insert any other preprocessing).
  2. **Camera FoV/pose**: whether the cube's on-screen size and viewpoint are similar to sim (front ~60°, head-on 90cm/55cm/45°down; wrist above the gripper between-jaws).
  3. **Color/exposure**: make sure it is not extreme (DR provides some tolerance).
  4. If it still does not work, confirm operation with the **state-based fallback** (§6).
- Note: check that the left/right and up/down orientation of both camera inputs, and the wrist roll direction, are the same as in sim (somewhat robust thanks to wrist roll ±20° DR in training).

---

## 8. Caveats (training state of these two checkpoints)
- Both students are at a **sim success rate of ~0.5 under heavy DR** (sim sr is moderate because DR is strong, but the goal is real-world robustness). **The real evaluation is transfer to the real robot.**
- Training is still ongoing (target 50000 successes), and an updated version can be delivered again once it converges further. The current version is a **snapshot for checking the pipeline and transfer**.

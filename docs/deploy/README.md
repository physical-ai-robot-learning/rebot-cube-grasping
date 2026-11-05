# reBot LiftCube — Real-World Deploy Package

Portable inference (no Isaac Sim/Lab — only `torch` + `numpy`). Copy this folder to the robot PC.
**Two policies are included:**

| | file | input | hardware you implement |
|---|---|---|---|
|  **VISUAL (recommended)** | `visual_deploy.py` + `results/checkpoints/dagger_dual128_final.pt` | front + wrist **cameras** (128×128) + joint encoders | joints + 2 images only |
| state-only | `state_deploy.py` + `results/checkpoints/c51_policy.pt` | 43D state incl. **cube pose** | joints + cube perception + FK + grasp heuristic |

> ##  VISUAL deploy (recommended — camera-only, much simpler hardware side)
> **dual-128 student v3** (DAgger-distilled from a **99.8% state-only teacher**, fully-random cube
> workspace, domain-randomized: cluttered background + lighting + **FoV jitter** + **wrist roll/pos
> jitter** + qpos noise; sim success ~86-90%). v3 fixes from real-robot feedback: **coupled elbow gate
> REMOVED** (fixes "wanders after align, never grasps") and **gripper open capped + convention corrected**
> (qpos↑=OPEN, grip 4cm cube @ q≈0.03 — fixes over-opening; see §gripper in DEPLOY_SPEC). It SEES the
> cube through the cameras, so you DON'T implement cube perception / gripper-link FK / grasp heuristic —
> only joint encoders + two camera images. (`dagger_dual128_last.pt` = backup checkpoint.)
> ```bash
> pip install -r requirements.txt
> python visual_deploy.py --checkpoint results/checkpoints/dagger_dual128_final.pt --mock --max-steps 30   # flow test
> ```
> Implement `VisualRobotInterface` (see `visual_deploy.py`):
> `readJointPos()[8]` · `getFrontImage()` HxWx3 RGB · `getWristImage()` HxWx3 RGB ·
> `sendJointTargets(arm6, gripper)` · `goToRest()`. Then `run(VisualPolicy("results/checkpoints/dagger_dual128_final.pt"), YourRobot())`.
> **CAMERA MOUNTS must match sim** (see `DEPLOY_SPEC.md` top): front = ~90cm fwd of arm base / ~55cm high
> / ~45° down / **head-on** (focal ~18); wrist = on the **camera bracket on top of the gripper**, looking
> **forward+down at the grasp** (focal ~12) — both downsampled to **128×128**.
>
> ###  Mac (Apple Silicon / Intel) — runs fine, GPU NOT needed
> The policy is tiny (dual-128 CNN, 2.3MB) at 10 Hz → a few ms/step on a Mac CPU. No Isaac, no CUDA.
> ```bash
> pip install torch numpy            # Mac CPU/MPS build of torch (no CUDA)
> python visual_deploy.py --checkpoint results/checkpoints/dagger_dual128_final.pt --mock --device auto
> ```
> `--device auto` picks **mps** (Apple-Silicon GPU) or **cpu** automatically — both are plenty (the
> image area-downsample always runs on CPU, so MPS op gaps don't bite). The Mac only needs the camera
> feeds + joint encoder reads + motor commands wired into `VisualRobotInterface`; that I/O — not compute —
> is the only Mac-side work.
> state(21) = qpos(8)+target_qpos(7)+dist_to_rest(6); same `ActionController` (scale 0.05 + joint limits +
> elbow gate) + same motor velocity limits as below.
>
> The state-only path below is the fallback (needs cube perception). Joint order / REST / gripper / motor
> limits / control rate are SHARED by both.

---

## state-only deploy  (`state_deploy.py`) — needs cube perception
Portable inference for a policy trained by `packages/sim/train_c51.py` (distributional-C51 SAC).

## Files
| file | purpose |
|---|---|
| `state_deploy.py` | standalone Actor + checkpoint loader + obs builder + delta action controller + deploy loop + `RobotInterface` ABC + `MockRobot` |
| `dump_sim_spec.py` | run on the TRAINING PC (needs Isaac) to print the exact joint order / obs layout to verify parity |
| `requirements.txt` | torch, numpy |

## Quick start
```bash
pip install -r requirements.txt
# 1) flow test (no hardware): obs -> policy -> action shapes
python state_deploy.py --checkpoint /path/to/c51_final.pt --mock --max-steps 30
# 2) real robot: implement RobotInterface (see below), then call run(actor, YourRobot())
```
Get a checkpoint from training: `rl_output/rl_c51/c51_ckpt_*.pt` (or `c51_final.pt` after 5000 successes).

## What you MUST implement (subclass `RobotInterface`)
The policy is **state-based** (no camera), so you must supply the cube pose from your own perception.
```python
class ReBot(RobotInterface):
    def readJointState(self):     # (joint_pos[8], joint_vel[8]) in JOINT_ORDER
    def getCubePose(self):        # (pos[3], quat_wxyz[4]) of the target cube, SIM WORLD frame
    def getGripperLinkPose(self):# (pos[3], quat_wxyz[4]) of gripper_link, SIM WORLD frame (your FK)
    def is_grasped(self):           # bool (e.g. gripper qpos>0.05 AND motor current high)
    def sendJointTargets(self, arm_target, gripper_target)  # arm rad x6, gripper m (drive BOTH fingers)
    def goToRest(self)            # move to REST before an episode
```
Then:
```python
from state_deploy import load_policy, run
run(load_policy("c51_final.pt"), ReBot())
```

## sim ↔ real PARITY — get these right or the policy fails
1. **Joint order** — `JOINT_ORDER = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw,
   wrist_roll, gripper, gripper_joint2]`. **Verify** with `dump_sim_spec.py` and fix `state_deploy.py`
   if your sim prints a different order. `gripper_joint2` mirrors `gripper`.
2. **Units** — arm = radians, gripper = **meters** (prismatic, stroke 0..0.0715). Gripper convention is
   INVERTED: `qpos > 0.05 = closed/gripping`, `< 0.035 = open`.
3. **Frame** — `obj_pos`, `gripper_link` pos are ABSOLUTE in the **sim world frame** (robot base spawned
   at `(0.35, -0.64, 0.01)`, z-up). Your perception + FK must report positions in that same frame, or
   the absolute-position obs (obj_pos, tcp_pos) won't match what the policy saw in sim.
4. **Control** — 10 Hz. Actions are **delta-on-target**: `target += clip(a,-1,1)*scale`
   (arm scale 0.1 rad, gripper 0.0143 m), gripper clamped to [0,0.0715]. `ActionController` handles
   this; just send its output to the motors. Initialize the target to the measured joints at reset.

## Observation vector (43-dim, in this exact order)
`qpos(8) | qvel(8) | target_qpos(7) | dist_to_rest(6) | obj_pos(3) | obj_quat(4) | tcp_pos(3) |
tcp_to_obj(3) | is_grasped(1)`
- `target_qpos(7)` = the commanded delta-target (arm6 + gripper1), tracked by `ActionController` — NOT a sensor.
- `dist_to_rest(6)` = `arm_qpos - REST_QPOS` (vector), `REST_QPOS=[0,-1.15,-1.00,0.55,0,0]`.
- `tcp_pos` = `gripper_link_pos + R(gripper_link_quat) @ (-0.086,0,0.103)` (computed by `tcp_from_gripper_link`).
- `tcp_to_obj` = `obj_pos - tcp_pos`. `is_grasped` = your heuristic.
- The policy ingests RAW obs (no normalization — the network's input LayerNorm handles scale), so the
  ONLY requirement is matching the sim's units/order/frame above.

## Success criterion (for reference, from the sim)
`success = lifted (cube_z ≥ rest+1mm) AND grasped AND reached_rest (||arm_target - REST|| < 0.2)`.
The policy is trained to grasp the cube, lift it, and return the arm to REST while holding.

## Notes / limitations
- Cube perception (`getCubePose`) is the main thing you provide — camera+pose-estimation, AprilTag, or
  motion capture, transformed into the sim world frame. Accuracy here directly limits real performance.
- `is_grasped` real heuristic is approximate; tune the threshold on hardware.
- Start at low speed and with E-stop ready. Validate `--mock` flow first, then dry-run above the table.

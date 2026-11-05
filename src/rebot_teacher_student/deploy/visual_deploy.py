"""reBot LiftCube — portable VISUAL real-world inference (Isaac-free).

Runs the VISUAL student policy (trained by train_dagger_visual.py: DAgger distillation of the
state-only C51 teacher) on the REAL reBot B601-DM. Only needs `torch` + `numpy`.

WHY THIS IS SIMPLER THAN rebot_deploy.py (state-only):
  the policy SEES the cube through the cameras, so you DO NOT implement cube perception, gripper-link
  FK, or a grasp heuristic. You only provide: joint encoders + two camera images.

WHAT YOU MUST IMPLEMENT (subclass VisualRobotInterface):
  - read_joint_pos()   -> joint_pos[8]  in JOINT_ORDER (gripper_joint2 mirrors gripper)
  - get_front_image()  -> HxWx3 uint8 RGB  (front camera)
  - get_wrist_image()  -> HxWx3 uint8 RGB  (eye-in-hand wrist camera)
  - send_joint_targets(arm_target[6], gripper_target)   (arm rad, gripper meters; both fingers)
  - go_to_rest()

CRITICAL sim<->real PARITY:
  1. JOINT_ORDER matches the sim articulation order.
  2. CAMERA MOUNTS must match sim (see DEPLOY_SPEC.md):
       front = ~90cm forward of the arm base, ~55cm high, ~45deg down, HEAD-ON (centered on the arm).
       wrist = on the camera bracket on TOP of the gripper, looking FORWARD+DOWN at the grasp /
               workspace ahead (the grasped cube sits low in frame, fingers at the bottom; upright).
     Camera intrinsics/FoV should roughly match (front ~focal18, wrist ~focal12 pinhole).
  3. state(21) = qpos(8) + target_qpos(7) + dist_to_rest(6).  arm=radians, gripper=METERS.
  4. Control rate 10 Hz. Actions are DELTA-on-target (ActionController, identical to state-only deploy).
  5. Images are downsampled in-policy to wrist 128x128 / front 128x128 (area). Give full-res RGB.
     (Trained with visual domain randomization — greenscreen bg, lighting, FoV jitter — so it is
      robust to background/lighting; just feed the raw camera RGB.)
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants (match env_config.py + visual_obs.py) ──
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw",
               "wrist_roll", "gripper", "gripper_joint2"]
N_JOINTS, N_ARM, N_ACT = 8, 6, 7
STATE_DIM, WRIST_SIZE, FRONT_SIZE = 21, 128, 128   # dual-128 student (wrist@128 + front@128)
REST_QPOS = np.array([0.0, -1.15, -1.00, 0.55, 0.0, 0.0], dtype=np.float32)
ARM_SCALE, GRIP_SCALE = 0.05, 0.0143
# Gripper qpos (meters) convention — MEASURED in sim (diag_gripper_stroke), CORRECTED 2026-06-10:
#   qpos UP = OPEN (jaw mouth wider), qpos DOWN = CLOSED.  mouth: q0->0cm(shut), q0.03->~3.5cm (grips a
#   4cm cube), q0.05->~6.6cm (open). The old "qpos>0.05=closed" was BACKWARDS. GRIP_LO MUST stay 0.0 — the
#   grip force comes from the close TARGET pushing well below the cube-contact qpos(~0.035); a higher
#   GRIP_LO (e.g. 0.03) leaves no push and the grip never forms (diag_gripper_stroke). Limit real grip
#   TORQUE via current/torque limits, NOT the position target.
#   Real mapping: smaller sim qpos = more CLOSED; map your real "closed/grip" to q~0.03 and "open" to q~0.05.
GRIP_LO, GRIP_HI = 0.0, 0.05
ARM_LO = np.array([-1.5708, -3.14, -3.14, -0.1745, -1.8326, -3.1416], dtype=np.float32)
ARM_HI = np.array([ 1.5708,  0.00,  0.00,  1.57,    1.8326,  3.1416], dtype=np.float32)
PAN_IDX, ELBOW_IDX, PAN_FORWARD = 0, 2, 1.54
ELBOW_TUCK_LO, ELBOW_TUCK_HI = -1.4, -0.7
CONTROL_HZ = 10.0


# ════════════════════════════════════════════════════════════════════════════════
#  Visual policy networks  (must match src/rebot_teacher_student/student/visual_nets.py so the ckpt loads)
# ════════════════════════════════════════════════════════════════════════════════
class CNNEncoder(nn.Module):
    REPR_DIM = 1024

    def __init__(self, image_size):
        super().__init__()
        if image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 128:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),       # 128 -> 31
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),      # 31 -> 14
                nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),      # 14 -> 6
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())  # 6 -> 4 => 64*4*4 = 1024
        else:
            raise ValueError(f"unsupported image size {image_size}")

    def forward(self, img_uint8):
        x = img_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(x)


class Projection(nn.Module):
    REPR_DIM = 50 + 50 + 256

    def __init__(self, n_state):
        super().__init__()
        self.wrist_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.front_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, wf, ff, state):
        return torch.cat([self.wrist_proj(wf), self.front_proj(ff), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state=STATE_DIM, n_act=N_ACT, hidden=256):
        super().__init__()
        self.proj = Projection(n_state)
        self.fc = nn.Sequential(
            nn.Linear(Projection.REPR_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)   # unused at inference; present so state_dict loads

    def mean_action(self, wf, ff, s):
        return torch.tanh(self.mean(self.fc(self.proj(wf, ff, s))))


def pick_device(device="auto"):
    """Resolve the inference device. 'auto' -> CUDA if present, else Apple-Silicon MPS, else CPU.
    On a Mac this gives 'mps' (Apple GPU) or 'cpu' — both run this tiny dual-128 policy at 10Hz with
    huge headroom (a few ms/step). The area-downsample runs on CPU regardless (see _ds), so MPS op gaps
    don't matter."""
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisualPolicy:
    """Loads enc_w + enc_f + actor from dagger_final.pt and runs deterministic inference."""

    def __init__(self, checkpoint_path, device="auto"):
        self.device = pick_device(device)
        self.enc_w = CNNEncoder(WRIST_SIZE).to(self.device).eval()
        self.enc_f = CNNEncoder(FRONT_SIZE).to(self.device).eval()
        self.actor = Actor().to(self.device).eval()
        ck = torch.load(checkpoint_path, map_location=self.device)
        self.enc_w.load_state_dict(ck["enc_w"])
        self.enc_f.load_state_dict(ck["enc_f"])
        self.actor.load_state_dict(ck["actor"])
        print(f"[deploy-v] loaded visual student from {checkpoint_path} (step {ck.get('step','?')}) "
              f"on device={self.device}")

    @staticmethod
    def _ds(rgb_uint8, size):
        """HxWx3 uint8 -> (1,size,size,3) uint8. CENTER-CROP to 1:1 FIRST, then area-downsample to size.
        The student trained on SQUARE frames (sim renders 192x192); resizing a real 16:9 (1920x1080) frame
        straight to square anamorphically SQUASHES it ~1.78x horizontally = guaranteed OOD (was the prime
        cause of actions~=0 / saturation). Center-cropping to the largest centered square first restores the
        sim geometry (no-op when the input is already square). NOTE: the crop must subtend ~the sim FoV
        (front ~60deg / wrist ~82deg) -> set the physical camera mount/zoom so the central 1:1 region frames
        the workspace like the sim render."""
        x = torch.as_tensor(rgb_uint8[..., :3], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        h, w = x.shape[-2], x.shape[-1]
        if h != w:                                          # center-crop to the largest centered square
            s = min(h, w); top = (h - s) // 2; left = (w - s) // 2
            x = x[:, :, top:top + s, left:left + s]
        x = F.interpolate(x, size=(size, size), mode="area")
        return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)

    @torch.no_grad()
    def act(self, front_rgb, wrist_rgb, state21):
        w = self._ds(wrist_rgb, WRIST_SIZE).to(self.device)
        f = self._ds(front_rgb, FRONT_SIZE).to(self.device)
        s = torch.as_tensor(state21, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor.mean_action(self.enc_w(w), self.enc_f(f), s)
        return a.squeeze(0).cpu().numpy()


def build_state(joint_pos, target_qpos7):
    """21D student state = qpos(8) + target_qpos(7) + dist_to_rest(6). arm=rad, gripper=m."""
    jp = np.asarray(joint_pos, dtype=np.float32)
    tq = np.asarray(target_qpos7, dtype=np.float32)
    dist = jp[:N_ARM] - REST_QPOS                       # 6D (arm only)
    s = np.concatenate([jp, tq, dist]).astype(np.float32)
    assert s.shape[0] == STATE_DIM, f"state {s.shape[0]} != {STATE_DIM}"
    return s


# ════════════════════════════════════════════════════════════════════════════════
#  Action controller  (IDENTICAL to the state-only deploy: delta-on-target + clamps + elbow gate)
# ════════════════════════════════════════════════════════════════════════════════
class ActionController:
    def __init__(self):
        self.arm_target = REST_QPOS.copy()
        self.grip_target = 0.02

    def reset(self, arm_qpos, grip_qpos):
        self.arm_target = np.asarray(arm_qpos, dtype=np.float32).copy()
        self.grip_target = float(grip_qpos)

    def step(self, action, pan_qpos):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.arm_target = np.clip(self.arm_target + a[:N_ARM] * ARM_SCALE, ARM_LO, ARM_HI)
        if float(pan_qpos) < PAN_FORWARD:
            self.arm_target[ELBOW_IDX] = float(np.clip(self.arm_target[ELBOW_IDX], ELBOW_TUCK_LO, ELBOW_TUCK_HI))
        self.grip_target = float(np.clip(self.grip_target + a[6] * GRIP_SCALE, GRIP_LO, GRIP_HI))
        return self.arm_target.copy(), self.grip_target

    @property
    def target_qpos7(self):
        return np.concatenate([self.arm_target, [self.grip_target]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Hardware abstraction
# ════════════════════════════════════════════════════════════════════════════════
class VisualRobotInterface:
    def read_joint_pos(self):
        """-> joint_pos[8] in JOINT_ORDER (gripper_joint2 mirrors gripper)."""
        raise NotImplementedError

    def get_front_image(self):
        """-> HxWx3 uint8 RGB from the FRONT camera (mounted per DEPLOY_SPEC)."""
        raise NotImplementedError

    def get_wrist_image(self):
        """-> HxWx3 uint8 RGB from the eye-in-hand WRIST camera."""
        raise NotImplementedError

    def send_joint_targets(self, arm_target, gripper_target):
        raise NotImplementedError

    def go_to_rest(self):
        raise NotImplementedError

    def close(self):
        pass


class MockVisualRobot(VisualRobotInterface):
    """No-hardware stand-in to smoke-test the obs->policy->action flow on any PC (torch only)."""
    def __init__(self):
        self._jp = np.concatenate([REST_QPOS, [0.02, 0.02]]).astype(np.float32)

    def read_joint_pos(self):
        return self._jp.copy()

    def get_front_image(self):
        return (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    def get_wrist_image(self):
        return (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    def send_joint_targets(self, arm_target, gripper_target):
        self._jp[:N_ARM] = arm_target
        self._jp[6] = gripper_target
        self._jp[7] = gripper_target

    def go_to_rest(self):
        self._jp = np.concatenate([REST_QPOS, [0.02, 0.02]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Deploy loop
# ════════════════════════════════════════════════════════════════════════════════
def run(policy: VisualPolicy, robot: VisualRobotInterface, hz=CONTROL_HZ, max_steps=0, verbose=True):
    ctrl = ActionController()
    robot.go_to_rest()
    jp = robot.read_joint_pos()
    ctrl.reset(jp[:N_ARM], jp[6])
    dt, step = 1.0 / hz, 0
    try:
        while max_steps == 0 or step < max_steps:
            t0 = time.perf_counter()
            jp = robot.read_joint_pos()
            front = robot.get_front_image()
            wrist = robot.get_wrist_image()
            state = build_state(jp, ctrl.target_qpos7)
            action = policy.act(front, wrist, state)          # (7,) in [-1,1]
            arm_t, grip_t = ctrl.step(action, jp[PAN_IDX])
            robot.send_joint_targets(arm_t, grip_t)
            if verbose and step % int(hz) == 0:
                print(f"[t={step}] grip_t={grip_t:.3f} arm_t={np.round(arm_t,2)}", flush=True)
            step += 1
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[deploy-v] stopped by user.")
    finally:
        robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dagger_final.pt (visual student)")
    ap.add_argument("--hz", type=float, default=CONTROL_HZ)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto -> cuda|mps|cpu (Mac: mps or cpu, both fine)")
    ap.add_argument("--mock", action="store_true", help="run with MockVisualRobot (no hardware)")
    args = ap.parse_args()
    policy = VisualPolicy(args.checkpoint, device=args.device)
    if args.mock:
        print("[deploy-v] MOCK mode — implement VisualRobotInterface for real hardware.")
        run(policy, MockVisualRobot(), hz=args.hz, max_steps=args.max_steps)
    else:
        raise SystemExit("Implement a VisualRobotInterface for the reBot and pass it to run(); "
                         "or use --mock to test the pipeline.")


if __name__ == "__main__":
    main()

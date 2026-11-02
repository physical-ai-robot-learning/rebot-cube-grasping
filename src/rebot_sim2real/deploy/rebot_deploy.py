"""reBot LiftCube — portable real-world inference (Isaac-free).

Runs a policy trained by train_c51.py (distributional-C51 SAC) on the REAL reBot B601-DM.
Only needs `torch` + `numpy` (no Isaac Sim / Isaac Lab). Copy this folder to the robot PC.

WHAT YOU MUST IMPLEMENT (subclass RobotInterface for your hardware):
  - read_joint_state()  -> (joint_pos[8], joint_vel[8])  in the EXACT JOINT_ORDER below
  - get_cube_pose()     -> (pos[3], quat_wxyz[4])         target cube, in the SIM WORLD frame
  - get_gripper_link_pose() -> (pos[3], quat_wxyz[4])     gripper_link, SIM WORLD frame (for TCP via FK)
  - is_grasped()        -> bool                            real grasp heuristic (current/qpos based)
  - send_joint_targets(arm_target[6], gripper_target)      command motors (arm rad, gripper meters)
  - go_to_rest()                                           move to REST_POSE at episode start

CRITICAL sim<->real PARITY (or the policy will fail):
  1. JOINT_ORDER must match the sim articulation order (verify with dump_sim_spec.py).
  2. Units: arm = radians, gripper = METERS (prismatic, stroke 0..0.045 capped). CONVENTION (measured,
     diag_gripper_stroke): gripper qpos UP=OPEN (jaws apart), DOWN=CLOSED; q~0.03 grips the 4cm cube.
     [the old "qpos>0.05=closed" was BACKWARDS]
  3. Frame: obj_pos / gripper_link pos are ABSOLUTE in the SIM WORLD frame (robot base spawned at
     (0.35,-0.64,0.01), z-up). Your perception+FK must report positions in that same frame.
  4. Control rate = 10 Hz (CONTROL_HZ). Actions are DELTA-on-target (see ActionController).
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (must match src/rebot_sim2real/env/env_config.py + maniskill_rewards.py)
# ════════════════════════════════════════════════════════════════════════════════
# Articulation joint order (verify with dump_sim_spec.py on the training PC!).
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw",
               "wrist_roll", "gripper", "gripper_joint2"]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
N_JOINTS = len(JOINT_ORDER)          # 8
N_ARM = len(ARM_JOINTS)              # 6
N_ACT = 7                            # 6 arm + 1 gripper
OBS_DIM = 43

REST_QPOS = np.array([0.0, -1.15, -1.00, 0.55, 0.0, 0.0], dtype=np.float32)   # arm rest (rad)
ARM_SCALE = 0.05                     # delta rad / step  (HALVED for calmer motion; sim scale=0.05)
GRIP_SCALE = 0.0143                  # delta m / step    (MirroredTargetDeltaGripperActionCfg.scale)
GRIP_LO, GRIP_HI = 0.0, 0.045        # gripper stroke (m); HI capped to the trained safe range (no over-open/breakage)
# HARD per-joint target limits (order = arm joints in JOINT_ORDER) -- real-robot safety + sim parity:
#  pan +-90deg | lift,elbow USD | wrist_flex UP -10deg / DOWN +90 | wrist_yaw +-105 | wrist_roll +-180
ARM_LO = np.array([-1.5708, -3.14, -3.14, -0.1745, -1.8326, -3.1416], dtype=np.float32)
ARM_HI = np.array([ 1.5708,  0.00,  0.00,  1.57,    1.8326,  3.1416], dtype=np.float32)
PAN_IDX, ELBOW_IDX = 0, 2            # local indices within the 6 arm joints
PAN_FORWARD = 1.54                   # elbow free only when measured pan >= this (~88deg, just under +90)
ELBOW_TUCK_LO, ELBOW_TUCK_HI = -1.4, -0.7   # else hold elbow in this compact tuck (no straighten / fold-back)
# Real-motor velocity limits to match sim: shoulder block 2.5 rad/s, wrist block 1.5 rad/s, gripper 0.2 m/s.
TCP_OFFSET = np.array([-0.086, 0.0, 0.103], dtype=np.float32)  # gripper_link-local TCP (ee_frame offset)
CONTROL_HZ = 10.0
EPISODE_STEPS = 50                   # 5.0 s @ 10 Hz (informational; deploy can run open-ended)

# obs term layout (name, dim) IN ORDER — exactly the sim ObservationManager 'policy' group.
OBS_LAYOUT = [("qpos", 8), ("qvel", 8), ("target_qpos", 7), ("dist_to_rest", 6),
              ("obj_pos", 3), ("obj_quat", 4), ("tcp_pos", 3), ("tcp_to_obj", 3), ("is_grasped", 1)]


# ════════════════════════════════════════════════════════════════════════════════
#  Policy network  (must match train_c51.py Actor exactly so the checkpoint loads)
# ════════════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state=OBS_DIM, n_act=N_ACT, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)   # unused at inference; present so state_dict loads

    @torch.no_grad()
    def act(self, obs_np):
        """Deterministic action in [-1,1]^7 from a (43,) obs vector."""
        x = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        x = self.fc(self.proj(x))
        return torch.tanh(self.mean(x)).squeeze(0).numpy()


def load_policy(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor = Actor()
    state = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt
    actor.load_state_dict(state)
    actor.eval()
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"[deploy] loaded actor from {checkpoint_path} (train step {step})")
    return actor


# ════════════════════════════════════════════════════════════════════════════════
#  Helpers: quaternion rotate (wxyz), FK for TCP
# ════════════════════════════════════════════════════════════════════════════════
def quat_rotate(q_wxyz, v):
    """Rotate vector v by quaternion q (w,x,y,z). Returns (3,)."""
    w, x, y, z = q_wxyz
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return R @ np.asarray(v, dtype=np.float32)


def tcp_from_gripper_link(gl_pos, gl_quat_wxyz):
    """TCP world pos = gripper_link pos + R(gripper_link quat) @ TCP_OFFSET. (Sim ee_frame, both
    target frames share the same offset, so their midpoint == this point.)"""
    return np.asarray(gl_pos, dtype=np.float32) + quat_rotate(gl_quat_wxyz, TCP_OFFSET)


# ════════════════════════════════════════════════════════════════════════════════
#  Observation builder  (EXACT sim parity)
# ════════════════════════════════════════════════════════════════════════════════
def build_obs(joint_pos, joint_vel, target_qpos, cube_pos, cube_quat_wxyz, tcp_pos, is_grasped):
    """Assemble the 43-dim policy obs IN SIM ORDER.
      joint_pos/joint_vel : (8,) in JOINT_ORDER
      target_qpos         : (7,) = [arm6 commanded target, gripper commanded target] (ActionController.target)
      cube_pos/cube_quat  : (3,)/(4, wxyz)  target cube, SIM WORLD frame
      tcp_pos             : (3,)  SIM WORLD frame (use tcp_from_gripper_link)
      is_grasped          : bool/0-1
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    target_qpos = np.asarray(target_qpos, dtype=np.float32)
    cube_pos = np.asarray(cube_pos, dtype=np.float32)
    cube_quat_wxyz = np.asarray(cube_quat_wxyz, dtype=np.float32)
    tcp_pos = np.asarray(tcp_pos, dtype=np.float32)
    arm_qpos = joint_pos[:N_ARM]                     # first 6 = arm (JOINT_ORDER)
    dist_to_rest = arm_qpos - REST_QPOS              # 6-dim VECTOR (sim _dist_to_rest_qpos)
    tcp_to_obj = cube_pos - tcp_pos
    obs = np.concatenate([
        joint_pos,                                   # 8
        joint_vel,                                   # 8
        target_qpos,                                 # 7
        dist_to_rest,                                # 6
        cube_pos,                                    # 3
        cube_quat_wxyz,                              # 4
        tcp_pos,                                     # 3
        tcp_to_obj,                                  # 3
        np.array([1.0 if is_grasped else 0.0], dtype=np.float32),  # 1
    ]).astype(np.float32)
    assert obs.shape[0] == OBS_DIM, f"obs dim {obs.shape[0]} != {OBS_DIM}"
    return obs


# ════════════════════════════════════════════════════════════════════════════════
#  Action controller  (DELTA-on-target, matches the sim action terms exactly)
# ════════════════════════════════════════════════════════════════════════════════
class ActionController:
    """Mirrors TargetDeltaJointPositionAction (arm) + MirroredTargetDeltaGripperAction (gripper):
        target += clip(action, -1, 1) * scale  ; gripper clamped to [0, 0.045].
    Initialize target to the current measured joint positions at reset (like the sim reset())."""

    def __init__(self):
        self.arm_target = REST_QPOS.copy()           # (6,) rad
        self.grip_target = 0.02                       # (1,) m  (REBOT_REST_POSE gripper = 0.02)

    def reset(self, arm_qpos, grip_qpos):
        self.arm_target = np.asarray(arm_qpos, dtype=np.float32).copy()
        self.grip_target = float(grip_qpos)

    def step(self, action, pan_qpos):
        """action: (7,) in [-1,1]; pan_qpos: measured shoulder_pan (rad). Returns (arm_target[6], grip_target).
        Replicates the sim action term EXACTLY: delta-on-target + hard per-joint clamps + coupled elbow gate."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.arm_target = np.clip(self.arm_target + a[:N_ARM] * ARM_SCALE, ARM_LO, ARM_HI)
        # coupled elbow gate: until the pan has swung forward (>= PAN_FORWARD) hold the elbow folded in the
        # tuck range so it cannot straighten out / extend backward; once pan is forward the elbow is free.
        if float(pan_qpos) < PAN_FORWARD:
            self.arm_target[ELBOW_IDX] = float(np.clip(self.arm_target[ELBOW_IDX], ELBOW_TUCK_LO, ELBOW_TUCK_HI))
        self.grip_target = float(np.clip(self.grip_target + a[6] * GRIP_SCALE, GRIP_LO, GRIP_HI))
        return self.arm_target.copy(), self.grip_target

    @property
    def target_qpos7(self):
        """The 7-dim commanded target for the obs (arm6 + gripper1)."""
        return np.concatenate([self.arm_target, [self.grip_target]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Hardware abstraction  (implement these for the real reBot)
# ════════════════════════════════════════════════════════════════════════════════
class RobotInterface:
    def read_joint_state(self):
        """-> (joint_pos[8], joint_vel[8]) in JOINT_ORDER. gripper_joint2 mirrors gripper."""
        raise NotImplementedError

    def get_cube_pose(self):
        """-> (pos[3], quat_wxyz[4]) of the TARGET cube, in the SIM WORLD frame (perception)."""
        raise NotImplementedError

    def get_gripper_link_pose(self):
        """-> (pos[3], quat_wxyz[4]) of gripper_link, SIM WORLD frame (your robot FK)."""
        raise NotImplementedError

    def is_grasped(self):
        """-> bool. Real grasp heuristic (e.g. gripper qpos>0.05 AND motor current high)."""
        raise NotImplementedError

    def send_joint_targets(self, arm_target, gripper_target):
        """Command the motors. arm_target[6] rad, gripper_target m (drive BOTH gripper fingers to it)."""
        raise NotImplementedError

    def go_to_rest(self):
        """Move smoothly to REST_QPOS (arm) + gripper 0.02 m before an episode."""
        raise NotImplementedError

    def close(self):
        pass


class MockRobot(RobotInterface):
    """No-hardware stand-in so you can smoke-test the pipeline (obs->policy->action) on any PC."""
    def __init__(self):
        self._jp = np.concatenate([REST_QPOS, [0.02, 0.02]]).astype(np.float32)
        self._t = 0

    def read_joint_state(self):
        return self._jp.copy(), np.zeros(N_JOINTS, dtype=np.float32)

    def get_cube_pose(self):
        return np.array([0.30, -0.30, 0.06], dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)

    def get_gripper_link_pose(self):
        return np.array([0.25, -0.35, 0.25], dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)

    def is_grasped(self):
        return False

    def send_joint_targets(self, arm_target, gripper_target):
        # pretend the motors instantly reach target (for shape/flow testing only)
        self._jp[:N_ARM] = arm_target
        self._jp[6] = gripper_target
        self._jp[7] = gripper_target
        self._t += 1

    def go_to_rest(self):
        self._jp = np.concatenate([REST_QPOS, [0.02, 0.02]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Deploy loop
# ════════════════════════════════════════════════════════════════════════════════
def run(actor, robot: RobotInterface, hz=CONTROL_HZ, max_steps=0, verbose=True):
    ctrl = ActionController()
    robot.go_to_rest()
    jp, _ = robot.read_joint_state()
    ctrl.reset(jp[:N_ARM], jp[6])
    dt = 1.0 / hz
    step = 0
    try:
        while max_steps == 0 or step < max_steps:
            t0 = time.perf_counter()
            jp, jv = robot.read_joint_state()
            cube_pos, cube_quat = robot.get_cube_pose()
            gl_pos, gl_quat = robot.get_gripper_link_pose()
            tcp = tcp_from_gripper_link(gl_pos, gl_quat)
            grasped = robot.is_grasped()

            obs = build_obs(jp, jv, ctrl.target_qpos7, cube_pos, cube_quat, tcp, grasped)
            action = actor.act(obs)                      # (7,) in [-1,1]
            arm_t, grip_t = ctrl.step(action, jp[PAN_IDX])   # pan-gated elbow needs measured pan
            robot.send_joint_targets(arm_t, grip_t)

            if verbose and step % int(hz) == 0:
                print(f"[t={step}] grip_t={grip_t:.3f} grasped={grasped} "
                      f"cube={np.round(cube_pos,3)} tcp={np.round(tcp,3)}", flush=True)
            step += 1
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[deploy] stopped by user.")
    finally:
        robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="c51_ckpt_*.pt / c51_final.pt from training")
    ap.add_argument("--hz", type=float, default=CONTROL_HZ)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock", action="store_true", help="run with MockRobot (no hardware, flow test)")
    args = ap.parse_args()

    actor = load_policy(args.checkpoint, device=args.device)
    if args.mock:
        robot = MockRobot()
        print("[deploy] MOCK mode — implement RobotInterface for real hardware.")
    else:
        raise SystemExit("Implement a RobotInterface for the reBot and pass it to run(); "
                         "or use --mock to test the pipeline. See README.md.")
    run(actor, robot, hz=args.hz, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

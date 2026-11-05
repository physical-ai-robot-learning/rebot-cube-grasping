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
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (must match src/rebot_teacher_student/env/env_config.py + maniskill_rewards.py)
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
GRIP_REST = 0.02                     # gripper target at REST (m), REBOT_REST_POSE
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
    def act(self, obsNp):
        """Deterministic action in [-1,1]^7 from a (43,) obs vector."""
        tX = torch.as_tensor(obsNp, dtype=torch.float32).unsqueeze(0)
        tX = self.fc(self.proj(tX))
        return torch.tanh(self.mean(tX)).squeeze(0).numpy()


def loadPolicy(checkpointPath, device="cpu"):
    tCkpt = torch.load(checkpointPath, map_location=device)
    tActor = Actor()
    tState = tCkpt["actor"] if isinstance(tCkpt, dict) and "actor" in tCkpt else tCkpt
    tActor.load_state_dict(tState)
    tActor.eval()
    tStep = tCkpt.get("step", "?") if isinstance(tCkpt, dict) else "?"
    print(f"[deploy] loaded actor from {checkpointPath} (train step {tStep})")
    return tActor


# ════════════════════════════════════════════════════════════════════════════════
#  Helpers: quaternion rotate (wxyz), FK for TCP
# ════════════════════════════════════════════════════════════════════════════════
def quatRotate(qWxyz, v):
    """Rotate vector v by quaternion q (w,x,y,z). Returns (3,)."""
    tW, tX, tY, tZ = qWxyz
    # rotation matrix from quaternion
    tR = np.array([
        [1 - 2 * (tY * tY + tZ * tZ), 2 * (tX * tY - tZ * tW),     2 * (tX * tZ + tY * tW)],
        [2 * (tX * tY + tZ * tW),     1 - 2 * (tX * tX + tZ * tZ), 2 * (tY * tZ - tX * tW)],
        [2 * (tX * tZ - tY * tW),     2 * (tY * tZ + tX * tW),     1 - 2 * (tX * tX + tY * tY)],
    ], dtype=np.float32)
    return tR @ np.asarray(v, dtype=np.float32)


def tcpFromGripperLink(glPos, glQuatWxyz):
    """TCP world pos = gripper_link pos + R(gripper_link quat) @ TCP_OFFSET. (Sim ee_frame, both
    target frames share the same offset, so their midpoint == this point.)"""
    return np.asarray(glPos, dtype=np.float32) + quatRotate(glQuatWxyz, TCP_OFFSET)


# ════════════════════════════════════════════════════════════════════════════════
#  Observation builder  (EXACT sim parity)
# ════════════════════════════════════════════════════════════════════════════════
def buildObs(jointPos, jointVel, targetQpos, cubePos, cubeQuatWxyz, tcpPos, isGrasped):
    """Assemble the 43-dim policy obs IN SIM ORDER.
      joint_pos/joint_vel : (8,) in JOINT_ORDER
      target_qpos         : (7,) = [arm6 commanded target, gripper commanded target] (ActionController.target)
      cube_pos/cube_quat  : (3,)/(4, wxyz)  target cube, SIM WORLD frame
      tcp_pos             : (3,)  SIM WORLD frame (use tcp_from_gripper_link)
      is_grasped          : bool/0-1
    """
    jointPos = np.asarray(jointPos, dtype=np.float32)
    jointVel = np.asarray(jointVel, dtype=np.float32)
    targetQpos = np.asarray(targetQpos, dtype=np.float32)
    cubePos = np.asarray(cubePos, dtype=np.float32)
    cubeQuatWxyz = np.asarray(cubeQuatWxyz, dtype=np.float32)
    tcpPos = np.asarray(tcpPos, dtype=np.float32)
    tArmQpos = jointPos[:N_ARM]                     # first 6 = arm (JOINT_ORDER)
    tDistToRest = tArmQpos - REST_QPOS              # 6-dim VECTOR (sim _dist_to_rest_qpos)
    tTcpToObj = cubePos - tcpPos
    tObs = np.concatenate([
        jointPos,                                   # 8
        jointVel,                                   # 8
        targetQpos,                                 # 7
        tDistToRest,                                # 6
        cubePos,                                    # 3
        cubeQuatWxyz,                              # 4
        tcpPos,                                     # 3
        tTcpToObj,                                  # 3
        np.array([1.0 if isGrasped else 0.0], dtype=np.float32),  # 1
    ]).astype(np.float32)
    assert tObs.shape[0] == OBS_DIM, f"obs dim {tObs.shape[0]} != {OBS_DIM}"
    return tObs


# ════════════════════════════════════════════════════════════════════════════════
#  Action controller  (DELTA-on-target, matches the sim action terms exactly)
# ════════════════════════════════════════════════════════════════════════════════
class ActionController:
    """Mirrors TargetDeltaJointPositionAction (arm) + MirroredTargetDeltaGripperAction (gripper):
        target += clip(action, -1, 1) * scale  ; gripper clamped to [0, 0.045].
    Initialize target to the current measured joint positions at reset (like the sim reset())."""

    def __init__(self):
        self.mArmTarget = REST_QPOS.copy()           # (6,) rad
        self.mGripTarget = GRIP_REST                  # (1,) m  (REBOT_REST_POSE gripper)

    def reset(self, armQpos, gripQpos):
        self.mArmTarget = np.asarray(armQpos, dtype=np.float32).copy()
        self.mGripTarget = float(gripQpos)

    def step(self, action, panQpos):
        """action: (7,) in [-1,1]; pan_qpos: measured shoulder_pan (rad). Returns (arm_target[6], grip_target).
        Replicates the sim action term EXACTLY: delta-on-target + hard per-joint clamps + coupled elbow gate."""
        tA = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.mArmTarget = np.clip(self.mArmTarget + tA[:N_ARM] * ARM_SCALE, ARM_LO, ARM_HI)
        # coupled elbow gate: until the pan has swung forward (>= PAN_FORWARD) hold the elbow folded in the
        # tuck range so it cannot straighten out / extend backward; once pan is forward the elbow is free.
        if float(panQpos) < PAN_FORWARD:
            self.mArmTarget[ELBOW_IDX] = float(np.clip(self.mArmTarget[ELBOW_IDX], ELBOW_TUCK_LO, ELBOW_TUCK_HI))
        self.mGripTarget = float(np.clip(self.mGripTarget + tA[6] * GRIP_SCALE, GRIP_LO, GRIP_HI))
        return self.mArmTarget.copy(), self.mGripTarget

    @property
    def targetQpos7(self):
        """The 7-dim commanded target for the obs (arm6 + gripper1)."""
        return np.concatenate([self.mArmTarget, [self.mGripTarget]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Hardware abstraction  (implement these for the real reBot)
# ════════════════════════════════════════════════════════════════════════════════
class RobotInterface:
    def readJointState(self):
        """-> (joint_pos[8], joint_vel[8]) in JOINT_ORDER. gripper_joint2 mirrors gripper."""
        raise NotImplementedError

    def getCubePose(self):
        """-> (pos[3], quat_wxyz[4]) of the TARGET cube, in the SIM WORLD frame (perception)."""
        raise NotImplementedError

    def getGripperLinkPose(self):
        """-> (pos[3], quat_wxyz[4]) of gripper_link, SIM WORLD frame (your robot FK)."""
        raise NotImplementedError

    def isGrasped(self):
        """-> bool. Real grasp heuristic (e.g. gripper qpos>0.05 AND motor current high)."""
        raise NotImplementedError

    def sendJointTargets(self, armTarget, gripperTarget):
        """Command the motors. arm_target[6] rad, gripper_target m (drive BOTH gripper fingers to it)."""
        raise NotImplementedError

    def goToRest(self):
        """Move smoothly to REST_QPOS (arm) + gripper 0.02 m before an episode."""
        raise NotImplementedError

    def close(self):
        pass


class MockRobot(RobotInterface):
    """No-hardware stand-in so you can smoke-test the pipeline (obs->policy->action) on any PC."""
    def __init__(self):
        self.mJointPos = np.concatenate([REST_QPOS, [GRIP_REST, GRIP_REST]]).astype(np.float32)
        self.mTick = 0

    def readJointState(self):
        return self.mJointPos.copy(), np.zeros(N_JOINTS, dtype=np.float32)

    def getCubePose(self):
        return np.array([0.30, -0.30, 0.06], dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)

    def getGripperLinkPose(self):
        return np.array([0.25, -0.35, 0.25], dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)

    def isGrasped(self):
        return False

    def sendJointTargets(self, armTarget, gripperTarget):
        # pretend the motors instantly reach target (for shape/flow testing only)
        self.mJointPos[:N_ARM] = armTarget
        self.mJointPos[6] = gripperTarget
        self.mJointPos[7] = gripperTarget
        self.mTick += 1

    def goToRest(self):
        self.mJointPos = np.concatenate([REST_QPOS, [GRIP_REST, GRIP_REST]]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Deploy loop
# ════════════════════════════════════════════════════════════════════════════════
def run(actor, robot: RobotInterface, hz=CONTROL_HZ, maxSteps=0, verbose=True):
    tCtrl = ActionController()
    robot.goToRest()
    tJp, _ = robot.readJointState()
    tCtrl.reset(tJp[:N_ARM], tJp[6])
    tDt = 1.0 / hz
    tStep = 0
    try:
        while maxSteps == 0 or tStep < maxSteps:
            tT0 = time.perf_counter()
            tJp, tJv = robot.readJointState()
            tCubePos, tCubeQuat = robot.getCubePose()
            tGlPos, tGlQuat = robot.getGripperLinkPose()
            tTcp = tcpFromGripperLink(tGlPos, tGlQuat)
            tGrasped = robot.isGrasped()

            tObs = buildObs(tJp, tJv, tCtrl.targetQpos7, tCubePos, tCubeQuat, tTcp, tGrasped)
            tAction = actor.act(tObs)                      # (7,) in [-1,1]
            tArmT, tGripT = tCtrl.step(tAction, tJp[PAN_IDX])   # pan-gated elbow needs measured pan
            robot.sendJointTargets(tArmT, tGripT)

            if verbose and tStep % int(hz) == 0:
                print(f"[t={tStep}] grip_t={tGripT:.3f} grasped={tGrasped} "
                      f"cube={np.round(tCubePos,3)} tcp={np.round(tTcp,3)}", flush=True)
            tStep += 1
            tSleep = tDt - (time.perf_counter() - tT0)
            if tSleep > 0:
                time.sleep(tSleep)
    except KeyboardInterrupt:
        print("\n[deploy] stopped by user.")
    finally:
        robot.close()


class DeployError(RuntimeError):
    """The deploy loop was asked to run without a robot implementation."""


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="Isaac-free deploy loop of the state-only C51 teacher.")
    tAp.add_argument("--checkpoint", required=True, help="c51_ckpt_*.pt / c51_final.pt from training")
    tAp.add_argument("--hz", type=float, default=CONTROL_HZ)
    tAp.add_argument("--max-steps", type=int, default=0, help="0 = run until Ctrl-C")
    tAp.add_argument("--device", default="cpu")
    tAp.add_argument("--mock", action="store_true", help="run with MockRobot (no hardware, flow test)")
    tArgs = tAp.parse_args(argv)

    tActor = loadPolicy(tArgs.checkpoint, device=tArgs.device)
    if tArgs.mock:
        tRobot = MockRobot()
        print("[deploy] MOCK mode — implement RobotInterface for real hardware.")
    else:
        print(DeployError("Implement a RobotInterface for the reBot and pass it to run(); "
                          "or use --mock to test the pipeline. See README.md."), file=sys.stderr)
        return 1
    run(tActor, tRobot, hz=tArgs.hz, maxSteps=tArgs.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

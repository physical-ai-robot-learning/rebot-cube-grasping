"""Real-robot I/O for the reBot B601-DM, backed by LeRobot (lerobot-robot-seeed-b601 + motorbridge)
for motor I/O and Pinocchio for FK. Runs in the deploy venv (.venv-deploy), NOT the Isaac venv.

Frame/convention bridge (sim policy <-> hardware), all VERIFY-on-arm flagged CALIB:
  * LeRobot reports/accepts joint angles in DEGREES (semantic names); the sim policy uses RADIANS
    (joint1..6). Map: sim_rad[j] = SIGN[j]*deg2rad(lerobot_deg[j]) + OFFSET_rad[j].
  * Joint name map: shoulder_pan->joint1 ... wrist_roll->joint6, gripper->gripper.
  * FK (Pinocchio + the repo URDF) gives the TCP in the base_link frame; the sim policy frame places
    the base at cfg.base_pos with cfg.base_quat (+90deg yaw), so tcp_env = R_base @ tcp_base + base_pos,
    then -= tcp_tip_drop in z (matches sim). Perception's cube_pos must be in this SAME env frame.
  * Observation has only .pos (no velocity) -> qvel by finite difference at the 15 Hz step.
  * Hardware joint limits are tighter on some axes (J6 +/-90 vs sim +/-180) -> clamp on command.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np

from rebot_sim2real.grasp_traj import APPROACH, ZERO, LIFTUP_STEPS, COMEBACK_STEPS, scripted_joint_path

# LeRobot semantic joint names in sim joint1..6 order, + gripper
LEROBOT_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
# CALIB (verify on arm): per-joint sim_rad = SIGN*deg2rad(deg) + OFFSET. Default assumes LeRobot's
# calibrated zero == the URDF zero and same sign (reBot ships the same URDF for Isaac/Pinocchio).
# Verify with verify_home(): at the URDF home (sim q=0) LeRobot should read ~0 on every joint.
SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
OFFSET_RAD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# hardware joint limits (deg) from lerobot_robot_seeed_b601 default config -> clamp commands
HW_LIMIT_DEG = {"shoulder_pan": (-145, 145), "shoulder_lift": (-170, 0), "elbow_flex": (-200, 0),
                "wrist_flex": (-80, 90), "wrist_yaw": (-90, 90), "wrist_roll": (-90, 90)}
# CRITICAL: LeRobot send_action() multiplies the input by joint_directions, but get_observation()
# returns RAW motor degrees (no direction). So to command the motor to a raw angle we must DIVIDE by
# the direction (counteract the internal *direction). (gripper has a -6 gear/scale factor.)
JOINT_DIR = {"shoulder_pan": -1.0, "shoulder_lift": -1.0, "elbow_flex": 1.0, "wrist_flex": 1.0,
             "wrist_yaw": 1.0, "wrist_roll": -1.0, "gripper": -6.0}
# is_grasped via gripper motor TORQUE (diag_grasp.py: air~0.07, cube~1.86) — the gap transiently
# passes the cube-width during closing and false-fires, but torque only rises when actually gripping.
GRIP_TORQUE_THRESH = 0.5
# add to fk_tcp z -> the policy drives the gripper that much lower. 0.12 was tuned for the OLD (z-biased)
# calib; with the FRESH calib_extrinsic, 0.12 drove the gripper to real fk_mid ~-0.06 (BELOW the table)
# -> it dug in and missed the cube -> reduced to 0.03. Tune by re-test (raise if it hovers high, lower
# if it digs into the table).
FK_Z_CORR = 0.09
# Real grasp lands CAMERA-SIDE (+y, "front") of the reported midpoint -> the gripper "closed IN FRONT
# of the cube". Shift the reported tcp toward the camera (= AGAINST the FK forward dir gripper_link->
# fingers, which points toward base) by JAW_FWD so the policy pulls the real jaw back onto the cube.
# Sign: subtract along fwd (=move +y at rest). 0.04 still closed "slightly in front"(slightly front) of the cube +
# shoved it -y during commit -> 0.06. Tune by re-test (raise if still front, lower if it closes behind).
JAW_FWD = 0.06


@dataclass
class Proprio:
    arm_qpos: np.ndarray   # [6] rad, sim convention
    arm_qvel: np.ndarray   # [6] rad/s (finite diff)
    grip_qpos: np.ndarray  # [2] m (mirrored fingers, sim convention)
    grip_qvel: np.ndarray  # [2]
    arm_torque: np.ndarray = None   # [6] N*m, sim convention (SIGN*motor_torque); for obs_torque policies
    grip_torque: float = 0.0        # gripper motor torque magnitude (same signal as is_grasped)


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


class RobotIO:
    n_arm = 6

    def read_proprio(self) -> Proprio: raise NotImplementedError
    def fk_tcp(self, arm_qpos: np.ndarray) -> np.ndarray: raise NotImplementedError
    def is_grasped(self, grip_qpos: np.ndarray) -> bool: raise NotImplementedError
    def command(self, arm_target: np.ndarray, grip_target: float) -> None: raise NotImplementedError
    def move_to(self, arm_qpos, grip_qpos, duration_s=2.0): raise NotImplementedError
    def initial_liftup(self, rest_arm, grip_open_qpos, steps=LIFTUP_STEPS): raise NotImplementedError
    def comeback_to_zero(self, grip_open_qpos, via=None, steps=COMEBACK_STEPS): raise NotImplementedError
    def estop(self): raise NotImplementedError


class DamiaoRobotIO(RobotIO):
    def __init__(self, cfg, urdf_path, port="/dev/ttyACM0", robot_id="follower1", dt=1.0 / 15.0,
                 grip_closed_deg=0.0, grip_open_deg=-90.0):
        """cfg = EnvConfig (base_pos/base_quat/tcp_tip_drop/grip_*). urdf_path = the with-gripper URDF.
        grip_closed_deg/grip_open_deg: CALIB endpoints mapping the gripper motor (deg) <-> sim qpos (m)."""
        from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
        import pinocchio as pin
        self.pin = pin
        self.cfg = cfg; self.dt = dt
        self.robot = SeeedB601DMFollower(SeeedB601DMFollowerConfig(port=port, id=robot_id, can_adapter="damiao"))
        self.model = pin.buildModelFromUrdf(urdf_path); self.fkdata = self.model.createData()
        self.fid_l = self.model.getFrameId("gripper_left"); self.fid_r = self.model.getFrameId("gripper_right")
        self.fid_glink = self.model.getFrameId("gripper_link")   # "behind" ref for the jaw forward direction
        self.R_base = _quat_wxyz_to_R(cfg.base_quat_wxyz); self.base_pos = np.asarray(cfg.base_pos, float)
        self.grip_closed_deg, self.grip_open_deg = grip_closed_deg, grip_open_deg
        # joint convention map from deploy/calib_convention.py (rest-anchored). Falls back to module
        # SIGN/OFFSET_RAD (identity) with a warning if absent.
        self.SIGN, self.OFFSET = SIGN.copy(), OFFSET_RAD.copy()
        mp = os.path.join(os.path.dirname(__file__), "joint_map.npz")
        if os.path.exists(mp):
            jm = np.load(mp, allow_pickle=True)
            self.SIGN = jm["sign"].astype(float)
            self.OFFSET = np.asarray(jm["rest_arm_rad"], float) - self.SIGN * np.deg2rad(jm["L_rest_deg"])
            self.grip_closed_deg, self.grip_open_deg = float(jm["grip_closed_deg"]), float(jm["grip_open_deg"])
            print(f"[DamiaoRobotIO] loaded joint_map.npz (rest-anchored); SIGN={self.SIGN}")
        else:
            print("[DamiaoRobotIO] WARNING no deploy/joint_map.npz -> identity convention (run calib_convention.py!)")
        self._prev_arm = None; self._prev_grip = None; self._last_grip = np.array([cfg.grip_open_qpos] * 2)
        self._last_grip_torque = 0.0

    # ---- gripper sim<->lerobot (linear between calibrated closed/open endpoints) ----
    def _grip_deg_to_qpos(self, deg):
        f = (deg - self.grip_closed_deg) / (self.grip_open_deg - self.grip_closed_deg + 1e-9)
        q = self.cfg.grip_close_qpos + f * (self.cfg.grip_open_qpos - self.cfg.grip_close_qpos)
        return float(np.clip(q, self.cfg.grip_close_qpos, self.cfg.grip_open_qpos))

    def _grip_qpos_to_deg(self, q):
        f = (q - self.cfg.grip_close_qpos) / (self.cfg.grip_open_qpos - self.cfg.grip_close_qpos + 1e-9)
        return self.grip_closed_deg + f * (self.grip_open_deg - self.grip_closed_deg)

    def connect(self): self.robot.connect(calibrate=False)   # assume already lerobot-calibrated

    def read_proprio(self) -> Proprio:
        obs = self.robot.get_observation()
        deg = np.array([obs[f"{lj}.pos"] for lj in LEROBOT_ARM], float)
        arm_qpos = self.SIGN * np.deg2rad(deg) + self.OFFSET
        gq = self._grip_deg_to_qpos(obs["gripper.pos"]); grip_qpos = np.array([gq, gq])
        self._last_grip_torque = abs(float(obs.get("gripper.torque", 0.0)))
        # joint torque (sim convention = SIGN*motor_torque, same map as qpos; OFFSET is position-only)
        arm_torque = self.SIGN * np.array([float(obs.get(f"{lj}.torque", 0.0)) for lj in LEROBOT_ARM], float)
        if self._prev_arm is None:
            arm_qvel = np.zeros(6); grip_qvel = np.zeros(2)
        else:
            arm_qvel = (arm_qpos - self._prev_arm) / self.dt
            grip_qvel = (grip_qpos - self._prev_grip) / self.dt
        self._prev_arm, self._prev_grip, self._last_grip = arm_qpos, grip_qpos, grip_qpos
        return Proprio(arm_qpos, arm_qvel, grip_qpos, grip_qvel,
                       arm_torque=arm_torque, grip_torque=self._last_grip_torque)

    def fk_tcp(self, arm_qpos: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq); q[0:6] = arm_qpos; q[6:8] = self._last_grip   # midpoint ~grip-invariant
        self.pin.forwardKinematics(self.model, self.fkdata, q)
        self.pin.updateFramePlacements(self.model, self.fkdata)
        fl = np.asarray(self.fkdata.oMf[self.fid_l].translation)
        fr = np.asarray(self.fkdata.oMf[self.fid_r].translation)
        glink = np.asarray(self.fkdata.oMf[self.fid_glink].translation)
        mid = self.R_base @ (0.5 * (fl + fr)) + self.base_pos   # finger-origin midpoint, env frame
        gl = self.R_base @ glink + self.base_pos                # gripper_link origin, env frame
        fwd = (mid - gl)[:2]; n = float(np.linalg.norm(fwd))    # horizontal forward (gripper_link -> fingers, ~-y)
        if n > 1e-6:
            mid[:2] -= JAW_FWD * fwd / n                        # shift toward camera/+y: pull real jaw onto cube
        mid[2] -= self.cfg.tcp_tip_drop
        mid[2] += FK_Z_CORR
        return mid

    def is_grasped(self, grip_qpos: np.ndarray) -> bool:
        return self._last_grip_torque > GRIP_TORQUE_THRESH   # torque, not gap (gap false-fires mid-close)

    def command(self, arm_target: np.ndarray, grip_target: float) -> None:
        deg = np.rad2deg((np.asarray(arm_target) - self.OFFSET) / self.SIGN)   # raw motor-deg targets
        action = {}
        for i, lj in enumerate(LEROBOT_ARM):
            lo, hi = HW_LIMIT_DEG[lj]
            motor_deg = float(np.clip(deg[i], lo, hi))
            action[f"{lj}.pos"] = motor_deg / JOINT_DIR[lj]    # /direction: counteract send_action's *direction
        action["gripper.pos"] = float(self._grip_qpos_to_deg(grip_target)) / JOINT_DIR["gripper"]
        self.robot.send_action(action)

    def move_to(self, arm_qpos, grip_qpos, duration_s=2.0):
        """Ramp from the current pose to the target (open-loop pre-position)."""
        import time
        cur = self.read_proprio().arm_qpos
        n = max(1, int(duration_s / self.dt))
        for k in range(1, n + 1):
            a = cur + (np.asarray(arm_qpos) - cur) * (k / n)
            self.command(a, grip_qpos); time.sleep(self.dt)

    def _play_path(self, arm_path, grip_path):
        """Replay a precomputed (arm, grip) waypoint list open-loop at the control rate."""
        import time
        for a, g in zip(arm_path, grip_path):
            self.command(np.asarray(a, float), float(g)); time.sleep(self.dt)

    def _grip_ramp(self, g_to, n):
        """Linear grip ramp from the current grip qpos to g_to over n waypoints."""
        g0 = float(self.read_proprio().grip_qpos[0])
        return [g0 + (g_to - g0) * ((k + 1) / n) for k in range(n)]

    def initial_liftup(self, rest_arm, grip_open_qpos, steps=LIFTUP_STEPS):
        """SCRIPTED ENTRY ('lift+open'): open the gripper and raise from the current pose to the pre-grasp
        hover (rest_arm) via a cosine-eased ramp, then hand off to the policy. Explicit, smooth, and
        identical in sim -> deterministic episode start for online-RLPD."""
        arm_path = scripted_joint_path([self.read_proprio().arm_qpos.tolist(), list(rest_arm)], steps)
        self._play_path(arm_path, self._grip_ramp(grip_open_qpos, steps))

    def comeback_to_zero(self, grip_open_qpos, via=None, steps=COMEBACK_STEPS):
        """SCRIPTED RETURN/PARK: open the gripper, lift clear through `via` (default the hover APPROACH so
        the jaw leaves the table before translating), then park at the URDF home ZERO. Ends every episode
        at a known safe pose; the controlled Zero<->Rest path doubles as a sysid probe of the s2r gap."""
        via = list(APPROACH if via is None else via)
        arm_path = scripted_joint_path([self.read_proprio().arm_qpos.tolist(), via, list(ZERO)], steps)
        self._play_path(arm_path, self._grip_ramp(grip_open_qpos, steps))

    def estop(self):
        self.robot.disconnect()   # disable_torque_on_disconnect=True -> motors go limp safely

    # ---- calibration aid (run once on the arm) ----
    def verify_home(self):
        """Move to the URDF home (sim q=0) and report LeRobot readings; nonzero => set OFFSET_RAD.
        Also command +5deg per joint and check the sign matches (else flip SIGN)."""
        raise NotImplementedError("run interactively during arm bring-up to fill SIGN/OFFSET_RAD")

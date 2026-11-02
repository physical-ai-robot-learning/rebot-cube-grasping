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
# HARD ceiling on the FORCE_POS torque/current ratio. This gripper has been BROKEN TWICE by
# over-torque (first at 0.3, then again by a raise ABOVE 0.15). Clamp EVERY caller here so a stray
# CLI value can't crush the fragile silicone/linkage again. 0.15 is the highest value observed to
# hold WITHOUT breaking; 0.18 leaves a small documented nudge band (0.10->0.15->0.18) and blocks the
# 0.2-0.3 range that broke it. Raise this ceiling ONLY after a hardware repair proves a higher value safe.
GRIP_FORCE_RATIO_MAX = 0.18
# Closed-loop grip FORCE-SERVO (grip_force_servo): creep the gripper CLOSED until the measured motor torque
# hits a target, then HOLD. IMPORTANT (hardware finding 2026-07-08): on a RIGID cube torque is a near-vertical
# wall in position (grip_tau ~0.1->1.3 within ~3 motor-deg), so grip force is really set by the FORCE_POS
# current cap = the --grip-force RATIO, not by servo depth. Deploy with the calibrated ratio (calib_grip.py A3)
# so the cap ≈ the target torque; then the servo just settles against that cap without oscillating on the wall.
GRIP_SERVO_STEP = 0.0003           # qpos closed-step/tick (~4 motor-deg; was 0.0008≈10.7deg = coarser than the wall)
GRIP_SERVO_FLOOR_MARGIN = 0.005    # deepest over-travel below grip_close_qpos the servo may command
GRIP_SERVO_BACKOFF = 1.3           # if torque > target*this, relieve one step (single-step overshoot guard)
GRIP_TORQUE_TARGET_DEFAULT = 0.8   # default closed-loop target (firm-but-gentle; > thresh 0.5, << crush ~1.86)
GRIP_TORQUE_TARGET_MAX = 1.2       # servo NEVER targets more than this (the ~1.86 grip broke the gripper)
# add to fk_tcp z -> the policy drives the gripper that much lower. 0.12 was tuned for the OLD (z-biased)
# calib; with the FRESH calib_extrinsic, 0.12 drove the gripper to real fk_mid ~-0.06 (BELOW the table)
# -> it dug in and missed the cube -> reduced to 0.03. Tune by re-test (raise if it hovers high, lower
# if it digs into the table). 2026-07-10 HW-MEASURED: with the gripper tip resting ON the table (real tip
# z=0), io fk_tcp read z=0.059 at FK_Z_CORR=0.09 -> the correction was ~0.06 too large (matched the stale
# comment's "reduced to 0.03" that the value had drifted away from). Set 0.03 so io FK z ≈ real tip; this
# also un-poisons auto_z, whose fk_bias(=io-base) had inflated to =FK_Z_CORR=0.09 and over-descended ~6cm.
# 2026-07-22 RE-FIT (deploy/main.py bring-up): a clean 4-pt table-touch sweep (fk_verify_z, radial 0.38-0.57,
# --z-reach-slope 0) showed io fk z STILL over-read ~9.3cm CONSTANT at the tip-on-table height (z_gap_io ~+0.074
# after the reach-slope, vs ideal -0.019). Reduced by 0.093 -> -0.063 so io FK z ≈ real tip again + auto_z sane.
# Pairs with cli defaults --z-reach-slope +0.0676 --z-reach-ref 0.486 (the sweep's radial slope). ⚠️ calib-drift
# suspect (J2/J3 motor-zero) — this is the band-aid; re-verify with fk_verify_z after any re-calibration.
# 2026-07-23 ROOT CAUSE FOUND (P4 adversarial review): the "~9.3cm constant over-read" + suspected "J2/J3
# motor-zero drift" was actually joint_map.npz NOT being loaded since P1 (dbe34de moved robot_io under src/
# but left the data file in deploy/ -> silent identity fallback, J2/J3/J4 offset off ~14-18° -> corrupt FK).
# Now that the joint_map read path is fixed above, this band-aid (fit AROUND that bug) is stale -> ZEROED,
# then re-fit clean. 2026-07-23 RE-FIT (fk_verify_z, joint_map FIXED, --z-reach-slope 0): 5 tight table-touch
# samples at the operating radial ~0.446 read z_gap_io = -0.034 ± 0.002 (vs ideal -0.019) => io fk z reads
# ~1.5cm LOW -> FK_Z_CORR = +0.015 raises it to ~real tip. HUGE improvement: the old ~9.3cm over-read is gone
# (that WAS the identity-offset FK error). Slope left 0 (near-radial samples were inconsistent, unreliable for
# a slope; operating region ~0.45 is the tight cluster). Re-fit near/far with clean touches if cross-radial
# grasps drift. Pairs with cli --z-reach-slope 0.
FK_Z_CORR = 0.015
# Real grasp lands CAMERA-SIDE (+y, "front") of the reported midpoint -> the gripper "closed IN FRONT
# of the cube". Shift the reported tcp toward the camera (= AGAINST the FK forward dir gripper_link->
# fingers, which points toward base) by JAW_FWD so the policy pulls the real jaw back onto the cube.
# Sign: subtract along fwd (=move +y at rest). 0.04 still closed slightly in front of the cube +
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


class JointSlew:
    """Acceleration-limited joint setpoint shaper — the SMOOTH, lag-free replacement for an output EMA.
    Each step() ramps a per-joint commanded velocity toward the velocity that closes the gap to `target`
    (capped at v_max), clamps the per-tick velocity CHANGE to a_max*dt (accel = the smoothness dial), and
    caps the step so it never overshoots the target. Built for a MOVING setpoint stream (the IK feeds a
    one-step-ahead target each tick): it cruises at the stream's velocity with smooth ramp-up/down.
    Contrast an EMA: it decays exponentially (fast-then-crawl, non-uniform velocity) and never reaches a
    MOVING target (steady-state lag) — the very thing that made the descend "stop too high". reset(q)
    snaps the state (scripted hand-off / freeze). The DM arm motor itself runs POS_VEL vlim=32 rad/s
    (snaps to each setpoint), so motion smoothness is set entirely by THIS setpoint stream + the rate.

    v_max: rad/s (set >= the IK's arm_step*rate so the IK stays the velocity authority).
    a_max: rad/s^2 (lower = smoother starts/stops = the main dial; higher = snappier)."""

    def __init__(self, q0, dt, v_max=2.5, a_max=8.0):
        self.q = np.asarray(q0, float).reshape(-1).copy()
        self.v = np.zeros_like(self.q)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def reset(self, q):
        self.q = np.asarray(q, float).reshape(-1).copy()
        self.v[:] = 0.0

    def step(self, target, dt=None):
        """Advance one tick toward target. Pass dt = the ACTUAL elapsed since the last call (the loop is
        camera-paced and jitters; using a fixed dt while the real period differs would scale the speed)."""
        dt = self.dt if dt is None else float(dt)
        target = np.asarray(target, float).reshape(-1)
        gap = target - self.q
        ag = np.abs(gap)
        a_dt = self.a_max * dt
        # Track a MOVING setpoint: desired velocity = the speed that closes the gap in one tick, capped at
        # v_max. (NOT a braking-distance limit — that assumes a FIXED distant target and, since the IK feeds
        # only a one-step-ahead target each tick, would wrongly throttle the cruise speed to a crawl.)
        v_des = np.clip(gap / dt, -self.v_max, self.v_max)
        dv = np.clip(v_des - self.v, -a_dt, a_dt)        # accel-limit the velocity CHANGE (the smoothness)
        self.v = self.v + dv
        self.v = np.clip(self.v, -ag / dt, ag / dt)      # never step PAST the target this tick (no overshoot)
        self.q = self.q + self.v * dt
        return self.q.copy()


def _quat_wxyz_to_R(q):
    """Quaternion -> 3x3 rotation matrix. Input q is in (w,x,y,z) order (scalar first); output is the standard direction-cosine rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


class RobotIO:
    n_arm = 6

    def read_proprio(self) -> Proprio:
        raise NotImplementedError

    def fk_tcp(self, arm_qpos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_grasped(self, grip_qpos: np.ndarray) -> bool:
        raise NotImplementedError

    def command(self, arm_target: np.ndarray, grip_target: float) -> None:
        raise NotImplementedError

    def move_to(self, arm_qpos, grip_qpos, duration_s=2.0):
        raise NotImplementedError

    def initial_liftup(self, rest_arm, grip_open_qpos, steps=LIFTUP_STEPS):
        raise NotImplementedError

    def comeback_to_zero(self, grip_open_qpos, via=None, steps=COMEBACK_STEPS):
        raise NotImplementedError

    def estop(self):
        raise NotImplementedError


class DamiaoRobotIO(RobotIO):
    def __init__(self, cfg, urdf_path, port="/dev/ttyACM0", robot_id="follower1", dt=1.0 / 15.0,
                 grip_closed_deg=0.0, grip_open_deg=-90.0, grip_force_ratio=0.10, grip_close_limit_deg=120.0,
                 arm_vlim_rad_s=None):
        """cfg = EnvConfig (base_pos/base_quat/tcp_tip_drop/grip_*). urdf_path = the with-gripper URDF.
        grip_closed_deg/grip_open_deg: CALIB endpoints mapping the gripper motor (deg) <-> sim qpos (m).
        grip_force_ratio: the gripper FORCE_POS torque/current ratio [0,1] = the motor clamp-force CAP. Default
        lowered to 0.10 (conservative) after the gripper BROKE TWICE from over-torque (0.3, then a raise >0.15).
        ⚠️ HARD-CLAMPED to GRIP_FORCE_RATIO_MAX (0.18) here regardless of what the caller passes — a higher value
        is silently unreachable, with a warning. If a cube slips, nudge up in SMALL steps (0.10->0.15) and watch
        grip_torque; prefer fixing the grasp GEOMETRY over cranking force. Fragile silicone/linkage.
        grip_close_limit_deg: UNLOCK the close direction. The gripper CLOSES at +motor-deg, but the driver's
        default gripper joint_limit max=0 CLIPPED every closing command at 0deg (HW: both reachable extremes
        0deg/-250deg were OPEN) -> the jaws never closed. Raise the max so close is commandable."""
        from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
        import pinocchio as pin
        self.pin = pin
        self.cfg = cfg
        self.dt = dt
        # ARM motor velocity limit (rad/s) for POS_VEL. The lerobot driver HARDCODES vlim=32 (~1833 deg/s) ->
        # the motor SNAPS to each setpoint and dwells (jerky between coarse, camera-paced control ticks). Set
        # this so the motor GLIDES between setpoints (smooth + still fast) -- POS_VEL's intended use. None = OFF
        # (driver's 32). Applied by re-issuing send_pos_vel per arm motor in command() (no vendored patch).
        self.arm_vlim = arm_vlim_rad_s
        # HARD safety clamp: block the over-torque that broke this gripper twice, no matter the caller/CLI.
        _req = float(grip_force_ratio)
        _ratio = float(np.clip(_req, 0.0, GRIP_FORCE_RATIO_MAX))
        if _req > GRIP_FORCE_RATIO_MAX:
            print(f"[DamiaoRobotIO] ⚠️ grip_force_ratio {_req:.2f} EXCEEDS the safety cap "
                  f"{GRIP_FORCE_RATIO_MAX:.2f} -> CLAMPED to {_ratio:.2f}. 0.3 and >0.15 have BROKEN this "
                  f"gripper; edit GRIP_FORCE_RATIO_MAX only after a hardware repair proves a higher value safe.")
        self.grip_force_ratio = _ratio
        _lcfg = SeeedB601DMFollowerConfig(
            port=port, id=robot_id, can_adapter="damiao",
            force_pos_torque_ration=_ratio)
        _gmin = _lcfg.joint_limits["gripper"][0]
        _lcfg.joint_limits["gripper"] = (_gmin, float(grip_close_limit_deg))   # extend max 0 -> + so jaws can close
        self.robot = SeeedB601DMFollower(_lcfg)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.fkdata = self.model.createData()
        self.fid_l = self.model.getFrameId("gripper_left")
        self.fid_r = self.model.getFrameId("gripper_right")
        self.fid_glink = self.model.getFrameId("gripper_link")   # "behind" ref for the jaw forward direction
        self.R_base = _quat_wxyz_to_R(cfg.base_quat_wxyz)
        self.base_pos = np.asarray(cfg.base_pos, float)
        self.grip_closed_deg, self.grip_open_deg = grip_closed_deg, grip_open_deg
        # joint convention map from src/tools/calib_convention.py (rest-anchored). Falls back to module
        # SIGN/OFFSET_RAD (identity) with a warning if absent.
        self.SIGN, self.OFFSET = SIGN.copy(), OFFSET_RAD.copy()
        # joint_map.npz lives in the repo's deploy/ data dir. P1 (dbe34de) moved robot_io.py from deploy/ to
        # src/rebot_sim2real/deploy/io/ but LEFT the data file in deploy/, so the old dirname(__file__)-relative
        # read silently missed it -> identity fallback (J2/J3/J4 offset off ~14-18°, corrupting all FK-based
        # positioning). Read the repo deploy/ copy explicitly (robot_io.py is 4 dirs below repo root).
        mp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "deploy", "joint_map.npz"))
        if os.path.exists(mp):
            jm = np.load(mp, allow_pickle=True)
            self.SIGN = jm["sign"].astype(float)
            self.OFFSET = np.asarray(jm["rest_arm_rad"], float) - self.SIGN * np.deg2rad(jm["L_rest_deg"])
            self.grip_closed_deg, self.grip_open_deg = float(jm["grip_closed_deg"]), float(jm["grip_open_deg"])
            print(f"[DamiaoRobotIO] loaded joint_map.npz (rest-anchored) from {mp}; SIGN={self.SIGN} OFFSET={np.round(self.OFFSET,3)}")
        else:
            print(f"[DamiaoRobotIO] WARNING no joint_map.npz at {mp} -> identity convention (run src/tools/calib_convention.py!)")
        self._prev_arm = None
        self._prev_grip = None
        self._last_read_t = None            # wall-clock of the last read_proprio -> REAL finite-diff interval (not a fixed dt)
        self._last_grip = np.array([cfg.grip_open_qpos] * 2)
        self._last_grip_torque = 0.0

    # ---- gripper sim<->lerobot (linear between calibrated closed/open endpoints) ----
    def _grip_deg_to_qpos(self, deg):
        f = (deg - self.grip_closed_deg) / (self.grip_open_deg - self.grip_closed_deg + 1e-9)
        q = self.cfg.grip_close_qpos + f * (self.cfg.grip_open_qpos - self.cfg.grip_close_qpos)
        # allow the READBACK to report BELOW grip_close_qpos (over-close) so a hard grip is visible for
        # debugging — the old hard clamp AT grip_close_qpos hid any close past the pinch floor. Upper bound
        # kept at grip_open_qpos; lower relaxed by 0.03 (still bounded against garbage readings).
        return float(np.clip(q, self.cfg.grip_close_qpos - 0.03, self.cfg.grip_open_qpos))

    def _grip_qpos_to_deg(self, q):
        f = (q - self.cfg.grip_close_qpos) / (self.cfg.grip_open_qpos - self.cfg.grip_close_qpos + 1e-9)
        return self.grip_closed_deg + f * (self.grip_open_deg - self.grip_closed_deg)

    def connect(self):
        self.robot.connect(calibrate=False)   # assume already lerobot-calibrated

    def read_proprio(self) -> Proprio:
        obs = self.robot.get_observation()
        deg = np.array([obs[f"{lj}.pos"] for lj in LEROBOT_ARM], float)
        arm_qpos = self.SIGN * np.deg2rad(deg) + self.OFFSET
        gq = self._grip_deg_to_qpos(obs["gripper.pos"])
        grip_qpos = np.array([gq, gq])
        self._last_grip_torque = abs(float(obs.get("gripper.torque", 0.0)))
        # joint torque (sim convention = SIGN*motor_torque, same map as qpos; OFFSET is position-only)
        arm_torque = self.SIGN * np.array([float(obs.get(f"{lj}.torque", 0.0)) for lj in LEROBOT_ARM], float)
        import time
        now = time.perf_counter()
        if self._prev_arm is None:                              # first read (or reset after a scripted move) -> vel 0
            arm_qvel = np.zeros(6)
            grip_qvel = np.zeros(2)
        else:
            # finite-diff over the REAL elapsed interval, not the hardcoded nominal self.dt: a jittery/stalled loop
            # otherwise mis-scales qvel (and a long gap would blow it up). Clamp to a sane band around the nominal.
            dtr = (now - self._last_read_t) if self._last_read_t is not None else self.dt
            dtr = float(np.clip(dtr, 0.2 * self.dt, 5.0 * self.dt))
            arm_qvel = (arm_qpos - self._prev_arm) / dtr
            grip_qvel = (grip_qpos - self._prev_grip) / dtr
        self._prev_arm, self._prev_grip, self._last_grip = arm_qpos, grip_qpos, grip_qpos
        self._last_read_t = now
        return Proprio(arm_qpos, arm_qvel, grip_qpos, grip_qvel,
                       arm_torque=arm_torque, grip_torque=self._last_grip_torque)

    def fk_tcp(self, arm_qpos: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[0:6] = arm_qpos
        q[6:8] = self._last_grip                                # midpoint ~grip-invariant
        self.pin.forwardKinematics(self.model, self.fkdata, q)
        self.pin.updateFramePlacements(self.model, self.fkdata)
        fl = np.asarray(self.fkdata.oMf[self.fid_l].translation)
        fr = np.asarray(self.fkdata.oMf[self.fid_r].translation)
        glink = np.asarray(self.fkdata.oMf[self.fid_glink].translation)
        mid = self.R_base @ (0.5 * (fl + fr)) + self.base_pos   # finger-origin midpoint, env frame
        gl = self.R_base @ glink + self.base_pos                # gripper_link origin, env frame
        fwd = (mid - gl)[:2]                                    # horizontal forward (gripper_link -> fingers, ~-y)
        n = float(np.linalg.norm(fwd))
        if n > 1e-6:
            mid[:2] -= JAW_FWD * fwd / n                        # shift toward camera/+y: pull real jaw onto cube
        mid[2] -= self.cfg.tcp_tip_drop
        mid[2] += FK_Z_CORR
        return mid

    def is_grasped(self, grip_qpos: np.ndarray) -> bool:
        return self._last_grip_torque > GRIP_TORQUE_THRESH   # torque, not gap (gap false-fires mid-close)

    def grip_force_servo(self, grip_cmd: float, grip_torque: float, target_torque: float) -> float:
        """ONE closed-loop force-servo step for grasping. Creep the gripper CLOSED until the measured motor
        torque reaches target_torque, then HOLD (never drive deeper). Returns the next grip_cmd (qpos); call
        it every control tick while closing, feeding back the live grip_torque. Safe by construction:
          (1) stops at the torque target -> no crush,
          (2) clamped to grip_close_qpos - GRIP_SERVO_FLOOR_MARGIN -> no linkage over-travel,
          (3) the FORCE_POS ratio cap (hard-capped GRIP_FORCE_RATIO_MAX) still bounds peak torque underneath.
        target_torque is clamped to GRIP_TORQUE_TARGET_MAX (well below the ~1.86 that broke the gripper)."""
        tgt = float(np.clip(target_torque, 0.0, GRIP_TORQUE_TARGET_MAX))
        floor = self.cfg.grip_close_qpos - GRIP_SERVO_FLOOR_MARGIN
        if grip_torque < tgt:
            grip_cmd = grip_cmd - GRIP_SERVO_STEP            # press a touch harder -> torque rises next tick
        elif grip_torque > tgt * GRIP_SERVO_BACKOFF:
            grip_cmd = grip_cmd + GRIP_SERVO_STEP            # single-step overshoot -> relieve a touch
        # else: within [tgt, tgt*backoff] -> HOLD (do not move)
        return float(np.clip(grip_cmd, floor, self.cfg.grip_open_qpos))

    def command(self, arm_target: np.ndarray, grip_target: float) -> None:
        deg = np.rad2deg((np.asarray(arm_target) - self.OFFSET) / self.SIGN)   # raw motor-deg targets
        action = {}
        arm_rad = {}
        for i, lj in enumerate(LEROBOT_ARM):
            lo, hi = HW_LIMIT_DEG[lj]
            motor_deg = float(np.clip(deg[i], lo, hi))
            action[f"{lj}.pos"] = motor_deg / JOINT_DIR[lj]    # /direction: counteract send_action's *direction
            arm_rad[lj] = float(np.deg2rad(motor_deg))         # motor-frame target (== what send_action sends)
        action["gripper.pos"] = float(self._grip_qpos_to_deg(grip_target)) / JOINT_DIR["gripper"]
        self.robot.send_action(action)                         # direction/limit + gripper FORCE_POS (arm vlim=32)
        if self.arm_vlim is not None:
            # OVERRIDE the arm's hardcoded vlim=32 -> re-issue each arm motor at OUR vlim so it GLIDES between
            # the coarse control setpoints (smooth, no snap). Same pos send_action just sent; gripper untouched.
            for lj, prad in arm_rad.items():
                self.robot.motors[lj].send_pos_vel(prad, float(self.arm_vlim))

    def move_to(self, arm_qpos, grip_qpos, duration_s=2.0):
        """Ramp from the current pose to the target (open-loop pre-position)."""
        import time
        cur = self.read_proprio().arm_qpos
        n = max(1, int(duration_s / self.dt))
        for k in range(1, n + 1):
            a = cur + (np.asarray(arm_qpos) - cur) * (k / n)
            self.command(a, grip_qpos)
            time.sleep(self.dt)
        self._prev_arm = self._prev_grip = None    # open-loop move (no read) left _prev stale -> reset so the NEXT
        #                                            read_proprio reports qvel 0, not a full-move-distance spike (OOD obs)

    def _play_path(self, arm_path, grip_path):
        """Replay a precomputed (arm, grip) waypoint list open-loop at the control rate."""
        import time
        for a, g in zip(arm_path, grip_path):
            self.command(np.asarray(a, float), float(g))
            time.sleep(self.dt)
        self._prev_arm = self._prev_grip = None    # scripted open-loop path left _prev stale -> reset so the next
        #                                            read_proprio reports qvel 0, not a full-path-distance spike (OOD obs)

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
        try:
            self.robot.disconnect()   # disable_torque_on_disconnect=True -> motors go limp safely
        except Exception as e:        # already-disconnected / double-estop: don't mask the real shutdown
            print(f"[DamiaoRobotIO] estop disconnect: {e}")

    # ---- calibration aid (run once on the arm) ----
    def verify_home(self):
        """Move to the URDF home (sim q=0) and report LeRobot readings; nonzero => set OFFSET_RAD.
        Also command +5deg per joint and check the sign matches (else flip SIGN)."""
        raise NotImplementedError("run interactively during arm bring-up to fill SIGN/OFFSET_RAD")

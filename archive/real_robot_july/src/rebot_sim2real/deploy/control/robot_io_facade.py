"""Robot IO facade: the single point through which the control loop talks to the robot (read/write/scripted moves/estop).

Bundles the 4 nested closures (send_arm / read_pr / run_scripted / _estop) inside the old debug_deepreach.py `main()`
into one. All four separately repeated the same decision branching on `io_tj is not None` (whether --io-thread is on);
that branch is gathered in this one place.

Two paths:
  - io-thread ON (io_tj present): control writes RAW commands to the Blackboard, and RobotIoThread performs shaping/servo/scripted
    moves on the bus. read is a blackboard snapshot.
  - io-thread OFF (io_tj None): synchronous single loop. This facade shapes (JointSlew+LPF+chunk) and commands io directly.

⚠️ The synchronous shaping logic is (intentionally) identical to io_thread.RobotIoThread._shape. io_thread is HW-verified
   working code, so it is not refactored into a shared helper now (avoiding needless churn) — the duplication is a follow-up task.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np


class RobotIoFacade:
    def __init__(self, io, bb, io_tj, shaper, *, use_shaper, cmd_lpf_tau, chunk_n, chunk_decay,
                 rest_np, cmd_snap_jump=0.5):
        """Inputs:
          io          : DamiaoRobotIO (CAN bus hardware)
          bb          : Blackboard (shared state in io-thread mode)
          io_tj       : RobotIoThread or None (None=synchronous path)
          shaper      : JointSlew acceleration-limited shaper (synchronous path only; initialized at REST)
          use_shaper  : True=soft JointSlew shaping, False=motor vlim glide (IK target sent directly)
          cmd_lpf_tau : first-order LPF time constant for joint setpoints (s; 0=OFF), synchronous path
          chunk_n     : action-chunk window size (1=OFF), synchronous path
          chunk_decay : action-chunk exponential weighting (0<d<=1), synchronous path
          rest_np     : REST joint array (initial_liftup target of run_scripted)
          cmd_snap_jump : if |Δq| > this value (rad), treat as a mode change/retarget and reset the shaper (the shaper is still acceleration-limited)
        """
        self._io = io
        self._bb = bb
        self._io_tj = io_tj
        self._shaper = shaper
        self._use_shaper = bool(use_shaper)
        self._cmd_lpf_tau = float(cmd_lpf_tau)
        self._chunk_n = max(1, int(chunk_n))
        self._chunk_decay = float(np.clip(chunk_decay, 1e-3, 1.0))
        self._rest_np = np.asarray(rest_np, dtype=float)
        self._cmd_snap_jump = float(cmd_snap_jump)
        # synchronous-path shaping state (nonlocal cmd_lpf / chunk_buf / _seq of the old send_arm)
        self._cmd_lpf = None       # LPF'd joint setpoint (rate-correct EMA); reset on snap/large jump
        self._chunk_buf = []       # ring of recent IK joint targets (temporal ensemble)
        self._seq = 0              # monotonically increasing command sequence in io-thread mode

    @property
    def uses_io_thread(self) -> bool:
        """Whether the io-thread path is used. (output: bool)"""
        return self._io_tj is not None

    def read_proprio(self):
        """Returns the current proprio. (output: proprio object or None)
        io-thread: blackboard publish snapshot (includes .tcp_io/.is_grasped). Synchronous: io.read_proprio() directly."""
        if self._io_tj is not None:
            snap = self._bb.proprio.snapshot()
            if snap.get("arm_qpos") is not None:
                return SimpleNamespace(**snap)
            return None
        return self._io.read_proprio()

    def tcp_io(self, pr):
        """TCP (grasp point) world coordinates (x,y,z) per robot_io FK. (input: proprio → output: list[3])
        io-thread: pr.tcp_io (already computed and published by the thread). Synchronous: io.fk_tcp(arm_qpos) directly."""
        if self._io_tj is not None:
            return list(pr.tcp_io)
        return list(map(float, self._io.fk_tcp(pr.arm_qpos)))

    def is_grasped(self, grip_qpos):
        """Whether the gripper is holding the cube (measured torque>0.5 etc.). (input: grip_qpos → output: bool)
        The check is owned by io, so both paths use io.is_grasped directly (same as the monolith)."""
        return bool(self._io.is_grasped(grip_qpos))

    def send_arm(self, q_target, grip, loop_dt, snap=False):
        """Arm command. (input: q_target[6] IK target, grip gripper setpoint, loop_dt measured loop period (s), snap whether handoff/freeze
        → output: none (commands to motors/blackboard)).
        io-thread: publish the RAW target to the blackboard (the thread shapes + sends). Synchronous: motor-vlim glide or
        JointSlew shaper (optional action-chunk→LPF in front of the shaper)."""
        qt = np.asarray(q_target, dtype=float).reshape(-1)
        if self._io_tj is not None:                    # --io-thread: publish RAW target, the IO thread shapes + sends
            self._seq += 1
            self._bb.command.publish(arm=qt.tolist(), grip=float(grip), grip_mode="hold",
                                     arm_snap=bool(snap), seq=self._seq)
            return
        if not self._use_shaper:                       # motor vlim glide: the motor smooths → send IK target directly
            self._io.command(qt, float(grip))
            return
        if snap:                                       # handoff/freeze: send directly + reset all shaper state
            self._shaper.reset(qt)
            self._cmd_lpf = qt.copy()
            self._chunk_buf.clear()
            self._io.command(qt, float(grip))
            return
        if self._cmd_lpf is None or float(np.max(np.abs(qt - self._cmd_lpf))) > self._cmd_snap_jump:
            # first call or mode change/large retarget: reset LPF+chunk to the new target (prevents laggy drift), but
            # the motion itself still goes through the acceleration shaper (no snap).
            self._cmd_lpf = qt.copy()
            self._chunk_buf.clear()
        else:
            if self._chunk_n > 1:                      # ACTION CHUNKING: exponentially weighted temporal ensemble (latest weight 1)
                self._chunk_buf.append(qt.copy())
                if len(self._chunk_buf) > self._chunk_n:
                    self._chunk_buf.pop(0)
                weights = np.array([self._chunk_decay ** (len(self._chunk_buf) - 1 - i)
                                    for i in range(len(self._chunk_buf))])
                qt = np.average(np.stack(self._chunk_buf), axis=0, weights=weights)
            if self._cmd_lpf_tau > 1e-6:               # LPF: first-order, rate-correct (measured loop_dt)
                alpha = 1.0 - float(np.exp(-loop_dt / self._cmd_lpf_tau))
                self._cmd_lpf = (1.0 - alpha) * self._cmd_lpf + alpha * qt
                qt = self._cmd_lpf.copy()
            else:
                self._cmd_lpf = qt.copy()
        q_out = self._shaper.step(qt, loop_dt)         # measured dt → correct speed even if --rate is not sustained
        self._io.command(q_out, float(grip))

    def reset_shaper(self, q):
        """Re-syncs the synchronous-path shaper (+LPF/chunk) to pose q. (input: q[6] → output: none)
        Used by command handlers to align the shaper with the reached pose after a scripted move/mode switch (old shaper.reset).
        Harmless in io-thread mode since the control-side shaper is unused (the io thread shaper is reset via the arm_snap flag)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        self._shaper.reset(q)
        self._cmd_lpf = q.copy()
        self._chunk_buf.clear()

    def run_scripted(self, name, grip):
        """Scripted blocking move ('initial_liftup' | 'comeback_to_zero'). (input: name, grip → output: none, blocks until done)
        io-thread: post io_task and wait until done (the thread performs it on the bus without contention). Synchronous: call io directly."""
        if self._io_tj is not None:
            self._bb.mode.publish(io_task={"name": name, "grip": float(grip)}, io_task_state="idle")
            deadline = time.time() + 8.0
            while self._bb.mode.get("io_task_state") != "done" and time.time() < deadline:
                time.sleep(0.02)
        elif name == "comeback_to_zero":
            self._io.comeback_to_zero(grip)
        else:
            self._io.initial_liftup(self._rest_np, grip)

    def grip_force_servo(self, arm_qpos, grip_cmd, grip_torque, target_torque, loop_dt):
        """One step of closed-loop gripper force-servo (arm frozen at the current pose; only the gripper gently closes to the target torque and holds).
        (input: arm_qpos[6], grip_cmd current setpoint, grip_torque measured torque, target_torque target, loop_dt → output: new grip_cmd)
        io-thread: publish grip_mode='force_servo' to the blackboard (the thread runs the servo and writes grip back).
        Synchronous: compute a new setpoint with io.grip_force_servo, then command arm+gripper directly."""
        if self._io_tj is not None:
            self._seq += 1
            self._bb.command.publish(arm=list(map(float, arm_qpos)), grip_mode="force_servo",
                                     grip_torque_target=float(target_torque), arm_snap=True, seq=self._seq)
            return float(self._bb.command.get("grip", grip_cmd))
        new_grip = self._io.grip_force_servo(grip_cmd, float(grip_torque), float(target_torque))
        self._io.command(np.asarray(arm_qpos, dtype=float).reshape(-1), new_grip)   # arm frozen; only the gripper converges
        return new_grip

    def estop(self):
        """Emergency stop (ABORT button). (output: none) io-thread: estop publish + thread stop. Synchronous: io.estop()."""
        if self._io_tj is not None:
            self._bb.mode.publish(estop=True)
            self._io_tj.stop()
        else:
            self._io.estop()

    def shutdown(self):
        """Shutdown sequence (control loop finally). Stops the io-thread first (if any), then ALWAYS calls motor estop
        — safety (motor stop) must not wait on the camera join, so motors FIRST. (same as the monolith finally)."""
        if self._io_tj is not None:
            self._bb.mode.publish(estop=True)
            self._io_tj.stop()
        self._io.estop()

"""Robot-IO thread for the deploy control stack (3-thread refactor Step 3, docs/28).

Owns the CAN bus EXCLUSIVELY: ONE thread that alternates READ (proprio -> blackboard) then WRITE
(shaped command -> motors). No other thread touches `io`, so there is never a concurrent read+write
on the single DamiaoRobotIO (hazard R1). It also owns everything that was woven into the old
synchronous control loop's send_arm/gripping/scripted-move paths:

  - the JointSlew shaper + --cmd-lpf-tau LPF + --action-chunk temporal ensemble (moved out of the
    send_arm closure; the control thread now writes a RAW joint target and IO shapes it -- hazard C1,
    so the shaper steps on the IO loop_dt, its OWN rate);
  - the closed-loop gripper force-servo (creep to grip_torque_target);
  - the SCRIPTED blocking moves (initial_liftup / comeback_to_zero), run as blackboard-requested
    io_tasks so they never freeze the control thread (hazard C-block);
  - estop, checked at the top of every loop so an abort can't wait on a blocking read (hazard R-estop).

Blackboard contract (docs/28):
  reads  command.{arm, arm_snap, grip, grip_mode, grip_torque_target, seq}, mode.{io_task, estop}
  writes proprio.{arm_qpos, arm_qvel, grip_qpos, grip_qvel, arm_torque, grip_torque, is_grasped,
                  tcp_io, ts}, mode.io_task_state, debug.gates.io

Wired via deploy/main.py (--io-thread): main.py constructs this RobotIoThread and the control loop
reads/writes the blackboard while this thread owns the bus. Still standalone + unit-testable in isolation.
"""

from __future__ import annotations
import logging
import threading
import time

import numpy as np

log = logging.getLogger("io")

CMD_SNAP_JUMP = 0.5   # |Δq|>this rad on any joint = mode change / re-target -> reset the smoothers (the shaper still
#   accel-limits the move, so this only prevents a long laggy LPF drift across a big jump). Mirrors debug_deepreach.


class RobotIoThread(threading.Thread):
    def __init__(self, io, bb, shaper, *, use_shaper=True, cmd_lpf_tau=0.0, action_chunk=1,
                 chunk_decay=0.5, rest_np=None, grip_open=None, hz=100.0):
        super().__init__(daemon=True, name="IO")
        self._io = io
        self._bb = bb
        self._shaper = shaper                       # JointSlew (created by the entrypoint at REST)
        self._use_shaper = bool(use_shaper)         # False = --arm-vlim glide (motor smooths; send raw)
        self._cmd_lpf_tau = float(cmd_lpf_tau)
        self._chunk_n = max(1, int(action_chunk))
        self._chunk_decay = float(np.clip(chunk_decay, 1e-3, 1.0))
        self._rest_np = None if rest_np is None else np.asarray(rest_np, float)
        self._grip_open = grip_open
        self._period = 1.0 / float(hz)
        # shaper-side smoother state (moved out of the send_arm closure; IO owns it now)
        self._cmd_lpf = None
        self._chunk_buf = []
        self._prev_arm = None                       # for the rate-correct qvel finite-diff (hazard R4: IO owns it)
        self._prev_t = None
        self._running = False
        self._last_seq = -1

    # ---- shaping: chunk -> LPF -> JointSlew accel-shaper (identical math to the old send_arm) ----
    def _shape(self, qt, loop_dt, snap):
        qt = np.asarray(qt, float).reshape(-1)
        if not self._use_shaper:                    # motor-vlim glide -> send the target straight through
            return qt
        if snap:                                    # handoff/freeze: reset ALL smoother state + send direct
            self._shaper.reset(qt); self._cmd_lpf = qt.copy(); self._chunk_buf.clear()
            return qt
        if self._cmd_lpf is None or float(np.max(np.abs(qt - self._cmd_lpf))) > CMD_SNAP_JUMP:
            self._cmd_lpf = qt.copy(); self._chunk_buf.clear()      # first call / big jump -> reset (shaper still ramps)
        else:
            if self._chunk_n > 1:                   # ACTION CHUNKING: exp-weighted temporal ensemble (newest weight 1)
                self._chunk_buf.append(qt.copy())
                if len(self._chunk_buf) > self._chunk_n:
                    self._chunk_buf.pop(0)
                w = np.array([self._chunk_decay ** (len(self._chunk_buf) - 1 - i)
                              for i in range(len(self._chunk_buf))])
                qt = np.average(np.stack(self._chunk_buf), axis=0, weights=w)
            if self._cmd_lpf_tau > 1e-6:            # LOW-PASS FILTER: first-order, rate-correct on the IO loop_dt
                a = 1.0 - float(np.exp(-loop_dt / self._cmd_lpf_tau))
                self._cmd_lpf = (1.0 - a) * self._cmd_lpf + a * qt
                qt = self._cmd_lpf.copy()
            else:
                self._cmd_lpf = qt.copy()
        return self._shaper.step(qt, loop_dt)

    def _publish_proprio(self, now):
        pr = self._io.read_proprio()
        # rate-correct qvel finite-diff owned HERE (hazard R4): read_proprio already returns qvel, but the IO
        # thread is the single owner of timing, so we keep read_proprio's qvel and just stamp freshness.
        tcp_io = list(map(float, self._io.fk_tcp(pr.arm_qpos)))
        self._bb.proprio.publish(
            arm_qpos=np.asarray(pr.arm_qpos, float), arm_qvel=np.asarray(pr.arm_qvel, float),
            grip_qpos=np.asarray(pr.grip_qpos, float), grip_qvel=np.asarray(pr.grip_qvel, float),
            arm_torque=np.asarray(pr.arm_torque, float), grip_torque=float(pr.grip_torque),
            is_grasped=bool(self._io.is_grasped(pr.grip_qpos)), tcp_io=tcp_io, ts=now)
        return pr

    def _run_io_task(self, task, grip_now):
        """Execute a control-requested SCRIPTED blocking move on THIS thread (so control keeps ticking).

        A scripted move relocates the arm OUT OF BAND, so the command the control thread wrote BEFORE the
        task (e.g. the last Follow/RUN target in bb.command) is now stale. Left in place, the loop's
        fallthrough would immediately drive the arm right back to it -- the "Reset returns to the Follow
        pose" bug. So after the move we RE-SYNC bb.command to the pose just achieved (arm_snap so the
        smoothers don't ramp from the old target, grip_mode "hold" so a prior force-servo can't resume).
        """
        name = task.get("name")
        self._bb.mode.publish(io_task_state="busy")
        log.info("io_task %s START", name)
        try:
            hold = grip = None
            if name == "initial_liftup":
                grip = float(task.get("grip", grip_now))
                self._io.initial_liftup(self._rest_np, grip)
                hold = None if self._rest_np is None else np.asarray(self._rest_np, float)
            elif name == "comeback_to_zero":
                grip = float(task.get("grip", self._grip_open))
                self._io.comeback_to_zero(grip)
                hold = np.zeros(6)
            if hold is not None:
                self._shaper.reset(hold); self._cmd_lpf = hold.copy(); self._chunk_buf.clear()
                self._bb.command.publish(arm=list(map(float, hold)), grip=grip,
                                         grip_mode="hold", arm_snap=True)
        finally:
            self._bb.mode.publish(io_task_state="done", io_task=None)
            log.info("io_task %s DONE", name)

    def run(self):
        self._running = True
        try:
            while self._running:
                t0 = time.perf_counter()
                loop_dt = self._period if self._prev_t is None else (t0 - self._prev_t)
                self._prev_t = t0
                if self._bb.mode.get("estop"):                      # priority abort: don't wait on anything
                    self._io.estop(); self._running = False; break
                pr = self._publish_proprio(now=time.time())         # (1) record current position -> blackboard
                task = self._bb.mode.get("io_task")
                if task and self._bb.mode.get("io_task_state") != "busy":
                    self._run_io_task(task, float(pr.grip_qpos[0]) if len(pr.grip_qpos) else 0.0)
                    continue                                        # scripted move consumed this cycle
                cmd = self._bb.command.snapshot()                   # (2) read control's raw target
                q = cmd.get("arm")
                if q is not None:
                    grip = float(cmd.get("grip", 0.0))
                    if cmd.get("grip_mode") == "force_servo":       # closed-loop grip torque servo (IO-side)
                        grip = self._io.grip_force_servo(grip, float(pr.grip_torque),
                                                         float(cmd.get("grip_torque_target", 0.8)))
                        self._bb.command.publish(grip=grip)         # write the servo'd setpoint back
                    q_out = self._shape(q, loop_dt, bool(cmd.get("arm_snap", False)))   # (3) shape -> motor command
                    self._io.command(q_out, grip)
                    seq = cmd.get("seq", self._last_seq)
                    if seq != self._last_seq:
                        self._last_seq = seq
                self._bb.debug.publish(io={"bus_hz": round(1.0 / max(1e-3, loop_dt), 1),
                                           "last_seq": self._last_seq})
                time.sleep(max(0.0, self._period - (time.perf_counter() - t0)))
        except Exception:
            import traceback
            traceback.print_exc()
            self._bb.proprio.publish(dead=True)     # control thread fail-stops on a dead IO
        finally:
            self._running = False

    def stop(self):
        self._running = False
        if self.is_alive():
            self.join(timeout=1.0)

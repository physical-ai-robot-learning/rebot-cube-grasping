<!-- Deploy control-stack 3-thread + Blackboard refactor DESIGN (review plan, not yet implemented).
     Generated 2026-07-20 (session S17c continuation), grounded in a parallel architecture map of
     debug_deepreach.py / robot_io.py / debug_dashboard.py / base_controller.py. -->

# reBot Deploy Control Stack — 3-Thread Refactor Design

This is a review plan, not an implementation. It is grounded in the current `deploy/debug_deepreach.py` (~2140 lines), `deploy/robot_io.py`, `deploy/debug_dashboard.py`, and `src/rebot_sim2real/rl/base_controller.py`. The guiding constraint: **the HW-verified sd3-wrist RUN→descend grasp path must keep working at every step** — no big-bang rewrite.

---

## 1. TARGET ARCHITECTURE

Four long-lived threads communicating **only** through one `Blackboard` object. No thread calls another thread's methods; all coupling is via blackboard read/write. This mirrors the two patterns that already work in-tree: `DebugDashboard._lock` (coarse mutex over `_state`/`_frames`/`_cmds`) and `AsyncPerception` (`set_ctx`/`snapshot` copy-in/copy-out).

```
        cameras (front/wrist USB)                       CAN bus (Damiao)
              │                                               │
              ▼                                               ▼
     ┌─────────────────┐                            ┌──────────────────┐
     │  VISION THREAD   │                            │  ROBOT-IO THREAD  │
     │ ~10-30Hz (cam)   │                            │ bus-limited       │
     │ AsyncPerception  │                            │ read+write serial │
     └───────┬─────────┘                            └────────┬─────────┘
             │ write perc.*                    write proprio.*│  ▲ read cmd.arm/grip
             │ read  ctrl.tcp_prior            read cmd.*     ▼  │ read mode.io_task
             ▼                                                    │
        ╔══════════════════════════════════════════════════════════╗
        ║                      BLACKBOARD                            ║
        ║  proprio.*  perception.*  command.*  mode/FSM.*  debug.*   ║
        ║  (lock-guarded sections; snapshot() returns copies)        ║
        ╚══════════════════════════════════════════════════════════╝
             ▲  read proprio.* + perception.*        ▲ read debug/state
             │  write cmd.arm/grip + mode.* + debug  │ write cmd (button)
     ┌───────┴─────────┐                     ┌───────┴──────────┐
     │ IK/FK + CONTROL │                     │  DASHBOARD (HTTP) │
     │ THREAD ~100Hz    │◄── FSM lives here   │  thread (exists)  │
     │ owns `base`      │                     │  :8899 poll/cmd   │
     └─────────────────┘                     └──────────────────┘
```

Per-tick data flow the user asked for — **[record Current Position] → [IK/FK compute] → [Robot movement command]** — becomes a producer/consumer chain across threads instead of one synchronous line:

```
IO thread:      read_proprio() ──► blackboard.proprio          (record current position)
Control thread: read proprio+perc ─► base.deep_reach_step ─► blackboard.command.arm  (IK/FK compute)
IO thread:      read command.arm ──► shaper.step ─► io.command()  (robot movement command)
```

### Thread responsibilities

**Vision Thread** (already exists as `AsyncPerception`, `debug_deepreach.py:515-709`) — minimal change.
- Owns front+wrist cameras + all detectors (CNN/OWLv2/HSV, align-seg, front-gripper-Z frame-diff).
- Rate: own camera rate, ~10-30Hz, self-throttled `time.sleep(period)`.
- Reads blackboard: `perception.ctx.tcp_prior` (control-published FK TCP for the front-Z search window — the existing `_tcp`/`set_ctx` back-channel), `perception.ctx.need_align`, `perception.ctx.cap_req`, `perception.ctx.ov_queries`.
- Writes blackboard: `perception.snap` (the `_snap` dict), `perception.cube_world`/`cube_raw`, `perception.ts`/`hz`/`dead`; pushes dashboard frames.

**Robot-IO Thread** (NEW; today inline in main loop) — owns the CAN bus **exclusively**, single thread that **alternates read then write** (never concurrent read+write on the one `SeeedB601DMFollower` — see hazard R1).
- Rate: bus-limited (`measure_latency.py` comm-mode). Natural: one read + one write per bus cycle.
- Reads blackboard: `command.arm`/`command.grip`/`command.grip_torque_target`, `mode.io_task` (scripted-move requests), `mode.estop`.
- Writes blackboard: `proprio.*` (arm/grip qpos/qvel/torque, `is_grasped`), `proprio.ts`, `mode.io_task_state`.
- Owns: the `JointSlew` shaper + `cmd_lpf` + `chunk_buf` (moved here — see hazard C1), the gripper `grip_force_servo`, and the scripted blocking moves (`initial_liftup`/`comeback_to_zero`/`move_to`) run as **blackboard-requested tasks** so they don't freeze control.

**IK/FK + Control Thread** (NEW; the monolith `while-True` at `debug_deepreach.py:1399-2130`) — owns `base` (`RuleBaseController`) exclusively, runs the mode FSM.
- Rate: `--control-hz` (~100Hz async) — decoupled from bus, so setpoint stream is finer than the bus can emit.
- Reads blackboard: `proprio.*` snapshot, `perception.snap`+`cube_world`, `command.buttons` (dashboard deque), `mode.*`.
- Writes blackboard: `command.arm`/`command.grip` (raw IK targets — NOT smoothed; shaper is now IO-side), `mode.*` FSM flags, `perception.ctx.*` (tcp_prior/need_align/cap_req), `debug.*` telemetry + `dash.update(**snap)`.
- Owns: auto-z, `deep_reach_step`/`_gimbal_lookat_ik`/`ik_oriented`, `_danger`, yaw-latch, all FSM latches.

**Dashboard Thread** — unchanged (`DebugDashboard`, `debug_dashboard.py`). Already a working blackboard for buttons/telemetry; keep it. Its `_cmds` deque is the FSM input; its `_state`/`_frames` are debug sinks.

---

## 2. BLACKBOARD SCHEMA

One `Blackboard` class, split into **named sections** each with its own lock (finer than `DebugDashboard`'s single `_lock`, so IO writes don't serialize behind control telemetry). Each section exposes `publish(**kw)` (writer, `dict.update` under its lock) and `snapshot()` (reader, returns a **copy** — dict/list copy, not reference — exactly like `AsyncPerception.snapshot()` at `:571`). Every producer section carries a `ts` (freshness) and, where a thread can die, a `dead`/`alive` flag — because the stale/dead fail-stop at `debug_deepreach.py:1447-1460` is a **safety contract** and must survive the split.

### proprio (writer: IO thread — rate: bus)
| field | type | readers | notes |
|---|---|---|---|
| `arm_qpos` | `float[6]` | Control | from `read_proprio` (`robot_io.py:240`) |
| `arm_qvel` | `float[6]` | Control | **finite-diff, rate-sensitive** (hazard R4); IO must own the diff |
| `grip_qpos`/`grip_qvel` | `float[2]` | Control | |
| `arm_torque` | `float[6]` | Control, debug | |
| `grip_torque` | `float` | IO(force-servo), Control | drives `grip_force_servo` + `is_grasped` |
| `is_grasped` | `bool` | Control | was `io.is_grasped`; now written atomically **with** proprio (hazard R7) |
| `ts` | `float` | Control | freshness for "hold at measured pose" idiom (hazard C6) |

### perception (writer: Vision thread — rate: cam) — **reuse existing `_snap`**
| field | type | readers | notes |
|---|---|---|---|
| `snap` | dict (wun,wvn,wvis,wdepth,wcos4,wsin4, s*, stopvn, gap_front, cube_live, ov_q/score, tm) | Control | verbatim the current `_snap` |
| `cube_world`/`cube_raw` | `float[2]` | Control | frozen captured cube (`_cube`/`_cube_raw`) |
| `ts`/`hz`/`dead` | float/float/bool | Control | **the safety freshness+liveness contract** (hazard V-dead) |

### perception.ctx (writer: Control thread — rate: control) — control→vision back-channel (existing `set_ctx`)
| field | type | reader | notes |
|---|---|---|---|
| `tcp_prior` | `float[3]` | Vision | front-gripper-Z search prior (`_tcp`) |
| `need_align` | `bool` | Vision | gates the heavy ~89ms align-seg |
| `cap_req` | `bool` | Vision | one-shot capture handshake |
| `ov_queries` | `list[str]` | Vision | live OWLv2 query from dashboard `ovquery` |

### command (writer: Control thread — rate: control; reader: IO thread)
| field | type | notes |
|---|---|---|
| `arm` | `float[6]` | **RAW IK target** (unshapened). IO applies shaper/LPF/chunk. |
| `arm_snap` | `bool` | request shaper `reset()` (mode change / "hold now") — replaces `send_arm(..., snap=True)` |
| `grip` | `float` | persistent grip setpoint (`grip_cmd`); IO fuses with arm in one `io.command` (hazard R-fuse) |
| `grip_mode` | enum `{hold, force_servo}` | force-servo closes the torque loop IO-side to `grip_torque_target` |
| `grip_torque_target` | `float` | ≤ `GRIP_TORQUE_MAX 1.2` |
| `seq` | int | monotonic; IO logs last-consumed seq so you can see dropped/duplicated commands |

### mode / FSM (writer: Control thread; some read by IO) — the FSM lives entirely in Control
| field | type | notes |
|---|---|---|
| `mode` | enum `{idle, run, align, follow, approach, grip}` | one active servo (preserves the single-servo-per-tick semantics) |
| `estop` | bool | IO reads; **priority abort** (hazard R-estop) |
| `io_task` | `{name: initial_liftup\|comeback\|none, args}` | Control requests a scripted blocking move; **IO runs it**, Control keeps ticking |
| `io_task_state` | enum `{idle, busy, done}` | Control gates: while `busy`, Control suppresses live `command.arm` writes so it doesn't fight the script (hazard C-block) |

### debug (writer: all threads; reader: Dashboard) — keep `DebugDashboard` as-is
- Control assembles the big telemetry snap (`debug_deepreach.py:1533-1569`) and calls `dash.update(**snap)`. IO adds `io.latency_ms`, `io.bus_hz`, `io.last_seq`. Vision keeps `dash.update_frame`. Per-producer sub-dicts (`gates.io`, `gates.perc`, `gates.ctrl`) avoid the cross-writer merge race noted in the dashboard map.

**Thread-safety approach:** per-section `threading.Lock`; writers do `with lock: dict.update`; readers do `with lock: return copy`. This is the proven `AsyncPerception` pattern. No lock-free ring buffers — rates are low (≤100Hz) and copies are cheap. The nested `tm` dict must be **deep-copied** (the existing shallow-copy note in the map is a latent bug to fix during the move).

---

## 3. FILE LAYOUT

New files, all under `deploy/`. Extractions are **move-with-thin-shim**: `debug_deepreach.py` keeps working by importing from the new modules until the last step.

```
deploy/
  blackboard.py        NEW  Blackboard + per-section publish/snapshot; Proprio dataclass moves here or is imported
  vision_thread.py     NEW  AsyncPerception moved out of debug_deepreach.py (515-709) + wrist/front detector
                            dispatchers (wrist_uvn_depth:1219, _align_seg:212, correct_top_xy:275,
                            annotate_*). Reads bb.perception.ctx, writes bb.perception.
  io_thread.py         NEW  RobotIoThread wrapping DamiaoRobotIO. OWNS JointSlew shaper + cmd_lpf +
                            chunk_buf (moved from send_arm closure 1300-1331), grip_force_servo loop,
                            scripted io_task runner (initial_liftup/comeback/move_to). read→write bus loop.
  control_thread.py    NEW  The FSM + IK/FK loop (1399-2130 minus IO/vision). Owns `base`. Contains
                            auto-z block, yaw-latch, command handler if/elif (1602-1834), mode servo
                            ladder (1835-2125), _return_to_rest (now non-blocking: posts io_task).
  kinematics.py        NEW  _gimbal_lookat_ik (427), _danger (490), _project_wrist_unvn (400) moved out
                            of the deploy script into a real module (they take `base` as arg today).
                            WRIST_UN/VN_A/B calib globals move here as module state with an explicit setter.
  run_deploy.py        NEW  Thin entrypoint: arg parse (was 716-1095), construct Blackboard + base + io +
                            dash, spawn the 3 threads, join, teardown (estop FIRST, then join vision).
  robot_io.py          KEEP DamiaoRobotIO unchanged (single-thread-use class); io_thread.py serializes it.
  debug_dashboard.py   KEEP unchanged (already the blackboard reference model).
  debug_deepreach.py   SHRINKS to a compat shim (or is deleted once run_deploy.py is validated).

src/rebot_sim2real/rl/base_controller.py
  Small surgical change only: make deep_reach_step take look_at as an ARG (remove the
  save/restore of self.reach_look_at at 318-321 → reentrancy, hazard K-reentrant).
  cube_top_z stays an attribute but is written ONLY by the control thread (single owner).
```

What moves where, precisely:
- **Vision (515-709, 1219, 212, 275, 235/296)** → `vision_thread.py`. The **sync inline fallback (1471-1492)** is dropped/kept behind a flag — but note hazard V-divergent: the sync path is not feature-equal, so make async the only path and delete sync, or explicitly reimplement it.
- **`send_arm` closure (1300-1331)** splits: raw target → `command.arm` (control side); shaper/LPF/chunk + `io.command` → `io_thread.py`.
- **Auto-z (1412-1421), yaw-latch (1493-1521), control-law (1570-1598), command handler (1602-1834), mode ladder (1835-2125)** → `control_thread.py`.
- **`_return_to_rest` (1378)** → `control_thread.py` but rewritten to post `mode.io_task=initial_liftup` instead of calling the blocking `io.initial_liftup` inline.

---

## 4. DATA FLOW PER CONTROL TICK

The old 13-step synchronous tick (map's `per_tick_steps`) splits into three independent loops. Ordering is now *eventual* via the blackboard, not lock-step.

**IO thread loop (bus rate):**
1. `pr = io.read_proprio()` → `bb.proprio.publish(arm_qpos, arm_qvel(finite-diff), grip_*, torques, is_grasped, ts=now)`. *(record current position)*
2. `cmd = bb.command.snapshot()`. If `bb.mode.io_task` pending → run scripted move to completion (set `io_task_state=busy→done`), skip live command this cycle.
3. Else: `q_out = shaper.step(cmd.arm, loop_dt_io)` (+ LPF/chunk; reset if `cmd.arm_snap`); if `cmd.grip_mode==force_servo` → `grip = io.grip_force_servo(grip, pr.grip_torque, cmd.grip_torque_target)`; `io.command(q_out, grip)`. *(robot movement command)*
4. Publish `bb.debug` io latency.

**Control thread loop (~100Hz — the FSM):**
1. `pr = bb.proprio.snapshot()`; `tcp_w = base.fk_world(pr.arm_qpos)`, `tcp_io = io_fk(pr.arm_qpos)` — see hazard R-2FK on which FK the control thread runs.
2. **Auto-z:** `fk_bias_ema` update, `base.cube_top_z = plane_z - z_bias` (single owner now). Note: EMA tau must be re-derived for the control rate, not the old fused-loop rate (hazard C-rate).
3. `perc = bb.perception.snapshot()`; publish `bb.perception.ctx(tcp_prior=tcp_w, need_align=...)`; **stale/dead check** (`perc.ts` age>2.0s or `perc.dead`) → set `mode=idle` + freeze.
4. Yaw-latch (ts-gated on `perc.snap['ts']` — one accumulate per NEW perception frame, hazard C-rate).
5. Control-law precompute: `q_cmd, at_cube = base.deep_reach_step(pr.arm_qpos, cube_xy, wun,wvn,wvis, look_at=..., jaw_fwd=...)`.
6. `dash.update(**snap)`.
7. **FSM event:** `c = dash.poll_cmd()`. If `c` → command handler (may set `mode`, may post `io_task`). **Preserve "command preempts servo this tick"**: if a command was serviced, skip the servo step (hazard C-ladder — decide explicitly; recommend keep).
8. **Mode servo:** based on `bb.mode.mode`, write `bb.command.publish(arm=q_cmd, grip=grip_cmd, arm_snap=...)`. This is the *only* thing that moves the arm.

**Vision thread loop (cam rate):** unchanged from the existing 11-step (map) — reads `bb.perception.ctx`, writes `bb.perception`.

The key change: Control writes a **target** to the blackboard; IO **decides when/how** to emit it to the bus (with shaping). "Hold at measured pose" (`send_arm(pr.arm_qpos, snap=True)`) becomes `command.publish(arm=bb.proprio.snapshot().arm_qpos, arm_snap=True)` — reading the IO thread's *latest* published joint state, not a stale local copy (hazard C6).

---

## 5. DEBUG LOGGING PLAN

Stdlib `logging`, one named logger per thread, gated by `--debug`. Each thread logs its stage so you can bisect *where* a bad grasp originates without a debugger, then set a PyCharm breakpoint in the one implicated file.

```python
# run_deploy.py
fmt = "%(asctime)s.%(msecs)03d [%(threadName)-8s] %(name)-10s %(levelname)-5s %(message)s"
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format=fmt, datefmt="%H:%M:%S")
log_io   = logging.getLogger("io")
log_vis  = logging.getLogger("vision")
log_ctrl = logging.getLogger("control")
```
Threads named `IO`/`Vision`/`Control`/`HTTP` (`threading.current_thread().name`), so every line is attributable.

**IO logger** (every write, DEBUG): `read qpos=[...] qvel=[...] grip_tau=1.86 | cmd.seq=812 arm=[...] shaped=[...] grip=0.03 | rd=4.1ms wr=3.8ms bus=118Hz`. On force-servo: `grip_servo tau=0.62→target 0.80 cmd 0.031→0.029`. On scripted: `io_task=initial_liftup START/step k/DONE`. This isolates: motor under-command, bus stalls, shaper overshoot, over-torque.

**Vision logger** (per published frame, DEBUG): `front cube_uv=(412,388) world=(-0.14,0.02) src=owlv2 score=0.71 | wrist un,vn=(0.02,-0.31) vis=0.9 depth=0.11 yaw=12° src=cnn | align skip | gz gap=0.006 | hz=14.2 tm{det_w:31 align:0 front:44 gz:9}`. Isolates: detector picking wrong blob, low conf, rate collapse, which stage eats the frame budget (`tm`).

**Control logger** (INFO on mode change + gate flips; DEBUG every tick): `mode RUN | cube_tgt=(-0.14,0.02) off=(0,0) track=off | zgap_io=0.031→goal xy_err=0.008 reached=F | yaw_lat=0.6 | q_cmd=[...]`. On decisions: `LATCH yaw seg=8°`, `DESCEND commit`, `DANGER xy_resid 0.04 → return_to_rest`, `STALE perc age 2.3s → HALT`. Isolates: wrong target, gate not firing, danger tripping, stale fail-stop.

**How it helps in PyCharm:** the log tells you the failing *stage and thread*; you open that one file (`io_thread.py` / `vision_thread.py` / `control_thread.py`), set a breakpoint, and — because each thread has an isolated loop reading/writing only the blackboard — you can pause it while the others run, or stub the blackboard section with canned values to reproduce offline. The old monolith made this impossible (one loop, all concerns interleaved).

Add a `--log-stage` filter (`io,vision,control`) and a rate-limited DEBUG (log every Nth tick) so 100Hz control doesn't flood.

---

## 6. MIGRATION PATH (incremental, RUN grasp works at every step)

Each step ends with the **same HW smoke test**: Capture → RUN → descend → (visual) grasp-ready with `zgap_io≈0`. If a step regresses it, revert just that step.

**Step 0 — Introduce `blackboard.py` as a pass-through, no thread split.** Create `Blackboard` and route the *existing* single-loop reads/writes through it (proprio publish/snapshot, command publish/snapshot) while still calling everything inline. Zero behavior change; validates the schema and copy semantics. Validate: RUN grasp unchanged.

**Step 1 — Extract `kinematics.py`** (`_gimbal_lookat_ik`, `_danger`, `_project_wrist_unvn`, `WRIST_*` calib). Pure moves, import back into `debug_deepreach.py`. Fix `deep_reach_step` reentrancy (look_at as arg) — this is a prerequisite for any threading and is low-risk (single caller today). Validate: RUN + approach unchanged.

**Step 2 — Extract `vision_thread.py`.** `AsyncPerception` is *already a thread*; just move the class + detector dispatchers to their own file and wire via `bb.perception`/`bb.perception.ctx` instead of the internal `_snap`/`_tcp`. **Delete or explicitly reimplement the sync inline path** (hazard V-divergent) — recommend async-only, since it's the HW-validated path. Validate: dashboard frames + RUN unchanged.

**Step 3 — Extract `io_thread.py` (highest risk — do carefully).** Move `send_arm`'s shaper/LPF/chunk + `io.command` and `grip_force_servo` and the scripted moves into the IO thread. Control now writes `command.arm`; IO shapes+sends. This is where the bus-serialization (R1) and shaper-placement (C1) hazards land. **Neutralize R1:** IO is ONE thread doing read-then-write; robot_io.py stays untouched and single-threaded-use. **Neutralize C1:** shaper steps on the IO loop_dt (its own rate), not control's; verify with `measure_latency.py` that the trapezoid still brakes without overshoot at bus rate. **Neutralize C-block:** `_return_to_rest`/reset/zero become `io_task` posts; Control keeps ticking and gates on `io_task_state==busy`. Validate: RUN grasp + a reset/liftup while STOP is still responsive (the thing the monolith couldn't do).

**Step 4 — Extract `control_thread.py` + `run_deploy.py`.** Move the FSM/servo loop. `base` owned solely here; `cube_top_z` single-writer. Preserve the command-preempts-servo ladder decision (C-ladder). Delete `debug_deepreach.py`. Validate: full RUN→Align→Grip→Lift cycle on HW.

**Step 5 — Re-tune rate-dependent constants** (`fk_bias_ema` tau, yaw-latch ts-gating, qvel finite-diff): now that IO reads at bus rate and control at 100Hz, verify the policy-visible `arm_qvel` magnitude and the auto-z EMA match the single-loop baseline (hazard R4/C-rate). This is the subtlest regression risk and deserves an explicit A/B against a logged baseline run.

**Highest-risk hazards → neutralization summary:**
| Hazard | Risk | Neutralization |
|---|---|---|
| R1 single bus, no locks | serial corruption | IO = one thread, read-then-write alternating; robot_io stays single-use |
| C1 shaper straddles IO/control | overshoot / rate bug | shaper moves fully to IO, steps on IO loop_dt; re-verify with measure_latency |
| R4 qvel finite-diff rate-coupled | policy OOD | IO owns the diff over real elapsed; preserve `_prev_arm=None` reset on scripted handoff |
| C-block scripted moves freeze loop | STOP/stale starved | scripted moves = `io_task` on IO thread; control gates on `io_task_state` |
| V-dead / stale fail-stop | safety | `perception.ts`+`dead` in blackboard; control halt logic preserved verbatim |
| R-estop | estop stuck behind blocking read | `mode.estop` checked at top of IO loop; estop = `io.disconnect()` runs before vision join |
| K-reentrant deep_reach_step | flag corruption | look_at as arg (Step 1) |
| C-rate 100Hz vs 30Hz | silent behavior change | ts-gate all per-tick consumption; re-derive EMA taus (Step 5) |

---

## 7. RISKS / OPEN QUESTIONS

1. **Two FKs (R-2FK).** `base.fk_world` and `io.fk_tcp` disagree ~40mm and the auto-z `fk_bias` hack reconciles them per tick. In the split, does the control thread compute `io_fk` itself (needs a *copy* of the pinocchio model — `fkdata` is mutated in-place, not shareable), or does IO publish `tcp_io` alongside proprio? **Recommend IO publishes `proprio.tcp_io`** so control never touches IO's pinocchio data. Confirm the ~1-tick staleness of `tcp_io` vs fresh control `tcp_w` doesn't drift the auto-z EMA.

2. **Command-preempts-servo (C-ladder).** Today a button tick skips the servo. Keep this, or let servo always run? At 100Hz control the semantics matter less, but reset/zero handlers assume no servo runs. **Please confirm** you want to preserve it.

3. **Control rate when IO is decoupled.** Control can now run at 100Hz emitting fine setpoints, but the bus + POS_VEL vlim=32 still gate what reaches the motor. Is 100Hz control genuinely useful, or should control run at ~30Hz (perception rate) to avoid re-emitting identical targets? Affects shaper tuning.

4. **`--track` / cube-of-record ownership.** Capture freezes `cube_world` in Vision, but `--track`, `cube_x_off/y_off`, and `track_anchor` live in control, and approach uses `perc['cube_live']` directly. Which thread owns the target-of-record? **Recommend: Vision owns raw estimates; Control owns the target** (applies track deadband + offsets), published as a `mode.cube_target` for debug clarity.

5. **calib_wrist (~9s inline mini-loop).** It bypasses the tick entirely (its own send_arm+sleep+snapshot loop). Model it as a dedicated `io_task` + control sub-state, or leave it as a special blocking maintenance mode outside the normal FSM? Lowest-priority since it's not on the grasp path.

6. **Per-section vs single lock.** I propose per-section locks; the dashboard uses one coarse lock and it's fine at these rates. Single lock is simpler and proven — worth the finer locks only if IO-write vs control-telemetry contention shows up. **Default to single lock unless profiling says otherwise.**

7. **GPU model sharing.** Vision stays one thread, so the shared `OpenVocabModel` + `CubeDetectorIO` instances remain safe (sequential). Do NOT split front/wrist into sub-threads without adding model locking / a GPU inference queue (already hangs under contention). This constrains "Vision" to remain a single thread.

Relevant files to review before implementation: `~/workspace/rebot_sim2real/deploy/debug_deepreach.py`, `~/workspace/rebot_sim2real/deploy/robot_io.py`, `~/workspace/rebot_sim2real/deploy/debug_dashboard.py`, `~/workspace/rebot_sim2real/src/rebot_sim2real/rl/base_controller.py`.

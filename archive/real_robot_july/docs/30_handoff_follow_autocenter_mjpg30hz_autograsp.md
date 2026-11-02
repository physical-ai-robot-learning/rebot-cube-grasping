# Handoff — (key) Follow wrist AUTO-CENTER (measured-Jacobian closed loop) + perception 30Hz via MJPG + AUTO-GRASP (commit-and-descend) in progress + io-thread/dashboard bug fixes (2026-07-20/21, session 19)

> Continuation of docs/29 (S18: 3-thread refactor + Follow descend FSM). **Very long session.** Start = docs/29 §6-1 Follow→Rest reconfirmation → io-thread bug fix → full redesign of wrist centering (user-driven iterative tuning) → perception 30Hz → started grasp automation. **Branch `perception-topwrist-coop`. 3 commits this session (`69ac4ba`, `9147e23`, `0cb5326`, all unpushed) + 3 uncommitted files (MJPG, look-at, AUTO-GRASP).**
>
> **One line**: (1) **Follow→Rest io-thread bug fix** (stale bb.command replayed after the scripted io_task → Reset returns to the Follow position; bb.command resynced in `_run_io_task`). (2) **dashboard /state numpy crash fix** (stray numpy → 500 → "?"). (3) (key) **Follow wrist AUTO-CENTER completed and committed**: deterministic cam-offset + **HW-measured decoupled Jacobian (dvn/dfwd=−19, dun/dlat=−9.5)** closed loop (measured sign → cannot run away) + ts-gate + hysteresis + rate-independent gain + grasp-framing target (0,0.44). (4) (key) **MJPG → perception ~10Hz (unstable) → 30Hz** (bottleneck = USB2 bandwidth, not GPU/model; both cams support MJPG = the handoff's "not supported" was wrong; K unchanged). (5) **AUTO-GRASP commit-and-descend in progress** (⊙centered = commit the optimal grasp xy → straight descent → close at depth; v4 just deployed, not HW-verified). (6) wrist-calib fit **failed** (unidentifiable, contaminated by front bias) → gave up on calibration and pivoted to an empirical approach. Look-at (gimbal) oscillates at 30Hz + latency → on hold.

---

## 0. Git / robot state
- Branch `perception-topwrist-coop`. **3 commits this session (all unpushed)**: `69ac4ba` (io-thread Reset fix) → `9147e23` (dashboard numpy-safe) → `0cb5326` (Follow auto-center feature).
- **3 uncommitted files** (later work, HW tuning in progress): `deploy/debug_deepreach.py` (MJPG flag + AUTO-GRASP sequencer + look-at + cam-offset tuning) · `deploy/debug_dashboard.py` (AUTO-GRASP/cam-center/CALIB buttons + generic cmd param) · `deploy/vision_thread.py` (MJPG FOURCC). **Uncommitted because AUTO-GRASP, MJPG and look-at are still being verified** — MJPG is settled (fine to commit), AUTO-GRASP is in HW iteration.
- Warning: **new untracked**: `deploy/fit_wrist_calib_constrained.py` (BA fit diagnostic script, used to establish why calibration failed). docs/20-30 still untracked (handoffs, need a separate commit).
- **Robot**: deploy running (port 8899, MJPG 30Hz). Camera symlinks `/dev/video_front`→video3 (udev re-enumeration absorbed automatically), `/dev/video_wrist`→video0. `.venv-deploy`. Warning: **when deploy is launched as a detached background job it intermittently exits** ("done torque off", unrelated to code) — relaunch repeatedly. Command below.

## 1. (key) Follow wrist AUTO-CENTER (headline, commit `0cb5326`)
Problem: when descending straight-down with the TCP over the cube, **the wrist cam looks ~22° ahead of the TCP + lens offset, so the cube leaves the bottom of the field of view**. User request = keep the cube centered using the wrist.
- (key) **Deterministic cam-offset** (`--follow-cam-fwd/lat`): shifts the Follow hover along the **gripper forward/lateral axes (rotating with J1)** → no feedback so no runaway, live-tunable (dashboard cam+/−).
- (key) **Closed-loop auto-center** (`--follow-auto-center`, default ON): drives the wrist (u,v) to a target. **HW-measured decoupled Jacobian** (2026-07-20 controlled fwd/lat sweep: +2cm cam-fwd → Δvn −0.38, i.e. `dvn/dfwd=−19/m`, un unchanged; +2cm cam-lat → Δun −0.19, i.e. `dun/dlat=−9.5/m`, vn unchanged). **Measured sign = stable Newton step → cannot run away** (the initial runaway was caused by a WRONG-sign geometry model — the u-sign was flipped, confirmed with pearson −0.69). Constants `FOLLOW_DVN_DFWD/DUN_DLAT`.
- **Triple defense against rate coupling** (repeatedly warned in handoffs): (a) **ts-gate** (update only on fresh ~7Hz frames, not every 30Hz tick → prevents 4x over-correction) (b) **hysteresis latch** (centered → stop when |err| < margin, resume when > 2×margin → removes boundary chatter) (c) **rate-independent gain** (step ∝ dt_perc/0.1 → even with perc 10→30Hz the per-second integration rate stays constant = removes slow back-and-forth servoing).
- **Grasp-framing target** (`--follow-center-vn 0.44`): puts the cube slightly below the wrist center (jaw landing point) — (key) **user confirmed: "⊙CENTERED = optimal grasp position"**.
- **HW-tuned defaults locked in**: auto-center ON, gain 0.1, margin 0.10, vn 0.44, cam-off seed fwd 0.03/lat −0.02. **HW-verified** (stops after ⊙CENTERED, servoing 0).
- **RISE separated**: separate `follow_front` (front-glued) → prevents front-vs-wrist bias from being mistaken for a fake move and causing rise/descend cycles.
- **Dashboard** (`debug_dashboard.py`, uncommitted additions): dual-cam cube XY (FRONT ray-plane / WRIST optical-unproject / Δ) · TCP FK (x,y,z) base_ctrl + robot_io · cam-center +/− · AUTO-GRASP · CALIB · Look-at toggle buttons · generic `action?k=v` cmd parsing.

## 2. (key) MJPG → perception 30Hz (uncommitted, settled)
- **Timing established** (`/state.gates.tm`): bottleneck = **camera read** (rd_w 48~76ms, rd_f 32~133ms fluctuating), detection 3~4ms, independent of GPU/model. Cause = **uncompressed YUYV 720p×2 saturating USB2 (480Mbps) bandwidth** → each contends at ~10fps.
- (key) **Both cameras support MJPG** (v4l2 fmt [1]) — the handoff's "MJPG not supported" was **wrong**. `--mjpg` (default ON): FOURCC MJPG → compressed transfer ~10x smaller → **perc fixed at 29.9Hz** (rd_w 76→**2ms**). **Resolution stays 720p → K/calibration unchanged** (OpenCV decodes MJPG in ~1-3ms). `vision_thread._read_rgb` + `--mjpg` flag.
- Side effect: control 30Hz = perc 30Hz matched → ts-gated centering updates at 30Hz → snappy tracking. GPU memory / smaller models are irrelevant (detection is not the bottleneck).

## 3. AUTO-GRASP one-shot (uncommitted, HW iteration in progress — key item for next session)
User request: one-shot button for Align + vertical descent + Grasp. **Old Align/Final chain discarded** (3 failures: align drifts, descent misses, grip closes on nothing). → Redesigned as an **extension of auto-center**.
- (key) **v4 (just deployed, not HW-verified) = commit-and-descend**: AUTO-GRASP **starts only when ⊙CENTERED** (= optimal grasp xy) → **XY + cam-offset FREEZE** (auto-center/tracking stopped during descent: movement shakes the framing and it cannot reconverge) → **pure straight descent** (over the committed xy) → **grip gate = exact `z_gap_io ≤ grasp_zgap_io+0.012` only** (ac_centered not required) → force-servo close. Sequencer (top of control loop, phase None/descend/grasp) + safety abort (stale / over-plunge `zgap<goal−0.03` / Follow interrupted / 30s). Manual commands (stop/reset/run/…) cancel the sequence.
- **Failure history (lessons)**: v1 Align→Final (discarded the auto-center framing) → v2 tied the floor to ac_centered (toggled −0.05↔0.07 → **descend/rise oscillation**) → v3 grip gate on ac_centered (**⊙ latched 0 times during descent** → grip never fired → **over-plunge** zgap −0.03) → v4 commit-and-descend. Established with a **trajectory sampler** (`scratchpad/ag_trace*.txt`, /state polled every 0.35s).
- **J6 yaw = not applied (fixed as in v1)**: with the current `--wrist-openvocab` there is no reliable yaw source (seg/CNN yaw 0/flaky). Yaw is separate work (enable the seg detector in Follow or a dedicated yaw detector).
- **Next**: v4 HW test → whether it descends straight and closes at depth (without over-plunge). If yes, add J6 yaw.

## 4. io-thread / dashboard bug fixes (committed)
- **`69ac4ba` io-thread Reset**: with `--io-thread` ON, Rest during Follow → heads to REST then **returns to and locks at the Follow position**. Root cause = after a scripted move (io_task) the old Follow target remains stale in `bb.command.arm` → the IO fallthrough replays it. FIX = `RobotIoThread._run_io_task` resyncs bb.command to the reached pose after the move (arm_snap, grip_mode hold). Reproduced and passed in mock-bus smoke. **HW-verified.**
- **`9147e23` dashboard numpy**: `json.dumps(dash._state)` had no default, so stray numpy (0-d ndarray etc.) → 500 → "?". Added a numpy-safe default. `/state` 200 confirmed.

## 5. wrist-calib attempt → failed → abandoned (empirical pivot)
- CALIB sweep (`calib_wrist`, J1+J4 diagonal) + `fit_wrist_calib.py` fit → **non-physical garbage** (lens 0.4~2m, focal 0.02~3x, RMS floor 0.15~0.20). Even with `fit_wrist_calib_constrained.py` fixing axes + bounds + **bundle adjustment** (cube position as free variable) it failed (BA moved the cube 18~31cm = overfit, ⊙data inconsistent, v-corr sign flips per position).
- **Root cause = the data is fundamentally inconsistent with the camera model** (the front-estimated cube reference is biased per position + accumulation/mixing + observation noise). In sim it worked because the cube is privileged, but the real front bias breaks the fit.
- **Conclusion: abandon the calibration fit → empirical deterministic + measured Jacobian (§1) is the practical winner.** Salvaged: **u-sign flip** (cause of the runaway); wrist detection works well with OWLv2/redmask.
- For reference, user measurements: optical axis 22° tilt (matches the model), lens = TCP + (−8cm y, +7.5cm z) (model Y off by ~11cm — frame sign ambiguous). The sd3 CNN missed an obvious red cube (sim2real gap) → **wrist uses OWLv2 or redmask** (`--wrist-detector none`).

## 6. rebot_eval GPU mystery = Ray jobs (user decision: leave it)
- `train_sac_det_xattn.py`/`calib_ui_official.py` reappear automatically = run by the **Ray JobSupervisor** (25 cluster submissions from a teammate + a calib_ui job from the local `<workstation>` account). **`kill` = whack-a-mole** (Ray restarts/queues), **must be stopped with `ray job stop <id>`**. User: leave them (coexisting with deploy is OK, GPU has headroom after MJPG). Warning: `nvidia-smi` before deploy.

## 7. Gotchas
- (key) **MJPG is the key to 30Hz** (YUYV 720p×2 = USB2 saturation). Both cams support MJPG. `--no-mjpg` = raw.
- (key) **auto-center runaway history = WRONG sign**; now settled with the measured Jacobian (dvn/dfwd=−19, dun/dlat=−9.5). If the rate changes the gain adapts automatically since it is rate-independent, but **look-at oscillates at 30Hz + command latency** (on hold).
- (key) **AUTO-GRASP = commit-and-descend**: freeze XY at ⊙centered, then straight descent (no re-centering during descent — movement shakes the framing → over-plunge). Grip = z_gap_io depth only.
- **Suppress RISE during descent** (moved=False during autograsp); do not tie the floor to ac_centered (oscillation).
- **io-thread OFF = byte-identical** (if problems, remove the flag → `a4f9ec5`). A scripted io_task must resync bb.command.
- **A deploy launched as a detached background job exits intermittently** → relaunch every time + confirm the dash is UP. Launching it directly by the user is more stable.

## Run (next session)
```
.venv-deploy/bin/python deploy/debug_deepreach.py --detector-device cuda \
  --front-openvocab --wrist-openvocab --front-roi 0.4,1.0,0.0,1.0 --track \
  --grip-force 0.05 --grip-torque-target 0.8 --vmax 5 --amax 8 \
  --x-corr 0.0235 --safety-reach-resid 0.15 --rate 30 --front-ov-period 0.5 \
  --cmd-lpf-tau 0.15 --action-chunk 20 --chunk-decay 0.9 \
  --follow-low 0.07 --follow-descend-step 0.004 --follow-move 0.04 --io-thread --lookat-j5-rate 0.05
```
(auto-center/MJPG/vn0.44 are all default ON — no flags needed.) Flow: **Capture → Follow (wait for ⊙CENTERED) → AUTO-GRASP ▶▶** (straight descent → close at grasp depth). Live = `curl localhost:8899/state`. Next key item = **AUTO-GRASP v4 HW verification** (closes without over-plunge?) → J6 yaw. Details: docs/29.

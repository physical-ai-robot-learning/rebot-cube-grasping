# Handoff — (key) gimbal look-at approach implemented + perception-stale identified and fixed + gripper recalibration + KP_APR reverted to 54 + front HSV+ROI (2026-07-15, session 15)

> Continuation of docs/24 (first part of session 15: KP_APR config A + initial perception-stale diagnosis). **A long real-robot session**, 6 workstreams. All **uncommitted** (branch perception-topwrist-coop). Collaborated with the user in front of the real robot (HW).
>
> **One line**: perception "stale" was identified as a **reporting bug** (startup msg lingering) and fixed (actual perception at a normal 5-8 Hz). Gripper over-opening + failure to grasp the cube identified and recalibrated. front 5k CNN **failed to transfer to the real robot** (25 cm jumps) → reverted to HSV+ROI. KP_APR **reverted to 54 (RAM)** at the user's request. (key) **Core new item = gimbal look-at approach mode** (null-space IK makes the wrist aim at the cube while descending) — offline perfect (0 mm/0°), **HW test in progress/pending**.

---

## 0. Robot/Git state
- Warning: **At session end, an `approach` test deploy was still running** (pid, `--track --detector-device cuda --front-roi 0.4,1.0,0.0,1.0 --grip-force 0.05 --grip-torque-target 0.8 --amax 6 --vmax 4`). **Before starting the next session, confirm it has been terminated** (Ctrl-C→estop) + torque off + KP_APR READ.
- **KP_APR = 54 (RAM, factory default)**. **Flash holds config A** (pan500/lift500/elbow400/wrist_flex150/yaw200/roll400) → **config A is restored on power reboot**. One-line config A restore: `damiao_params --set-kp-apr <v>` for each joint.
- **All uncommitted**. Files changed this session: `deploy/debug_deepreach.py` (substantial: approach mode + perception-stale fix + detector-device + front-roi), `src/rebot_sim2real/perception.py` (red_mask roi/s_min), `deploy/calib_grip.py` (--close-max/step/on-cube), `deploy/debug_dashboard.py` (Approach button), `deploy/joint_map.npz` (grip_open_deg -59, grip_closed_deg 60), `ckpt/cube_detector_front_5k.pt` (new).

## 1. (key) gimbal look-at APPROACH mode (new, headline) — offline verified, waiting for HW
**Goal (user)**: front rough xy → correct xy via wrist look-at → approach while maintaining look-at → track when the cube moves. All CNNs on GPU.

**Warning: key finding (reversed twice)**: at first `ik_oriented(look_at)` diverged (TCP 100-170 mm off), so I concluded "look-at is geometrically impossible" → **that was wrong**. Re-verified after the user asked "like a gimbal?": joint margin **J4 60-70° / J5 90°** (look-at tilt needs only ~30°) is **sufficient**. The cause was the crude **coupled DLS** in `ik_oriented` (position + axis least-squares at once → a half-way compromise when they conflict) and a bug in my first gimbal IK, not geometric impossibility.
- **Solution = `_gimbal_lookat_ik` (new module function in debug_deepreach)**: task-priority **null-space** — position is primary (damped pinv), cam-aim is secondary as a **projected gradient** in the position null space (aims while holding the TCP). Offline (`scratchpad/test_gimbal_ik2.py`): **at dz≥6 cm, tcp 0 mm / cam 0°**; only fails at dz<~2 cm (just before grasp, cam beside the cube = extreme angle) → switches to straight-down. Speed: 80 iters 96 ms/call (control loop ~10 Hz during approach; OK since the descent is slow).

**Implementation (opt-in, baseline untouched)**: dashboard/curl **`approach`** command + branch `elif approaching:`.
1. front rough xy (HSV+ROI, `--track`, deadband).
2. wrist (u,v)→world slow fuse (±12 cm agreement gate, `--lookat-fuse` 0.2).
3. descend gate: when xy is aligned (`--lookat-xy-eps` 0.02), descend by `--lookat-descend-step` (3 mm) each step; on a gross miss (>6×) return to hover.
4. IK: dz>`--lookat-taper` (0.05) → **gimbal look-at** (cam aims at the cube), below that → straight-down. Stops at grasp-ready (surf+`--lookat-grasp-clear` 0.01) → hands off to Align/close.
- flags: `--lookat-hover/grasp-clear/descend-step/xy-eps/deadband/fuse/iters(80)/taper`.
- fail-stop (perception stale) · RUN · close · follow halt approach (mutually exclusive).
**(key) HW result (2026-07-15)**: **look-at is unstable at low TCP** — near the cube the aim tilt becomes large and changes quickly, and combined with **KP_APR 54's ~700 ms latency** the wrist lags behind → oscillation (offline static checks cannot see this dynamic latency). **Stable at high TCP (hover, small and slow tilt).** → **Fix: `--lookat-taper` default 0.05→0.10** (look-at only up high, straight-down for the low descent). Warning: **fundamental trade-off**: **look-at depth is gated by motor latency** — at 54, look-at is stable only near hover (dz>0.10) (short look-at window). **To run look-at deeper (truly throughout the approach), config A** (fast motors, lower latency) must be restored = the very values the user reverted. I.e. a choice between **54 (stable but shallow look-at) vs config A (deep look-at, but ~1° tremor at rest)**. If it stutters, use `--lookat-iters 40`.

## 2. perception "DEAD/STALE" = reporting bug (identified and fixed)
In `debug_deepreach.py` the stale msg kept appearing, so it seemed RUN/follow did not work — but **the cameras and perception were fine** (frame MD5 kept changing; perc-diag: after iter0 3.5 s [first CNN + CUDA init], **a normal 5-8 Hz**). The real issue = **the msg set during the 3.5 s startup stale period is never cleared in idle (no mode) and lingers**. → **Fix**: (a) clear the msg on recovery (`perc_was_stale`), (b) staleness 0.5→**2 s** (tolerates CPU CNN spikes of ~0.8 s; a true freeze is unbounded so it is still caught), (c) expose **truthful `perc_hz`/`perc_age_ms`** in `/state` (judge health by these, not the msg), (d) `--detector-device` (default cpu; **cuda opt-in**, GPU 3 ms).

## 3. KP_APR (docs/24 config A → reverted to 54)
After config A was finalized and saved, at the user's request **all joints were reverted to 54 (RAM)** (flash keeps config A). Reason not stated (config A residual ~1° "jitter" or factory reset). Warning: **at 54, J2/J3 ~12° oscillation was measured at rest hover** (worse than config A's ~1°) — but that was a vlim=32 fixed-setpoint hold test, so actual deploy (glide/shaper) may differ.

## 4. Gripper recalibration (over-opening + failure to grasp the cube resolved)
- **Over-opening**: `grip_open_deg` -179→**-59** (calib_grip `--open-max 10`).
- **Two root causes of failing to grasp the cube**: (a) the jaw no longer self-closes (mechanical change after 2 breakages; torque does not spike even up to +120°) → **firm-close ≠ cube-close; set grip_closed_deg separately as the cube clamp**. (b) `grip_closed_deg=29` was too shallow, so the force-servo only reached +46.7° (could not reach the cube's +60° clamp, 0.3 cm short) → wrote **grip_closed_deg 29→60** → confirmed it reaches. **`--grip-force 0.05`** (grip_tau 0.93@ratio0.05). Added `--close-max/step/on-cube` to calib_grip.py (hard-coded -8 → dynamic, handles zero-shift). Warning: actual deploy grip behavior not verified (calib ③ + calculation confirmed).

## 5. Front cam: 5k CNN failed to transfer to the real robot → HSV+ROI
`ckpt/cube_detector_front_5k.pt` (retrained, sim val median 3.0 px / IoU 0.92, better than 1500). **BUT garbage on the real robot** (cube_top jumps 25 cm, crosshair off the cube) = sim→real transfer failure (only camera pose aligned, no intrinsics/distortion/roll warp). → **Reverted to HSV** (cube_top perfectly stable). Mitigating false detections on the back of the hand = `--front-roi 0.4,1.0,0.0,1.0` (top 40% = excludes the person) + `--front-red-smin` (rejects low-saturation skin). **To adopt the 5k CNN, v2 is needed: fine-tune on real front images or an intrinsic/distortion warp**.

## 6. Next session actions (priority)
1. **approach (gimbal look-at) HW verification** — descent + tracking + taper stability. If good, full cycle Capture→Approach→Align→close→lift.
2. **Verify actual gripper deploy grip behavior** (does close with `--grip-force 0.05` bite the cube).
3. KP_APR: observe rest behavior at 54 → if oscillation is severe, restore config A or do load-aware re-tuning.
4. **Commit** (HW-verified parts): perception-stale fix + gimbal approach + gripper calib + front ROI + calib_grip improvements. Large, so splitting into logical commits is recommended.
5. front 5k CNN v2 (transfer) or keep as-is (if HSV+ROI is sufficient).

## 7. Gotchas
- **The perception stale msg may be lingering** — judge actual health by `perc_hz`/`perc_age_ms` in `/state` (not the msg).
- **look-at is geometrically possible** (gimbal/null-space) — only impossible just before grasp (dz<~2 cm) (taper). `ik_oriented(look_at)` alone diverges (coupled DLS); use `_gimbal_lookat_ik`.
- **front 5k CNN does not transfer to the real robot** — deploy default is HSV. Turning on the CNN with `--front-detector` causes jumps on the real robot (no transfer).
- **The gripper jaw does not self-close** (mechanical change) — grip_closed_deg is based on the cube clamp (+60), not self-contact.
- **KP_APR 54 in RAM / config A in flash** — after reboot it is config A. Do not confuse them.
- **The offline-verification habit paid off twice this time** (early detection of the look_at divergence + rediscovering that it is in fact possible with a gimbal). Use `scratchpad/test_*.py` before HW.
- compile OK (3 files). approach verified offline only, **not verified on HW**.

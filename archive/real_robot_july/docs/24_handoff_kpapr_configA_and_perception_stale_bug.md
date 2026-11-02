# Handoff — KP_APR (key) config A (re-correction: root fix for J4 oscillation) + 5k detector + (key) perception intermittent-stale bug identified (2026-07-14 evening, session 15)

> Follows docs/23 (session 14, multiclass detector + initial KP_APR tuning); **a consecutive session on the same evening**. The user collaborated in front of the HW, running and observing the real robot.
> **One line**: leaving KP_APR at the session 14 values (J4=500) made **J4 blow up to 24° at the real-robot REST hover** → re-measurement **identified a non-monotonic resonance → config A (J4 150) removed the blow-up, saved to flash**. hover-follow remains unfinished, blocked by a bug where **perception "DEAD/STALE" intermittently false-alarms and keeps halting RUN/follow** (the cameras are fine). The 5k front detector was retrained and evaluated. **All uncommitted.**

---

## 0. Warning: robot safety state (at session end)
- **All motor torque OFF** (explicit estop confirmed), no deploy process, cameras released.
- **KP_APR = config A, saved to flash**: `pan 500 · lift 500 · elbow 400 · wrist_flex 150 · wrist_yaw 200 · wrist_roll 400`. (Expected to persist across power reboot; recommended to do a final check with `deploy/damiao_params.py` READ after reboot.)

## 1. (key) KP_APR re-correction = config A (today's main result)
Leaving the **session 14 values** (pan500/lift500/elbow400/**wrist_flex500**/yaw200/roll400) unchanged and starting deploy, **J4 oscillated violently when moving to the real-robot REST hover** (user observation). This was despite having passed the step/track tests.

**Root cause (scratchpad/j4_hold_test.py = move_to(rest) + 3 s fixed-setpoint hold, per-joint p2p measurement):**
- (key) **step/track are misleading** — clean in a light folded pose, but in the **loaded REST hover** (J2≈-90°, J3≈-87°, J4≈60°, arm extended) the gravity-loaded joints (J2/J3/J4) go into a **limit cycle**.
- (key) **Non-monotonic in KP_APR** (hold p2p): J4 {60→1.42°, **150→0.81° (min)**, 600→**24° (blow-up)**} · J2 {120→5.22°, **500→0.87° (min)**} · J3 {90→7.37°, **400→1.20° (min)**, 600→5.01°}. **Gets worse whether raised or lowered; each joint has a resonance minimum**.
- **all-54 (the historical 'known-good') is actually the worst in this pose** (J2/J3 **12°+**) — the docs/22 grasp+lift was in folded/descending poses, so it did not experience this load.
- **Unloaded joints (J1/J5/J6) are OK at high gain** (J1@500 0.02°). J2/J3 at +5° (up) did not move because gravity does not push the torque → characterized at -5° (gravity direction).
- → **config A = each joint's resonance minimum = the practical optimum for the single KP_APR knob.** Blow-up removed; residual J2 0.87 / J3 1.20 / J4 0.81° (std 0.1-0.17°).

**Residual ~1° limit cycle (= the user's "jittering"):**
- `--arm-vlim` (glide 2.5 vs snap 32) **no change**, `KP_ASR` (0.004→0.02) **no change** → **the floor of POS_VEL**, circumstantially mechanical (backlash: reassembled J4 + loaded J3/J2).
- **Full removal / "smooth and fast" = requires MIT mode (kp/kd damping).** The user asked again (deferred this time). git `c9df753` (2-thread MIT impl, mock verified), `657fdee` (probe) can be reused.
- **Lesson: motor gains must be hold-verified in the actual working pose (loaded). Do not judge from step/track alone.**

## 2. (key) perception "DEAD/STALE" = intermittent false-alarm bug (biggest hover-follow blocker)
**Symptom**: when deploy starts, the `/state` msg says "PERCEPTION DEAD/STALE -> auto/align halted (frozen camera feed)". RUN/Align/Follow do not proceed (arm stuck at rest).
**Diagnosis**: the cameras are **fine** (the user's "the cameras look fine" was correct). The front and wrist frame MD5s keep changing = **thread LIVE, frames updating**. No traceback in the log (not dead).
- Real cause = the **staleness fail-stop** at `debug_deepreach.py:930` `(time.time() - perc["ts"]) > 0.5` **fires intermittently**. One perception-thread iteration **occasionally exceeds 0.5 s** (heavy: 2×**720p** read + wrist CNN + front HSV + front-gripper-Z frame-diff + 2 overlays @ target 30 Hz). → RUN/align repeatedly halt at `line 931` → arm stalls.
- **NOT a camera hang, NOT dead**. Being intermittent, it also recovers (which is why FOLLOW occasionally got down to 3 mm).
**FIX candidates (not applied)**: ① raise the `line 930` threshold 0.5 s→~2 s (simple; trade-off of a weaker fail-stop) ② profile and slim down the perception thread (front read from 720p→lower, skip wrist CNN for follow/RUN, front-gripper-Z only for descend) ③ expose thread hz in /state and measure it. **This must be fixed for the hover-follow/RUN/grasp deploy flow to proceed.**

## 3. (B) 5k front detector — retrained and evaluated (uncommitted, not adopted as deploy default)
- `ckpt/cube_detector_front_5k.pt` (trained locally, 40 ep). Evaluation (scratchpad/eval_seg_centroid.py, same held-out 5k val 500, deploy metric = seg_summary centroid cls0): **5k seg-centroid median 3.0 px / mean 5.2 px / IoU 0.92** vs 1500 at 3.7/5.8/0.88 → **better on every metric**.
- The conf≥0.30 gate passes 99% = gross failures (7 cases >50 px for both models) are confident-but-wrong (occlusion/edge), so conf does not filter them (common limitation; v2 needs stronger not-visible handling).
- **Not adopted as deploy default** — `--front-detector ckpt/cube_detector_front_5k.pt` only after HW (u,v) alignment verification.

## 4. hover-follow — unfinished
- The `follow` mode control law itself is fine (TCP tracks the cube xy straight-down at plane_z+0.16 m, shaped, deadband). Dashboard Follow button / curl `cmd?a=follow`.
- On the real robot, FOLLOW tracking down to 3 mm was briefly observed, but a stable demo was not completed because of **§2 perception-stale + §1 J4 oscillation**.
- **Fixing §2 and retrying under config A should work.**

## 5. Git / commit state (branch perception-topwrist-coop, all uncommitted)
Same as the docs/23 list + no changes this session (zero code edits — KP_APR is a motor register, the detector is a ckpt file). **Pending commit**: KP_APR tools (damiao_params/measure_latency, HW verification + store done) · 5k detector · deploy wiring. **The §2 fix requires an actual code edit in debug_deepreach.py (not done yet).**

## 6. Next session actions (priority)
1. **§2 perception intermittent-stale fix** (raise the threshold at debug_deepreach.py:930 or slim down the thread) — blocker for the entire deploy flow.
2. **hover-follow HW retry** (after config A + §2 fix = stability expected).
3. **Revisit MIT mode** (user's "smooth and fast" — remove the residual POS_VEL limit cycle; git c9df753).
4. **Commit**: KP_APR tools + config A + 5k detector (HW-verified parts).

## 7. Gotchas
- **KP_APR config A ≠ session 14 values** (J4 500→150). The values in session 14 handoff (docs/23) §0 are discarded.
- **Passing step/track tests ≠ stable on the real robot** — always hold-verify in a loaded pose (j4_hold_test.py).
- **"PERCEPTION DEAD/STALE" appearing does not mean a camera problem** — first confirm liveness via /frame MD5 diff. Intermittent stale = slow thread (720p+CNN).
- The user presses dashboard buttons directly (Capture/RUN/Follow) — overlapping with curl-driven control causes confusion. In real-robot sessions, confirm the division of roles.
- The residual limit cycle may be mechanical (backlash), so complete removal in software is unlikely → MIT damping or mechanical inspection.

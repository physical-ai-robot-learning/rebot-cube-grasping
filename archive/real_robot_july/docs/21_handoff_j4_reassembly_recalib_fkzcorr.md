# Handoff — Full recalibration after J4 disassembly→reassembly + (key) FK_Z_CORR=0.09 bug found (2026-07-13, session 11)

> Follows docs/19 (session 10, deploy wrist-centric increments + motor CAN blocker). This session: **the user disassembled→reassembled J4 (wrist_flex)** → full recalibration (motor zero + rest anchor + gripper) → during debug_deepreach grasp verification, **a z problem where the TCP slams into the table** → traced to the root → confirmed by HW measurement that **`robot_io.FK_Z_CORR=0.09` is too large (the comment already says "0.03" but the value had drifted)** → fixed to 0.03 (uncommitted). **Session ended right before grasp re-verification** (SSH dropped due to a remote MacBook update).
>
> **One line**: J4 reassembly recalibration, all 3 steps done (motor zero→rest anchor→gripper). Root cause of TCP-table collision in deploy verification = **FK_Z_CORR 0.09→0.03** (io FK z was 6 cm higher than reality, so auto_z over-descended). After the uncommitted fix, **re-verification (RUN→Align→Grip→Lift) is the top priority next**.

---

## SESSION 12 done (2026-07-13) — re-verification passed + Lift bug fixed + commit/push
Completed the §0 re-verification on the real robot, and fixed a **Lift bug** exposed by running the full grip+lift cycle end-to-end for the first time. The robot **actually picked up and lifted the cube** (end-to-end).

- **FK_Z_CORR=0.03 verification passed**: after RUN, `fk_bias=0.03` (0.09→normalized, auto_z contamination resolved), `tcp_io.z≈0.05` hover (does not hit the table), `z_gap_io≈goal`. The grasp reached the cube accurately.
- **Full grasp+Lift success**: Capture→RUN→(jog descent)→Grip(grasped=True, grip_tau 0.93)→**Lift (tcp_io_z 0.024→0.323, cube lifted off the table)**→put-down done.
- **Lift bug fixed**: the idle `elif gripping:` (force-servo hold) re-pinned the arm every tick with `send_arm(pr.arm_qpos, snap)` → after grasp, the target sent by Lift was cancelled on the next tick → the arm did not rise. **FIX = `gripping=False` in the `lift`/`jog` handlers** (grip_cmd stays closed = cube held). A bug exposed by running the full grip+lift cycle on HW for the first time.
- **2 commits (branch `perception-topwrist-coop`, pushed)**: `bfe1ea5` (robot_io FK_Z_CORR + joint_map + calib_grip) + `bd94fb5` (debug_deepreach lift-fix+cube-offset + debug_dashboard, including session 10 track/align-seg increments). `uv.lock` and `deepreach_sac_world.py` (another workstream) remain uncommitted.

**Warning: issues for next session (SESSION 12 notes):**
1. **"PERCEPTION DEAD/STALE" is a startup-latch false positive** — do not chase it by restarting. The thread is fine (frames update). Confirm liveness via `/frame?n=wrist` md5 changing or `capture`→`cube_raw` being populated. `cube_raw=null` is normal at REST hover. (This session wasted 4 restarts on this.)
2. **Open-jaw descent physically catches on the cube** — the real reason `Final`/RUN cannot descend after contact (separate from the reached gate). Consider redesigning the descent approach / gripper width.
3. **Align J4 framing overshoots when starting near-centered** (vn -0.02→0.73, cube pushed out of view) — `align_j4_gain` needs near-range tuning/deceleration. From a far start (vn -0.84) it settled well at 0.11.
4. **Grip depth**: RUN/Final stops at `z_gap_io≈goal(0.030)±0.015` (grasp-HOVER) → actual gripping needs more depth (tcp_io_z≈0.025). jog `j=2 d=1` (J3) ≈ −4 mm/step, preserves wrist (u,v) centering. Put-down = `run` (gently lower the held cube to the captured target)→`stop`→jog→`open`.
5. **Nature of the XY offset not yet determined** (§4): the cube moved, so the offset had to be re-converged — position-dependent vs systematic is still open.

---

## 0. (key) [done] FK_Z_CORR 0.03 re-verification → passed (see SESSION 12 above)
The code (`robot_io.py`) was modified, so **a restart is required**. After Ctrl+C:
```bash
cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/debug_deepreach.py \
  --grip-force 0.05 --grasp-zgap-io 0.03 --cube-x-off -0.029 --cube-y-off -0.042
```
→ After **RUN**, check with `curl -s localhost:8899/state`:
1. **Does io tcp.z match the actual hover** (it should no longer touch the table — with FK_Z_CORR 0.03, io FK z ≈ actual tip)
2. **Has `fk_bias` dropped to around 0.03** (before it was 0.09 = FK_Z_CORR itself = source of auto_z contamination)
3. dx/dy residual (was 1.2 cm previously)

If OK → **complete Align → Final → Grip → Lift** to confirm an actual grasp success → commit (§5).

---

## 1. Git / uncommitted state (branch `perception-topwrist-coop`)
Latest commit top = `210b225` (docs/19 handoff). **Uncommitted (working tree, stays local even if SSH drops)**:
```
 M deploy/robot_io.py        ← (key) FK_Z_CORR 0.09→0.03 (core fix this session)
 M deploy/debug_deepreach.py ← added --cube-x-off/--cube-y-off args (this session) + session 10 --track/--align-seg (still uncommitted)
 M deploy/debug_dashboard.py ← session 10 uncommitted (essentials/details cleanup)
 M deploy/calib_grip.py      ← CLOSE_SWEEP refined to 1-step (this session)
 M deploy/joint_map.npz      ← J4 reassembly recalibration result (this session)
 M uv.lock                   ← unrelated
 ?? deploy/deepreach_sac_world.py ← SAC-policy deploy tool from a parallel work session (07-13) (docs/20), not my work
```
Warning: **`debug_deepreach.py`/`debug_dashboard.py` mix session 10 uncommitted changes (`--track`, `--align-seg`, mask) with this session's changes.** When committing, decide whether to logically separate this session's grasp-verification changes (cube offset, FK_Z_CORR, calib_grip, joint_map) from the session 10 increments.

**Current `joint_map.npz` (recalibration result):**
```
L_rest_deg  = [0.1, -76.24, -51.69, 18.79, 1.41, 0.34]   (J4 +18.79° = correct sign, cam-down)
grip_closed = 29.05   grip_open = -179.15
```

---

## 2. J4 reassembly → 3-step recalibration (done, HW verified)
The user disassembled→reassembled J4 (wrist_flex). In order:

**① Reset lerobot motor zero** (sit-down) — J4 reassembly shifted the motor's physical zero (J4 +z at home). `connect(calibrate=True)` is skipped because of the existing file, so force it:
```bash
.venv-deploy/bin/python -c "
import sys; sys.path.insert(0,'deploy'); sys.path.insert(0,'src')
from lerobot_robot_seeed_b601 import SeeedB601DMFollower, SeeedB601DMFollowerConfig
r = SeeedB601DMFollower(SeeedB601DMFollowerConfig(port='/dev/ttyACM0', id='follower1', can_adapter='damiao'))
r.connect(calibrate=False)
try: r.calibrate()          # enter 'c' → place in sit-down pose and press ENTER → set_zero_position for all joints
finally: r.disconnect()
"
```
(seeed_b601_follower.py:177 "ZERO POSITION = manual sit-down". All 6 joints are reset → the sit-down pose must be reproduced exactly.)

**② calib_convention.py** (rest anchor) — the key is to tilt **J4 so the gripper cam points down** in A1. If L_rest J4 comes out **positive (+18.79°)** it is correct (the previous failed calibration was −38.3° = opposite sign). B2 rise and B3 descent passed = MAP OK.

**③ calib_grip.py** (gripper) — the zero changed, so it cannot be restored from git; recalibration is mandatory. Warning: **CLOSE_SWEEP refined to 1-step (`[0,-1,-2,…]`)** (code modified) — the self-contact wall is very steep (torque 0.13→1.84 within ~6 motor-deg), so the old −3 step skipped STRAIN (0.5) and hit ABORT. Result: closed 29.0 / open −179.2 / **`--grip-force 0.05`** (target 0.8). Warning: in ② the self-close torque reached 1.84, close to the breakage threshold (1.86), so **do not repeat ②**.

> Reassembly recalibration details are documented separately (deploy motor-connect recovery notes).

---

## 3. (key) Core finding: `FK_Z_CORR=0.09` bug (root cause, verified by HW measurement)
In deploy verification, **RUN was reached but the gripper tip slammed into the table**. Root cause:
```python
# robot_io.py fk_tcp():
mid[2] -= self.cfg.tcp_tip_drop
mid[2] += FK_Z_CORR          # ← artificially adds 0.09 to io FK z
```
- **HW measurement**: with the gripper tip touching the table (actual tip z=0), io `fk_tcp` z=**0.059**. → Setting FK_Z_CORR to `0.09−0.059=0.031`, i.e. **0.03**, gives io FK z ≈ 0 = actual tip. (The comment already said "reduced to 0.03" but the value had drifted to 0.09. git: `5361299` 0.12 → `fa0675e` 0.09, never changed to 0.03 = comment-value mismatch.)
- **Double contamination**: (a) io FK z is 6 cm higher than reality + (b) `auto_z` **treats io as truth** (debug_deepreach:734), so `fk_bias = io−base = FK_Z_CORR = 0.09` → z_bias too large → descent target 6 cm lower → **over-descends and slams into the table**.
- **Fix (uncommitted)**: `robot_io.py` FK_Z_CORR **0.09 → 0.03** (+ comment with the measurement basis). py_compile OK.
- Warning: **this is an original bug unrelated to the reassembly** — FK_Z_CORR 0.09 was too large but was previously cancelled out by chance through base_z/grasp_zgap_io tuning, and it surfaced when z was re-tuned after reassembly.

---

## 4. RUN XY offset tuning (band-aid, beware position dependence)
Recalibration shifted the FK so the TCP misses the cube when RUN is reached → **added `--cube-x-off`/`--cube-y-off` world-offset arguments** (debug_deepreach.py, applied to `cube_tx/cube_ty` = single entry point for the RUN servo + reached gate).
- Converged values (for this cube position): `--cube-x-off -0.029 --cube-y-off -0.042` → dx −6 mm, dy +10 mm, xy 1.2 cm.
- Warning: **x/y change even in sign with cube position/pose** (the xy diff between base_ctrl and io FK is pose-dependent). The offset is a band-aid for this position — **next session, determine systematic vs pose-dependent using multiple cube positions**. If pose-dependent, the root fix is FK alignment, not an offset.

---

## 5. Debugging infrastructure (use next session too)
- **Read the dashboard live state directly with curl** (no need to have values read out):
  ```bash
  curl -s localhost:8899/state | python -c "import sys,json; d=json.load(sys.stdin); print(d['tcp_io'], d['cube_top'], d['gates']['z_gap_io'])"
  ```
  Key fields: `tcp_io` (actual FK, accurate), `tcp` (base_ctrl, under-reading shadow), `cube_top`, `grasp_xy`, `xy_err`, `gates` (z_gap_io/fk_bias/z_bias/align_top_vn/…), `reached`.
- **Must distinguish the two FKs**: `tcp_io` (robot_io, actual) vs `tcp` (base_ctrl, FK under-reads → negative z is a shadow, physically impossible). auto_z uses io as truth.

---

## 6. Next session actions
1. **FK_Z_CORR 0.03 re-verification** (§0): rerun → RUN → check via curl io tcp.z (does it touch the table?) / fk_bias (0.03?) / dx-dy.
2. **Complete the grasp**: Align → Final → Grip (force 0.05) → Lift. Whether the cube actually lifts = the real goal of the reassembly verification.
3. **Commit on success**: `robot_io.py` (FK_Z_CORR) + `debug_deepreach.py` (cube offset) + `calib_grip.py` + `joint_map.npz`. Decide on separating from session 10 uncommitted changes (`--track`/`--align-seg`/dashboard).
4. **Determine the nature of the x/y offset**: measure dx/dy at multiple cube positions → systematic (offset OK) vs pose-dependent (FK alignment needed).

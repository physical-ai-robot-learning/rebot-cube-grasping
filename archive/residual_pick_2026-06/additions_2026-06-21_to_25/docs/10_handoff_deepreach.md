# Deep-Reach + Deploy handoff (2026-06-24 session 2)

> Session after docs/09 (residual-RL v6 Phase 1·2a). **lift-hold + deploy calib + Deep-Reach (reach the cube before grasping)**.
> **One line: lift-hold slip solved (grip hold-latch, sim best 0.90) + deploy reach working (calib fix) + Deep-Reach Rule (top+wrist closed-loop→cube, 90%/79%) + Deep-Reach Debugger. 6 commits (branch `residual-rl-v6-phase1-2a`).**

---
## 1. Results this session (all sim-verified, some not deployed)

| Item | Result | Files |
|---|---|---|
| **lift-hold slip solved** | Diagnosis = the policy lifts the cube 14cm and **re-opens the grip** (independent of effort/friction/tilt). → `grip hold-latch`. Retrain `pick_v6a_yaw_holdlatch`: grasp≈success, **fall 23-44%→6-12%, best 0.90** | `pick_env._apply_action` (commit e52eb4b), backup `ckpt/pick_v6a_yaw_holdlatch_best_0p90.pt` |
| **DEPLOY reach working** | calib.npz was the old FK-calib (cam 45cm/measured 57, X↔Y swap). `solvePnPRansac` **diverges on coplanar points** → **single `solvePnP ITERATIVE`**: world 11mm, **reached=true (err_xy 3mm)** | `calib_extrinsic_known.py` (c476e6e), `debug_reach` (7268c0b) |
| **Deep-Reach Rule** | `deep_reach_step`: A=straight-down reach→hover (latch), B=after xy alignment, descend to cube + wrist fusion + J6, **gripper OPEN**. **90% (top exact)/79% (11mm calib)** | `base_controller` (3e463ae) |
| **Deep-Reach Debugger** | real-hardware debugger (top+wrist closed-loop, gripper open), dashboard xy_err/z_gap/at_cube | `deploy/debug_deepreach.py`+`debug_dashboard.py` (a4cfb0f) |
| **wrist detector label fix** | `--gt-label` ignored occlusion (65px). Occlusion gate (projection + red present). Regenerated red-blob `percep_wrist_lookat.npz` (visible 0.30, n4000) | `gen_perception_data_wrist.py` (4ba25a6). **CNN retraining pending** |
| **deploy grip-latch confirmed unnecessary** | `run_pick` FSM (approach→commit→**hold=force-close**) is stronger than the sim latch — not ported | (workflow analysis) |

## 2. Core logic (Deep-Reach Rule)
```
deep_reach_step(arm_q, cube_xy_top, wrist_un/vn/vis, [wrist_cos4/sin4]):
  clamp top xy to the reach radius (0.70) (prevents edge-ray divergence)
  Phase A: compute_q_base(reach_look_at=False temporarily) → hover, latch   # STRAIGHT-DOWN (look-at diverges the reach)
  Phase B (latched):
    wxy = _wrist_uv_to_world(un,vn)  (inverse of project_cube_to_wrist, round-trip 0mm)
    fused_xy = wxy if (visible & |wxy-top|<12cm) else top   # wrist corrects top (closed-loop)
    z_target = cube surface if xy aligned (<4.5cm) else hover   # xy first → then descend (prevents hitting the cube)
    q = _ik_step6(target=[fused_xy,z], cam=STRAIGHT-DOWN)
    q[J6] -= jaw_gain·(image_yaw)·vis                        # J6 face alignment (residual 14-16°)
  gripper OPEN throughout
```

## 3. Identified causes (all from data, probe_deep_reach.py)
- **Divergence (xy 14m)** = **edge-ray** in top `_cube_est`/wrist back-projection (near-horizontal → t explodes). → clamp.
- **Hitting the cube (15m)** = the jaw strikes the cube while descending before xy alignment. → xy first, then descend.
- **reach 22%** = **look-at (aiming the wrist) tilts the cam and makes the 6-DOF reach diverge**. → straight-down (94%). (decisive)
- **J6 yaw increasing** = sign was reversed → flipped (residual 14-16°, proportional limit → residual learns it).
- **lift-hold** = the policy re-opens the grip (finger_gap→grip_open_qpos).
- **Large wrist correction (5cm)** = binary fusion oscillates and diverges, and look-at during descent also diverges. With the actual 11mm calib, top-only 79% is sufficient (wrist is a bonus).

## 4. Commits (6, branch `residual-rl-v6-phase1-2a`, v6 main body @cd35b61)
```
c476e6e fix(deploy): known-calib single solvePnP (coplanar RANSAC divergence)
7268c0b feat(deploy): debug_reach world-frame TCP + debug_grasp vertical_ok/engaged-open
e52eb4b fix(rl): grip hold-latch (lift-hold slip)
4ba25a6 fix(perception): wrist label occlusion gate
3e463ae feat(rl): deep_reach_step
a4cfb0f feat(deploy): Deep-Reach Debugger
```
Uncommitted (`.gitignore`/outputs): `scripts/probes/*` (probe_lifthold·probe_deep_reach·probe_wrist_unproject), `data/`, `deploy/*.png`, `calib.npz`, `extrinsic_known_raw.npz`, `calib_broken_ippe_backup.npz`.

## 5. Remaining work (priority order)
1. **Real-hardware verification of the Deep-Reach Debugger** — `debug_deepreach.py` after connecting the hardware. cube top accurate / two TCPs / xy_err↓ / z_gap→0 / at_cube.
2. **Residual SAC + A/B** (user's step 4): deep_reach rule (79-90%) baseline **vs** rule+residual. J6 alignment and grasp fine-tuning are left to the residual to learn.
3. **Retrain wrist CNN**: `train_cube_detector.py` with `percep_wrist_lookat.npz` (visible 0.30) → measure px (improves on 65px?). If low, raise the weight of grasp-side samples (gen `a∈[0.25,1]`).
4. **grasp (close+lift)** = the next bottleneck after deep_reach. lift-hold (rotated grasp variance) also remains.

## 6. Run state (at the end of the session)
- **All training/monitor stopped, GPU freed** (15.8GB free). Hardware not connected.
- tensorboard :6006 may still be alive (lightweight).
- Policy backups: `pick_v6a_yaw_holdlatch_best_0p90.pt` (lift-hold, slip solved), `pick_v6a_yaw_phase2a_94pct.pt` (previous).
- Data: `data/percep_wrist_lookat.npz` (red-blob, look-at, n4000 visible0.30) — for wrist CNN training.

## 7. Commands to reproduce/continue
```bash
# Deep-Reach sim verification (probe; --top-offset/--no-wrist/--no-lookat/--jaw-gain)
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/probes/probe_deep_reach.py --config configs/train_v6a_yaw.yaml --num-envs 48
# Deep-Reach Debugger (real hardware)
.venv-deploy/bin/python deploy/debug_deepreach.py --port 8899 --config configs/train_v6a_yaw.yaml
# hold-latch retraining
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py --config configs/train_v6a_yaw.yaml --exp pick_v6a_yaw_holdlatch
# wrist CNN retraining (percep_wrist_lookat data)
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train_cube_detector.py   # check the data path
```

## 8. Caveats
- **`pkill` as a standalone command** (bundling it with the next launch gives exit 144). **SIGTERM does not kill Isaac training immediately → SIGKILL** is needed. **When stopping training, also clean up the monitor** (it keeps polling dead ckpts and wastes 5.5GB of GPU → looks "as if training is running").
- **look-at makes both the reach and the descent of deep_reach diverge → fixed to straight-down.** (However, the pick policy's `reach_look_at` is kept; only deep_reach_step turns it off temporarily internally.)
- deep_reach measurements with privileged top do not show the fusion effect → measure by injecting calib error with `--top-offset`.
- `scripts/probes/`, `deploy/calib.npz`, `data/` are in `.gitignore`.

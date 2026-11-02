# Robust-Hybrid handoff (2026-06-25 session 3)

> After docs/10 (Deep-Reach + deploy session 2). **Deploy control architecture settled + v6b deep_reach residual base wins A/B + pedestal z-gap root cause identified + IBVS dropped.**
> **One line: deploy = XY (wrist world-unproj) + Z (front-gripper geometry + pedestal base_pos correction); IBVS dropped after verification. v6b (deep_reach base+residual) overwhelms v6a (hover base) in a fair A/B, 0.97 vs 0.75 (fall 6x↓). All sim — real-hardware verification is next.**

---

## 1. Results this session (all sim-verified; not verified on real hardware)

| Item | Result | Files |
|---|---|---|
| **Pedestal = root cause of the z-gap** (key) | The J1 base sits on a 2cm pedestal = actual z≈+0.02, while the config had −0.03 (5cm error) → base-FK reads the TCP 5cm too low = most of the z-gap/z_bias so far. **With base_pos z→+0.02, z_bias≈0**. calib/cube are table-referenced, so unchanged. reach 0.97@+0.02 | `debug_deepreach --base-z 0.02`, `train_v6b` config base_pos |
| **IBVS dropped after evaluation** (key) | sim-verified the option of closing deep_reach XY with IBVS → **unproj fusion is better**. Real xy under noise: top-only 50mm / unproj 8mm / IBVS 14-21mm / IBVS+EMA 39-76mm (lag diverges during descent). Planar constraint → unproj absolute target + clamp low-pass is robust | `base_controller._wrist_ibvs_dxy` (code kept, OFF) |
| **Z = front-gripper** (key) | cube z is known from the plane; the unknown = gripper z (sag). The front cam detects the gripper by frame-diff → ray + cube xy → gripper z → gap. CPU verification **0mm exact, robust to +1cm xy residual** | `deploy/front_gripper_z.py` (new) |
| **v6b deep_reach residual base** (key) | `residual_base="deep_reach"`: base=deep_reach_step (reach+descend, gripper open), residual=grasp+lift. **Wins the fair A/B decisively** (§3 below) | `pick_env._apply_action`, `configs/train_v6b_deepreach.yaml` (new) |
| wrist CNN + depth head | `perception_cnn` `with_depth`; `cube_detector_wrist_sd3.pt`. CNN primary + red_mask fallback. Warning: straight-down recall 6% (the occlusion label mislabels straight-down as not-visible) | `perception_cnn`, `train_cube_detector`, `gen_perception_data_wrist`, `deploy/cube_detector_io.wrist_uvn_depth` |
| deploy wiring | `debug_deepreach` = CNN wrist (unproj XY) + front_gripper_z (Z) + 3-source Z + pedestal base-z + stronger grip (overdrive+sustain) + **one-shot RUN mode** (continuous automatic descent, `--rule-step` velocity) | `deploy/debug_deepreach.py`, `debug_dashboard.py` |
| Ray training infra | the rtx5080a node was up without `--resources`, so jobs stayed PENDING → `ray stop`+`ray start --resources='{rtx5080a:1}'`. Added data/npz to submit_train.sh excludes (413 fix) | `scripts/submit_train.sh` |

## 2. Settled deploy control architecture (a diagram Artifact exists)
```
Top/front cam (fixed, calib) → cube world xy (ray-plane, FK-free) ─┐
                          → gripper detection (frame-diff) → gripper z ─┤
Wrist cam (eye-in-hand, CNN) → cube (u,v) → unproj XY (sag-corrected) ────┤
                                                                  ↓
A. COARSE: front cube → DLS-IK reach → hover(latch)              base = rule(deep_reach)
B. FINE: XY=unproj fusion(noise robust) / Z=3-source median(front-gripper/wrist-depth/FK+pedestal) / J6=yaw
   → residual SAC does grasp(close)+lift+fine adjustment
```

## 3. v6b A/B (fair 64-ep, same eval_seed 12345 + holdout dr_mass 0.06-0.10, `scripts/probes/eval_ab.py`)
| pipeline | best ckpt | success | grasp | fall |
|---|---|---|---|---|
| A: v6a hover-base | `_latest` | 0.750 | 0.859 | 0.203 |
| **B: v6b deep_reach-base** | `_best` | **0.969** | **0.984** | **0.031** |
- v6b decisive advantage: success +0.22, grasp +0.13, **fall 0.20→0.03 (6x↓)**. + faithful to deploy (the rule does the geometric reach/descend → smaller residual role = smaller sim2real gap).
- Warning: **The v6a "0.90/94%" was an optimistic peak of the 16-ep eval during training** — 0.46–0.75 with the fair 64-ep eval. (The saved _best ckpt is also worse than _latest = side effect of selecting best on a noisy 16-ep eval. → From now on, select/report best with large eval_episodes.)
- Deploy candidate = **`ckpt/pick_v6b_deepreach_best.pt`** (+ backup `_1M_0p84_best.pt`).

## 4. Limits of the verification (honest)
- **All sim.** detector px/depth/recall, IBVS, unproj, front-gripper "0mm", A/B — all sim render/physics. Real cameras + real sag not verified.
- **Sag modeled as translation only** (rotational sag not verified), front-gripper **frame-diff detection not verified** (only the geometry is 0mm), pedestal reconciled with a single observation.
- → **The real verification = hardware.**

## 5. Remaining work (priority order)
1. **Hardware deploy verification** (when back on the real robot): **confirm z_bias≈0** with the pedestal correction via `debug_deepreach --base-z 0.02` (confirms the hypothesis); tune front-gripper frame-diff (diff_thresh/box); RUN mode; **deploy the v6b policy** (port deep_reach base + the stack into run_pick).
2. **Fix CNN straight-down recall**: the datagen occlusion gate mislabels straight-down as not-visible → relax the gate/blob centroid + add close-descent samples → retrain.
4. demos for the deep_reach-base (currently demo_frac 0).

## 6. Run/reproduce commands
```bash
# Reproduce A/B (fair eval)
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/probes/eval_ab.py --config configs/train_v6b_deepreach.yaml --ckpt ckpt/pick_v6b_deepreach_best.pt --episodes 64
# Retrain v6b (Ray job; the node needs the rtx5080a resource)
#   First: ray stop; ray start --address=<ray-head>:6379 --node-ip-address=<cluster-node> --num-gpus=1 --resources='{"rtx5080a":1}'
EXP=pick_v6b_deepreach STEPS=3000000 ./scripts/submit_train.sh --config configs/train_v6b_deepreach.yaml
RAY_ADDRESS=http://<ray-head>:8265 ray job {status,logs,stop} rebot-pick_v6b_deepreach
# Real-hardware deploy debugger (pedestal correction + fast continuous RUN)
.venv-deploy/bin/python deploy/debug_deepreach.py --port 8899 --config configs/train_v6a_yaw.yaml --base-z 0.02 --rule-step 0.20
# front-gripper Z geometry self-test (Isaac not required)
.venv-deploy/bin/python deploy/front_gripper_z.py
```

## 7. Caveats
- **`--steps` counts policy steps** (not env_steps) — v6b reaches eval 0.84 at step ~16k and 0.95-1.0 at 210k. Leaving it to 3M takes days → **`ray job stop` after convergence**.
- **Run eval with large eval_episodes** (the 16-ep "best" is optimistic — the A/B revealed v6a 0.90 → actually 0.46-0.75).
- Ray detached jobs have **no automatic harness completion notification** — check with `ray job status`.
- `scripts/probes/`, `data/`, `ckpt/`, `*.png`, `*.npz`, `viz_out/` are gitignored/artifacts.

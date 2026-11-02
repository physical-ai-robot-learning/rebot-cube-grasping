# Cluster training — Ray submission + MLflow (reBot SAC pick)

How to train this project remotely on the BH 5-GPU Ray cluster (head `<ray-head>`, MLflow `:5000`, Ray `:8265`).

> **This project is not dataset training; it is Isaac sim RL.** There is no "dataset" to move; experience is generated in real time by the sim.
> Therefore the **NFS/parquet/WebDataset staging, MacBook MPS smoke test, and multi-node DDP** from the general GPU-cluster workflow doc **do not apply.**
> Here "data" = **the static inputs `assets/` (USD/meshes ~74M) + `demos/` (~46M)**, and only these need to be placed on the worker.

---

## 0. Model — 1 job = 1 GPU

Each Isaac training run uses a **single GPU** (num_envs in parallel on one GPU). **DDP is unnecessary/not recommended** (WiFi all-reduce bottleneck).
For sweeps, **submit multiple jobs** with different `--exp`/`--seed` → Ray distributes them across the 5 GPUs. (e.g. v3 + 3 seeds at the same time)

```
[this box/MacBook] submit_train.sh ──ray job submit (code only)──▶ [GPU worker] 1 Isaac process
                                                              ├─▶ MLflow(:5000)  metrics·ckpt
                                                              └─▶ NFS/local      ckpt copy
```

---

## 1. One-time worker setup (prerequisites)

The following must be in place **in advance on the worker** that `ray job submit` schedules onto (cu128 is currently OK, **Isaac must be installed separately**):

1. Install **Isaac Sim + cu128 torch + pure-Python deps** into a venv (same configuration as this box's `.venv`).
   - Third-party packages needed by the training process: `isaacsim*`, `torch==2.7.0+cu128`, `numpy`, `pyyaml`, `tensorboard`, **`mlflow-skinny`**.
   - The project code (`rebot_sim2real`) is picked up via `sys.path` from `src/` in the working-dir that ray sends, so **it does not need to be installed in the worker venv**.
   - Only add the MLflow client: `uv pip install --python <isaac-venv>/bin/python mlflow-skinny` (does not touch torch, verified).
2. **Stage `assets/` + `demos/` at a path the worker can read** (= `$REBOT_DATA_ROOT`, default `/mnt/shared/rebot`).
   ```bash
   # Upload once to the central NFS (@<nfs-host> <nfs-host>), then the worker pulls it locally or mounts NFS directly
   rsync -a assets/ <nfs-host>:/srv/data/rebot/assets/
   rsync -a demos/pick_demos_state43.pt <nfs-host>:/srv/data/rebot/demos/
   # Worker: make it visible as /mnt/shared/rebot/{assets,demos} (NFS mount) or rsync-stage it to /data
   ```
   - `scripts/train.py` resolves `robot_usd`/`demo_path` (relative paths) against `$REBOT_DATA_ROOT`.
   - **To save WiFi:** rather than reading directly from NFS every time, staging once to the worker's local NVMe (`/data`) and reading from there is recommended.

---

## 2. MLflow (already integrated in the code)

`src/rebot_sim2real/rl/train.py` logs to the central MLflow **in parallel with tensorboard** (best-effort):

- tracking_uri = `$MLFLOW_TRACKING_URI` (default `http://<ray-head>:5000`), experiment = `exp_name`.
- **Sends only at low frequency** (WiFi-friendly): 200-step summaries (success/return/fps/alpha) + eval points (eval/eval_holdout/best) + **uploads best/final ckpt as artifacts**.
- Server unavailable / `mlflow` not installed / `MLFLOW_DISABLE=1` → **silently disabled**, tensorboard continues. (Training never breaks)
- Connection verified: `http://<ray-head>:5000` reachable OK.

---

## 3. Submission

```bash
# 0) Set the worker Isaac venv python path to match the helper (once)
export WORKER_PY=/opt/isaac/.venv/bin/python      # <-- set to the actual worker path
export REBOT_DATA_ROOT=/mnt/shared/rebot           # location of assets/ + demos/

# 1) (recommended) quick smoke test on a GPU box before submitting — checks the pipeline only
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py --smoke-test

# 2) Submit full training (sends code only; excludes assets/demos/runs/ckpt/.venv)
./scripts/submit_train.sh                          # exp=pick_state_v3, steps=2,000,000

# Sweep (distributed across 5 GPUs):
EXP=pick_state_v3_s0 ./scripts/submit_train.sh --seed 0
EXP=pick_state_v3_s1 ./scripts/submit_train.sh --seed 1
```
> The ray flag names in `submit_train.sh` may differ between versions → check with `ray job submit --help`.
> The excludes drop `.venv` (several GB), `runs` (301M), `assets`, and `demos` to keep the working-dir to code only (<1MB).

---

## 4. Monitoring & retrieving results

| What | Where |
|---|---|
| Metrics/loss·success curves, run comparison | MLflow UI `http://<ray-head>:5000` |
| best/final checkpoints | MLflow run artifacts (+ `ckpt_dir` to NFS if desired) |
| Job status/logs | Ray Dashboard `http://<ray-head>:8265`, `ray job logs <id> -f` |
| GPU utilization/temperature | Grafana `http://<ray-head>:3000` |

When training finishes, take the best ckpt and deploy:
```bash
# After retrieving ckpt/pick_state_v3_best.pt from the MLflow artifacts or NFS
.venv-deploy/bin/python deploy/run_pick.py --ckpt ckpt/pick_state_v3_best.pt --max-steps 200
```

---

## 5. Local training (this box, <workstation>) works as before

You can also run it the existing way without Ray (MLflow runs alongside automatically; to turn it off use `MLFLOW_DISABLE=1`):
```bash
# To also use MLflow locally, install the client into the Isaac venv once:
uv pip install --python .venv/bin/python mlflow-skinny
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py
```

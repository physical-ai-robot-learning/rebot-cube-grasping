# Cluster training — Ray submission + MLflow (reBot SAC pick)

How to train this project remotely on the BH 5-GPU Ray cluster (head `<cluster-node>`, MLflow `:5000`, Ray `:8265`).

> **This project is Isaac simulation RL, not dataset training.** There is no "dataset" to move; experience is generated live by the simulator.
> Therefore the **NFS/parquet/WebDataset staging, MacBook MPS smoke and multi-node DDP** from the general GPU-cluster workflow docs **do not apply.**
> Here "data" = **the static inputs `assets/` (USD/meshes ~74M) + `demos/` (~46M)**; only these need to be on the worker.

---

## 0. Model — 1 job = 1 GPU

Each Isaac training run uses a **single GPU** (num_envs in parallel on one GPU). **DDP not needed / not recommended** (WiFi all-reduce bottleneck).
For sweeps, **submit several jobs** varying `--exp`/`--seed` → Ray distributes them over the 5 GPUs. (e.g. v3 + 3 seeds concurrently)

```
[this box/MacBook] submit_train.sh ──ray job submit (code only)──▶ [GPU worker] 1 Isaac process
                                                              ├─▶ MLflow(:5000)  metrics·ckpt
                                                              └─▶ NFS/local      ckpt copy
```

---

## 1. One-time worker setup (prerequisites) Warning:

The following must be in place **beforehand on the worker** scheduled by `ray job submit` (cu128 is currently OK, **Isaac needs a separate install**):

1. **Install Isaac Sim + cu128 torch + pure-Python deps** in a venv (same setup as this box's `.venv`).
   - Third-party packages needed by the training process: `isaacsim*`, `torch==2.7.0+cu128`, `numpy`, `pyyaml`, `tensorboard`, **`mlflow-skinny`**.
   - The project code (`rebot_sim2real`) is picked up via `sys.path` from `src/` in the working-dir that ray ships, so it **does not need to be installed in the worker venv**.
   - Only add the MLflow client: `uv pip install --python <isaac-venv>/bin/python mlflow-skinny` (does not touch torch, verified).
2. **Stage `assets/` + `demos/` at a path the worker can read** (= `$REBOT_DATA_ROOT`, default `/mnt/shared/rebot`).
   ```bash
   # upload once to the central NFS (@<nfs-host> <cluster-node>), then the worker pulls locally or mounts NFS directly
   rsync -a assets/ <cluster-node>:/srv/data/rebot/assets/
   rsync -a demos/pick_demos_state43.pt <cluster-node>:/srv/data/rebot/demos/
   # worker: make it visible as /mnt/shared/rebot/{assets,demos} (NFS mount) or rsync-stage to /data
   ```
   - `scripts/train.py` resolves `robot_usd`/`demo_path` (relative paths) against `$REBOT_DATA_ROOT`.
   - **Save WiFi:** rather than reading NFS directly every time, stage once to the worker's local NVMe (`/data`) and read from there.

---

## 2. MLflow (already integrated in the code)

`src/rebot_sim2real/rl/train.py` logs to the central MLflow **alongside tensorboard** (best-effort):

- tracking_uri = `$MLFLOW_TRACKING_URI` (default `http://<cluster-node>:5000`), experiment = `exp_name`.
- **Low-frequency only** (WiFi-friendly): 200-step summary (success/return/fps/alpha) + eval points (eval/eval_holdout/best) + **best/final ckpt uploaded as artifacts**.
- Server unavailable / `mlflow` not installed / `MLFLOW_DISABLE=1` → **silently disabled**, tensorboard continues. (Training never breaks.)
- Connection confirmed: `http://<cluster-node>:5000` reachable OK.

---

## 3. Submission

```bash
# 0) set the worker Isaac venv python path for the helper (once)
export WORKER_PY=/opt/isaac/.venv/bin/python      # <-- set to the actual worker path
export REBOT_DATA_ROOT=/mnt/shared/rebot           # location of assets/ + demos/

# 1) (recommended) quick smoke on the GPU box before submitting — checks the pipeline only
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py --smoke-test

# 2) submit full training (code only; excludes assets/demos/runs/ckpt/.venv)
./scripts/submit_train.sh                          # exp=pick_state_v3, steps=2,000,000

# sweep (distributed over 5 GPUs):
EXP=pick_state_v3_s0 ./scripts/submit_train.sh --seed 0
EXP=pick_state_v3_s1 ./scripts/submit_train.sh --seed 1
```
> The ray flag names in `submit_train.sh` may differ between versions → check with `ray job submit --help`.
> The excludes drop `.venv` (several GB), `runs` (301M), `assets`, `demos` so the working-dir stays code only (<1MB).

---

## 4. Monitoring & retrieving results

| What | Where |
|---|---|
| metrics/loss·success curves, run comparison | MLflow UI `http://<cluster-node>:5000` |
| best/final checkpoints | MLflow run artifacts (+ optionally `ckpt_dir` on NFS) |
| job status/logs | Ray Dashboard `http://<cluster-node>:8265`, `ray job logs <id> -f` |
| GPU utilization/temperature | Grafana `http://<cluster-node>:3000` |

When training finishes, fetch the best ckpt and deploy:
```bash
# after retrieving ckpt/pick_state_v3_best.pt from MLflow artifacts or NFS
.venv-deploy/bin/python deploy/run_pick.py --ckpt ckpt/pick_state_v3_best.pt --max-steps 200
```

---

## 5. Local training (this box, <workstation>) works as before

You can also run the existing way without Ray (MLflow runs alongside automatically; disable with `MLFLOW_DISABLE=1`):
```bash
# to use MLflow locally too, install the client in the Isaac venv once:
uv pip install --python .venv/bin/python mlflow-skinny
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/train.py
```

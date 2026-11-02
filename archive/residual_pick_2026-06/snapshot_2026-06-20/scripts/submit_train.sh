#!/usr/bin/env bash
# Submit a reBot SAC pick training job to the BH Ray cluster. VALIDATED on the local node
# (<workstation> / rtx5080a), which has the Isaac venv; the 5090/others need Isaac installed first.
#
# Model: each Isaac run is a SINGLE-GPU job (num_envs parallel on one GPU). No DDP.
# Sweep = submit several jobs (different --exp/--seed); Ray spreads them across the GPUs.
#
# Ships CODE ONLY via the ray working-dir (<~12M after excludes). The bulky static inputs
# (assets/ USD+meshes ~74M, demos/ ~46M, runs/ ~301M, .venv multi-GB) are NOT shipped:
#   - assets/demos come from $REBOT_DATA_ROOT on the worker (train.py resolves them there)
#   - ckpts persist to $REBOT_CKPT_DIR (ray's working-dir is ephemeral -> would be wiped)
#
# Pins to a node via $NODE_RESOURCE (a per-node custom resource: rtx5080a/rtx5090/...), because
# the absolute $WORKER_PY (Isaac venv python) only exists on the node that has Isaac installed.
#
# Usage:
#   ./scripts/submit_train.sh                              # local node, exp=pick_state_v3, 2M steps
#   SMOKE=1 ./scripts/submit_train.sh                      # tiny pipeline check (4 envs/300 steps)
#   EXP=pick_state_v3_s1 ./scripts/submit_train.sh --seed 1
#   NODE_RESOURCE=rtx5090 WORKER_PY=/opt/isaac/.venv/bin/python REBOT_DATA_ROOT=/mnt/shared/rebot \
#     ./scripts/submit_train.sh                            # remote worker (once Isaac is installed there)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RAY_ADDRESS="${RAY_ADDRESS:-http://<ray-head>:8265}"
RAY_BIN="${RAY_BIN:-$HOME/ray-env/bin/ray}"   # cluster's ray CLI
NODE_RESOURCE="${NODE_RESOURCE:-rtx5080a}"                           # per-node resource to pin the job
WORKER_PY="${WORKER_PY:-$REPO/.venv/bin/python}"                     # that node's Isaac venv python (absolute)
REBOT_DATA_ROOT="${REBOT_DATA_ROOT:-$REPO}"                          # where assets/ + demos/ live on the node
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://<ray-head>:5000}"
EXP="${EXP:-pick_state_v3}"
STEPS="${STEPS:-2000000}"

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke-test)
  CKPT_DIR="${REBOT_CKPT_DIR:-/tmp/rebot_smoke_ckpt}"
  EXP="${EXP}_smoke"
else
  CKPT_DIR="${REBOT_CKPT_DIR:-$REPO/ckpt}"
fi

echo "[submit] node=$NODE_RESOURCE  exp=$EXP  steps=$STEPS  smoke=${SMOKE:-0}"
echo "[submit] WORKER_PY=$WORKER_PY  REBOT_DATA_ROOT=$REBOT_DATA_ROOT  CKPT_DIR=$CKPT_DIR  MLFLOW=$MLFLOW_URI"

# NOTE: flag names can vary by Ray version -- verify with `ray job submit --help` (validated on 2.55.1).
RAY_ADDRESS="$RAY_ADDRESS" "$RAY_BIN" job submit --no-wait \
  --submission-id "rebot-${EXP}" \
  --working-dir "$REPO" \
  --runtime-env-json "{
    \"excludes\": [\".git\", \".venv\", \".venv-deploy\", \"runs\", \"ckpt\", \"assets\", \"demos\",
                   \"**/__pycache__\", \"**/*.png\", \"**/*.jpeg\", \"**/*.mp4\"],
    \"env_vars\": {\"OMNI_KIT_ACCEPT_EULA\": \"YES\",
                  \"REBOT_DATA_ROOT\": \"$REBOT_DATA_ROOT\",
                  \"REBOT_CKPT_DIR\": \"$CKPT_DIR\",
                  \"MLFLOW_TRACKING_URI\": \"$MLFLOW_URI\"}
  }" \
  --entrypoint-num-gpus 1 \
  --entrypoint-resources "{\"$NODE_RESOURCE\": 1}" \
  -- "$WORKER_PY" scripts/train.py --exp "$EXP" --steps "$STEPS" "${EXTRA_ARGS[@]}" "$@"

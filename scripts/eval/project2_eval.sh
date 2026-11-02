#!/usr/bin/env bash
# Seeded re-evaluation of the Project 2 checkpoints (staged DET-xattn C51-SAC), using the trainer's own fixed
# eval-set benchmark mode (TP_EVAL=1) with opt-in per-episode output (src/det_xattn/eval_hooks.py).
#
# Three runs, one Isaac process each, written to $OUT/<run>/{episodes.csv,trainer_summary.json,summary.json,log.txt}:
#   A  bench86   - exact replica of the 2026-08-03 bench (86-cell grid, bench config, checkpoints of bench e3712k)
#   B  grid400   - same policy and config on the 400-case 20x20 grid, each case once (EVAL_SINGLE=1)
#   R  render     - rollout videos of env 0 (1 env, 4 fixed cases from the 86-cell grid, EVAL_VIDEO_DIR): MP4 per
#                  episode, front | wrist detector-input view with detection overlay; outcome in the file name
#   C  staged_e2e - EXPERIMENTAL, untested: GRASP_ON=1 and RSR_DCUBE=0 so episodes are not ended at the reach and the
#                  rule grasp+lift can run; the "lift" column is the end-to-end criterion
# Fixed: seed (EVAL_SEED), checkpoints (checkpoints/project2), config (configs/project2_bench_2026-08-03.env),
# success criteria (RSR_DCUBE / RSR_HOLD / APP_* / FRAC_GRASP from that config).
#
# Needs (on the Isaac machine): Isaac Sim 5.1 + Isaac Lab venv with the LeIsaac reBot fork ($PY),
# LeIsaac assets ($LEISAAC_ASSETS_ROOT, default $REBOT_ISAAC_WS/leisaac/assets), the YOLO-OBB cube detector ($DETECTOR), one GPU.
# Usage (repository root):  OUT=/path/to/out bash scripts/eval/project2_eval.sh [A|B|C|R ...]   (default: A B)
set -euo pipefail
: "${OUT:?set OUT to an output directory}"
REBOT_ISAAC_WS="${REBOT_ISAAC_WS:-$HOME/workspace/rebot_isaac_rl}"
PY="${PY:-$REBOT_ISAAC_WS/.venv/bin/python}"
DETECTOR="${DETECTOR:-$HOME/det_cube_obb_yolo26m_ft3.pt}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-32}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(A B)
mkdir -p "$OUT"
if [ "${DRY_RUN:-0}" != "1" ]; then
  for f in "$PY" "$DETECTOR" "$ROOT/checkpoints/project2/search_last.pt" "$ROOT/checkpoints/project2/approach_last.pt"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
  done
fi
run() {  # name grid extra-env...
  local name=$1 grid=$2; shift 2
  local d="$OUT/$name"; mkdir -p "$d"; rm -f "$d/episodes.csv"
  local cmd=(env $(grep -v '^#' "$ROOT/configs/project2_bench_2026-08-03.env" | xargs)
    "$@" EVAL_SET="$ROOT/results/project2/eval_grids/$grid" TP_EVAL_DIR="$ROOT/checkpoints/project2"
    CURRIC_WRITE="$d/cw.npy" LEISAAC_ASSETS_ROOT="${LEISAAC_ASSETS_ROOT:-$REBOT_ISAAC_WS/leisaac/assets}" PYTHONPATH="$ROOT/src/det_xattn"
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
    EVAL_OUT_DIR="$d" EVAL_SEED="$SEED" EVAL_EXIT_WHEN_DONE=1
    "$PY" -u "$ROOT/src/det_xattn/train_sac_det_xattn.py" --num-envs "${RUN_ENVS:-$NUM_ENVS}" --total-timesteps 1000000000
    --learning-starts 999999999 --buffer-size 1000000 --buffer-device cuda --batch-size 512 --num-updates 128
    --alpha-floor 0.1 --alpha-lr 1.5e-5 --target-entropy -3.5 --single-cube --headless --render-size 256
    --detector "$DETECTOR" --det-conf 0.15 --gamma 0.97 --v-min -20 --v-max 400 --http-port 8967 --save-dir "$d/save")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  echo "[project2-eval] $name -> $d"
  "${cmd[@]}" > "$d/log.txt" 2>&1 || echo "[project2-eval] $name exited with $?" | tee -a "$d/log.txt"
  python3 "$ROOT/analysis/project2_eval_summary.py" "$d" || true
}
for r in "${RUNS[@]}"; do
  case $r in
    A) run bench86 evalgrid_86cells_2026-08-03.npy EVAL_SINGLE=0 ;;
    B) run grid400 eval_set_400cells_2026-07-21.npy EVAL_SINGLE=1 EVAL_HEAT_NX=20 EVAL_HEAT_NY=20 ;;
    R) RUN_ENVS=1 run render render_cases_4_from_86cells.npy EVAL_SINGLE=1 EVAL_HEAT_NX=4 EVAL_HEAT_NY=1 EVAL_VIDEO_DIR="$OUT/render/videos" EVAL_VIDEO_MAX=4 ;;
    C) run staged_e2e evalgrid_86cells_2026-08-03.npy EVAL_SINGLE=1 GRASP_ON=1 RSR_DCUBE=0 ;;
    *) echo "unknown run $r" >&2; exit 2 ;;
  esac
done

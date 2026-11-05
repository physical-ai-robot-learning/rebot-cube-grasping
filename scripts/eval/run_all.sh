#!/usr/bin/env bash
# Evaluate every shipped teacher / student checkpoint with python -m rebot_teacher_student.evaluate (one Isaac process each).
#   CKPT_ROOT  folder that holds the original rl_output/ trees:
#              $CKPT_ROOT/rebot-sim2real-rl/rl_output/...  and  $CKPT_ROOT/rebot-dagger-visual/rl_output/...
#   PY         Isaac Sim 5.1 / Isaac Lab 2.3 python with LeIsaac and the reBot assets
#   OUT        output folder (CSV + JSON per checkpoint)
# Run from the repository root:  CKPT_ROOT=... PY=... OUT=... bash scripts/eval/run_all.sh
set -euo pipefail
: "${CKPT_ROOT:?}" "${PY:?}" "${OUT:?}"
EPISODES="${EPISODES:-128}"; NUM_ENVS="${NUM_ENVS:-64}"; SEED="${SEED:-0}"
T="$CKPT_ROOT/rebot-sim2real-rl/rl_output"; S="$CKPT_ROOT/rebot-dagger-visual/rl_output"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p "$OUT"
ev() { local tag="$2"; "$PY" -m rebot_teacher_student.evaluate --num-envs "$NUM_ENVS" --episodes "$EPISODES" --seed "$SEED" --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/$tag.log" | grep -E '^\[eval\]|Error|Traceback' || true; }
ev --tag teacher_v5g        --mode single  --ckpts "$T/rl_c51_v5g/c51_final.pt"
ev --tag teacher_joint3     --mode staged3 --ckpts "$T/rl_c51_joint/s0_final.pt" "$T/rl_c51_joint/s1_final.pt" "$T/rl_c51_joint/s2_final.pt"
ev --tag teacher_v3         --mode single  --ckpts "$T/rl_c51_v3/c51_final.pt"
ev --tag teacher_mixed99    --mode single  --ckpts "$T/rl_c51_mixed99/c51_final.pt"
ev --tag teacher_smooth_v1  --mode single  --ckpts "$T/rl_c51_smooth/c51_final.pt"
ev --tag student_dual128_v3 --mode student --ckpts "$S/rl_dagger_dual128_v3/dagger_final.pt"

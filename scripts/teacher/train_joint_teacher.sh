#!/usr/bin/env bash
# Stage 1b: jointly trained multi-stage teacher (one C51 actor-critic per stage, co-trained).
# NOTE: the code in this repository is the 4-stage version (commit c1407a3); the committed checkpoints
# rl_c51_joint/s{0,1,2}_final.pt are from the 3-stage version (commit ca032b5). See docs/provenance.md.
set -euo pipefail
source scripts/isaac_env.sh
"$PY" -m rebot_sim2real.teacher.train_c51_joint --num-envs 1024 --save-dir rl_output/rl_c51_joint "$@"

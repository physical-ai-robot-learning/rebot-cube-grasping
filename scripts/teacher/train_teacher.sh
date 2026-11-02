#!/usr/bin/env bash
# Stage 1: state-only distributional-C51 SAC teacher (privileged 43-D state).
# The command of the first teacher (rl_c51_smooth) as recorded in the original README. Later teachers were trained with
# the same script plus --resume / --success-rate-target / --min-sr-episodes / --min-sr-hits; their exact flags were not
# recorded. Dashboard: http://localhost:8890
set -euo pipefail
source scripts/isaac_env.sh
"$PY" -m rebot_sim2real.teacher.train_c51 --target-cube cube --num-envs 1024 \
  --success-target 5000 --save-dir rl_output/rl_c51_smooth "$@"

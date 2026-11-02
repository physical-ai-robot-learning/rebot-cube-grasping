#!/usr/bin/env bash
# Stage 4: Isaac-free deploy loop, run against the mock robot (no hardware). torch + numpy only.
# For the real robot a RobotInterface / VisualRobotInterface subclass has to be written (docs/deploy/).
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PY="${PY:-python}"
"$PY" -m rebot_sim2real.deploy.rebot_deploy --checkpoint checkpoints/teacher/c51_v5g_final.pt --mock --max-steps 20 --hz 100
"$PY" -m rebot_sim2real.deploy.rebot_deploy_visual --checkpoint checkpoints/student/dagger_dual128_v3_final.pt \
  --mock --max-steps 20 --hz 100 --device cpu

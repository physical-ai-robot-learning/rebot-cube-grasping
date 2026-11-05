#!/usr/bin/env bash
# Stage 4: Isaac-free deploy loop, run against the mock robot (no hardware). torch + numpy only.
# For the real robot a RobotInterface / VisualRobotInterface subclass has to be written (docs/deploy/).
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PY="${PY:-python}"
"$PY" -m rebot_teacher_student.deploy.state_deploy --checkpoint results/checkpoints/teacher/c51_v5g_final.pt --mock --max-steps 20 --hz 100
"$PY" -m rebot_teacher_student.deploy.visual_deploy --checkpoint results/checkpoints/student/dagger_dual128_v3_final.pt \
  --mock --max-steps 20 --hz 100 --device cpu

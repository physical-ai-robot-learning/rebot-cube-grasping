#!/usr/bin/env bash
# Stage 2: DAgger distillation of a state-only teacher into a visual student (front + wrist RGB + 21-D proprio).
# Command as recorded in the original README; add --dr for visual domain randomisation (background composite,
# lighting, image noise, proprio noise, wrist-mount jitter). TEACHER defaults to the shipped v5 teacher.
set -euo pipefail
source scripts/isaac_env.sh
TEACHER="${TEACHER:-results/checkpoints/teacher/c51_v5g_final.pt}"
"$PY" -m rebot_teacher_student.student.train_dagger_visual --num-envs 128 --teacher-checkpoint "$TEACHER" \
  --success-target 5000 --beta-decay-steps 150000 --save-dir rl_output/rl_dagger_visual "$@"

# scripts/ — thin launchers

**Status:** operational entry points only; all library code is in `src/`. Run from the repository root.

| Script | What | Needs |
|---|---|---|
| `isaac_env.sh` | `source` it: sets `REBOT_ISAAC_WS`, `LEISAAC_ASSETS_ROOT`, `PYTHONPATH=src`, `PY` | — |
| `eval/project2_eval.sh` | Project 2 seeded re-evaluation runs A / B / C and render run R (`python -m rebot_grasp_rl.train` with `TP_EVAL=1`); `DRY_RUN=1` prints the commands | Isaac, GPU, detector weights (dry run: nothing) |
| `eval/run_all.sh` | June: evaluate every teacher / student checkpoint with `python -m rebot_teacher_student.evaluate` | Isaac, the original `rl_output` trees |
| `teacher/train_teacher.sh`, `teacher/train_joint_teacher.sh` | June teachers | Isaac |
| `student/train_student.sh` | June DAgger student | Isaac |
| `deploy/mock_deploy.sh` | both Isaac-free deploy loops against mock robots with the shipped checkpoints | torch |

**Inputs → outputs:** see each script's header. **Provenance:** written for this repository; the Project 2 log tools
that used to be here are in `archive/project2_log_tools/`.

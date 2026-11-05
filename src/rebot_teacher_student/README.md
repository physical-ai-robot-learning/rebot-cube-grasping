# rebot_teacher_student — earlier stage (June 2026): privileged teacher → camera students

**Status:** superseded by `rebot_grasp_rl`, kept with its checkpoints. Training and evaluation need Isaac Sim; the
two deploy loops and `student/visual_nets.py` run with torch + numpy only and are tested on the shipped checkpoints
(`tests/test_deploy_mock.py`, `tests/test_characterization.py`). No RL policy from this stage worked on the robot.

**Inputs → outputs.** LeIsaac LiftCube scene (simulation S1) → state-only C51-SAC teachers (`c51_*.pt`), DAgger
camera students (`dagger_*.pt`); deploy loops: checkpoint + robot interface → motor targets.

**Entry points**
* `python -m rebot_teacher_student.teacher.train_c51` (`scripts/teacher/train_teacher.sh`), `...teacher.train_c51_joint`
* `python -m rebot_teacher_student.student.train_dagger_visual` (`scripts/student/train_student.sh`)
* `python -m rebot_teacher_student.evaluate` (`scripts/eval/run_all.sh`)
* `python -m rebot_teacher_student.deploy.state_deploy --checkpoint results/checkpoints/teacher/c51_v5g_final.pt --mock`
* `python -m rebot_teacher_student.deploy.visual_deploy --checkpoint results/checkpoints/student/dagger_dual128_v3_final.pt --mock --device cpu`

**Layout:** `env/` (June environment), `teacher/` (single, joint and sequential-stage teachers, staged inference),
`student/` (DAgger and visual-RL students, `visual_nets.py`), `deploy/` (Isaac-free loops, `dump_sim_spec.py`,
`requirements.txt`), `evaluate.py` (seeded evaluation; was `scripts/eval/eval_policies.py`), `dashboard.py`
(training web UI; was `tools/rebot_dashboard.py`). One-off diagnostics and renders are in `archive/june_sim_tools/`.

**Provenance:** was `src/rebot_sim2real/`; origin in `docs/provenance.md` §1–§4. Details of the stage:
`docs/stage_2026-06_teacher_student.md`; deploy contract: `docs/deploy/`.

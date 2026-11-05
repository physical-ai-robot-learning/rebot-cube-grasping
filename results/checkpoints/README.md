# results/checkpoints/

| File | What | Trained in | Loads with |
|---|---|---|---|
| `project2/search_last.pt`, `project2/approach_last.pt` (17.7 MB) | Project 2 search (8,038,400 env-steps) and approach (3,712,000) policies: the pair of the last logged bench (`results/project2/bench_daemon.log`, bench e3712k, 2026-08-03) | runs 8951 (search) and `det_TP304_refac_smoke9t` (approach), July–August 2026 | `src/rebot_grasp_rl/train.py` with `TP_EVAL=1`, `TP_EVAL_DIR=results/checkpoints/project2` |
| `teacher/c51_v5g_final.pt` | single-actor state-only C51 SAC teacher (`actor`, `critic`, `log_alpha`, `step`), environment version v5 (= the environment code in `src/`) | `rebot-sim2real-rl` `ca032b5` | `rebot_teacher_student.student.visual_nets.StateOnlyActor(43, 7)`; `deploy/state_deploy.py` |
| `teacher/c51_smooth_v1_final.pt` | first teacher (v1 environment: 5 s episodes, cube on one side, wider gripper range); also the original state-only deploy checkpoint `c51_policy.pt` | `c5fc4df` | same |
| `student/dagger_dual128_v3_final.pt` | visual DAgger student, 128×128 front + 128×128 wrist RGB + 21-D proprio (`enc_w`, `enc_f`, `actor`, `step`), committed with the v3 teacher in the v3 environment (no elbow gate, 5 s episodes); also the original visual deploy checkpoint `dagger_dual128_final.pt` | `rebot-dagger-visual` `b5e597d` | `deploy/visual_deploy.py`; `src/rebot_teacher_student/evaluate.py --mode student` |

MD5 sums and the checkpoints that are not included: [docs/provenance.md §4](../../docs/provenance.md#4-checkpoints) (June) and [§9](../../docs/provenance.md#9-july--august-2026-sources) (July–August).

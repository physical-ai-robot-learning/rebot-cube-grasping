# results/project2/ — raw evidence for Project 2 (staged DET-xattn C51-SAC in simulation)

| File | What | Written by |
|---|---|---|
| `bench_daemon.log` | 120 benches (124 heatmaps) of approach checkpoints 128k → 3,712k env-steps of run `det_TP304_refac_smoke9t`, each together with the latest search checkpoint, on the fixed eval-set grid (90 cells, later 86), deterministic, 32 envs, no domain randomisation; 2026-08-02 19:27 → 08-03 12:43 | `scripts/project2_tools/bench_daemon.py` (submitted one eval job per new checkpoint) |
| `ntfy_poller.log` | 114 curriculum self-eval sweeps posted from the dashboards of training runs 8951 (search training) and 8953 (approach training, search frozen); lines carry times, not dates (day 0 is presumably 2026-07-31, when the notifier was added) | `scripts/project2_tools/ntfy_sweep_poller.py` |
| `CHANGES_8953_20260731.md` | run notes of 2026-07-31: configuration, gates, mouth-offset fix | the author's work session |
| `heatmaps/bench_e*.png` | per-cell heatmaps of each bench (deployable gate / privileged reach / search) | bench daemon |
| `eval_grids/*.npy` | eval-set grids (x, y, yaw, wrist start) | trainer tools |
| `benches.csv`, `sweeps.csv`, `log_summary.json` | parsed from the two logs | `analysis/parse_logs.py` |
| `render_2026-11-02/` | per-episode CSV + summaries of the 4-episode rollout render (seed 0, included checkpoints, simulation S4): search 4/4, privileged reach 4/4, deployable gate 0/4 | `scripts/eval/project2_eval.sh R` |
| `reeval/{bench86,grid400,staged_e2e}/` | seeded re-evaluation of 2026-11-02 (seed 0): `episodes.csv` (one row per episode), `summary.json` (rates + Wilson 95 %), `trainer_summary.json` (trainer aggregate), `log.txt` (trainer log; paths redacted) | `scripts/eval/project2_eval.sh A B C`, `analysis/project2_eval_summary.py` |

## Success criteria

| Name in the logs | Criterion | Threshold in the benches (`configs/project2_bench_2026-08-03.env`) |
|---|---|---|
| `SEARCH` | episode reached the APPROACH phase | `SEARCH_HOLD=1` |
| `RSR` | deployable gate: cube centred in the wrist image (`APP_CEN`), jaw mouth low enough (`APP_MZ`), cube area fraction (`FRAC_GRASP`), held `RSR_HOLD` frames | 0.15 / 0.07 m / 0.23 / 1 |
| `RSR_DCUBE`, `rsr_old` | privileged: true 3-D jaw-mouth-to-cube distance < `RSR_DCUBE`, held `RSR_HOLD` frames | 0.07 m / 1 |
| sweep `RSR` (`ntfy_poller.log`) | the privileged criterion (training entrypoints set `EVAL_SUCC_BY_OLD=1`), threshold per run configuration | 0.07–0.14 m across the entrypoint files of 07-29 → 08-02 |

The sweep segments of run 8953 are separated where the step counter restarts (new job). Their configuration files
are not linked to the log lines, so the threshold of a given segment is not recorded.

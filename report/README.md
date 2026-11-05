# report/ — technical report

**Status:** current, English only, status of 5 November 2026. `report_en.tex` + `figures/` → `report_en.pdf`
(IEEEtran two-column).

**What it is.** The technical report of the repository: problem, related work, set-up, method (June teacher /
students; Project 2 staged C51-SAC on detections; privileged vs deployable criteria), experiments (benches, sweeps,
seeded re-evaluation A/B/C), results, the hardware path (RL failed; perception + kinematics + state machine recorded as
working, uncounted), failure analysis of the deployable gate, limitations and reproducibility.

**Build** (from this folder; [tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd report && tectonic -X compile report_en.tex
```

**How the numbers are checked.** The report adds no new numbers. [`../analysis/verify_report.py`](../analysis/verify_report.py)
extracts every number from the body of `report_en.tex` and asserts that it appears in the repository file it is mapped
to (`README.md`, `docs/claims.md`, `docs/stage_2026-06_teacher_student.md`, `docs/simulation_versions.md`,
`results/project2/README.md`, the re-evaluation `summary.json` files, the bench configuration). It also checks
required statements (no RL policy shown working on hardware, no hardware success rate, deployable gate 0.0 %) and
forbidden phrases. It runs in CI and prints `N claims checked, 0 failed`.

**Layout:** `report_en.tex`, `report_en.pdf`, `figures/` — `fig_log_curves.png`, `fig_bench_heatmap.png`,
`fig_student_inputs.png` (copies of files in `media/`), `fig_s4_rollout_frames.jpg` (three frames of the first episode
of `media/project2/hero_s4_rollouts_e3712k_2026-11-02.gif`, stacked).

**Provenance:** written by the author on 2026-11-05 from the repository's documents; no number was re-measured.

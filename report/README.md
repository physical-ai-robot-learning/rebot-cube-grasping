# report/ — technical report

**Status:** current, English only, status of 5 November 2026. [`report_en.pdf`](report_en.pdf)
(IEEEtran two-column). The LaTeX source is not published here.

**What it is.** The technical report of the repository: problem, related work, set-up, method (June teacher /
students; Project 2 staged C51-SAC on detections; privileged vs deployable criteria), experiments (benches, sweeps,
seeded re-evaluation A/B/C), results, the hardware path (RL failed; perception + kinematics + state machine recorded as
working, uncounted), failure analysis of the deployable gate, limitations and reproducibility.

**How the numbers are checked.** The report adds no new numbers. A checker script (included with the code on
request, [README §3](../README.md#3-code-availability)) extracts every number from the body of the LaTeX source and asserts that it appears in the repository file it is mapped
to (`README.md`, `docs/claims.md`, `docs/stage_2026-06_teacher_student.md`, `docs/simulation_versions.md`,
`results/project2/README.md`, the re-evaluation `summary.json` files, the bench configuration). It also checks
required statements (no RL policy shown working on hardware, no hardware success rate, deployable gate 0.0 %) and
forbidden phrases.

**Figures:** `fig_log_curves.png`, `fig_bench_heatmap.png` and `fig_student_inputs.png` are copies of files in
[`media/`](../media/README.md); `fig_s4_rollout_frames.jpg` is three frames of the first episode of
[`media/project2/hero_s4_rollouts_e3712k_2026-11-02.gif`](../media/project2/hero_s4_rollouts_e3712k_2026-11-02.gif), stacked.

**Provenance:** written by the author on 2026-11-05 from the repository's documents; no number was re-measured.

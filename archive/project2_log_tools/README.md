# archive/project2_log_tools/ — the tools that wrote the Project 2 logs (verbatim)

* `bench_daemon.py` — watched the approach checkpoint folder and submitted one bench job per new checkpoint; wrote
  `results/project2/bench_daemon.log`.
* `ntfy_sweep_poller.py` — polled the training dashboards and posted curriculum sweeps; wrote
  `results/project2/ntfy_poller.log`.

Moved unchanged from `scripts/project2_tools/` on 2026-11-05. They depend on the training cluster (Ray head, node
names and paths are redacted placeholders) and are kept as a record of how the evidence was produced.

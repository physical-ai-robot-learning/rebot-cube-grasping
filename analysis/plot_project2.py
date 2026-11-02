#!/usr/bin/env python3
"""Figures for README from results/project2/{benches,sweeps}.csv (written by analysis/parse_logs.py). Needs matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
D = R / "results" / "project2"
OUT = R / "media" / "project2"
b = list(csv.DictReader(open(D / "benches.csv")))
s = list(csv.DictReader(open(D / "sweeps.csv")))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), dpi=110)
x = [int(r["approach_step_k"]) / 1000 for r in b]
ax[0].plot(x, [float(r["search_pct"]) for r in b], ".-", label="search stage (reached APPROACH)")
ax[0].plot(x, [float(r["rsr_dcube_pct"]) for r in b], ".-", label="approach, privileged gate (true jaw-cube < 0.07 m)")
ax[0].plot(x, [float(r["rsr_pct"]) for r in b], ".-", label="approach, deployable gate")
ax[0].set_xlabel("approach checkpoint (M env-steps)"); ax[0].set_ylabel("% of eval cells")
ax[0].set_title("Fixed eval-set benches, 2026-08-02/03 (bench_daemon.log)", fontsize=9)
ax[0].set_ylim(-3, 103); ax[0].legend(fontsize=7, loc="center right")
seg = [r for r in s if r["run"] == "8953" and r["segment"] == "3"]
ax[1].plot([int(r["step_k"]) / 1000 for r in seg], [int(r["sweep_success_pct"]) for r in seg], ".-", label="sweep success (privileged distance gate)")
ax[1].plot([int(r["step_k"]) / 1000 for r in seg], [int(r["search_pct"]) for r in seg], ".-", label="search stage")
ax[1].set_xlabel("approach training (M env-steps)"); ax[1].set_ylim(-3, 103)
ax[1].set_title("Run 8953 curriculum sweeps, 3rd segment (ntfy_poller.log)", fontsize=9)
ax[1].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "project2_log_curves.png")
print("wrote", OUT / "project2_log_curves.png")

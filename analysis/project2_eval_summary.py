#!/usr/bin/env python3
"""Summarise a seeded Project 2 re-evaluation (episodes.csv written by src/det_xattn/eval_hooks.py).

For every stage criterion the rate over episodes and a Wilson 95 % interval are computed:
  search   = reached the APPROACH phase            (search stage)
  rsr_old  = privileged gate held (true jaw-mouth-to-cube distance < RSR_DCUBE)   (approach, true cube position)
  rsr      = deployable gate held                   (approach, deployable criterion)
  lift     = rule grasp+lift latch                  (end-to-end; only meaningful in the GRASP_ON=1 run)
Usage: python analysis/project2_eval_summary.py <run_dir> [<run_dir> ...]   -> writes <run_dir>/summary.json
"""
import csv
import json
import math
import sys
from pathlib import Path

CRITERIA = ("search", "rsr_old", "rsr", "lift")


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def summarise(run_dir):
    run_dir = Path(run_dir)
    rows = list(csv.DictReader(open(run_dir / "episodes.csv")))
    n = len(rows)
    out = {"run_dir": run_dir.name, "episodes": n, "cases": len({r["case"] for r in rows}),
           "seeds": sorted({int(r["seed"]) for r in rows})}
    for c in CRITERIA:
        k = sum(int(r[c]) for r in rows)
        out[c] = {"successes": k, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}
    tj = run_dir / "trainer_summary.json"
    if tj.exists():
        out["trainer_summary"] = json.loads(tj.read_text())
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        print(json.dumps(summarise(d)))

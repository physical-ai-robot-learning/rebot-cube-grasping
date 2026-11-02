#!/usr/bin/env python3
"""Rebuild every Project 2 (staged DET-xattn C51-SAC) rate reported in README.md / docs/claims.md from the raw logs.

Inputs (copied unchanged except redactions, see docs/provenance.md):
  results/project2/bench_daemon.log  fixed eval-set benches of approach checkpoints, 2026-08-02 19:27 -> 08-03 12:43
  results/project2/ntfy_poller.log   curriculum self-eval sweeps of the Ray runs "8951" / "8953", posted to a phone
Outputs:
  results/project2/benches.csv       one row per bench (approach checkpoint step, rates, cells)
  results/project2/sweeps.csv        one row per sweep post (run, segment, day offset, step, rates)
  results/project2/log_summary.json  the aggregates quoted in the README
Success criteria (from the code and the entrypoint files, see results/project2/README.md):
  SEARCH   = the episode reached the APPROACH phase (search stage handed the cube over)
  RSR      = deployable gate (wrist-image centring + jaw-mouth height + cube area fraction, held RSR_HOLD frames)
  RSR_DCUBE / rsr_old = privileged gate: true 3-D jaw-mouth-to-cube distance < RSR_DCUBE, held RSR_HOLD frames
  In the curriculum sweeps the per-cell "RSR" is the privileged gate, because every training entrypoint of those
  days set EVAL_SUCC_BY_OLD=1 (train_sac_det_xattn.py: _chh = _rsr_old_hit if EVAL_SUCC_BY_OLD else _rsr_hit).
Run: python analysis/parse_logs.py [--check]   (--check: recompute and compare with the committed outputs)
"""
import csv
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
D = R / "results" / "project2"
BENCH = re.compile(r"\[bench-daemon\] bench e(\d+)k: RSR ([\d.]+)% \| RSR_DCUBE ([\d.]+)% \| SEARCH ([\d.]+)% \| (\d+)/(\d+)")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})")
SWEEP = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] (\d+) sweep@(\d+)k -> ntfy \d+ \(RSR (\d+)% \| SEARCH (\d+)%")


def parse_bench():
    rows, last_stamp, first_stamp = [], None, None
    for line in (D / "bench_daemon.log").read_text(encoding="utf-8").splitlines():
        m = STAMP.match(line)
        if m:
            last_stamp = f"{m.group(1)} {m.group(2)}"
            first_stamp = first_stamp or last_stamp
        m = BENCH.search(line)
        if m:
            k, rsr, rd, s, cov, tot = m.groups()
            rows.append({"approach_step_k": int(k), "rsr_pct": float(rsr), "rsr_dcube_pct": float(rd),
                         "search_pct": float(s), "covered": int(cov), "cells": int(tot), "after_ray_log": last_stamp})
    return rows, first_stamp, last_stamp


def parse_sweeps():
    rows, prev = [], {}
    day, last_sec = 0, None
    for line in (D / "ntfy_poller.log").read_text(encoding="utf-8").splitlines():
        m = SWEEP.match(line)
        if not m:
            continue
        hh, mm, ss, run, step, rsr, srch = m.groups()
        sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
        if last_sec is not None and sec < last_sec:
            day += 1                                     # clock rolled over midnight (log lines carry no date)
        last_sec = sec
        step = int(step)
        p = prev.get(run)
        seg = (p[1] + 1 if (p is None or step < p[0]) else p[1]) if p else 1
        prev[run] = (step, seg)
        rows.append({"run": run, "segment": seg, "day_offset": day, "time": f"{hh}:{mm}:{ss}", "step_k": step,
                     "sweep_success_pct": int(rsr), "search_pct": int(srch)})
    return rows


def rng(vals):
    return [min(vals), max(vals)] if vals else None


def summarise():
    b, b0, b1 = parse_bench()
    s = parse_sweeps()
    full = [r for r in b if r["covered"] == r["cells"]]
    out = {
        "bench": {
            "n_benches": len(b), "n_full_coverage": len(full), "first_ray_stamp": b0, "last_ray_stamp": b1,
            "cells": sorted({r["cells"] for r in b}),
            "approach_step_k": rng([r["approach_step_k"] for r in b]),
            "search_pct": rng([r["search_pct"] for r in b]),
            "rsr_dcube_pct": rng([r["rsr_dcube_pct"] for r in b]),
            "rsr_pct": rng([r["rsr_pct"] for r in b]),
            "n_rsr_zero": sum(1 for r in b if r["rsr_pct"] == 0.0),
            "full_coverage_only": {"search_pct": rng([r["search_pct"] for r in full]),
                                   "rsr_dcube_pct": rng([r["rsr_dcube_pct"] for r in full]),
                                   "rsr_pct": rng([r["rsr_pct"] for r in full])},
            "last": b[-1] if b else None,
        },
        "sweeps": {"n_posts": len(s), "runs": {}},
    }
    for run in sorted({r["run"] for r in s}):
        rr = [r for r in s if r["run"] == run]
        segs = {}
        for seg in sorted({r["segment"] for r in rr}):
            sr = [r for r in rr if r["segment"] == seg]
            segs[str(seg)] = {"n": len(sr), "day_offsets": sorted({r["day_offset"] for r in sr}),
                              "step_k": rng([r["step_k"] for r in sr]),
                              "sweep_success_pct": rng([r["sweep_success_pct"] for r in sr]),
                              "search_pct": rng([r["search_pct"] for r in sr])}
        best = max(rr, key=lambda r: r["sweep_success_pct"])
        out["sweeps"]["runs"][run] = {"n": len(rr), "segments": segs,
                                      "sweep_success_pct": rng([r["sweep_success_pct"] for r in rr]),
                                      "best": best}
    return b, s, out


def write(b, s, out):
    for name, rows in (("benches.csv", b), ("sweeps.csv", s)):
        with open(D / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (D / "log_summary.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    b, s, out = summarise()
    if "--check" in sys.argv:
        old = json.loads((D / "log_summary.json").read_text())
        if old != json.loads(json.dumps(out)):
            print("FAIL: results/project2/log_summary.json differs from the logs")
            return 1
        print("PASS: log_summary.json matches the logs")
        return 0
    write(b, s, out)
    bb = out["bench"]
    print(f"benches: {bb['n_benches']} (full coverage {bb['n_full_coverage']}), SEARCH {bb['search_pct']}, "
          f"RSR_DCUBE {bb['rsr_dcube_pct']}, RSR {bb['rsr_pct']} (zero in {bb['n_rsr_zero']})")
    for run, v in out["sweeps"]["runs"].items():
        print(f"sweeps {run}: n={v['n']} success {v['sweep_success_pct']} segments "
              + "; ".join(f"{k}: steps {x['step_k']} succ {x['sweep_success_pct']} search {x['search_pct']} days {x['day_offsets']}"
                          for k, x in v["segments"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

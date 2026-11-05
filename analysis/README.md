# analysis/ — evidence checkers

**Status:** all run in CI without Isaac. **Inputs → outputs:** repository files → pass / fail and printed checks.

| Script | What it checks / writes |
|---|---|
| `parse_logs.py` | parses `results/project2/bench_daemon.log` and `ntfy_poller.log` into `benches.csv`, `sweeps.csv`, `log_summary.json`; `--check` verifies the committed summary equals the logs |
| `plot_project2.py` | rebuilds `media/project2/project2_log_curves.png` from the parsed tables |
| `project2_eval_summary.py` | rates and Wilson 95 % intervals from a re-evaluation `episodes.csv` |
| `verify_claims.py` | every number in `README.md` against the file it comes from |
| `verify_report.py` | every number in `report/report_en.tex` against the file it comes from |
| `ast_normalize.py`, `code_fingerprints*.json` | normalised-AST fingerprints of the code taken from the original repositories (provenance) |
| `refactor_equivalence.py`, `refactor_2026-11_map.json` | the 2026-11 restructure is a pure move + rename of commit `47398fd` (syntax tree and per-scope bindings) |

**Entry points:** `python analysis/<script>.py` from anywhere. **Provenance:** written for this repository.

#!/usr/bin/env python3
"""Check every number in README.md against the repository file it comes from.

* A number directly followed by "%" must be a value of results/project2/log_summary.json (rebuilt from the raw logs by
  analysis/parse_logs.py; this script first checks that the summary still matches the logs), or of a re-evaluation
  summary under results/project2/reeval/ once it exists.
* Every other number must be listed in SOURCES with the file it was read from and must appear in that file
  (compared by value; digit-group commas, underscores and a trailing "k" are ignored: 3,712k = 3712).
* A mapped number the README no longer prints fails ("stale"), so the map stays in step with the text.
Skipped: link targets and URLs, ISO dates and times, commit hashes, version strings (1.0.0), numbers glued to letters
(B601-DM, C51, 6-DoF, e3712k, docs/14), heading and list numbers.
Run from anywhere:  python analysis/verify_claims.py      Exit code 1 if any check fails.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
BENCHCFG = "configs/project2_bench_2026-08-03.env"
TRAINCFG = "configs/project2_train_9t_dacq_2026-08-02.env"
SUMMARY = "results/project2/log_summary.json"
PROV = "docs/provenance.md"
TPCFG = "src/det_xattn/tp_config.py"

SOURCES = {
    "4": "docs/deploy/DEPLOY_SPEC.md",   # 4 cm cube
    "6": "src/rebot_sim2real/deploy/rebot_deploy.py",  # 6 joints (N_ARM)
    "0.07": BENCHCFG,                    # RSR_DCUBE=0.07 (privileged threshold in the benches)
    "0.14": "results/project2/CHANGES_8953_20260731.md",   # upper end of sweep thresholds (also _ep files)
    "1": BENCHCFG,                       # RSR_HOLD=1
    "120": SUMMARY,                      # n_benches
    "124": "results/project2/README.md", # 124 heatmaps
    "86": SUMMARY,                       # cells
    "3712": SUMMARY,                     # approach checkpoint step (k)
    "8038400": PROV,                     # search checkpoint step
    "3712000": PROV,                     # approach checkpoint step
    "36": SUMMARY,                       # sweeps in segment 3 of run 8953 (n)
    "1101": SUMMARY,                     # step (k) of the best sweep
    "61": SUMMARY,                       # sweeps of run 8951
    "8951": SUMMARY, "8953": SUMMARY,    # run names (dashboard ports)
    "101": TPCFG,                        # num_atoms
    "10": BENCHCFG,                      # CURRIC_NX / NY = 10 (10x10 grid)
    "32": "scripts/eval/project2_eval.sh",   # bench / re-eval --num-envs 32
    "256": "results/project2/CHANGES_8953_20260731.md",   # training envs 256, render 256
    "5000": TPCFG, "1000000": TPCFG, "1000000000": TPCFG, "512": TPCFG, "128": "scripts/eval/project2_eval.sh",
    "0.1": "scripts/eval/project2_eval.sh", "1.5e-5": "scripts/eval/project2_eval.sh", "-3.5": "scripts/eval/project2_eval.sh",
    "0.15": "scripts/eval/project2_eval.sh", "0.97": "scripts/eval/project2_eval.sh", "-20": "scripts/eval/project2_eval.sh",
    "400": "scripts/eval/project2_eval.sh",
    "5.1": "scripts/isaac_env.sh", "2.3": "scripts/isaac_env.sh", "3.11": "scripts/isaac_env.sh",
    "188": "docs/provenance.md",         # detector weights size (MB), recorded with the not-included items
    "2026": "CITATION.cff",
    "0": "scripts/eval/project2_eval.sh",   # seed 0 (SEED default)
    "160": "results/project2/reeval/bench86/summary.json",   # episodes of re-evaluation A
    "20": "scripts/eval/project2_eval.sh",  # 20x20 grid (EVAL_HEAT_NX=20)
}
# where a SOURCES file does not hold the literal, a second file that does
ALSO = {"10": TRAINCFG}


def norm(s):
    s = s.replace(",", "").replace("_", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return repr(int(v)) if v == int(v) and "." not in s and "e" not in s.lower() else repr(v)


def readme_tokens():
    t = (R / "README.md").read_text(encoding="utf-8")
    t = re.sub(r"```.*?```", lambda m: re.sub(r"\d", "", m.group(0)) if False else m.group(0), t, flags=re.S)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", t)
    t = re.sub(r"\b\d{2}-\d{2}\b", "", t)                      # month-day (07-23)
    t = re.sub(r"\b\d{2}:\d{2}(:\d{2})?\b", "", t)            # times
    t = re.sub(r"`[0-9a-f]{7,40}`", "", t)
    t = re.sub(r"\b\d+\.\d+\.\d+\b", "", t)
    t = re.sub(r"^#+ \d+\.", "#", t, flags=re.M)
    t = re.sub(r"^\s*\d+\. ", "", t, flags=re.M)
    t = re.sub(r"<[A-Za-z][^<>\n]*>", "", t)
    t = re.sub(r"Parts? \d", "", t)                            # "Part 1" labels
    t = re.sub(r"§\d+", "", t)                                  # section signs
    t = re.sub(r"`?docs/\d+`?[–-]`?\d+`?", "", t)              # note ranges docs/08–34
    t = re.sub(r"\[\d+\. ", "[", t)                             # contents links [2. Results]
    t = re.sub(r"\b3-D\b", "", t)
    t = re.sub(r"Project \d", "", t)
    t = re.sub(r"\*\*\d\. ", "**", t)                          # bold part labels "**2. RL on hardware**"
    t = re.sub(r"notes \d+[–-]\d+", "notes", t)
    t = re.sub(r"\b16:9\b", "", t)                             # aspect ratio
    t = re.sub(r"(\d+)×(\d+)", r"\1 \2", t)
    t = re.sub(r"(\d)k\b", r"\1", t)                            # 3,712k -> 3,712
    pct, other = set(), set()
    for m in re.finditer(r"(?<![\w.\-/#])(\d+(?:\.\d+)?)[–-](\d+(?:\.\d+)?)\s?%", t):   # ranges "a–b %": both ends
        pct.add(norm(m.group(1)))
    for m in re.finditer(r"%\s*\((\d+(?:\.\d+)?)[–-](\d+(?:\.\d+)?)\)", t):    # Wilson intervals "x % (lo–hi)"
        pct.add(norm(m.group(1)))
        pct.add(norm(m.group(2)))
    for m in re.finditer(r"(?<![\w.\-/#])(-?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?)(?![\w])(\s?%)?", t):
        n = norm(m.group(1))
        if n is None:
            continue
        if n in pct:
            continue
        (pct if m.group(2) else other).add(n)
    return pct, other


def numbers_in(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(norm(str(obj)))
        if isinstance(obj, float) and obj == int(obj):
            out.add(norm(str(int(obj))))


def file_numbers(rel):
    t = (R / rel).read_text(encoding="utf-8", errors="ignore")
    toks = re.findall(r"-?\d+(?:[,_]\d{3})*(?:\.\d+)?(?:e-?\d+)?", t)
    return {norm(x) for x in toks} | {norm(x.lstrip("-")) for x in toks}


def main():
    r = subprocess.run([sys.executable, str(R / "analysis" / "parse_logs.py"), "--check"], capture_output=True, text=True)
    fails = [] if r.returncode == 0 else ["log_summary.json does not match the raw logs: " + r.stdout.strip()]
    results = set()
    numbers_in(json.loads((R / SUMMARY).read_text()), results)
    for p in (R / "results" / "project2" / "reeval").glob("*/summary.json") if (R / "results/project2/reeval").is_dir() else []:
        s = json.loads(p.read_text())
        numbers_in(s, results)
        for c in ("search", "rsr_old", "rsr", "lift"):     # rates and Wilson bounds are printed as percentages
            if isinstance(s.get(c), dict):
                for v in [s[c].get("rate")] + list(s[c].get("wilson95") or []):
                    if v is not None:
                        results.add(norm(f"{100 * v:.1f}"))
                        results.add(norm(str(int(round(100 * v)))) if abs(100 * v - round(100 * v)) < 1e-9 else None)
    pct, other = readme_tokens()
    for n in sorted(pct, key=float):
        if n in results:
            print(f"ok   {n:>10} %  <- results")
        elif n in ("95",):   # "Wilson 95 % intervals" (a confidence level, not a result)
            print(f"ok   {n:>10} %  (confidence level)")
        else:
            fails.append(f"percentage not in results: {n}")
    srcmap = {norm(k): v for k, v in SOURCES.items()}
    for n in sorted(other, key=float):
        if n not in srcmap:
            fails.append(f"unmapped: {n}")
            continue
        if n not in file_numbers(srcmap[n]):
            fails.append(f"not in source: {n} ({srcmap[n]})")
            continue
        print(f"ok   {n:>10}    <- {srcmap[n]}")
    for k, v in ALSO.items():
        if norm(k) in other and norm(k) not in file_numbers(v):
            fails.append(f"not in second source: {k} ({v})")
    for k in srcmap:
        if k not in other and k not in pct:
            fails.append(f"stale: {k}")
    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails))
        return 1
    print(f"\nPASS: {len(pct)} percentages and {len(other)} other numbers in README.md checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

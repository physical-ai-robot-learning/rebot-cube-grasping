#!/usr/bin/env python3
"""Check every number in the technical report (report/report_en.tex) against the repository file it comes from.

The report adds no new numbers: each number it prints was taken from a file of this repository (README.md,
docs/claims.md, docs/stage_2026-06_teacher_student.md, docs/simulation_versions.md, results/project2/README.md, the
re-evaluation summaries, the bench configuration). This script extracts every number from the body of
report/report_en.tex and asserts that it appears, as a whole number token, in the file SOURCES maps it to. A number
in the report but not in SOURCES fails ("unmapped"); a mapping no longer used fails ("stale").

Skipped (not claims): the preamble and the bibliography, LaTeX labels / references / citations, file paths and code
names (\\code, \\texttt, \\url), figure file names and widths, table column specifications, and digits glued to
letters (B601-DM, C51, 6-DoF, S4, 128k). The TikZ figure is layout; it is checked to print no numbers.

Wording checks: the report states that no RL policy was shown working on the hardware, that no hardware success rate is
reported, and that the deployable gate was 0.0 %, and it uses none of the phrases listed in FORBIDDEN.

Modelled on the checker of the org's garment repository. Run from anywhere:  python analysis/verify_report.py
Exit code 1 if any check fails.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = "report/report_en.tex"
README = "README.md"
CLAIMS = "docs/claims.md"
STAGE = "docs/stage_2026-06_teacher_student.md"
SIMVER = "docs/simulation_versions.md"
RESULTS = "results/project2/README.md"
BENCHCFG = "configs/project2_bench_2026-08-03.env"
REEVAL_A = "results/project2/reeval/bench86/summary.json"
REEVAL_B = "results/project2/reeval/grid400/summary.json"

# number as printed in the report -> the repository file it was taken from
SOURCES = {
    # dates and counts used throughout
    "2026": CLAIMS, "0": CLAIMS, "1": README, "2": STAGE, "3": README, "4": README, "5": STAGE, "6": README,
    "7": STAGE, "9": README, "10": CLAIMS, "13": README, "14": README, "16": README, "23": README,
    # set-up and method
    "22": SIMVER, "43": STAGE, "0.05": STAGE, "20": STAGE, "101": STAGE, "0.9": STAGE, "0.2": STAGE, "21": STAGE,
    "150000": STAGE, "8951": CLAIMS, "8953": CLAIMS, "256": README, "128": README, "512": README, "0.97": README,
    "400": CLAIMS, "0.07": CLAIMS, "0.15": BENCHCFG, "0.23": BENCHCFG, "5.1": README, "2.3": CLAIMS,
    # experiments
    "90": RESULTS, "86": CLAIMS, "32": RESULTS, "120": CLAIMS, "3712": README, "114": RESULTS, "0.14": CLAIMS,
    "160": CLAIMS, "646": CLAIMS,
    # results
    "92.1": CLAIMS, "99.4": CLAIMS, "84.0": CLAIMS, "95.2": CLAIMS, "0.0": CLAIMS, "97.8": CLAIMS, "89.9": CLAIMS,
    "96.5": CLAIMS, "99.9": CLAIMS, "88.1": CLAIMS, "82.2": CLAIMS, "92.3": CLAIMS, "98.5": CLAIMS, "96.8": CLAIMS,
    "99.3": CLAIMS, "73.0": CLAIMS, "68.5": CLAIMS, "77.1": CLAIMS, "97.7": CLAIMS, "91.9": CLAIMS, "4.3": CLAIMS,
    "159": REEVAL_A, "141": REEVAL_A, "394": REEVAL_B, "292": REEVAL_B, "95": CLAIMS, "100": CLAIMS, "61": CLAIMS,
    "71": CLAIMS, "81": CLAIMS, "36": CLAIMS, "1101": CLAIMS,
}
REQUIRED = [
    "No reinforcement-learning policy was shown working on the hardware",
    "no hardware success rate is reported",
    "the deployable gate was 0.0\\,\\% in every bench",
    "0 of 646 episodes",
]
FORBIDDEN = ["sim-to-real transfer succeeded", "successful sim-to-real", "validated", "validation", "RL policy grasped",
             "success rate of the rule controller"]

NUM = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?![0-9])")


def normSource(text):
    tText = re.sub(r"(\d)\\,(\d{3})(?!\d)", r"\1\2", text)
    tText = re.sub(r"(\d)\{,\}(\d{3})(?!\d)", r"\1\2", tText)
    return re.sub(r"(\d),(\d{3})(?![\d])", r"\1\2", tText)


def bodyOf(tex):
    tTex = re.sub(r"(?<!\\)%.*", "", tex)
    tTex = tTex[tTex.index("\\begin{document}"):]
    if "\\begin{thebibliography}" in tTex:
        tTex = tTex[:tTex.index("\\begin{thebibliography}")]
    return tTex


def numbersIn(tex):
    """Return (numbers in the text, numbers printed inside TikZ figures)."""
    tBody = normSource(bodyOf(tex))
    tTikz = "".join(re.findall(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", tBody, re.S))
    tBody = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", tBody, flags=re.S)
    for tCmd in ("label", "ref", "cite", "code", "texttt", "url", "includegraphics", "bibitem", "hypersetup"):
        tBody = re.sub(r"\\" + tCmd + r"(\[[^\]]*\])?\{[^{}]*\}", " ", tBody)
    tBody = re.sub(r"\\begin\{(tabular|table\*?|figure\*?)\}(\[[^\]]*\])?(\{(?:[^{}]|\{[^{}]*\})*\})?", " ", tBody)
    tBody = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", " ", tBody)
    tBody = re.sub(r"\\\\\[[^\]]*\]", " ", tBody)
    tBody = re.sub(r"width=[0-9.]+\\(textwidth|columnwidth)", " ", tBody)
    tBody = tBody.replace("\\\\", " ")
    tBody = re.sub(r"\\[A-Za-z]+", " ", tBody)
    tText = " ".join(re.findall(r"\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", tTikz))
    tText = re.sub(r"\\[A-Za-z]+", " ", tText.replace("\\\\", " "))
    return NUM.findall(tBody), NUM.findall(tText)


def main(argv: list[str] | None = None) -> int:
    tRows = []
    tTex = (ROOT / REPORT).read_text(encoding="utf-8")
    tNums, tTikzNums = numbersIn(tTex)
    tCache = {}
    for tNum in dict.fromkeys(tNums):
        tSrc = SOURCES.get(tNum)
        if tSrc is None:
            tRows.append((f"{tNum}: unmapped (no source)", "-", False))
            continue
        if tSrc not in tCache:
            tCache[tSrc] = normSource((ROOT / tSrc).read_text(encoding="utf-8"))
        tHit = re.search(r"(?<![0-9.])" + re.escape(tNum) + r"(?![0-9])(?!\.\d)", tCache[tSrc]) is not None
        tRows.append((f"{tNum} appears in its source", tSrc, tHit))
    for tNum in sorted(set(SOURCES) - set(tNums), key=float):
        tRows.append((f"{tNum}: mapped but not in the report (stale)", SOURCES[tNum], False))
    tRows.append((f"TikZ figure prints no numbers (found {sorted(set(tTikzNums))})", REPORT, not tTikzNums))
    tFlat = " ".join(bodyOf(tTex).split())
    for tPhrase in REQUIRED:
        tRows.append((f"states: {tPhrase}", REPORT, tPhrase in tFlat))
    for tPhrase in FORBIDDEN:
        tRows.append((f"does not say: {tPhrase}", REPORT, tPhrase.lower() not in tFlat.lower()))
    tFails = [tRow for tRow in tRows if not tRow[2]]
    for tClaim, tSrc, tOk in tRows:
        print(f"{'ok  ' if tOk else 'FAIL'} {tClaim}  <- {tSrc}")
    print(f"\n{len(tRows)} claims checked, {len(tFails)} failed.")
    return 1 if tFails else 0


if __name__ == "__main__":
    sys.exit(main())

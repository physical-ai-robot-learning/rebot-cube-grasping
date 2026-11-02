"""Static checks on the tree: required files, no Korean text, no machine-specific paths or addresses."""
import re
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
TEXT = {".py", ".md", ".sh", ".yaml", ".yml", ".toml", ".txt", ".cff", ".json", ".csv", ""}


def tracked_text_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=R, capture_output=True, text=True, check=True).stdout.split()
        files = [R / f for f in out]
    except Exception:
        files = [p for p in R.rglob("*") if ".git" not in p.parts]
    return [p for p in files if p.is_file() and p.suffix in TEXT]


def test_required_files():
    for f in ["README.md", "LICENSE", "NOTICE", "CITATION.cff", "docs/claims.md", "docs/provenance.md",
              "docs/third_party.md", "analysis/verify_claims.py", 
              "LICENSES/MIT-Squint.txt", "scripts/eval/eval_policies.py", "scripts/eval/project2_eval.sh",
              "analysis/parse_logs.py", "results/project2/bench_daemon.log", "results/project2/ntfy_poller.log",
              "results/project2/log_summary.json", "checkpoints/project2/search_last.pt", "checkpoints/project2/approach_last.pt"]:
        assert (R / f).is_file(), f


def test_no_hangul():
    bad = [str(p.relative_to(R)) for p in tracked_text_files()
           if re.search("[\uac00-\ud7a3]", p.read_text(encoding="utf-8", errors="ignore"))]
    assert not bad, bad


def test_no_machine_specific_strings():
    pat = re.compile(r"/home/[a-z]|/Users/[a-z]|100\.64\.\d+\.\d+|192\.168\.\d+\.\d+|bh-aiteam|ghp_[A-Za-z0-9]{20}"
                     r"|hf_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}")
    bad = []
    for p in tracked_text_files():
        if p.name == "test_layout.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{p.relative_to(R)}:{i}")
    assert not bad, bad


def test_no_ai_attribution():
    pat = re.compile(r"Co-Authored-By|Generated with \[Claude|noreply@anthropic", re.I)
    bad = [str(p.relative_to(R)) for p in tracked_text_files()
           if p.name != "test_layout.py" and pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not bad, bad


def test_media_small():
    for p in (R / "media").rglob("*"):
        if p.is_file():
            assert p.stat().st_size < 5 * 1024 * 1024, p


def test_checkpoints_total_under_40mb():
    tot = sum(p.stat().st_size for p in (R / "checkpoints").rglob("*.pt"))
    assert tot < 40 * 1024 * 1024, tot

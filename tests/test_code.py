"""Code checks: every Python file parses; every in-repo import of the src/ packages resolves to a file; no sys.path edits
in src/; the provenance fingerprints still hold.

Provenance chain (docs/provenance.md, docs/refactor_2026-11.md):
  original repositories --(normalised-AST fingerprints, analysis/code_fingerprints*.json)--> commit 47398fd
  commit 47398fd --(pure move + rename, analysis/refactor_equivalence.py)--> this tree.
Files moved verbatim to archive/ keep their fingerprints; the fingerprints of restyled files are checked on their
47398fd version, which needs the git history (skipped in a shallow clone).
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "analysis"))
import ast_normalize  # noqa: E402
import refactor_equivalence  # noqa: E402

PY = [p for d in ("src", "scripts", "analysis", "tests", "archive") for p in (R / d).rglob("*.py")]
PACKAGES = ("rebot_grasp_rl", "rebot_grasp_env", "rebot_teacher_student", "rebot_grasp_common")
REFACTOR_MAP = refactor_equivalence.loadMap()
BASE = REFACTOR_MAP["base"]


def _hasBase():
    tResult = subprocess.run(["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=R, capture_output=True)
    return tResult.returncode == 0


HAS_BASE = _hasBase()
needsBase = pytest.mark.skipif(not HAS_BASE, reason=f"git history with {BASE} not available (shallow clone)")


@pytest.mark.parametrize("path", PY, ids=lambda p: str(p.relative_to(R)))
def test_parses(path):
    ast.parse(path.read_text(encoding="utf-8"))


def _resolves(mod):
    tPath = R / "src" / Path(*mod.split("."))
    return tPath.with_suffix(".py").is_file() or (tPath / "__init__.py").is_file()


def test_internal_imports_resolve():
    tMissing = []
    for tFile in list((R / "src").rglob("*.py")) + list((R / "scripts").rglob("*.py")):
        for tNode in ast.walk(ast.parse(tFile.read_text(encoding="utf-8"))):
            tMods = []
            if isinstance(tNode, ast.Import):
                tMods = [tAlias.name for tAlias in tNode.names]
            elif isinstance(tNode, ast.ImportFrom) and tNode.module and tNode.level == 0:
                tMods = [tNode.module] + [m for m in (tNode.module + "." + a.name for a in tNode.names) if _resolves(m)]
            for tMod in tMods:
                if tMod.split(".")[0] in PACKAGES and not _resolves(tMod):
                    tMissing.append(f"{tFile.relative_to(R)}: {tMod}")
    assert not tMissing, tMissing


OLD_MODULES = ("packages", "rebot_sim2real", "det_xattn_nets", "tp_config", "tp_net", "ui_render", "runtime",
               "robot_profile", "eval_hooks")


def test_no_old_package_names_or_sys_path_in_src():
    tBad = []
    for tFile in (R / "src").rglob("*.py"):
        tText = tFile.read_text(encoding="utf-8")
        for tNode in ast.walk(ast.parse(tText)):
            if isinstance(tNode, ast.Import):
                tNames = [tAlias.name for tAlias in tNode.names]
            elif isinstance(tNode, ast.ImportFrom) and tNode.level == 0:
                tNames = [tNode.module or ""]
            else:
                continue
            tBad += [f"{tFile.relative_to(R)}: import {n}" for n in tNames if n.split(".")[0] in OLD_MODULES]
        for tNeedle in ("sys.path.insert", "sys.path.append"):
            if tNeedle in tText:
                tBad.append(f"{tFile.relative_to(R)}: {tNeedle}")
    assert not tBad, tBad


FP = json.loads((R / "analysis" / "code_fingerprints.json").read_text())
FP_JULY = json.loads((R / "analysis" / "code_fingerprints_july.json").read_text())
VERBATIM = {tOld: tNew for tNew, tOld in REFACTOR_MAP["moved_verbatim"].items()}
RESTYLED = {tEntry["old"]: tNew for tNew, tEntry in REFACTOR_MAP["files"].items()}


def _fingerprintCase(rel, expected):
    if rel in VERBATIM or (R / rel).is_file():          # verbatim move (archive/) or untouched file
        assert ast_normalize.fingerprint(R / VERBATIM.get(rel, rel)) == expected, f"{rel} changed"
        return
    assert rel in RESTYLED, f"{rel} is neither in the tree nor in analysis/refactor_2026-11_map.json"
    if not HAS_BASE:
        pytest.skip(f"git history with {BASE} not available")
    tOld = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=R, capture_output=True, text=True, check=True).stdout
    tTmp = R / ".pytest_cache" / "fingerprint_base.py"
    tTmp.parent.mkdir(exist_ok=True)
    tTmp.write_text(tOld, encoding="utf-8")
    assert ast_normalize.fingerprint(tTmp) == expected, f"{rel} at {BASE} differs from its fingerprint"


@pytest.mark.parametrize("rel", sorted(FP), ids=str)
def test_fingerprint_unchanged(rel):
    _fingerprintCase(rel, FP[rel]["fingerprint"])


def test_fingerprint_record_complete():
    tEq = [k for k, v in FP.items() if v["ast_equal"]]
    tNeq = {k: v.get("note") for k, v in FP.items() if not v["ast_equal"]}
    assert len(tEq) + len(tNeq) == len(FP)
    assert set(tNeq) == {"src/rebot_sim2real/tools/rebot_dashboard.py"}, tNeq


@pytest.mark.parametrize("rel", sorted(FP_JULY), ids=str)
def test_july_fingerprint_unchanged(rel):
    _fingerprintCase(rel, FP_JULY[rel])


@needsBase
@pytest.mark.parametrize("path", sorted(REFACTOR_MAP["files"]), ids=str)
def test_refactor_is_move_plus_rename(path):
    if REFACTOR_MAP["files"][path].get("manual"):
        pytest.skip("restyled by hand; covered by tests/test_characterization.py")
    refactor_equivalence.checkEntry(path, REFACTOR_MAP["files"][path])


@needsBase
def test_refactor_check_detects_a_wrong_binding():
    """Self-test: put back the original name at one use of a renamed local (same syntax tree after the inverse
    rename, but the name now binds to a global). The scope check must reject it on every Python version."""
    tPath = "src/rebot_teacher_student/teacher/train_c51.py"
    tEntry = REFACTOR_MAP["files"][tPath]
    tSource = (R / tPath).read_text(encoding="utf-8")
    tAt = tSource.rindex("tA)")
    tBroken = tSource[:tAt] + "A)" + tSource[tAt + len("tA)"):]
    assert tEntry["renames"]["tA"] == "A"
    with pytest.raises(refactor_equivalence.EquivalenceError):
        refactor_equivalence.checkEntry(tPath, tEntry, tBroken)


def test_refactor_map_covers_src():
    tSrc = {str(p.relative_to(R)) for p in (R / "src").rglob("*.py")}
    tMapped = set(REFACTOR_MAP["files"]) | set(REFACTOR_MAP.get("new_files", {}))
    assert tSrc == tMapped, (sorted(tSrc - tMapped), sorted(tMapped - tSrc))

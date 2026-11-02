"""Every Python file parses; every in-repo import of rebot_sim2real resolves to a file; the code moved from the original
repositories has not drifted from the normalised-AST fingerprints recorded in analysis/code_fingerprints.json."""
import ast
import json
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "analysis"))
import ast_normalize  # noqa: E402

PY = [p for d in ("src", "scripts", "analysis", "tests", "archive") for p in (R / d).rglob("*.py")]


@pytest.mark.parametrize("path", PY, ids=lambda p: str(p.relative_to(R)))
def test_parses(path):
    ast.parse(path.read_text(encoding="utf-8"))


def _resolves(mod, roots=("src", "src/det_xattn")):
    for root in roots:
        p = R / root / Path(*mod.split("."))
        if p.with_suffix(".py").is_file() or (p / "__init__.py").is_file():
            return True
    return False


def test_internal_imports_resolve():
    missing = []
    for p in list((R / "src").rglob("*.py")) + list((R / "scripts").rglob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module] + [n.module + "." + a.name for a in n.names]
                mods = [n.module] + [m for m in mods[1:] if _resolves(m)]
            for m in mods:
                if m.split(".")[0] in ("rebot_sim2real", "packages") and not _resolves(m):
                    missing.append(f"{p.relative_to(R)}: {m}")
    assert not missing, missing


def test_no_old_package_imports_in_src():
    bad = [str(p.relative_to(R)) for p in (R / "src").rglob("*.py")
           if "det_xattn" not in p.parts and "packages.sim" in p.read_text(encoding="utf-8")]
    assert not bad, bad


FP = json.loads((R / "analysis" / "code_fingerprints.json").read_text())


@pytest.mark.parametrize("rel", sorted(FP), ids=str)
def test_fingerprint_unchanged(rel):
    assert ast_normalize.fingerprint(R / rel) == FP[rel]["fingerprint"], f"{rel} changed since the provenance check"


def test_fingerprint_record_complete():
    eq = [k for k, v in FP.items() if v["ast_equal"]]
    neq = {k: v.get("note") for k, v in FP.items() if not v["ast_equal"]}
    assert len(eq) + len(neq) == len(FP)
    assert set(neq) == {"src/rebot_sim2real/tools/rebot_dashboard.py"}, neq


FP2 = json.loads((R / "analysis" / "code_fingerprints_july.json").read_text())


@pytest.mark.parametrize("rel", sorted(FP2), ids=str)
def test_july_fingerprint_unchanged(rel):
    assert ast_normalize.fingerprint(R / rel) == FP2[rel], f"{rel} changed since it was added"

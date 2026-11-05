#!/usr/bin/env python3
"""Check that the 2026-11 restructure / restyle of the code is a pure move + rename (no behaviour change).

For every entry of analysis/refactor_2026-11_map.json ({new path: {"old": old path, "renames": {new: old}}}), the file
at the pre-refactor commit (git show BASE:old) and the current file are compared on two levels:

  1. syntax tree: both trees are normalised (docstrings dropped; `sys.path.insert/append(...)` statements dropped;
     every import statement replaced by its bindings, with module names mapped old -> new by MODULE_MAP; the paths
     module's ISAAC_WS / LEISAAC_ASSETS expressions mapped to the original REBOT_ISAAC_WS expression), the current
     tree is renamed back with the entry's rename map, and the two serialisations must be identical;
  2. scopes: for every symbol-table scope (module, class, function, lambda, comprehension) the set of names and their
     binding flags (local / global / free / parameter / imported / assigned / referenced) must be identical after the
     rename map is applied. This catches a renamed local that would silently resolve to a different binding.

Entries whose "renames" is empty are pure moves. Entries with a "manual" note were restyled by hand beyond renaming
(small Isaac-free modules); they are skipped here and covered by the golden tests in tests/test_characterization.py. Files moved verbatim to archive/ are not listed here; their
fingerprints are checked directly against analysis/code_fingerprints*.json (tests/test_code.py).

Usage (repository root, needs the git history that contains BASE):
    python analysis/refactor_equivalence.py              # all entries, exit 1 on any difference
    python analysis/refactor_equivalence.py src/x.py     # one entry, prints the first differing node
"""
import ast
import json
import subprocess
import sys
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "analysis" / "refactor_2026-11_map.json"
BASE = "47398fd"
MODULE_MAP = [  # old module -> new module; longest prefix first
    ("packages.sim.env_setup.maniskill_rewards", "rebot_grasp_env.rewards"),
    ("packages.sim.env_setup", "rebot_grasp_env"),
    ("det_xattn.packages.sim.env_setup", "rebot_grasp_env"),
    ("rebot_sim2real.env.maniskill_rewards", "rebot_teacher_student.env.rewards"),
    ("rebot_sim2real.deploy.rebot_deploy_visual", "rebot_teacher_student.deploy.visual_deploy"),
    ("rebot_sim2real.deploy.rebot_deploy", "rebot_teacher_student.deploy.state_deploy"),
    ("rebot_sim2real.tools.rebot_dashboard", "rebot_teacher_student.dashboard"),
    ("rebot_sim2real", "rebot_teacher_student"),
    ("tp_config", "rebot_grasp_rl.config"),
    ("det_xattn_nets", "rebot_grasp_rl.nets"),
    ("tp_net", "rebot_grasp_rl.target_projection"),
    ("runtime", "rebot_grasp_rl.runtime"),
    ("ui_render", "rebot_grasp_rl.dashboard_render"),
    ("robot_profile", "rebot_grasp_rl.robot_profile"),
    ("eval_hooks", "rebot_grasp_rl.eval_hooks"),
]
PATHS_MODULE = "rebot_grasp_common.paths"
# Expressions of the original code that the paths module now provides (all string valued).
ISAAC_WS_EXPR = 'os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl"))'
PATH_EXPRS = {
    "ISAAC_WS": ISAAC_WS_EXPR,
    "LEISAAC_ASSETS": ISAAC_WS_EXPR + ' + "/leisaac/assets"',
    "WRIST_PROBE_DIR": 'os.path.expanduser("~/rebot_eval")',
    "YOLO_EXPORT_DIR": 'os.path.expanduser("~/yolo_cube")',
}


class EquivalenceError(RuntimeError):
    """A refactored file is not a pure move + rename of its original."""


def mapModule(name: str) -> str:
    for tOld, tNew in MODULE_MAP:
        if name == tOld or name.startswith(tOld + "."):
            return tNew + name[len(tOld):]
    return name


def _resolveRelative(node: ast.ImportFrom, path: str) -> str:
    tPkg = Path(path).with_suffix("").parts
    tPkg = tPkg[tPkg.index("src") + 1:-1] if "src" in tPkg else tPkg[:-1]
    tBase = list(tPkg[:len(tPkg) - node.level + 1])
    return ".".join(tBase + ([node.module] if node.module else []))


class _Normaliser(ast.NodeTransformer):
    """Normalise one tree (see the module docstring); `renames` maps current names back to the original ones."""

    def __init__(self, path: str, renames: dict, isNew: bool):
        self.mPath = path
        self.mRenames = renames
        self.mIsNew = isNew
        self.mPathNames = set()

    def _name(self, name):
        return self.mRenames.get(name, name) if name is not None else None

    def _strip(self, node):
        tBody = getattr(node, "body", None)
        if tBody and isinstance(tBody[0], ast.Expr) and isinstance(tBody[0].value, ast.Constant) \
                and isinstance(tBody[0].value.value, str):
            node.body = tBody[1:] or [ast.Pass()]
        return node

    def _dropSysPath(self, body):
        tOut = []
        for tStmt in body:
            if isinstance(tStmt, ast.Expr) and isinstance(tStmt.value, ast.Call) and \
                    ast.unparse(tStmt.value.func) in ("sys.path.insert", "sys.path.append"):
                continue
            if isinstance(tStmt, ast.ImportFrom) and self.mIsNew and (tStmt.module or "") == PATHS_MODULE:
                self.mPathNames.update(tAlias.asname or tAlias.name for tAlias in tStmt.names)
                continue
            tOut.append(tStmt)
        return tOut or [ast.Pass()]

    def generic_visit(self, node):
        for tField in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, tField, None), list):
                setattr(node, tField, self._dropSysPath(getattr(node, tField)))
        return super().generic_visit(node)

    def visit_Module(self, node):
        self._strip(node)
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):   # a docstring-only module is an empty one
            node.body = []
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._strip(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._strip(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if self.mIsNew and node.id in self.mPathNames and node.id in PATH_EXPRS:
            return ast.copy_location(ast.parse(PATH_EXPRS[node.id], mode="eval").body, node)
        node.id = self._name(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._name(node.arg)
        node.annotation = None
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node.attr = self._name(node.attr)
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_Global(self, node):
        node.names = [self._name(tName) for tName in node.names]
        return node

    visit_Nonlocal = visit_Global

    def _importMarker(self, node, bindings):
        tText = "IMPORT " + " ".join(sorted(f"{self._name(tLocal)}={tTarget}" for tLocal, tTarget in bindings))
        return ast.copy_location(ast.Expr(ast.Constant(tText)), node)

    def visit_Import(self, node):
        tBindings = []
        for tAlias in node.names:
            tTarget = mapModule(tAlias.name)
            tLocal = tAlias.asname or tAlias.name.split(".")[0]
            tBindings.append((tLocal, tTarget if tAlias.asname else mapModule(tLocal)))
        return self._importMarker(node, tBindings)

    def visit_ImportFrom(self, node):
        tModule = _resolveRelative(node, self.mPath) if node.level else (node.module or "")
        tBindings = [(tAlias.asname or tAlias.name, mapModule(f"{tModule}.{self._name(tAlias.name)}"))
                     for tAlias in node.names]
        return self._importMarker(node, tBindings)


def _canon(node) -> str:
    if isinstance(node, ast.AST):
        tParts = []
        for tField in node._fields:
            tValue = getattr(node, tField, None)
            if tValue is None or tValue == [] or tField in ("type_comment", "kind", "type_params"):
                continue
            tParts.append(tField + "=" + _canon(tValue))
        return type(node).__name__ + "(" + ",".join(tParts) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_canon(tItem) for tItem in node) + "]"
    return repr(node)


def normalisedDump(source: str, path: str, renames: dict, isNew: bool) -> str:
    return _canon(_Normaliser(path, renames, isNew).visit(ast.parse(source)))


SCOPE_FLAGS = ("is_parameter", "is_global", "is_declared_global", "is_local", "is_free", "is_imported",
               "is_assigned", "is_namespace", "is_nonlocal")
IGNORED_MODULE_NAMES = {"sys", "os", "__file__"}   # used only by the dropped sys.path lines / replaced path expressions


def _scopes(table, renames, out, prefix=""):
    tKey = f"{prefix}/{table.get_type()}:{renames.get(table.get_name(), table.get_name())}"
    tSymbols = []
    for tSym in table.get_symbols():
        tName = renames.get(tSym.get_name(), tSym.get_name())
        if table.get_type() == "module" and tName in IGNORED_MODULE_NAMES:
            continue
        tSymbols.append((tName, tuple(getattr(tSym, tFlag)() for tFlag in SCOPE_FLAGS)))
    out.append((tKey, tuple(sorted(tSymbols))))
    for tChild in table.get_children():
        _scopes(tChild, renames, out, tKey)
    return out


def scopeSignature(source: str, renames: dict, pathNames=()) -> list:
    tTable = symtable.symtable(source, "<file>", "exec")
    tOut = _scopes(tTable, renames, [])
    if pathNames:   # names of the paths module stand for the original os.* expressions
        tOut = [(tKey, tuple(tSym for tSym in tSyms if tSym[0] not in pathNames)) for tKey, tSyms in tOut]
    return tOut


def oldSource(oldPath: str) -> str:
    return subprocess.run(["git", "show", f"{BASE}:{oldPath}"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def checkEntry(newPath: str, entry: dict) -> None:
    tOld = oldSource(entry["old"])
    tNew = (ROOT / newPath).read_text(encoding="utf-8")
    tRenames = entry.get("renames", {})
    tNorm = _Normaliser(newPath, tRenames, True)
    tNewDump = _canon(tNorm.visit(ast.parse(tNew)))
    tOldDump = _canon(_Normaliser(entry["old"], {}, False).visit(ast.parse(tOld)))
    if tNewDump != tOldDump:
        tIdx = next(i for i, (a, b) in enumerate(zip(tNewDump, tOldDump)) if a != b) if any(
            a != b for a, b in zip(tNewDump, tOldDump)) else min(len(tNewDump), len(tOldDump))
        raise EquivalenceError(f"{newPath}: syntax tree differs from {entry['old']} near\n"
                               f"  new: ...{tNewDump[max(0, tIdx - 150):tIdx + 150]}\n"
                               f"  old: ...{tOldDump[max(0, tIdx - 150):tIdx + 150]}")
    tPathNames = tNorm.mPathNames
    if scopeSignature(tNew, tRenames, tPathNames) != scopeSignature(tOld, {}, tPathNames):
        tA = scopeSignature(tNew, tRenames, tPathNames)
        tB = scopeSignature(tOld, {}, tPathNames)
        tDiff = next(((x[0], sorted(set(x[1]) ^ set(y[1]))) for x, y in zip(tA, tB) if x != y), (len(tA), len(tB)))
        raise EquivalenceError(f"{newPath}: scope / binding structure differs: {tDiff}")


def loadMap() -> dict:
    return json.loads(MAP_PATH.read_text())


def main(argv: list[str] | None = None) -> int:
    tArgs = sys.argv[1:] if argv is None else argv
    tMap = loadMap()["files"]
    tTargets = tArgs or sorted(tMap)
    tFails = 0
    tManual = 0
    for tPath in tTargets:
        if tMap[tPath].get("manual"):
            tManual += 1
            print(f"skip {tPath}  (manual: {tMap[tPath]['manual']})")
            continue
        try:
            checkEntry(tPath, tMap[tPath])
            print(f"ok   {tPath}  ({len(tMap[tPath].get('renames', {}))} renames)")
        except EquivalenceError as tError:
            tFails += 1
            print(f"FAIL {tError}")
    tChecked = len(tTargets) - tManual
    print(f"\n{'PASS' if not tFails else 'FAIL'}: {tChecked - tFails}/{tChecked} files equivalent to {BASE} "
          f"({tManual} restyled by hand, covered by tests/test_characterization.py)")
    return 1 if tFails else 0


if __name__ == "__main__":
    sys.exit(main())

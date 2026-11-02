"""Normalised-AST fingerprint of a Python file, used to show that the code moved into this repository behaves like the
original files.

Normalisation (applied to both the original and the current file):
  * docstrings are dropped and comments do not exist in the AST (text-only changes are ignored);
  * imports of the original package layout are renamed to the new one
    (packages.sim.env_setup.* -> rebot_sim2real.env.*, packages.sim.visual_nets -> rebot_sim2real.student.visual_nets,
     packages.sim.rebot_dashboard -> rebot_sim2real.tools.rebot_dashboard);
  * machine-specific path defaults are mapped to one placeholder: the original absolute workspace prefix and the new
    expression  os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/..."  both
    become the string "<WS>/..."; the original ".../sim2real_so101/delivery/" prefix of rl_output defaults is dropped.

Usage:  python analysis/ast_normalize.py FILE [--original]     prints the SHA-256 of the normalised AST dump.
The original workspace prefix is passed through the environment variable ORIG_WS_PREFIX (it contains a machine user
name and is therefore not stored in this repository); without it only current files can be fingerprinted.
"""
import ast
import hashlib
import os
import sys

IMPORT_MAP = [
    ("packages.sim.env_setup.", "rebot_sim2real.env."),
    ("packages.sim.env_setup", "rebot_sim2real.env"),
    ("packages.sim.visual_nets", "rebot_sim2real.student.visual_nets"),
    ("packages.sim.rebot_dashboard", "rebot_sim2real.tools.rebot_dashboard"),
]


def _map_mod(name):
    for a, b in IMPORT_MAP:
        if name == a or (a.endswith(".") and name.startswith(a)):
            return b + name[len(a):]
    return name


class _Norm(ast.NodeTransformer):
    def __init__(self, orig_prefix):
        self.orig_prefix = orig_prefix

    def _strip_doc(self, node):
        b = getattr(node, "body", None)
        if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_doc(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_doc(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = _map_mod(node.module)
        if node.module == "rebot_sim2real.tools" or node.module == "packages.sim":
            node.module = "rebot_sim2real.tools" if any(a.name == "rebot_dashboard" for a in node.names) else node.module
        return node

    def visit_Import(self, node):
        for a in node.names:
            a.name = _map_mod(a.name)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and self.orig_prefix:
            v = node.value
            dl = self.orig_prefix + "/sim2real_so101/delivery/"
            if v.startswith(dl):
                return ast.copy_location(ast.Constant(v[len(dl):]), node)
            if v.startswith(self.orig_prefix + "/"):
                return ast.copy_location(ast.Constant("<WS>" + v[len(self.orig_prefix):]), node)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        l = node.left
        if isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant) and isinstance(l, ast.Call) \
                and "REBOT_ISAAC_WS" in ast.dump(l):
            return ast.copy_location(ast.Constant("<WS>" + node.right.value), node)
        return node


def _canon(node):
    """Version-independent serialisation: node type + non-empty fields (Python 3.10-3.14 differ in ast.dump defaults
    and add empty fields such as type_params)."""
    if isinstance(node, ast.AST):
        parts = []
        for f in node._fields:
            v = getattr(node, f, None)
            if v is None or v == [] or (f == "kind" and v is None):
                continue
            parts.append(f + "=" + _canon(v))
        return type(node).__name__ + "(" + ",".join(parts) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_canon(x) for x in node) + "]"
    return repr(node)


def fingerprint(path, original=False):
    tree = ast.parse(open(path, encoding="utf-8").read())
    prefix = os.environ.get("ORIG_WS_PREFIX", "") if original else ""
    tree = _Norm(prefix).visit(tree)
    return hashlib.sha256(_canon(tree).encode()).hexdigest()


if __name__ == "__main__":
    print(fingerprint(sys.argv[1], original="--original" in sys.argv))

"""The user guide's Python blocks parse, every kika name they import exists, and every keyword
they pass to a kika function is one of its parameters. They are not run: they need full
evaluations that are not in the repository."""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "source" / "user_guide"
FENCE = re.compile(r"```python[^\S\n]*\n(.*?)```", re.DOTALL)


def _blocks():
    for page in sorted(GUIDE.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for match in FENCE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            yield pytest.param(page.name, match.group(1), id=f"{page.stem}:{line}")


BLOCKS = list(_blocks())


def _kika_import(module: str, name: str):
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return getattr(mod, name)
    return importlib.import_module(f"{module}.{name}")


def _imported_names(tree: ast.AST) -> dict:
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "kika":
            for alias in node.names:
                names[alias.asname or alias.name] = _kika_import(node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "kika":
                    module = importlib.import_module(alias.name)
                    names[alias.asname or alias.name.split(".")[0]] = (
                        module if alias.asname else importlib.import_module("kika"))
    return names


def _unknown_keywords(tree: ast.AST, names: dict) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = None
        if isinstance(func, ast.Name):
            target = names.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in names:
            target = getattr(names[func.value.id], func.attr, None)
        if target is None or not callable(target):
            continue
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
            continue
        found += [f"{target.__qualname__}({kw.arg}=...)" for kw in node.keywords
                  if kw.arg is not None and kw.arg not in parameters]
    return found


def test_the_guide_is_there():
    assert BLOCKS, f"no Python blocks found under {GUIDE}"


@pytest.mark.parametrize("page, code", BLOCKS)
def test_a_guide_block_matches_the_api(page, code):
    tree = ast.parse(code)
    names = _imported_names(tree)
    assert _unknown_keywords(tree, names) == []

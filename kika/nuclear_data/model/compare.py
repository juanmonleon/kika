"""Where two model objects differ, field by field.

Dataclass ``==`` stops at the first numpy array (``truth value of an array is
ambiguous``), so a round-trip gate over anything holding an ``XYs1d`` cannot
use it. :func:`modelDifferences` walks dataclasses, lists, tuples, dicts and
arrays and returns a path for every leaf that differs -- exactly, because a
round trip either keeps a value or it does not. ``NaN`` equals ``NaN`` here.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Iterable, List

import numpy as np

__all__ = ["modelDifferences"]


def modelDifferences(a, b, path: str = "", ignore: Iterable[str] = ()) -> List[str]:
    """Every path at which *a* and *b* differ; ``[]`` when they are the same.

    *ignore* names attributes skipped wherever they occur (``report``,
    ``provenance``…): bookkeeping that is not the physics being compared.
    """
    ignore = frozenset(ignore)
    out: List[str] = []
    _walk(a, b, path or "<root>", ignore, out)
    return out


def _walk(a, b, path, ignore, out) -> None:
    if a is b:
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            out.append(f"{path}: {type(a).__name__} != {type(b).__name__}")
            return
        for f in dataclasses.fields(a):
            if f.name in ignore:
                continue
            _walk(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", ignore, out)
        return
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append(f"{path}: keys {sorted(map(str, a))} != "
                       f"{sorted(map(str, b)) if isinstance(b, dict) else type(b).__name__}")
            return
        for key in a:
            _walk(a[key], b[key], f"{path}[{key!r}]", ignore, out)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape:
            out.append(f"{path}: shape {x.shape} != {y.shape}")
        elif not np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f"):
            out.append(f"{path}: arrays differ")
        return
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            out.append(f"{path}: length {len(a)} != "
                       f"{len(b) if isinstance(b, (list, tuple)) else type(b).__name__}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", ignore, out)
        return
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return
    try:
        same = a == b
        if not isinstance(same, bool):
            same = bool(np.all(same))
    except Exception:                                          # noqa: BLE001
        same = False
    if not same:
        out.append(f"{path}: {a!r} != {b!r}")

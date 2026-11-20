"""Isotope and module-definition loaders for NJOY."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, ENDF_MAT_TO_ZAID

_DATA_DIR = Path(__file__).parent / "data"
_MODULES_DIR = _DATA_DIR / "modules"

_isotopes_cache: dict[str, int] | None = None
_modules_cache: dict[str, Any] | None = None


def _build_isotope_map() -> dict[str, int]:
    """Build isotope name -> MAT mapping from ``ENDF_MAT_TO_ZAID``.

    Ground-state isotopes get plain names (e.g. ``"U235"``).
    When multiple MATs share the same ZAID the lowest MAT is treated as
    ground state; subsequent ones get an ``"m"`` suffix (e.g. ``"Am242m"``).
    """
    # Group MATs by ZAID
    zaid_to_mats: dict[int, list[int]] = defaultdict(list)
    for mat, zaid in ENDF_MAT_TO_ZAID.items():
        zaid_to_mats[zaid].append(mat)

    mapping: dict[str, int] = {}
    for zaid, mats in zaid_to_mats.items():
        z = zaid // 1000
        a = zaid % 1000
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"Z{z}")
        base_name = f"{symbol}{a}" if a else f"{symbol}-nat"

        mats_sorted = sorted(mats)
        # First (lowest) MAT = ground state
        mapping[base_name] = mats_sorted[0]
        # Additional MATs = metastable states
        for i, mat in enumerate(mats_sorted[1:], start=1):
            suffix = "m" if i == 1 else f"m{i}"
            mapping[f"{base_name}{suffix}"] = mat

    return mapping


def load_isotopes() -> dict[str, int]:
    """Return the MAT number of every isotope name kika knows, as a dict.

    The names are built from kika's table of ENDF MAT numbers and their ZA values
    (``ENDF_MAT_TO_ZAID`` in ``kika/_constants.py``), not from a file. A name is
    the element symbol followed by the mass number (``"U235"``, no hyphen), or
    ``"<symbol>-nat"`` for a natural element (mass number 0). When several MATs
    share a ZA, the lowest MAT keeps the plain name and the next ones take the
    suffix ``m``, then ``m2`` and so on (``"Am242m"``).

    The dict is built on the first call and the same object is returned on the
    next ones, so a caller must not modify it.
    """
    global _isotopes_cache
    if _isotopes_cache is None:
        _isotopes_cache = _build_isotope_map()
    return _isotopes_cache


def load_module_definitions() -> dict[str, Any]:
    """Return the definitions of the NJOY modules shipped with kika, by module name.

    The definitions are the ``*.json`` files in ``kika/njoy/data/modules/``, one per
    module (``acer.json``, ``reconr.json`` ...). Each key is the upper-case ``name``
    of a file (``"ACER"``) and each value is the file's content: the keys ``name``,
    ``description``, ``tooltip`` and ``cards``, where ``cards`` lists the cards of
    the module with the parameters of each (name and type always; default and
    help text in most).

    The dict is built on the first call and the same object is returned on the
    next ones, so a caller must not modify it.
    """
    global _modules_cache
    if _modules_cache is None:
        _modules_cache = {}
        for filename in os.listdir(_MODULES_DIR):
            if filename.endswith(".json"):
                with open(_MODULES_DIR / filename, "r") as f:
                    mod = json.load(f)
                    _modules_cache[mod["name"].upper()] = mod
    return _modules_cache


def get_mat_number(isotope: str) -> int:
    """Return the MAT number of an isotope name (``'U235'`` gives ``9228``).

    Parameters
    ----------
    isotope
        A name as ``load_isotopes`` spells it, with the same case. Any other
        spelling (``"U-235"``, ``"u235"``) is not recognised.

    Returns
    -------
    int
        The MAT number. An unknown name does not raise: it returns 9228, the MAT
        of U-235.
    """
    return load_isotopes().get(isotope, 9228)

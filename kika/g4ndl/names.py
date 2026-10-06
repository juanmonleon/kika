"""Isotope identity: G4NDL file names and the targets users type.

A G4NDL file is named ``<Z>_<A>[m<M>]_<Element>`` — ``26_56_Iron``,
``27_58m1_Cobalt``, ``6_nat_Carbon`` (``G4ParticleHPNames::GetName``). The
English element name is checked against ``Z`` but is never the identity: the
library spells some of them its own way (``Berylium``), and ``Z`` is what
Geant4 builds the name from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER

__all__ = ["GEANT4_ELEMENT_NAMES", "IsotopeKey", "file_name", "parse_file_name",
           "parse_target"]

_FILE_RE = re.compile(r"^(\d+)_(\d+|nat)(?:m(\d+))?_([A-Za-z]+)$")
_TARGET_RE = re.compile(r"^([A-Za-z]{1,2})-?(\d+|nat)(?:-?[mM](\d*))?$")


@dataclass(frozen=True, order=True)
class IsotopeKey:
    """``Z``, ``A`` and isomeric state ``M``. ``A`` is ``None`` for ``nat``."""

    Z: int
    A: Optional[int]
    M: int = 0

    @property
    def symbol(self) -> str:
        return ATOMIC_NUMBER_TO_SYMBOL.get(self.Z, f"Z{self.Z}")

    @property
    def isNatural(self) -> bool:
        return self.A is None

    def __str__(self) -> str:
        mass = "nat" if self.A is None else str(self.A)
        return f"{self.symbol}{mass}" + (f"m{self.M}" if self.M else "")


def parse_file_name(stem: str) -> Optional[Tuple[IsotopeKey, str]]:
    """``"26_56_Iron"`` -> ``(IsotopeKey(26, 56, 0), "Iron")``; ``None`` if not a G4NDL name.

    ``stem`` is the file name without a ``.z`` suffix.
    """
    m = _FILE_RE.match(stem)
    if not m:
        return None
    a = None if m.group(2) == "nat" else int(m.group(2))
    return IsotopeKey(int(m.group(1)), a, int(m.group(3) or 0)), m.group(4)


#: ``G4ParticleHPNames::theString`` (``src/G4ParticleHPNames.cc:55-71``), Z = 1..100,
#: with Geant4's own spellings (``Berylium``, ``Phosphorous``, ``Platinium``):
#: the consumer builds the file name from this table, so a writer must too.
GEANT4_ELEMENT_NAMES = (
    "Hydrogen", "Helium", "Lithium", "Berylium", "Boron", "Carbon",
    "Nitrogen", "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium",
    "Aluminum", "Silicon", "Phosphorous", "Sulfur", "Chlorine", "Argon",
    "Potassium", "Calcium", "Scandium", "Titanium", "Vanadium", "Chromium",
    "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc",
    "Gallium", "Germanium", "Arsenic", "Selenium", "Bromine", "Krypton",
    "Rubidium", "Strontium", "Yttrium", "Zirconium", "Niobium", "Molybdenum",
    "Technetium", "Ruthenium", "Rhodium", "Palladium", "Silver", "Cadmium",
    "Indium", "Tin", "Antimony", "Tellurium", "Iodine", "Xenon",
    "Cesium", "Barium", "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
    "Promethium", "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium",
    "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium", "Hafnium",
    "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinium",
    "Gold", "Mercury", "Thallium", "Lead", "Bismuth", "Polonium",
    "Astatine", "Radon", "Francium", "Radium", "Actinium", "Thorium",
    "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium", "Curium",
    "Berkelium", "Californium", "Einsteinium", "Fermium",
)


def file_name(key: "IsotopeKey") -> str:
    """``IsotopeKey(26, 56)`` -> ``"26_56_Iron"``: the name Geant4 opens, without ``.z``."""
    if not 1 <= key.Z <= len(GEANT4_ELEMENT_NAMES):
        raise ValueError(f"Z={key.Z}: Geant4 names elements for Z = 1..100 only")
    mass = "nat" if key.A is None else str(key.A)
    iso = f"m{key.M}" if key.M else ""
    return f"{key.Z}_{mass}{iso}_{GEANT4_ELEMENT_NAMES[key.Z - 1]}"


TargetLike = Union[str, int, Tuple[int, ...], IsotopeKey]


def parse_target(target: TargetLike) -> IsotopeKey:
    """What a user may pass as ``target``.

    ``"Fe56"``, ``"Fe-56"``, ``"Am242m1"`` (``"Am242m"`` means ``m1``),
    ``"Cnat"``/``"C-nat"``, a ZA integer ``26056`` (``A=0`` is natural), a
    tuple ``(Z, A)`` or ``(Z, A, M)``, or an :class:`IsotopeKey`. The MCNP
    ``+400`` isomer offset is **not** accepted on a ZA: ``95642`` is
    ambiguous between conventions, and an ambiguous identity is the thing
    this module exists to prevent.
    """
    if isinstance(target, IsotopeKey):
        return target
    if isinstance(target, tuple):
        z, a, *m = target
        return IsotopeKey(int(z), None if a in (0, None, "nat") else int(a),
                          int(m[0]) if m else 0)
    if isinstance(target, int) or (isinstance(target, str) and target.strip().isdigit()):
        za = int(target)
        z, a = divmod(za, 1000)
        if a > 300:
            raise ValueError(f"ZA {za}: A={a} looks like an isomer offset; "
                             "write the isomer as 'Am242m1' or (95, 242, 1)")
        if z not in ATOMIC_NUMBER_TO_SYMBOL:
            raise ValueError(f"ZA {za}: unknown Z={z}")
        return IsotopeKey(z, a or None, 0)
    m = _TARGET_RE.match(str(target).strip())
    if not m:
        raise ValueError(f"cannot parse target {target!r}; use e.g. 'Fe56', 'Am242m1', 26056")
    symbol = m.group(1).capitalize()
    z = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
    if z is None:
        raise ValueError(f"unknown element symbol {symbol!r} in {target!r}")
    a = None if m.group(2) == "nat" else int(m.group(2))
    iso = m.group(3)
    return IsotopeKey(z, a, 0 if iso is None else int(iso or 1))

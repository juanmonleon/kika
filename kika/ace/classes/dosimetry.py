"""A dosimetry ACE table (``.NNy``), decoded.

A dosimetry table holds reaction cross sections and nothing else: no energy
grid shared between reactions, no secondary distributions, no total. Each
reaction keeps its own energies and its own interpolation law, which is why it
is not an :class:`~kika.ace.classes.ace.Ace` with fewer blocks. MCNP reads it
for FM tallies and other reaction-rate edits; it is never transported.

**Layout** (NJOY2016 ``acedo.f90``, ``dosout``). NXS(1) is the length of XSS,
NXS(2) the target's ZA and NXS(4) the number of reactions NTR. JXS(1) points at
LONE (always 1), JXS(3) at MTR, JXS(6) at LSIG, JXS(7) at SIGD and JXS(22) at
the end of the table. MTR holds NTR reaction numbers; LSIG the location of each
reaction's data relative to SIGD (the first is 1). Each reaction's data is a
TAB1 without its CONT: NR, then NBT(1..NR) and INT(1..NR) as two arrays (NR = 0
for a single lin-lin region), NE, the energies in MeV and the cross sections
in barns.

**Production cross sections** (MF 9/10) are named as IRDFF-II's ACE library
names them:

- ``MT + 1000·(10 + LFS)`` for a reaction with one product, as ACER and FRENDY
  name them: MT 4 to the first isomer (LFS 1) is 11004;
- ``(50 + LFS)·10⁶ + ZA`` for MT 5, whose products are many and are told apart
  only by what they are: Mg(n,x)Na-24 is 50011024.

:func:`production_mt`, :func:`split_production_mt` and :func:`production_za`
convert. Older tables (NJOY's ACER) write every MT 5 product of one LFS under
one ``MT + 1000·(10 + LFS)`` number; a table read with a repeated number keeps
every reaction, in order, and :attr:`DosimetryAce.repeated_numbers` names them,
since a code that looks a number up finds only one.

**Units.** Every energy here is in **eV**, the cross sections in **barns**
(eV·b for the energy-deposition numbers MT 301 and MT 444).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

#: MeV → eV.
EV_PER_MEV = 1.0e6


#: The one reaction whose productions are numbered by their product.
MANY_PRODUCTS_MT = 5


def production_mt(mt: int, lfs: int, za: Optional[int] = None) -> int:
    """The dosimetry-table number of the production cross section of ``mt`` to final state ``lfs``.

    ``za`` (1000·Z + A of the product) is needed for MT 5 only.
    """
    if not 0 <= int(lfs) <= 98:
        raise ValueError(f"LFS = {lfs} is not a final state (0-98)")
    if int(mt) == MANY_PRODUCTS_MT:
        if za is None or not 0 < int(za) < 1000000:
            raise ValueError(f"a production of MT 5 is numbered by its product; ZA = {za} is not one")
        return (50 + int(lfs)) * 1000000 + int(za)
    return int(mt) + 1000 * (10 + int(lfs))


def split_production_mt(number: int) -> Optional[Tuple[int, int]]:
    """``(MT, LFS)`` of a production number from :func:`production_mt`, ``None`` for a plain MT."""
    number = int(number)
    if number >= 50000000:
        return MANY_PRODUCTS_MT, number // 1000000 - 50
    if number < 10000:
        return None
    return number % 1000, number // 1000 - 10


def production_za(number: int) -> Optional[int]:
    """The product's ZA a production number names (MT 5 only), ``None`` otherwise."""
    number = int(number)
    return number % 1000000 if number >= 50000000 else None


@dataclass
class DosimetryReaction:
    """One reaction of a dosimetry table: σ(E) with its interpolation regions.

    ``breakpoints`` and ``interpolation`` are the TAB1's NBT and INT; both are
    empty for a single lin-lin region, which the table spells NR = 0.
    """

    mt: int
    energies: np.ndarray          # eV
    cross_section: np.ndarray     # b
    breakpoints: List[int] = field(default_factory=list)
    interpolation: List[int] = field(default_factory=list)

    @property
    def production(self) -> Optional[Tuple[int, int]]:
        """``(MT, LFS)`` when this is an MF 10 production cross section, else ``None``."""
        return split_production_mt(self.mt)

    @property
    def lin_lin(self) -> bool:
        return not self.interpolation or set(self.interpolation) == {2}


@dataclass
class DosimetryAce:
    """The decoded reactions of a dosimetry ACE table. See the module docstring for units."""

    filename: Optional[str] = None
    header: object = None
    xss_data: Optional[np.ndarray] = None
    xss_int_words: Optional[np.ndarray] = None
    za: Optional[int] = None
    reactions: List[DosimetryReaction] = field(default_factory=list)

    @property
    def mt_numbers(self) -> List[int]:
        """The reaction numbers in the order the table writes them."""
        return [r.mt for r in self.reactions]

    @property
    def repeated_numbers(self) -> List[int]:
        """The reaction numbers the table writes more than once (empty for a well-formed table)."""
        numbers = self.mt_numbers
        return sorted({n for n in numbers if numbers.count(n) > 1})

    def reaction(self, mt: int) -> DosimetryReaction:
        """The reaction numbered ``mt``; KeyError if absent, ValueError if the table repeats it."""
        found = [r for r in self.reactions if r.mt == int(mt)]
        if not found:
            raise KeyError(f"no reaction {mt} in the table")
        if len(found) > 1:
            raise ValueError(f"the table writes reaction {mt} {len(found)} times (several MF 10 products)")
        return found[0]

    def __repr__(self) -> str:
        name = f"{self.header.zaid}{self.header.extension}" if self.header is not None else "?"
        return f"DosimetryAce({name}, {len(self.reactions)} reactions)"

"""MF26: secondary distributions of electro-atomic reactions (ENDF-6 §26).

The record layout is MF6's (a HEAD, then NK products, each a TAB1 yield and
the body its LAW states), so a section is an :class:`MF26MT`, MF6's class with
``MF=26``. MF26 uses three laws: LAW=1 (continuum energy spectra: the
bremsstrahlung photon, the knock-on electron of ionisation), LAW=2 (two-body
angular distribution, tabulated: the elastic electron) and LAW=8, which MF6
does not have: the energy transfer of excitation and bremsstrahlung, a TAB1 of
ET(E), the energy the incident electron loses (:class:`MF26LawEnergyTransfer`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ...utils import format_tab1
from ..mf6.base import MF6MT
from ..mf6.laws import MF6Law

__all__ = ["MF26MT", "MF26LawEnergyTransfer"]


@dataclass(repr=False)
class MF26MT(MF6MT):
    """One MT section of MF26: MF6's layout, with LAW=8 among its laws."""

    _mf: int = 26
    _lct: int = 0

    def __repr__(self) -> str:
        laws = ",".join(str(p.law) for p in self.products)
        return f"MF26MT({self.number}, NK={self.num_products}, LAW=[{laws}])"


@dataclass
class MF26LawEnergyTransfer(MF6Law):
    """LAW=8: ET(E), the energy the incident electron loses (eV), one TAB1."""

    law: int = 8
    c1: float = 0.0
    c2: float = 0.0
    l1: int = 0
    l2: int = 0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    transfer: List[float] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num, pad):
        return format_tab1(self.c1, self.c2, self.l1, self.l2, self.interp, self.energies, self.transfer,
                           mat, mf, mt, line_num, pad=pad.pairs, interp_pad=pad.interp)

    def describe(self) -> str:
        return f"LAW=8 (energy transfer, {len(self.energies)} points)"

"""MF14: photon angular distributions (ENDF-6 §14).

One section per MT that has photons in MF12 or MF13, describing each photon's
angular distribution in the laboratory frame. ``LI=1`` says every photon of the
MT is isotropic and the section is its HEAD alone — 30 556 of the 30 579 MF14
sections in ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 (census of 2026-10-09). With
``LI=0`` the first ``NI`` photons are isotropic, one CONT record ``EG, ES``
each, and the remaining ``NK-NI`` are anisotropic, one TAB2 over incident
energy each, holding Legendre coefficients (``LTT=1``, a LIST per energy) or a
tabulated p(μ) (``LTT=2``, a TAB1 per energy).

**LTT=2 has no witness** in the three libraries. It is implemented from ENDF-6
and exercised against kika's own emitter, as MF6 LAW=5 was before its witness
turned up.

The photons are matched to MF12/MF13 by ``(EG, ES)``, never by position:
``NK`` here differs from the number of photons of MF12 LO=2 (NT) in 267 of
29 581 sections, so the order cannot be trusted to line up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mf12.base import emit_cont, emit_list
from ..mt import MT
from ...utils import (
    ENDF_FORMAT_PRECISE,
    PadStyle,
    format_endf_send_record,
    format_tab1,
    format_tab2,
)

#: ``LTT`` → how an anisotropic photon's distribution is given.
LTT_NAMES = {1: "Legendre coefficients", 2: "tabulated p(mu)"}


@dataclass
class IsotropicPhoton:
    """An isotropic photon of an LI=0 section: one CONT record."""

    eg: float = 0.0
    es: float = 0.0
    #: The record's four unused fields, kept as read.
    rest: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class AngularNode:
    """The distribution at one incident energy: a LIST (LTT=1) or a TAB1 (LTT=2)."""

    energy: float = 0.0
    c1: float = 0.0
    l1: int = 0
    l2: int = 0
    #: LTT=1: the LIST header's sixth field (unused). LTT=2: unused.
    n2: int = 0
    #: LTT=1: a_1 … a_NL (a_0 = 1 is implied and not written).
    coefficients: List[float] = field(default_factory=list)
    #: LTT=2: the TAB1 of p(μ).
    interp: List[Tuple[int, int]] = field(default_factory=list)
    mu: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)


@dataclass
class AnisotropicPhoton:
    """An anisotropic photon: a TAB2 over incident energy, one node per energy."""

    eg: float = 0.0
    es: float = 0.0
    l1: int = 0
    l2: int = 0
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    nodes: List[AngularNode] = field(default_factory=list)


@dataclass
class MF14MT(MT):
    """One MT section of MF14."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    #: ``LI``: 1 all photons isotropic (HEAD only), 0 otherwise.
    _li: int = 1
    #: ``LTT``: 1 Legendre, 2 tabulated. Written as read even when LI=1.
    _ltt: int = 0
    #: ``NK`` as written. With LI=1 there is no body to count it from, and it
    #: is not always the photon count of MF12 (see the module docstring).
    _nk: int = 0
    #: ``NI`` as written, for the same reason as ``_nk``: with LI=1 the HEAD is
    #: all there is.
    _ni: int = 0
    _mat: Optional[int] = None
    _mf: int = 14
    isotropic: List[IsotropicPhoton] = field(default_factory=list)
    anisotropic: List[AnisotropicPhoton] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def li(self) -> int:
        return self._li

    @property
    def ltt(self) -> int:
        return self._ltt

    @property
    def all_isotropic(self) -> bool:
        return self._li == 1

    @property
    def num_photons(self) -> int:
        return self._nk if self._li == 1 else len(self.isotropic) + len(self.anisotropic)

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        ni = self._ni if self._li == 1 else len(self.isotropic)
        head, line_num = emit_cont(self._za, self._awr, self._li, self._ltt,
                                   self.num_photons, ni, mat, mf, mt, 1)
        lines = [head]
        if self._li != 1:
            for photon in self.isotropic:
                line, line_num = emit_cont(photon.eg, photon.es, *photon.rest,
                                           mat, mf, mt, line_num)
                lines.append(line)
            for photon in self.anisotropic:
                block, line_num = format_tab2(
                    photon.eg, photon.es, photon.l1, photon.l2, photon.tab2_interp,
                    len(photon.nodes), mat, mf, mt, line_num, interp_pad=self.pad.interp)
                lines.extend(block)
                for node in photon.nodes:
                    if self._ltt == 2:
                        block, line_num = format_tab1(
                            node.c1, node.energy, node.l1, node.l2, node.interp,
                            node.mu, node.p, mat, mf, mt, line_num,
                            pad=self.pad.pairs, interp_pad=self.pad.interp,
                            data_format=ENDF_FORMAT_PRECISE)
                    else:
                        block, line_num = emit_list(
                            node.c1, node.energy, node.l1, node.l2, node.n2,
                            node.coefficients, mat, mf, mt, line_num, self.pad.values)
                    lines.extend(block)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        if self._li == 1:
            return f"MF14MT({self.number}, LI=1, NK={self._nk})"
        return (f"MF14MT({self.number}, LTT={self._ltt}, NI={len(self.isotropic)}, "
                f"anisotropic={len(self.anisotropic)})")

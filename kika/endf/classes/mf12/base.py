"""MF12 and MF13: photon production multiplicities and cross sections (ENDF-6 §12, §13).

Both files list the photons a reaction emits, one TAB1 per photon. They differ
in what the TAB1 holds and in one option:

* **MF12 LO=1** gives a *multiplicity* y_k(E) per photon, and an MF12 section
  can instead be **LO=2**, a table of *transition probabilities* between the
  levels of the residual nucleus (the level scheme of a two-body reaction).
  LO=2 is 97 % of the MF12 sections in ENDF/B-VIII.1, JEFF-4.0 and JENDL-5
  (census of 2026-10-09, ``docs/library/endf_photons_e5_plan.md`` §2.2 in the
  workspace repo).
* **MF13** gives a *production cross section* σ_k(E) per photon. It has no LO:
  its layout is MF12 LO=1's, and the HEAD's third field is unused.

A subsection's header carries ``EG, ES, LP, LF``: the photon energy (0 for a
continuum, whose spectrum is in MF15), the energy of the level it comes from,
the primary-photon flag (LP=2: E_γ = EG + AWR/(AWR+1)·E) and the spectrum flag
(LF=1 tabulated in MF15, LF=2 discrete). With NK>1 a total TAB1 comes first.

**Everything a record header says is stored, not only what ENDF-6 gives a
meaning.** The sections are re-emitted byte for byte, and a field the format
calls unused is still a field some evaluation may have written. What the
format derives (NR, NP, NK, NW, NT) is recomputed from the data, so a section
built or edited in memory cannot write a count that disagrees with its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_INT,
    ENDF_FORMAT_PRECISE,
    PadStyle,
    format_data_values,
    format_endf_data_line,
    format_endf_send_record,
    format_tab1,
)

#: Floats go through the *precise* formatter: the canonical ``d.dddddd+e``
#: unless a fixed-point field is strictly closer to the value, which is what
#: keeps JEFF-4.0 B-10's ``20000000.1`` (a step at 20 MeV) from coming back as
#: ``2.000000+7``, a different number. A value read from a canonical field ties
#: with it and keeps it, so nothing canonical moves.
_HEAD_FORMATS = [ENDF_FORMAT_PRECISE, ENDF_FORMAT_PRECISE, ENDF_FORMAT_INT,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]


def emit_cont(c1, c2, l1, l2, n1, n2, mat, mf, mt, line_num) -> Tuple[str, int]:
    """One CONT/HEAD record, floats in C1-C2 and integers in C3-C6."""
    return format_endf_data_line([c1, c2, l1, l2, n1, n2], mat, mf, mt, line_num,
                                 formats=_HEAD_FORMATS), line_num + 1


def emit_list(c1, c2, l1, l2, n2, values, mat, mf, mt, line_num,
              pad: str) -> Tuple[List[str], int]:
    """A LIST record: the header carrying NPL = len(values), then the values."""
    head, line_num = emit_cont(c1, c2, l1, l2, len(values), n2, mat, mf, mt, line_num)
    body, line_num = format_data_values(list(values), mat, mf, mt, line_num,
                                        formats=[ENDF_FORMAT_PRECISE] * len(values),
                                        pad=pad)
    return [head] + body, line_num


@dataclass
class PhotonTable:
    """One TAB1 of MF12 LO=1, MF13 or MF15, with its header fields as read.

    In MF12/MF13 ``c1, c2, l1, l2`` are ``EG, ES, LP, LF`` of a photon (or the
    zeros of the total); in MF15 they are the weight's ``0, 0, 0, LF`` or a
    distribution's ``0, E, 0, 0``. Named for what they are in MF12, the file
    that has most of them.
    """

    c1: float = 0.0
    c2: float = 0.0
    l1: int = 0
    l2: int = 0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    @property
    def eg(self) -> float:
        """``EG``: the photon energy (eV); 0 for a continuum."""
        return self.c1

    @property
    def es(self) -> float:
        """``ES``: the energy of the level the photon comes from (eV)."""
        return self.c2

    @property
    def lp(self) -> int:
        """``LP``: 0 or 1 an ordinary photon, 2 a primary one."""
        return self.l1

    @property
    def lf(self) -> int:
        """``LF``: 1 the spectrum is tabulated in MF15, 2 a discrete line."""
        return self.l2

    def emit(self, mat, mf, mt, line_num, pad: PadStyle) -> Tuple[List[str], int]:
        return format_tab1(self.c1, self.c2, self.l1, self.l2, self.interp,
                           self.x, self.y, mat, mf, mt, line_num,
                           pad=pad.pairs, interp_pad=pad.interp,
                           data_format=ENDF_FORMAT_PRECISE)


@dataclass
class MF12MT(MT):
    """One MT section of MF12 (``LO=1`` multiplicities or ``LO=2`` a level scheme)."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    #: ``LO``: 1 multiplicities, 2 transition probabilities.
    _lo: int = 1
    #: The HEAD's fourth field: ``LG`` for LO=2 (1 without and 2 with the
    #: conditional photon probabilities GP), unused (0) for LO=1.
    _l2: int = 0
    #: The HEAD's sixth field, unused by ENDF-6 and stored for the bytes.
    _n2: int = 0
    #: ``NS`` as the HEAD writes it for LO=2. ENDF-6 makes it the number of
    #: levels below the one this section describes, which is *not* the number
    #: of transitions NT, so it cannot be derived and is kept.
    _ns: int = 0
    _mat: Optional[int] = None
    _mf: int = 12
    #: LO=1 with NK>1: the total multiplicity (or, in MF13, the total cross
    #: section). ``None`` with a single photon, where the format writes none.
    total: Optional[PhotonTable] = None
    #: LO=1: one table per photon, in the file's order.
    photons: List[PhotonTable] = field(default_factory=list)
    #: LO=2: the LIST header's ``ES_NS`` (energy of the level the section
    #: describes), ``LP`` and its unused fourth field, and the flat body:
    #: ``(ES_i, TP_i[, GP_i])`` per lower level, ``LG+1`` values each.
    es_ns: float = 0.0
    list_c2: float = 0.0
    lp: int = 0
    list_l2: int = 0
    transition_values: List[float] = field(default_factory=list)
    #: The LIST header's ``NT`` as written. Normally NPL/(LG+1); kept rather
    #: than derived so a section whose NPL and NT disagree is written as it came.
    nt: int = 0
    pad: PadStyle = field(default_factory=PadStyle)

    # ------------------------------------------------------------------
    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def atomic_weight_ratio(self) -> Optional[float]:
        return self._awr

    @property
    def lo(self) -> int:
        return self._lo

    @property
    def lg(self) -> int:
        """``LG`` of an LO=2 section: 2 when GP is given, 1 when it is not."""
        return self._l2

    @property
    def num_photons(self) -> int:
        """``NK`` for LO=1, ``NS`` (the levels below this one) for LO=2."""
        if self._lo == 2:
            return self._ns
        return len(self.photons)

    @property
    def transitions(self) -> List[Tuple[float, ...]]:
        """LO=2: ``(ES_i, TP_i)`` or ``(ES_i, TP_i, GP_i)`` per lower level."""
        width = self._l2 + 1
        v = self.transition_values
        return [tuple(v[i:i + width]) for i in range(0, len(v), width)]

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        line_num = 1
        if self._lo == 2:
            head, line_num = emit_cont(self._za, self._awr, self._lo, self._l2,
                                       self._ns, self._n2, mat, mf, mt, line_num)
            body, line_num = emit_list(self.es_ns, self.list_c2, self.lp, self.list_l2,
                                       self.nt, self.transition_values,
                                       mat, mf, mt, line_num, self.pad.values)
            lines = [head] + body
        else:
            head, line_num = emit_cont(self._za, self._awr, self._head_l1(), self._l2,
                                       len(self.photons), self._n2, mat, mf, mt, line_num)
            lines = [head]
            tables = ([self.total] if self.total is not None else []) + self.photons
            for table in tables:
                block, line_num = table.emit(mat, mf, mt, line_num, self.pad)
                lines.extend(block)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def _head_l1(self) -> int:
        return self._lo

    def report_gaps(self) -> List[str]:
        """Nothing: every record of the section is read and kept."""
        return []

    def __repr__(self) -> str:
        if self._lo == 2:
            return (f"MF{self._mf}MT({self.number}, LO=2, LG={self._l2}, "
                    f"NS={self._ns}, NT={self.nt})")
        return f"MF{self._mf}MT({self.number}, NK={len(self.photons)})"


@dataclass
class MF13MT(MF12MT):
    """One MT section of MF13: photon production cross sections.

    The layout of MF12 LO=1, so the class is MF12's with ``LO`` fixed; the
    HEAD's third field, which carries LO in MF12, is unused here and is kept
    as read in ``_l1`` rather than assumed to be 0.
    """

    _mf: int = 13
    _l1: int = 0

    def _head_l1(self) -> int:
        return self._l1

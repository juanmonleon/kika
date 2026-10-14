"""MF9 and MF10: production of radioactive nuclides (ENDF-6 §9, §10).

Both files say, for one reaction MT, how much of each radioactive final state
LFS the reaction makes. They share one layout and differ in what the table holds:

    HEAD  ZA, AWR, LIS, 0, NS, 0
    for each of the NS final states:
        TAB1  QM, QI, IZAP, LFS, NR, NP / E_int / Y(E)   (MF9: multiplicity)
                                                  σ(E)   (MF10: cross section)
    SEND

``IZAP`` is the ZA of the product, ``LFS`` its level index, ``QM`` the Q to the
product's ground state and ``QI`` the Q to the level, so ``QM - QI`` is the
level's excitation energy. MF8 lists the same final states and points at MF9 or
MF10 through LMF; the MF9/MF10 records name their product themselves, so they
are read and written without MF8.

**Everything a record header says is stored**, unused fields included, as in
:mod:`..mf12.base`: the sections are re-emitted byte for byte. What the format
derives (NS, NR, NP) is recomputed from the data, so a section edited in memory
cannot write a count that disagrees with its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mf12.base import emit_cont
from ..mt import MT
from ...utils import ENDF_FORMAT_PRECISE, PadStyle, format_endf_send_record, format_tab1


@dataclass
class ProductionState:
    """One final state of MF9/MF10: ``QM, QI, IZAP, LFS`` and its TAB1."""

    qm: float = 0.0
    qi: float = 0.0
    izap: int = 0
    lfs: int = 0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    @property
    def level_energy(self) -> float:
        """``QM - QI``: the excitation energy of the final state (eV)."""
        return float(self.qm) - float(self.qi)

    def emit(self, mat, mf, mt, line_num, pad: PadStyle):
        return format_tab1(self.qm, self.qi, self.izap, self.lfs, self.interp,
                           self.x, self.y, mat, mf, mt, line_num,
                           pad=pad.pairs, interp_pad=pad.interp,
                           data_format=ENDF_FORMAT_PRECISE,
                           header_format=ENDF_FORMAT_PRECISE)


@dataclass
class MF9MT(MT):
    """One MT section of MF9: multiplicities of the radioactive final states."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    #: ``LIS``: the target's level index.
    _lis: int = 0
    #: The HEAD's fourth and sixth fields, unused by ENDF-6 and kept for the bytes.
    _l2: int = 0
    _n2: int = 0
    _mat: Optional[int] = None
    _mf: int = 9
    states: List[ProductionState] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def atomic_weight_ratio(self) -> Optional[float]:
        return self._awr

    @property
    def num_states(self) -> int:
        """``NS``, from the states held."""
        return len(self.states)

    def get_state(self, izap: int, lfs: int) -> Optional[ProductionState]:
        for state in self.states:
            if int(state.izap) == int(izap) and int(state.lfs) == int(lfs):
                return state
        return None

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        head, line_num = emit_cont(self._za, self._awr, self._lis, self._l2,
                                   len(self.states), self._n2, mat, mf, mt, 1)
        lines = [head]
        for state in self.states:
            block, line_num = state.emit(mat, mf, mt, line_num, self.pad)
            lines.extend(block)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        """Nothing: every record of the section is read and kept."""
        return []

    def __repr__(self) -> str:
        states = [(int(s.izap), int(s.lfs)) for s in self.states]
        return f"MF{self._mf}MT({self.number}, NS={len(self.states)}, states={states})"


@dataclass
class MF10MT(MF9MT):
    """One MT section of MF10: production cross sections of the final states.

    MF9's layout exactly, with σ(E) in barns where MF9 has a multiplicity.
    """

    _mf: int = 10

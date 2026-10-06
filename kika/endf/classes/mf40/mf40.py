"""MF40: covariances for the production of radioactive nuclei (ENDF-6 §40).

MF40 is the covariance file of MF10, the cross sections for producing each
final state LFS of a reaction. Its records are File 33's, one level deeper:

    HEAD  ZA, AWR, LIS, 0, NS, 0
    for each of the NS final states:
        CONT  QM, QI, IZAP, LFS, 0, NL
        NL subsections, each exactly an MF33 subsection
            CONT  XMF1, XLFS1, MAT1, MT1, NC, NI
            NC and NI sub-subsections (LTY, LB as in §33)

So a block is named by two pairs, (MT, LFS) for the row and (MT1, XLFS1) for
the column, and everything below the subsection CONT is read and written by
the MF33 code (:func:`~kika.endf.parsers.parse_mf33.parse_subsection`,
:func:`~kika.endf.classes.mf33.mf33.emit_subsection`). :meth:`MF40State.as_mf33`
gives one final state the shape of an ``MF33MT``, so the MF33 decoders sum its
records without a copy of them here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..mf33.mf33 import MF33MT, Subsection, emit_subsection
from ..mt import MT


@dataclass
class MF40State:
    """One final state: ``QM, QI, IZAP, LFS, NL`` and its subsections."""

    qm: float = 0.0
    qi: float = 0.0
    izap: int = 0
    lfs: int = 0
    nl: int = 0
    subsections: List[Subsection] = field(default_factory=list)

    def as_mf33(self, mt: int, mat: Optional[int], za=None, awr=None) -> MF33MT:
        """This state's subsections as an ``MF33MT`` (``_mf`` = 40), sharing them."""
        view = MF33MT(number=mt, _za=za, _awr=awr, _mtl=0, _nl=self.nl, _mat=mat, _mf=40)
        view._subsections = self.subsections
        return view


@dataclass
class MF40MT(MT):
    """One MT section of MF40: the covariances of its NS final states."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _lis: Optional[int] = None
    _ns: Optional[int] = None
    _mat: Optional[int] = None
    _mf: int = 40
    states: List[MF40State] = field(default_factory=list)

    @property
    def num_states(self) -> int:
        return self._ns if self._ns is not None else len(self.states)

    def get_state(self, lfs: int) -> Optional[MF40State]:
        for state in self.states:
            if int(state.lfs) == int(lfs):
                return state
        return None

    def __str__(self) -> str:
        from ...utils import (
            ENDF_FORMAT_FLOAT,
            ENDF_FORMAT_INT,
            ENDF_FORMAT_INT_ZERO,
            format_endf_data_line,
        )

        mat = self._mat if self._mat is not None else 0
        mf, mt = 40, self.number

        def blank(line: str) -> str:
            return line[:75] + "     "

        lines = [blank(format_endf_data_line(
            [self._za, self._awr, self._lis or 0, 0, self.num_states, 0],
            mat, mf, mt, 0,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        ))]
        for state in self.states:
            lines.append(blank(format_endf_data_line(
                [state.qm, state.qi, state.izap, state.lfs, 0, state.nl],
                mat, mf, mt, 0,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT],
            )))
            for subsection in state.subsections:
                lines.extend(emit_subsection(subsection, mat, mf, mt))
        lines.append(format_endf_data_line(
            [0, 0, 0, 0, 0, 0], mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        ))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MF40MT({self.number}, NS={self.num_states}, LFS="
                f"{[int(s.lfs) for s in self.states]})")

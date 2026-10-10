"""MF28: atomic relaxation data (ENDF-6 §28).

When a photon or electron knocks an electron out of a subshell, the vacancy
is filled by an electron from a higher subshell and the energy released
leaves as a fluorescence photon (radiative) or a second, Auger or
Coster-Kronig, electron (non-radiative). MF28 tabulates, per subshell, every
such transition and its probability. It lives on its own sublibrary (NSUB=6,
the ``ard`` tapes of ENDF/B-VIII.0, from EADL) under one MT, 533.

Layout of MT533::

    HEAD  [ZA, AWR, 0, 0, NSS, 0]
    NSS × LIST [SUBI, 0, 0, 0, 6·(NTR+1), NTR /
                EBI, ELN, 0, 0, 0, 0,
                (SUBJ, SUBK, ETR, FTR, 0, 0) × NTR]

``SUBI`` is the subshell's ENDF designator (1 = K, 2 = L1, ...; the MF23
subshell MT is ``533 + SUBI``), ``EBI`` its binding energy in eV and ``ELN``
its number of electrons when the atom is neutral. A transition fills the
vacancy from ``SUBJ`` and, if ``SUBK`` is not zero, ejects an electron from
``SUBK``; ``SUBK = 0`` means a photon is emitted instead. ``ETR`` is the
energy of the emitted particle in eV and ``FTR`` the fraction of vacancies
in SUBI filled this way.

Everything the LIST bodies carry is kept in ``values``, the four fields
ENDF-6 calls zero included, so a section re-emits as it came.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mf12.base import emit_cont, emit_list
from ..mf23.base import subshell_name
from ..mt import MT
from ...utils import PadStyle, format_endf_send_record


@dataclass
class RelaxationTransition:
    """One way of filling a vacancy: from ``subj``, ejecting from ``subk`` (0: a photon)."""

    subj: int
    subk: int
    energy: float
    probability: float

    @property
    def radiative(self) -> bool:
        return self.subk == 0


@dataclass
class RelaxationSubshell:
    """One subshell's LIST: its binding energy, occupancy and transitions.

    ``values`` is the LIST body as read; the properties decode it, so an edit
    to ``values`` is what gets written.
    """

    subi: int
    c2: float = 0.0
    l1: int = 0
    l2: int = 0
    ntr_as_written: int = 0
    values: List[float] = field(default_factory=list)

    @property
    def name(self) -> Optional[str]:
        return subshell_name(self.subi)

    @property
    def binding_energy(self) -> float:
        """``EBI`` in eV."""
        return self.values[0]

    @property
    def electrons(self) -> float:
        """``ELN``: the number of electrons in the subshell of the neutral atom."""
        return self.values[1]

    @property
    def transitions(self) -> List[RelaxationTransition]:
        v = self.values
        return [RelaxationTransition(int(round(v[i])), int(round(v[i + 1])), v[i + 2], v[i + 3])
                for i in range(6, len(v), 6)]


@dataclass
class MF28MT(MT):
    """MF28/MT533: the relaxation data of every subshell of an element."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _l1: int = 0
    _l2: int = 0
    _n2: int = 0
    _mat: Optional[int] = None
    _mf: int = 28
    subshells: List[RelaxationSubshell] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def z(self) -> Optional[int]:
        za = self.zaid
        return za // 1000 if za is not None else None

    @property
    def num_subshells(self) -> int:
        return len(self.subshells)

    def subshell(self, designator: int) -> Optional[RelaxationSubshell]:
        for shell in self.subshells:
            if shell.subi == designator:
                return shell
        return None

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        head, line_num = emit_cont(self._za, self._awr, self._l1, self._l2,
                                   len(self.subshells), self._n2, mat, mf, mt, 1)
        lines = [head]
        for shell in self.subshells:
            block, line_num = emit_list(float(shell.subi), shell.c2, shell.l1, shell.l2,
                                        shell.ntr_as_written, shell.values,
                                        mat, mf, mt, line_num, self.pad.values)
            lines.extend(block)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        """Nothing: every record of the section is read and kept."""
        return []

    def binding_energies(self) -> List[Tuple[str, float]]:
        """``[(name, EBI eV), ...]`` in the file's subshell order."""
        return [(s.name or str(s.subi), s.binding_energy) for s in self.subshells]

    def __repr__(self) -> str:
        return f"MF28MT({self.number}, NSS={len(self.subshells)})"

"""MF15: continuous photon energy spectra (ENDF-6 §15).

The spectrum of the continuum photon (``EG=0, LF=1``) of an MT in MF12 or
MF13, as ``NC`` partial distributions: each a weight p_j(E) and, for ``LF=1``
— the only law ENDF-6 defines for MF15 — a TAB2 over incident energy holding a
tabulated g_j(E'←E) per energy. Shaped like MF5's ``LF=1``, which is why the
classes look alike.

In the three libraries every section has ``NC=1`` (850 of 850, census of
2026-10-09). ``NC>1`` costs nothing here — it is the same subsection repeated —
and is exercised against kika's own emitter. An LF other than 1 is refused by
the parser: ENDF-6 gives MF15 no other layout, so a guessed length would be a
guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mf12.base import PhotonTable, emit_cont
from ..mt import MT
from ...utils import PadStyle, format_endf_send_record, format_tab2


@dataclass
class PhotonSpectrum:
    """One partial distribution: its weight p(E) and the tabulated g(E'←E)."""

    #: The weight's TAB1: ``0, 0, 0, LF`` in the header and p(E) in the table.
    weight: PhotonTable = field(default_factory=PhotonTable)
    #: The TAB2 header's four leading fields, as read (``0, 0, 0, 0``).
    tab2_fields: Tuple[float, float, int, int] = (0.0, 0.0, 0, 0)
    tab2_interp: List[Tuple[int, int]] = field(default_factory=list)
    #: One TAB1 per incident energy: header ``0, E, 0, 0``, then g(E').
    distributions: List[PhotonTable] = field(default_factory=list)

    @property
    def lf(self) -> int:
        return self.weight.l2

    @property
    def incident_energies(self) -> List[float]:
        return [d.c2 for d in self.distributions]


@dataclass
class MF15MT(MT):
    """One MT section of MF15."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    #: The HEAD's third, fourth and sixth fields, unused by ENDF-6.
    _l1: int = 0
    _l2: int = 0
    _n2: int = 0
    _mat: Optional[int] = None
    _mf: int = 15
    spectra: List[PhotonSpectrum] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def num_partials(self) -> int:
        return len(self.spectra)

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        head, line_num = emit_cont(self._za, self._awr, self._l1, self._l2,
                                   len(self.spectra), self._n2, mat, mf, mt, 1)
        lines = [head]
        for spectrum in self.spectra:
            block, line_num = spectrum.weight.emit(mat, mf, mt, line_num, self.pad)
            lines.extend(block)
            block, line_num = format_tab2(*spectrum.tab2_fields, spectrum.tab2_interp,
                                          len(spectrum.distributions), mat, mf, mt,
                                          line_num, interp_pad=self.pad.interp)
            lines.extend(block)
            for distribution in spectrum.distributions:
                block, line_num = distribution.emit(mat, mf, mt, line_num, self.pad)
                lines.extend(block)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        return f"MF15MT({self.number}, NC={len(self.spectra)})"

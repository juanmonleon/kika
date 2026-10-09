"""An MF8 section of a neutron tape (radioactive products of a reaction, roadmap E6).

ENDF-6 §8.2: an incident-neutron evaluation's MF8 says which radioactive
nuclides a reaction produces and points at MF9/MF10 for how many. kika does not
model activation yet (E6, deferred until someone needs it), so the section is
kept as the lines it was read from and written back unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..mt import MT


@dataclass
class MF8Raw(MT):
    """Any MF8 MT other than 454, 457 and 459, kept verbatim."""

    _mat: Optional[int] = None
    _mf: int = 8
    raw_lines: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        from ...utils import format_endf_send_record

        mat = self._mat if self._mat is not None else 0
        return "\n".join(list(self.raw_lines) + [format_endf_send_record(mat, 8)])

    def report_gaps(self) -> List[str]:
        return [f"MF8/MT{self.number}: radioactive products of a reaction (roadmap "
                f"E6) are kept as read and not interpreted"]

    def __repr__(self) -> str:
        return f"MF8Raw(MT{self.number}, {len(self.raw_lines)} lines)"

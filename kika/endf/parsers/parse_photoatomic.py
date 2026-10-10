"""Parsers for the photo-atomic files MF23 and MF27, and the relaxation file MF28.

MF23 and MF27 are the photo-atomic sublibrary (NSUB=3); MF28 is the atomic
relaxation sublibrary (NSUB=6). They are parsed together because a user of
one needs the others: the subshell photoelectric cross sections of MF23 are
indexed by the same subshell designators MF28 relaxes.

The record walk has the discipline of :mod:`~kika.endf.parsers.parse_photons`:
every record of a section must be read (a section with lines left over raises,
naming its MT), and a section that fails costs only its own MT, which is
recorded in ``mf.parse_errors``.

**Fortran ``D`` exponents.** The EADL relaxation tapes of ENDF/B-VIII.0 write
some fields as ``6.00794D-6``, which is a Fortran double-precision literal and
not ENDF-6. :func:`kika._records.parse_number` returns None for it, and a LIST
body would then lose the field silently and shift every value after it. So
these three parsers rewrite ``D`` to ``E`` in the data columns before reading;
nothing else on the line is touched. The value is kept; the spelling is not,
and the section re-emits it canonically.
"""
import re
from typing import List

from ..classes.mf23.base import MF23MT, MF27MT, PhotoatomicTabMT
from ..classes.mf28.base import MF28MT, RelaxationSubshell
from ..utils import PaddingProbe, parse_line
from .parse_photons import _check_consumed, _file_parser, _int, _list, _table
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)

_FORTRAN_D = re.compile(r"(\d)[dD]([+-]?\d)")


def _without_fortran_d(lines: List[str]) -> List[str]:
    """The lines with ``1.0D+11`` rewritten as ``1.0E+11`` in columns 1-66."""
    out = []
    for line in lines:
        data = line[:66]
        if "D" in data or "d" in data:
            line = _FORTRAN_D.sub(r"\1E\2", data) + line[66:]
        out.append(line)
    return out


def _parse_tab_section(cls, lines: List[str], mt: int, mf: int) -> PhotoatomicTabMT:
    """HEAD + one TAB1, the whole of an MF23 or MF27 section."""
    if not lines:
        raise ValueError(f"MF{mf}/MT{mt}: empty section")
    lines = _without_fortran_d(lines)
    head = parse_line(lines[0])
    section = cls(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                  _l1=_int(head, "C3"), _l2=_int(head, "C4"),
                  _n1=_int(head, "C5"), _n2=_int(head, "C6"),
                  _mat=head.get("MAT"))
    if len(lines) < 3:
        raise ValueError(f"MF{mf}/MT{mt}: {len(lines)} record(s); a HEAD and a TAB1 need at least 3")
    probe = PaddingProbe()
    section.table, idx = _table(lines, 1, probe)
    if len(section.table.x) != len(section.table.y):
        raise ValueError(f"MF{mf}/MT{mt}: the TAB1 body is short of its NP")
    section.pad = probe.resolve()
    _check_consumed(idx, lines, mf, mt)
    return section


def parse_mf23_mt(lines: List[str], mt: int) -> MF23MT:
    """One MF23 section: σ(E) for a photon interaction (HEAD + TAB1)."""
    return _parse_tab_section(MF23MT, lines, mt, 23)


def parse_mf27_mt(lines: List[str], mt: int) -> MF27MT:
    """One MF27 section: a form factor or anomalous scattering factor (HEAD + TAB1)."""
    return _parse_tab_section(MF27MT, lines, mt, 27)


def parse_mf28_mt(lines: List[str], mt: int) -> MF28MT:
    """MF28/MT533: HEAD with NSS, then one LIST per subshell."""
    if not lines:
        raise ValueError(f"MF28/MT{mt}: empty section")
    lines = _without_fortran_d(lines)
    head = parse_line(lines[0])
    section = MF28MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                     _l1=_int(head, "C3"), _l2=_int(head, "C4"), _n2=_int(head, "C6"),
                     _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    for k in range(_int(head, "C5")):
        if idx >= len(lines):
            raise ValueError(f"MF28/MT{mt}: NSS={_int(head, 'C5')} but the section ends after {k} subshell(s)")
        header, values, idx = _list(lines, idx, probe)
        npl, ntr = _int(header, "C5"), _int(header, "C6")
        if len(values) != npl:
            raise ValueError(f"MF28/MT{mt} subshell {k + 1}: LIST declares NW={npl}, body has {len(values)}")
        if npl != 6 * (ntr + 1):
            raise ValueError(
                f"MF28/MT{mt} subshell {k + 1}: NW={npl} but NTR={ntr}; ENDF-6 requires "
                f"NW = 6·(NTR+1), so the transitions cannot be told apart"
            )
        subi = header.get("C1")
        section.subshells.append(RelaxationSubshell(
            subi=int(round(subi or 0)), c2=header.get("C2") or 0.0,
            l1=_int(header, "C3"), l2=_int(header, "C4"),
            ntr_as_written=ntr, values=list(values)))
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 28, mt)
    return section


parse_mf23 = _file_parser(23, parse_mf23_mt)
parse_mf27 = _file_parser(27, parse_mf27_mt)
parse_mf28 = _file_parser(28, parse_mf28_mt)

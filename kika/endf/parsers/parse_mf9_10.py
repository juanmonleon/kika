"""Parsers for MF9 and MF10 (production of radioactive nuclides, ENDF-6 §9, §10).

One module for two files because they are one layout: a HEAD, then one TAB1 per
final state. The record walk is strict, as for MF12-15 (:mod:`.parse_photons`):
every record of a section has to be read, and a section with records left over
raises, naming the MT, rather than being cut at a guessed length.
"""
from typing import List, Type

from ..classes.mf import MF
from ..classes.mf9.base import MF9MT, MF10MT, ProductionState
from ..utils import PaddingProbe, group_lines_by_mt_with_positions, parse_line, parse_tab1
from .parse_photons import _check_consumed, _float, _int, _pair_lines
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def _parse_section(lines: List[str], mt: int, cls: Type[MF9MT]) -> MF9MT:
    head = parse_line(lines[0])
    section = cls(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                  _lis=_int(head, "C3"), _l2=_int(head, "C4"), _n2=_int(head, "C6"),
                  _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    for _ in range(_int(head, "C5")):
        header, interp, x, y, idx = parse_tab1(lines, idx)
        probe.observe_pairs(lines, idx, len(x))
        probe.observe_interp(lines, idx - _pair_lines(len(x)), len(interp))
        section.states.append(ProductionState(
            qm=_float(header, "C1"), qi=_float(header, "C2"),
            izap=_int(header, "C3"), lfs=_int(header, "C4"),
            interp=[(int(a), int(b)) for a, b in interp], x=list(x), y=list(y)))
    section.pad = probe.resolve()
    _check_consumed(idx, lines, section._mf, mt)
    return section


def parse_mf9_mt(lines: List[str], mt: int) -> MF9MT:
    """One MF9 section: multiplicities of the radioactive final states."""
    return _parse_section(lines, mt, MF9MT)


def parse_mf10_mt(lines: List[str], mt: int) -> MF10MT:
    """One MF10 section: production cross sections of the radioactive final states."""
    return _parse_section(lines, mt, MF10MT)


def _parse_file(lines: List[str], number: int, parse_mt) -> MF:
    mf = MF(number=number)
    mf.num_lines = len(lines)
    groups, counts = group_lines_by_mt_with_positions(lines)
    for mt, mt_lines in groups.items():
        if mt == 0:
            continue
        body = [line for line in mt_lines if line[72:75].strip() not in ("", "0")]
        try:
            section = parse_mt(body, mt)
            section.num_lines = counts.get(mt, len(body))
            mf.add_section(section)
        except Exception as exc:                                 # noqa: BLE001
            mf.parse_errors[mt] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Error parsing MT{mt} in MF{number}: {mf.parse_errors[mt]}")
    return mf


def parse_mf9(lines: List[str]) -> MF:
    """Every MT of an MF9 block."""
    return _parse_file(lines, 9, parse_mf9_mt)


def parse_mf10(lines: List[str]) -> MF:
    """Every MT of an MF10 block."""
    return _parse_file(lines, 10, parse_mf10_mt)

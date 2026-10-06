"""Parser for MF40 (covariances for the production of radioactive nuclei).

ENDF-6 §40: a HEAD, then one CONT per final state LFS, each followed by NL
subsections that are File 33's records exactly. Those are read by
:func:`~kika.endf.parsers.parse_mf33.parse_subsection`, so MF33, MF31 and MF40
share one reader of the NC/NI sub-subsections.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf40.mf40 import MF40MT, MF40State
from ..utils import group_lines_by_mt_with_positions, parse_endf_id, parse_line
from ...utils import get_endf_logger
from .parse_mf33 import parse_subsection

logger = get_endf_logger(__name__)


def parse_mf40(lines: List[str]) -> MF:
    """Parse MF40 into an :class:`MF` of :class:`MF40MT`."""
    mf = MF(number=40)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf40_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
        except Exception as exc:
            mf.parse_errors[mt] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Error parsing MT{mt} in MF40: {exc}")
    return mf


def parse_mf40_mt(lines: List[str], mt: int) -> MF40MT:
    """One MT section: HEAD, then NS final states of NL subsections each."""
    if not lines:
        raise ValueError(f"no lines for MF40/MT{mt}")
    head = parse_line(lines[0])
    mat, _, _ = parse_endf_id(lines[0])
    section = MF40MT(
        number=mt,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _lis=int(head.get("C3") or 0),
        _ns=int(head.get("C5") or 0),
        _mat=mat,
    )

    idx = 1
    for _ in range(section._ns):
        if idx >= len(lines):
            break
        cont = parse_line(lines[idx])
        idx += 1
        state = MF40State(
            qm=cont.get("C1"),
            qi=cont.get("C2"),
            izap=int(cont.get("C3") or 0),
            lfs=int(cont.get("C4") or 0),
            nl=int(cont.get("C6") or 0),
        )
        for _ in range(state.nl):
            if idx >= len(lines):
                break
            subsection, idx = parse_subsection(lines, idx)
            state.subsections.append(subsection)
        section.states.append(state)
    return section

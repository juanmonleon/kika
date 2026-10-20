"""Parser for MF26 (electro-atomic secondary distributions, ENDF-6 §26).

MF6's record walk (:mod:`.parse_mf6`) with MF26's own law, LAW=8 (a TAB1 of
the energy transfer). Every record must be walked, as in MF6.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf26 import MF26LawEnergyTransfer, MF26MT
from ..utils import PaddingProbe, group_lines_by_mt_with_positions, parse_line, parse_tab1
from ...utils import get_endf_logger
from . import parse_mf6 as _mf6

logger = get_endf_logger(__name__)


def parse_mf26(lines: List[str]) -> MF:
    """Parse MF26 into an :class:`MF` of :class:`MF26MT` sections."""
    mf = MF(number=26)
    mf.num_lines = len(lines)
    groups, counts = group_lines_by_mt_with_positions(lines)
    for mt, mt_lines in groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf26_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in counts:
                section.num_lines = counts[mt]
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF26: {exc}")
    return mf


def parse_mf26_mt(lines: List[str], mt: int) -> MF26MT:
    head = parse_line(lines[0])
    section = MF26MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"), _jp=int(head.get("C3") or 0),
                     _lct=int(head.get("C4") or 0), _nk=int(head.get("C5") or 0), _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    for index in range(section._nk):
        product, idx = _product(lines, idx, mt, index, probe)
        section.products.append(product)
    section.pad = probe.resolve()
    if idx != len(lines):
        raise ValueError(f"MF26/MT{mt}: parsed {idx} of {len(lines)} records")
    return section


def _product(lines, idx, mt, index, probe):
    header = parse_line(lines[idx])
    if int(header.get("C4") or 0) != 8:
        return _mf6._parse_product(lines, idx, mt, index, probe)
    from ..classes.mf6.products import MF6Product
    head, y_interp, y_e, y_v, idx = parse_tab1(lines, idx)
    probe.observe_pairs(lines, idx, len(y_e))
    th, t_interp, t_e, t_v, idx = parse_tab1(lines, idx)
    probe.observe_pairs(lines, idx, len(t_e))
    law = MF26LawEnergyTransfer(c1=th.get("C1") or 0.0, c2=th.get("C2") or 0.0, l1=int(th.get("C3") or 0),
                                l2=int(th.get("C4") or 0), interp=list(t_interp), energies=list(t_e),
                                transfer=list(t_v))
    return MF6Product(zap=head.get("C1"), awp=head.get("C2"), lip=int(head.get("C3") or 0),
                      y_interp=list(y_interp), y_energies=list(y_e), y_values=list(y_v), law_data=law), idx

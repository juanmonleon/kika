"""Parser for MF8: decay data (MT457), fission yields (MT454/459), and the rest kept.

The record walk is strict, as for MF12-15 (:mod:`.parse_photons`): every record
of a section has to be read, and a layout ENDF-6 does not define raises, naming
the MT, rather than being stepped over. Measured on ENDF/B-VIII.1's decay
(3 821 nuclides) and fission-yield (31 evaluations) sublibraries, roadmap E7.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf12.base import PhotonTable
from ..classes.mf8.decay import ContinuousSpectrum, DecaySpectrum, DiscreteLine, MF8MT457
from ..classes.mf8.raw import MF8Raw
from ..classes.mf8.yields import FissionYields, MF8FissionYields
from ..utils import (PaddingProbe, group_lines_by_mt_with_positions, parse_data_values,
                     parse_line, parse_tab1)
from .parse_photons import _check_consumed, _float, _int, _pair_lines
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)

LINE_END = chr(13) + chr(10)


def _list(lines, idx, probe):
    header = parse_line(lines[idx])
    values, idx = parse_data_values(lines, idx + 1, _int(header, "C5"))
    probe.observe_values(lines, idx, len(values))
    return header, list(values), idx


def _fields(header) -> list:
    return [_float(header, "C1"), _float(header, "C2"), _int(header, "C3"),
            _int(header, "C4"), _int(header, "C5"), _int(header, "C6")]


def parse_mf8_mt457(lines: List[str], mt: int = 457) -> MF8MT457:
    head = parse_line(lines[0])
    section = MF8MT457(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                       _lis=_int(head, "C3"), _liso=_int(head, "C4"),
                       _nst=_int(head, "C5"), _mat=head.get("MAT"))
    nsp = _int(head, "C6")
    probe = PaddingProbe()
    header, values, idx = _list(lines, 1, probe)
    section.halflife, section.dhalflife = _float(header, "C1"), _float(header, "C2")
    section.energies_l1, section.energies_l2 = _int(header, "C3"), _int(header, "C4")
    section.energies_n2 = _int(header, "C6")
    section.average_energies = values
    header, values, idx = _list(lines, idx, probe)
    section.spin, section.parity = _float(header, "C1"), _float(header, "C2")
    section.modes_l1, section.modes_l2 = _int(header, "C3"), _int(header, "C4")
    section.modes_n2 = _int(header, "C6")
    section.modes_raw = values
    for _ in range(nsp):
        header, values, idx = _list(lines, idx, probe)
        spectrum = DecaySpectrum(styp=_float(header, "C2"), lcon=_int(header, "C3"),
                                 head=values, c1=_float(header, "C1"),
                                 l2=_int(header, "C4"), n2=_int(header, "C6"))
        ner = _int(header, "C6")
        if spectrum.lcon != 1:
            for _ in range(ner):
                header, values, idx = _list(lines, idx, probe)
                spectrum.lines.append(DiscreteLine(
                    er=_float(header, "C1"), der=_float(header, "C2"), values=values,
                    l1=_int(header, "C3"), l2=_int(header, "C4"), n2=_int(header, "C6")))
        if spectrum.lcon != 0:
            tab_header, interp, x, y, idx = parse_tab1(lines, idx)
            probe.observe_pairs(lines, idx, len(x))
            probe.observe_interp(lines, idx - _pair_lines(len(x)), len(interp))
            table = PhotonTable(c1=_float(tab_header, "C1"), c2=_float(tab_header, "C2"),
                                l1=_int(tab_header, "C3"), l2=_int(tab_header, "C4"),
                                interp=list(interp), x=list(x), y=list(y))
            covariance = None
            if table.l2 != 0:
                header, values, idx = _list(lines, idx, probe)
                covariance = (_fields(header), values)
            spectrum.continuum = ContinuousSpectrum(table=table, covariance=covariance)
        section.spectra.append(spectrum)
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 8, mt)
    return section


def parse_mf8_yields(lines: List[str], mt: int) -> MF8FissionYields:
    head = parse_line(lines[0])
    section = MF8FissionYields(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                               _l2=_int(head, "C4"), _n1=_int(head, "C5"),
                               _n2=_int(head, "C6"), _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    for _ in range(_int(head, "C3")):
        header, values, idx = _list(lines, idx, probe)
        section.energies.append(FissionYields(energy=_float(header, "C1"),
                                              interpolation=_int(header, "C3"),
                                              values=values, c2=_float(header, "C2"),
                                              l2=_int(header, "C4")))
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 8, mt)
    return section


def parse_mf8(lines: List[str]) -> MF:
    """Every MT of an MF8 block: 457, 454 and 459 parsed, the rest kept (E6)."""
    mf = MF(number=8)
    mf.num_lines = len(lines)
    groups, counts = group_lines_by_mt_with_positions(lines)
    for mt, mt_lines in groups.items():
        if mt == 0:
            continue
        body = [line for line in mt_lines if line[72:75].strip() not in ("", "0")]
        try:
            if mt == 457:
                section = parse_mf8_mt457(body, mt)
            elif mt in (454, 459):
                section = parse_mf8_yields(body, mt)
            else:
                section = MF8Raw(number=mt, raw_lines=[l.rstrip(LINE_END) for l in body],
                                 _mat=parse_line(body[0]).get("MAT") if body else None)
            section.num_lines = counts.get(mt, len(body))
            mf.add_section(section)
        except Exception as exc:                                 # noqa: BLE001
            mf.parse_errors[mt] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Error parsing MT{mt} in MF8: {mf.parse_errors[mt]}")
    return mf

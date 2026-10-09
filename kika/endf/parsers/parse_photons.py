"""Parsers for the photon production files MF12, MF13, MF14 and MF15.

One module for four files because they are one subject: MF12 or MF13 lists an
MT's photons, MF14 gives their angular distributions and MF15 the spectrum of
the continuum among them. The record walk follows
:mod:`~kika.endf.parsers.parse_mf6` in discipline — :func:`_check_consumed`
requires every record of a section to have been read, and a layout ENDF-6 does
not define raises, naming the MT, rather than being stepped over by a guessed
length.

**The walk was measured before it was written.** An independent column parser
walked every MF12-15 section of ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 (1 946
tapes, about 61 000 sections) on 2026-10-09 with no format error and no line
left over; ``docs/library/endf_photons_e5_plan.md`` §2 in the workspace repo,
and ``kika/endf/tests/photon_census.py`` here, which keeps it re-runnable.
"""
from typing import Callable, List, Tuple

from ..classes.mf import MF
from ..classes.mf12.base import MF12MT, MF13MT, PhotonTable
from ..classes.mf14.base import AngularNode, AnisotropicPhoton, IsotropicPhoton, MF14MT
from ..classes.mf15.base import MF15MT, PhotonSpectrum
from ..utils import (
    PaddingProbe,
    group_lines_by_mt_with_positions,
    parse_data_values,
    parse_line,
    parse_tab1,
    parse_tab2,
)
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def _int(header: dict, key: str) -> int:
    return int(header.get(key) or 0)


def _float(header: dict, key: str) -> float:
    value = header.get(key)
    return 0.0 if value is None else value


def _pair_lines(n_pairs: int) -> int:
    return -(-n_pairs // 3)


def _table(lines: List[str], idx: int, probe: PaddingProbe) -> Tuple[PhotonTable, int]:
    """A TAB1 with its header fields, observing how its writer padded it."""
    header, interp, x, y, idx = parse_tab1(lines, idx)
    probe.observe_pairs(lines, idx, len(x))
    probe.observe_interp(lines, idx - _pair_lines(len(x)), len(interp))
    return PhotonTable(
        c1=_float(header, "C1"), c2=_float(header, "C2"),
        l1=_int(header, "C3"), l2=_int(header, "C4"),
        interp=list(interp), x=list(x), y=list(y),
    ), idx


def _list(lines: List[str], idx: int, probe: PaddingProbe) -> Tuple[dict, List[float], int]:
    header = parse_line(lines[idx])
    values, idx = parse_data_values(lines, idx + 1, _int(header, "C5"))
    probe.observe_values(lines, idx, len(values))
    return header, values, idx


def _tab2(lines: List[str], idx: int, probe: PaddingProbe):
    header, interp, idx = parse_tab2(lines, idx)
    probe.observe_interp(lines, idx, len(interp))
    return header, list(interp), idx


def _check_consumed(idx: int, lines: List[str], mf: int, mt: int) -> None:
    if idx != len(lines):
        raise ValueError(
            f"MF{mf}/MT{mt}: parsed {idx} of {len(lines)} records; "
            f"{len(lines) - idx} left over, so the record walk is wrong"
        )


# ----------------------------------------------------------------------
# MF12 and MF13
# ----------------------------------------------------------------------

def parse_mf12_mt(lines: List[str], mt: int) -> MF12MT:
    """One MF12 section: LO=1 multiplicities or an LO=2 level scheme."""
    head = parse_line(lines[0])
    lo = _int(head, "C3")
    if lo not in (1, 2):
        raise ValueError(f"MF12/MT{mt}: LO={lo}; ENDF-6 defines only LO=1 and LO=2")
    section = MF12MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"), _lo=lo,
                     _l2=_int(head, "C4"), _n2=_int(head, "C6"), _mat=head.get("MAT"))
    probe = PaddingProbe()
    if lo == 2:
        section._ns = _int(head, "C5")
        header, values, idx = _list(lines, 1, probe)
        section.es_ns = _float(header, "C1")
        section.list_c2 = _float(header, "C2")
        section.lp = _int(header, "C3")
        section.list_l2 = _int(header, "C4")
        section.nt = _int(header, "C6")
        section.transition_values = list(values)
    else:
        idx = _read_photon_tables(section, lines, _int(head, "C5"), probe)
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 12, mt)
    return section


def parse_mf13_mt(lines: List[str], mt: int) -> MF13MT:
    """One MF13 section: MF12 LO=1's layout, cross sections instead of yields."""
    head = parse_line(lines[0])
    section = MF13MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                     _l1=_int(head, "C3"), _l2=_int(head, "C4"), _n2=_int(head, "C6"),
                     _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = _read_photon_tables(section, lines, _int(head, "C5"), probe)
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 13, mt)
    return section


def _read_photon_tables(section: MF12MT, lines: List[str], nk: int,
                        probe: PaddingProbe) -> int:
    """The optional total (NK>1), then NK photon tables."""
    idx = 1
    if nk > 1:
        section.total, idx = _table(lines, idx, probe)
    for _ in range(nk):
        table, idx = _table(lines, idx, probe)
        section.photons.append(table)
    return idx


# ----------------------------------------------------------------------
# MF14
# ----------------------------------------------------------------------

def parse_mf14_mt(lines: List[str], mt: int) -> MF14MT:
    """One MF14 section: HEAD only for LI=1, else NI CONTs and NK-NI TAB2s."""
    head = parse_line(lines[0])
    li, ltt = _int(head, "C3"), _int(head, "C4")
    nk, ni = _int(head, "C5"), _int(head, "C6")
    section = MF14MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                     _li=li, _ltt=ltt, _nk=nk, _ni=ni, _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    if li != 1:
        if ltt not in (1, 2) and nk > ni:
            raise ValueError(
                f"MF14/MT{mt}: LTT={ltt} with {nk - ni} anisotropic photon(s); "
                f"ENDF-6 defines only LTT=1 (Legendre) and LTT=2 (tabulated)"
            )
        for _ in range(ni):
            record = parse_line(lines[idx])
            section.isotropic.append(IsotropicPhoton(
                eg=_float(record, "C1"), es=_float(record, "C2"),
                rest=(_int(record, "C3"), _int(record, "C4"),
                      _int(record, "C5"), _int(record, "C6")),
            ))
            idx += 1
        for _ in range(nk - ni):
            header, interp, idx = _tab2(lines, idx, probe)
            photon = AnisotropicPhoton(
                eg=_float(header, "C1"), es=_float(header, "C2"),
                l1=_int(header, "C3"), l2=_int(header, "C4"), tab2_interp=interp,
            )
            for _ in range(_int(header, "C6")):
                if ltt == 2:
                    table, idx = _table(lines, idx, probe)
                    photon.nodes.append(AngularNode(
                        energy=table.c2, c1=table.c1, l1=table.l1, l2=table.l2,
                        interp=table.interp, mu=table.x, p=table.y))
                else:
                    record, values, idx = _list(lines, idx, probe)
                    photon.nodes.append(AngularNode(
                        energy=_float(record, "C2"), c1=_float(record, "C1"),
                        l1=_int(record, "C3"), l2=_int(record, "C4"),
                        n2=_int(record, "C6"), coefficients=list(values)))
            section.anisotropic.append(photon)
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 14, mt)
    return section


# ----------------------------------------------------------------------
# MF15
# ----------------------------------------------------------------------

def parse_mf15_mt(lines: List[str], mt: int) -> MF15MT:
    """One MF15 section: NC partial distributions, each LF=1."""
    head = parse_line(lines[0])
    section = MF15MT(number=mt, _za=head.get("C1"), _awr=head.get("C2"),
                     _l1=_int(head, "C3"), _l2=_int(head, "C4"), _n2=_int(head, "C6"),
                     _mat=head.get("MAT"))
    probe = PaddingProbe()
    idx = 1
    for j in range(_int(head, "C5")):
        weight, idx = _table(lines, idx, probe)
        if weight.l2 != 1:
            raise ValueError(
                f"MF15/MT{mt} partial {j}: LF={weight.l2}; ENDF-6 defines only "
                f"LF=1 (tabulated) for MF15, so the length of this body is unknown"
            )
        header, interp, idx = _tab2(lines, idx, probe)
        spectrum = PhotonSpectrum(
            weight=weight,
            tab2_fields=(_float(header, "C1"), _float(header, "C2"),
                         _int(header, "C3"), _int(header, "C4")),
            tab2_interp=interp,
        )
        for _ in range(_int(header, "C6")):
            table, idx = _table(lines, idx, probe)
            spectrum.distributions.append(table)
        section.spectra.append(spectrum)
    section.pad = probe.resolve()
    _check_consumed(idx, lines, 15, mt)
    return section


# ----------------------------------------------------------------------
# File level
# ----------------------------------------------------------------------

def _file_parser(number: int, section_parser: Callable) -> Callable[[List[str]], MF]:
    def parse(lines: List[str]) -> MF:
        mf = MF(number=number)
        mf.num_lines = len(lines)
        mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
        for mt, mt_lines in mt_groups.items():
            if mt == 0:
                continue
            try:
                section = section_parser(mt_lines, mt)
                if mt in line_counts:
                    section.num_lines = line_counts[mt]
                mf.add_section(section)
            except Exception as exc:                       # noqa: BLE001
                # One bad section costs its own MT, as in MF5 and MF6, and the
                # loss is recorded where a reader of the file can find it.
                mf.parse_errors[mt] = str(exc)
                logger.warning(f"Error parsing MT{mt} in MF{number}: {exc}")
        return mf

    parse.__name__ = f"parse_mf{number}"
    parse.__doc__ = f"Parse MF{number} into an :class:`MF` of its sections."
    return parse


parse_mf12 = _file_parser(12, parse_mf12_mt)
parse_mf13 = _file_parser(13, parse_mf13_mt)
parse_mf14 = _file_parser(14, parse_mf14_mt)
parse_mf15 = _file_parser(15, parse_mf15_mt)

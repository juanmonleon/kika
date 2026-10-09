"""MF12, MF13, MF14 and MF15: the read-write gate of the photon files (roadmap E5a).

**The gate.** Every photon section must parse and re-emit as the bytes it came
in, with one stated exception: a field the source writes in a non-canonical
form (``6012.00000``, ``1.00000E-5``, ``4.3188799+6``) comes back canonical.
Re-emitting it verbatim would mean storing the text of every number on the
tape; the value is kept bit for bit, which is what is asserted. It is the
divergence ``test_mf6_roundtrip.py`` pins on C-12's MF6 HEAD, met at scale:
ENDF/B-VIII.1 writes most of its photon HEADs in plain decimal.

**Where the gate runs.** On six committed cuts, which between them carry every
form the census found worth a fixture (``PHOTON_FIXTURES`` in
``test_micro_tape_regen.py``), and — ``tape``- and ``slow``-marked — on every
MF12-15 section of ENDF/B-VIII.1, JEFF-4.0 and JENDL-5: 61 361 sections, about
three minutes. The census that sweep re-asserts was first taken by a column
parser that shares no code with kika (kika-workspace
``docs/library/endf_photons_e5_plan.md`` §2); the two agree section for section.

**What has no witness, and is gated by kika's own emitter only:** MF14 LTT=2
(tabulated p(μ)) and MF15 NC>1. Both are cheap — the first is a TAB1 where
LTT=1 has a LIST, the second the same subsection repeated — and neither occurs
in the three libraries. The tests that cover them say so in their names.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from kika._records import format_endf_number, parse_number
from kika.endf import read_endf
from kika.endf.classes.mf12.base import MF12MT, MF13MT, PhotonTable
from kika.endf.classes.mf14.base import AngularNode, AnisotropicPhoton, MF14MT
from kika.endf.classes.mf15.base import MF15MT, PhotonSpectrum
from kika.endf.parsers.parse_endf import MF_PARSERS
from kika.endf.parsers.parse_photons import (
    parse_mf12_mt,
    parse_mf13_mt,
    parse_mf14_mt,
    parse_mf15_mt,
)
from kika.endf.utils import parse_endf_id

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = {key: DATA / f"micro_{key}_photons.endf"
            for key in ("n14", "fe56", "u235", "s36", "hf182", "cm243")}
SECTION_PARSERS = {12: parse_mf12_mt, 13: parse_mf13_mt,
                   14: parse_mf14_mt, 15: parse_mf15_mt}
PHOTON_MF = tuple(SECTION_PARSERS)


def section_lines(text: str):
    """``{(MF, MT): [lines]}`` for the photon files of an ENDF text, SEND excluded."""
    out = defaultdict(list)
    for line in text.splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf in SECTION_PARSERS and mt:
            out[(mf, mt)].append(line)
    return out


def textual_divergence(source: list, kika: list) -> list:
    """Every field where *kika* differs from *source*, if each is allowed to.

    Returns the offending ``(line, column, source field, kika field)`` list —
    empty when the section is byte-identical, or differs only where the source
    writes a number non-canonically and kika writes the same value back.
    """
    if len(source) != len(kika):
        return [("line count", len(source), len(kika), None)]
    bad = []
    for i, (want, have) in enumerate(zip(source, kika)):
        want, have = want[:75], have[:75]
        if want == have:
            continue
        if want[66:] != have[66:]:
            bad.append((i, "MAT/MF/MT", want[66:], have[66:]))
            continue
        for col in range(0, 66, 11):
            a, b = want[col:col + 11], have[col:col + 11]
            if a == b:
                continue
            value = parse_number(a)
            canonical = a.strip() and format_endf_number(value).strip() == a.strip()
            if value != parse_number(b) or canonical:
                bad.append((i, col, a, b))
    return bad


def roundtrip(lines: list, mf: int, mt: int) -> list:
    section = SECTION_PARSERS[mf](lines, mt)
    return textual_divergence(lines, str(section).splitlines()[:-1])


# ---------------------------------------------------------------------------
# Registration and the committed cuts
# ---------------------------------------------------------------------------

def test_the_photon_files_are_registered():
    assert {12, 13, 14, 15} <= set(MF_PARSERS)


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_every_fixture_section_roundtrips(key):
    sections = section_lines(FIXTURES[key].read_text())
    assert sections, key
    for (mf, mt), lines in sections.items():
        assert not roundtrip(lines, mf, mt), f"{key} MF{mf}/MT{mt}"


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_read_endf_keeps_every_photon_section(key):
    """``read_endf`` no longer drops MF12-15: each section is in ``endf.mf``."""
    on_tape = set(section_lines(FIXTURES[key].read_text()))
    endf = read_endf(str(FIXTURES[key]))
    read = {(mf, mt) for mf in PHOTON_MF if mf in endf.mf for mt in endf.mf[mf].mt}
    assert read == on_tape
    assert not any(endf.mf[mf].parse_errors for mf in PHOTON_MF if mf in endf.mf)


def test_n14_writes_its_plain_decimal_head_back_canonical():
    """The one divergence, on the cut that shows it: same value, canonical text."""
    lines = section_lines(FIXTURES["n14"].read_text())[(12, 102)]
    section = parse_mf12_mt(lines, 102)
    head = str(section).splitlines()[0]
    assert section._za == pytest.approx(7014.0)
    assert parse_number(lines[0][:11]) == parse_number(head[:11])


# ---------------------------------------------------------------------------
# What each form carries
# ---------------------------------------------------------------------------

def all_sections(key):
    endf = read_endf(str(FIXTURES[key]))
    return {mf: endf.mf[mf].mt for mf in PHOTON_MF if mf in endf.mf}


def test_lo1_carries_the_total_when_nk_exceeds_one():
    sections = all_sections("n14")
    capture = sections[12][102]
    assert capture.lo == 1 and len(capture.photons) == 59
    assert capture.total is not None
    single = sections[13][32]
    assert len(single.photons) == 1 and single.total is None


def test_lo1_subsections_say_discrete_primary_or_continuum():
    """EG=0 goes with LF=1 and EG>0 with LF=2, with no exception in the census."""
    capture = all_sections("n14")[12][102]
    assert {p.lp for p in capture.photons} == {0, 2}
    for photon in capture.photons:
        assert (photon.eg == 0) == (photon.lf == 1)


def test_lo2_transitions_are_lg_plus_one_wide():
    """LG=2 states (ES, TP, GP); LG=1 states (ES, TP)."""
    for key, lg in (("fe56", 2), ("s36", 2), ("cm243", 1)):
        for mt, section in all_sections(key)[12].items():
            assert section.lo == 2 and section.lg == lg, (key, mt)
            assert len(section.transition_values) == (lg + 1) * section.nt
            assert all(len(t) == lg + 1 for t in section.transitions)
            assert sum(t[1] for t in section.transitions) == pytest.approx(1.0)


def test_lp_of_a_level_scheme_is_kept():
    assert {s.lp for s in all_sections("s36")[12].values()} == {1}
    assert {s.lp for s in all_sections("fe56")[12].values()} == {0}


def test_mf13_head_has_no_lo_and_keeps_its_field():
    for mt, section in all_sections("n14")[13].items():
        assert isinstance(section, MF13MT)
        assert section._l1 == 0, mt


def test_mf14_isotropic_and_anisotropic_split_on_ni():
    mf14 = all_sections("n14")[14]
    inelastic = mf14[4]
    assert inelastic.li == 0 and inelastic.ltt == 1
    assert len(inelastic.isotropic) == 41 and len(inelastic.anisotropic) == 2
    for photon in inelastic.anisotropic:
        assert photon.nodes and all(n.coefficients for n in photon.nodes)
    assert mf14[103].all_isotropic and mf14[103].num_photons == 11


def test_mf15_is_one_lf1_partial_with_a_distribution_per_energy():
    spectrum = all_sections("hf182")[15][3].spectra
    assert len(spectrum) == 1 and spectrum[0].lf == 1
    assert len(spectrum[0].distributions) == len(spectrum[0].incident_energies) > 1


def test_mf13_on_an_mt_with_no_mf3_is_read():
    """N-14's MT28 and MT32 state photons and no cross section."""
    endf = read_endf(str(FIXTURES["n14"]))
    for mt in (28, 32):
        assert mt in endf.mf[13].mt and mt not in endf.mf[3].mt


def test_the_mt18_deferral_points_at_a_file_that_was_read():
    """U-235 MT18: MF6 LAW=-15 subsections beside MF12/14/15 of the same MT."""
    endf = read_endf(str(FIXTURES["u235"]))
    assert any(p.law == -15 for p in endf.mf[6].mt[18].products)
    assert endf.mf[6].mt[18].report_gaps() == []
    assert 18 in endf.mf[15].mt


# ---------------------------------------------------------------------------
# What ENDF-6 does not define is refused, naming it
# ---------------------------------------------------------------------------

def _head(c1, c2, l1, l2, n1, n2, mf, mt=4):
    from kika.endf.classes.mf12.base import emit_cont
    return emit_cont(c1, c2, l1, l2, n1, n2, 125, mf, mt, 1)[0]


def test_mf12_lo_other_than_one_or_two_raises():
    with pytest.raises(ValueError, match="LO=3"):
        parse_mf12_mt([_head(7014.0, 13.9, 3, 0, 1, 0, 12)], 4)


def test_mf14_ltt_other_than_one_or_two_raises():
    lines = [_head(7014.0, 13.9, 0, 3, 1, 0, 14)]
    with pytest.raises(ValueError, match="LTT=3"):
        parse_mf14_mt(lines, 4)


def test_mf15_lf_other_than_one_raises():
    weight = PhotonTable(l2=2, interp=[(2, 2)], x=[1e-5, 2e7], y=[1.0, 1.0])
    lines = [_head(7014.0, 13.9, 0, 0, 1, 0, 15)]
    lines += weight.emit(125, 15, 4, 2, MF15MT(number=4).pad)[0]
    with pytest.raises(ValueError, match="LF=2"):
        parse_mf15_mt(lines, 4)


def test_a_bad_section_costs_only_its_own_mt(tmp_path):
    good = section_lines(FIXTURES["fe56"].read_text())[(12, 52)]
    bad = [_head(26056.0, 55.4, 3, 0, 1, 0, 12, mt=53)]
    mf = MF_PARSERS[12](good + bad)
    assert 52 in mf.mt and 53 not in mf.mt
    assert "LO=3" in mf.parse_errors[53]


# ---------------------------------------------------------------------------
# No witness: kika's own emitter
# ---------------------------------------------------------------------------

def test_mf14_ltt2_roundtrips_through_our_own_emitter():
    """LTT=2 occurs in none of the three libraries."""
    section = MF14MT(number=4, _za=7014.0, _awr=13.88, _li=0, _ltt=2, _mat=725)
    section.anisotropic.append(AnisotropicPhoton(
        eg=1.6e6, es=2.3e6, tab2_interp=[(2, 2)],
        nodes=[AngularNode(energy=e, interp=[(3, 2)], mu=[-1.0, 0.0, 1.0],
                           p=[0.4, 0.5, 0.6]) for e in (2.5e6, 2.0e7)]))
    text = str(section).splitlines()[:-1]
    back = parse_mf14_mt(text, 4)
    assert str(back) == str(section)
    assert back.anisotropic[0].nodes[1].p == [0.4, 0.5, 0.6]


def test_mf15_nc_above_one_roundtrips_through_our_own_emitter():
    """NC>1 occurs in none of the three libraries (850 of 850 are NC=1)."""
    def partial(weight):
        return PhotonSpectrum(
            weight=PhotonTable(l2=1, interp=[(2, 2)], x=[1e-5, 2e7], y=[weight, weight]),
            tab2_interp=[(2, 2)],
            distributions=[PhotonTable(c2=e, interp=[(2, 2)], x=[0.0, 1e6, 2e6],
                                       y=[0.0, 1e-6, 0.0]) for e in (1e-5, 2e7)])
    section = MF15MT(number=102, _za=7014.0, _awr=13.88, _mat=725,
                     spectra=[partial(0.25), partial(0.75)])
    back = parse_mf15_mt(str(section).splitlines()[:-1], 102)
    assert str(back) == str(section)
    assert [s.weight.y[0] for s in back.spectra] == [0.25, 0.75]


# ---------------------------------------------------------------------------
# The whole of three libraries
# ---------------------------------------------------------------------------

#: Sections per file, re-measured by the sweep below; the independent census of
#: 2026-10-09 found the same.
CENSUS_SECTIONS = {
    "endfb81": {12: 14574, 13: 52, 14: 14626, 15: 526},
    "jeff40": {12: 15171, 13: 38, 14: 15209, 15: 76},
    "jendl5": {12: 735, 13: 9, 14: 744, 15: 248},
}
CENSUS_LO = {"endfb81": {1: 509, 2: 14065}, "jeff40": {1: 75, 2: 15096},
             "jendl5": {1: 315, 2: 420}}
CENSUS_MF14_LI0 = {"endfb81": 11, "jeff40": 11, "jendl5": 1}


@pytest.mark.slow
def test_every_photon_section_of_three_libraries_roundtrips(neutron_libraries):
    sections, lo, li0, mf15_shapes, mf13_mts = (Counter(), Counter(), Counter(),
                                                Counter(), Counter())
    failures = []
    for lib, directory in neutron_libraries.items():
        files = sorted(p for p in Path(directory).iterdir()
                       if p.suffix in (".endf", ".jeff", ".dat"))
        for path in files:
            text = path.read_text(encoding="latin-1")
            for (mf, mt), lines in section_lines(text).items():
                sections[(lib, mf)] += 1
                section = SECTION_PARSERS[mf](lines, mt)
                bad = textual_divergence(lines, str(section).splitlines()[:-1])
                if bad and len(failures) < 10:
                    failures.append((lib, path.name, mf, mt, bad[:2]))
                if mf == 12:
                    lo[(lib, section.lo)] += 1
                elif mf == 13:
                    mf13_mts[mt] += 1
                elif mf == 14 and section.li == 0:
                    li0[lib] += 1
                elif mf == 15:
                    mf15_shapes[(len(section.spectra),
                                 tuple(sorted({s.lf for s in section.spectra})))] += 1
    assert not failures, failures
    for lib, by_mf in CENSUS_SECTIONS.items():
        for mf, n in by_mf.items():
            assert sections[(lib, mf)] == n, (lib, mf)
        for value, n in CENSUS_LO[lib].items():
            assert lo[(lib, value)] == n, (lib, "LO", value)
        assert li0[lib] == CENSUS_MF14_LI0[lib], (lib, "LI=0")
    # MF15 is NC=1, LF=1 everywhere; MF13 never on fission, capture or a level.
    assert dict(mf15_shapes) == {(1, (1,)): 850}
    assert not {18, 102} & set(mf13_mts)
    assert not any(51 <= mt <= 91 for mt in mf13_mts)

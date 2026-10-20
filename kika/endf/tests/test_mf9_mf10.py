"""MF9 and MF10: the read-write gate of the radioactive-production files.

**The gate** is the photon files' (``test_photon_files.py``): every section must
parse and re-emit as the bytes it came in, except that a number the source
writes non-canonically comes back canonical with the same value.

**Where it runs.** On the two committed activation cuts
(``ACTIVATION_FIXTURES`` in ``test_micro_tape_regen.py``), and -- ``slow``,
needing the shared libraries -- on every MF9/MF10 section of ENDF/B-VIII.1,
JEFF-4.0 and JENDL-5. At home the three libraries sit under
``NuclearData/endf``: point ``KIKA_LIB_TAPES`` there. The full sweep (2026-10-14)
caught one section, In-115 MF9/MT102 of B-VIII.1, whose 8-digit fixed-point QM
the TAB1 header rounded to 7.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.classes.mf9.base import MF9MT, MF10MT, ProductionState
from kika.endf.parsers.parse_endf import MF_PARSERS
from kika.endf.parsers.parse_mf9_10 import parse_mf9_mt, parse_mf10_mt
from kika.endf.tests.test_photon_files import textual_divergence
from kika.endf.utils import parse_endf_id

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = {key: DATA / f"micro_{key}_activation.endf" for key in ("co59", "am241")}
SECTION_PARSERS = {9: parse_mf9_mt, 10: parse_mf10_mt}


def section_lines(text: str):
    """``{(MF, MT): [lines]}`` for MF9 and MF10 of an ENDF text, SEND excluded."""
    out = defaultdict(list)
    for line in text.splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf in SECTION_PARSERS and mt:
            out[(mf, mt)].append(line)
    return out


def roundtrip(lines: list, mf: int, mt: int) -> list:
    section = SECTION_PARSERS[mf](lines, mt)
    return textual_divergence(lines, str(section).splitlines()[:-1])


def test_mf9_and_mf10_are_registered():
    assert {9, 10} <= set(MF_PARSERS)


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_every_fixture_section_roundtrips(key):
    sections = section_lines(FIXTURES[key].read_text())
    assert sections, key
    for (mf, mt), lines in sections.items():
        assert not roundtrip(lines, mf, mt), f"{key} MF{mf}/MT{mt}"


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_read_endf_keeps_every_section(key):
    on_tape = set(section_lines(FIXTURES[key].read_text()))
    endf = read_endf(str(FIXTURES[key]))
    read = {(mf, mt) for mf in (9, 10) if mf in endf.mf for mt in endf.mf[mf].mt}
    assert read == on_tape
    assert not any(endf.mf[mf].parse_errors for mf in (9, 10) if mf in endf.mf)


def test_a_state_names_its_product_and_its_level():
    """Am-241 capture: the ground state of Am-242 and the isomer, level 2 at 48.63 keV."""
    section = read_endf(str(FIXTURES["am241"])).mf[9].mt[102]
    assert isinstance(section, MF9MT) and section._lis == 0
    assert [(s.izap, s.lfs) for s in section.states] == [(95242, 0), (95242, 2)]
    assert section.states[0].level_energy == 0.0
    assert section.states[1].level_energy == pytest.approx(48630.0)
    # The two branches of the capture sum to one at every tabulated energy.
    ground, isomer = section.states
    assert ground.x == isomer.x
    assert all(a + b == pytest.approx(1.0) for a, b in zip(ground.y, isomer.y))


def test_mf10_holds_cross_sections_per_state():
    endf = read_endf(str(FIXTURES["co59"]))
    section = endf.mf[10].mt[16]
    assert isinstance(section, MF10MT)
    assert [(s.izap, s.lfs) for s in section.states] == [(27058, 0), (27058, 1)]
    assert section.get_state(27058, 1).level_energy == pytest.approx(24950.0)


def test_records_left_over_raise_naming_the_mt():
    lines = section_lines(FIXTURES["am241"].read_text())[(9, 102)]
    with pytest.raises(ValueError, match="MF9/MT102"):
        parse_mf9_mt(lines + [lines[-1]], 102)


def test_the_counts_are_derived_from_the_body():
    """NS, NR and NP come from the data, so an edit cannot write a stale count."""
    section = MF10MT(number=16, _za=27059.0, _awr=58.43, _mat=2725, states=[
        ProductionState(qm=-1.0e7, qi=-1.0e7, izap=27058, lfs=0,
                        interp=[(3, 2)], x=[1.1e7, 1.5e7, 2.0e7], y=[0.0, 0.5, 0.7])])
    section.states.append(ProductionState(qm=-1.0e7, qi=-1.0249e7, izap=27058, lfs=1,
                                          interp=[(2, 2)], x=[1.1e7, 2.0e7], y=[0.0, 0.2]))
    back = parse_mf10_mt(str(section).splitlines()[:-1], 16)
    assert str(back) == str(section)
    assert back.num_states == 2 and len(back.states[1].x) == 2


def test_an_eight_digit_q_value_keeps_its_digits():
    """ENDF/B-VIII.1 In-115, MF9/MT102, writes QM as ``  6783389.7``. The
    canonical field is ``6.783390+6``, a different number; the state's header
    goes through the precise formatter, as its data does."""
    section = MF9MT(number=102, _za=49115.0, _awr=113.81, _mat=4931, states=[
        ProductionState(qm=6783389.7, qi=6656120.0, izap=49116, lfs=1,
                        interp=[(2, 2)], x=[1.0e-5, 2.0e7], y=[0.5, 0.5])])
    text = str(section)
    assert "  6783389.7" in text
    back = parse_mf9_mt(text.splitlines()[:-1], 102)
    assert back.states[0].qm == 6783389.7


# ---------------------------------------------------------------------------
# The whole of three libraries
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_every_mf9_mf10_section_of_three_libraries_roundtrips(neutron_libraries):
    sections, failures = 0, []
    for lib, directory in neutron_libraries.items():
        for path in sorted(p for p in Path(directory).iterdir()
                           if p.suffix in (".endf", ".jeff", ".dat")):
            for (mf, mt), lines in section_lines(path.read_text(encoding="latin-1")).items():
                sections += 1
                bad = roundtrip(lines, mf, mt)
                if bad and len(failures) < 10:
                    failures.append((lib, path.name, mf, mt, bad[:2]))
    assert sections, "the sweep found no MF9/MF10 section at all"
    assert not failures, failures

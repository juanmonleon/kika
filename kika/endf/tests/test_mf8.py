"""MF8: decay data (MT457) and fission product yields (MT454/459), read and written back.

The fixtures are whole ENDF/B-VIII.1 sublibrary files: Co-60 (one beta- mode,
gamma/beta/e-/x-ray spectra), Fe-56 (stable, NST=1), Cs-137, Rb-93 (beta- and
beta-delayed n, with an LCON=1 neutron continuum) and the Cm-242 fission yields.

Library gate (``-m tape``): every file of ``KIKA_DECAY_TAPES`` and
``KIKA_NFY_TAPES`` -- 3 821 decay and 31 yield evaluations of B-VIII.1 on
2026-10-09, 3 883 sections, all written back byte for byte.
"""
import os
from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.classes.mf8 import MF8FissionYields, MF8MT457
from kika.endf.tests.test_photon_files import textual_divergence

DATA = Path(__file__).parent / "data"
FIXTURES = ["dec-027_Co_060", "dec-026_Fe_056", "dec-055_Cs_137", "dec-037_Rb_093",
            "nfy-096_Cm_242"]


def _source(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if len(line) >= 75 and line[70:72] == " 8" and line[72:75].strip() not in ("", "0"):
            out.setdefault(int(line[72:75]), []).append(line)
    return out


def _assert_round_trip(path: Path) -> int:
    endf = read_endf(str(path), mf_numbers=[8])
    mf8 = endf.mf[8]
    assert not mf8.parse_errors, mf8.parse_errors
    source = _source(path)
    assert set(source) == set(mf8.mt)
    for mt, lines in source.items():
        written = str(mf8.mt[mt]).splitlines()
        assert written[-1][70:75] == " 8  0"
        assert textual_divergence(lines, written[:-1]) == [], (path.name, mt)
    return len(source)


@pytest.mark.parametrize("name", FIXTURES)
def test_mf8_is_written_back_byte_for_byte(name):
    assert _assert_round_trip(DATA / f"{name}.endf") >= 1


def test_mt457_names_what_it_holds():
    co60 = read_endf(str(DATA / "dec-027_Co_060.endf"), mf_numbers=[8]).mf[8].mt[457]
    assert isinstance(co60, MF8MT457)
    assert co60.zaid == 27060 and not co60.is_stable
    assert co60.halflife == pytest.approx(1.663442e8)
    (mode,) = co60.decay_modes
    assert mode.rtyp == 1 and mode.br == 1
    assert [s.radiation for s in co60.spectra] == ["gamma", "beta-", "e-", "x-ray"]
    assert len(co60.average_energy_pairs) == 3

    fe56 = read_endf(str(DATA / "dec-026_Fe_056.endf"), mf_numbers=[8]).mf[8].mt[457]
    assert fe56.is_stable and fe56.decay_modes == [] and fe56.spectra == []

    rb93 = read_endf(str(DATA / "dec-037_Rb_093.endf"), mf_numbers=[8]).mf[8].mt[457]
    assert [m.rtyp for m in rb93.decay_modes] == [1, 1.5]
    neutrons = next(s for s in rb93.spectra if s.radiation == "n")
    assert neutrons.lcon == 1 and neutrons.lines == [] and neutrons.continuum is not None


def test_fission_yields_name_what_they_hold():
    mf8 = read_endf(str(DATA / "nfy-096_Cm_242.endf"), mf_numbers=[8]).mf[8]
    independent, cumulative = mf8.mt[454], mf8.mt[459]
    assert isinstance(independent, MF8FissionYields)
    assert (independent.kind, cumulative.kind) == ("independent", "cumulative")
    (block,) = independent.energies
    entries = block.entries
    assert len(entries) == len(block.values) // 4 > 100
    assert sum(e.y for e in entries) == pytest.approx(2.0, rel=0.02)


def _library(env: str):
    root = os.environ.get(env)
    if not root:
        pytest.skip(f"{env} not set")
    paths = sorted(Path(root).glob("*.endf"))
    if not paths:
        pytest.fail(f"{env}={root!r} holds no *.endf file")
    return paths


@pytest.mark.tape
@pytest.mark.slow
@pytest.mark.parametrize("env", ["KIKA_DECAY_TAPES", "KIKA_NFY_TAPES"])
def test_every_mf8_of_a_sublibrary_is_written_back(env):
    paths = _library(env)
    assert sum(_assert_round_trip(p) for p in paths) >= len(paths)

"""A GNDS ``resonances`` holding the scattering radius alone is written as MF2.

NNDC's H-2 (ENDF/B-VIII.1) states AP and no region. The tape kika writes from
the GNDS carries the LRU=0 range ENDF/B-VIII.1's own tape has, field for field:
ZAI 1002, ABN 1, EL 1e-5 eV and EH 1e8 eV (the radius' domain, not the
projectile's 1.5e8), SPI 1, AP 0.51977 (10^-12 cm) -- with LRP=0.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kika
from kika.endf import read_endf

DATA = Path(__file__).resolve().parents[3] / "gnds" / "tests" / "data"


def test_the_radius_alone_is_written_as_the_tape_states_it(tmp_path):
    out = tmp_path / "h2.endf"
    kika.write(kika.read(str(DATA / "n-001_H_002.endf.gnds.xml")), str(out), format="endf")
    tape = read_endf(str(out))
    assert tape.mf[1].mt[451]._lrp == 0
    (isotope,) = tape.mf[2].mt[151].isotopes
    assert (isotope.za, isotope.abn, isotope.lfw) == (1002, 1.0, 0)
    (energy_range,) = isotope.energy_ranges
    assert (energy_range.lru, energy_range.lrf, energy_range.nro, energy_range.naps) == (0, 0, 0, 0)
    assert (energy_range.el, energy_range.eh) == (1e-5, 1e8)
    assert energy_range.parameters.spi == 1.0
    assert energy_range.parameters.ap == pytest.approx(0.51977, abs=1e-9)


def test_a_target_without_a_spin_is_not_given_one(tmp_path):
    suite = kika.read(str(DATA / "n-001_H_002.endf.gnds.xml"))
    suite.PoPs.particles.get(suite.target).spin = None
    out = tmp_path / "h2.endf"
    report = kika.write(suite, str(out), format="endf")
    assert 2 not in read_endf(str(out)).mf
    assert any("SPI is not assumed" in message for message in report.losses)

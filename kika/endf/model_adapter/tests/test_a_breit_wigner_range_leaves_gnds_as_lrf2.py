"""A Breit-Wigner range read from GNDS is written as the LRF=1/2 range the tape had.

GNDS already holds it one spin group per L, as ENDF does; what it lacks is the
bookkeeping (LRF, NAPS, SPI, AP, AWRI, QX, LRX). Ne-20 (MLBW) comes back with
every record of the source and its 0 K resonance cross sections bit for bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import kika
from kika.endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TAPE = DATA / "micro_ne20_covcheck.endf"


def _viaGnds(tmp_path):
    gnds = tmp_path / "ne20.gnds.xml"
    kika.write(kika.read(str(TAPE)), str(gnds), format="gnds")
    out = tmp_path / "ne20.endf"
    kika.write(kika.read(str(gnds)), str(out), format="endf")
    return out


def _records(path):
    (isotope,) = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151].isotopes
    (energyRange,) = isotope.energy_ranges
    p = energyRange.parameters
    head = (isotope.za, isotope.abn, isotope.lfw, energyRange.el, energyRange.eh,
            energyRange.lru, energyRange.lrf, energyRange.nro, energyRange.naps,
            p.spi, p.ap, p.nls)
    blocks = [(b.l, b.awri, b.apl_or_qx, b.lrx,
               [(r.energy, r.spin, r.c3, r.c4, r.c5, r.c6) for r in b.resonances])
              for b in p.l_values]
    return head, blocks


def test_the_range_comes_back_record_for_record(tmp_path):
    assert _records(_viaGnds(tmp_path)) == _records(TAPE)


def test_its_resonance_cross_sections_are_the_tapes(tmp_path):
    from kika.endf.reconstruction import context_from_endf
    from kika.processing.resonances.prepare import prepare_resonances

    sigma = []
    for path in (TAPE, _viaGnds(tmp_path)):
        suite = kika.read(str(path))
        context, _ = context_from_endf(read_endf(str(path)), suite)
        region = suite.resonances.resolved[0]
        energies = np.geomspace(region.domainMin, region.domainMax, 4001)[:-1]
        sigma.append(prepare_resonances(suite.resonances, context).evaluate(energies))
    assert sigma[0].keys() == sigma[1].keys()
    for mt in sigma[0]:
        assert np.array_equal(sigma[0][mt], sigma[1][mt]), mt

"""MF7/MT2: the elastic cross sections, from the closed forms ENDF-102 states.

Both elastic laws are a formula over the tabulated data, with no integration:

* coherent (Bragg): σ = S(E, T) / E, S cumulative and held between edges;
* incoherent (Debye-Waller): σ = SB/2 · (1 − e^(−4EW′)) / (2EW′).

The assertions are the properties a wrong reading would break, not values
copied out of the implementation: zero below the first edge, the jump *at* an
edge and not one record later, the exact 1/E fall between edges, SB as the
E → 0 limit, and refusals where an answer would have to be invented.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf


def mt2_of(path):
    return read_endf(str(path)).mf[7].sections[2]


def test_coherent_is_zero_below_the_first_bragg_edge(micro_tsl_bemetal_elastic_tape):
    coherent = mt2_of(micro_tsl_bemetal_elastic_tape).coherent
    first = coherent.energies[0]
    sigma = coherent.cross_section([first * 0.5, first * (1 - 1e-9)], 293.6)
    assert np.all(sigma == 0.0)


def test_coherent_steps_at_each_edge_and_falls_as_one_over_e_between(
        micro_tsl_bemetal_elastic_tape):
    """At edge *i* σ is S_i / E_i; just below it, S_(i-1) / E_i."""
    coherent = mt2_of(micro_tsl_bemetal_elastic_tape).coherent
    edges = np.asarray(coherent.energies)
    s = np.asarray(coherent.table.at_temperature(293.6))
    for i in (1, 10, 500):
        at, below = coherent.cross_section([edges[i], edges[i] * (1 - 1e-12)], 293.6)
        assert at == pytest.approx(s[i] / edges[i], rel=1e-12)
        assert below == pytest.approx(s[i - 1] / edges[i], rel=1e-9)
    # Halfway between two edges the value is the left edge's S over E, not
    # an interpolation towards the right one.
    mid = 0.5 * (edges[3] + edges[4])
    assert coherent.cross_section(mid, 293.6) == pytest.approx(s[3] / mid)


def test_coherent_holds_the_last_s_beyond_the_last_edge(micro_tsl_bemetal_elastic_tape):
    coherent = mt2_of(micro_tsl_bemetal_elastic_tape).coherent
    last_s = coherent.table.at_temperature(1200.0)[-1]
    energies = np.array([10.0, 20.0])
    np.testing.assert_allclose(coherent.cross_section(energies, 1200.0),
                               last_s / energies)


def test_coherent_temperatures_are_exact_not_interpolated(micro_tsl_bemetal_elastic_tape):
    coherent = mt2_of(micro_tsl_bemetal_elastic_tape).coherent
    with pytest.raises(KeyError, match="not tabulated"):
        coherent.cross_section(0.1, 350.0)


def test_coherent_refuses_a_table_that_is_not_a_histogram(micro_tsl_bemetal_elastic_tape):
    coherent = mt2_of(micro_tsl_bemetal_elastic_tape).coherent
    coherent.table.interp = [(len(coherent.energies), 2)]
    with pytest.raises(ValueError, match="histogram"):
        coherent.cross_section(0.1, 293.6)


def test_incoherent_tends_to_the_bound_cross_section_at_zero_energy(micro_tsl_sch4_tape):
    incoherent = mt2_of(micro_tsl_sch4_tape).incoherent
    sigma = incoherent.cross_section([0.0, 1e-12], 22.0)
    np.testing.assert_allclose(sigma, incoherent.sb, rtol=1e-9)


def test_incoherent_matches_the_formula_and_its_high_energy_limit(
        micro_tsl_un_elastic_tape):
    incoherent = mt2_of(micro_tsl_un_elastic_tape).incoherent
    w = incoherent.debye_waller(296.0)
    assert w == pytest.approx(2.888359)
    e = 0.0253
    expected = incoherent.sb / 2 * (1 - np.exp(-4 * e * w)) / (2 * e * w)
    assert incoherent.cross_section(e, 296.0) == pytest.approx(expected, rel=1e-12)
    # 4EW' = 1155 at 100 eV: the exponential is gone and σ is SB / 4EW'.
    assert incoherent.cross_section(100.0, 296.0) == pytest.approx(
        incoherent.sb / (4 * 100.0 * w), rel=1e-12)


def test_debye_waller_interpolates_inside_the_table_and_refuses_outside(
        micro_tsl_un_elastic_tape):
    incoherent = mt2_of(micro_tsl_un_elastic_tape).incoherent
    # lin-lin between 296 K (2.888359) and 400 K (3.640491).
    assert incoherent.interp == [(8, 2)]
    expected = 2.888359 + (3.640491 - 2.888359) * (348.0 - 296.0) / (400.0 - 296.0)
    assert incoherent.debye_waller(348.0) == pytest.approx(expected)
    with pytest.raises(KeyError, match="outside"):
        incoherent.debye_waller(20.0)


def test_lthr3_returns_both_terms_and_their_sum(micro_tsl_un_elastic_tape):
    mt2 = mt2_of(micro_tsl_un_elastic_tape)
    energies = np.geomspace(1e-4, 1.0, 50)
    sigma = mt2.cross_sections(energies, 296.0)
    assert set(sigma) == {"coherent", "incoherent", "total"}
    np.testing.assert_allclose(sigma["total"], sigma["coherent"] + sigma["incoherent"])


def test_a_single_block_has_no_total(micro_tsl_bemetal_elastic_tape):
    sigma = mt2_of(micro_tsl_bemetal_elastic_tape).cross_sections([0.1], 293.6)
    assert set(sigma) == {"coherent"}

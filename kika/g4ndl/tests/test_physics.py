"""Phase 5: physics checks and p(mu|E) of an elastic suite read from G4NDL.

The whole-library sweep, with the anomalies each library ships, is in
``test_full_libraries.py``; the comparison against Geant4's own construction
of p(mu|E) is in ``kika-workspace/myworkspace/G4NDL/harness``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

import kika.g4ndl as g4ndl
from kika.g4ndl import IsotopeKey
from kika.g4ndl.decode import decodeElastic
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.physics import (
    angularPdf, checkElastic, differentialCrossSection, legendreDensity,
    legendreMinimum, tableIntegral,
)
from kika.g4ndl.tokens import TokenStream

DATA = Path(__file__).parent / "data"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
SYNTH = g4ndl.open(DATA / "synthetic")
_CS = "0 0\n3\n1.0e-5 4.0 1.0e6 3.0 2.0e7 1.0\n"


def _suite(fs_text, cs_text=_CS):
    suite, _ = decodeElastic(parse_cross_section(TokenStream(cs_text)),
                             parse_elastic_fs(TokenStream(fs_text)), IsotopeKey(1, 1))
    return suite


# ------------------------------------------------------- building blocks

@pytest.mark.parametrize("seed", range(5))
def test_legendre_minimum_is_exact_not_a_grid_value(seed):
    rng = np.random.default_rng(seed)
    a = np.concatenate([[1.0], rng.normal(0, 0.3, 12)])
    pmin, mu = legendreMinimum(a)
    dense = np.linspace(-1, 1, 2_000_001)
    assert pmin <= legendreDensity(a, dense).min() + 1e-12
    assert legendreDensity(a, mu) == pytest.approx(pmin)


def test_a_negative_lobe_between_grid_points_is_found():
    # p = 1/2 + c P_20: its deepest dip sits between the nodes of a 101-point grid.
    a = np.zeros(21)
    a[0] = 1.0
    a[20] = 0.0617
    coarse = legendreDensity(a, np.linspace(-1, 1, 101)).min()
    pmin, _ = legendreMinimum(a)
    assert coarse > 0 > pmin


@pytest.mark.parametrize("code,x,y,exact", [
    (2, [0.0, 1.0], [1.0, 3.0], 2.0),
    (4, [0.0, 1.0], [1.0, np.e], np.e - 1),                      # ln y linear
    (3, [1.0, np.e], [0.0, 1.0], 1.0),                           # y = ln x
    (5, [1.0, 2.0], [1.0, 4.0], 7.0 / 3.0),                      # y = x^2
    (1, [0.0, 2.0], [1.5, 9.0], 3.0),
])
def test_table_integral_is_exact_on_each_law(code, x, y, exact):
    assert tableIntegral(x, y, [2], [code]) == pytest.approx(exact, rel=1e-14)


def test_table_integral_uses_each_intervals_own_law():
    # Regions (2 lin-lin, 3 log-lin): the boundary node closes region one.
    mu = [-1.0, 0.0, 1.0]
    p = [0.5, 0.5, 0.5 * np.e]
    expected = 0.5 + 0.5 * (np.e - 1)
    assert tableIntegral(mu, p, [2, 3], [2, 4]) == pytest.approx(expected, rel=1e-14)


# ------------------------------------------------------------ the checks

@pytest.mark.parametrize("target", ["H1", "He3", "C12", "N14", "Co58m1"])
def test_real_fixtures_are_normalised_and_return_their_moments(target):
    check = checkElastic(JEFF.read(target))
    assert not check.byKind("moments") and not check.byKind("nonfinite")
    assert not [f for f in check.byKind("normalisation") if f.block == "Legendre"]
    assert check.maxNormalisationError < 1e-4


def test_loglin_tables_are_integrated_on_their_law_not_by_trapezoids():
    check = checkElastic(JEFF.read("C12"))
    assert check.maxNormalisationError < 1e-6
    table = JEFF.elasticFinalState("C12").tabulated.records[-1]
    trapezoid = abs(np.trapezoid(table.probability, table.mu) - 1)
    assert trapezoid > 100 * check.maxNormalisationError


def test_a_negative_legendre_density_is_reported_not_corrected():
    suite = _suite("1 1.0 2\n2\n1 2 2\n0.0 1.0e-5 0 1 0.0\n0.0 2.0e7 0 1 0.5\n")
    check = checkElastic(suite)
    (finding,) = check.byKind("negative")
    assert finding.energy == 2.0e7 and finding.value == pytest.approx(-0.25)
    assert "mu = -1" in finding.detail
    # Nothing was changed on the way: the coefficient is still the file's.
    assert angularPdf(suite, 2.0e7, -1.0) == pytest.approx(-0.25)


def test_a_slipped_factor_on_a0_is_caught():
    suite = _suite("1 1.0 2\n1\n1 1 2\n0.0 1.0e6 0 1 0.1\n")
    leg = suite.reactions[2].outputChannel.products.byPid("n")[0] \
        .distribution["eval"].angular.function1ds[0]
    leg.coefficients[0] = 4 * np.pi
    assert checkElastic(suite).byKind("normalisation")


def test_an_unnormalised_and_a_negative_table_are_reported():
    suite = _suite("2 1.0 2\n2\n1 2 2\n"
                   "0.0 1.0e6 0 2 1 2 2 -1.0 0.5 1.0 0.5\n"
                   "0.0 2.0e6 0 2 1 2 2 -1.0 0.0 1.0 0.0\n")
    check = checkElastic(suite)
    (f,) = check.byKind("normalisation")
    assert f.energy == 2.0e6 and f.value == -1.0


def test_transition_jump_is_measured():
    # C-12 (JEFF-4.0): Legendre and table disagree by 8 % in L1 at the
    # transition -- a property of the file, measured, not a tolerance.
    check = checkElastic(JEFF.read("C12"))
    assert check.transitionJump == pytest.approx(0.0819295, rel=1e-5)
    assert checkElastic(JEFF.read("H1")).transitionJump is None


def test_isotropic_is_clean():
    check = checkElastic(SYNTH.read("H1"))
    assert check.isClean and check.minDensity == 0.5


# ------------------------------------------------------------ evaluation

def test_at_a_node_the_pdf_is_the_record():
    suite = JEFF.read("He3")
    rec = JEFF.elasticFinalState("He3").tabulated.records[3]
    assert angularPdf(suite, rec.energy, rec.mu).tolist() == rec.probability.tolist()


def test_between_nodes_linlin_in_energy():
    suite = _suite("1 1.0 2\n2\n1 2 2\n0.0 1.0e6 0 1 0.1\n0.0 3.0e6 0 1 0.3\n")
    assert angularPdf(suite, 2.0e6, 1.0) == pytest.approx(0.5 + 1.5 * 0.2)


def test_between_nodes_linlog_in_energy():
    suite = _suite("1 1.0 2\n2\n1 2 3\n0.0 1.0e6 0 1 0.1\n0.0 1.0e8 0 1 0.3\n")
    # Halfway in ln E is 1e7: a_1 = 0.2.
    assert angularPdf(suite, 1.0e7, 1.0) == pytest.approx(0.5 + 1.5 * 0.2)


def test_legendre_of_different_order_is_zero_padded():
    suite = _suite("1 1.0 2\n2\n1 2 2\n0.0 1.0e6 0 0\n0.0 3.0e6 0 2 0.0 0.4\n")
    mu = np.linspace(-1, 1, 7)
    expected = legendreDensity([1.0, 0.0, 0.2], mu)
    assert np.allclose(angularPdf(suite, 2.0e6, mu), expected, rtol=0, atol=1e-15)


def test_side_picks_the_record_at_a_repeated_energy():
    suite = _suite("1 1.0 2\n3\n1 3 2\n0.0 1.0e6 0 1 0.1\n0.0 2.0e6 0 1 0.2\n"
                   "0.0 2.0e6 0 1 0.3\n")
    assert angularPdf(suite, 2.0e6, 1.0, side="left") == pytest.approx(0.5 + 1.5 * 0.2)
    assert angularPdf(suite, 2.0e6, 1.0, side="right") == pytest.approx(0.5 + 1.5 * 0.3)


def test_side_picks_the_representation_at_the_transition():
    suite = JEFF.read("C12")
    fs = JEFF.elasticFinalState("C12")
    et = fs.transitionEnergy
    mu = fs.tabulated.records[0].mu
    left = angularPdf(suite, et, mu, side="left")
    right = angularPdf(suite, et, mu, side="right")
    assert np.allclose(left, legendreDensity(np.r_[1.0, fs.legendre.records[-1].coefficients], mu))
    # LOGLIN in mu: evaluating at a node goes through exp(log(...)), 1 ulp off.
    assert np.allclose(right, fs.tabulated.records[0].probability, rtol=1e-14, atol=0)


def test_no_extrapolation():
    suite = JEFF.read("H1")
    e = JEFF.elasticFinalState("H1").legendre.energies
    with pytest.raises(ValueError, match="outside the data"):
        angularPdf(suite, e[-1] * 1.01, 0.0)
    with pytest.raises(ValueError, match="side"):
        angularPdf(suite, e[0], 0.0, side="middle")


def test_differential_cross_section_integrates_to_sigma():
    suite = JEFF.read("C12")
    E = 5.0e6
    nodes, weights = npleg.leggauss(64)
    total = 2 * np.pi * np.sum(weights * differentialCrossSection(suite, E, nodes))
    sigma = float(suite.reactions[2].crossSection["recon"].evaluate(E))
    assert total == pytest.approx(sigma, rel=1e-4)

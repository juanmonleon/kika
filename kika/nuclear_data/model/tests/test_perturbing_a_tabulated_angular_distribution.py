"""The arithmetic of perturbing a tabulated f(mu) by Legendre-order factors.

T1 of ``docs/library/mf4_tabulated_perturbation_roadmap.md``: one table, one
incident energy, no model nodes. The energy dimension and the gate against the
Legendre applier are further down the roadmap and further down this file.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import legendre as L

from kika.nuclear_data.model.angular_tables import (legendreMoments,
                                                    perturbTabulatedAngular,
                                                    tableIntegral)


def _pdf(coefficients, mu):
    """f(mu) = sum (2l+1)/2 a_l P_l(mu)."""
    series = [0.5 * (2 * l + 1) * a for l, a in enumerate(coefficients)]
    return L.legval(mu, series)


def _forwardPeaked(n=91):
    """A real-looking table: smooth plus a forward peak no low order holds."""
    mu = np.concatenate([np.linspace(-1.0, 0.9, n - 20), np.linspace(0.905, 1.0, 20)])
    p = 0.3 + 0.2 * mu + 5.0 * np.exp(-((1.0 - mu) / 0.03))
    return mu, p / tableIntegral(mu, p)


def test_every_factor_one_gives_the_table_back_bit_for_bit():
    mu, p = _forwardPeaked()
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.0, 2: 1.0, 3: 1.0})
    assert np.array_equal(pPrime, p)
    assert pPrime is not p
    assert all(delta == 0.0 for delta in info["deltas"].values())


def test_the_projection_of_a_linlin_table_is_exact():
    """f linear between nodes: each panel integrates to rounding, at any order.

    Checked against the closed form for a single panel spanning [-1, 1], where
    f = a + b mu gives a_0 = 2a, a_1 = 2b/3 and nothing above.
    """
    mu = np.array([-1.0, 1.0])
    p = np.array([0.2, 0.8])                   # a = 0.5, b = 0.3
    moments = legendreMoments(mu, p, range(0, 7))
    assert moments[0] == pytest.approx(1.0, abs=1e-15)
    assert moments[1] == pytest.approx(0.2, abs=1e-15)
    for order in range(2, 7):
        assert moments[order] == pytest.approx(0.0, abs=1e-15)


def test_the_moments_shift_by_exactly_the_deltas_and_nothing_else_moves():
    """Project ``f'`` and ``f``: the named orders differ by delta, others by ~0.

    Exact up to the lin-lin interpolation of the correction between nodes,
    which is what D3 measures; at 401 nodes it is below 1e-5.
    """
    mu = np.cos(np.linspace(np.pi, 0.0, 401))
    p = _pdf([1.0, 0.4, 0.2, 0.1], mu)
    factors = {1: 1.10, 2: 0.95, 3: 1.20}
    pPrime, info = perturbTabulatedAngular(mu, p, factors)
    before = legendreMoments(mu, p, range(0, 8))
    after = legendreMoments(mu, pPrime, range(0, 8))
    for order in range(0, 8):
        expected = info["deltas"].get(order, 0.0)
        assert after[order] - before[order] == pytest.approx(expected, abs=1e-5), order


def test_the_integral_is_kept_to_the_linearisation():
    mu, p = _forwardPeaked()
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.1, 2: 0.9})
    assert info["integral_before"] == pytest.approx(1.0, abs=1e-12)
    assert abs(info["integral_after"] - info["integral_before"]) < 1e-3


def test_an_analytic_case_comes_out_exact():
    """f = 1/2 (1 + 3 a1 mu), 201 nodes, c_1 = 1.1: f' = 1/2 (1 + 3.3 a1 mu)."""
    a1 = 0.3
    mu = np.linspace(-1.0, 1.0, 201)
    p = 0.5 * (1.0 + 3.0 * a1 * mu)
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.1})
    np.testing.assert_allclose(pPrime, 0.5 * (1.0 + 3.0 * 1.1 * a1 * mu),
                               rtol=0, atol=1e-14)
    assert info["integral_after"] == pytest.approx(1.0, abs=1e-14)


def test_the_forward_peak_survives_a_low_order_perturbation():
    """The point of D-B: the naive resum would flatten the peak; this keeps it."""
    mu, p = _forwardPeaked()
    pPrime, _info = perturbTabulatedAngular(mu, p, {1: 1.05, 2: 1.05})
    peak = mu > 0.95
    assert np.max(np.abs(pPrime[peak] / p[peak] - 1.0)) < 0.05


def test_a_negative_node_is_reported_and_left_alone():
    mu = np.linspace(-1.0, 1.0, 51)
    p = _pdf([1.0, 0.3], mu)                   # 0.05 at mu = -1
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.5})
    assert info["n_negative"] > 0 and info["min_p"] < 0.0
    assert pPrime[0] == pytest.approx(0.5 * (1.0 - 3.0 * 0.45))


def test_the_magnitude_order_and_unknown_laws_are_refused():
    mu = np.linspace(-1.0, 1.0, 5)
    p = np.full(5, 0.5)
    with pytest.raises(ValueError, match="magnitude"):
        perturbTabulatedAngular(mu, p, {0: 1.1})
    with pytest.raises(NotImplementedError, match="transcendental"):
        perturbTabulatedAngular(mu, p, {1: 1.1}, pairs=[(5, 5)])
    with pytest.raises(ValueError, match=r"not \[-1, 1\]"):
        perturbTabulatedAngular(np.linspace(-0.5, 1.0, 5), p, {1: 1.1})


def test_a_histogram_table_is_projected_exactly():
    """INT=1: f = 0.25 on [-1, 0), 0.75 on [0, 1]; a_1 = int mu f = 0.25."""
    mu = np.array([-1.0, 0.0, 1.0])
    p = np.array([0.25, 0.75, 0.75])
    moments = legendreMoments(mu, p, [0, 1], pairs=[(3, 1)])
    assert moments[0] == pytest.approx(1.0, abs=1e-15)
    assert moments[1] == pytest.approx(0.25, abs=1e-15)


# ----------------------------------------------------------------------
# T2: the gate against the Legendre applier
# ----------------------------------------------------------------------
#
# Fabricate an LTT=2 distribution whose Legendre truth is known -- the Legendre
# region of real JEFF-4.0 Fe-56 MT2, tabulated at every one of its energies --
# and require the two appliers to agree on f'(mu, E) at the table's nodes, with
# the same MF34 grids and the same factors. They differ by exactly one thing:
# the tabulated applier projects the table, and the projection of a lin-lin
# table is not the Legendre coefficients it was tabulated from. That
# difference, measured per order, is the floor; the gate is that the appliers
# disagree by no more than the floor propagates to.

from pathlib import Path

from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF4MT
from kika.nuclear_data.model.functions.higher import XYs2d
from kika.nuclear_data.model.functions.xys1d import XYs1d
from kika.nuclear_data.model.perturbation import (_legendreRegions,
                                                  applyLegendreFactors,
                                                  applyTabulatedFactors)
from kika.sampling.model_blocks import legendre_covariance_index

FE56 = str(Path(__file__).resolve().parents[3] / "endf" / "tests" / "data"
           / "micro_fe56_structural.endf")
ORDERS = range(1, 7)


def _tabulated(region, mu):
    tables = [XYs1d(xs=mu.copy(), ys=_pdf(f.coefficients, mu),
                    outerDomainValue=f.outerDomainValue, index=f.index)
              for f in region.function1ds]
    return XYs2d(function1ds=tables, interpolation=region.interpolation,
                 axes=region.axes)


@pytest.fixture(scope="module")
def fe56():
    distribution, _provenance, _report = decodeMF4MT(
        read_endf(FE56).get_file(4).sections[2])
    ((_container, _position, legendre),) = _legendreRegions(distribution.angular)
    suite, _report = decodeCovarianceSuite(read_endf(FE56))
    (entry,) = legendre_covariance_index(suite, relative=True).values()
    factors, edges = {}, {}
    for triplet in entry["triplets"]:
        grid = np.asarray(entry["grids"][triplet], dtype=float)
        order = int(triplet[2])
        factors[order] = 1.0 + 0.05 * np.cos(np.arange(grid.size - 1) + order)
        edges[order] = grid
    return legendre, factors, edges


def _gate(fe56, nMu):
    legendre, factors, edges = fe56
    mu = np.cos(np.linspace(np.pi, 0.0, nMu))
    table = _tabulated(legendre, mu)
    floor = max(
        abs(legendreMoments(t.xs, t.ys, ORDERS)[order]
            - (f.coefficients[order] if order < f.coefficients.size else 0.0))
        for f, t in zip(legendre.function1ds, table.function1ds) for order in ORDERS)

    byLegendre, _ = applyLegendreFactors(legendre, factors, edges)
    byTable, info = applyTabulatedFactors(table, factors, edges)
    energiesL = [f.outerDomainValue for f in byLegendre.function1ds]
    energiesT = [f.outerDomainValue for f in byTable.function1ds]
    assert np.array_equal(energiesL, energiesT)
    worst = max(float(np.max(np.abs(_pdf(a.coefficients, b.xs) - b.ys)))
                for a, b in zip(byLegendre.function1ds, byTable.function1ds))
    largestFactor = max(float(np.max(np.abs(np.asarray(f) - 1.0)))
                        for f in factors.values())
    bound = largestFactor * floor * sum(0.5 * (2 * l + 1) for l in ORDERS)
    return worst, bound, floor, info


def test_the_two_appliers_agree_to_the_projection_floor(fe56):
    """Measured 2026-10-05 at 401 nodes: 3.1e-5 against a bound of 4.2e-4."""
    worst, bound, floor, info = _gate(fe56, 401)
    assert floor < 1e-3
    assert worst <= bound, f"{worst:.2e} > {bound:.2e} (floor {floor:.2e})"
    assert info["max_integral_change"] < 1e-5
    assert info["n_inserted"] > 0


def test_a_coarse_mu_grid_is_what_decision_d3_is_about(fe56):
    """At 21 nodes the gate still holds against its own floor, but the floor is
    0.13 in a_l and the integral moves 1.7e-3 -- the size that D3 (refining mu)
    has to answer, measured rather than argued."""
    worst, bound, floor, info = _gate(fe56, 21)
    assert worst <= bound
    assert floor > 1e-2
    assert 1e-4 < info["max_integral_change"] < 1e-2


# ----------------------------------------------------------------------
# T3: the incident-energy dimension
# ----------------------------------------------------------------------

def _threeTables():
    """Tables at 1, 2 and 4 MeV, lin-lin in E, a_1 = 0.1, 0.2, 0.4."""
    mu = np.linspace(-1.0, 1.0, 41)
    tables = [XYs1d(xs=mu.copy(), ys=_pdf([1.0, a1], mu), outerDomainValue=e,
                    index=i)
              for i, (e, a1) in enumerate([(1e6, 0.1), (2e6, 0.2), (4e6, 0.4)])]
    return XYs2d(function1ds=tables)


def _a1(node):
    return [legendreMoments(f.xs, f.ys, [1])[1] for f in node.function1ds]


def test_an_edge_between_two_tables_gets_the_interpolated_table_twice():
    angular = _threeTables()
    out, info = applyTabulatedFactors(angular, {1: np.array([1.1, 0.9])},
                                      {1: np.array([1e6, 3e6, 4e6])})
    energies = [f.outerDomainValue for f in out.function1ds]
    assert energies == [1e6, 2e6, 3e6, 3e6, 4e6]
    assert info["n_inserted"] == 2
    # 3 MeV is halfway between a_1 = 0.2 and 0.4; below the edge x1.1, above
    # x0.9. 4 MeV sits on the block's last edge and is outside it -- the rule
    # _flatFactors states for MF3 and the Legendre applier alike.
    np.testing.assert_allclose(_a1(out), [0.11, 0.22, 0.33, 0.27, 0.40],
                               rtol=1e-12)


def test_an_edge_on_a_tabulated_energy_duplicates_that_table():
    out, info = applyTabulatedFactors(_threeTables(), {1: np.array([1.1, 0.9])},
                                      {1: np.array([1e6, 2e6, 4e6])})
    assert [f.outerDomainValue for f in out.function1ds] == [1e6, 2e6, 2e6, 4e6]
    assert info["n_inserted"] == 1
    np.testing.assert_allclose(_a1(out), [0.11, 0.22, 0.18, 0.40], rtol=1e-12)


def test_coverage_that_ends_inside_the_table_steps_or_ramps():
    """A block over [1, 3] MeV: "step" writes the edge at 3 MeV, "ramp" not."""
    factors, edges = {1: np.array([1.2])}, {1: np.array([1e6, 3e6])}
    stepped, infoStep = applyTabulatedFactors(_threeTables(), factors, edges)
    ramped, infoRamp = applyTabulatedFactors(_threeTables(), factors, edges,
                                             coverageEdges="ramp")
    assert [f.outerDomainValue for f in stepped.function1ds] == [1e6, 2e6, 3e6, 3e6, 4e6]
    np.testing.assert_allclose(_a1(stepped), [0.12, 0.24, 0.36, 0.30, 0.40],
                               rtol=1e-12)
    assert [f.outerDomainValue for f in ramped.function1ds] == [1e6, 2e6, 4e6]
    assert infoStep["n_inserted"] == 2 and infoRamp["n_inserted"] == 0


def test_a_block_outside_the_table_changes_nothing():
    angular = _threeTables()
    out, info = applyTabulatedFactors(angular, {1: np.array([1.5])},
                                      {1: np.array([1e7, 2e7])})
    assert all(a is b for a, b in zip(out.function1ds, angular.function1ds))
    assert info["per_order"][1]["n_scaled"] == 0 and info["n_inserted"] == 0


def test_a_repair_callback_sees_every_negative_node_and_its_answer_is_kept():
    seen = []

    def repair(mu, p, pPrime, info, pairs):
        seen.append(info["min_p"])
        return np.clip(pPrime, 0.0, None), {"what": "clip, for the test only"}

    mu = np.linspace(-1.0, 1.0, 21)
    angular = XYs2d(function1ds=[XYs1d(xs=mu.copy(), ys=_pdf([1.0, 0.3], mu),
                                       outerDomainValue=e) for e in (1e6, 2e6)])
    out, info = applyTabulatedFactors(angular, {1: np.array([1.5])},
                                      {1: np.array([5e5, 3e6])}, repair=repair)
    assert len(seen) == 2 and all(v < 0 for v in seen)
    assert info["n_negative_nodes"] == 2 and len(info["positivity_events"]) == 2
    assert min(float(f.ys.min()) for f in out.function1ds) == 0.0

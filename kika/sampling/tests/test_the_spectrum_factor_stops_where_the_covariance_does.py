"""PD-1 and PD-2: a PFNS factor acts where MF35 states it, and stops there.

Three places where the model applier used to perturb more, or less, than the
covariance covers -- each decided by Juan on 2026-10-08 and each fixed on the
model only (the legacy applier is the fixed reference, PD-4):

* **PD-1, a band stated at a point** (``E1 == E2``). B-VII.1 Pu-240 states its
  only band as ``[0.5, 0.5]`` MeV and JEFF-4.0 U-238 its first as
  ``[1e-5, 1e-5]`` eV. The half-open band test matched such a band only when it
  was the last one, so U-238's band 0 perturbed **nothing**, silently. Now the
  MF5 node at exactly ``E1`` is perturbed, nothing around it, and the run says
  "covariance stated at a point, not over a band".
* **PF-2, the factor leaking above a band's top.** Where no band starts at a
  band's ``E2`` the next incident node is unperturbed, so the factor ramped
  across the whole interval up to it. Measured on ENDF/B-VIII.1 U-233 (one band
  to 5 MeV, next node 6 MeV), seed 11: 63 % at 5.01 MeV, 36 % at 5.5 MeV. Fixed
  by an unperturbed node at ``E2 (1 + s)``.
* **PF-3, the ramp across the MF35 grid's outer edge.** Where the MF5 table
  reaches past ``[g_0, g_N]`` the factor steps to 1 there, and the edges were
  not stepping candidates. Measured on B-VIII.1 U-233 as the mass outside the
  grid moved net of the renormalisation: 2.1 % of it (3.3e-10 of chi) before,
  6.2e-5 of it (6.0e-13) after; B-VIII.1 Pu-240 38 % before, 0.37 % after.

The synthetic tests fail on the applier as it was before 2026-10-08 and pass
on the fixed one; the tape tests pin the measured numbers on real witnesses
(JEFF-4.0 U-238 for PD-1, ENDF/B-VIII.1 U-233 for PF-2/PF-3 -- B-VII.1 Pu-240
is not on this box). Plan: ``kika-workspace/docs/pfns/pfns_mf5_mf35_roadmap.md``,
Q2 and "Decisions that are Juan's".
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter.energy import decodeMF5MT
from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.functions import XYs1d, XYs2d
from kika.nuclear_data.model.perturbation import (POINT_BAND_NOTE,
                                                  SPECTRUM_OUTER_SHOULDER,
                                                  applySpectrumFactors)
from kika.sampling.mf35_sampling import (band_grids, build_pfns_covariance,
                                         generate_pfns_samples, pfns_ratio_rule)

S = SPECTRUM_OUTER_SHOULDER


# ======================================================================
# A synthetic spectrum small enough to reason about by hand
# ======================================================================

#: Outgoing grid of every child: it reaches past the "MF35" grid below at
#: both ends, and neither grid edge (1.0, 5.0) is a node -- the PF-3 shape.
XS = np.array([0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5, 6.0])
GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
RATIOS = np.array([1.3, 0.8, 1.2, 0.7])


def _spectrum(nodes=(1.0, 2.0, 3.0, 4.0, 5.0)):
    """An ``XYs2d`` of normalised lin-lin tables, one shape per incident node."""
    children = []
    for k, energy in enumerate(nodes):
        ys = XS * np.exp(-XS / (1.0 + 0.1 * k))
        ys = ys / np.trapezoid(ys, XS)
        child = XYs1d(xs=XS.copy(), ys=ys, interpolation=Interpolation.linlin)
        child.outerDomainValue = float(energy)
        child.index = k
        children.append(child)
    return XYs2d(function1ds=children)


def _fixed(ratios=RATIOS):
    """A ``ratiosFor`` that hands every node the same ratios."""
    return lambda band, p0, covered: (np.asarray(ratios, dtype=float), {})


def _moved(form, perturbed, energy):
    """``max |P/P0 - 1|`` of the perturbed spectrum at incident *energy*."""
    from kika.algebra import group_integrals

    x0, y0 = form.evaluateAtOuter(energy)
    x1, y1 = perturbed.evaluateAtOuter(energy)
    p0 = group_integrals(x0, y0, 2, GRID)
    p1 = group_integrals(x1, y1, 2, GRID)
    return float(np.max(np.abs(p1 / p0 - 1.0)))


# ======================================================================
# PD-1 -- a band stated at a point
# ======================================================================

def test_a_point_band_perturbs_the_node_at_its_point_and_nothing_else():
    """``[3, 3]``: the node at 3 moves, the shoulders either side do not.

    Before PD-1 a point band that was not the last one matched no node at all;
    this one is the only band, so it matched by accident of position, and the
    factor then ramped on both sides of it (PF-1 below, PF-2 above).
    """
    form = _spectrum()
    perturbed, diagnostics = applySpectrumFactors(
        form, {0: (3.0, 3.0)}, {0: GRID}, _fixed())

    assert [entry["outer_value"] for entry in diagnostics["per_node"]] == [3.0]
    assert sorted(perturbed.outerDomainValues) == pytest.approx(
        [1.0, 2.0, 3.0 * (1 - S), 3.0, 3.0 * (1 + S), 4.0, 5.0], rel=1e-12)
    assert _moved(form, perturbed, 3.0) > 0.1
    for energy in (2.5, 3.0 * (1 - S), 3.0 * (1 + S), 3.5):
        assert _moved(form, perturbed, energy) < 1e-12, energy

    record = diagnostics["point_bands"][0]
    assert record["band_kind"] == "point" and record["node_at_point"]
    assert POINT_BAND_NOTE in record["note"]


def test_a_point_band_takes_its_node_from_a_band_that_starts_there():
    """JEFF-4.0 U-238's layout: ``[1, 1]`` and then ``[1, 3)``.

    The half-open test gave the node at 1 to the range band and the point band
    nothing. The node is the one thing the point band is a statement about, so
    it takes it, and the range band starts at the shoulder just above.
    """
    form = _spectrum()
    point, wide = RATIOS, np.array([0.9, 1.1, 0.95, 1.05])
    rules = {0: _fixed(point), 1: _fixed(wide)}
    perturbed, diagnostics = applySpectrumFactors(
        form, {0: (1.0, 1.0), 1: (1.0, 3.0)}, {0: GRID, 1: GRID},
        lambda band, p0, covered: rules[band](band, p0, covered))

    owner = {entry["outer_value"]: entry["band"]
             for entry in diagnostics["per_node"]}
    assert owner[1.0] == 0
    shoulder, = [v for v in owner if 1.0 < v < 2.0]
    assert shoulder == pytest.approx(1.0 * (1 + S), rel=1e-12)
    assert owner[shoulder] == 1
    assert owner[2.0] == 1
    assert owner[3.0] == 1          # the last range band is closed at the top


def test_a_point_band_with_no_node_at_its_point_moves_nothing_and_says_so():
    form = _spectrum()
    perturbed, diagnostics = applySpectrumFactors(
        form, {0: (2.5, 2.5)}, {0: GRID}, _fixed())
    assert diagnostics["per_node"] == []
    assert diagnostics["n_outer_inserted"] == 0
    assert list(perturbed.outerDomainValues) == list(form.outerDomainValues)
    record = diagnostics["point_bands"][0]
    assert not record["node_at_point"]
    assert "nothing is perturbed" in record["note"]


def test_jeff40_u238_band_0_is_a_point_and_is_perturbed_at_it(u238_tape):
    """The real witness of PD-1 on this box (B-VII.1 Pu-240 is not here).

    JEFF-4.0 U-238 states band 0 as ``[1e-5, 1e-5]`` eV and band 1 as
    ``[1e-5, 1e3)``. Before PD-1, band 0 perturbed no node and its draw was
    thrown away without a word; now it perturbs the 1e-5 eV node, band 1 starts
    at the shoulder 1.001e-5 eV, and the run says so by name.
    """
    from kika.sampling.model_perturbation import perturbFromModel

    endf = read_endf(str(u238_tape), mf_numbers=[5, 35])
    suite, mf5, bands = build_pfns_covariance(endf, mt=18)
    assert bands[0] == (1e-5, 1e-5)
    drawn, _info = generate_pfns_samples(suite, 1, seed=11, verbose=False)
    deltas = {key[-1]: value[0] for key, value in drawn.items()}
    form, _provenance, _report = decodeMF5MT(mf5)
    _perturbed, diagnostics = applySpectrumFactors(
        form, dict(enumerate(bands)), dict(enumerate(band_grids(suite))),
        pfns_ratio_rule(deltas))

    byBand = {}
    for entry in diagnostics["per_node"]:
        byBand.setdefault(entry["band"], []).append(entry["outer_value"])
    assert byBand[0] == [1e-5]
    assert byBand[1][0] == pytest.approx(1e-5 * (1 + S), rel=1e-12)
    assert diagnostics["point_bands"][0]["node_at_point"]

    run = perturbFromModel(str(u238_tape), {35: {"index": [0, 1]}}, nSamples=1,
                           seed=11, dryRun=True)
    assert any(POINT_BAND_NOTE in note and "band 0" in note
               for note in run.notes), run.notes


# ======================================================================
# PF-2 -- the factor stops at the top of a band
# ======================================================================

def test_the_factor_stops_at_the_top_of_the_last_band():
    """Band ``[1, 3]`` on a grid to 5: nothing above 3 may move.

    Before the fix the unperturbed node at 4 was the only thing stopping the
    factor, and halfway there the spectrum was still moved by half of it.
    """
    form = _spectrum()
    perturbed, diagnostics = applySpectrumFactors(
        form, {0: (1.0, 3.0)}, {0: GRID}, _fixed())
    assert 3.0 * (1 + S) == pytest.approx(
        min(v for v in perturbed.outerDomainValues if v > 3.0), rel=1e-12)
    assert _moved(form, perturbed, 3.0) > 0.1
    for energy in (3.0 * (1 + S), 3.25, 3.5, 3.9):
        assert _moved(form, perturbed, energy) < 1e-12, energy


def test_the_factor_stops_below_a_gap_between_two_bands():
    """``[1, 2)`` and ``[4, 5]``: between 2 and 4 nothing is perturbed.

    The half-open band does not own its top node, so the step is just *below*
    it: a node at ``2 (1 - s)`` carries band 0's factor and the node at 2 none.
    """
    form = _spectrum()
    perturbed, _diagnostics = applySpectrumFactors(
        form, {0: (1.0, 2.0), 1: (4.0, 5.0)}, {0: GRID, 1: GRID}, _fixed())
    assert any(v == pytest.approx(2.0 * (1 - S), rel=1e-12)
               for v in perturbed.outerDomainValues)
    assert _moved(form, perturbed, 1.5) > 0.1
    for energy in (2.0, 2.5, 3.0, 3.5, 4.0 * (1 - S)):
        assert _moved(form, perturbed, energy) < 1e-12, energy


def test_contiguous_bands_get_no_new_outer_node():
    """The whole-band request on a tiling: exactly the nodes it got before.

    Every top edge is the next band's lower edge, so its shoulder is the one
    that band already inserts and the de-duplication drops it. This is why the
    JEFF-4.0 U-235 and B-VIII.1 Cf-252 tape gates stay byte-identical.
    """
    form = _spectrum()
    _perturbed, diagnostics = applySpectrumFactors(
        form, {0: (1.0, 3.0), 1: (3.0, 5.0)}, {0: GRID, 1: GRID}, _fixed())
    assert diagnostics["n_outer_inserted"] == 1        # 3 (1 - s), as before


@pytest.fixture(scope="module")
def u233_b81_perturbed(request):
    """ENDF/B-VIII.1 U-233, seed 11 sample 0, through the model applier."""
    tape = request.getfixturevalue("u233_b81_tape")
    endf = read_endf(str(tape), mf_numbers=[5, 35])
    suite, mf5, bands = build_pfns_covariance(endf, mt=18)
    grids = band_grids(suite)
    drawn, _info = generate_pfns_samples(suite, 1, seed=11, verbose=False)
    deltas = {key[-1]: value[0] for key, value in drawn.items()}
    form, _provenance, _report = decodeMF5MT(mf5)
    perturbed, diagnostics = applySpectrumFactors(
        form, dict(enumerate(bands)), dict(enumerate(grids)),
        pfns_ratio_rule(deltas))
    return form, perturbed, diagnostics, bands, grids


def test_the_factor_stops_at_the_top_of_a_partial_band(u233_b81_perturbed):
    """PF-2 on its real witness: one band ``[1e-5 eV, 5 MeV]``, MF5 to 20 MeV.

    Measured before the fix (seed 11): the 5 MeV node moves its group
    probabilities by up to 63 %, and the factor was still 63 % at 5.01 MeV,
    36 % at 5.5 MeV and 0.8 % at 5.99 MeV. After: 1.8e-12 at worst.
    """
    from kika.algebra import group_integrals

    form, perturbed, _diagnostics, bands, grids = u233_b81_perturbed
    assert bands == [(1e-5, 5e6)]

    def moved(energy):
        x0, y0 = form.evaluateAtOuter(energy)
        x1, y1 = perturbed.evaluateAtOuter(energy)
        p0 = group_integrals(x0, y0, 2, grids[0])
        p1 = group_integrals(x1, y1, 2, grids[0])
        live = p0 > 1e-6 * p0.sum()
        return float(np.max(np.abs(p1[live] / p0[live] - 1.0)))

    assert moved(5e6) > 0.1
    for fraction in (0.01, 0.1, 0.25, 0.5, 0.75, 0.99):
        assert moved(5e6 + fraction * 1e6) < 1e-11, fraction


def test_the_factor_stops_at_the_edge_of_the_mf35_grid(u233_b81_perturbed):
    """PF-3 on its real witness: grid 10 eV-30 MeV, tables 0-31 MeV.

    The quantity is the mass outside the grid, net of the renormalisation
    scalar -- the mass the factor should not touch. Measured before (seed 11):
    3.3e-10 of chi, 2.1 % of that mass; after, 6.0e-13 and 6.2e-5 of it.
    """
    form, perturbed, diagnostics, _bands, grids = u233_b81_perturbed
    lo, hi = grids[0][0], grids[0][-1]
    worst = 0.0
    values = list(perturbed.outerDomainValues)
    for entry in diagnostics["per_node"]:
        k = entry["node"]
        energy = entry["outer_value"]
        occurrence = values[:k].count(energy)
        same = [i for i, v in enumerate(form.outerDomainValues) if v == energy]
        if occurrence >= len(same):
            continue                                  # an inserted shoulder
        before = form.function1ds[same[occurrence]]
        after = perturbed.function1ds[k]
        outside = (before.integrate() - before.integrate(lo, hi))
        moved = (after.integrate() - after.integrate(lo, hi)) \
            / entry["renormalisation_scalar"] - outside
        worst = max(worst, abs(moved) / before.integrate())
    assert worst < 1e-11
    assert diagnostics["max_group_mass_error"] < 1e-7

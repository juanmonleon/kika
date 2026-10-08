"""PD-6: an MF5 table under INT=3, 4 or 5 is perturbed from MF35 exactly.

The one real MF35 tape with a log-interpolated spectrum is JEFF-3.1.1 U-233
(outgoing INT 2+4, six bands), and it is **not on this box**. So every test
here is on a **synthetic** spectrum: the committed micro Cf-252 tape, cut to
its first band's incident nodes and relabelled INT=2 below 1 MeV and INT=4
(log-lin) above -- the JEFF-3.1.1 U-233 layout -- then renormalised under the
new laws. The functions it states are therefore not Cf-252's; what matters is
that they are stated in log-lin and the applier has to respect that.

What was missing, and what each test pins:

1. **The integrals** -- closed form for all five laws since kika.algebra
   (7-oct-2026); gated here once more on a multi-region table against scipy
   quadrature, group by group, at 1e-12.
2. **Exact insertion.** A group boundary inside a log-lin panel was valued
   with ``np.interp``, a lin-lin number on a log-lin panel. It is now valued by
   the panel's own law, so away from the shoulders the perturbed spectrum *is*
   ``rescale * r_g * chi`` to rounding.
3. **Region-preserving ``replaceTable``.** A multi-region child was refused;
   it now keeps its regions, each new interval inheriting its panel's law.
4. **PF-4, pinned on the legacy applier:** ``perturb_pfns_partial`` writes
   every table back as one lin-lin region, so a zero perturbation of an INT 2+4
   spectrum changes it. Measured on this fixture below.
5. **What is still refused:** an *incident* node inserted between two
   log-interpolated children (every interior band edge needs one) has no exact
   table -- the blend of two log-lin functions is not log-lin -- and is refused
   by name. This is what still stops the six-band real witness; see the report.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from kika.algebra import evaluate, interval_laws, laws_on_refinement
from kika.endf import read_endf
from kika.endf.model_adapter.energy import decodeMF5MT, encodeMF5MT
from kika.nuclear_data.model.functions import Regions1d
from kika.nuclear_data.model.perturbation import (SPECTRUM_STEP_SHOULDER,
                                                  applySpectrumFactors)
from kika.sampling.mf35_sampling import (band_grids, build_pfns_covariance,
                                         generate_pfns_samples,
                                         perturb_pfns_partial, pfns_ratio_rule)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
SPLIT = 1.0e6          # INT=2 below, INT=4 above: the JEFF-3.1.1 U-233 layout


def _logLinSection(laws=(2, 4)):
    """``(mf5 section, band, grid, one sample's deltas)``, synthetic (see module)."""
    endf = read_endf(str(DATA / "micro_cf252_pfns.endf"), mf_numbers=[5, 35])
    suite, mf5, bands = build_pfns_covariance(endf, mt=18)
    grid = band_grids(suite)[0]
    band = bands[0]

    section = copy.deepcopy(mf5)
    partial = section.partials[0]
    keep = [k for k, e in enumerate(partial.incident_energies)
            if band[0] <= e <= band[1]]
    partial.incident_energies = [partial.incident_energies[k] for k in keep]
    partial.outgoing_grids = [partial.outgoing_grids[k] for k in keep]
    partial.chi = [partial.chi[k] for k in keep]
    partial.outgoing_interp = []
    partial.tab2_interp = [(len(keep), 2)]
    for k in range(len(keep)):
        x = np.asarray(partial.outgoing_grids[k])
        split = int(np.searchsorted(x, SPLIT)) + 1
        partial.outgoing_interp.append([(split, laws[0]), (x.size, laws[1])])
        norm = partial.normalisation(k)
        partial.chi[k] = list(np.asarray(partial.chi[k]) / norm)

    drawn, _info = generate_pfns_samples(suite, 1, seed=11, verbose=False)
    delta = {key[-1]: value[0] for key, value in drawn.items()}[0]
    return section, band, grid, delta


@pytest.fixture(scope="module")
def perturbed():
    section, band, grid, delta = _logLinSection()
    form, provenance, _report = decodeMF5MT(section)
    out, diagnostics = applySpectrumFactors(
        form, {0: band}, {0: grid}, pfns_ratio_rule({0: delta}))
    return section, form, provenance, out, diagnostics, grid, delta


def test_the_fixture_is_what_it_claims():
    section, _band, _grid, _delta = _logLinSection()
    form, _provenance, _report = decodeMF5MT(section)
    child = form.function1ds[0]
    assert isinstance(child, Regions1d)
    assert [r.endfInterpolationCode for r in child.function1ds] == [2, 4]
    assert form.normalisation(0) == pytest.approx(1.0, rel=1e-13)


@pytest.mark.parametrize("laws", [(2, 3), (2, 4), (2, 5)])
def test_group_integrals_under_each_log_law_are_the_quadrature(laws):
    """Closed form, group by group, against scipy (INT=3, 4, 5).

    The bar is 1e-12 relative **or 1e-14 of the spectrum**, whichever is
    looser. The second is what limits it, and it is not the laws:
    :func:`kika.algebra.group_integrals` differences a cumulative integral, so
    a group holding 3.6e-5 of chi carries the ~1e-16 rounding of a sum near 1,
    which is 1.8e-12 of that group (measured against a 40-digit evaluation of
    the same log-lin panels; scipy agrees with the 40-digit value). An MF35
    matrix is an *absolute* covariance of these numbers, so the absolute bar
    is the one that matters.
    """
    section, _band, grid, _delta = _logLinSection(laws)
    form, _provenance, _report = decodeMF5MT(section)
    xs, ys = form.table(0)
    codes = form.tableLaws(0)
    exact = form.groupIntegrals(0, grid)
    lo, hi = xs[1], xs[-2]        # a log-x law needs x > 0: skip E'=0
    for j in range(len(grid) - 1):
        a, b = max(grid[j], lo), min(grid[j + 1], hi)
        if b <= a or grid[j] < lo or grid[j + 1] > hi:
            continue
        cuts = np.unique(np.concatenate([[a, b], xs[(xs > a) & (xs < b)]]))
        want = sum(quad(lambda t: float(evaluate(xs, ys, codes, t)), u, v,
                        epsabs=0, epsrel=1e-13, limit=200)[0]
                   for u, v in zip(cuts[:-1], cuts[1:]))
        assert exact[j] == pytest.approx(want, rel=1e-12, abs=1e-14), j


def test_a_refinement_inherits_each_panels_law():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    laws = np.array([2, 4, 5])
    assert list(laws_on_refinement(x, laws, [1.0, 1.5, 2.0, 3.0, 4.0, 8.0])) \
        == [2, 2, 4, 4, 5]
    with pytest.raises(ValueError, match="crosses a node"):
        laws_on_refinement(x, laws, [1.0, 3.0, 8.0])


def test_the_perturbed_spectrum_is_the_factor_times_chi_between_steps(perturbed):
    """Away from the shoulders, chi' = rescale * r_g * chi, to rounding.

    The discrimination, stated: a point inserted in a log-lin panel by
    ``np.interp`` -- what the applier did before -- is off the log-lin curve by
    up to the relative amount asserted at the end, and the identity above
    would fail by that much in every group whose edge it was.
    """
    _section, form, _provenance, out, diagnostics, grid, delta = perturbed
    assert diagnostics["total_outgoing_inserted"] > 0
    worstInterp = 0.0
    for entry in diagnostics["per_node"]:
        k = entry["node"]
        ratios = pfns_ratio_rule({0: delta})(
            0, form.groupIntegrals(k, grid), entry["covered_before"])[0]
        xs, ys = form.table(k)
        codes = form.tableLaws(k)
        xNew, yNew = out.table(k)
        codesNew = out.tableLaws(k)
        for j in range(len(grid) - 1):
            a, b = grid[j], grid[j + 1]
            # Stay off the shoulder below the group's top edge.
            probe = np.linspace(a, b - 2 * SPECTRUM_STEP_SHOULDER * (b - a), 7)[1:-1]
            probe = probe[(probe > xs[0]) & (probe < xs[-1])]
            if probe.size == 0:
                continue
            want = entry["renormalisation_scalar"] * ratios[j] * evaluate(
                xs, ys, codes, probe)
            got = evaluate(xNew, yNew, codesNew, probe)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300)
        inserted = np.setdiff1d(xNew, xs)
        inside = inserted[(inserted > SPLIT) & (inserted < xs[-2])]
        if inside.size:
            law = evaluate(xs, ys, codes, inside)
            linear = np.interp(inside, xs, ys)
            live = law > 0
            worstInterp = max(worstInterp, float(np.max(
                np.abs(linear[live] / law[live] - 1.0))))
    assert worstInterp > 1e-6, "the fixture does not discriminate np.interp"


def test_the_regions_survive_and_reach_the_endf_section(perturbed):
    """Every perturbed table keeps INT=2 below 1 MeV and INT=4 above it."""
    _section, _form, provenance, out, diagnostics, _grid, _delta = perturbed
    for entry in diagnostics["per_node"]:
        xs, _ys = out.table(entry["node"])
        codes = out.tableLaws(entry["node"])
        wide = np.diff(xs) > 0
        middle = 0.5 * (xs[:-1] + xs[1:])
        assert np.all(codes[wide & (middle < SPLIT)] == 2)
        assert np.all(codes[wide & (middle > SPLIT)] == 4)

    encoded, _report = encodeMF5MT(out, provenance, 18)
    written = encoded.partials[0]
    for k in range(len(written.incident_energies)):
        assert [code for _nbt, code in written.outgoing_interp[k]] == [2, 4]
        x = np.asarray(written.outgoing_grids[k])
        assert written.outgoing_interp[k][0][0] == int(np.searchsorted(x, SPLIT)) + 1


def test_a_zero_perturbation_gives_the_log_lin_spectrum_back(perturbed):
    section, form, _provenance, _out, _diagnostics, grid, _delta = perturbed
    band = (section.partials[0].incident_energies[0],
            section.partials[0].incident_energies[-1])
    same, diagnostics = applySpectrumFactors(
        form, {0: band}, {0: grid},
        pfns_ratio_rule({0: np.zeros(len(grid) - 1)}))
    assert diagnostics["max_renormalisation_error"] < 1e-12
    for k in range(len(form.outerDomainValues)):
        # Points may be inserted (an empty top group freezes at a ratio of its
        # own), but every one sits on the log-lin curve: same function.
        xs, ys = form.table(k)
        codes = form.tableLaws(k)
        xNew, yNew = same.table(k)
        assert np.all(np.isin(xs, xNew))
        probe = np.union1d(xNew, 0.5 * (xNew[:-1] + xNew[1:]))
        np.testing.assert_allclose(
            evaluate(xNew, yNew, same.tableLaws(k), probe),
            evaluate(xs, ys, codes, probe), rtol=1e-13, atol=1e-300)


def test_the_legacy_applier_relabels_a_log_lin_spectrum_as_lin_lin():
    """PF-4, pinned: legacy, given delta = 0, still changes the spectrum.

    ``perturb_pfns_partial`` replaces every table with ``[(n, 2)]``. On this
    fixture that relabels the log-lin region above 1 MeV as lin-lin; the
    renormalisation then hides the change in the integral, and what is left is
    a different function at every point between nodes of that region.
    Measured here: relabelling moves the integral by up to 1.3e-4, and the
    written chi departs from the log-lin one between nodes by up to 4.3e-4. Not fixed, on purpose: the
    legacy applier is the fixed reference (PD-4) and the model applier is the
    one that runs on such a tape.
    """
    section, band, grid, _delta = _logLinSection()
    partial = copy.deepcopy(section.partials[0])
    before = copy.deepcopy(partial)
    out, _diagnostics = perturb_pfns_partial(
        partial, {0: np.zeros(len(grid) - 1)}, [band], [grid])
    k = 0
    assert before.outgoing_interp[k][-1][1] == 4
    assert out.outgoing_interp[k] == [(len(out.outgoing_grids[k]), 2)]

    relabelled = copy.deepcopy(before)
    relabelled.outgoing_interp[k] = [(len(before.outgoing_grids[k]), 2)]
    assert abs(relabelled.normalisation(k) - 1.0) > 1e-4

    x, y = before.table(k)
    probe = 0.5 * (x[:-1] + x[1:])
    probe = probe[(probe > SPLIT) & (probe < x[-2])]
    lawful = evaluate(x, y, interval_laws(x.size, before.outgoing_interp[k]), probe)
    xo, yo = out.table(k)
    written = np.interp(probe, xo, yo)
    live = lawful > 1e-12 * lawful.max()
    assert np.max(np.abs(written[live] / lawful[live] - 1.0)) > 1e-4


def test_an_incident_node_between_log_tables_is_refused_by_name():
    """The step at an interior band edge needs a node there; this cannot be one.

    Between two log-lin children the interpolant at a new incident energy is a
    blend of two log-lin functions, which no single law states. So a request
    whose bands split the incident range is refused rather than linearised in
    silence -- the remaining blocker for JEFF-3.1.1 U-233 (six bands).
    """
    section, band, grid, delta = _logLinSection()
    form, _provenance, _report = decodeMF5MT(section)
    nodes = form.outerDomainValues
    middle = nodes[len(nodes) // 2]
    with pytest.raises(NotImplementedError, match="lin-lin"):
        applySpectrumFactors(
            form, {0: (nodes[0], middle), 1: (middle, nodes[-1])},
            {0: grid, 1: grid}, pfns_ratio_rule({0: delta, 1: delta}))

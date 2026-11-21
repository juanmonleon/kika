"""Spectrum adjustment against closed forms, the linear kernel and a closure test on its own model."""
import numpy as np
import pytest

from kika.algebra import linear_update, product_group_integrals
from kika.dosimetry import (BARN_CM2, AdjustmentNotConverged, Dosimeter, SpectrumPrior,
                            adjust_spectrum, response_cells)
import kika.dosimetry.adjust as adjust_module

EDGES = np.geomspace(1e5, 2e7, 31)
MID = np.sqrt(EDGES[:-1] * EDGES[1:])
FINE = np.geomspace(1e5, 2e7, 400)
THRESHOLDS = (6e5, 1.2e6, 2.8e6, 3.5e6, 4.5e6, 7e6)


def _flux():
    e = MID / 1e6
    return 1e10 * np.sqrt(e) * np.exp(-e / 1.3) * np.diff(EDGES) / 1e6


def _prior_covariance(sd=0.2, width=1.0, normalisation=0.05):
    ell = np.log(MID)
    corr = np.exp(-0.5 * ((ell[:, None] - ell[None, :]) / width) ** 2)
    return sd**2 * corr + normalisation**2


def _root(cov):
    w, q = np.linalg.eigh(cov)
    return q * np.sqrt(np.clip(w, 0.0, None))


def _dosimeter(threshold, sd=0.04, name=None):
    sigma = 0.5 / (1 + np.exp(-(FINE - threshold) / (0.08 * threshold + 1e5)))
    cov_edges = np.geomspace(1e5, 2e7, 9)
    ell = np.log(np.sqrt(cov_edges[:-1] * cov_edges[1:]))
    cov = sd**2 * np.exp(-0.5 * ((ell[:, None] - ell[None, :]) / 1.5) ** 2)
    return Dosimeter(name or f"thr{threshold:.2e}", FINE, sigma, 2, cov_edges, cov)


def test_cells_add_up_to_the_rate_by_direct_integration():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    d = _dosimeter(2.8e6)
    cells = response_cells(prior, d)
    flat = (EDGES, np.r_[_flux() / np.diff(EDGES), 0.0], 1)
    direct = BARN_CM2 * product_group_integrals((d.energies, d.cross_section, 2), flat, EDGES[[0, -1]])[0]
    assert cells.shape == (30, 9)
    assert cells.sum() == pytest.approx(direct, rel=1e-12)
    assert cells[:, -1].sum() == pytest.approx(0.0, abs=1e-30 * direct)


def test_one_group_one_dosimeter_is_the_closed_form():
    prior = SpectrumPrior([1e6, 2e6], [3e9], [[0.1**2]])
    d = Dosimeter("flat", [1e6, 2e6], [0.2, 0.2], 2, [1e6, 2e6], [[0.03**2]])
    calculated = response_cells(prior, d).sum()
    result = adjust_spectrum(prior, [d], [1.15 * calculated], [[0.04**2]])
    c, s, v = 0.1**2, 0.03**2, 0.04**2
    assert np.log(result.flux[0] / 3e9) == pytest.approx(c / (c + s + v) * np.log(1.15), rel=1e-12)
    assert result.log_covariance[0, 0] == pytest.approx(c - c**2 / (c + s + v), rel=1e-12)
    assert result.chi2 == pytest.approx(np.log(1.15) ** 2 / (c + s + v), rel=1e-12)
    assert result.iterations <= 2


def test_the_first_iteration_is_the_linear_one_step_update():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    cells = [response_cells(prior, d) for d in dosimeters]
    rates = np.array([c.sum() for c in cells])
    measured = rates * np.exp([0.1, -0.05, 0.2, 0.15, -0.1, 0.05])
    V = np.diag(np.full(6, 0.04**2))
    result = adjust_spectrum(prior, dosimeters, measured, V)

    G = np.vstack([c.sum(axis=1) for c in cells]) / rates[:, None]
    folded = np.diag([(c.sum(axis=0) / r) @ d.cell_covariance @ (c.sum(axis=0) / r)
                      for c, r, d in zip(cells, rates, dosimeters)])
    one = linear_update(np.zeros(30), prior.relative_covariance, G, np.log(measured / rates), V + folded)
    np.testing.assert_allclose(result.one_step_flux, prior.flux * np.exp(one.mean), rtol=1e-12)
    assert result.iterations > 1
    assert np.max(np.abs(np.log(result.flux / result.one_step_flux))) < 0.02


def test_the_posterior_covers_the_truth_in_a_closure_test():
    rng = np.random.default_rng(11)
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    V = np.diag(np.full(6, 0.04**2))
    L = _root(prior.relative_covariance)
    above = EDGES[:-1] >= 1e6
    inside, chi2 = [], []
    for _ in range(300):
        truth = prior.flux * np.exp(L @ rng.normal(size=30))
        true_prior = SpectrumPrior(EDGES, truth, prior.relative_covariance)
        rates = []
        for d in dosimeters:
            k = d.relative_covariance.shape[0]
            factor = np.exp(_root(d.relative_covariance) @ rng.normal(size=k))
            cells = response_cells(true_prior, d)
            rates.append((cells[:, :k] * factor).sum() + cells[:, k].sum())
        measured = np.array(rates) * np.exp(rng.normal(scale=0.04, size=6))
        result = adjust_spectrum(prior, dosimeters, measured, V)
        value, sd = result.flux_above(EDGES[np.argmax(above)])
        inside.append(abs(np.log(truth[above].sum() / value)) < sd / value)
        chi2.append(result.chi2)
    assert 0.60 < np.mean(inside) < 0.76
    assert np.mean(chi2) / 6 == pytest.approx(1, abs=0.15)


def test_flux_above_needs_a_group_boundary_and_integral_needs_one_weight_per_group():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    d = _dosimeter(2.8e6)
    result = adjust_spectrum(prior, [d], [response_cells(prior, d).sum()], [[0.04**2]])
    with pytest.raises(ValueError):
        result.flux_above(1.234e6)
    with pytest.raises(ValueError):
        result.integral(np.ones(3))
    value, sd = result.flux_above(EDGES[0])
    assert value == pytest.approx(prior.flux.sum(), rel=1e-12) and sd > 0


def test_a_dosimeter_blind_to_the_spectrum_is_refused():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    blind = Dosimeter("below", [1.0, 10.0], [5.0, 5.0], 2)
    with pytest.raises(ValueError, match="no response"):
        adjust_spectrum(prior, [blind], [1.0], [[0.01]])


@pytest.mark.parametrize("measured, cov", [([1.0, 2.0], [[0.01]]), ([-1.0], [[0.01]]), ([1.0], np.eye(2))])
def test_malformed_measurements_are_refused(measured, cov):
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    with pytest.raises(ValueError):
        adjust_spectrum(prior, [_dosimeter(2.8e6)], measured, cov)


def test_a_group_of_zero_prior_flux_stays_zero_and_carries_no_parameter():
    flux = _flux()
    flux[25:] = 0.0
    prior = SpectrumPrior(EDGES, flux, _prior_covariance())
    d = _dosimeter(2.8e6)
    result = adjust_spectrum(prior, [d], [1.2 * response_cells(prior, d).sum()], [[0.04**2]])
    assert np.all(result.flux[25:] == 0.0) and np.all(result.log_covariance[25:] == 0.0)
    assert result.flux[:25].sum() > flux.sum()
    with pytest.raises(ValueError, match="zero in every group"):
        SpectrumPrior(EDGES, np.zeros(30), _prior_covariance())


def test_no_convergence_raises(monkeypatch):
    monkeypatch.setattr(adjust_module, "MAX_ITERATIONS", 1)
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    rates = np.array([response_cells(prior, d).sum() for d in dosimeters])
    with pytest.raises(AdjustmentNotConverged):
        adjust_spectrum(prior, dosimeters, rates * 1.3, np.diag(np.full(6, 0.04**2)))


def test_two_measurements_of_one_reaction_share_its_parameters():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    d = _dosimeter(2.8e6, name="Ni58p")
    rate = response_cells(prior, d).sum()
    v = 0.04**2
    twice = adjust_spectrum(prior, [d, d], [1.10 * rate, 1.20 * rate], np.diag([v, v]))
    once = adjust_spectrum(prior, [d], [np.sqrt(1.10 * 1.20) * rate], [[v / 2]])
    assert list(twice.cross_section_factors) == ["Ni58p"]
    np.testing.assert_allclose(twice.flux, once.flux, rtol=1e-10)
    np.testing.assert_allclose(twice.cross_section_factors["Ni58p"], once.cross_section_factors["Ni58p"], rtol=1e-10)
    np.testing.assert_allclose(twice.log_covariance, once.log_covariance, rtol=1e-8, atol=1e-14)


def test_two_reactions_of_one_name_with_different_data_are_refused():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    with pytest.raises(ValueError, match="different data"):
        adjust_spectrum(prior, [_dosimeter(2.8e6, name="X"), _dosimeter(4.5e6, name="X")],
                        [1.0, 1.0], np.eye(2) * 0.01)


def test_the_posterior_correlates_flux_and_cross_sections():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    rates = np.array([response_cells(prior, d).sum() for d in dosimeters])
    result = adjust_spectrum(prior, dosimeters, rates * 1.1, np.diag(np.full(6, 0.04**2)))
    name = dosimeters[2].name
    assert result.cross_section_factors[name].shape == (8,)
    assert result.flux_cross_section_log_covariance[name].shape == (30, 8)
    assert np.min(result.flux_cross_section_log_covariance[name]) < 0
    prior_sd = np.sqrt(np.diag(dosimeters[2].relative_covariance))
    assert np.all(np.sqrt(np.diag(result.cross_section_log_covariance[name])) <= prior_sd + 1e-15)


def test_the_report_is_one_html_page_with_every_dosimeter(tmp_path):
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    rates = np.array([response_cells(prior, d).sum() for d in dosimeters])
    good = adjust_spectrum(prior, dosimeters, rates * np.array([1.1, 1.05, 1.12, 1.08, 0.98, 1.15]),
                          np.diag(np.full(6, 0.04**2)))
    assert 0.3 <= good.chi2 / good.dof <= 2.0
    page = good.to_html()
    assert page.startswith("<!doctype html>") and page.count("<svg") == 2
    assert all(name in page for name in good.names)
    assert 'class="verdict pass"' in page and "Above 0.1 MeV" in page
    bad = adjust_spectrum(prior, dosimeters, rates * np.array([1, 1, 2.5, 1, 1, 1]), np.diag(np.full(6, 0.01**2)))
    assert 'class="verdict warn"' in bad.to_html()
    path = tmp_path / "adjustment.html"
    good.write_html(path)
    assert path.read_text(encoding="utf-8") == page


def test_negative_eigenvalues_within_round_off_are_a_note_in_the_report():
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    cov = q @ np.diag([0.04, 0.02, -1.5e-17]) @ q.T
    cov = 0.5 * (cov + cov.T)
    edges = np.array([1e6, 2e6, 4e6, 8e6])
    prior = SpectrumPrior(edges, [1e9, 8e8, 3e8], cov)
    d = Dosimeter("flat", [1e6, 8e6], [0.3, 0.3], 2)
    result = adjust_spectrum(prior, [d], [1.1 * response_cells(prior, d).sum()], [[0.03**2]])
    rounded = [note for note in result.notes if "round-off" in note]
    assert len(rounded) == 1 and "the prior spectrum" in rounded[0]
    assert 'id="ka-notes"' in result.to_html()


def test_a_reaction_without_covariance_is_named_in_the_notes():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    exact = Dosimeter("exact", FINE, 0.5 / (1 + np.exp(-(FINE - 3e6) / 3e5)), 2)
    result = adjust_spectrum(prior, [exact], [1.1 * response_cells(prior, exact).sum()], [[0.04**2]])
    assert any("No covariance for exact" in note for note in result.notes)


def test_a_free_normalization_leaves_the_level_to_the_data():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    dosimeters = [_dosimeter(t) for t in THRESHOLDS]
    rates = np.array([response_cells(prior, d).sum() for d in dosimeters])
    measured = rates * np.exp([0.1, -0.05, 0.2, 0.15, -0.1, 0.05])
    V = np.diag(np.full(6, 0.04**2))
    free = adjust_spectrum(prior, dosimeters, measured, V, free_normalization=True)
    far = SpectrumPrior(EDGES, 1e7 * _flux(), _prior_covariance())
    free_far = adjust_spectrum(far, dosimeters, measured, V, free_normalization=True)
    scaled = adjust_spectrum(prior, dosimeters, 1.3 * measured, V, free_normalization=True)

    assert free.dof == 5 and free_far.normalization == pytest.approx(1e-7 * free.normalization, rel=1e-9)
    np.testing.assert_allclose(free_far.flux, free.flux, rtol=1e-8)
    np.testing.assert_allclose(free_far.log_covariance, free.log_covariance, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(scaled.flux, 1.3 * free.flux, rtol=1e-6)
    assert scaled.chi2 == pytest.approx(free.chi2, rel=1e-6)
    np.testing.assert_allclose(free.prior_rates, free.normalization * rates, rtol=1e-12)
    assert any("Free normalization" in note for note in free.notes)


def test_a_free_normalization_needs_two_dosimeters():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    with pytest.raises(ValueError, match="two dosimeters"):
        adjust_spectrum(prior, [_dosimeter(2.8e6)], [1.0], [[0.01]], free_normalization=True)


def test_a_covered_foil_shares_the_bare_reactions_parameters():
    from kika.dosimetry import covered

    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    bare = _dosimeter(2.8e6, name="Ni58p")
    half = covered(bare, ([1e5, 2e7], [0.5, 0.5]), name="Ni58p Cd")
    assert half.reaction == "Ni58p"
    rate = response_cells(prior, bare).sum()
    assert response_cells(prior, half).sum() == pytest.approx(0.5 * rate, rel=1e-6)
    v = 0.04**2
    both = adjust_spectrum(prior, [bare, half], [1.10 * rate, 0.5 * 1.20 * rate], np.diag([v, v]))
    once = adjust_spectrum(prior, [bare], [np.sqrt(1.10 * 1.20) * rate], [[v / 2]])
    assert list(both.cross_section_factors) == ["Ni58p"]
    np.testing.assert_allclose(both.flux, once.flux, rtol=1e-5)


def test_dosimeters_of_one_reaction_with_different_covariances_are_refused():
    prior = SpectrumPrior(EDGES, _flux(), _prior_covariance())
    a = _dosimeter(2.8e6, name="A")
    b = Dosimeter("B", a.energies, a.cross_section, 2, a.covariance_edges, 2 * a.relative_covariance, reaction="A")
    with pytest.raises(ValueError, match="different covariances"):
        adjust_spectrum(prior, [a, b], [1.0, 1.0], np.eye(2) * 0.01)


def test_a_free_normalization_of_one_group_is_the_weighted_mean_in_closed_form():
    prior = SpectrumPrior([1e6, 2e6], [3e9], [[0.2**2]])
    a = Dosimeter("a", [1e6, 2e6], [0.2, 0.2], 2)
    b = Dosimeter("b", [1e6, 2e6], [0.5, 0.5], 2)
    rates = np.array([response_cells(prior, d).sum() for d in (a, b)])
    r = np.log([1.30, 1.10])
    v = np.array([0.03**2, 0.05**2])
    result = adjust_spectrum(prior, [a, b], rates * np.exp(r), np.diag(v), free_normalization=True)

    w = (1 / v) / np.sum(1 / v)
    mean = w @ r
    assert result.dof == 1
    assert result.normalization == pytest.approx(np.exp(mean), rel=1e-12)
    assert result.chi2 == pytest.approx((r[0] - r[1]) ** 2 / v.sum(), rel=1e-4)
    assert result.flux[0] == pytest.approx(3e9 * np.exp(mean), rel=1e-4)
    left = v - 1 / np.sum(1 / v)
    np.testing.assert_allclose(result.standardized_residuals, (r - mean) / np.sqrt(left), rtol=1e-9)
    assert result.log_covariance[0, 0] == pytest.approx(1 / np.sum(1 / v), rel=1e-4)

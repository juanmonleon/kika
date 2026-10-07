"""MF35 read against MF5: ``sqrt(C_ii) / P_i`` at any incident energy.

The band matrix covaries group probabilities (see ``test_mf35_roundtrip``), so
a relative uncertainty needs ``P_i`` from MF5, and ``P_i`` depends on where in
the band the incident energy sits. Comparing libraries is what makes that
energy the selector: their bands differ (five on ENDF/B-VIII.1 U-235, eight on
JEFF-4.0), so "band 2" is a different thing on each tape and "1 MeV" is not.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf
from kika.plotting import NotPlottable, plottable


def _pfns(path):
    endf = read_endf(str(path), mf_numbers=[1, 5, 35])
    return endf, endf.files[35].mt[18], endf.files[5].mt[18]


def test_group_integrals_at_a_node_is_group_integrals(micro_pfns_tape):
    _endf, mf35, mf5 = _pfns(micro_pfns_tape)
    partial = mf5.tabulated_partials()[0][1]
    band = mf35.subsections[0]
    k = list(partial.incident_energies).index(band.e1)
    expected = partial.group_integrals(k, band.energy_grid())
    np.testing.assert_array_equal(
        partial.group_integrals_at(band.e1, band.energy_grid()), expected)


def test_group_integrals_between_nodes_blend_and_stay_normalised(micro_pfns_tape):
    """Between nodes the blend is exact, so the groups still sum to the
    interpolated normalisation, which is computed by an independent route."""
    _endf, mf35, mf5 = _pfns(micro_pfns_tape)
    partial = mf5.tabulated_partials()[0][1]
    energies = np.asarray(partial.incident_energies)
    energy = float(np.sqrt(energies[1] * energies[2]))
    edges = np.concatenate([[0.0], mf35.subsections[0].energy_grid()[1:-1], [1e9]])
    total = partial.group_integrals_at(energy, edges).sum()
    assert total == pytest.approx(partial.normalisation_at_incident(energy), rel=1e-12)

    lo = partial.group_integrals_at(energies[1], edges)
    hi = partial.group_integrals_at(energies[2], edges)
    mid = partial.group_integrals_at(energy, edges)
    assert np.all(mid >= np.minimum(lo, hi) - 1e-15)
    assert np.all(mid <= np.maximum(lo, hi) + 1e-15)


def test_relative_uncertainty_selects_the_band_by_energy(micro_pfns_tape):
    _endf, mf35, mf5 = _pfns(micro_pfns_tape)
    for index, band in enumerate(mf35.subsections):
        energy = float(np.sqrt(max(band.e1, 1e-5) * band.e2))
        result = mf35.relative_uncertainty(mf5, incident_energy=energy)
        assert result.band_index == index
        assert result.exact
        assert result.incident_energy == energy
        assert result.relative is not None
        assert result.relative.shape == (band.order,)
        np.testing.assert_allclose(
            result.relative, result.sigma / result.probabilities, equal_nan=True)


def test_relative_uncertainty_defaults_to_the_band_lower_edge(micro_pfns_tape):
    _endf, mf35, mf5 = _pfns(micro_pfns_tape)
    result = mf35.relative_uncertainty(mf5, band_index=1)
    assert result.incident_energy == mf35.subsections[1].e1
    assert result.incident_energy in result.incident_nodes


def test_relative_uncertainty_without_mf5_is_absolute_only(micro_pfns_tape):
    _endf, mf35, _mf5 = _pfns(micro_pfns_tape)
    result = mf35.relative_uncertainty(None, band_index=0)
    assert result.relative is None
    assert result.sigma.size == mf35.subsections[0].order
    assert any("no MF5" in w for w in result.warnings)


def test_relative_uncertainty_outside_every_band_says_so(micro_pfns_tape):
    _endf, mf35, mf5 = _pfns(micro_pfns_tape)
    with pytest.raises(ValueError, match="outside every MF35/MT18 band"):
        mf35.relative_uncertainty(mf5, incident_energy=1e12)


def test_plottable_spectrum_relative_uncertainty(micro_pfns_tape):
    endf, mf35, _mf5 = _pfns(micro_pfns_tape)
    band = mf35.subsections[-1]
    energy = float(np.sqrt(band.e1 * band.e2))
    item = plottable(endf, "spectrum_relative_uncertainty", incident_energy=energy)
    assert item.data.plot_type == "step"
    np.testing.assert_array_equal(item.data.x, band.energy_grid())
    assert item.data.y.size == item.data.x.size
    assert item.data.y_unit == "%"
    assert "band" in (item.data.provenance.detail or "")


def test_plottable_spectrum_relative_uncertainty_needs_mf35(micro_pfns_tape):
    endf = read_endf(str(micro_pfns_tape), mf_numbers=[1, 5])
    with pytest.raises(NotPlottable, match="MF35/MT18"):
        plottable(endf, "spectrum_relative_uncertainty", incident_energy=1e6)


@pytest.mark.parametrize("tape_fixture,band_index,order", [
    ("u235_b81_tape", 1, 640),
    ("u235_tape", 3, 64),
])
def test_one_mev_lands_in_each_librarys_own_band(request, tape_fixture, band_index, order):
    """The same energy, different bands and grids: the point of selecting by E."""
    path = request.getfixturevalue(tape_fixture)
    _endf, mf35, mf5 = _pfns(path)
    result = mf35.relative_uncertainty(mf5, incident_energy=1.0e6)
    assert result.band_index == band_index
    assert result.relative.size == order
    assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    # A few per cent near the peak of the spectrum, on both evaluations.
    peak = np.searchsorted(result.boundaries, 1.5e6) - 1
    assert 0.005 < result.relative[peak] < 0.05


def test_a_zero_first_boundary_is_drawable_on_a_log_scale():
    # ENDF/B-VIII U-235 bands 1-4 start at E' = 0; drawn at 1e-300 the first
    # group filled the whole log axis and the heatmap came out one colour.
    from kika.endf.classes.mf35 import MF35SubSection

    grid = [0.0, 10.0, 1e3, 1e6]
    tri = [1.0, 0.1, 0.0, 1.0, 0.2, 1.0]
    band = MF35SubSection(e1=1.0, e2=2.0, ls=1, lb=7, nt=len(grid) + len(tri), ne=len(grid),
                          boundaries=grid, upper_triangle=tri)
    assert band.to_heatmap_data(scale="log").energy_grid[0] == 1.0
    assert band.to_heatmap_data(scale="linear").energy_grid[0] == 0.0
    assert band.boundaries[0] == 0.0  # the band itself is untouched


def test_a_band_heatmap_has_the_mf33_layout_and_crops(micro_pfns_tape):
    # The builder only draws energy ticks and honours limits for data laid out
    # as MF33's is: transformed edges plus a one-block block_info.
    section = read_endf(str(micro_pfns_tape)).files[35].mt[18]
    band = section.subsections[0]
    grid = band.energy_grid()
    full = band.to_heatmap_data(scale="log", mt=18)
    assert full.block_info["mts"] == [18]
    assert full.block_info["G"] == band.order
    assert len(full.x_edges) == len(grid)
    assert full.extent[1] == pytest.approx(full.x_edges[-1])

    lo, hi = grid[len(grid) // 4], grid[3 * len(grid) // 4]
    pct = np.arange(band.order, dtype=float)
    cropped = band.to_heatmap_data(scale="log", mt=18, relative_pct=pct, energy_range=(lo, hi))
    n = cropped.block_info["G"]
    assert 0 < n < band.order
    assert cropped.matrix_data.shape == (n, n)
    assert len(cropped.uncertainty_data[18]) == n
    with pytest.raises(ValueError):
        band.to_heatmap_data(energy_range=(grid[-1] * 2, grid[-1] * 3))


def test_the_incident_grid_spans_every_band(micro_pfns_tape):
    endf = read_endf(str(micro_pfns_tape))
    section = endf.files[35].mt[18]
    result = section.relative_uncertainty(endf.files[5].mt[18], band_index=0)
    edges = [e for b in section.subsections for e in (b.e1, b.e2)]
    assert result.bands == [(b.e1, b.e2) for b in section.subsections]
    assert result.incident_grid[0] == min(edges) and result.incident_grid[-1] == max(edges)
    assert set(result.incident_nodes) <= set(result.incident_grid)
    assert result.incident_grid == sorted(set(result.incident_grid))

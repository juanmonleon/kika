"""The dpa fold against closed forms."""
import numpy as np
import pytest

from kika.processing.heating import dpa


def test_a_pointwise_fold_is_exact_on_the_union_of_two_grids():
    # sigma = E on [0, 4], phi = E on [1, 3]: int_1^3 E^2 dE = 26/3, though no grid has both.
    sigma = ([0.0, 4.0], [0.0, 4.0])
    phi = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert dpa(sigma, phi) == pytest.approx(26 / 3 * 1e-24, rel=1e-14)


def test_a_step_in_the_cross_section_stays_a_step():
    sigma = ([1.0, 2.0, 2.0, 3.0], [1.0, 1.0, 5.0, 5.0])
    phi = ([1.0, 3.0], [1.0, 1.0])
    assert dpa(sigma, phi) == pytest.approx(6e-24, rel=1e-14)


@pytest.mark.parametrize("weight, means", [("energy", [5.5, 55.0]),
                                           ("lethargy", [9 / np.log(10), 90 / np.log(10)])])
def test_a_group_flux_takes_the_group_mean_under_its_weight(weight, means):
    sigma = ([1.0, 100.0], [1.0, 100.0])
    edges = [1.0, 10.0, 100.0]
    assert dpa(sigma, [2.0, 3.0], edges, weight=weight) == pytest.approx(
        (2 * means[0] + 3 * means[1]) * 1e-24, rel=1e-13)
    cells = dpa(sigma, [[2.0, 3.0], [1.0, 0.0]], edges, weight=weight)
    assert cells.shape == (2,) and cells[1] == pytest.approx(means[0] * 1e-24, rel=1e-13)


def test_a_flat_group_flux_is_the_flat_pointwise_one():
    x = np.geomspace(1e-5, 2e7, 400)
    sigma = (x, 3.0 + np.sqrt(x))
    edges = np.geomspace(1e-5, 2e7, 30)
    width = np.diff(edges)
    pointwise = (np.repeat(edges, 2)[1:-1], np.repeat(1.0 / width, 2))   # one n/cm^2 per group, flat
    assert dpa(sigma, np.ones(29), edges, weight="energy") == pytest.approx(dpa(sigma, pointwise), rel=1e-12)


def test_flux_outside_the_table_is_refused_unless_it_is_zero():
    sigma = ([1e-5, 2e7], [1.0, 1.0])
    with pytest.raises(ValueError, match="outside"):
        dpa(sigma, ([1e-6, 1.0], [1.0, 1.0]))
    with pytest.raises(ValueError, match=r"groups \[0\]"):
        dpa(sigma, [1.0, 1.0], [0.0, 1.0, 2e7], weight="lethargy")
    # MCNP's first group starts at 0: fine when it carries nothing, even under 1/E.
    assert dpa(sigma, [0.0, 2.0], [0.0, 1.0, 2e7], weight="lethargy") == pytest.approx(2e-24)
    assert dpa(sigma, ([1e-6, 1.0, 2.0], [0.0, 0.0, 1.0])) == pytest.approx(0.5e-24)


def test_the_group_weight_is_required_and_only_for_groups():
    sigma = ([1.0, 10.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="weight"):
        dpa(sigma, [1.0], [1.0, 10.0])
    with pytest.raises(ValueError, match="weight"):
        dpa(sigma, ([1.0, 10.0], [1.0, 1.0]), weight="energy")
    with pytest.raises(ValueError, match="one value per group"):
        dpa(sigma, [1.0, 2.0], [1.0, 10.0], weight="energy")
    with pytest.raises(ValueError, match="increasing"):
        dpa(sigma, [1.0, 2.0], [1.0, 1.0, 10.0], weight="energy")

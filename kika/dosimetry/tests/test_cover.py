"""Cover transmission: E2 of the cover's optical thickness, and covered responses."""
import numpy as np
import pytest
from scipy.special import expn

from kika.dosimetry import Dosimeter, cover_transmission, covered


def test_a_constant_absorption_transmits_e2_of_the_optical_thickness():
    x, t = cover_transmission([1e-5, 1e7], [2500.0, 2500.0], 2, 1e-3)
    np.testing.assert_allclose(t, expn(2, 2.5), rtol=1e-12)


def test_a_1_over_v_absorber_is_refined_to_the_tolerance():
    e = np.geomspace(1e-5, 1e3, 9)
    sigma = 2500.0 * np.sqrt(0.0253 / e)
    x, t = cover_transmission(e, sigma, 5, 1e-3, tol=1e-4)
    q = np.geomspace(1e-5, 1e3, 2000)
    exact = expn(2, 1e-3 * 2500.0 * np.sqrt(0.0253 / q))
    floor = 1e-9 * exact.max()
    assert np.all(np.abs(np.interp(q, x, t) - exact) <= 1e-4 * exact + floor + 1e-12)


def test_a_covered_cross_section_is_the_product_and_keeps_steps():
    bare = Dosimeter("X", [1.0, 10.0, 10.0, 100.0], [1.0, 1.0, 3.0, 3.0], 2, [1.0, 100.0], [[0.01]])
    cut = covered(bare, ([1.0, 50.0, 100.0], [0.2, 0.2, 1.0]))
    assert cut.name == "X covered" and cut.reaction == "X"
    np.testing.assert_array_equal(cut.relative_covariance, bare.relative_covariance)
    q = np.array([5.0, 20.0, 75.0])
    assert np.interp(q, cut.energies, cut.cross_section) == pytest.approx([0.2, 0.6, 3.0 * 0.6], rel=1e-3)
    step = np.flatnonzero(np.diff(cut.energies) == 0)
    assert cut.energies[step].tolist() == [10.0]


def test_a_negative_areal_density_is_refused():
    with pytest.raises(ValueError, match="areal_density"):
        cover_transmission([1.0, 2.0], [1.0, 1.0], 2, -1.0)

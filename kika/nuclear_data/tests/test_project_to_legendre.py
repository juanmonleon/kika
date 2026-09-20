"""AngularDistribution.project_to_legendre returns ENDF-convention moments.

Until 2026-09 it returned (2l+1) a_l: a_1 three times too large, a_2 five times,
and a PDF rebuilt from them that no longer matched its own table. Found when the
plotting layer overlaid ACE-projected coefficients on the MF4 of the same
evaluation (kika/plotting/tests/test_quantities.py has that tape-backed check).
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

from kika.nuclear_data.angular_distribution import AngularDistribution

A = {1: 0.30, 2: 0.12, 3: -0.05}


def _pdf(mu):
    """f(mu) = sum (2l+1)/2 a_l P_l(mu), with a_0 = 1."""
    coeffs = [0.5] + [(2 * l + 1) / 2 * A[l] for l in sorted(A)]
    return legval(mu, coeffs)


def _tabulated():
    mu = np.linspace(-1.0, 1.0, 401)
    f = _pdf(mu)
    return AngularDistribution(
        energies=np.array([1.0e6, 2.0e6]),
        coefficients={},
        reaction=2,
        nuclide_id=26056,
        frame="CM",
        representation="tabulated",
        tabulated_data={"cosines": [mu.tolist()] * 2, "probabilities": [f.tolist()] * 2,
                        "angular_interpolation": []},
    )


def test_projection_recovers_the_endf_coefficients():
    ad = _tabulated()
    ad.project_to_legendre(max_order=3)
    assert ad.coefficients[0] == pytest.approx([1.0, 1.0])
    for order, value in A.items():
        assert ad.coefficients[order] == pytest.approx([value, value], abs=1e-5)


def test_projection_is_insensitive_to_the_table_normalisation():
    ad = _tabulated()
    ad.tabulated_data["probabilities"] = [(np.asarray(p) * 3.0).tolist()
                                          for p in ad.tabulated_data["probabilities"]]
    ad.project_to_legendre(max_order=2)
    assert ad.coefficients[1] == pytest.approx([A[1], A[1]], abs=1e-5)

"""AngularDistribution.evaluate_pdf_vs_energy is evaluate_pdf, one cosine at a time.

The fixed-angle sweep that dσ/dΩ(E) needs calls the PDF at ~10^6 energies when
it folds, so it cannot be one evaluate_pdf call each. This pins the fast path to
the slow one, on synthetic sections of every representation and on the real
tapes in the local cache when they are there.
"""

import numpy as np
import pytest

import kika
from kika.nuclear_data.angular_distribution import AngularDistribution

QUERY = np.concatenate([np.geomspace(1e-5, 2e7, 701), [1.0e6, 1.5e6, 2.0e6, 3.0e6]])


def _slow(ad, mu, energies):
    return np.array([ad.evaluate_pdf(float(e), np.array([mu]))[1][0] for e in energies])


def _legendre():
    e = np.array([1e-5, 1e5, 1e6, 3e6, 2e7])
    return AngularDistribution(
        energies=e, coefficients={0: [1.0] * 5, 1: [0.0, 0.05, 0.3, 0.5, 0.7],
                                  2: [0.0, 0.0, 0.1, 0.2, 0.4]},
        reaction=2, nuclide_id=26056, frame="CM", representation="legendre",
    )


def _tabulated(energies):
    mu = np.linspace(-1, 1, 51)
    rows = [(0.5 + 0.3 * k * mu / len(energies)).tolist() for k in range(len(energies))]
    return mu, rows


def _mixed():
    leg = np.array([1e-5, 1e5, 1e6, 2e6])
    tab = np.array([2e6, 5e6, 2e7])
    mu, rows = _tabulated(tab)
    return AngularDistribution(
        energies=np.concatenate([leg, tab]),
        coefficients={0: [1.0] * 4, 1: [0.0, 0.05, 0.3, 0.45]},
        reaction=2, nuclide_id=26056, frame="CM", representation="mixed",
        tabulated_data={"cosines": [mu.tolist()] * 3, "probabilities": rows,
                        "angular_interpolation": [], "legendre_energies": leg.tolist(),
                        "tabulated_energies": tab.tolist()},
    )


def _pure_tabulated():
    e = np.array([1e5, 1e6, 2e7])
    mu, rows = _tabulated(e)
    return AngularDistribution(
        energies=e, coefficients={}, reaction=2, nuclide_id=26056, frame="CM",
        representation="tabulated",
        tabulated_data={"cosines": [mu.tolist()] * 3, "probabilities": rows,
                        "angular_interpolation": []},
    )


@pytest.mark.parametrize("build", [_legendre, _mixed, _pure_tabulated])
@pytest.mark.parametrize("mu", [1.0, 0.37, -0.8])
def test_matches_evaluate_pdf(build, mu):
    ad = build()
    np.testing.assert_allclose(ad.evaluate_pdf_vs_energy(mu, QUERY), _slow(ad, mu, QUERY),
                               rtol=1e-12, atol=1e-14)


def test_isotropic_is_a_half():
    ad = AngularDistribution(energies=np.array([1.0]), coefficients={}, reaction=2,
                             nuclide_id=1, frame="CM", representation="isotropic")
    assert ad.evaluate_pdf_vs_energy(0.2, [1.0, 2.0]).tolist() == [0.5, 0.5]


@pytest.mark.parametrize("library", ["jeff4.0", "endfb8.1"])
def test_real_tapes(library):
    from pathlib import Path
    path = Path.home() / ".kika" / "endf_cache" / library / "n" / "26056.endf"
    if not path.exists():
        pytest.skip(f"{path} not in the local cache")
    ad = AngularDistribution.from_endf(kika.read_endf(str(path), mf_numbers=[4]).files[4].mt[2])
    query = QUERY[::7]  # evaluate_pdf on a 32-order tape is the slow side
    for mu in (1.0, -0.9):
        np.testing.assert_allclose(ad.evaluate_pdf_vs_energy(mu, query), _slow(ad, mu, query),
                                   rtol=1e-10, atol=1e-12)

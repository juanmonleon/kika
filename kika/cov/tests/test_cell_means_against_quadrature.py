"""The cell means of the MF34 collapse against an independent integral.

``compute_base_cell_means`` integrates a piecewise-linear a_l under the flux
alone: phi = 1 by the trapezoid (exact for a linear a_l), phi = 1/E segment by
segment in closed form, and the Maxwellian and fission fluxes by the trapezoid
on the integration mesh. Each is compared here with scipy's adaptive
quadrature of the same integrals.
"""
import numpy as np
import pytest
from scipy import integrate

from kika.cov.multigroup.collapse import LegendreSource, compute_base_cell_means
from kika.processing.multigroup import WeightingFunction as W

#: a_1 tabulated on seven energies, linear in E between them.
_E = np.array([1e-5, 1.0, 1e3, 1e5, 1e6, 5e6, 2e7])
_A1 = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 0.7, 0.9])
_GRID = np.array([1e-5, 0.1, 10.0, 1e4, 3e5, 2e6, 2e7])
_SOURCE = LegendreSource(native_energies=_E,
                         coefficients={0: np.ones_like(_E), 1: _A1}, frame="LAB")


def _quad_means(phi):
    def a1(e):
        return np.interp(e, _E, _A1)
    out = []
    for lo, hi in zip(_GRID[:-1], _GRID[1:]):
        pts = [p for p in _E if lo < p < hi] or None
        kw = dict(points=pts, limit=500, epsrel=1e-12, epsabs=0.0)
        num = integrate.quad(lambda e: phi(e) * a1(e), lo, hi, **kw)[0]
        den = integrate.quad(phi, lo, hi, **kw)[0]
        out.append(num / den if den > 0 else np.nan)
    return np.array(out)


@pytest.mark.parametrize("name", ["constant", "lethargy"])
def test_constant_and_lethargy_weights_are_exact_on_a_linear_coefficient(name):
    phi = getattr(W, name)
    means = compute_base_cell_means(_GRID, _SOURCE, [0, 1], phi,
                                    getattr(W, f"{name}_antiderivative"))
    assert np.allclose(means[0], 1.0, rtol=0, atol=1e-14)
    assert np.allclose(means[1], _quad_means(phi), rtol=1e-12, atol=0)


@pytest.mark.parametrize("name", ["maxwellian", "fission_spectrum"])
def test_maxwellian_and_fission_weights_converge_with_the_mesh(name):
    """The trapezoid is as good as the mesh: coarse, it misses; on a 10 000-point
    mesh (what a sigma table supplies) it agrees with the quadrature to 1e-5."""
    phi = getattr(W, name)
    anti = getattr(W, f"{name}_antiderivative")
    ref = _quad_means(phi)
    # Groups where the Maxwellian at kT = 0.0253 eV has underflowed carry no
    # weight and are left at zero; only the groups with flux are compared.
    live = np.isfinite(ref) & (ref > 0)
    if name == "maxwellian":
        live[2:] = False
    coarse = compute_base_cell_means(_GRID, _SOURCE, [1], phi, anti)[1]
    assert np.max(np.abs(coarse[live] / ref[live] - 1)) > 1e-2
    dense = compute_base_cell_means(_GRID, _SOURCE, [1], phi, anti,
                                    pendf_energies=np.geomspace(1e-5, 2e7, 10_000))[1]
    assert np.max(np.abs(dense[live] / ref[live] - 1)) < 1e-5

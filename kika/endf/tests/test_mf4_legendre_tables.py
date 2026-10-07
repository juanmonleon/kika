"""a_l(E) is a table in incident energy, and its cell means are that table's integral.

MF34's cell means of a_l and the covariance check's bound used to sample a_l at
5 and 9 equally spaced energies per cell. On JEFF-4.0 Fe-56 elastic, whose a_l
carry the resonance structure, that missed a_1 by 0.13 in 0.65-0.75 MeV
(``kika_dev/processing/function_math/check_al_cell_averages.py``). Each MF4
class now states a_l(E) as one table (``legendre_table``) that reads back as
``extract_legendre_coefficients`` does, and the means integrate it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from kika.endf import read_endf
from kika.endf.classes.mf4.tabulated import MF4MTTabulated
from kika.endf.utils import project_tabulated_to_legendre

DATA = Path(__file__).resolve().parent / "data"


def _tabulated() -> MF4MTTabulated:
    mu = [-1.0, 0.0, 0.9, 1.0]
    return MF4MTTabulated(
        number=2,
        _energies=[1.0e6, 2.0e6, 4.0e6],
        _cosines=[mu, mu, mu],
        _probabilities=[[0.1, 0.2, 1.0, 8.0], [0.1, 0.2, 2.0, 16.0],
                        [0.5, 0.5, 0.5, 0.5]],
        _angular_interpolation=[[(4, 2)]] * 3,
        _interpolation=[(3, 2)],
    )


def _fe56_mixed():
    return read_endf(str(DATA / "micro_fe56_xs_and_angular.endf"),
                     mf_numbers=[4]).get_file(4).mt[2]


@pytest.mark.parametrize("section", [_tabulated, _fe56_mixed], ids=["LTT2", "LTT3"])
def test_the_table_reads_back_as_the_coefficients(section):
    from kika.algebra import evaluate

    section = section()
    E, A, laws, hold = section.legendre_table(4)
    lo, hi = E[0], E[-1]
    q = np.sort(np.r_[np.geomspace(lo, hi, 400), E[:200]])
    kw = {"trim": False} if hasattr(section, "_tab_interpolation") else {}
    want = section.extract_legendre_coefficients(q, 4, out_of_range="zero", **kw)
    for l in range(5):
        assert np.array_equal(evaluate(E, A[:, l], laws, q), np.asarray(want[l])), l


def test_a_cell_mean_is_the_integral_of_the_coefficient():
    section = _tabulated()
    edges = np.array([0.5e6, 1.3e6, 3.1e6, 5.0e6])  # reaches past both ends: held
    got = section.legendre_cell_averages(edges, 3)
    for l in (1, 3):
        def a(e):
            return section.extract_legendre_coefficients(e, 3)[l]
        exact = [quad(a, x0, x1, points=[1e6, 2e6, 4e6], limit=200)[0] / (x1 - x0)
                 for x0, x1 in zip(edges[:-1], edges[1:])]
        assert got[l] == pytest.approx(exact, rel=1e-10, abs=1e-14)


def test_the_smallest_magnitude_in_a_cell_is_zero_across_a_sign_change():
    section = _tabulated()          # a_1 > 0 at 1 and 2 MeV, 0 at 4 MeV (isotropic)
    edges = np.array([1.0e6, 1.5e6, 3.0e6, 4.0e6])
    smallest = section.legendre_cell_min_abs(edges, 1)
    a1 = section.extract_legendre_coefficients(np.array([1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6]), 1)[1]
    assert smallest[0] == pytest.approx(min(abs(a1[0]), abs(a1[1])))
    assert smallest[2] == pytest.approx(0.0, abs=1e-15)


def test_a_tabulated_projection_is_exact_on_its_panels():
    """f lin-lin in mu with a kink at 0.9: the moments are the integrals."""
    from scipy.special import eval_legendre

    mu = np.array([-1.0, 0.0, 0.9, 1.0])
    f = np.array([0.1, 0.2, 2.0, 16.0])
    got = project_tabulated_to_legendre(mu, f, 8, [(4, 2)])
    norm = sum(quad(lambda t: np.interp(t, mu, f), a, b)[0] for a, b in zip(mu[:-1], mu[1:]))
    for l in range(9):
        m = sum(quad(lambda t: np.interp(t, mu, f) * eval_legendre(l, t), a, b)[0]
                for a, b in zip(mu[:-1], mu[1:]))
        assert got[l] == pytest.approx(m / norm, abs=1e-14)

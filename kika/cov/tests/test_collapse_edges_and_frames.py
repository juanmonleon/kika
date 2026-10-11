"""The MF34 multigroup collapse at its edges and across frames.

Three defects found while documenting ``kika.cov.multigroup.collapse``
(11-oct-2026): weight outside the group structure clipped into the end groups,
the MF34 covariance copied beyond its own grid, and the CM->lab change done
with elastic kinematics for every reaction.
"""
import numpy as np
import pytest

from kika._legendre_frames import transferMatrix, twoBodyGamma
from kika.cov.multigroup.collapse import (_build_overlap_from_flux, _two_body_kinematics,
                                          build_union_grid, cm_to_lab_legendre,
                                          expand_matrix_to_union_grid)


def test_cells_outside_the_groups_carry_no_weight():
    mf34 = np.array([1.0, 10.0, 100.0, 1000.0])
    mg = np.array([10.0, 50.0, 100.0])
    union = build_union_grid(mf34, mg)          # 1, 10, 50, 100, 1000
    flux = np.diff(union)
    w = _build_overlap_from_flux(union, mg, flux)
    # The cells [1, 10] and [100, 1000] belong to no group.
    assert np.array_equal(w.sum(axis=0), [40.0, 50.0])
    assert w[0].sum() == 0.0 and w[-1].sum() == 0.0


def test_the_covariance_stops_where_mf34_does():
    mf34 = np.array([10.0, 100.0])
    union = build_union_grid(mf34, np.array([1.0, 10.0, 100.0, 1000.0]))
    expanded = expand_matrix_to_union_grid(np.array([[0.04]]), mf34, union)
    assert np.array_equal(expanded, np.diag([0.0, 0.04, 0.0]))


def _cm_coefficients(n):
    a = {l: np.array([0.5 ** l]) for l in range(n)}
    a[0] = np.array([1.0])
    return a


@pytest.mark.parametrize("gamma", [1.0 / 55.45, 0.3])
def test_the_lab_coefficients_match_the_frame_module(gamma):
    a = _cm_coefficients(6)
    lab = cm_to_lab_legendre(a, gamma)
    vec = np.array([a[l][0] for l in range(6)])
    expected = transferMatrix(gamma, 6, 6, "CM->LAB") @ vec
    assert np.allclose([lab[l][0] for l in range(6)], expected, atol=1e-10)


def test_one_gamma_per_energy_is_one_conversion_per_energy():
    gammas = np.array([0.02, 0.3, np.inf])
    a = {l: np.array([0.5 ** l] * 3) for l in range(4)}
    lab = cm_to_lab_legendre(a, gammas)
    for e, g in enumerate(gammas[:2]):
        single = cm_to_lab_legendre({l: v[e:e + 1] for l, v in a.items()}, g)
        assert np.allclose([lab[l][e] for l in a], [single[l][0] for l in a])
    # At or below threshold (gamma infinite) the coefficients are left alone.
    assert [lab[l][2] for l in a] == [a[l][2] for l in a]


class _Sec:
    def __init__(self, q):
        self.q_reaction = q


class _MF:
    def __init__(self, mts):
        self.mt = mts


class _Tape:
    def __init__(self, mf3):
        self.mf = {3: _MF(mf3)} if mf3 is not None else {}


class _MF4:
    _awr = 55.45


def test_elastic_and_a_level_get_their_own_kinematics():
    tape = _Tape({51: _Sec(-846.8e3)})
    assert _two_body_kinematics(tape, 2, _MF4()) == (55.45, 0.0)
    awr, q = _two_body_kinematics(tape, 51, _MF4())
    assert q == -846.8e3
    # Well above threshold a level's gamma differs from elastic's 1/AWR.
    assert twoBodyGamma(2e6, awr, q) > 1.0 / awr * 1.2


def test_a_reaction_that_is_not_two_body_is_refused():
    with pytest.raises(ValueError, match="two-body"):
        _two_body_kinematics(_Tape({}), 91, _MF4())
    with pytest.raises(ValueError, match="MF3/MT52"):
        _two_body_kinematics(_Tape(None), 52, _MF4())

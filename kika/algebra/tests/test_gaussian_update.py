"""The linear Gaussian update against closed forms, identities and a Monte Carlo of its own model."""
import numpy as np
import pytest

from kika.algebra import LinearUpdate, linear_update


def _spd(rng, n, scale=1.0, floor=0.1):
    a = rng.normal(size=(n, n))
    return scale * (a @ a.T / n + floor * np.eye(n))


def test_repeated_measurements_of_one_value_give_the_weighted_mean_with_the_prior():
    values = np.array([10.2, 9.7, 10.5, 10.0])
    sigma = np.array([0.3, 0.5, 0.4, 0.2])
    w = 1 / sigma**2
    prior_variance = 25.0
    u = linear_update([0.0], [[prior_variance]], np.ones((4, 1)), values, np.diag(sigma**2))
    assert isinstance(u, LinearUpdate)
    assert u.mean[0] == pytest.approx((w @ values) / (w.sum() + 1 / prior_variance), rel=1e-14)
    assert u.covariance[0, 0] == pytest.approx(1 / (w.sum() + 1 / prior_variance), rel=1e-13)
    assert u.dof == 4


def test_a_sequential_update_equals_the_joint_one_and_the_chi2_adds():
    rng = np.random.default_rng(1)
    n, m1, m2 = 8, 3, 4
    C0, x0 = _spd(rng, n), rng.normal(size=n)
    G1, G2 = rng.normal(size=(m1, n)), rng.normal(size=(m2, n))
    V1, V2 = np.diag(rng.uniform(0.1, 0.5, m1)), np.diag(rng.uniform(0.1, 0.5, m2))
    d1, d2 = rng.normal(size=m1), rng.normal(size=m2)
    V = np.zeros((m1 + m2, m1 + m2))
    V[:m1, :m1], V[m1:, m1:] = V1, V2
    joint = linear_update(x0, C0, np.vstack([G1, G2]), np.concatenate([d1 - G1 @ x0, d2 - G2 @ x0]), V)
    a = linear_update(x0, C0, G1, d1 - G1 @ x0, V1)
    b = linear_update(a.mean, a.covariance, G2, d2 - G2 @ a.mean, V2)
    np.testing.assert_allclose(b.mean, joint.mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.covariance, joint.covariance, rtol=0, atol=1e-12)
    assert a.chi2 + b.chi2 == pytest.approx(joint.chi2, rel=1e-12)


def test_an_identity_prior_over_lambda_is_ridge_regression():
    rng = np.random.default_rng(2)
    lam = 1e-2
    R = np.abs(rng.normal(size=(5, 12)))
    N = 3 * np.abs(rng.normal(size=5))
    sN = 0.05 * N
    guess = np.ones(12)
    Wd = np.diag(1 / sN**2)
    ridge = np.linalg.solve(R.T @ Wd @ R + lam * np.eye(12), R.T @ Wd @ N + lam * guess)
    u = linear_update(guess, np.eye(12) / lam, R, N - R @ guess, np.diag(sN**2))
    np.testing.assert_allclose(u.mean, ridge, rtol=1e-8)


def test_a_joint_parameter_vector_equals_folding_the_second_block_into_the_data():
    rng = np.random.default_rng(3)
    nf, ns, m = 10, 14, 4
    Cf = np.diag(rng.uniform(0.05, 0.3, nf) ** 2)
    Cs = 0.03**2 * _spd(rng, ns, floor=1.0)
    Ff, Fs = rng.uniform(0, 1, (m, nf)), rng.uniform(0, 1, (m, ns))
    V = np.diag(rng.uniform(0.02, 0.05, m) ** 2)
    r = rng.normal(scale=0.1, size=m)
    C = np.zeros((nf + ns, nf + ns))
    C[:nf, :nf], C[nf:, nf:] = Cf, Cs
    joint = linear_update(np.zeros(nf + ns), C, np.hstack([Ff, Fs]), r, V)
    folded = linear_update(np.zeros(nf), Cf, Ff, r, V + Fs @ Cs @ Fs.T)
    np.testing.assert_allclose(joint.mean[:nf], folded.mean, rtol=0, atol=1e-14)
    np.testing.assert_allclose(joint.covariance[:nf, :nf], folded.covariance, rtol=0, atol=1e-14)
    assert joint.chi2 == pytest.approx(folded.chi2, rel=1e-12)


def test_a_rank_one_prior_moves_only_its_common_factor():
    rng = np.random.default_rng(4)
    n, m = 10, 4
    C = 0.1**2 * np.ones((n, n))
    G = rng.uniform(0, 1, (m, n))
    u = linear_update(np.zeros(n), C, G, rng.normal(scale=0.1, size=m), np.diag(np.full(m, 0.03**2)))
    np.testing.assert_allclose(u.mean, u.mean[0], rtol=1e-12)
    assert u.covariance[0, 0] < C[0, 0]


def test_chi2_drop_is_the_chi2_lost_by_leaving_each_datum_out():
    rng = np.random.default_rng(5)
    n, m = 6, 5
    C, G = _spd(rng, n), rng.normal(size=(m, n))
    V = _spd(rng, m, scale=0.5)
    r = rng.normal(size=m)
    u = linear_update(np.zeros(n), C, G, r, V)
    for i in range(m):
        keep = np.arange(m) != i
        without = linear_update(np.zeros(n), C, G[keep], r[keep], V[np.ix_(keep, keep)])
        assert u.chi2 - without.chi2 == pytest.approx(u.chi2_drop[i], rel=1e-10)
    np.testing.assert_allclose(u.standardized_residuals, r / np.sqrt(np.diag(u.predictive_covariance)))


def test_chi2_has_m_degrees_of_freedom_and_the_posterior_covers_the_truth():
    rng = np.random.default_rng(6)
    n, m, draws = 20, 6, 3000
    C = 0.1**2 * _spd(rng, n, floor=0.2)
    G = rng.uniform(0, 1, (m, n))
    V = np.diag(rng.uniform(0.02, 0.06, m) ** 2)
    Lc, Lv = np.linalg.cholesky(C), np.linalg.cholesky(V)
    chi2, pulls, coverage = [], [], []
    for _ in range(draws):
        truth = Lc @ rng.normal(size=n)
        u = linear_update(np.zeros(n), C, G, G @ truth + Lv @ rng.normal(size=m), V)
        chi2.append(u.chi2)
        pulls.append(u.standardized_residuals)
        w, Q = np.linalg.eigh(u.covariance)
        coverage.append(np.sum((Q.T @ (truth - u.mean)) ** 2 / w) / n)
    assert np.mean(chi2) / m == pytest.approx(1, abs=0.05)
    assert np.var(chi2) / (2 * m) == pytest.approx(1, abs=0.12)
    np.testing.assert_allclose(np.std(pulls, axis=0), 1, atol=0.05)
    assert np.mean(coverage) == pytest.approx(1, abs=0.03)


def test_a_predictive_covariance_that_is_not_positive_definite_raises():
    with pytest.raises(np.linalg.LinAlgError, match="smallest eigenvalue"):
        linear_update([0.0, 0.0], np.zeros((2, 2)), np.eye(2), [1.0, 1.0], np.diag([1.0, -0.5]))


def test_an_indefinite_prior_warns_and_is_not_repaired():
    with pytest.warns(RuntimeWarning, match="negative eigenvalue"):
        u = linear_update([0.0, 0.0], np.diag([1.0, -0.5]), [[1.0, 0.0]], [0.3], [[0.1]])
    assert u.covariance[1, 1] == -0.5


@pytest.mark.parametrize("bad", [
    dict(prior_covariance=[[1.0, 0.2], [0.0, 1.0]]),
    dict(sensitivity=np.ones((1, 3))),
    dict(residual=[]),
    dict(data_covariance=[[np.nan]]),
])
def test_malformed_input_raises(bad):
    args = dict(prior_mean=[0.0, 0.0], prior_covariance=np.eye(2), sensitivity=[[1.0, 1.0]],
                residual=[0.5], data_covariance=[[0.1]])
    args.update(bad)
    with pytest.raises(ValueError):
        linear_update(**args)


def test_a_nearly_uninformative_prior_shows_in_the_condition_number():
    values = np.array([10.2, 9.7, 10.5, 10.0])
    sigma = np.array([0.3, 0.5, 0.4, 0.2])
    u = linear_update([0.0], [[1e12]], np.ones((4, 1)), values, np.diag(sigma**2))
    assert u.condition_number > 1e12

"""The prior covariance from sampled runs recovers the nuclear-data covariance under known Monte Carlo noise."""
import numpy as np
import pytest

from kika.dosimetry import nuclear_data_covariance, prior_covariance


def test_replicas_with_known_noise_give_back_the_nuclear_data_covariance():
    rng = np.random.default_rng(4)
    n, draws = 12, 20000
    loadings = rng.normal(scale=0.05, size=(n, 3))
    true = loadings @ loadings.T
    mean = np.linspace(1.0, 3.0, n)
    noise = np.linspace(0.01, 0.06, n)
    replicas = mean * (1 + rng.normal(size=(draws, 3)) @ loadings.T) * (1 + noise * rng.normal(size=(draws, n)))
    errors = np.tile(noise, (draws, 1))
    got = nuclear_data_covariance(replicas, errors)
    naive = np.cov((replicas / replicas.mean(axis=0)).T)
    np.testing.assert_allclose(got, true, atol=4e-4)
    assert np.all(np.diag(naive) - np.diag(true) > 0.5 * noise**2)


def test_a_group_of_zero_mean_has_zero_rows():
    replicas = np.array([[1.0, 0.0], [1.2, 0.0], [0.9, 0.0]])
    got = nuclear_data_covariance(replicas, np.zeros((3, 2)))
    assert np.all(got[1] == 0.0) and np.all(got[:, 1] == 0.0) and got[0, 0] > 0


def test_the_prior_adds_the_calculations_noise_and_a_full_normalisation():
    nd = np.diag([0.01, 0.02])
    C = prior_covariance(nd, statistical=[0.1, 0.0], normalization=0.05)
    np.testing.assert_allclose(C, [[0.01 + 0.01 + 0.0025, 0.0025], [0.0025, 0.02 + 0.0025]])
    assert nd[0, 0] == 0.01


@pytest.mark.parametrize("call", [
    lambda: nuclear_data_covariance(np.ones((1, 3)), np.zeros((1, 3))),
    lambda: nuclear_data_covariance(np.ones((3, 3)), np.zeros((3, 2))),
    lambda: nuclear_data_covariance(np.ones((3, 3)), -np.ones((3, 3))),
    lambda: prior_covariance(np.eye(2), statistical=[0.1]),
    lambda: prior_covariance(np.eye(2), normalization=-0.1),
])
def test_malformed_input_raises(call):
    with pytest.raises(ValueError):
        call()

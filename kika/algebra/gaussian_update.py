"""The linear Gaussian update: a prior, linear data, and the posterior.

A parameter vector ``x ~ N(x0, C)`` and data that depend on it linearly,
``d = G x + e`` with ``e ~ N(0, V)``, give a Gaussian posterior. This is the
generalized least-squares adjustment of spectrum unfolding (STAY'SL), of
cross-section adjustment to integral experiments (GLLS) and of a Kalman filter
step; here it is mathematics only, and whoever calls it says what ``x`` and
``d`` are.

The update is solved in the **data space**: only the ``m x m`` predictive
covariance ``S = G C G^T + V`` is factorised, never ``C``. A singular prior
covariance (a common normalisation of rank one, a sample covariance of fewer
draws than parameters) needs nothing special. ``S`` that is not positive
definite is an error, not something repaired here: covariance repairs are a
step a person runs first.

The data-space form loses digits when the prior is nearly uninformative, when
``G C G^T`` exceeds ``V`` by ten orders of magnitude or more; the condition
number of ``S`` is returned so that a caller can see it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

__all__ = ["LinearUpdate", "linear_update"]

#: Relative asymmetry tolerated in a covariance, against its largest entry.
SYMMETRY_RTOL = 1e-10

#: Negative eigenvalues of the posterior within this fraction of the largest prior entry are rounding.
POSTERIOR_EIGENVALUE_RTOL = 1e-10


@dataclass(frozen=True)
class LinearUpdate:
    """The posterior of a linear Gaussian update, and how well the data fit the prior.

    Returned by :func:`linear_update`.

    Attributes
    ----------
    mean : numpy.ndarray
        The posterior mean, ``n`` values.
    covariance : numpy.ndarray
        The posterior covariance, ``n x n``, symmetric.
    chi2 : float
        ``r^T S^-1 r``: the residual against its predictive covariance.
    dof : int
        The degrees of freedom of *chi2*: the number of data ``m``. The prior
        counts as information, so they are not ``m - n``.
    standardized_residuals : numpy.ndarray
        ``r_i / sqrt(S_ii)``, ``m`` values; each is N(0, 1) when the prior, the
        model and the data are consistent. When ``S`` has correlations their
        squares do not add up to *chi2*.
    chi2_drop : numpy.ndarray
        ``m`` values: *chi2* minus the chi-square of the same update with datum
        ``i`` left out, ``(S^-1 r)_i^2 / (S^-1)_ii``. The influence of each datum
        on the consistency, correlations included.
    predictive_covariance : numpy.ndarray
        ``S = G C G^T + V``, ``m x m``.
    condition_number : float
        The 2-norm condition number of ``S``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    chi2: float
    dof: int
    standardized_residuals: np.ndarray
    chi2_drop: np.ndarray
    predictive_covariance: np.ndarray
    condition_number: float


def _square(name: str, matrix: np.ndarray, size: int) -> None:
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be {size} x {size}, got {matrix.shape}")
    scale = np.max(np.abs(matrix)) if matrix.size else 0.0
    if np.max(np.abs(matrix - matrix.T), initial=0.0) > SYMMETRY_RTOL * scale:
        raise ValueError(f"{name} is not symmetric")


def _check_posterior(posterior_covariance, prior_covariance) -> None:
    """Warn, without repairing, if the posterior has a negative eigenvalue beyond the prior's rounding."""
    if not np.size(posterior_covariance):
        return
    smallest = float(np.linalg.eigvalsh(posterior_covariance)[0])
    if smallest < -POSTERIOR_EIGENVALUE_RTOL * np.max(np.abs(prior_covariance)):
        warnings.warn(
            f"the posterior covariance has a negative eigenvalue "
            f"({smallest:.6g}); is the prior covariance positive semi-definite?",
            RuntimeWarning, stacklevel=3)


def linear_update(prior_mean, prior_covariance, sensitivity, residual,
                  data_covariance, check: bool = True) -> LinearUpdate:
    """Update a Gaussian prior with data that depend on the parameters linearly.

    With ``x ~ N(prior_mean, C)`` and ``residual = G (x - prior_mean) + e``,
    ``e ~ N(0, V)``, the posterior is ``N(prior_mean + K r, C - K S K^T)`` with
    ``S = G C G^T + V`` and ``K = C G^T S^-1``. For a model linearised at a
    point ``x_k`` other than the prior mean (one Gauss-Newton step towards the
    maximum a posteriori), pass ``residual = d - h(x_k) + G (x_k - prior_mean)``
    with ``G`` the Jacobian at ``x_k``.

    Parameters
    ----------
    prior_mean : array_like
        The prior mean, ``n`` values.
    prior_covariance : array_like
        The prior covariance ``C``, ``n x n``, symmetric. It may be singular.
    sensitivity : array_like
        ``G``, ``m x n``: the derivative of each datum with respect to each
        parameter.
    residual : array_like
        ``r``, ``m`` values: the data minus their prediction at *prior_mean*.
    data_covariance : array_like
        ``V``, ``m x m``, symmetric: the covariance of the data about the model.
    check : bool
        Whether to look for negative eigenvalues of the posterior covariance
        (an ``n x n`` eigendecomposition, which dominates
        the cost when ``n`` is much larger than ``m``). A caller that has already
        checked the prior and iterates can check the last posterior alone.

    Returns
    -------
    LinearUpdate
        The posterior mean and covariance, the chi-square with its degrees of
        freedom, and the per-datum diagnostics.

    Raises
    ------
    ValueError
        If a shape does not match, there are no data, a value is not finite, or
        a covariance is not symmetric.
    numpy.linalg.LinAlgError
        If ``S`` is not positive definite. The message gives its smallest
        eigenvalue.

    Warns
    -----
    RuntimeWarning
        If *check* is true and the posterior covariance has a negative
        eigenvalue beyond rounding, which happens when *prior_covariance* is not
        positive semi-definite. The posterior is returned as computed, not
        repaired.
    """
    x0 = np.asarray(prior_mean, dtype=float)
    C = np.asarray(prior_covariance, dtype=float)
    G = np.asarray(sensitivity, dtype=float)
    r = np.asarray(residual, dtype=float)
    V = np.asarray(data_covariance, dtype=float)

    if x0.ndim != 1:
        raise ValueError(f"prior_mean must be one-dimensional, got shape {x0.shape}")
    if r.ndim != 1 or r.size == 0:
        raise ValueError(f"residual must be a non-empty one-dimensional array, got shape {r.shape}")
    n, m = x0.size, r.size
    if G.shape != (m, n):
        raise ValueError(f"sensitivity must be {m} x {n}, got {G.shape}")
    _square("prior_covariance", C, n)
    _square("data_covariance", V, m)
    for name, value in (("prior_mean", x0), ("prior_covariance", C), ("sensitivity", G),
                        ("residual", r), ("data_covariance", V)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} has a value that is not finite")

    CGt = C @ G.T
    S = G @ CGt + V
    S = 0.5 * (S + S.T)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        smallest = float(np.linalg.eigvalsh(S)[0])
        raise np.linalg.LinAlgError(
            f"the predictive covariance G C G^T + V is not positive definite "
            f"(smallest eigenvalue {smallest:.6g})") from None

    W = solve_triangular(L, CGt.T, lower=True)
    z = solve_triangular(L, r, lower=True)
    mean = x0 + W.T @ z
    covariance = C - W.T @ W
    covariance = 0.5 * (covariance + covariance.T)

    Sinv = solve_triangular(L.T, solve_triangular(L, np.eye(m), lower=True), lower=False)
    Sinv_r = Sinv @ r
    eigenvalues = np.linalg.eigvalsh(S)

    if check:
        _check_posterior(covariance, C)

    return LinearUpdate(
        mean=mean,
        covariance=covariance,
        chi2=float(z @ z),
        dof=m,
        standardized_residuals=r / np.sqrt(np.diag(S)),
        chi2_drop=Sinv_r ** 2 / np.diag(Sinv),
        predictive_covariance=S,
        condition_number=float(eigenvalues[-1] / eigenvalues[0]),
    )

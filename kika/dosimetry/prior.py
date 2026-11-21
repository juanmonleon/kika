"""The relative covariance of a calculated spectrum, built from what is known about the calculation.

Two pieces, added by :func:`prior_covariance`:

- the nuclear-data part, from transport runs with sampled nuclear data
  (:func:`nuclear_data_covariance`). The scatter of the replicas holds both the
  nuclear data and each replica's own Monte Carlo noise; the noise is
  subtracted as fast TMC does for one tally (:func:`kika.UQ.fastTMC`), here on
  the diagonal, since two independent runs' noises are uncorrelated;
- the nominal calculation's own statistical uncertainty, and the normalisation
  (source strength, power history) that sampling nuclear data does not see.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["nuclear_data_covariance", "prior_covariance"]


def nuclear_data_covariance(replicas, relative_errors) -> np.ndarray:
    """The relative covariance of group fluxes due to nuclear data, from sampled transport runs.

    Parameters
    ----------
    replicas : array_like
        ``N x n``: the group fluxes of ``N`` runs with sampled nuclear data, ``N >= 2``.
    relative_errors : array_like
        ``N x n``: the relative one-sigma Monte Carlo uncertainty of each of those
        values (a fraction, not a percentage).

    Returns
    -------
    numpy.ndarray
        ``n x n``: the sample covariance of the replicas relative to their mean,
        minus the mean statistical variance on the diagonal. Rows and columns of
        groups whose mean is zero are zero. It is not repaired: when the
        statistical noise is close to the scatter, the subtraction can leave it
        indefinite, and :func:`~kika.dosimetry.adjust_spectrum` will say so.

    Raises
    ------
    ValueError
        If the shapes differ, there are fewer than two replicas, a value is not
        finite, or a relative error is negative.
    """
    y = np.asarray(replicas, dtype=float)
    s = np.asarray(relative_errors, dtype=float)
    if y.ndim != 2 or y.shape != s.shape:
        raise ValueError(f"replicas and relative_errors must be two arrays of one N x n shape, "
                         f"got {y.shape} and {s.shape}")
    if y.shape[0] < 2:
        raise ValueError("at least two replicas are needed for a covariance")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))):
        raise ValueError("a replica or a relative error is not finite")
    if np.any(s < 0):
        raise ValueError("a relative error is negative")
    mean = y.mean(axis=0)
    scale = np.where(mean != 0, 1.0 / np.where(mean != 0, mean, 1.0), 0.0)
    deviations = (y - mean) * scale
    covariance = deviations.T @ deviations / (y.shape[0] - 1)
    statistical = np.mean((s * y) ** 2, axis=0) * scale**2
    covariance[np.diag_indices_from(covariance)] -= statistical
    return 0.5 * (covariance + covariance.T)


def prior_covariance(nuclear_data, statistical: Optional[np.ndarray] = None,
                     normalization: float = 0.0) -> np.ndarray:
    """The relative covariance of a prior spectrum: nuclear data, the calculation's own noise, and normalisation.

    Parameters
    ----------
    nuclear_data : array_like
        ``n x n``, the relative covariance due to nuclear data
        (:func:`nuclear_data_covariance`).
    statistical : array_like, optional
        ``n`` relative one-sigma Monte Carlo uncertainties of the calculation the
        prior flux comes from; None adds none.
    normalization : float
        The relative one-sigma uncertainty of the absolute scale, common to every
        group (fully correlated).

    Returns
    -------
    numpy.ndarray
        ``n x n``: ``nuclear_data + diag(statistical^2) + normalization^2``.

    Raises
    ------
    ValueError
        If *nuclear_data* is not square, *statistical* does not have one value per
        group or is negative, or *normalization* is negative.
    """
    C = np.array(nuclear_data, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"nuclear_data must be square, got shape {C.shape}")
    if statistical is not None:
        sd = np.asarray(statistical, dtype=float)
        if sd.shape != (C.shape[0],) or np.any(sd < 0):
            raise ValueError(f"statistical must be {C.shape[0]} values, none negative")
        C[np.diag_indices_from(C)] += sd**2
    if normalization < 0:
        raise ValueError("normalization must not be negative")
    return C + normalization**2

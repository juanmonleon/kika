"""Generalized least-squares adjustment of a group spectrum to measured dosimeter rates.

The parameters are ``x_g = ln(phi_g / phi_g^prior)``, the data the logarithms of
the measured rates. The model ``ln R_i(x)`` is not linear in ``x``, so the update
is iterated (Gauss-Newton) to the maximum of the posterior; the first iteration
is the one-step linear adjustment of Perey's STAY'SL in relative variables. The
cross sections are not adjusted: their relative covariance is folded into that of
the data, ``V + F C_sigma F^T`` with ``F`` the share of each rate in each
covariance interval at the current spectrum. Each step is
:func:`kika.algebra.linear_update`.

The adjusted flux ``phi^prior exp(x)`` is the median of the log-normal posterior,
not its mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

from kika.algebra import LinearUpdate, linear_update

from .response import Dosimeter, SpectrumPrior, response_cells

__all__ = ["AdjustmentNotConverged", "SpectrumAdjustment", "adjust_spectrum"]

#: A Gauss-Newton step below this in every ``ln phi_g`` ends the iteration.
STEP_TOLERANCE = 1e-12

#: Gauss-Newton steps allowed before the adjustment is declared not converged.
MAX_ITERATIONS = 100


class AdjustmentNotConverged(RuntimeError):
    """The Gauss-Newton iteration of :func:`adjust_spectrum` did not converge; nothing is returned."""


@dataclass(frozen=True)
class SpectrumAdjustment:
    """The adjusted spectrum, its covariance and how well the dosimeters agree with the prior.

    Returned by :func:`adjust_spectrum`.

    Attributes
    ----------
    edges : numpy.ndarray
        Group boundaries in eV, as in the prior.
    prior_flux : numpy.ndarray
        The prior group fluxes.
    flux : numpy.ndarray
        The adjusted group fluxes (posterior median), in the prior's unit.
    log_covariance : numpy.ndarray
        The posterior covariance of ``ln phi_g``, ``n x n``.
    one_step_flux : numpy.ndarray
        The flux after the first iteration alone (the linear one-step adjustment).
    names : tuple of str
        The dosimeters, in the order of every per-dosimeter array.
    measured_rates, prior_rates, adjusted_rates : numpy.ndarray
        The measured rates and the rates of the prior and of the adjusted
        spectrum, per target atom and per unit time of the flux.
    chi2 : float
        The chi-square of the last linearisation, with *dof* degrees of freedom:
        at convergence, the minimum of the posterior objective.
    dof : int
        The number of dosimeters.
    standardized_residuals : numpy.ndarray
        Each dosimeter's residual over its predictive standard deviation, from the
        last linearisation.
    chi2_drop : numpy.ndarray
        The chi-square lost by leaving each dosimeter out.
    iterations : int
        The Gauss-Newton steps taken.
    """

    edges: np.ndarray
    prior_flux: np.ndarray
    flux: np.ndarray
    log_covariance: np.ndarray
    one_step_flux: np.ndarray
    names: Tuple[str, ...]
    measured_rates: np.ndarray
    prior_rates: np.ndarray
    adjusted_rates: np.ndarray
    chi2: float
    dof: int
    standardized_residuals: np.ndarray
    chi2_drop: np.ndarray
    iterations: int

    def integral(self, weights) -> Tuple[float, float]:
        """``sum_g w_g phi_g`` of the adjusted flux and its standard deviation.

        Parameters
        ----------
        weights : array_like
            One weight per group: 1 and 0 select groups, a response averaged over
            each group gives a rate.

        Returns
        -------
        tuple of float
            The value and its one-sigma uncertainty, propagated to first order from
            *log_covariance*.

        Raises
        ------
        ValueError
            If there is not one weight per group.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != self.flux.shape:
            raise ValueError(f"one weight per group: {self.flux.size} expected, got shape {w.shape}")
        part = w * self.flux
        return float(part.sum()), float(np.sqrt(max(part @ self.log_covariance @ part, 0.0)))

    def flux_above(self, energy: float) -> Tuple[float, float]:
        """The adjusted flux above a group boundary and its standard deviation.

        Parameters
        ----------
        energy : float
            The lower limit in eV. It must be one of the group boundaries other than
            the last, so that no group is split.

        Returns
        -------
        tuple of float
            The flux summed over the groups above *energy*, and its one-sigma
            uncertainty (:meth:`integral`).

        Raises
        ------
        ValueError
            If *energy* is not one of the group boundaries other than the last.
        """
        where = np.flatnonzero(self.edges == energy)
        if where.size != 1 or where[0] == self.edges.size - 1:
            raise ValueError(f"{energy!r} eV is not a lower group boundary")
        return self.integral(np.arange(self.flux.size) >= where[0])


def _linearise(x, cells, sigma_covariances):
    scale = np.exp(x)[:, None]
    parts = [c * scale for c in cells]
    rates = np.array([p.sum() for p in parts])
    group_share = np.vstack([p.sum(axis=1) for p in parts]) / rates[:, None]
    interval_share = [p.sum(axis=0) / r for p, r in zip(parts, rates)]
    folded = block_diag(*[s @ cov @ s for s, cov in zip(interval_share, sigma_covariances)])
    return rates, group_share, folded


def adjust_spectrum(prior: SpectrumPrior, dosimeters: Sequence[Dosimeter], measured_rates,
                    measured_covariance) -> SpectrumAdjustment:
    """Adjust a group spectrum to measured dosimeter rates, with every covariance.

    Parameters
    ----------
    prior : SpectrumPrior
        The calculated spectrum with its relative covariance.
    dosimeters : sequence of Dosimeter
        The reactions measured, with their cross sections and relative covariances.
        Different dosimeters' cross sections are taken as uncorrelated.
    measured_rates : array_like
        One positive rate per dosimeter, per target atom and per unit time of the
        prior's flux (saturation rates, with every decay, burn-up and cover
        correction already applied).
    measured_covariance : array_like
        The relative covariance of the measured rates, square, symmetric;
        correlations between dosimeters are allowed.

    Returns
    -------
    SpectrumAdjustment
        The adjusted spectrum and its covariance, the rates before and after, and
        the consistency diagnostics.

    Raises
    ------
    ValueError
        If there is not one positive rate per dosimeter, the covariance shape does
        not match, or a dosimeter has no response in the prior spectrum.
    numpy.linalg.LinAlgError
        If the predictive covariance of the rates is not positive definite.
    AdjustmentNotConverged
        If the Gauss-Newton iteration does not converge.
    """
    dosimeters = list(dosimeters)
    measured = np.asarray(measured_rates, dtype=float)
    m = len(dosimeters)
    if m == 0 or measured.shape != (m,):
        raise ValueError(f"one measured rate per dosimeter: {m} dosimeters, rates of shape {measured.shape}")
    if not np.all(measured > 0):
        raise ValueError("every measured rate must be positive")
    V = np.asarray(measured_covariance, dtype=float)
    if V.shape != (m, m):
        raise ValueError(f"measured_covariance must be {m} x {m}, got {V.shape}")

    cells = [response_cells(prior, d) for d in dosimeters]
    for d, c in zip(dosimeters, cells):
        if not c.sum() > 0:
            raise ValueError(f"{d.name} has no response in the prior spectrum")
    sigma_covariances = [d.cell_covariance for d in dosimeters]
    n = prior.flux.size
    x0 = np.zeros(n)
    x = x0.copy()
    data = np.log(measured)
    prior_rates = np.array([c.sum() for c in cells])
    one_step = None
    update: LinearUpdate
    for iteration in range(1, MAX_ITERATIONS + 1):
        rates, G, folded = _linearise(x, cells, sigma_covariances)
        update = linear_update(x0, prior.relative_covariance, G, data - np.log(rates) + G @ (x - x0),
                               V + folded)
        step = float(np.max(np.abs(update.mean - x)))
        x = update.mean
        if one_step is None:
            one_step = prior.flux * np.exp(x)
        if step < STEP_TOLERANCE:
            break
    else:
        raise AdjustmentNotConverged(
            f"no convergence in {MAX_ITERATIONS} Gauss-Newton steps (last step {step:.3g} in ln phi)")

    adjusted_rates = _linearise(x, cells, sigma_covariances)[0]
    return SpectrumAdjustment(
        edges=prior.edges,
        prior_flux=prior.flux,
        flux=prior.flux * np.exp(x),
        log_covariance=update.covariance,
        one_step_flux=one_step,
        names=tuple(d.name for d in dosimeters),
        measured_rates=measured,
        prior_rates=prior_rates,
        adjusted_rates=adjusted_rates,
        chi2=update.chi2,
        dof=update.dof,
        standardized_residuals=update.standardized_residuals,
        chi2_drop=update.chi2_drop,
        iterations=iteration,
    )

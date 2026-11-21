"""Generalized least-squares adjustment of a group spectrum and its dosimetry cross sections.

The parameters are ``ln(phi_g / phi_g^prior)`` for every group with a positive
prior flux, and ``ln(sigma_k / sigma_k^prior)`` for every covariance interval of
every reaction; the data are the logarithms of the measured rates, whose
covariance is that of the measurements alone. Flux and cross sections are
adjusted together, so the posterior holds their correlation. The model
``ln R_i(x)`` is not linear in ``x``, so the update is iterated (Gauss-Newton) to
the maximum of the posterior; each step is :func:`kika.algebra.linear_update`.
The first iteration agrees to first order with Perey's one-step STAY'SL
adjustment, which is linear in relative deviations; it is not equal to it (here
the residual is ``ln(M/C)`` and the flux ``phi exp(x)``).

Adjusted values ``prior x exp(x)`` are medians of the log-normal posterior, not
means. A group with zero prior flux is not a parameter: it stays zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

from kika.algebra import LinearUpdate, linear_update
from kika.algebra.gaussian_update import _check_posterior
from kika.cov.conditioning import ROUNDOFF_RATIO

from .response import Dosimeter, SpectrumPrior, response_cells

__all__ = ["AdjustmentNotConverged", "SpectrumAdjustment", "adjust_spectrum"]

#: A Gauss-Newton step below this in every parameter, times the prior's largest variance when
#: it exceeds one, ends the iteration (the update's round-off grows with that variance).
STEP_TOLERANCE = 1e-12

#: Gauss-Newton steps allowed before the adjustment is declared not converged.
MAX_ITERATIONS = 100

#: Two energies closer than this, relatively, are the same group boundary.
EDGE_RTOL = 1e-9

#: The prior variance of ``ln`` of a free normalization: wide enough that the data alone set it
#: (it moves the posterior by about the data's variance over it: 1e-4 at a 10 % level).
FREE_NORMALIZATION_VARIANCE = 1e2


class AdjustmentNotConverged(RuntimeError):
    """The Gauss-Newton iteration of :func:`adjust_spectrum` did not converge; nothing is returned."""


@dataclass(frozen=True)
class SpectrumAdjustment:
    """The adjusted spectrum and cross sections, their covariances, and how well the dosimeters agree.

    Returned by :func:`adjust_spectrum`.

    Attributes
    ----------
    edges : numpy.ndarray
        Group boundaries in eV, as in the prior.
    prior_flux : numpy.ndarray
        The prior group fluxes, times *normalization*.
    flux : numpy.ndarray
        The adjusted group fluxes (posterior median), in the prior's unit; zero
        where the prior is zero.
    log_covariance : numpy.ndarray
        The posterior covariance of ``ln phi_g``, ``n x n``; zero rows and columns
        for groups of zero prior flux.
    one_step_flux : numpy.ndarray
        The flux after the first Gauss-Newton iteration alone; it agrees with a linear
        one-step adjustment to first order.
    names : tuple of str
        The dosimeters, in the order of every per-dosimeter array.
    measured_rates, prior_rates, adjusted_rates : numpy.ndarray
        The measured rates and the rates of the prior and of the adjusted
        spectrum and cross sections, per target atom and per unit time of the flux.
    chi2 : float
        The chi-square of the last linearisation, with *dof* degrees of freedom:
        at convergence, the minimum of the posterior objective.
    dof : int
        The number of dosimeters, minus one with a free normalization.
    standardized_residuals : numpy.ndarray
        Each dosimeter's residual over its predictive standard deviation, from the
        last linearisation; with a free normalization, the residual once the
        common factor that best fits all of them is taken out, over the standard
        deviation that leaves.
    chi2_drop : numpy.ndarray
        The chi-square lost by leaving each dosimeter out.
    iterations : int
        The Gauss-Newton steps taken.
    cross_section_factors : dict
        Reaction (the dosimeters' *reaction*) to the adjusted over prior cross section on each of its
        covariance intervals (posterior median), for every reaction with a
        covariance.
    cross_section_log_covariance : dict
        Reaction name to the posterior covariance of the logarithms of those factors.
    flux_cross_section_log_covariance : dict
        Reaction name to the posterior covariance between ``ln phi_g`` (rows) and
        the logarithms of its factors (columns).
    prior_log_covariance : numpy.ndarray
        The prior's relative covariance, ``n x n``, with zero rows and columns for
        groups of zero prior flux; with a free normalization, that of its shape.
    notes : tuple of str
        What the report should say about the inputs: reactions with no covariance
        (taken as exact), covariances with negative eigenvalues no larger than
        eigensolver round-off, and a free normalization with its factor.
    normalization : float
        The factor the prior was scaled by before the adjustment: the data's
        estimate with a free normalization, 1 otherwise.
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
    cross_section_factors: Dict[str, np.ndarray] = field(default_factory=dict)
    cross_section_log_covariance: Dict[str, np.ndarray] = field(default_factory=dict)
    flux_cross_section_log_covariance: Dict[str, np.ndarray] = field(default_factory=dict)
    prior_log_covariance: np.ndarray = None
    notes: Tuple[str, ...] = ()
    normalization: float = 1.0

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

    def prior_integral(self, weights) -> Tuple[float, float]:
        """``sum_g w_g phi_g`` of the prior flux and its standard deviation, as :meth:`integral` does for the adjusted one.

        Parameters
        ----------
        weights : array_like
            One weight per group.

        Returns
        -------
        tuple of float
            The value and its one-sigma uncertainty from *prior_log_covariance*.

        Raises
        ------
        ValueError
            If there is not one weight per group.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != self.prior_flux.shape:
            raise ValueError(f"one weight per group: {self.prior_flux.size} expected, got shape {w.shape}")
        part = w * self.prior_flux
        return float(part.sum()), float(np.sqrt(max(part @ self.prior_log_covariance @ part, 0.0)))

    def to_html(self) -> str:
        """The report of the adjustment as one HTML page: verdict, dosimeters, integrals, spectrum, cross sections, method."""
        from .report import adjustment_html

        return adjustment_html(self)

    def write_html(self, path) -> None:
        """Write :meth:`to_html` to *path*, in UTF-8.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write; it is replaced if it exists.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_html())

    def flux_above(self, energy: float) -> Tuple[float, float]:
        """The adjusted flux above a group boundary and its standard deviation.

        Parameters
        ----------
        energy : float
            The lower limit in eV. It must be one of the group boundaries other than
            the last (to a relative 1e-9, so that a grid converted from MeV matches), so
            that no group is split.

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
        where = np.flatnonzero(np.isclose(self.edges, energy, rtol=EDGE_RTOL, atol=0.0))
        if where.size != 1 or where[0] == self.edges.size - 1:
            raise ValueError(f"{energy!r} eV is not a lower group boundary")
        return self.integral(np.arange(self.flux.size) >= where[0])


def _reactions(dosimeters: List[Dosimeter]) -> Dict[str, Dosimeter]:
    """One entry per reaction; its dosimeters must state the same covariance, and those of one name the same table."""
    out: Dict[str, Dosimeter] = {}
    names: Dict[str, Dosimeter] = {}
    for d in dosimeters:
        first = names.setdefault(d.name, d)
        if first is not d and not (first.reaction == d.reaction and np.array_equal(first.energies, d.energies)
                                   and np.array_equal(first.cross_section, d.cross_section)
                                   and np.array_equal(first.laws, d.laws)):
            raise ValueError(f"two dosimeters are named {d.name!r} but hold different data; name a covered "
                             f"foil apart and give it the bare one's reaction")
        first = out.setdefault(d.reaction, d)
        if first is d:
            continue
        same = (np.array_equal(first.cell_covariance, d.cell_covariance)
                and (first.covariance_edges is None) == (d.covariance_edges is None)
                and (d.covariance_edges is None or np.array_equal(first.covariance_edges, d.covariance_edges)))
        if not same:
            raise ValueError(f"the dosimeters of reaction {d.reaction!r} state different covariances; "
                             f"they share its cross-section parameters, so they must state the same one")
    return out


def _definiteness(label: str, matrix: np.ndarray):
    """``(smallest, largest)`` when *matrix* is negative only by round-off; a ValueError when it is not positive semi-definite."""
    if matrix.size == 0:
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if smallest >= 0.0:
        return None
    if largest > 0.0 and -smallest / largest <= matrix.shape[0] * ROUNDOFF_RATIO:
        return smallest, largest
    raise ValueError(
        f"the relative covariance of {label} is not positive semi-definite: {int((eigenvalues < 0).sum())} "
        f"negative eigenvalues, the smallest {smallest:.3g} against a largest of {largest:.3g}. Repair it "
        f"before the adjustment, with the pre-flight of kika.cov.conditioning (inspect_matrix says how much "
        f"each remedy changes it, apply_plan applies the one chosen).")


class _Layout:
    """Where each parameter sits: the positive-flux groups, then each reaction's intervals."""

    def __init__(self, prior: SpectrumPrior, reactions: Dict[str, Dosimeter]):
        self.groups = np.flatnonzero(prior.flux > 0)
        self.slices: Dict[str, slice] = {}
        start = self.groups.size
        blocks = [prior.relative_covariance[np.ix_(self.groups, self.groups)]]
        for name, d in reactions.items():
            k = d.cell_covariance.shape[0] - 1
            if k:
                self.slices[name] = slice(start, start + k)
                blocks.append(d.cell_covariance[:k, :k])
                start += k
        self.size = start
        self.prior_covariance = block_diag(*blocks)
        labels = ["the prior spectrum"] + [name for name in reactions if name in self.slices]
        rounded = [(label, found) for label, found in ((label, _definiteness(label, block))
                                                       for label, block in zip(labels, blocks)) if found]
        notes = []
        exact = [name for name in reactions if name not in self.slices]
        if exact:
            notes.append(f"No covariance for {', '.join(exact)}: taken as exact, so the adjustment puts "
                         f"all of their disagreement on the flux.")
        if rounded:
            worst = max(-small / large for _, (small, large) in rounded)
            notes.append(f"{len(rounded)} of {len(blocks)} covariances have negative eigenvalues within "
                         f"eigensolver round-off ({', '.join(label for label, _ in rounded)}); the largest "
                         f"ratio to the largest eigenvalue is {worst:.2g}. They were used as they are.")
        self.notes = tuple(notes)


def _rates_and_jacobian(x, layout, n_groups, keys, cells):
    phi_scale = np.ones(n_groups)
    phi_scale[layout.groups] = np.exp(x[:layout.groups.size])
    rates = np.empty(len(cells))
    G = np.zeros((len(cells), layout.size))
    for i, (key, c) in enumerate(zip(keys, cells)):
        sigma_scale = np.ones(c.shape[1])
        where = layout.slices.get(key)
        if where is not None:
            sigma_scale[:-1] = np.exp(x[where])
        part = c * phi_scale[:, None] * sigma_scale[None, :]
        rates[i] = part.sum()
        G[i, :layout.groups.size] = part.sum(axis=1)[layout.groups] / rates[i]
        if where is not None:
            G[i, where] = part.sum(axis=0)[:-1] / rates[i]
    return rates, G


def _normalization(G, C, V, residual):
    """The GLS estimate of a common log factor of the flux, and the covariance of the residuals it leaves.

    With ``S = G C G^T + V`` the factor is ``c = 1^T S^-1 r / 1^T S^-1 1`` and the
    residuals ``r - c`` have the covariance ``S - 1 1^T / (1^T S^-1 1)``.
    """
    S = G @ C @ G.T + V
    ones = np.ones(residual.size)
    Sinv_1 = np.linalg.solve(S, ones)
    weight = float(ones @ Sinv_1)
    return float(Sinv_1 @ residual) / weight, S - 1.0 / weight


def adjust_spectrum(prior: SpectrumPrior, dosimeters: Sequence[Dosimeter], measured_rates,
                    measured_covariance, free_normalization: bool = False) -> SpectrumAdjustment:
    """Adjust a group spectrum and the dosimetry cross sections to measured rates, with every covariance.

    Parameters
    ----------
    prior : SpectrumPrior
        The calculated spectrum with its relative covariance.
    dosimeters : sequence of Dosimeter
        One per measured rate. Dosimeters of the same reaction
        (their *reaction*: two foils, or one foil bare and one covered)
        share its cross-section parameters; different reactions' cross sections are
        taken as uncorrelated.
    measured_rates : array_like
        One positive rate per dosimeter, per target atom and per unit time of the
        prior's flux (saturation rates, with every decay and burn-up correction
        already applied; a cover is part of the dosimeter, :func:`~kika.dosimetry.covered`).
    measured_covariance : array_like
        The relative covariance of the measured rates, square, symmetric;
        correlations between dosimeters are allowed.
    free_normalization : bool
        Whether the absolute level of the prior is unknown and left to the data
        (a calculation in arbitrary units, or one whose source strength is not
        known). The prior is first scaled by the generalized least-squares estimate
        of the common factor (the result's *normalization*), and that factor then
        gets a prior variance of 100 in ``ln``, so its posterior comes from the
        data alone; the chi-square has one degree of freedom fewer. False takes
        the prior's level, with the uncertainty its covariance states.

    Returns
    -------
    SpectrumAdjustment
        The adjusted spectrum and cross sections with their covariances, the rates
        before and after, and the consistency diagnostics.

    Raises
    ------
    ValueError
        If there is not one positive rate per dosimeter, the covariance shape does
        not match, a dosimeter has no response in the prior spectrum, two
        dosimeters of one name hold different data, two of one reaction state
        different covariances, the normalization is free with fewer than two
        dosimeters, or the prior's or a reaction's relative covariance is not
        positive semi-definite beyond eigensolver round-off (``order x machine
        epsilon`` relative to its largest eigenvalue, as in
        :mod:`kika.cov.conditioning`).
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
    if free_normalization and m < 2:
        raise ValueError("a free normalization needs two dosimeters or more: one only fixes the level")
    V = np.asarray(measured_covariance, dtype=float)
    if V.shape != (m, m):
        raise ValueError(f"measured_covariance must be {m} x {m}, got {V.shape}")

    reactions = _reactions(dosimeters)
    cells = [response_cells(prior, d) for d in dosimeters]
    for d, c in zip(dosimeters, cells):
        if not c.sum() > 0:
            raise ValueError(f"{d.name} has no response in the prior spectrum")
    names = [d.name for d in dosimeters]
    keys = [d.reaction for d in dosimeters]
    layout = _Layout(prior, reactions)
    n = prior.flux.size
    x0 = np.zeros(layout.size)
    x = x0.copy()
    data = np.log(measured)
    shape_covariance = layout.prior_covariance
    prior_covariance = shape_covariance
    prior_flux = prior.flux
    notes = layout.notes
    normalization = 1.0
    if free_normalization:
        rates, G = _rates_and_jacobian(x, layout, n, keys, cells)
        c, _ = _normalization(G, shape_covariance, V, data - np.log(rates))
        normalization = float(np.exp(c))
        cells = [cell * normalization for cell in cells]
        prior_flux = prior.flux * normalization
        k = layout.groups.size
        prior_covariance = shape_covariance.copy()
        prior_covariance[:k, :k] += FREE_NORMALIZATION_VARIANCE
        notes = notes + (f"Free normalization: the prior was scaled by {normalization:.6g}, the "
                         f"generalized least-squares estimate from the rates, and its level left to the "
                         f"data; the prior uncertainties shown are those of its shape.",)
    prior_rates = np.array([c.sum() for c in cells])

    def flux_of(params):
        out = np.zeros(n)
        out[layout.groups] = prior_flux[layout.groups] * np.exp(params[:layout.groups.size])
        return out

    tolerance = STEP_TOLERANCE * max(1.0, float(np.max(np.diag(prior_covariance))))
    one_step = None
    update: LinearUpdate
    for iteration in range(1, MAX_ITERATIONS + 1):
        rates, G = _rates_and_jacobian(x, layout, n, keys, cells)
        residual = data - np.log(rates) + G @ (x - x0)
        update = linear_update(x0, prior_covariance, G, residual, V, check=False)
        step = float(np.max(np.abs(update.mean - x)))
        x = update.mean
        if one_step is None:
            one_step = flux_of(x)
        if step < tolerance:
            break
    else:
        raise AdjustmentNotConverged(
            f"no convergence in {MAX_ITERATIONS} Gauss-Newton steps (last step {step:.3g})")
    _check_posterior(update.covariance, prior_covariance)

    dof = update.dof
    standardized = update.standardized_residuals
    if free_normalization:
        dof -= 1
        c, left = _normalization(G, shape_covariance, V, residual)
        standardized = (residual - c) / np.sqrt(np.diag(left))

    g = layout.groups
    prior_log_covariance = np.zeros((n, n))
    prior_log_covariance[np.ix_(g, g)] = prior.relative_covariance[np.ix_(g, g)]
    log_covariance = np.zeros((n, n))
    log_covariance[np.ix_(g, g)] = update.covariance[:g.size, :g.size]
    factors, sigma_cov, cross_cov = {}, {}, {}
    for name, where in layout.slices.items():
        factors[name] = np.exp(x[where])
        sigma_cov[name] = update.covariance[where, where]
        block = np.zeros((n, where.stop - where.start))
        block[g] = update.covariance[:g.size, where]
        cross_cov[name] = block

    return SpectrumAdjustment(
        edges=prior.edges,
        prior_flux=prior_flux,
        flux=flux_of(x),
        log_covariance=log_covariance,
        one_step_flux=one_step,
        names=tuple(names),
        measured_rates=measured,
        prior_rates=prior_rates,
        adjusted_rates=_rates_and_jacobian(x, layout, n, keys, cells)[0],
        chi2=update.chi2,
        dof=dof,
        standardized_residuals=standardized,
        chi2_drop=update.chi2_drop,
        iterations=iteration,
        cross_section_factors=factors,
        cross_section_log_covariance=sigma_cov,
        flux_cross_section_log_covariance=cross_cov,
        prior_log_covariance=prior_log_covariance,
        notes=notes,
        normalization=normalization,
    )

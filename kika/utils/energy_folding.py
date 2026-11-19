"""
Energy folding utilities for differential cross section calculations.

This module provides tools for computing energy-folded differential cross sections,
accounting for TOF (Time-of-Flight) energy resolution effects in experimental data.

The folding can be applied to:
- Cross sections only (xs_only): Default mode, folds σ(E) with Gaussian kernel
- Angular distributions only (angular_only): Folds f(μ,E) shape with Gaussian kernel
- Both (both): Folds both XS and angular distribution

Theory:
    The measured differential cross section at energy E₀ with resolution σE is:

    dσ/dΩ(μ, E₀) = σ_folded(E₀) × f_folded(μ, E₀)

    where:
    - σ_folded = ∫ σ(E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE
    - f_folded = ∫ f(μ, E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE
    - G(E; E₀, σE) = exp(-0.5 × ((E - E₀) / σE)²) is the Gaussian kernel

TOF Energy Resolution:
    The energy resolution σE is computed from TOF parameters:
    1. FWHM_E = E × 2 × Δt / t  (TOF differential relation)
    2. σE = FWHM_E / 2.35482    (FWHM to standard deviation)

    IMPORTANT: delta_t_ns is interpreted as the FWHM of the time resolution,
    following the standard experimental convention used by GELINA, ORELA, n_TOF, etc.

Usage:
    >>> from kika.utils.energy_folding import (
    ...     EnergyFoldingConfig,
    ...     compute_energy_resolution_tof,
    ...     compute_folded_differential_xs,
    ... )
    >>>
    >>> # Create config with GELINA defaults (delta_t_ns is FWHM)
    >>> config = EnergyFoldingConfig(delta_t_ns=10.0, flight_path_m=27.037)
    >>>
    >>> # Compute σE at 1.406 MeV
    >>> sigma_E = compute_energy_resolution_tof(1.406, config)
    >>> print(f"σE = {sigma_E * 1000:.2f} keV")  # ~7.2 keV
    >>>
    >>> # Compute folded differential cross section
    >>> mu_grid, dsigma_domega, info = compute_folded_differential_xs(
    ...     ace_data=ace,
    ...     endf_coeffs=coeffs,  # ENDF format [a1, a2, ...]
    ...     target_energy_mev=1.406,
    ...     mt=2,
    ...     config=config,
    ...     mode="xs_only",  # or "angular_only" or "both"
    ... )
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import warnings

import numpy as np
from scipy.special import legendre

from kika._constants import (
    FWHM_TO_SIGMA as _FWHM_TO_SIGMA,
    NEUTRON_MASS_MEV,
    SPEED_OF_LIGHT_M_NS,
)
from kika.algebra import fold_tabulated


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnergyFoldingConfig:
    """
    Time-of-flight parameters of a measurement, from which the energy resolution follows.

    Passed as ``config`` to :func:`compute_energy_resolution_tof` and
    :func:`compute_folded_differential_xs`. The fold itself has nothing to configure:
    it is exact, so there is no window or sample count.

    Attributes
    ----------
    flight_path_m : float
        Flight path length in metres. Default 27.037.
    delta_t_ns : float
        Time resolution in nanoseconds, as a full width at half maximum (FWHM), not a
        standard deviation. Default 5.0.
    """
    flight_path_m: float = 27.037  # GELINA default
    delta_t_ns: float = 5.0        # Time resolution FWHM (not sigma!)


# =============================================================================
# TOF Energy Resolution
# =============================================================================

# Re-exported from kika._constants so existing `from kika.utils import
# FWHM_TO_SIGMA` keeps working.  The value is now the exact 2*sqrt(2*ln 2)
# rather than the 6-digit 2.35482 this module used to carry.
FWHM_TO_SIGMA = _FWHM_TO_SIGMA


def tof_energy_resolution(
    energy_mev: float,
    *,
    flight_path_m: float,
    delta_t_ns: float,
    delta_t_is_fwhm: bool = True,
) -> float:
    r"""TOF energy resolution :math:`\sigma_E` in MeV.

    Time-of-flight relates energy spread to timing spread by
    :math:`\Delta E / E = 2\,\Delta t / t`, with
    :math:`t = L / v` and :math:`v = c\sqrt{2E/m_n}` (non-relativistic; the
    correction is below 1% up to ~10 MeV).

    This function is the single definition of that formula. The result is a standard
    deviation in MeV (absolute, not relative); the two readings of ``delta_t_ns``
    differ by the factor ``FWHM_TO_SIGMA`` (see ``delta_t_is_fwhm``).

    Parameters
    ----------
    energy_mev : float
        Neutron energy in MeV.
    flight_path_m : float
        Flight path in metres.
    delta_t_ns : float
        Timing spread in nanoseconds.  See ``delta_t_is_fwhm`` for what it means.
    delta_t_is_fwhm : bool, default True
        Whether ``delta_t_ns`` is a **FWHM** (the usual experimental convention:
        pulse widths and detector timing are normally quoted that way) or
        already a standard deviation.  When True the result is divided by
        ``FWHM_TO_SIGMA`` (:math:`2\sqrt{2\ln 2}`).  There is no safe default that
        suits both readings, so callers that care should pass this explicitly.

    Returns
    -------
    float
        :math:`\sigma_E` in MeV (a standard deviation, either way); 0.0 when
        ``energy_mev`` is not positive. The time of flight is ``flight_path_m``
        divided by the non-relativistic neutron speed, in ns.
    """
    if energy_mev <= 0.0:
        return 0.0
    velocity_m_per_ns = SPEED_OF_LIGHT_M_NS * np.sqrt(2.0 * energy_mev / NEUTRON_MASS_MEV)
    t_ns = flight_path_m / velocity_m_per_ns
    width_mev = energy_mev * 2.0 * delta_t_ns / t_ns
    return width_mev / FWHM_TO_SIGMA if delta_t_is_fwhm else width_mev


def compute_energy_resolution_tof(
    energy_mev: float,
    config: Optional[EnergyFoldingConfig] = None,
    *,
    flight_path_m: Optional[float] = None,
    delta_t_ns: Optional[float] = None,
) -> float:
    """
    Calculate TOF energy resolution σE in MeV.

    The energy resolution for time-of-flight measurements is derived from:
        ΔE/E = 2 × Δt/t  (TOF differential relation)

    where Δt is the FWHM of the time resolution (standard experimental convention).

    This function:
    1. Computes the FWHM of energy resolution: FWHM_E = E × 2 × Δt / t
    2. Converts to standard deviation: σE = FWHM_E / FWHM_TO_SIGMA (2.35482...)

    It is :func:`tof_energy_resolution` with ``delta_t_is_fwhm=True``, and the time of
    flight ``t`` is the flight path over the non-relativistic neutron speed.

    Parameters
    ----------
    energy_mev
        Neutron energy in MeV.
    config
        Source of the flight path and time resolution. None is a default
        :class:`EnergyFoldingConfig`.
    flight_path_m
        Flight path in metres. Overrides ``config.flight_path_m`` when given.
    delta_t_ns
        Time resolution in nanoseconds, the FWHM of the time response and not its
        standard deviation. Overrides ``config.delta_t_ns`` when given.

    Returns
    -------
    float
        Energy resolution σE, a standard deviation in MeV (absolute, not relative);
        0.0 when ``energy_mev`` is not positive.

    Examples
    --------
        >>> # Using config
        >>> config = EnergyFoldingConfig(flight_path_m=27.037, delta_t_ns=10.0)
        >>> sigma_E = compute_energy_resolution_tof(1.406, config)
        >>> print(f"σE = {sigma_E * 1000:.2f} keV")  # ~7.2 keV

        >>> # Using explicit parameters
        >>> sigma_E = compute_energy_resolution_tof(1.0, flight_path_m=27.037, delta_t_ns=5.0)
    """
    # Deprecated shim over tof_energy_resolution, which is now the single
    # definition of this formula.  Kept because six workspace notebooks import
    # this name directly.  Behaviour is unchanged (delta_t_ns is a FWHM here).
    if config is None:
        config = EnergyFoldingConfig()

    L = flight_path_m if flight_path_m is not None else config.flight_path_m
    dt = delta_t_ns if delta_t_ns is not None else config.delta_t_ns

    return tof_energy_resolution(
        energy_mev, flight_path_m=L, delta_t_ns=dt, delta_t_is_fwhm=True,
    )


# =============================================================================
# Cross Section Folding
# =============================================================================

def fold_cross_section(
    ace_data,
    target_energy_mev: float,
    mt: int,
    sigma_E_mev: float,
) -> Tuple[float, float]:
    """
    Compute energy-folded cross section using ACE data.

    Thin ACE adapter over :func:`kika.algebra.fold_tabulated`:

        σ_folded = ∫ σ(E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE

    Exact for the lin-lin interpolant of the ACE cross section (closed form,
    :func:`kika.algebra.fold_tabulated`), held at its end values past the table.

    Parameters
    ----------
    ace_data
        ACE data object, as returned by ``kika.read_ace()``.
    target_energy_mev
        Central energy E₀ of the fold, in MeV.
    mt
        Reaction MT number (e.g. 2 for elastic).
    sigma_E_mev
        Standard deviation σE of the Gaussian kernel, in MeV (absolute). Zero or
        negative gives the unfolded value.

    Returns
    -------
    tuple of float
        ``(folded_xs, unfolded_xs)`` in barns. The unfolded value is the table
        linearly interpolated at ``target_energy_mev`` and held at its end values past
        the table.
    """
    # Get cross section data from ACE
    xs_data = ace_data.cross_section.to_plot_data(mt=mt)
    energies_mev = np.asarray(xs_data.x)
    xs_values = np.asarray(xs_data.y)

    folded_xs = fold_tabulated(energies_mev, xs_values, target_energy_mev, sigma_E_mev)
    unfolded_xs = float(np.interp(target_energy_mev, energies_mev, xs_values))

    return folded_xs, unfolded_xs


# =============================================================================
# Angular Distribution Functions
# =============================================================================

def endf_angular_distribution(
    mu: np.ndarray,
    a_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized angular distribution from ENDF Legendre coefficients.

    ENDF format:
        f(μ) = (1/2) × Σ_{l=0}^{L} (2l+1) × a_l × P_l(μ)

    where a_0 = 1 (implicit) and a_coeffs = [a_1, a_2, ..., a_L]

    The result integrates to 1 over μ ∈ [-1, 1].

    Parameters
    ----------
    mu : array_like
        Cosine of the scattering angle.
    a_coeffs : array_like
        ENDF coefficients ``[a_1, a_2, ..., a_L]``; a_0 = 1 is implicit. An empty
        sequence gives the isotropic value 0.5.

    Returns
    -------
    numpy.ndarray
        f(μ), per unit μ, with the shape of ``mu`` (at least one-dimensional).
    """
    mu = np.atleast_1d(mu)
    
    # P_0(μ) = 1, a_0 = 1 by convention
    result = 0.5 * np.ones_like(mu, dtype=float)  # (2×0+1)/2 × 1 × P_0 = 0.5
    
    # Add higher orders: l = 1, 2, ..., L
    for l, a_l in enumerate(a_coeffs, start=1):
        P_l = legendre(l)(mu)
        result += 0.5 * (2 * l + 1) * a_l * P_l
    
    return result


def fold_angular_distribution(
    ace_data,
    target_energy_mev: float,
    mt: int,
    sigma_E_mev: float,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute energy-folded angular distribution shape using ACE data.

    This folds the angular distribution PDF (shape only, not cross section),
    through the ACE container's own fold: nodes on every incident energy the
    distribution tabulates, weights exact for its lin-lin interpolation in
    energy (:func:`kika.algebra.gaussian_fold_nodes`).

    Parameters
    ----------
    ace_data
        ACE data object, as returned by ``kika.read_ace()``.
    target_energy_mev
        Central energy of the fold, in MeV.
    mt
        Reaction MT number.
    sigma_E_mev
        Standard deviation σE of the Gaussian kernel in energy, in MeV (absolute).
    num_mu_points
        Number of points of the μ grid.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mu_grid, folded_pdf, unfolded_pdf)``, each of shape ``(num_mu_points,)``.
        ``mu_grid`` runs from -1 to 1 ascending; both PDFs are the neutron angular
        PDF per unit μ, interpolated linearly onto it. ``unfolded_pdf`` is the
        distribution at ``target_energy_mev``.
    """
    mu_grid = np.linspace(-1, 1, num_mu_points)

    def on_grid(cosines, values):
        order = np.argsort(np.asarray(cosines))
        return np.interp(mu_grid, np.asarray(cosines)[order], np.asarray(values)[order])

    cosines, folded = ace_data.angular_distributions._apply_energy_folding(
        mt, target_energy_mev, sigma_E_mev, ace_data, 'neutron', 0,
        num_mu_points, False, 'b',
    )
    folded_pdf = on_grid(cosines, folded)

    unfolded_data = ace_data.angular_distributions.to_plot_data(
        mt=mt,
        energy=target_energy_mev,
        ace=ace_data,
        interpolate=True,
        num_points=num_mu_points,
        normalize_to_xs=False,
    )
    unfolded_pdf = on_grid(unfolded_data.x, unfolded_data.y)

    return mu_grid, folded_pdf, unfolded_pdf


# =============================================================================
# Main Entry Point
# =============================================================================

FoldingMode = Literal["xs_only", "angular_only", "both"]


def compute_folded_differential_xs(
    ace_data,
    target_energy_mev: float,
    mt: int,
    config: Optional[EnergyFoldingConfig] = None,
    *,
    mode: FoldingMode = "xs_only",
    endf_coeffs: Optional[np.ndarray] = None,
    sigma_E_mev: Optional[float] = None,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute energy-folded differential cross section dσ/dΩ(μ).
    
    This is the main entry point for computing differential cross sections
    with TOF energy resolution effects.
    
    Modes:
        - "xs_only" (default): Fold cross section only, use unfolded angular shape
        - "angular_only": Fold angular distribution only, use unfolded XS
        - "both": Fold both cross section and angular distribution
    
    The result is ``dσ/dΩ = σ × f(μ) / (2π)``, with σ the folded cross section in
    "xs_only" and "both" (the unfolded one in "angular_only") and f(μ) the angular
    PDF per unit μ.

    Parameters
    ----------
    ace_data
        ACE data object, as returned by ``kika.read_ace()``.
    target_energy_mev
        Central energy of the fold, in MeV.
    mt
        Reaction MT number (e.g. 2 for elastic scattering).
    config
        Source of the flight path and time resolution (FWHM) from which σE is
        computed. None is a default :class:`EnergyFoldingConfig`. Unused for σE when
        ``sigma_E_mev`` is given; it is still returned in ``info["config"]``.
    mode : {"xs_only", "angular_only", "both"}
        What is folded; see above.
    endf_coeffs
        ENDF Legendre coefficients ``[a_1, ..., a_L]`` for the angular distribution.
        Used only in "xs_only" mode; None takes the ACE angular distribution at
        ``target_energy_mev``. In the other modes it is ignored.
    sigma_E_mev
        Standard deviation σE of the energy resolution, in MeV (absolute). None
        computes it with :func:`compute_energy_resolution_tof` at
        ``target_energy_mev`` from ``config``.
    num_mu_points
        Number of points of the μ grid.

    Returns
    -------
    mu_grid : numpy.ndarray
        cos(θ), ascending from -1 to 1, shape ``(num_mu_points,)``.
    dsigma_domega : numpy.ndarray
        dσ/dΩ in b/sr, shape ``(num_mu_points,)``.
    info : dict
        ``sigma_E_mev`` and ``sigma_E_kev`` (the σE used), ``folded_xs`` and
        ``unfolded_xs`` (barns; equal in "angular_only"), ``xs_ratio``
        (``folded_xs / unfolded_xs``, 1.0 when ``unfolded_xs`` is not positive),
        ``mode`` and ``config``.

    Examples
    --------
        >>> import kika
        >>> ace = kika.read_ace("path/to/file.ace")
        >>> 
        >>> # XS-only folding (default, recommended)
        >>> mu, dsigma, info = compute_folded_differential_xs(
        ...     ace, target_energy_mev=1.0, mt=2
        ... )
        >>> 
        >>> # Full folding with custom ENDF coefficients
        >>> mu, dsigma, info = compute_folded_differential_xs(
        ...     ace, target_energy_mev=1.0, mt=2,
        ...     mode="both",
        ...     endf_coeffs=[0.1, 0.05, 0.02],  # a1, a2, a3
        ... )
    """
    if config is None:
        config = EnergyFoldingConfig()
    
    # Compute energy resolution if not provided
    if sigma_E_mev is None:
        sigma_E_mev = compute_energy_resolution_tof(target_energy_mev, config)
    
    # Create μ grid
    mu_grid = np.linspace(-1, 1, num_mu_points)
    
    # Get cross sections (always needed for normalization)
    if mode in ("xs_only", "both"):
        folded_xs, unfolded_xs = fold_cross_section(
            ace_data, target_energy_mev, mt, sigma_E_mev
        )
    else:
        # angular_only: no XS folding
        _, unfolded_xs = fold_cross_section(
            ace_data, target_energy_mev, mt, sigma_E_mev
        )
        folded_xs = unfolded_xs
    
    # Select which XS to use for normalization
    xs_for_norm = folded_xs if mode in ("xs_only", "both") else unfolded_xs
    
    # Get angular distribution
    if mode in ("angular_only", "both"):
        # Fold angular distribution
        _, angular_pdf, _ = fold_angular_distribution(
            ace_data, target_energy_mev, mt, sigma_E_mev, num_mu_points
        )
    else:
        # xs_only: use unfolded angular distribution
        if endf_coeffs is not None:
            # Use provided ENDF coefficients
            angular_pdf = endf_angular_distribution(mu_grid, endf_coeffs)
        else:
            # Use ACE angular distribution at target energy
            plot_data = ace_data.angular_distributions.to_plot_data(
                mt=mt,
                energy=target_energy_mev,
                ace=ace_data,
                interpolate=True,
                num_points=num_mu_points,
                normalize_to_xs=False,
            )
            # Ensure proper sorting for interpolation (np.interp requires increasing xp)
            ace_mu = np.asarray(plot_data.x)
            ace_pdf = np.asarray(plot_data.y)
            sort_idx = np.argsort(ace_mu)
            angular_pdf = np.interp(mu_grid, ace_mu[sort_idx], ace_pdf[sort_idx])
    
    # Compute differential cross section: dσ/dΩ = σ × f(μ) / (4π)
    # Since f(μ) integrates to 1 over μ ∈ [-1,1], and solid angle integral is 4π,
    # we have: dσ/dΩ = σ × f(μ) / (2) to get b/sr
    # Actually, f(μ) is per unit μ, so dσ/dΩ = σ × f(μ) / (2π)
    # Note: The factor depends on the normalization convention.
    # For ENDF: ∫ f(μ) dμ = 1, and dσ/dΩ = σ × f(μ) / (2π) gives b/sr
    dsigma_domega = xs_for_norm * angular_pdf / (2 * np.pi)
    
    # Build info dict
    info = {
        "sigma_E_mev": sigma_E_mev,
        "sigma_E_kev": sigma_E_mev * 1000,
        "folded_xs": folded_xs,
        "unfolded_xs": unfolded_xs,
        "xs_ratio": folded_xs / unfolded_xs if unfolded_xs > 0 else 1.0,
        "mode": mode,
        "config": config,
    }
    
    return mu_grid, dsigma_domega, info


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_unfolded_differential_xs(
    ace_data,
    target_energy_mev: float,
    mt: int,
    endf_coeffs: Optional[np.ndarray] = None,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute unfolded (raw) differential cross section dσ/dΩ(μ).
    
    This is equivalent to calling compute_folded_differential_xs with
    σE = 0, but more efficient as it skips the folding calculations.
    
    The result is ``dσ/dΩ = σ × f(μ) / (2π)``, with σ the cross section interpolated
    linearly at the energy and f(μ) the angular PDF per unit μ.

    Parameters
    ----------
    ace_data
        ACE data object, as returned by ``kika.read_ace()``.
    target_energy_mev
        Energy in MeV.
    mt
        Reaction MT number.
    endf_coeffs
        ENDF Legendre coefficients ``[a_1, ..., a_L]`` for the angular distribution.
        None takes the ACE angular distribution at ``target_energy_mev``.
    num_mu_points
        Number of points of the μ grid.

    Returns
    -------
    mu_grid : numpy.ndarray
        cos(θ), ascending from -1 to 1, shape ``(num_mu_points,)``.
    dsigma_domega : numpy.ndarray
        dσ/dΩ in b/sr, shape ``(num_mu_points,)``.
    info : dict
        ``sigma_E_mev`` and ``sigma_E_kev`` (0.0), ``folded_xs`` and ``unfolded_xs``
        (both the cross section in barns), ``xs_ratio`` (1.0) and ``mode``
        (``"unfolded"``). It has no ``config`` key.
    """
    mu_grid = np.linspace(-1, 1, num_mu_points)
    
    # Get cross section at target energy
    xs_data = ace_data.cross_section.to_plot_data(mt=mt)
    xs = np.interp(target_energy_mev, xs_data.x, xs_data.y)
    
    # Get angular distribution
    if endf_coeffs is not None:
        angular_pdf = endf_angular_distribution(mu_grid, endf_coeffs)
    else:
        plot_data = ace_data.angular_distributions.to_plot_data(
            mt=mt,
            energy=target_energy_mev,
            ace=ace_data,
            interpolate=True,
            num_points=num_mu_points,
            normalize_to_xs=False,
        )
        # Ensure proper sorting for interpolation (np.interp requires increasing xp)
        ace_mu = np.asarray(plot_data.x)
        ace_pdf = np.asarray(plot_data.y)
        sort_idx = np.argsort(ace_mu)
        angular_pdf = np.interp(mu_grid, ace_mu[sort_idx], ace_pdf[sort_idx])
    
    dsigma_domega = xs * angular_pdf / (2 * np.pi)
    
    info = {
        "sigma_E_mev": 0.0,
        "sigma_E_kev": 0.0,
        "folded_xs": xs,
        "unfolded_xs": xs,
        "xs_ratio": 1.0,
        "mode": "unfolded",
    }
    
    return mu_grid, dsigma_domega, info

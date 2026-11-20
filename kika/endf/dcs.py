r"""Differential cross sections from ENDF MF4 (angular) + MF3 (cross section).

The module holds every choice that turns the MF4 angular data and the MF3 cross
section into a differential cross section:

- the angular distribution :math:`f(\mu) = \sum_\ell \tfrac{2\ell+1}{2}\,a_\ell
  P_\ell(\mu)` from Legendre coefficients, with :math:`a_0 = 1` implicit,
- the interpolation of those coefficients **in energy** under the MF4
  ``(NBT, INT)`` law (lin-lin when none is given),
- the elastic LAB :math:`\leftrightarrow` CM frame transform,
- the three readings of :math:`\sigma(E)`: nominal, bin-averaged, TOF-folded.

The two products are :func:`differential_xs_vs_angle` (:math:`d\sigma/d\Omega`
against :math:`\mu` at fixed :math:`E`) and :func:`differential_xs_vs_energy`
(against :math:`E` at fixed :math:`\mu`).

Every function takes plain arrays, not an ENDF object.

Conventions
-----------
Energies are in **eV** unless a name says ``_mev``; cross sections in **barns**;
:math:`d\sigma/d\Omega` in **barn/sr** and :math:`d\sigma/d\mu` in **barn**.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from kika._constants import FWHM_TO_SIGMA, NEUTRON_MASS_AMU
from kika.processing.interpolation import interpolate_1d
from kika.utils.energy_folding import tof_energy_resolution
from kika.algebra import fold_tabulated, gaussian_fold_nodes, group_averages, interval_laws
from kika._legendre_frames import cos_cm_from_cos_lab, cos_lab_from_cos_cm, jacobian_cm_to_lab

__all__ = [
    "TofResolution",
    "legendre_basis",
    "angular_pdf",
    "angular_pdf_uncertainty",
    "frame_alpha",
    "cos_lab_from_cos_cm",
    "cos_cm_from_cos_lab",
    "jacobian_cm_to_lab",
    "transform_angular_curve",
    "interpolate_log_log",
    "bin_edges_for_energy",
    "bin_edges_for_width",
    "sigma_nominal",
    "sigma_bin_averaged",
    "sigma_folded",
    "resolve_sigma",
    "max_legendre_order",
    "legendre_provenance",
    "LegendreProvenance",
    "DEFAULT_PROJECTION_ORDER",
    "coefficients_at_energies",
    "coefficients_bin_averaged",
    "coefficients_folded",
    "resolve_coefficients",
    "resolution_fold_nodes",
    "coefficients_sigma_weighted_folded",
    "coefficients_sigma_weighted",
    "product_reading_nodes",
    "differential_xs_factor",
    "differential_xs_vs_angle",
    "differential_xs_vs_energy",
    "dsigma_vs_energy_grid",
]

Frame = Literal["lab", "cm"]
XsMode = Literal["nominal", "binavg", "folded"]
Weighting = Literal["lethargy", "constant"]


# =============================================================================
# TOF resolution
# =============================================================================

@dataclass(frozen=True)
class TofResolution:
    r"""Energy resolution of a time-of-flight spectrometer, for the folded readings.

    An instance gives the standard deviation :math:`\sigma_E` of the Gaussian
    kernel that the folded readings of :math:`\sigma` and :math:`a_\ell` average
    over. Build it by hand: from a flight path and a timing spread, or from a
    width the experiment declared.

    Parameters
    ----------
    flight_path_m : float
        Flight path :math:`L` in metres (27.037 m by default).
    delta_t_ns : float
        Timing spread :math:`\delta t` in nanoseconds (10 ns by default).
    delta_t_is_fwhm : bool
        True (default): ``delta_t_ns`` is a FWHM and is divided by
        ``kika._constants.FWHM_TO_SIGMA`` (about 2.355) to give a standard
        deviation. False: it is already a standard deviation. The two readings
        give kernels 2.355 times apart; there is no safe default for both, so
        pass it explicitly.
    min_sigma_e_kev : float
        Floor on :math:`\sigma_E`, in keV. The default 0 means no floor. The
        floor applies to every branch, a declared width included, and at
        non-positive energies as well.
    declared_fwhm_ev : float, optional
        A resolution the experiment declared itself, as a FWHM in eV constant in
        energy. When it is positive it replaces the flight-path geometry.
    declared_fwhm_fraction : float, optional
        The same as a fraction of the incident energy (0.01 is 1 %, not 1).
        Used only when ``declared_fwhm_ev`` is None or not positive, and only
        when positive itself.
    """

    flight_path_m: float = 27.037
    delta_t_ns: float = 10.0
    delta_t_is_fwhm: bool = True
    min_sigma_e_kev: float = 0.0

    #: A resolution the experiment declared itself, as a **FWHM in eV**,
    #: constant in energy.  When set it replaces the flight-path geometry: see
    #: the class note below.
    declared_fwhm_ev: Optional[float] = None

    #: The same, as a fraction of the incident energy (EXFOR's PER-CENT form).
    #: Ignored when :attr:`declared_fwhm_ev` is set.
    declared_fwhm_fraction: Optional[float] = None

    def sigma_e_mev(self, energy_mev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Standard deviation :math:`\sigma_E` of the kernel at the given energies.

        A positive declared width wins over the flight-path geometry. The two are
        not interchangeable: the TOF relation makes :math:`\sigma_E` grow as
        :math:`E^{3/2}`, whereas a declared resolution is quoted as a constant
        or as a fixed fraction of :math:`E`.

        Parameters
        ----------
        energy_mev : float or numpy.ndarray
            Incident energy in MeV.

        Returns
        -------
        float or numpy.ndarray
            :math:`\sigma_E` in MeV (a standard deviation, not a FWHM): a float
            for a scalar input, an array of the input's shape otherwise. It is 0
            at a non-positive energy, or when the geometry has a non-positive
            ``flight_path_m`` or ``delta_t_ns``, unless ``min_sigma_e_kev`` lifts it.
        """
        e = np.atleast_1d(np.asarray(energy_mev, dtype=float))
        out = np.empty_like(e)

        if self.declared_fwhm_ev is not None and self.declared_fwhm_ev > 0:
            out[:] = (self.declared_fwhm_ev / 1e6) / FWHM_TO_SIGMA
            out[e <= 0] = 0.0
        elif self.declared_fwhm_fraction is not None and self.declared_fwhm_fraction > 0:
            out[:] = np.where(e > 0, e * self.declared_fwhm_fraction / FWHM_TO_SIGMA, 0.0)
        else:
            for i, ei in enumerate(e):
                if ei <= 0 or self.flight_path_m <= 0 or self.delta_t_ns <= 0:
                    out[i] = 0.0
                else:
                    out[i] = tof_energy_resolution(
                        float(ei),
                        flight_path_m=self.flight_path_m,
                        delta_t_ns=self.delta_t_ns,
                        delta_t_is_fwhm=self.delta_t_is_fwhm,
                    )

        np.maximum(out, self.min_sigma_e_kev / 1000.0, out=out)
        return float(out[0]) if np.ndim(energy_mev) == 0 else out


# =============================================================================
# Angular reconstruction
# =============================================================================

def legendre_basis(mu: np.ndarray, max_order: int) -> np.ndarray:
    r"""Legendre polynomials :math:`P_l(\mu)` for :math:`l = 0 \ldots L`.

    Evaluated by Bonnet's recurrence
    :math:`(l+1)P_{l+1} = (2l+1)\mu P_l - l P_{l-1}`.

    Parameters
    ----------
    mu : array_like
        Cosines, in :math:`[-1, 1]`. A scalar is treated as one point.
    max_order : int
        Highest order :math:`L` returned.

    Returns
    -------
    np.ndarray
        Shape ``(max_order + 1, len(mu))``; row ``l`` is :math:`P_l` at every ``mu``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    basis = np.empty((max_order + 1, mu.size), dtype=float)
    basis[0] = 1.0
    if max_order >= 1:
        basis[1] = mu
    for l in range(2, max_order + 1):
        basis[l] = ((2 * l - 1) * mu * basis[l - 1] - (l - 1) * basis[l - 2]) / l
    return basis


def angular_pdf(mu: np.ndarray, a_coeffs: Sequence[float]) -> np.ndarray:
    r"""Normalized angular distribution :math:`f(\mu)` from ENDF coefficients.

    .. math::
        f(\mu) = \tfrac{1}{2} \sum_{l=0}^{L} (2l+1)\, a_l\, P_l(\mu)

    with :math:`a_0 = 1` implicit by the MF4 normalization convention, so
    ``a_coeffs`` is :math:`[a_1, a_2, \ldots, a_L]` and :math:`\int f\,d\mu = 1`.

    Equivalent to :func:`kika.utils.energy_folding.endf_angular_distribution`;
    this one evaluates the basis by recurrence instead of
    ``scipy.special.legendre``, which is faster when sweeping thousands of
    energies.

    Parameters
    ----------
    mu : array_like
        Cosines, in the frame the coefficients are given in. The function does
        not check the range.
    a_coeffs : sequence of float
        :math:`a_1 \ldots a_L`, without :math:`a_0`. An empty sequence gives the
        isotropic :math:`f = 1/2`.

    Returns
    -------
    np.ndarray
        :math:`f(\mu)` per unit cosine (the distribution normalised to 1 over
        :math:`[-1, 1]`), one value per ``mu``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    a = np.asarray(a_coeffs, dtype=float)
    basis = legendre_basis(mu, a.size)
    orders = np.arange(1, a.size + 1)
    result = 0.5 * basis[0]
    if a.size:
        result = result + 0.5 * ((2 * orders + 1) * a) @ basis[1:]
    return result


def angular_pdf_uncertainty(
    mu: np.ndarray,
    a_coeffs: Sequence[float],
    a_sigmas: Sequence[float],
) -> np.ndarray:
    r"""Diagonal (uncorrelated) uncertainty on :math:`f(\mu)`.

    .. math::
        \sigma_f(\mu)^2 = \sum_l \left[\tfrac{2l+1}{2} P_l(\mu)\right]^2
                          \sigma_{a_l}^2

    Diagonal-only: it ignores the MF34 off-diagonal blocks, so it is a display
    aid and not a defensible uncertainty on the reconstructed shape.

    Parameters
    ----------
    mu : array_like
        Cosines, in the frame the coefficients are given in.
    a_coeffs : sequence of float
        :math:`a_1 \ldots a_L`, without :math:`a_0`. Only its length is used.
    a_sigmas : sequence of float
        Absolute standard deviation of each coefficient, in the same order and of
        the same length as ``a_coeffs`` (the units of :math:`a_\ell`, which are
        dimensionless).

    Returns
    -------
    np.ndarray
        :math:`\sigma_f(\mu)`, one value per ``mu``, in the units of :math:`f`
        (per unit cosine). Zeros when the sequences are empty.

    Raises
    ------
    ValueError
        If ``a_coeffs`` and ``a_sigmas`` differ in length.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    a = np.asarray(a_coeffs, dtype=float)
    s = np.asarray(a_sigmas, dtype=float)
    if a.size != s.size:
        raise ValueError(f"a_coeffs ({a.size}) and a_sigmas ({s.size}) must have equal length")
    if a.size == 0:
        return np.zeros_like(mu)
    basis = legendre_basis(mu, a.size)
    orders = np.arange(1, a.size + 1)
    weights = (0.5 * (2 * orders + 1) * s) ** 2
    return np.sqrt(weights @ (basis[1:] ** 2))


# =============================================================================
# Elastic frame transform
# =============================================================================

def frame_alpha(*, awr: Optional[float] = None, mass_number: Optional[float] = None) -> float:
    r"""Mass ratio :math:`\alpha = m_n / m_{target}` for the elastic transform.

    Prefers the ENDF atomic weight ratio (:math:`AWR = m_{target}/m_n`, so
    :math:`\alpha = 1/AWR`); falls back to the mass number
    (:math:`\alpha \approx m_n / A`), good to ~0.1%.  Returns ``0.0`` when
    neither is known, which the functions that take ``alpha`` here treat as "no
    transform".

    Parameters
    ----------
    awr : float, optional
        ENDF atomic weight ratio of the target (target mass in neutron masses).
        Used when positive.
    mass_number : float, optional
        Target mass number :math:`A`. Used only when ``awr`` is missing or not
        positive; the neutron mass is taken as 1.00866 u.

    Returns
    -------
    float
        :math:`\alpha`, dimensionless; 0.0 when neither argument is positive.
    """
    if awr and awr > 0:
        return 1.0 / float(awr)
    if mass_number and mass_number > 0:
        return NEUTRON_MASS_AMU / float(mass_number)
    return 0.0


# `cos_lab_from_cos_cm` and `cos_cm_from_cos_lab` live in `kika._legendre_frames`,
# where the sampler's MF34 frame change uses them too; imported above.


def transform_angular_curve(
    mu: np.ndarray,
    y: np.ndarray,
    alpha: float,
    direction: Literal["cm2lab", "lab2cm"],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Move a per-solid-angle curve between the LAB and CM frames.

    A PDF :math:`f(\mu)` and a :math:`d\sigma/d\Omega` are both per-solid-angle
    densities, so the same Jacobian applies::

        cm2lab:  y_L(\mu_L) = y_C(\mu_C) · J,   \mu_L = cos_lab_from_cos_cm(\mu_C)
        lab2cm:  y_C(\mu_C) = y_L(\mu_L) / J,   \mu_C = cos_cm_from_cos_lab(\mu_L)

    **Elastic two-body kinematics only** (MT=2 with a neutron projectile).  For
    a discrete inelastic level the mapping also depends on the incident energy
    through the Q value, which this does not model.

    The returned cosine grid is generally non-uniform.  ``alpha = 0`` is a
    passthrough.

    Parameters
    ----------
    mu : array_like
        Cosines in the source frame (CM for ``"cm2lab"``, LAB for ``"lab2cm"``).
    y : array_like
        The density at each ``mu``, per steradian or per unit cosine (the
        Jacobian is the same), in any unit; it is multiplied or divided by
        :math:`J` and keeps its unit.
    alpha : float
        Mass ratio from :func:`frame_alpha`. Zero or negative returns copies of
        ``mu`` and ``y`` unchanged, whatever ``direction`` says.
    direction : {"cm2lab", "lab2cm"}
        The frame change.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The cosines in the target frame and the density there, both of the shape
        of ``mu``. For ``"cm2lab"`` the cosines are the images of ``mu`` and
        the density is ``y`` times :math:`J`. For ``"lab2cm"`` the cosines are
        the images of ``mu`` and the density is ``y`` divided by :math:`J`
        evaluated at the CM cosine.

    Raises
    ------
    ValueError
        If ``direction`` is neither string and ``alpha`` is positive.
    """
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    if not alpha or alpha <= 0:
        return mu.copy(), y.copy()
    if direction == "cm2lab":
        return cos_lab_from_cos_cm(mu, alpha), y * jacobian_cm_to_lab(mu, alpha)
    if direction == "lab2cm":
        mu_cm = cos_cm_from_cos_lab(mu, alpha)
        return mu_cm, y / jacobian_cm_to_lab(mu_cm, alpha)
    raise ValueError(f"direction must be 'cm2lab' or 'lab2cm', got {direction!r}")


# =============================================================================
# sigma(E)
# =============================================================================

def interpolate_log_log(x_grid, y_grid, xq):
    """Log-log interpolation of ``y(x)``, clamped to the endpoints.

    The ENDF convention for pointwise cross sections (INT=5).  Falls back to
    linear on any interval where a value is non-positive.

    Parameters
    ----------
    x_grid, y_grid : array_like
        The table. ``x_grid`` must be ascending; the function does not check it.
    xq : float or array_like
        Query abscissae, in the units of ``x_grid``. At or below ``x_grid[0]``
        the result is ``y_grid[0]``, at or above ``x_grid[-1]`` it is
        ``y_grid[-1]``.

    Returns
    -------
    float or numpy.ndarray
        ``y`` at ``xq``, in the units of ``y_grid``: a float for a scalar
        ``xq``, otherwise an array of its shape. An empty table gives NaN; a
        one-point table gives its single value.
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    scalar = np.ndim(xq) == 0
    q = np.atleast_1d(np.asarray(xq, dtype=float))
    if x.size == 0:
        out = np.full(q.shape, np.nan)
        return float(out[0]) if scalar else out
    if x.size == 1:
        out = np.full(q.shape, y[0])
        return float(out[0]) if scalar else out

    k = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[k], x[k + 1]
    y0, y1 = y[k], y[k + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(x1 == x0, 0.0, (q - x0) / (x1 - x0))
        out = y0 + t * (y1 - y0)
        log_ok = (x0 > 0) & (x1 > 0) & (q > 0) & (y0 > 0) & (y1 > 0)
        if np.any(log_ok):
            tt = (np.log(q[log_ok]) - np.log(x0[log_ok])) / (np.log(x1[log_ok]) - np.log(x0[log_ok]))
            out[log_ok] = np.exp(
                (1.0 - tt) * np.log(y0[log_ok]) + tt * np.log(y1[log_ok])
            )

    out = np.where(q <= x[0], y[0], out)
    out = np.where(q >= x[-1], y[-1], out)
    return float(out[0]) if scalar else out


def bin_edges_for_energy(energies: Sequence[float], index: int) -> Tuple[float, float]:
    """Bin ``[lo, hi]`` around ``energies[index]``, as midpoints to its neighbours.

    The first/last point takes a one-sided half-bin mirrored to the other side.
    A single-point grid yields a degenerate bin.

    Parameters
    ----------
    energies : sequence of float
        Ascending energy grid, in eV.
    index : int
        Position of the point whose bin is wanted, from 0 to ``len(energies) - 1``.
        A negative index is not supported: it is treated as the first point's
        branch of the edges.

    Returns
    -------
    (float, float)
        ``(lo, hi)`` in eV, with ``lo`` clipped at 0.

    Raises
    ------
    ValueError
        If ``energies`` is empty.
    """
    e = np.asarray(energies, dtype=float)
    n = e.size
    if n == 0:
        raise ValueError("energies is empty")
    ei = float(e[index])
    if n == 1:
        return ei, ei
    if index <= 0:
        hi = 0.5 * (e[0] + e[1])
        lo = ei - (hi - ei)
    elif index >= n - 1:
        lo = 0.5 * (e[n - 2] + e[n - 1])
        hi = ei + (ei - lo)
    else:
        lo = 0.5 * (e[index - 1] + ei)
        hi = 0.5 * (ei + e[index + 1])
    return max(float(lo), 0.0), float(hi)


BinWindow = Literal["mf4grid", "relative", "tof"]


def bin_edges_for_width(
    energy_ev: float,
    *,
    mode: BinWindow = "relative",
    relative_width: float = 0.02,
    tof: Optional["TofResolution"] = None,
    n_sigma: float = 1.0,
) -> Tuple[float, float]:
    r"""Bin ``[lo, hi]`` around ``energy_ev`` from an explicit width.

    The companion to :func:`bin_edges_for_energy`, which takes its width from
    the spacing of the MF4 incident-energy grid. That window is as coarse as the
    grid, and for a nuclide whose angular data is far sparser than its cross
    section it can span a wide range.

    Parameters
    ----------
    energy_ev : float
        Centre of the window, in eV. A non-positive value returns ``(0.0, 0.0)``.
    mode : {"relative", "tof", "mf4grid"}
        ``'relative'``: a fixed :math:`\Delta E / E` equal to ``relative_width``,
        centred on ``energy_ev``. ``'tof'``: ``n_sigma`` times the standard
        deviation :math:`\sigma_E` of the time-of-flight resolution at this
        energy on each side, so a bin average and a fold can be compared on
        equal terms. ``'mf4grid'`` needs the grid and is refused: call
        :func:`bin_edges_for_energy`.
    relative_width : float
        Full width of the window as a fraction of ``energy_ev`` (0.02 is 2 %).
        Used by ``'relative'`` only.
    tof : TofResolution, optional
        The resolution for ``'tof'``; a default :class:`TofResolution` when None.
        Used by ``'tof'`` only.
    n_sigma : float
        Half-width of the window in units of :math:`\sigma_E`. Used by ``'tof'``
        only.

    Returns
    -------
    (float, float)
        ``(lo, hi)`` in eV, with ``lo`` clipped at 0.

    Raises
    ------
    ValueError
        If ``mode`` is ``'mf4grid'``, or (for a positive ``energy_ev``) not one
        of the three strings.
    """
    if mode == "mf4grid":
        raise ValueError(
            "mode='mf4grid' takes its width from the energy grid; "
            "call bin_edges_for_energy(energies, index) instead"
        )
    e = float(energy_ev)
    if not (e > 0.0):
        return 0.0, 0.0
    if mode == "relative":
        half = 0.5 * e * float(relative_width)
    elif mode == "tof":
        resolution = tof or TofResolution()
        half = float(n_sigma) * resolution.sigma_e_mev(e / 1e6) * 1e6
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"unknown bin window mode {mode!r}")
    return max(e - half, 0.0), e + half


def _xs_pairs(xs_energies_ev, xs_interpolation):
    """The table's ``(NBT, INT)`` regions; ``None`` is one lin-lin region.

    Lin-lin is the default because it is what a PENDF and an ACE file state,
    and what an evaluation's MF3 states above the resonance region. A table
    with other laws passes its own regions.
    """
    n = len(xs_energies_ev)
    return list(xs_interpolation) if xs_interpolation else [(n, 2)]


def _sigma_at(xs_energies_ev, xs_values, energy_ev, xs_interpolation=None):
    """sigma at *energy_ev* under the table's own law, held at the ends."""
    return interpolate_1d(xs_energies_ev, xs_values,
                          _xs_pairs(xs_energies_ev, xs_interpolation),
                          energy_ev, out_of_range="hold")


def sigma_nominal(xs_energies_ev, xs_values, energy_ev, xs_interpolation=None):
    r"""Pointwise :math:`\sigma(E)`, interpolated under the table's own law.

    ``xs_interpolation`` is the table's ``(NBT, INT)`` pairs; ``None`` means
    one lin-lin region, which is what a PENDF or an ACE file states. Values
    outside the table hold its end values.

    Parameters
    ----------
    xs_energies_ev, xs_values
        The table: energies in eV, ascending, and :math:`\sigma` in barns.
    energy_ev : float or array_like
        Energies in eV at which to read :math:`\sigma`.
    xs_interpolation : sequence of (int, int), optional
        The table's ``(NBT, INT)`` regions. None is one lin-lin region.

    Returns
    -------
    float or numpy.ndarray
        :math:`\sigma` in barns at ``energy_ev``: a float for a scalar input,
        an array otherwise.
    """
    return _sigma_at(xs_energies_ev, xs_values, energy_ev, xs_interpolation)


def sigma_bin_averaged(
    xs_energies_ev,
    xs_values,
    e_lo_ev: float,
    e_hi_ev: float,
    weighting: Weighting = "lethargy",
    xs_interpolation=None,
) -> float:
    r"""Flux-averaged :math:`\sigma` over ``[e_lo, e_hi]``.

    .. math::
        \langle\sigma\rangle = \frac{\int \sigma(E) w(E)\,dE}{\int w(E)\,dE}

    Exact for the table under its own interpolation law
    (:func:`kika.algebra.group_averages`; ``xs_interpolation`` as in
    :func:`sigma_nominal`). ``weighting``: ``'lethargy'`` uses :math:`w = 1/E`
    (flux flat per unit lethargy, the neutronics default); ``'constant'`` uses
    :math:`w = 1`. The window is clipped to the table's domain, so the average
    is over the part the table covers. A degenerate window returns
    :math:`\sigma(e_{lo})`.

    The lethargy weight is not integrable at :math:`E = 0`, and the lowest bin
    of an MF4 grid does reach zero once its half-width is mirrored.  Both
    integrals diverge there while their ratio tends to :math:`\sigma` at the
    lower end, so such a bin is truncated at the first positive grid point
    inside it, and is not evaluated at the singularity.

    Parameters
    ----------
    xs_energies_ev, xs_values
        The table: energies in eV, ascending, and :math:`\sigma` in barns.
    e_lo_ev, e_hi_ev : float
        Window edges in eV.
    weighting : {"lethargy", "constant"}
        The flux weight :math:`w`: ``'lethargy'`` is :math:`1/E`, ``'constant'``
        is 1. Any other string is read as ``'constant'``.
    xs_interpolation : sequence of (int, int), optional
        The table's ``(NBT, INT)`` regions. None is one lin-lin region.

    Returns
    -------
    float
        The average, in barns. When the window does not overlap the table, or
        the average is not finite, :math:`\sigma` read nominally at the lower
        edge.
    """
    grid = np.asarray(xs_energies_ev, dtype=float)
    values = np.asarray(xs_values, dtype=float)
    if not (e_hi_ev > e_lo_ev):
        return float(sigma_nominal(grid, values, e_lo_ev, xs_interpolation))

    if weighting == "lethargy" and e_lo_ev <= 0.0:
        positive = grid[(grid > 0.0) & (grid < e_hi_ev)]
        e_lo_ev = float(positive[0]) if positive.size else e_hi_ev * 1e-6
        if not (e_hi_ev > e_lo_ev):
            return float(sigma_nominal(grid, values, e_hi_ev, xs_interpolation))

    lo, hi = max(float(e_lo_ev), float(grid[0])), min(float(e_hi_ev), float(grid[-1]))
    if not (hi > lo):
        return float(sigma_nominal(grid, values, e_lo_ev, xs_interpolation))
    laws = interval_laws(grid.size, _xs_pairs(grid, xs_interpolation))
    average = group_averages(grid, values, laws, [lo, hi],
                             weight="1/x" if weighting == "lethargy" else None)[0]
    return float(average) if np.isfinite(average) else float(
        sigma_nominal(grid, values, e_lo_ev, xs_interpolation))


def sigma_folded(xs_energies_ev, xs_values, energy_ev, tof: TofResolution):
    r"""TOF-resolution-folded :math:`\sigma(E)`.

    Averages :math:`\sigma` over a Gaussian kernel
    :math:`N(E_0, \sigma_E^2)`, exactly for the lin-lin interpolant of the
    table (:func:`kika.algebra.fold_tabulated`), with :math:`\sigma_E`
    from ``tof``.  Vectorized over ``energy_ev``.

    Note the folding samples :math:`\sigma` **linearly** (``numpy.interp``
    semantics), unlike :func:`sigma_nominal`, which honours the table's own
    law. The fold is an integral of the lin-lin interpolant, and log-log
    sampling inside it would bias the quadrature near zeros.

    Parameters
    ----------
    xs_energies_ev, xs_values
        The table: energies in eV, ascending, and :math:`\sigma` in barns.
    energy_ev : float or array_like
        Kernel centroids :math:`E_0` in eV.
    tof : TofResolution
        Gives the kernel's standard deviation :math:`\sigma_E` at each centroid.
        Where it is 0 the result is the table read at the centroid.

    Returns
    -------
    float or numpy.ndarray
        The folded :math:`\sigma` in barns: a float for a scalar ``energy_ev``,
        otherwise an array of its shape.
    """
    e0 = np.asarray(energy_ev, dtype=float)
    sigma_e_ev = np.asarray(tof.sigma_e_mev(e0 / 1e6), dtype=float) * 1e6
    return fold_tabulated(
        np.asarray(xs_energies_ev, dtype=float),
        np.asarray(xs_values, dtype=float),
        e0,
        sigma_e_ev,
    )


def resolve_sigma(
    xs_energies_ev,
    xs_values,
    energy_ev,
    *,
    mode: XsMode = "nominal",
    bin_edges: Optional[Tuple[float, float]] = None,
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
    xs_interpolation=None,
):
    r"""Dispatch to the requested reading of :math:`\sigma(E)`.

    ``'nominal'`` → :func:`sigma_nominal`; ``'binavg'`` → :func:`sigma_bin_averaged`
    over ``bin_edges``; ``'folded'`` → :func:`sigma_folded` with ``tof``.  A
    mode whose inputs are missing (``bin_edges`` or ``tof`` None), or a mode
    string that is none of the three, falls back to nominal and does not raise.

    Parameters
    ----------
    xs_energies_ev, xs_values
        The table: energies in eV, ascending, and :math:`\sigma` in barns.
    energy_ev : float or array_like
        Energies in eV. Not used by ``'binavg'``, which reads ``bin_edges``
        (one window, so a single value in the result).
    mode : {"nominal", "binavg", "folded"}
        The reading.
    bin_edges : (float, float), optional
        Window ``(lo, hi)`` in eV, for ``'binavg'``.
    tof : TofResolution, optional
        The resolution, for ``'folded'``.
    weighting : {"lethargy", "constant"}
        The flux weight of ``'binavg'``; see :func:`sigma_bin_averaged`.
    xs_interpolation : sequence of (int, int), optional
        The table's ``(NBT, INT)`` regions, for ``'nominal'`` and ``'binavg'``.
        ``'folded'`` always reads the table linearly.

    Returns
    -------
    float or numpy.ndarray
        :math:`\sigma` in barns, as the chosen function returns it.
    """
    if mode == "binavg" and bin_edges is not None:
        return sigma_bin_averaged(xs_energies_ev, xs_values, bin_edges[0], bin_edges[1],
                                  weighting, xs_interpolation)
    if mode == "folded" and tof is not None:
        return sigma_folded(xs_energies_ev, xs_values, energy_ev, tof)
    return sigma_nominal(xs_energies_ev, xs_values, energy_ev, xs_interpolation)


# =============================================================================
# Coefficients in energy
# =============================================================================

def max_legendre_order(section) -> int:
    r"""Highest Legendre order :math:`L` an MF4 section actually carries.

    Per ENDF-6, an MF4 LTT=1 LIST record holds exactly ``NL`` values,
    :math:`a_1 \ldots a_{NL}` — :math:`a_0 = 1` is the normalization convention
    and is **never stored**.  So the row length *is* the order, with no
    off-by-one to correct.  Sections vary ``NL`` with incident energy (a
    forward-peaked distribution at 45 MeV needs far more orders than a nearly
    isotropic one at thermal), so the answer is the maximum over the grid.

    Returns ``0`` for a section with no Legendre part, e.g. a purely isotropic
    one, which is the correct order for :math:`f(\mu) = 1/2`.

    Parameters
    ----------
    section
        An MF4 section. Its ``legendre_coefficients`` attribute (one row
        :math:`a_1 \ldots a_{NL}` per incident energy) is read first; if that is
        missing or empty, its ``num_legendre_coefficients`` count.

    Returns
    -------
    int
        The longest row, or the count, or 0 when neither is available.
    """
    rows = getattr(section, "legendre_coefficients", None)
    if rows:
        lengths = [len(r) for r in rows if r is not None and len(r)]
        if lengths:
            return int(max(lengths))
    # Some representations expose only a count rather than the rows.
    count = getattr(section, "num_legendre_coefficients", None)
    try:
        return max(0, int(count or 0))
    except (TypeError, ValueError):
        return 0


DEFAULT_PROJECTION_ORDER = 20
r"""Orders to offer for a distribution the file stores as :math:`f(\mu)`.

A tabulated section carries no :math:`a_\ell`, so any order shown for it is one
we computed by projection.  How far to go is a choice, not a property of the
file: the projection is defined for every :math:`\ell` and the coefficients
just get small.

Twenty, because the forward peak is what sets the answer and it sharpens with
energy.  Reconstructing JEFF-4.0 U-235 elastic from its own projection and
comparing against the stored 91-point table, worst error relative to the peak:

=========  ==========  ==========
energy     L = 10      L = 20
=========  ==========  ==========
0.1 MeV    0.00 %      0.00 %
1 MeV      0.03 %      0.04 %
10 MeV     9.66 %      0.26 %
18 MeV     25.73 %     0.47 %
=========  ==========  ==========

Ten is exact where the distribution is nearly flat and visibly wrong where it
is not.  Past 20 nothing moves: the half-percent left at 18 MeV is the table's
own kink structure, not a missing order.

Deliberately *not* measured per file at parse time: that would mean projecting
every tabulated section of every file the moment it is opened, on the read path
that already dominates the cost of opening a heavy evaluation.
"""


@dataclass(frozen=True)
class LegendreProvenance:
    r"""Where an MF4 section's :math:`a_\ell(E)` come from.

    A viewer built only around stored coefficients shows nothing for the half
    of the evaluations that tabulate :math:`f(\mu)` instead, and shows a mixed
    section as if its whole energy range were stored.  The provenance is carried
    next to the numbers so that neither is inferred. Get one from
    :func:`legendre_provenance`.

    Attributes
    ----------
    ltt
        The file's own representation flag: 0 isotropic, 1 Legendre,
        2 tabulated, 3 mixed.
    kind
        ``ltt`` as a name, for anything that has to say it out loud.
    stored_max_order
        Highest :math:`\ell` the file actually writes.  0 for a tabulated or
        isotropic section, which store no coefficients at all.
    max_order
        Highest :math:`\ell` that can be *asked* for -- stored where the file
        stores them, projected where it does not.  This is the one a list of
        available orders should be built from.
    projected
        ``"none"``, ``"all"``, or ``"above_boundary"`` for a mixed section,
        whose coefficients are read below ``boundary_energy`` and projected
        above it.
    boundary_energy
        Where a mixed section switches representation, in eV; None otherwise.
    """

    ltt: int
    kind: str
    stored_max_order: int
    max_order: int
    projected: str
    boundary_energy: Optional[float] = None

    @property
    def is_projected(self) -> bool:
        """True when some of the orders are computed by projection, not stored.

        It is ``projected != "none"``: true for ``"all"`` and ``"above_boundary"``.
        """
        return self.projected != "none"


_LTT_KINDS = {0: "isotropic", 1: "legendre", 2: "tabulated", 3: "mixed"}


def legendre_provenance(
    section, *, projection_order: int = DEFAULT_PROJECTION_ORDER
) -> LegendreProvenance:
    r"""Describe what :math:`a_\ell(E)` a section can produce, and from what.

    Pairs with :func:`max_legendre_order`, which stays a statement about the
    file: this one is a statement about what can be shown.  For a tabulated
    section the two differ by exactly the point of this function -- the file
    stores nothing, and the projection can still give you :math:`a_\ell` up to
    whatever order you ask for.

    Parameters
    ----------
    section
        An MF4 section. Its ``ltt`` attribute (missing or None is 0) selects the
        case; ``legendre_energies`` gives the boundary of a mixed section.
    projection_order : int
        Highest order offered for the projected part. For a tabulated section it
        is the ``max_order``; for a mixed one ``max_order`` is the larger of it
        and the stored maximum. An isotropic or Legendre section ignores it.

    Returns
    -------
    LegendreProvenance
        The file's LTT, its name, the highest stored and the highest askable
        order, which part is projected, and for a mixed section the last
        energy of the stored Legendre part in eV (None when
        ``legendre_energies`` is missing or empty).
    """
    ltt = int(getattr(section, "ltt", 0) or 0)
    stored = max_legendre_order(section)
    kind = _LTT_KINDS.get(ltt, "legendre" if stored else "isotropic")

    if ltt == 2:
        return LegendreProvenance(
            ltt=ltt,
            kind=kind,
            stored_max_order=0,
            max_order=int(projection_order),
            projected="all",
        )
    if ltt == 3:
        legendre_energies = getattr(section, "legendre_energies", None) or []
        boundary = float(legendre_energies[-1]) if legendre_energies else None
        return LegendreProvenance(
            ltt=ltt,
            kind=kind,
            stored_max_order=stored,
            max_order=max(stored, int(projection_order)),
            projected="above_boundary",
            boundary_energy=boundary,
        )
    # LTT=0 stores nothing but needs nothing: f(mu) = 1/2 is a_0 = 1 exactly,
    # and every higher order is zero rather than unknown.
    return LegendreProvenance(
        ltt=ltt,
        kind=kind,
        stored_max_order=stored,
        max_order=stored,
        projected="none",
    )


def _as_coefficient_matrix(
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    n_energies: int,
    max_order: Optional[int] = None,
) -> np.ndarray:
    """Normalize the app's ``{order: [a_l(E)]}`` wire shape to ``(L, n_energies)``.

    Order 0 is dropped when present: MF4 normalizes :math:`a_0 = 1`, and
    :func:`angular_pdf` puts it back implicitly.  Rows are indexed by
    :math:`l - 1`, and a missing order becomes a zero row rather than shifting
    the ones above it down.
    """
    if isinstance(coefficients, Mapping):
        orders = sorted(int(k) for k in coefficients.keys())
        orders = [o for o in orders if o >= 1]
        if max_order is not None:
            orders = [o for o in orders if o <= max_order]
        if not orders:
            return np.zeros((0, n_energies), dtype=float)
        top = max(orders)
        matrix = np.zeros((top, n_energies), dtype=float)
        for o in orders:
            row = np.asarray(
                coefficients.get(o, coefficients.get(str(o))), dtype=float
            )
            if row.size != n_energies:
                raise ValueError(
                    f"order {o} has {row.size} values but the energy grid has {n_energies}"
                )
            matrix[o - 1] = row
        return matrix

    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if matrix.shape[1] != n_energies:
        raise ValueError(
            f"coefficient matrix is {matrix.shape} but the energy grid has {n_energies} points"
        )
    if max_order is not None:
        matrix = matrix[:max_order]
    return matrix


def coefficients_at_energies(
    energies_ev: Sequence[float],
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    query_ev: Union[float, Sequence[float]],
    *,
    nbt_int_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""Interpolate :math:`a_l(E)` onto ``query_ev``, honouring the MF4 law.

    Outside the tabulated range the edge value is held; mask on
    ``energies_ev[0]`` and ``energies_ev[-1]`` to show a gap.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : mapping or numpy.ndarray
        Either ``{order: values}`` with integer or string keys and one value per
        grid energy (order 0 is dropped, a missing order below the highest
        becomes a row of zeros), or an array of shape ``(L, len(energies_ev))``
        whose row ``i`` is :math:`a_{i+1}`. The coefficients are the
        dimensionless :math:`a_\ell` of ``f(mu) = sum (2l+1)/2 a_l P_l(mu)``.
    query_ev : float or sequence of float
        Energies in eV at which to interpolate.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions of the incident-energy grid. None is
        lin-lin (INT=2) over the whole grid.
    max_order : int, optional
        Keep orders up to this one. None keeps all.

    Returns
    -------
    np.ndarray
        Shape ``(L, len(query_ev))``, rows indexed by :math:`l - 1`. With no
        orders, shape ``(0, len(query_ev))``.

    Raises
    ------
    ValueError
        If a row, or the array, does not have one value per grid energy.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = _as_coefficient_matrix(coefficients, grid.size, max_order)
    q = np.atleast_1d(np.asarray(query_ev, dtype=float))
    if matrix.shape[0] == 0:
        return np.zeros((0, q.size), dtype=float)

    pairs = list(nbt_int_pairs) if nbt_int_pairs else [(grid.size, 2)]
    out = np.empty((matrix.shape[0], q.size), dtype=float)
    for i, row in enumerate(matrix):
        out[i] = np.atleast_1d(
            interpolate_1d(grid, row, pairs, q, out_of_range="hold")
        )
    return out


# =============================================================================
# Plot products
# =============================================================================

def coefficients_folded(
    energies_ev,
    coefficients,
    query_ev,
    tof: TofResolution,
    *,
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""TOF-resolution-folded :math:`a_\ell(E)`.

    The angular counterpart of :func:`sigma_folded`: each coefficient is
    averaged over the same Gaussian energy-resolution kernel, on the same
    quadrature (the MF4 grid's own points inside the window).

    Why this exists.  Folding only :math:`\sigma` gives
    :math:`\langle\sigma\rangle\,f(\mu, E)` — the measured normalization with
    the evaluated *shape*.  A detector integrating over its own resolution sees
    the shape smeared too, and where the angular distribution changes quickly
    with energy the difference is not small.

    ``a_0 = 1`` identically, so it is neither stored nor folded; ``coefficients``
    is :math:`a_1 \ldots a_L` per energy, exactly as
    :func:`coefficients_at_energies` returns them.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : numpy.ndarray
        Array of shape ``(L, len(energies_ev))``, row ``i`` being :math:`a_{i+1}`.
        A mapping is not accepted here.
    query_ev : float or sequence of float
        Kernel centroids in eV.
    tof : TofResolution
        Gives the kernel's standard deviation at each centroid.
    nbt_int_pairs
        Not used: the fold reads each coefficient linearly between the grid
        points, whatever the MF4 law says.
    max_order : int, optional
        Keep orders up to this one. None keeps all.

    Returns
    -------
    np.ndarray
        Shape ``(L, len(query_ev))``.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    q = np.atleast_1d(np.asarray(query_ev, dtype=float))
    if max_order is not None:
        matrix = matrix[:max_order]

    sigma_e_ev = np.asarray(tof.sigma_e_mev(q / 1e6), dtype=float) * 1e6
    out = np.empty((matrix.shape[0], q.size), dtype=float)
    for i, row in enumerate(matrix):
        out[i] = np.atleast_1d(fold_tabulated(grid, row, q, sigma_e_ev))
    return out


def coefficients_bin_averaged(
    energies_ev,
    coefficients,
    bin_edges: Sequence[Tuple[float, float]],
    *,
    weighting: Weighting = "lethargy",
    max_order: Optional[int] = None,
    nbt_int_pairs=None,
) -> np.ndarray:
    r"""Flux-averaged :math:`a_\ell` over one window per query point.

    The angular counterpart of :func:`sigma_bin_averaged`, and averaged the same
    way: exactly, under the MF4 interpolation law ``nbt_int_pairs`` (lin-lin by
    default) and the same flux weight.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : numpy.ndarray
        Array of shape ``(L, len(energies_ev))``, row ``i`` being :math:`a_{i+1}`.
        A mapping is not accepted here.
    bin_edges : sequence of (float, float)
        One window ``(lo, hi)`` in eV per output point.
    weighting : {"lethargy", "constant"}
        The flux weight, as in :func:`sigma_bin_averaged`.
    max_order : int, optional
        Keep orders up to this one. None keeps all.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions of the incident-energy grid. None is
        lin-lin.

    Returns
    -------
    np.ndarray
        Shape ``(L, len(bin_edges))``.
    """
    grid = np.asarray(energies_ev, dtype=float)
    matrix = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if max_order is not None:
        matrix = matrix[:max_order]

    edges = list(bin_edges)
    out = np.empty((matrix.shape[0], len(edges)), dtype=float)
    for i, row in enumerate(matrix):
        for j, (lo, hi) in enumerate(edges):
            out[i, j] = sigma_bin_averaged(grid, row, lo, hi, weighting, nbt_int_pairs)
    return out


def resolve_coefficients(
    energies_ev,
    coefficients,
    query_ev,
    *,
    mode: XsMode = "nominal",
    bin_edges: Optional[Sequence[Tuple[float, float]]] = None,
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
) -> np.ndarray:
    r"""Dispatch to the requested reading of :math:`a_\ell(E)`.

    Deliberately the same three modes, the same argument names and the same
    fallback behaviour as :func:`resolve_sigma`, because the two are meant to
    be chosen independently:

    ======================  ======================  ==================================
    :math:`\sigma` mode     :math:`a_\ell` mode     result
    ======================  ======================  ==================================
    ``nominal``             ``nominal``             the evaluation as written
    ``folded``              ``nominal``             measured normalization, evaluated shape
    ``nominal``             ``folded``              evaluated normalization, smeared shape
    ``folded``              ``folded``              both, the *factor average*
    ======================  ======================  ==================================

    The last row is :math:`\langle\sigma\rangle\,F(\langle a_\ell\rangle)`, not
    :math:`\langle\sigma\,F(a_\ell)\rangle`.  The two differ by the
    :math:`\mathrm{Cov}(\sigma, a_\ell)` term across the window, and this module
    follows the factor average because MF3 :math:`\sigma` and MF4
    :math:`a_\ell` come from different underlying data — a product fold would
    impose a point-by-point energy correlation the evaluation does not carry.
    That is the same choice ``scripts/precompute_chi2_folded_al_c0.py`` makes,
    and the two must not disagree.

    **Deprecated** for any ``mode`` other than ``'nominal'``: it then issues a
    ``DeprecationWarning`` saying that it averages the coefficients apart from
    sigma, and to use :func:`coefficients_sigma_weighted` for what a
    measurement sees.  A detector counts :math:`\langle\sigma f\rangle`, and the
    evaluation states :math:`\sigma(E)` and :math:`f(\mu, E)` at every energy,
    so their product at each energy is what it asserts; averaging the factors
    apart assumes they do not vary together inside the window.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : mapping or numpy.ndarray
        As in :func:`coefficients_at_energies` for ``'nominal'``; an array of
        shape ``(L, len(energies_ev))`` for the two averaged modes.
    query_ev : float or sequence of float
        Energies in eV, for ``'nominal'`` and ``'folded'``. ``'binavg'`` reads
        ``bin_edges`` instead.
    mode : {"nominal", "binavg", "folded"}
        The reading. ``'binavg'`` needs ``bin_edges`` and ``'folded'`` needs
        ``tof``; without it the result is the nominal one.
    bin_edges : sequence of (float, float), optional
        One window ``(lo, hi)`` in eV per output point, for ``'binavg'``.
    tof : TofResolution, optional
        The resolution, for ``'folded'``.
    weighting : {"lethargy", "constant"}
        The flux weight of ``'binavg'``.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions, for ``'nominal'`` and ``'binavg'``; the
        fold ignores them.
    max_order : int, optional
        Keep orders up to this one. None keeps all.

    Returns
    -------
    np.ndarray
        Shape ``(L, n)`` with ``n`` the number of query energies, or of windows
        for ``'binavg'``.
    """
    if mode != "nominal":
        warnings.warn(
            "resolve_coefficients averages the coefficients apart from sigma; "
            "use coefficients_sigma_weighted for what a measurement sees",
            DeprecationWarning,
            stacklevel=2,
        )
    if mode == "binavg" and bin_edges is not None:
        return coefficients_bin_averaged(
            energies_ev, coefficients, bin_edges, weighting=weighting, max_order=max_order,
            nbt_int_pairs=nbt_int_pairs,
        )
    if mode == "folded" and tof is not None:
        return coefficients_folded(
            energies_ev, coefficients, query_ev, tof,
            nbt_int_pairs=nbt_int_pairs, max_order=max_order,
        )
    return coefficients_at_energies(
        energies_ev, coefficients, query_ev,
        nbt_int_pairs=nbt_int_pairs, max_order=max_order,
    )


# =============================================================================
# The sigma-weighted fold
# =============================================================================

def resolution_fold_nodes(
    energy_ev: float,
    sigma_e_ev: float,
    grids: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Quadrature nodes and weights for a Gaussian fold over tabulated data.

    :func:`kika.algebra.gaussian_fold_nodes` in eV: every point of every
    grid inside :math:`E_0 \pm 6\sigma_E` and the window edges, with weights
    exact for the integrand's lin-lin interpolant between them.  The same
    integral :func:`sigma_folded` and :func:`coefficients_folded` take through
    :func:`~kika.algebra.fold_tabulated`.

    Parameters
    ----------
    energy_ev : float
        Kernel centroid :math:`E_0` in eV.
    sigma_e_ev : float
        Standard deviation :math:`\sigma_E` of the kernel in eV.
    grids : sequence of sequence of float
        The tables whose points the nodes include, in eV.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The nodes in eV and their weights, which sum to 1. For
        ``sigma_e_ev <= 0`` the single node ``energy_ev`` with weight 1, whatever
        ``grids`` holds. With ``sigma_e_ev > 0``, empty ``grids`` is refused with
        the error of :func:`kika.algebra.gaussian_fold_nodes`.
    """
    return gaussian_fold_nodes(energy_ev, sigma_e_ev, grids)


def product_reading_nodes(
    energy_ev: float,
    mode: XsMode,
    grids: Sequence[Sequence[float]],
    *,
    tof: Optional[TofResolution] = None,
    bin_edges: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Nodes and normalised weights for one reading of a product such as :math:`\sigma f`.

    - ``'folded'``: :func:`resolution_fold_nodes` with :math:`\sigma_E` from
      ``tof``, a Gaussian over :math:`E_0 \pm 6\sigma_E`.
    - ``'binavg'``: the window ``bin_edges`` under a flux flat per unit lethargy
      (:math:`w = 1/E`), on its edges and every grid point strictly inside,
      trapezoid rule.  A window reaching :math:`E \le 0` is cut at the first
      positive grid point inside it, as :func:`sigma_bin_averaged` does.
    - ``'nominal'``, or a mode whose inputs are missing: the single node
      :math:`E_0`.

    Whatever the reading, the average of a product is :math:`\sum_i w_i\,
    \sigma(E_i)\,f(E_i)`, so the resolution acts on what a detector counts and
    not on the factors one at a time.

    Parameters
    ----------
    energy_ev : float
        The nominal energy :math:`E_0` in eV.
    mode : {"nominal", "binavg", "folded"}
        The reading. ``'folded'`` needs ``tof`` and ``'binavg'`` needs
        ``bin_edges``; without it the result is the single node of ``'nominal'``.
    grids : sequence of sequence of float
        The tables of the product (cross-section grid and MF4 grid), in eV. Their
        points become the nodes.
    tof : TofResolution, optional
        The resolution, for ``'folded'``.
    bin_edges : (float, float), optional
        The window ``(lo, hi)`` in eV, for ``'binavg'``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The nodes in eV and their weights, which sum to 1.
    """
    e0 = float(energy_ev)
    if mode == "folded" and tof is not None:
        sigma_e = float(tof.sigma_e_mev(e0 / 1e6)) * 1e6
        return resolution_fold_nodes(e0, sigma_e, grids)
    if mode == "binavg" and bin_edges is not None:
        lo, hi = float(bin_edges[0]), float(bin_edges[1])
        parts = []
        for grid in grids:
            g = np.asarray(grid, dtype=float)
            if g.size:
                parts.append(g[(g > lo) & (g < hi)])
        inside = np.unique(np.concatenate(parts)) if parts else np.zeros(0)
        if lo <= 0.0:
            positive = inside[inside > 0.0]
            lo = float(positive[0]) if positive.size else hi * 1e-6
            inside = inside[inside > lo]
        if hi > lo:
            nodes = np.unique(np.concatenate(([lo], inside, [hi])))
            w = 1.0 / nodes
            trap = np.zeros_like(nodes)
            spans = np.diff(nodes)
            trap[:-1] += 0.5 * spans
            trap[1:] += 0.5 * spans
            weights = w * trap
            total = weights.sum()
            if total > 0:
                return nodes, weights / total
    return np.array([e0]), np.array([1.0])


def coefficients_sigma_weighted(
    energies_ev,
    coefficients,
    xs_energies_ev,
    xs_values,
    energy_ev: float,
    *,
    mode: XsMode = "folded",
    tof: Optional[TofResolution] = None,
    bin_edges: Optional[Tuple[float, float]] = None,
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
    xs_interpolation=None,
) -> Tuple[np.ndarray, float]:
    r"""The angular shape and cross section a resolution-limited measurement sees.

    :func:`coefficients_sigma_weighted_folded` for any reading of
    :func:`product_reading_nodes`: :math:`a_\ell^\mathrm{eff} =
    \langle\sigma a_\ell\rangle / \langle\sigma\rangle` and
    :math:`\langle\sigma\rangle`, so :math:`d\sigma/d\Omega(\mu) =
    \langle\sigma\rangle f_\mathrm{eff}(\mu)/2\pi = \langle\sigma f\rangle/2\pi`.

    :math:`\sigma` is read linearly inside a fold (``numpy.interp``, as
    :func:`sigma_folded`) and under the table's own law otherwise
    (``xs_interpolation``, as :func:`sigma_nominal`; lin-lin by default).
    Nominal, it is the file at :math:`E_0`.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : mapping or numpy.ndarray
        :math:`a_1 \ldots a_L` on that grid, as in :func:`coefficients_at_energies`.
    xs_energies_ev, xs_values
        The cross-section table: energies in eV, ascending, and :math:`\sigma`
        in barns.
    energy_ev : float
        The nominal energy :math:`E_0` in eV.
    mode : {"nominal", "binavg", "folded"}
        The reading of :func:`product_reading_nodes`. The ``'binavg'`` weight is
        always :math:`1/E`. ``'folded'`` needs ``tof`` and ``'binavg'`` needs
        ``bin_edges``; without it the reading is nominal.
    tof : TofResolution, optional
        The resolution, for ``'folded'``.
    bin_edges : (float, float), optional
        The window ``(lo, hi)`` in eV, for ``'binavg'``.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions of ``energies_ev``. None is lin-lin.
    max_order : int, optional
        Keep orders up to this one. None keeps all.
    xs_interpolation : sequence of (int, int), optional
        The table's ``(NBT, INT)`` regions. None is one lin-lin region. Not used
        by a fold.

    Returns
    -------
    (numpy.ndarray, float)
        The effective coefficients :math:`a_1^\mathrm{eff} \ldots a_L^\mathrm{eff}`
        (shape ``(L,)``) and :math:`\langle\sigma\rangle` in barns. When
        :math:`\langle\sigma\rangle` is not positive, the coefficients are the
        mean over the nodes with the reading's own quadrature weights (Gaussian
        or the :math:`1/E` trapezoid), not weighted by :math:`\sigma`.
    """
    grid = np.asarray(energies_ev, dtype=float)
    xs_grid = np.asarray(xs_energies_ev, dtype=float)
    xs = np.asarray(xs_values, dtype=float)
    nodes, weights = product_reading_nodes(
        energy_ev, mode, [xs_grid, grid], tof=tof, bin_edges=bin_edges,
    )
    folded = mode == "folded" and tof is not None
    sigma_at = (
        np.interp(nodes, xs_grid, xs) if folded
        else np.atleast_1d(_sigma_at(xs_grid, xs, nodes, xs_interpolation))
    )
    a_at = coefficients_at_energies(
        grid, coefficients, nodes, nbt_int_pairs=nbt_int_pairs, max_order=max_order,
    )
    sigma_avg = float(np.sum(weights * sigma_at))
    if not (sigma_avg > 0):
        # No counts anywhere in the window: the shape is undefined, and the
        # unweighted mean is the only answer that is not a division by zero.
        return a_at @ weights, sigma_avg
    return (a_at @ (weights * sigma_at)) / sigma_avg, sigma_avg


def coefficients_sigma_weighted_folded(
    energies_ev,
    coefficients,
    xs_energies_ev,
    xs_values,
    energy_ev: float,
    tof: TofResolution,
    *,
    nbt_int_pairs=None,
    max_order: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    r"""What a resolution-limited measurement of :math:`d\sigma/d\Omega` sees.

    A detector at nominal :math:`E_0` counts

    .. math::
        \left\langle\frac{d\sigma}{d\Omega}\right\rangle(\mu, E_0)
        = \frac{1}{2\pi}\int R(E; E_0)\,\sigma(E)\,f(\mu, E)\,dE
        = \frac{\langle\sigma\rangle}{2\pi}\,f_\mathrm{eff}(\mu, E_0),

    and because :math:`f` is linear in the :math:`a_\ell`, the effective shape
    is a Legendre series with the **sigma-weighted** mean coefficients

    .. math::
        a_\ell^\mathrm{eff}(E_0) = \frac{\langle\sigma\,a_\ell\rangle}{\langle\sigma\rangle}.

    This is the product fold.  :func:`resolve_coefficients` deliberately does
    not do it -- its ``folded`` mode averages :math:`a_\ell` unweighted, the
    factor average the chi-square scripts are built on -- so this is a separate
    function and not a fourth mode there.  The two agree where :math:`\sigma`
    is flat across the window and part where it is not, which above the
    resolved range of a structural material is almost everywhere below a few
    MeV: the counts come from the energies where :math:`\sigma` is large.

    :math:`\sigma(E)` is sampled linearly between its table points, as in
    :func:`sigma_folded`, and :math:`a_\ell(E)` under the MF4 law, as in
    :func:`coefficients_at_energies`.  The quadrature is
    :func:`resolution_fold_nodes` on the union of both grids.

    Parameters
    ----------
    energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    coefficients : mapping or numpy.ndarray
        :math:`a_1 \ldots a_L` on that grid, as in :func:`coefficients_at_energies`.
    xs_energies_ev, xs_values
        The cross-section table: energies in eV, ascending, and :math:`\sigma`
        in barns.
    energy_ev : float
        The nominal energy :math:`E_0` in eV.
    tof : TofResolution
        Gives the kernel's standard deviation :math:`\sigma_E` at ``energy_ev``.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions of ``energies_ev``. None is lin-lin.
    max_order : int, optional
        Keep orders up to this one. None keeps all.

    Returns
    -------
    (np.ndarray, float)
        :math:`a_1^\mathrm{eff} \ldots a_L^\mathrm{eff}` and :math:`\langle\sigma\rangle`
        in barns.
    """
    return coefficients_sigma_weighted(
        energies_ev, coefficients, xs_energies_ev, xs_values, energy_ev,
        mode="folded", tof=tof, nbt_int_pairs=nbt_int_pairs, max_order=max_order,
    )


def differential_xs_factor(sigma, per_steradian: bool):
    r"""Scale turning :math:`f(\mu)` into a differential cross section.

    :math:`d\sigma/d\Omega = \sigma/(2\pi)\,f(\mu)` [barn/sr] when
    ``per_steradian``, else :math:`d\sigma/d\mu = \sigma\,f(\mu)` [barn].
    Both follow from :math:`\int f\,d\mu = 1`.

    Parameters
    ----------
    sigma : float or array_like
        Cross section in barns.
    per_steradian : bool
        True for :math:`d\sigma/d\Omega` (divides by :math:`2\pi`), False for
        :math:`d\sigma/d\mu`.

    Returns
    -------
    numpy.ndarray
        ``sigma / (2*pi)`` (barn per steradian, since :math:`f` is per unit
        cosine) or ``sigma`` (barn), of the shape of ``sigma``; 0-d for a scalar.
    """
    return (np.asarray(sigma, dtype=float) / (2.0 * np.pi)) if per_steradian else np.asarray(sigma, dtype=float)


def differential_xs_vs_angle(
    mu: Sequence[float],
    a_coeffs: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    *,
    pdf_values: Optional[Sequence[float]] = None,
    per_steradian: bool = True,
    alpha: float = 0.0,
    native_frame: Frame = "cm",
    output_frame: Optional[Frame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r""":math:`d\sigma/d\Omega` (or the bare PDF) against :math:`\mu`.

    ``mu`` is interpreted in ``native_frame`` — the frame the MF4 coefficients
    are given in, from the section's LCT flag.  With ``sigma=None`` the
    normalized PDF :math:`f(\mu)` (per unit cosine) is returned instead.

    When ``output_frame`` differs from ``native_frame`` the curve is moved with
    :func:`transform_angular_curve` (elastic only), so the returned cosine grid
    is non-uniform and is returned alongside the values.

    ``pdf_values`` short-circuits the reconstruction: a section that stores
    :math:`f(\mu)` as a table can hand its own values in and skip the round
    trip through a truncated expansion.

    Parameters
    ----------
    mu : sequence of float
        Cosines in ``native_frame``.
    a_coeffs : sequence of float, optional
        :math:`a_1 \ldots a_L` at the energy of interest, without :math:`a_0`.
        Not used when ``pdf_values`` is given.
    sigma : float, optional
        Cross section in barns. None returns the PDF with no scaling.
    pdf_values : sequence of float, optional
        :math:`f(\mu)` at ``mu``, same shape as ``mu``. It takes precedence over
        ``a_coeffs``.
    per_steradian : bool
        True: the scale is :math:`\sigma/2\pi` and the values are
        :math:`d\sigma/d\Omega` in barn/sr. False: :math:`d\sigma/d\mu` in barn.
        Ignored when ``sigma`` is None.
    alpha : float
        Mass ratio from :func:`frame_alpha`. The frame change needs it positive;
        with 0 the curve is returned in ``native_frame`` whatever
        ``output_frame`` says.
    native_frame : {"cm", "lab"}
        The frame of ``mu`` and of the coefficients.
    output_frame : {"cm", "lab"}, optional
        The frame of the result. None keeps ``native_frame``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The cosines (in the output frame) and the values there, both of the
        shape of ``mu``.

    Raises
    ------
    ValueError
        If neither ``a_coeffs`` nor ``pdf_values`` is given, or ``pdf_values`` does
        not have the shape of ``mu``.
    """
    mu = np.asarray(mu, dtype=float)
    if pdf_values is not None:
        values = np.asarray(pdf_values, dtype=float)
        if values.shape != mu.shape:
            raise ValueError(
                f"pdf_values has shape {values.shape}, expected {mu.shape} to match mu"
            )
    elif a_coeffs is not None:
        values = angular_pdf(mu, a_coeffs)
    else:
        raise ValueError("pass either a_coeffs or pdf_values")
    if sigma is not None:
        values = values * differential_xs_factor(sigma, per_steradian)

    if output_frame is not None and output_frame != native_frame and alpha > 0:
        direction = "cm2lab" if native_frame == "cm" else "lab2cm"
        return transform_angular_curve(mu, values, alpha, direction)
    return mu, values


def differential_xs_vs_energy(
    *,
    energies_ev: Sequence[float],
    coefficients: Union[Mapping[Union[int, str], Sequence[float]], np.ndarray],
    mu: float,
    xs_energies_ev: Optional[Sequence[float]] = None,
    xs_values: Optional[Sequence[float]] = None,
    mu_frame: Frame = "lab",
    native_frame: Frame = "cm",
    alpha: float = 0.0,
    per_steradian: bool = True,
    xs_mode: XsMode = "nominal",
    tof: Optional[TofResolution] = None,
    weighting: Weighting = "lethargy",
    query_energies_ev: Optional[Sequence[float]] = None,
    nbt_int_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_order: Optional[int] = None,
    pdf_at_energies: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    xs_interpolation: Optional[Sequence[Tuple[int, int]]] = None,
) -> dict:
    r""":math:`d\sigma/d\Omega` at a **fixed angle**, against incident energy.

    The complement of :func:`differential_xs_vs_angle`: pin :math:`\mu` and
    sweep :math:`E`.  Because the elastic frame relation depends only on the
    mass ratio and not on energy, a fixed ``mu`` in ``mu_frame`` maps to a
    single fixed cosine in ``native_frame``, evaluated once up front.

    Parameters
    ----------
    energies_ev, coefficients
        The MF4 incident-energy grid in eV, ascending, and its :math:`a_l(E)`
        as ``{order: [...]}`` or as an ``(L, n_energies)`` matrix (see
        :func:`coefficients_at_energies`). With ``pdf_at_energies`` the
        coefficients are not read.
    mu : float
        Target cosine, expressed in ``mu_frame``.
    xs_energies_ev, xs_values
        :math:`\sigma(E)`: energies in eV and cross section in barns.  Omit
        either to get the bare PDF :math:`f(\mu, E)`, which is the useful view
        when only the shape's energy dependence matters.
    mu_frame, native_frame : {"cm", "lab"}
        The frame of ``mu`` (default LAB) and the frame the coefficients are
        given in (default CM). When they differ and ``alpha`` is positive,
        ``mu`` is mapped once to ``mu_native`` and the density is multiplied by
        the Jacobian (``jacobian`` in the result). With ``alpha`` 0 no
        conversion is made, whatever the frames say.
    alpha : float
        Mass ratio from :func:`frame_alpha`. Elastic scattering only.
    per_steradian : bool
        True (default): :math:`d\sigma/d\Omega` in barn/sr. False:
        :math:`d\sigma/d\mu` in barn.
    xs_mode : {"nominal", "binavg", "folded"}
        The reading of :math:`\sigma`. The two averaged readings average the
        product :math:`\sigma(E)\,f(\mu, E)` over the window, which is what a
        detector at a fixed angle counts (:func:`product_reading_nodes`);
        ``sigma`` in the result is then :math:`\langle\sigma\rangle`.  A bin is
        the MF4 bin of the output energy, weighted by :math:`1/E`.  ``'folded'``
        without ``tof`` gives the nominal reading; without a cross-section table
        no reading applies. Pair a fold with :func:`dsigma_vs_energy_grid`
        for ``query_energies_ev``.
    tof : TofResolution, optional
        The resolution, for ``'folded'``.
    weighting : {"lethargy", "constant"}
        Not used: ``'binavg'`` is always weighted by :math:`1/E` here.
    query_energies_ev : sequence of float, optional
        Output grid in eV.  Defaults to the MF4 grid; a denser grid is
        interpolated through :func:`coefficients_at_energies` under
        ``nbt_int_pairs``, and the energies outside the MF4 range are dropped.
    nbt_int_pairs : sequence of (int, int), optional
        The MF4 ``(NBT, INT)`` regions of ``energies_ev``. None is lin-lin.
    max_order : int, optional
        Keep orders up to this one. None keeps all.
    pdf_at_energies : callable, optional
        ``f(mu_native, energies) -> array`` with one value per energy, for a
        section that stores the distribution itself: it replaces the
        coefficients, and ``nbt_int_pairs`` and ``max_order`` are then not read.
        The sigma modes, the Jacobian and the units are the same either way.
    xs_interpolation : sequence of (int, int), optional
        The sigma table's ``(NBT, INT)`` pairs. None is one lin-lin region,
        which is what a PENDF or an ACE file states; pass an evaluation's MF3
        regions to read it under its own laws. A fold reads the table linearly.

    Returns
    -------
    dict
        ``energies`` (the output grid, eV), ``values`` (in ``y_unit``, with the
        Jacobian applied), ``sigma`` (the :math:`\sigma(E)` used in barns, as an
        array on the output grid; :math:`\langle\sigma\rangle` for an averaged
        reading; None when no table was given), ``mu_requested`` and
        ``mu_native`` (the cosines in ``mu_frame`` and ``native_frame``),
        ``jacobian`` (1.0 when no frame change), ``frame``, ``native_frame``,
        ``xs_mode`` (as passed) and ``y_unit`` (``"barn/sr"``, ``"barn"``, or
        ``"Probability Density"`` for the bare PDF).

    Raises
    ------
    ValueError
        If ``energies_ev`` is empty, ``pdf_at_energies`` returns a different
        number of values from the energies, or a coefficient row does not have
        one value per grid energy.
    """
    grid = np.asarray(energies_ev, dtype=float)
    if grid.size == 0:
        raise ValueError("energies_ev is empty")

    # 1. Fixed angle: map once into the frame the coefficients live in.
    jacobian = 1.0
    mu_native = float(mu)
    if mu_frame != native_frame and alpha > 0:
        if native_frame == "cm":
            # `mu` is a LAB cosine; evaluate the CM-frame expansion at the
            # matching CM cosine, then push the density out with y_L = y_C · J.
            mu_native = float(cos_cm_from_cos_lab(mu, alpha))
            jacobian = float(jacobian_cm_to_lab(mu_native, alpha))
        else:
            # `mu` is already the CM cosine; the expansion is in LAB, so
            # evaluate there and pull the density back with y_C = y_L / J.
            mu_native = float(cos_lab_from_cos_cm(mu, alpha))
            jacobian = 1.0 / float(jacobian_cm_to_lab(float(mu), alpha))

    # 2. Output grid, clipped to where MF4 actually has data.
    if query_energies_ev is None:
        out_e = grid.copy()
    else:
        out_e = np.asarray(query_energies_ev, dtype=float)
        out_e = out_e[(out_e >= grid[0]) & (out_e <= grid[-1])]

    # 3. f(mu_native, E) at any set of energies.
    def pdf_at(energies: np.ndarray, on_grid: bool = False) -> np.ndarray:
        if pdf_at_energies is not None:
            # The section holds the distribution; ask it, and never build the
            # expansion at all.
            values = np.asarray(pdf_at_energies(mu_native, energies), dtype=float).ravel()
            if values.size != energies.size:
                raise ValueError(
                    f"pdf_at_energies returned {values.size} values for {energies.size} energies"
                )
            return values
        if on_grid:
            return pdf_on_grid()
        if linear_in_y:
            # f is linear in the a_l, and a law linear in y commutes with a
            # linear combination: interpolating f(mu, E) itself is exact, and
            # costs one row where the coefficients cost L. A fold reads ~200
            # energies per output point, so this is most of its time.
            return np.asarray(
                interpolate_1d(grid, pdf_on_grid(), pairs, energies, out_of_range="hold"),
                dtype=float,
            )
        return legendre_sum(coefficients_at_energies(
            grid, coefficients, energies, nbt_int_pairs=nbt_int_pairs, max_order=max_order
        ))

    def legendre_sum(coeffs: np.ndarray) -> np.ndarray:
        # One Legendre basis, reused across the sweep.
        basis = legendre_basis(np.array([mu_native]), coeffs.shape[0])[:, 0]
        orders = np.arange(1, coeffs.shape[0] + 1)
        return 0.5 * basis[0] + 0.5 * ((2 * orders + 1) * basis[1:]) @ coeffs

    grid_pdf: list = []

    def pdf_on_grid() -> np.ndarray:
        if not grid_pdf:
            grid_pdf.append(legendre_sum(_as_coefficient_matrix(coefficients, grid.size, max_order)))
        return grid_pdf[0]

    pairs = list(nbt_int_pairs) if nbt_int_pairs else [(grid.size, 2)]
    # INT 1-3 (histogram, lin-lin, lin-log) interpolate y linearly.
    linear_in_y = all(int(law) in (1, 2, 3) for _, law in pairs)

    has_xs = xs_energies_ev is not None and xs_values is not None and len(xs_values) > 0

    # 4a. An averaged reading: <sigma f> over the window, as one quantity.
    averaged = (xs_mode == "folded" and tof is not None) or xs_mode == "binavg"
    if averaged and has_xs:
        sigma, folded_product = _sigma_weighted_average_at_mu(
            np.asarray(xs_energies_ev, dtype=float),
            np.asarray(xs_values, dtype=float),
            grid,
            out_e,
            xs_mode,
            tof,
            pdf_at,
            xs_interpolation,
        )
        values = differential_xs_factor(folded_product, per_steradian) * jacobian
        return {
            "energies": out_e,
            "values": values,
            "sigma": sigma,
            "mu_requested": float(mu),
            "mu_native": mu_native,
            "jacobian": jacobian,
            "frame": mu_frame,
            "native_frame": native_frame,
            "y_unit": "barn/sr" if per_steradian else "barn",
            "xs_mode": xs_mode,
        }

    pdf = pdf_at(out_e, on_grid=query_energies_ev is None)

    # 4b. Nominal: sigma(E) at the energy.
    sigma = None
    if has_xs:
        sigma = np.atleast_1d(sigma_nominal(xs_energies_ev, xs_values, out_e,
                                            xs_interpolation))

    if sigma is None:
        values = pdf
        y_unit = "Probability Density"
    else:
        values = pdf * differential_xs_factor(sigma, per_steradian)
        y_unit = "barn/sr" if per_steradian else "barn"

    values = values * jacobian

    return {
        "energies": out_e,
        "values": values,
        "sigma": sigma,
        "mu_requested": float(mu),
        "mu_native": mu_native,
        "jacobian": jacobian,
        "frame": mu_frame,
        "native_frame": native_frame,
        "y_unit": y_unit,
        "xs_mode": xs_mode,
    }


def _sigma_weighted_average_at_mu(
    xs_grid: np.ndarray,
    xs: np.ndarray,
    mf4_grid: np.ndarray,
    energies_ev: np.ndarray,
    mode: XsMode,
    tof: Optional[TofResolution],
    pdf_at: Callable[[np.ndarray], np.ndarray],
    xs_interpolation=None,
) -> Tuple[np.ndarray, np.ndarray]:
    r""":math:`\langle\sigma\rangle` and :math:`\langle\sigma f\rangle` at one cosine, per energy.

    :func:`coefficients_sigma_weighted` at a single :math:`\mu`, for a whole
    sweep at once.  Every energy gets its own :func:`product_reading_nodes` on
    the union of the two grids (a bin is the MF4 bin of the energy); the nodes
    of all energies are concatenated so :math:`\sigma` and :math:`f` are each
    evaluated in one vectorised call, and the window sums are segment sums.
    """
    node_parts, weight_parts = [], []
    for e0 in energies_ev:
        edges = None
        if mode == "binavg":
            i = int(np.clip(np.searchsorted(mf4_grid, e0), 0, mf4_grid.size - 1))
            edges = bin_edges_for_energy(mf4_grid, i)
        nodes, weights = product_reading_nodes(
            float(e0), mode, (xs_grid, mf4_grid), tof=tof, bin_edges=edges,
        )
        node_parts.append(nodes)
        weight_parts.append(weights)
    if not node_parts:
        return np.zeros(0), np.zeros(0)
    counts = np.array([n.size for n in node_parts])
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    nodes = np.concatenate(node_parts)
    weights = np.concatenate(weight_parts)

    sigma_n = (
        np.interp(nodes, xs_grid, xs) if mode == "folded"
        else np.atleast_1d(_sigma_at(xs_grid, xs, nodes, xs_interpolation))
    )
    f_n = pdf_at(nodes)
    sigma_avg = np.add.reduceat(weights * sigma_n, starts)
    product = np.add.reduceat(weights * sigma_n * f_n, starts)
    return sigma_avg, product


def dsigma_vs_energy_grid(
    xs_energies_ev: Sequence[float],
    mf4_energies_ev: Sequence[float],
    *,
    tof: Optional[TofResolution] = None,
) -> np.ndarray:
    r"""The incident energies a :math:`d\sigma/d\Omega(E)` curve is drawn on.

    :math:`d\sigma/d\Omega(E) = \sigma(E)\,f(\mu, E)/2\pi`, and :math:`\sigma`
    has structure on a far finer grid than the few hundred energies MF4
    tabulates.  Drawn on the MF4 grid alone, the curve steps over every
    resonance between two of them.

    - **Nominal** (``tof=None``): the union of both grids, inside the MF4
      range.  Between two points both factors are the file's own
      interpolants, so this grid loses nothing.
    - **Folded**: the same union, thinned so consecutive points are at least
      :math:`\sigma_E/4` apart.  The folded curve is smooth on the scale of
      the kernel, so a point every quarter width resolves it, and where the
      table is coarser than that every table point is kept.

    Parameters
    ----------
    xs_energies_ev : sequence of float
        The cross-section grid in eV. Only the points inside the MF4 range are
        used.
    mf4_energies_ev : sequence of float
        The MF4 incident-energy grid in eV, ascending.
    tof : TofResolution, optional
        None for the nominal grid; with it, the thinned folded grid.

    Returns
    -------
    numpy.ndarray
        Ascending energies in eV, with the two ends of the MF4 grid always
        present; empty when ``mf4_energies_ev`` is empty.
    """
    xs_grid = np.asarray(xs_energies_ev, dtype=float)
    mf4 = np.asarray(mf4_energies_ev, dtype=float)
    if mf4.size == 0:
        return np.zeros(0)
    lo, hi = float(mf4[0]), float(mf4[-1])
    inside = xs_grid[(xs_grid >= lo) & (xs_grid <= hi)]
    union = np.unique(np.concatenate([mf4, inside]))
    if tof is None or union.size < 3:
        return union

    step = np.atleast_1d(np.asarray(tof.sigma_e_mev(union / 1e6), dtype=float)) * 1e6 / 4.0
    keep = np.zeros(union.size, dtype=bool)
    keep[0] = True
    next_allowed = union[0] + step[0]
    for i in range(1, union.size - 1):
        if union[i] >= next_allowed:
            keep[i] = True
            next_allowed = union[i] + step[i]
    keep[-1] = True
    return union[keep]

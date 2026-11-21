"""Covers: a foil inside an absorbing shell (cadmium, gadolinium, boron) sees less of the flux.

The transmission is a function of the cover's optical thickness ``tau = N sigma(E)``
(``N`` its atoms per barn, ``sigma`` the cross section the caller chooses: absorption,
or absorption plus a fraction of scattering). Two limits are offered, and the choice
is the caller's: ``exp(-tau)`` for a beam at normal incidence, the exponential
attenuation most dosimetry codes apply with an effective thickness, and ``E2(tau)``
for a thin foil inside a thin slab in an isotropic flux. Covers that need more (a
thick cover, a sphere, scattering in the cover) need a transport calculation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expn

from kika.algebra import evaluate, refine, sample_on_union, union, validate

from .response import Dosimeter

__all__ = ["COVER_TOLERANCE", "INCIDENCES", "covered", "cover_transmission"]

#: Relative tolerance of the lin-lin tables of a transmission and of a covered cross section.
COVER_TOLERANCE = 1e-3

#: A covered cross section is not refined where it is below this fraction of its maximum.
_NEGLIGIBLE = 1e-9


def _refined(x, y, value, tol):
    """*value* as lin-lin to *tol*, from nodes *x* whose values *y* keep each step's two sides."""
    x = np.asarray(x, dtype=float)
    floor = _NEGLIGIBLE * float(np.max(np.abs(y), initial=0.0)) + 1e-300

    def exceeds(actual, chord):
        return np.abs(actual - chord) / (tol * np.abs(actual) + floor)

    result = refine(x, y, lambda q, owner: value(q), exceeds, log_x=np.diff(x) > 0,
                    max_points=2_000_000)
    return result.x, result.y


#: The transmission of each incidence, as a function of the optical thickness.
INCIDENCES = {"normal": lambda tau: np.exp(-tau), "isotropic": lambda tau: expn(2, tau)}


def cover_transmission(energies, cross_section, laws, areal_density: float, incidence: str,
                       tol: float = COVER_TOLERANCE):
    """The fraction of the flux that reaches a thin foil inside a cover.

    Parameters
    ----------
    energies, cross_section, laws
        The cover's attenuating cross section (eV, barns; one ENDF law per interval
        or one for all), for an element its natural mixture. IRDFF-II gives the
        total, elastic and absorption cross sections of B, B-10, Cd and Gd for
        this.
    areal_density : float
        The cover's atoms per barn: number density in atoms/(b cm) times the
        thickness in cm (an effective thickness, for a code that fits one).
    incidence : {"normal", "isotropic"}
        ``"normal"``: ``exp(-tau)``, a beam through the cover. ``"isotropic"``:
        ``E2(tau)``, a thin slab cover in an isotropic flux.
    tol : float
        Relative tolerance of the lin-lin table.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, t)``, lin-lin, ``t`` between 0 and 1 for ``tau = areal_density x
        sigma``, on the cross section's grid refined to *tol*. Outside the
        table's energies the cover is taken as transparent.

    Raises
    ------
    ValueError
        If the table is not valid, *areal_density* is negative or *incidence* is
        not one of the two.
    """
    x, y, laws = validate(energies, cross_section, laws)
    if not areal_density >= 0:
        raise ValueError(f"areal_density must be zero or positive, got {areal_density!r}")
    if incidence not in INCIDENCES:
        raise ValueError(f"incidence must be one of {sorted(INCIDENCES)}, got {incidence!r}")
    transmit = INCIDENCES[incidence]

    def value(q):
        return transmit(areal_density * evaluate(x, y, laws, q))

    return _refined(x, transmit(areal_density * sample_on_union(x, y, laws, x)), value, tol)


def covered(dosimeter: Dosimeter, transmission, name: str = None,
            tol: float = COVER_TOLERANCE) -> Dosimeter:
    """The dosimeter of a covered foil: its cross section times the cover's transmission.

    Parameters
    ----------
    dosimeter : Dosimeter
        The bare reaction.
    transmission : tuple
        ``(x, t)`` lin-lin, from :func:`cover_transmission`; 1 outside its energies.
    name : str, optional
        The covered dosimeter's label; ``"<name> covered"`` by default.
    tol : float
        Relative tolerance of the lin-lin table of the product.

    Returns
    -------
    Dosimeter
        The covered response, lin-lin, with the bare dosimeter's covariance and
        *reaction*: an adjustment gives both the same
        cross-section parameters.

    Raises
    ------
    ValueError
        If *transmission* is not a valid lin-lin table.
    """
    tx, ty, _ = validate(*transmission, 2)
    sx, sy, slaws = dosimeter.energies, dosimeter.cross_section, dosimeter.laws

    def value(q):
        t = np.interp(q, tx, ty, left=1.0, right=1.0)
        return evaluate(sx, sy, slaws, q) * t

    grid = union([sx, tx[(tx > sx[0]) & (tx < sx[-1])]])
    start = sample_on_union(sx, sy, slaws, grid) * np.interp(grid, tx, ty, left=1.0, right=1.0)
    x, y = _refined(grid, start, value, tol)
    return Dosimeter(name if name is not None else f"{dosimeter.name} covered", x, y, 2,
                     dosimeter.covariance_edges, dosimeter.relative_covariance, dosimeter.reaction)

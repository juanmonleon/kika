"""Covers: a foil inside an absorbing shell (cadmium, gadolinium, boron) sees less of the flux.

For a thin foil inside a thin slab cover in an isotropic flux, the fraction of the
scalar flux that reaches the foil is ``E2(N sigma_a(E))``, with ``N`` the cover's
atoms per barn and ``sigma_a`` its absorption cross section (a scattered neutron is
not lost in an isotropic field, to first order). ``E2`` is the second exponential
integral; at normal incidence the fraction would be ``exp(-N sigma_a)``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expn

from kika.algebra import evaluate, refine, sample_on_union, union, validate

from .response import Dosimeter

__all__ = ["COVER_TOLERANCE", "covered", "cover_transmission"]

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


def cover_transmission(energies, cross_section, laws, areal_density: float,
                       tol: float = COVER_TOLERANCE):
    """The fraction of an isotropic flux that reaches a thin foil inside a slab cover.

    Parameters
    ----------
    energies, cross_section, laws
        The cover's absorption cross section (eV, barns; one ENDF law per interval
        or one for all), for an element its natural mixture (IRDFF-II gives Cd,
        Gd and B as covers, MT101).
    areal_density : float
        The cover's atoms per barn: number density in atoms/(b cm) times the
        thickness in cm.
    tol : float
        Relative tolerance of the lin-lin table.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, t)``, lin-lin: ``t = E2(areal_density x sigma_a)``, between 0 and 1,
        on the cross section's grid refined to *tol*. Outside the table's
        energies the cover is taken as transparent.

    Raises
    ------
    ValueError
        If the table is not valid or *areal_density* is negative.
    """
    x, y, laws = validate(energies, cross_section, laws)
    if not areal_density >= 0:
        raise ValueError(f"areal_density must be zero or positive, got {areal_density!r}")

    def value(q):
        return expn(2, areal_density * evaluate(x, y, laws, q))

    return _refined(x, expn(2, areal_density * sample_on_union(x, y, laws, x)), value, tol)


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
        :attr:`~Dosimeter.reaction`: an adjustment gives both the same
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

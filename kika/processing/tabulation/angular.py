"""Angular distributions P(mu|E) → tables in mu with their cumulative.

A Legendre series is evaluated exactly and refined in mu from ``[-1, 0, 1]``
until lin-lin interpolation reproduces it to the tolerance -- the shape ACER
gives: three points where the distribution is nearly isotropic, tens to a
hundred where it is forward-peaked. A tabulated distribution keeps its own
points (re-expressed lin-lin first if stated under another law). A
``regions2d`` (ENDF LTT=3) is flattened, the boundary energy taken once, from
the region above it. Negative densities follow
:attr:`TabulationOptions.negative`.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .tables import TabulationOptions, pdfTable, refine

__all__ = ["tabulateAngular", "legendreDensity", "muTable"]


def _m():
    from kika.nuclear_data import model

    return model


def legendreDensity(coefficients):
    """``p(mu) = sum_l (2l+1)/2 a_l P_l(mu)`` as a vectorised function."""
    from numpy.polynomial import legendre

    c = np.asarray(coefficients, dtype=float)
    scaled = c * (2.0 * np.arange(c.size) + 1.0) / 2.0
    return lambda mu: legendre.legval(np.asarray(mu, dtype=float), scaled)


def _linlinXY(function):
    """A tabulated function's nodes, refined to lin-lin if it is stated otherwise."""
    m = _m()
    if isinstance(function, m.XYs1d):
        x, y, law = function.xs, function.ys, function.interpolation
        if law in (m.Interpolation.linlin, m.Interpolation.flat):
            return x, y, law == m.Interpolation.flat
        regions = m.Regions1d(function1ds=[function])
    elif isinstance(function, m.Regions1d):
        regions = function
        laws = {r.interpolation for r in regions.function1ds}
        if laws == {m.Interpolation.linlin}:
            x, y, _ = regions.toEndfRegions()
            return x, y, False
        if laws == {m.Interpolation.flat}:
            x, y, _ = regions.toEndfRegions()
            return x, y, True
    else:
        raise TypeError(f"a {type(function).__name__} is not a tabulated function")
    from kika.algebra import interval_laws, to_linlin

    x, y, pairs = regions.toEndfRegions()
    lx, ly = to_linlin(x, y, interval_laws(len(x), pairs), tol=1e-4)
    return lx, ly, False


def muTable(function, options: TabulationOptions, outer=None, what="angular"):
    """One P(mu) (Legendre or tabulated) → ``(Xs_pdf_cdf1d, clipped)``."""
    m = _m()
    if isinstance(function, m.Legendre):
        density = legendreDensity(function.coefficients)
        # A degree-N series has up to N-1 extrema: 4N+1 seeds leave none of
        # them alone between two seeds for the refinement to miss.
        order = len(function.coefficients) - 1
        seeds = np.linspace(-1.0, 1.0, max(3, options.angularSeeds, 4 * order + 1))
        mu, p = refine(density, seeds, options)
        return pdfTable(mu, p, options=options, outer=outer, what=what)
    x, y, histogram = _linlinXY(function)
    return pdfTable(x, y, histogram=histogram, options=options, outer=outer, what=what)


def _functions(form) -> List:
    """The per-energy functions of an XYs2d or regions2d, boundary energies once."""
    m = _m()
    if isinstance(form, m.XYs2d):
        return list(form.function1ds)
    out: List = []
    for region in form.function2ds:
        for f in _functions(region):
            if out and f.outerDomainValue == out[-1].outerDomainValue:
                out[-1] = f
            else:
                out.append(f)
    return out


def tabulateAngular(form, options: TabulationOptions = TabulationOptions()) -> Tuple[object, int]:
    """P(mu|E) → ``(XYs2d of Xs_pdf_cdf1d, number of tables clipped)``.

    ``Isotropic2d`` and ``None`` come back as :class:`Isotropic2d`: the writer
    leaves the angle out and OpenMC samples mu uniformly.
    """
    m = _m()
    if form is None or isinstance(form, m.Isotropic2d):
        return m.Isotropic2d(), 0
    if not isinstance(form, (m.XYs2d, m.Regions2d)):
        raise TypeError(f"an angular distribution as a {type(form).__name__}")
    tables, clipped = [], 0
    for function in _functions(form):
        energy = function.outerDomainValue
        table, wasClipped = muTable(function, options, outer=energy, what=f"P(mu|E={energy:g} eV)")
        tables.append(table)
        clipped += wasClipped
    return m.XYs2d(function1ds=tables, label="MonteCarlo"), clipped

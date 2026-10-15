"""Angular distributions P(mu|E) → tables in mu with their cumulative.

A Legendre series is evaluated exactly and refined in mu from ``[-1, 0, 1]``
until lin-lin interpolation reproduces it to the tolerance -- the shape ACER
gives: three points where the distribution is nearly isotropic, tens to a
hundred where it is forward-peaked. A tabulated distribution keeps its own
points (re-expressed lin-lin first if stated under another law). A
``regions2d`` (ENDF LTT=3) is flattened with both sides of each boundary
retained, since their densities may differ. Negative densities follow
:attr:`TabulationOptions.negative`.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .tables import TabulationOptions, pdfTable, refine

from importlib import import_module
try:
    _polynomial_native = import_module("kika.algebra._polynomial_native")
except ModuleNotFoundError as exc:
    if exc.name != "kika.algebra._polynomial_native":raise
    _polynomial_native = None
if _polynomial_native is not None and _polynomial_native.API_VERSION != 1:
    raise ImportError("rebuild kika's native polynomial extension (API version mismatch)")

__all__ = ["tabulateAngular", "legendreDensity", "muTable"]


def _m():
    from kika.nuclear_data import model

    return model


def legendreDensity(coefficients):
    """``p(mu) = sum_l (2l+1)/2 a_l P_l(mu)`` as a vectorised function."""
    from numpy.polynomial import legendre

    c = np.asarray(coefficients, dtype=float)
    scaled = c * (2.0 * np.arange(c.size) + 1.0) / 2.0
    if _polynomial_native is None or scaled.ndim != 1 or not len(scaled):
        return lambda mu: legendre.legval(np.asarray(mu,dtype=float),scaled)
    scaled = np.ascontiguousarray(scaled)
    def density(mu):
        query=np.asarray(mu,dtype=float)
        points=np.ascontiguousarray(query).ravel()
        out=np.empty_like(points)
        if _polynomial_native.legendre_values(scaled,points,out):
            result=out.reshape(query.shape)
            return result[()] if query.ndim==0 else result
        return legendre.legval(query,scaled)
    return density


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


def _legendreMoments(coefficients, negative):
    """Exact polynomial integrals, split at real roots for positive clipping."""
    from numpy.polynomial import Legendre
    c = np.asarray(coefficients, dtype=float)
    if len(c) and np.all(np.isfinite(c)) and c[0] > 0:
        # |P_l(mu)| <= 1 proves positivity when the constant term dominates.
        # Orthogonality then gives both moments without finding any roots.
        bound = np.sum((2*np.arange(1,len(c))+1)*np.abs(c[1:]))
        if c[0] > bound*(1+64*np.finfo(float).eps):
            first = c[1]/c[0] if len(c)>1 else 0.
            second = 1/3+(2/3)*(c[2]/c[0] if len(c)>2 else 0.)
            return np.array([first,second])
    law = Legendre(c * (2*np.arange(len(c))+1)/2)
    roots = law.roots()
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
    edges = np.unique(np.r_[-1., real[(real > -1) & (real < 1)], 1.])
    values = np.zeros(3)
    mu = Legendre([0., 1.])
    for a, b in zip(edges[:-1], edges[1:]):
        if law((a+b)/2) < 0:
            if negative == "raise":
                raise ValueError("angular Legendre law has a negative interval")
            continue
        for power in range(3):
            integral = (law * mu**power).integ()
            values[power] += integral(b)-integral(a)
    if not np.isfinite(values).all() or values[0] <= 0:
        raise ValueError("angular Legendre law has no positive finite integral")
    return values[1:]/values[0]


def _tableMoments(table):
    a, b = np.asarray(table.xs[:-1]), np.asarray(table.xs[1:])
    pa, pb = np.asarray(table.pdf[:-1]), np.asarray(table.pdf[1:])
    dx = b-a
    first = np.sum(dx*(pa*(2*a+b)+pb*(a+2*b))/6)
    second = np.sum(dx*(pa*(3*a*a+2*a*b+b*b)+pb*(a*a+2*a*b+3*b*b))/12)
    return np.array([first, second])


def muTable(function, options: TabulationOptions, outer=None, what="angular"):
    """One P(mu) (Legendre or tabulated) → ``(Xs_pdf_cdf1d, clipped)``."""
    m = _m()
    if isinstance(function, m.Legendre):
        density = legendreDensity(function.coefficients)
        # Order-aware seeds reduce midpoint cancellations; the exact moment
        # check below remains the accuracy gate.
        degree = len(function.coefficients) - 1
        seeds = np.linspace(-1.0, 1.0, max(3, options.angularSeeds, 4 * degree + 1))
        target = _legendreMoments(function.coefficients, options.negative)
        mu, p = refine(density, seeds, options)
        for iteration in range(30):
            table, clipped = pdfTable(mu, p, options=options, outer=outer, what=what)
            if np.all(abs(_tableMoments(table)-target) <= options.angularMomentAtol):
                return table, clipped
            # A midpoint-only density check can miss cancellations in a
            # higher-order polynomial. Probe three interior fractions and
            # refine panels with the largest integrated shape discrepancy.
            fraction = np.array([.25, .5, .75])
            width = np.diff(mu)
            probes = mu[:-1,None] + width[:,None]*fraction
            actual = np.maximum(density(probes), 0.)
            endpoints = np.maximum(p, 0.)
            chord = endpoints[:-1,None] + np.diff(endpoints)[:,None]*fraction
            score = width*np.max(abs(actual-chord), axis=1)
            available = options.maxPoints-len(mu)
            if available <= 0:
                raise ValueError("angular moment refinement exceeds maxPoints")
            order = np.argsort(score)[::-1]
            count = max(1, int(np.searchsorted(np.cumsum(score[order]), .5*score.sum()))+1)
            selected = order[:min(count, available)]
            mu = np.sort(np.r_[mu, (mu[selected]+mu[selected+1])/2])
            p = density(mu)
        raise ValueError("angular moment refinement did not converge")
    x, y, histogram = _linlinXY(function)
    return pdfTable(x, y, histogram=histogram, options=options, outer=outer, what=what)


def _functions(form) -> List:
    """Per-energy functions in source order, including both sides of a jump."""
    m = _m()
    if isinstance(form, m.XYs2d):
        return list(form.function1ds)
    out: List = []
    for region in form.function2ds:
        for f in _functions(region):
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
    # OpenMC searches with lower_bound: a repeated coordinate selects the
    # left law at the exact boundary, while NumPy/model interpolation selects
    # the right one. Store the left limits on adjacent representable energies
    # so both consumers retain each side and choose the right law at the jump.
    # Only the processed table coordinates move; source functions are untouched.
    if any(a.outerDomainValue > b.outerDomainValue for a, b in zip(tables, tables[1:])):
        raise ValueError("angular incident energies must be nondecreasing")
    for i in range(len(tables) - 2, -1, -1):
        if tables[i].outerDomainValue >= tables[i + 1].outerDomainValue:
            tables[i].outerDomainValue = float(np.nextafter(
                tables[i + 1].outerDomainValue, -np.inf))
    return m.XYs2d(function1ds=tables, label="MonteCarlo"), clipped

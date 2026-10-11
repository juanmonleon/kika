"""Correlated energy-angle distributions → Monte Carlo tables.

- **Kalbach-Mann** (MF6 LAW=1 LANG=2): ``f`` becomes a normalised table with
  its cumulative, ``r`` stays on the same outgoing grid, and the slope ``a``
  is the evaluator's when given and Kalbach's systematics otherwise
  (:mod:`.kalbach`).
- **energyAngular** (MF6 LAW=1 LANG=1, NA > 0): at each (E, E') the series
  ``sum (2l+1)/2 f_l P_l(mu)``; ``f_0`` is the outgoing spectrum, and the
  angular table at E' is the series divided by it, tabulated in mu
  (``energyAngularMC``).
- **angularEnergy** (MF6 LAW=7, laboratory): mu outermost, the joint density
  f(mu, E') per cosine. Re-sorted to E' outermost: on the union of the
  cosines' outgoing grids, the spectrum is the integral over mu and the
  angular table at each E' the joint density at that E', normalised.
  OpenMC's exporter cannot write LAW=7 at all (``laboratory.py:141-142``);
  this is the re-expression the ``correlated`` layout admits.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .angular import _linlinXY, muTable
from .energy import mapFunction2d
from .kalbach import kalbachSlope
from .tables import TabulationOptions, pdfTable

__all__ = ["tabulateKalbachMann", "tabulateEnergyAngular", "tabulateAngularEnergy"]


def _m():
    from kika.nuclear_data import model

    return model


def _flat(form):
    m = _m()
    if isinstance(form, m.XYs2d):
        return list(form.function1ds)
    return [f for region in form.function2ds for f in _flat(region)]


def tabulateKalbachMann(form, options: TabulationOptions, *, zaEmitted: int, zaTarget: int,
                        discrete: Optional[Sequence[int]] = None):
    """KalbachMann(f, r[, a]) → KalbachMann of tables, ``r`` and ``a`` on f's grid."""
    m = _m()
    rs, as_ = _flat(form.r), (_flat(form.a) if form.a is not None else None)
    lines = {}

    def convert(f, k):
        x, y, histogram = _linlinXY(f)
        nd = int(discrete[k]) if discrete is not None and k < len(discrete) else 0
        table, _ = pdfTable(x, y, histogram=histogram, options=options, discrete=nd,
                            outer=f.outerDomainValue)
        lines[k] = (table, histogram)
        return table

    fTables = mapFunction2d(form.f, convert)

    def column(k, values):
        table, histogram = lines[k]
        law = m.Interpolation.flat if histogram else m.Interpolation.linlin
        return m.XYs1d(xs=table.xs, ys=values, interpolation=law,
                       outerDomainValue=table.outerDomainValue)

    def rAt(f, k):
        r = rs[k]
        rx, ry, _ = _linlinXY(r)
        return column(k, np.interp(lines[k][0].xs, rx, ry))

    def aAt(f, k):
        table = lines[k][0]
        if as_ is not None:
            ax, ay, _ = _linlinXY(as_[k])
            return column(k, np.interp(table.xs, ax, ay))
        return column(k, kalbachSlope(float(table.outerDomainValue), table.xs, zaEmitted, zaTarget))

    rForm = mapFunction2d(form.f, rAt)
    aForm = mapFunction2d(form.f, aAt)
    return m.KalbachMann(f=fTables, r=rForm, a=aForm, label="MonteCarlo",
                         productFrame=form.productFrame)


def _isotropicTable(outer):
    table, _ = pdfTable([-1.0, 1.0], [0.5, 0.5], outer=outer)
    return table


def tabulateEnergyAngular(form, options: TabulationOptions, *,
                          discrete: Optional[Sequence[int]] = None):
    """energyAngular (Legendre in mu per E') → energyAngularMC."""
    m = _m()
    xys3d = form.xys3d
    spectra, angular = [], []
    for k, surface in enumerate(xys3d.function2ds):
        energy = surface.outerDomainValue
        nodes = list(surface.function1ds)
        eOut = np.array([n.outerDomainValue for n in nodes], dtype=float)
        f0 = np.array([_magnitude(n) for n in nodes], dtype=float)
        histogram = surface.interpolation == m.Interpolation.flat
        nd = int(discrete[k]) if discrete is not None and k < len(discrete) else 0
        table, _ = pdfTable(eOut, f0, histogram=histogram, options=options, discrete=nd, outer=energy)
        spectra.append(table)
        tables = []
        for node, magnitude in zip(nodes, f0):
            if magnitude > 0:
                normalised = _normalisedNode(node, magnitude)
                mu, _ = muTable(normalised, options, outer=node.outerDomainValue)
            else:
                mu = _isotropicTable(node.outerDomainValue)
            tables.append(mu)
        angular.append(m.XYs2d(function1ds=tables, outerDomainValue=energy))
    energyForm = m.XYs2d(function1ds=spectra, interpolation=xys3d.interpolation,
                         interpolationQualifier=xys3d.interpolationQualifier)
    return m.EnergyAngularMC(energy=energyForm,
                             energyAngular=m.XYs3d(function2ds=angular,
                                                   interpolation=xys3d.interpolation,
                                                   interpolationQualifier=xys3d.interpolationQualifier),
                             label="MonteCarlo", productFrame=form.productFrame)


def _magnitude(node) -> float:
    m = _m()
    if isinstance(node, m.Legendre):
        return float(node.coefficients[0]) if node.coefficients.size else 0.0
    x, y, histogram = _linlinXY(node)
    dx = np.diff(x)
    return float(np.sum(y[:-1] * dx) if histogram else np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def _normalisedNode(node, magnitude):
    m = _m()
    if isinstance(node, m.Legendre):
        return m.Legendre(coefficients=node.coefficients / magnitude)
    x, y, histogram = _linlinXY(node)
    return m.XYs1d(xs=x, ys=y / magnitude,
                   interpolation=m.Interpolation.flat if histogram else m.Interpolation.linlin)


def tabulateAngularEnergy(form, options: TabulationOptions):
    """angularEnergy (LAW=7, mu outermost) → energyAngularMC (E' outermost)."""
    m = _m()
    xys3d = form.xys3d
    spectra, angular = [], []
    for surface in xys3d.function2ds:
        energy = surface.outerDomainValue
        cosines = list(surface.function1ds)
        mu = np.array([c.outerDomainValue for c in cosines], dtype=float)
        curves = [_linlinXY(c) for c in cosines]
        grid = np.unique(np.concatenate([x for x, _, _ in curves]))
        joint = np.array([np.interp(grid, x, y, left=0.0, right=0.0) for x, y, _ in curves])
        if len(mu) > 1:
            spectrum = np.trapezoid(joint, mu, axis=0)
        else:
            spectrum = joint[0]
        table, _ = pdfTable(grid, spectrum, options=options, outer=energy)
        spectra.append(table)
        tables = []
        for j, eOut in enumerate(grid):
            column = joint[:, j]
            if spectrum[j] > 0 and len(mu) > 1:
                t, _ = pdfTable(mu, column, options=options, outer=eOut)
            else:
                t = _isotropicTable(eOut)
            tables.append(t)
        angular.append(m.XYs2d(function1ds=tables, outerDomainValue=energy))
    return m.EnergyAngularMC(energy=m.XYs2d(function1ds=spectra, interpolation=xys3d.interpolation),
                             energyAngular=m.XYs3d(function2ds=angular,
                                                   interpolation=xys3d.interpolation),
                             label="MonteCarlo", productFrame=m.Frame.lab)

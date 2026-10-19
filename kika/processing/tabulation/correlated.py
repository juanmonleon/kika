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


def _jointNodes(surface, options, discrete):
    """Refine a linear joint Legendre law before splitting marginal/conditional PDFs.

    Interpolating normalised conditional coefficients alone loses the joint law
    when its marginal changes. The coefficient L1 bound uses |P_l(mu)| <= 1.
    Zero-marginal endpoints take the adjacent limiting conditional distribution.
    """
    m = _m()
    nodes = list(surface.function1ds)
    if discrete or surface.interpolation != m.Interpolation.linlin or not all(isinstance(n, m.Legendre) for n in nodes):
        return nodes
    import copy
    order = max(len(n.coefficients) for n in nodes)
    weights = (2*np.arange(order)+1)/2
    pairs = [(float(n.outerDomainValue), np.pad(np.asarray(n.coefficients,float), (0,order-len(n.coefficients)))) for n in nodes]
    peak=max(float(np.sum(weights*abs(c))) for _,c in pairs)
    for iteration in range(40):
        refined=[];changed=False;worst=0.;bad=None
        for left,right in zip(pairs[:-1],pairs[1:]):
            a,ca=left;b,cb=right;refined.append(left)
            if b<=a:continue
            mid=(ca+cb)/2
            if mid[0]<=0:continue
            na=ca/ca[0] if ca[0]>0 else cb/cb[0]
            nb=cb/cb[0] if cb[0]>0 else na
            linear=(na+nb)/2
            error=float(np.sum(weights*abs(mid-mid[0]*linear)))
            scale=max(float(np.sum(weights*abs(mid))),1e-300)
            if error > options.rtol*scale+options.atol*peak:
                if error/scale>worst:worst=error/scale;bad=(a,b,ca.tolist(),cb.tolist())
                refined.append(((a+b)/2,mid));changed=True
        refined.append(pairs[-1])
        if len(refined)>options.maxPoints:raise ValueError("joint energy-angle refinement exceeds maxPoints")
        pairs=refined
        if not changed:break
    else:raise ValueError(f"joint refinement did not converge: E={surface.outerDomainValue}, worst={worst}, panel={bad}")
    result=[]
    for x,c in pairs:
        node=copy.copy(nodes[0]);node.outerDomainValue=x;node.coefficients=c.copy();result.append(node)
    return result


def tabulateEnergyAngular(form, options: TabulationOptions, *,
                          discrete: Optional[Sequence[int]] = None):
    """energyAngular (Legendre in mu per E') → energyAngularMC."""
    m = _m()
    # The ordinary tolerances, as ACER's. Caps of 1e-4/1e-6 for this law (13-oct, 78693cf)
    # made Fe-56's ACE 2.7 times larger (U-235's 2.1) and moved no product's <E'> by 1e-4
    # nor the KERMA by 2e-5 (Fe-56, O-16, Be-9, U-235; runbook PXB-065).
    xys3d = form.xys3d
    spectra, angular = [], []
    # Repeated normalized Legendre laws are common across outgoing energies.
    # Keep this cache local and bounded; returned mutable tables own their arrays.
    from collections import OrderedDict
    angularCache = OrderedDict()
    def angleTable(node, outer):
        import copy
        key = (np.asarray(node.coefficients,dtype=float).tobytes()
               if isinstance(node,m.Legendre) else None)
        if key is not None and key in angularCache:
            angularCache.move_to_end(key)
            table = copy.deepcopy(angularCache[key])
            table.outerDomainValue = outer
            return table
        table, _ = muTable(node, options, outer=outer)
        if key is not None:
            angularCache[key] = table
            if len(angularCache)>256:angularCache.popitem(last=False)
        return table
    for k, surface in enumerate(xys3d.function2ds):
        energy = surface.outerDomainValue
        nd = int(discrete[k]) if discrete is not None and k < len(discrete) else 0
        nodes = _jointNodes(surface, options, nd)
        eOut = np.array([n.outerDomainValue for n in nodes], dtype=float)
        f0 = np.array([_magnitude(n) for n in nodes], dtype=float)
        histogram = surface.interpolation == m.Interpolation.flat
        nd = int(discrete[k]) if discrete is not None and k < len(discrete) else 0
        table, _ = pdfTable(eOut, f0, histogram=histogram, options=options, discrete=nd, outer=energy)
        spectra.append(table)
        tables = []
        for inode, (node, magnitude) in enumerate(zip(nodes, f0)):
            if magnitude > 0:
                normalised = _normalisedNode(node, magnitude)
                mu = angleTable(normalised, node.outerDomainValue)
            else:
                neighbours = [j for j in (inode-1,inode+1) if 0 <= j < len(nodes) and f0[j] > 0]
                if not nd and not histogram and isinstance(node,m.Legendre) and neighbours:
                    j = neighbours[0]
                    mu = angleTable(_normalisedNode(nodes[j],f0[j]),node.outerDomainValue)
                else:mu = _isotropicTable(node.outerDomainValue)
            tables.append(mu)
        angular.append(m.XYs2d(function1ds=tables, outerDomainValue=energy,
                                interpolation=surface.interpolation))
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

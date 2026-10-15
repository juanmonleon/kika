"""Outgoing-energy distributions P(E'|E) → what a Monte Carlo library stores.

=====================================  =========================================
evaluated form                         processed form
=====================================  =========================================
table per incident energy (MF5 LF=1,   the same points, normalised, with the
MF6 LAW=1 NA=0, MF15)                  cumulative (``xs_pdf_cdf1d``); the first
                                       ND points kept as discrete lines
Maxwell (LF=7), evaporation (LF=9),    kept: OpenMC samples them analytically
Watt (LF=11)                           (``maxwell``, ``evaporation``, ``watt``)
general evaporation (LF=5),            tabulated on an adaptive grid in E' at the
Madland-Nix (LF=12)                    parameters' incident energies (OpenMC's
                                       C++ samples neither)
discreteGamma / primaryGamma           :class:`DiscretePhotonEnergy` (ACE law 2,
                                       LP = 0 / 2)
NBodyPhaseSpace                        kept (``nbody``)
=====================================  =========================================

A ``weightedFunctionals`` is split by :mod:`.distribution` into a mixture,
one term per functional.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .angular import _linlinXY
from .tables import TabulationOptions, pdfTable, refine

__all__ = ["tabulateEnergy", "mapFunction2d"]


def _m():
    from kika.nuclear_data import model

    return model


def mapFunction2d(form, convert):
    """The same XYs2d / regions2d with every inner function replaced by ``convert(f, k)``."""
    m = _m()
    counter = [0]

    def walk(node):
        if isinstance(node, m.XYs2d):
            out = []
            for f in node.function1ds:
                out.append(convert(f, counter[0]))
                counter[0] += 1
            return m.XYs2d(function1ds=out, interpolation=node.interpolation,
                           interpolationQualifier=node.interpolationQualifier,
                           outerDomainValue=node.outerDomainValue, index=node.index)
        if isinstance(node, m.Regions2d):
            return m.Regions2d(function2ds=[walk(r) for r in node.function2ds],
                               outerDomainValue=node.outerDomainValue, index=node.index)
        raise TypeError(f"a {type(node).__name__} is not a two-dimensional table")

    return walk(form)


def _tableAt(function, k, options, discrete, clipped):
    x, y, histogram = _linlinXY(function)
    nd = int(discrete[k]) if discrete is not None and k < len(discrete) else 0
    table, wasClipped = pdfTable(x, y, histogram=histogram, options=options, discrete=nd,
                                 outer=function.outerDomainValue,
                                 what=f"P(E'|E={function.outerDomainValue:g} eV)")
    clipped[0] += wasClipped
    if nd:
        # OpenMC 0.16 interpolates discrete entries on the lin-lin branch.
        # Two positive constant subpanels can retain a linear panel's mass,
        # first and second moments exactly; a CDF bound sets their width.
        table = _mixedHistogram(table, options, outer=function.outerDomainValue)
    return table


def _mixedHistogram(table, options, outer):
    """Transport-safe mixed measure with exact panel moments through order two."""
    nd = int(table.numberOfDiscrete)
    x, p = np.asarray(table.xs[nd:]), np.asarray(table.pdf[nd:])
    flat = str(table.interpolation) == "flat"
    if len(x) < 2:
        table.interpolation = _m().Interpolation.flat
        return table
    if flat and len(x) >= 3:
        return table
    edges, density = [float(x[0])], []
    for a, b, pa, pb in zip(x[:-1], x[1:], p[:-1], p[1:]):
        if b <= a:
            continue
        if flat:
            pb = pa
        # On each subpanel the CDF error is at most |dp|*dx/24. Mass
        # and moments through order two agree exactly at every panel end.
        # This is a projection, not an adaptive refinement, so maxPoints does
        # not cap it: |dp|*dx <= 2 m on a panel of mass m, which bounds the
        # table by N + 2*sqrt(N/(12*mixedCdfAtol)) points for N panels of a
        # normalised continuum. Capping it dropped the whole product instead:
        # Pu-242's MF6 capture photons of ENDF/B-VIII.1 need 2 057 points at
        # 6.5 MeV and were lost.
        count = max(1, int(np.ceil(np.sqrt(abs(pb-pa)*(b-a)/(24*options.mixedCdfAtol)))))
        for k in range(count):
            left, right = a+(b-a)*k/count, a+(b-a)*(k+1)/count
            pl, pr = pa+(pb-pa)*k/count, pa+(pb-pa)*(k+1)/count
            density.extend([pl+(pr-pl)/6, pl+5*(pr-pl)/6])
            edges.extend([(left+right)/2, right])
    if not density:
        return table
    density.append(float(p[-1]))
    converted, _ = pdfTable(np.r_[table.xs[:nd], edges], np.r_[table.pdf[:nd], density],
        histogram=True, options=options, discrete=nd, outer=outer, what="mixed line/continuum spectrum")
    return converted


def _histogramLaw(law) -> bool:
    """A general evaporation whose g(x) is one histogram (flat) table."""
    m = _m()
    g = getattr(law, "g", None)
    return (isinstance(law, m.GeneralEvaporation) and isinstance(g, m.XYs1d)
            and getattr(g, "interpolation", None) == m.Interpolation.flat)


def _tabulateLaw(law, options: TabulationOptions, incident: Optional[Sequence[float]] = None):
    """A parametrised spectrum → an XYs2d of xs_pdf_cdf1d on adaptive outgoing grids."""
    m = _m()
    energies = law.incidentEnergies() if incident is None else np.asarray(incident, dtype=float)
    tables = []
    for energy in energies:
        seeds = law.outgoingGrid(float(energy), options.energySeeds)
        if seeds.size < 2:
            # Below the law's support: a spectrum concentrated at E' = 0.
            seeds = np.array([0.0, max(float(energy) * 1e-9, 1e-11)])
            table, _ = pdfTable(seeds, [1.0, 1.0], histogram=True, options=options, outer=energy)
            tables.append(table)
            continue
        if _histogramLaw(law):
            # g(x) is a histogram (LF=5 with INT=1: delayed-neutron spectra):
            # P(E'|E) = g(E'/theta)/theta is itself a histogram on theta*x_i,
            # written exactly as one. A lin-lin refinement of its steps never
            # converges (ACER also writes these as histograms).
            mids = 0.5 * (seeds[:-1] + seeds[1:])
            y = np.append(law.evaluate(float(energy), mids), 0.0)
            table, _ = pdfTable(seeds, y, histogram=True, options=options, outer=energy,
                                what=f"{type(law).__name__} at {energy:g} eV")
            tables.append(table)
            continue
        x, y = refine(lambda e: law.evaluate(float(energy), e), seeds, options)
        table, _ = pdfTable(x, y, options=options, outer=energy, what=f"{type(law).__name__} at {energy:g} eV")
        tables.append(table)
    return m.XYs2d(function1ds=tables, label="MonteCarlo")


def tabulateEnergy(form, options: TabulationOptions = TabulationOptions(), *, awr: float = None,
                   discrete=None) -> Tuple[object, int]:
    """``P(E'|E)`` → ``(processed form, number of tables clipped)``.

    ``discrete`` gives, per incident energy, how many leading points are lines
    (ENDF's ND, which the model keeps in provenance). ``awr`` is needed for a
    primary photon.
    """
    m = _m()
    clipped = [0]
    if isinstance(form, (m.XYs2d, m.Regions2d)):
        return mapFunction2d(form, lambda f, k: _tableAt(f, k, options, discrete, clipped)), clipped[0]
    if isinstance(form, (m.SimpleMaxwellianFission, m.Evaporation, m.Watt)):
        return form, 0
    if isinstance(form, (m.GeneralEvaporation, m.MadlandNix)):
        return _tabulateLaw(form, options), 0
    if isinstance(form, m.DiscreteGamma):
        return m.DiscretePhotonEnergy(primaryFlag=0, energy=float(form.value),
                                      atomicWeightRatio=float(awr or 0.0)), 0
    if isinstance(form, m.PrimaryGamma):
        if awr is None:
            raise ValueError("a primary photon needs the atomic weight ratio")
        return m.DiscretePhotonEnergy(primaryFlag=2, energy=float(form.value),
                                      atomicWeightRatio=float(awr)), 0
    if isinstance(form, (m.NBodyPhaseSpace, m.LevelEnergy, m.DiscretePhotonEnergy)):
        return form, 0
    raise TypeError(f"no tabulation for an energy distribution of type {type(form).__name__}")

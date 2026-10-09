"""G4b: MF5's bookkeeping from the model -- the ``"mf5"`` block the encoder rebuilds from.

:func:`~kika.endf.model_adapter.energy.encodeMF5MT` writes a section from a
record per subsection: ``U``, ``LF``, the weight ``p(E)`` and, for every
subsection the model form does not carry by itself, its records as bytes. This
module builds those records from the model, by FUDGE's rules
(``toENDF6/productData/distributions/energy.py``):

- **a table** (``XYs2d``/``regions2d``, NK=1) is LF=1 with U=0 and the weight
  ``[(E_first, 1), (E_last, 1)]``; its tables come from the model, as for a
  section read from ENDF;
- **a parametrised law** (§18.3: ``evaporation``, ``generalEvaporation``,
  ``simpleMaxwellianFission``, ``Watt``, ``MadlandNix``) is LF=9/5/7/11/12 with
  its own ``U`` (``EFL``/``EFH`` for Madland-Nix) and the weight over its first
  parameter's domain; its parameter TAB1s are formatted here, with the format's
  full precision;
- **a ``weightedFunctionals``** is one subsection per member, each with its
  ``weight`` as ``p(E)``;
- **MT455's precursor families** (§18.4) are one subsection each, weighted by
  the family's multiplicity over the total delayed nu-bar -- FUDGE's
  ``divideIgnoring0DividedBy0``, done here lin-lin on the union of the two
  grids by :mod:`kika.algebra`, and declared, because it is a division.

Every parametrised section is checked on the way out by the encoder itself:
``encodeMF5MT`` re-decodes the bytes built here and refuses unless they give
back the model's form, so a law formatted wrongly cannot reach a tape.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["mf5Fields", "familyFields"]


def _tab1Lines(c1, c2, function) -> List[str]:
    """Columns 1-66 of a TAB1 of *function* (an XYs1d or regions1d)."""
    from kika.endf.utils import ENDF_FORMAT_PRECISE, format_tab1

    x, y, pairs = function.toEndfRegions()
    lines, _ = format_tab1(float(c1), float(c2), 0, 0, [tuple(map(int, p)) for p in pairs],
                           [float(v) for v in x], [float(v) for v in y], 0, 5, 0, 1,
                           data_format=ENDF_FORMAT_PRECISE)
    return [line[:66] for line in lines]


def _domain(function):
    x, _, _ = function.toEndfRegions()
    return float(x[0]), float(x[-1])


def _record(index, lf, u, weight, rawLines=None) -> dict:
    """One subsection's record, in ``energy._partialRecord``'s shape."""
    if isinstance(weight, tuple):
        lo, hi = weight
        pairs, px, py = [(2, 2)], [lo, hi], [1.0, 1.0]
    else:
        x, y, pairs = weight.toEndfRegions()
        pairs, px, py = [tuple(map(int, p)) for p in pairs], [float(v) for v in x], [float(v) for v in y]
    record = {"index": index, "lf": int(lf), "u": float(u), "p_interp": pairs,
              "p_energies": px, "p_values": py}
    if rawLines is not None:
        record["raw_lines"] = rawLines
    return record


def _incidentDomain(table):
    from kika.nuclear_data.model import toEndfTab2

    functions, _ = toEndfTab2(table)
    return float(functions[0].outerDomainValue), float(functions[-1].outerDomainValue)


def _tableLines(table) -> List[str]:
    """An LF=1 table's records after its subsection header, as bytes."""
    from kika.endf.classes.mf5.partials import MF5PartialTabulated
    from kika.nuclear_data.model import toEndfTab2

    from ..energy import _spectrumRecord, _verbatimBody

    functions, pairs = toEndfTab2(table)
    records = [_spectrumRecord(f) for f in functions]
    lo, hi = _incidentDomain(table)
    partial = MF5PartialTabulated(
        u=0.0, lf=1, p_interp=[(2, 2)], p_energies=[lo, hi], p_values=[1.0, 1.0],
        tab2_interp=pairs, incident_energies=[float(f.outerDomainValue) for f in functions],
        outgoing_grids=[r[0] for r in records], chi=[r[1] for r in records],
        outgoing_interp=[r[2] for r in records])
    return _verbatimBody(partial, 0, 5, 0)


def _lawRecord(form, index, weight=None) -> Optional[dict]:
    """A §18.3 law or an LF=1 table → its subsection record with its bytes."""
    from kika.nuclear_data.model.energy_spectra import (Evaporation, GeneralEvaporation,
                                                        MadlandNix, SimpleMaxwellianFission,
                                                        Watt)

    u = float(form.U.value) if getattr(form, "U", None) is not None else 0.0
    if isinstance(form, GeneralEvaporation):
        lf, lines, first = 5, _tab1Lines(0, 0, form.theta) + _tab1Lines(0, 0, form.g), form.theta
    elif isinstance(form, SimpleMaxwellianFission):
        lf, lines, first = 7, _tab1Lines(0, 0, form.theta), form.theta
    elif isinstance(form, Evaporation):
        lf, lines, first = 9, _tab1Lines(0, 0, form.theta), form.theta
    elif isinstance(form, Watt):
        lf, lines, first = 11, _tab1Lines(0, 0, form.a) + _tab1Lines(0, 0, form.b), form.a
    elif isinstance(form, MadlandNix):
        lf, u, first = 12, 0.0, form.T_M
        lines = _tab1Lines(form.EFL.value, form.EFH.value, form.T_M)
    elif hasattr(form, "function1ds") or hasattr(form, "function2ds"):
        return _record(index, 1, 0.0, weight or _incidentDomain(form), _tableLines(form))
    else:
        return None
    return _record(index, lf, u, weight or _domain(first), lines)


def _header(context) -> dict:
    return {"mat": context.mat, "za": context.za, "awr": context.awr}


def mf5Fields(form, context, report, mt) -> Optional[dict]:
    """The ``"mf5"`` block for a neutron's energy *form*, or ``None`` with the reason reported."""
    from kika.nuclear_data.model.energy_spectra import WeightedFunctionals

    if hasattr(form, "function1ds") or hasattr(form, "function2ds"):
        return {**_header(context), "nk": 1, "modelled": 0,
                "partials": [_record(0, 1, 0.0, _incidentDomain(form))]}
    if isinstance(form, WeightedFunctionals):
        records = [_lawRecord(member.functional, k, member.weight)
                   for k, member in enumerate(form.weighted)]
    else:
        records = [_lawRecord(form, 0)]
    if any(r is None for r in records):
        report.lost(f"MF5/MT{mt}: a {type(form).__name__} energy form has no MF5 law "
                    f"kika writes; MF5/MT{mt} is not written")
        return None
    return {**_header(context), "nk": len(records), "modelled": None,
            "parametrised": True, "partials": records}


def _ratio(numerator, denominator):
    """``numerator / denominator`` lin-lin on the union of their grids; 0/0 is 0."""
    from kika.algebra.evaluate import sample_on_union
    from kika.algebra.grid import union

    from ..photons import _linlin

    nx, ny = _linlin(numerator)
    dx, dy = _linlin(denominator)
    grid = union([nx, dx[(dx >= nx[0]) & (dx <= nx[-1])]])
    top = sample_on_union(nx, ny, 2, grid)
    bottom = sample_on_union(dx, dy, 2, grid)
    zero = bottom == 0.0
    return grid, np.where(zero, 0.0, top / np.where(zero, 1.0, bottom))


def _thin(x, y, rtol: float = 1e-12):
    """Drop interior points that lin-lin interpolation of their kept neighbours
    reproduces to *rtol* -- FUDGE's ``thinWeights``, made exact: a constant
    weight comes back as its two end points, and nothing else moves."""
    keep = [0]
    for k in range(1, len(x) - 1):
        a, b = keep[-1], k + 1
        t = (x[k] - x[a]) / (x[b] - x[a]) if x[b] != x[a] else 0.0
        guess = y[a] + t * (y[b] - y[a])
        if not np.isclose(guess, y[k], rtol=rtol, atol=0.0):
            keep.append(k)
    keep.append(len(x) - 1)
    return np.asarray(x)[keep], np.asarray(y)[keep]


def familyFields(suite, families, context, report) -> Optional[dict]:
    """MF5/MT455's block: one subsection per precursor family (§18.4).

    The weight of family *k* is its multiplicity over the total delayed nu-bar
    (the MT455 ``multiplicitySum``, or the families' sum when there is none),
    lin-lin on the union of the two grids -- FUDGE's rule, and a division, so
    declared once.
    """
    from kika.nuclear_data.model import EVAL_LABEL, XYs1d

    total = suite.sums.multiplicitySums.byENDF_MT(455)
    if total is None or total.multiplicity is None:
        report.lost("MF5/MT455: no total delayed nu-bar (MT455 multiplicitySum) to "
                    "weigh the families by; MF5/MT455 is not written")
        return None
    records = []
    for k, family in enumerate(families):
        product = getattr(family, "product", None)
        form = (product.distribution.get(EVAL_LABEL)
                if product is not None and product.distribution is not None else None)
        energy = getattr(form, "energy", None)
        multiplicity = getattr(product, "multiplicity", None)
        if energy is None or multiplicity is None or not multiplicity.isEvaluable:
            report.lost(f"MF5/MT455: family {family.label!r} has no spectrum or no "
                        f"multiplicity to weigh it by; MF5/MT455 is not written")
            return None
        x, y = _thin(*_ratio(multiplicity.form, total.multiplicity.form))
        record = _lawRecord(energy, k, XYs1d(xs=x, ys=y))
        if record is None:
            report.lost(f"MF5/MT455: family {family.label!r} has a "
                        f"{type(energy).__name__} spectrum, which has no MF5 law")
            return None
        records.append(record)
    report.approximated(
        "MF5/MT455: each family's p(E) is its multiplicity over the total delayed "
        "nu-bar, divided lin-lin on the union of the two grids (FUDGE's rule)")
    return {**_header(context), "nk": len(records), "modelled": None,
            "parametrised": True, "partials": records}

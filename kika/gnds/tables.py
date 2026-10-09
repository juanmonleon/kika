"""A suite read from GNDS, as the plain arrays and rows a viewer draws.

A GNDS ``reactionSuite`` decodes onto the same model a G4NDL isotope does, and
the arrays a viewer wants out of it are the same: every cross section
pointwise, and each reaction's angular distribution in the MF4 bulk shape. So
:mod:`kika.g4ndl.tables` already does the work, format-blind, and this module
re-exports it (:func:`crossSections`, :func:`angularMTs`, :func:`angularBulk`)
next to the one thing GNDS needs of its own: :func:`isotopeSummary`, the
header a viewer shows above the plots.

**Which cross section is "the" cross section.** A file translated by FUDGE
carries, beside each ``resonancesWithBackground`` form, a ``recon`` ``XYs1d``
under a ``crossSectionReconstructed`` style: FUDGE's own resonance
reconstruction at 0 K. Every other reaction is pointwise in ``eval`` as it
stands. :func:`kika.g4ndl.tables.pointwiseCrossSection` takes ``recon`` first
and ``eval`` otherwise, so nothing here reconstructs anything; a reaction whose
only form is resonance parameters is reported in ``not_pointwise`` rather than
drawn.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from kika.g4ndl.tables import (  # noqa: F401  (re-exported)
    _angularShape, _range, angularBulk, angularMTs, crossSections, pointwiseCrossSection,
)

__all__ = ["angularBulk", "angularMTs", "crossSections", "isotopeSummary",
           "pointwiseCrossSection"]


def _formLabel(node) -> str:
    labels = list(node.crossSection)
    return "recon" if "recon" in labels else (labels[0] if labels else "")


def _resonances(suite) -> Dict[str, Any]:
    res = suite.resonances
    if res is None:
        return dict(present=False, resolved=[], unresolved=None)
    resolved = [dict(energy_min=float(r.domainMin), energy_max=float(r.domainMax),
                     formalism=type(r.formalism).__name__ if r.formalism is not None else None)
                for r in res.resolved]
    u = res.unresolved
    return dict(present=True, resolved=resolved,
                unresolved=None if u is None else dict(energy_min=float(u.domainMin),
                                                       energy_max=float(u.domainMax)))


def isotopeSummary(suite) -> Dict[str, Any]:
    """What a viewer shows above a GNDS target's plots: the evaluation, the
    styles, the resonance regions, one row per cross section and what the
    decode lost."""
    angular = set(angularMTs(suite))
    pointwise = crossSections(suite)
    sums = {int(n.ENDF_MT) for n in suite.sums if n.ENDF_MT is not None}
    nodes = {int(n.ENDF_MT): n for n in list(suite.reactions) + list(suite.sums)
             if n.ENDF_MT is not None}

    rows: List[Dict[str, Any]] = []
    for mt, (e, s) in pointwise.items():
        node = nodes[mt]
        q = None
        if mt not in sums:
            q = getattr(getattr(node.outputChannel, "Q", None), "value", None)
        positive = np.nonzero(s > 0)[0]
        row = dict(mt=mt, label=str(node.label), form=_formLabel(node),
                   q_value=None if q is None else float(q),
                   threshold=float(e[positive[0]]) if positive.size else None,
                   sum=mt in sums, angular=mt in angular, **_range((e, s)))
        if mt in angular:
            row.update(_angularShape(suite, mt))
        rows.append(row)

    report = suite.report
    styles = [dict(label=str(st.label), kind=type(st).__name__,
                   derived_from=getattr(st, "derivedFrom", None) or None)
              for st in suite.styles]
    return dict(
        target=str(suite.target),
        projectile=str(suite.projectile),
        evaluation=suite.evaluation,
        format=str(suite.format) if suite.format is not None else None,
        interaction=getattr(suite, "interaction", None),
        styles=styles,
        reconstructed=any(st["kind"] == "CrossSectionReconstructed" for st in styles),
        resonances=_resonances(suite),
        reactions=rows,
        angular_mts=sorted(angular),
        not_pointwise=sorted(mt for mt in nodes if mt not in pointwise),
        counts=dict(reactions=len(suite.reactions), sums=len(suite.sums),
                    productions=len(suite.productions),
                    fission_components=len(suite.fissionComponents)),
        report=dict(warnings=list(report.warnings), losses=list(report.losses),
                    approximations=list(report.approximations),
                    unsupported=list(report.unsupported)) if report is not None else None,
    )

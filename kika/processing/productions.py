"""The forms a radionuclide production (ENDF MF8-10) takes at a processed label.

A production shares its MT with the reaction that makes it -- MT102's isomer is
MT102 -- so it is not a cross section of the material, and the resonance
reconstruction and Doppler broadening, which project reactions by MT, leave it
out (:func:`kika.processing.resonances.suite._entries`). This module gives it
the new label instead, after the reactions have theirs:

* a ``reference`` (MF9, and MF8 ``LMF=3``) becomes a reference to its
  reaction's form at the new label. The reaction is what was reconstructed or
  broadened, and the production follows it: that is what the reference is for,
  and it is how the resonance contribution reaches an isomer made by capture.
* a table (MF10) is its own curve, to which MF2 adds nothing. RECONR processes
  MF10 by linearising it (``reconr.f90``: "process mf3, mf10, mf12, mf13 and
  mf23 only") and BROADR leaves it as it is, so the reconstruction re-expresses
  it lin-lin to the reconstruction's tolerance and Doppler broadening copies it.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Mapping, Optional, Tuple

import numpy as np

__all__ = ["production_forms", "publish_production_forms"]


def _linlin(form, label: str, tolerance: float):
    """*form* read lin-lin to *tolerance*: copied when it already is, refined when not."""
    from kika.algebra import LINLIN, interval_laws, split_at_discontinuities, to_linlin
    from kika.nuclear_data.model import Interpolation, Regions1d, XYs1d, crossSectionAxes

    regions = form.function1ds if isinstance(form, Regions1d) else [form]
    if all(r.interpolation == Interpolation.linlin for r in regions):
        out = deepcopy(form)
    else:
        whole = form if isinstance(form, Regions1d) else Regions1d(function1ds=[form])
        x, y, pairs = whole.toEndfRegions()
        lx, ly = to_linlin(x, y, interval_laws(len(x), pairs), tol=tolerance)
        pieces = split_at_discontinuities(np.asarray(lx, float), np.asarray(ly, float),
                                          np.full(max(len(lx) - 1, 0), LINLIN))
        curves = [XYs1d(xs=px, ys=py) for px, py, _ in pieces]
        if len(curves) == 1:
            out = curves[0]
            out.axes = crossSectionAxes()
        else:
            out = Regions1d(function1ds=curves, axes=crossSectionAxes())
    out.label = label
    return out


def production_forms(suite, source_label: str, label: str, new_forms: Mapping,
                     *, linearise_to: Optional[float] = None) -> List[Tuple[object, object]]:
    """``[(production, form at label)]`` for every radionuclide production of *suite*.

    *new_forms* maps the reconstruction's or broadening's ``ReactionKey`` to the
    form it made, which is what a reference now points at. *linearise_to* is the
    relative tolerance a non-lin-lin MF10 table is re-expressed to; ``None``
    copies it. Raises, naming the production, before anything is published.
    """
    from kika.nuclear_data.model import isRadionuclideProduction
    from kika.nuclear_data.model.cross_section_forms import Reference, crossSectionFormTag

    from .resonances.suite import _entries, _path, _target

    entries = _entries(suite)
    out = []
    for production in suite.productions:
        if not isRadionuclideProduction(production):
            continue
        if label in production.crossSection:
            raise ValueError(f"{production.label} already has a {label!r} form")
        source = production.crossSection.get(source_label)
        if source is None:
            raise ValueError(f"{production.label} has no {source_label!r} form to process")
        if isinstance(source, Reference):
            base = f"/reactionSuite/productions/production[@label='{production.label}']/crossSection"
            host, _ = _target(source.href, base + f"/reference[@label='{source_label}']",
                              entries, source_label)
            made = new_forms.get(host)
            if made is None:
                raise ValueError(f"{production.label} refers to {host.label}, which has no "
                                 f"{label!r} form")
            form = Reference(href=f"{_path(host)}/{crossSectionFormTag(made)}[@label='{label}']",
                             label=label)
        elif linearise_to is not None:
            form = _linlin(source, label, linearise_to)
        else:
            form = deepcopy(source)
            form.label = label
        out.append((production, form))
    return out


def publish_production_forms(pending) -> None:
    """Set each ``(production, form)`` of :func:`production_forms`. Cannot fail."""
    for production, form in pending:
        production.crossSection[form.label] = form

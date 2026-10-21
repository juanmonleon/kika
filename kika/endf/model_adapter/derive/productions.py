"""The ENDF bookkeeping of a ``production``, derived from the model.

Gives a production read from GNDS -- or built by hand -- exactly the provenance
:func:`~kika.endf.model_adapter.productions.attachProductions` gives one read
from ENDF, so :func:`~kika.endf.model_adapter.productions.encodeProductionSections`
writes both the same way and the oracle compares them field by field:

* which file states it: MF9 when the cross section is a reference and the
  multiplicity is tabulated, MF10 when the cross section is a table, MF8 alone
  (``LMF=3``) when it is a reference with a multiplicity of one;
* ``QM`` = Q + the product's level energy in PoPs (``QM - QI`` is that energy);
* the interpolation regions of the table, adjacent equal laws merged (a GNDS
  file splits a step into two regions; ENDF keeps it in one);
* the state's position among its MT's states of the same file, in the order the
  suite lists them, and the target's ``LIS``.
"""
from __future__ import annotations

from typing import Optional

from . import DERIVED, DerivationContext, register


def _isProduction(node) -> bool:
    from kika.nuclear_data.model import Reaction

    return (isinstance(node, Reaction)
            and getattr(getattr(node, "outputChannel", None), "genre", None) == "production")


def deriveProduction(production, path: str, context: DerivationContext, report) -> Optional[object]:
    from kika.nuclear_data.model import EVAL_LABEL, EndfProvenance
    from kika.nuclear_data.model.cross_section_forms import Reference
    from kika.nuclear_data.model.functions.simple import Constant1d

    from ..photons import _tab1Of
    from ..productions import _levelOf, _mergedRegions

    if production.ENDF_MT is None:
        return None
    products = [p for p in production.outputChannel.products if p.pid != "photon"]
    sigma = production.crossSection.get(EVAL_LABEL)
    if len(products) != 1 or sigma is None or products[0].multiplicity is None:
        return None
    multiplicity = products[0].multiplicity.form
    if isinstance(sigma, Reference):
        constantOne = isinstance(multiplicity, Constant1d) and float(multiplicity.constant) == 1.0
        kind, table = ("mf8", None) if constantOne else ("mf9", multiplicity)
    else:
        kind, table = "mf10", sigma

    header = {}
    suiteProvenance = getattr(context.suite, "provenance", None)
    lis = int((getattr(suiteProvenance, "headerFields", None) or {}).get("lis") or 0)
    if kind == "mf8":
        header[kind] = {}
        qm, regions = None, []
    else:
        siblings = [p for p in context.suite.productions
                    if p.ENDF_MT == production.ENDF_MT and _kindOf(p) == kind]
        order = next(k for k, p in enumerate(siblings) if p is production)
        header[kind] = {"order": order, "lis": lis, "l2": 0, "n2": 0}
        q = getattr(production.outputChannel.Q, "value", None)
        _, energy = _levelOf(context.suite, products[0].pid)
        qm = None if q is None else float(q) + energy
        regions = _mergedRegions(_tab1Of(table)[0])
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, qm=qm,
                          interpolationRegions=[tuple(pair) for pair in regions],
                          headerFields=header)


def _kindOf(production) -> Optional[str]:
    from kika.nuclear_data.model import EVAL_LABEL
    from kika.nuclear_data.model.cross_section_forms import Reference
    from kika.nuclear_data.model.functions.simple import Constant1d

    sigma = production.crossSection.get(EVAL_LABEL)
    products = [p for p in production.outputChannel.products if p.pid != "photon"]
    if sigma is None or len(products) != 1 or products[0].multiplicity is None:
        return None
    if not isinstance(sigma, Reference):
        return "mf10"
    form = products[0].multiplicity.form
    return "mf8" if isinstance(form, Constant1d) and float(form.constant) == 1.0 else "mf9"


register("productions", _isProduction, deriveProduction)

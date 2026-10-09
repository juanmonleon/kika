"""G3 (and G4b): a product's MF4 bookkeeping — LTT, LI, LCT, NM — from the model.

A product decoded from ENDF carries the MF4 header in its provenance
(``angular._headerProvenance``); one read from GNDS carries nothing, and the
tape writer emits MF4 only where ``"ltt"`` is in that provenance. This is the
rule that puts it there, stated in the model's terms:

- **Which product.** The one MF4 is about: ``angular.mf4Ejectile(MT)`` (the
  neutron, or the light charged particle of MT600-849), on the reaction's own
  output channel -- never a decay product -- and only when the reaction is not
  one FUDGE's ``ENDFconversionFlags`` send to MF6 (that is G4c).
- **LTT** from the form: ``isotropic2d`` is 0 (with LI=1); an ``XYs2d`` of
  ``Legendre`` is 1, of ``XYs1d`` is 2; a ``regions2d`` whose first region is
  Legendre and second tabulated is 3. An ``uncorrelated`` contributes its
  angular half (its energy half is MF5, G4b).
- **LCT** from the ``productFrame``: lab 1, centre of mass 2.
- **NM**, written by LTT=3 only, is the highest Legendre order of its Legendre
  region -- ENDF-102's definition, and what FUDGE writes
  (``toENDF6/productData/distributions/angular.py:168``). The libraries do not
  agree with it: on 74 of 131 sampled tapes the LTT=3 section states NM=0, and
  JEFF-4.0 Fe-56 states 31 beside a 32nd-order row. The oracle names that
  difference instead of copying a number the data contradicts.

Anything else (a Legendre and a tabulated function mixed in one XYs2d, a third
region) is refused by name rather than given a header that would misdescribe it.
"""
from __future__ import annotations

from typing import Optional

from . import DERIVED, DerivationContext, register

__all__ = ["deriveProduct", "angularHeader"]

def _owners(context: DerivationContext) -> dict:
    """``{id(product): reaction}`` for the products of every reaction's and sum's
    own output channel -- the only ones MF4 can be about. Built once per pass.

    By identity and not by path: a suite read from GNDS names its reactions by
    label (``n + S36``), so an ``MT`` in the path is not there to be read.
    """
    owners = getattr(context, "_productOwners", None)
    if owners is None:
        owners = {}
        for container in (context.suite.reactions, context.suite.sums):
            for reaction in container:
                channel = getattr(reaction, "outputChannel", None)
                if reaction.ENDF_MT is None or channel is None:
                    continue
                for product in channel.products:
                    owners[id(product)] = reaction
        context._productOwners = owners
    return owners


def _frameCode(frame) -> Optional[int]:
    from kika.nuclear_data.model import Frame

    return {Frame.lab: 1, Frame.centerOfMass: 2}.get(frame)


def _kind(function) -> str:
    from kika.nuclear_data.model import Legendre

    return "legendre" if isinstance(function, Legendre) else "tabulated"


def angularHeader(form, mt: int) -> dict:
    """``{"ltt", "li", "lct", "nm"}`` for the angular form *form* of MT *mt*.

    Raises ``ValueError`` naming the shape when no LTT describes it.
    """
    from kika.nuclear_data.model import AngularTwoBody, Isotropic2d, Regions2d, XYs2d

    if isinstance(form, Isotropic2d):
        return {"ltt": 0, "li": 1, "lct": _frameCode(form.productFrame), "nm": None}
    if not isinstance(form, AngularTwoBody) or form.angular is None:
        raise ValueError(f"MT{mt}: {type(form).__name__} is not an MF4 form")
    lct = _frameCode(form.productFrame)
    angular = form.angular
    if isinstance(angular, Isotropic2d):
        # §18.1.1 puts an isotropic two-body angle inside angularTwoBody (the
        # (n,alpha) levels of NNDC's Fe-56): MF4 LTT=0, LI=1.
        return {"ltt": 0, "li": 1, "lct": lct, "nm": None}
    if isinstance(angular, XYs2d):
        kinds = {_kind(f) for f in angular.function1ds}
        if len(kinds) != 1:
            raise ValueError(f"MT{mt}: one XYs2d mixes Legendre and tabulated functions, "
                             f"which no single MF4 LTT describes")
        return {"ltt": 1 if kinds == {"legendre"} else 2, "li": 0, "lct": lct, "nm": None}
    if isinstance(angular, Regions2d):
        parts = [{_kind(f) for f in region.function1ds} for region in angular.function2ds]
        if all(p == {"legendre"} for p in parts):
            return {"ltt": 1, "li": 0, "lct": lct, "nm": None}
        if all(p == {"tabulated"} for p in parts):
            return {"ltt": 2, "li": 0, "lct": lct, "nm": None}
        if len(parts) == 2 and parts[0] == {"legendre"} and parts[1] == {"tabulated"}:
            nm = max(len(f.coefficients) - 1 for f in angular.function2ds[0].function1ds)
            return {"ltt": 3, "li": 0, "lct": lct, "nm": int(nm)}
        raise ValueError(f"MT{mt}: a regions2d of {parts} is no MF4 LTT "
                         f"(LTT=3 is one Legendre region then one tabulated)")
    raise ValueError(f"MT{mt}: an angular {type(angular).__name__} is not an MF4 form")


def _sentToMF6(context: DerivationContext, mt: int) -> bool:
    """Whether FUDGE's flags put this reaction's distributions in MF6 (G4c)."""
    from .reactions import reactionHref

    reaction = context.suite.findReactionByENDF_MT(mt)
    if reaction is None:
        return False
    href = reactionHref(reaction)
    return any(text.split(",")[0].strip() == "MF6"
               for where, text in context.conversionFlags.items()
               if where.startswith(href))


def deriveProduct(product, path, context: DerivationContext, report):
    """MF4's provenance for the product MF4 is about, or ``None``."""
    from kika.nuclear_data.model import EVAL_LABEL, EndfProvenance, Uncorrelated

    from ..angular import mf4Ejectile

    reaction = _owners(context).get(id(product))
    if reaction is None:
        return None
    mt = int(reaction.ENDF_MT)
    from .energy_angle import goesToMF6

    if goesToMF6(reaction, context):
        return None
    distribution = getattr(product, "distribution", None)
    if distribution is None or EVAL_LABEL not in distribution:
        return None
    form = distribution[EVAL_LABEL]

    header = {}
    if product.pid == mf4Ejectile(mt):
        angular = form
        if isinstance(form, Uncorrelated):
            # §18.3 keeps the angular XYs2d bare under <angular>; MF4 is the
            # same table, so it is looked at as the angularTwoBody it would be
            # in §18.2.
            from kika.nuclear_data.model import AngularTwoBody, Isotropic2d

            angular = form.angular
            if angular is not None and not isinstance(angular, Isotropic2d):
                angular = AngularTwoBody(angular=angular, productFrame=form.productFrame)
        if angular is not None:
            try:
                header.update(angularHeader(angular, mt))
            except ValueError as refused:
                if "is not an MF4 form" not in str(refused):
                    report.lost(f"{refused}; MF4 is not written for it")
    if product.pid == "n" and isinstance(form, Uncorrelated) and form.energy is not None:
        # G4b: MF5 is always about the neutron (ENDF-6 §5).
        from .energy import mf5Fields

        fields = mf5Fields(form.energy, context, report, mt)
        if fields is not None:
            header["mf5"] = fields
    if not header:
        return None
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header)


def deriveDelayedNeutrons(families, path, context: DerivationContext, report):
    """MF5/MT455's provenance on §18.4's families (G4b)."""
    from kika.nuclear_data.model import EndfProvenance

    from .energy import familyFields

    if not len(families):
        return None
    fields = familyFields(context.suite, list(families), context, report)
    if fields is None:
        return None
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields={"mf5": fields})


def _isDelayedNeutrons(node) -> bool:
    from kika.nuclear_data.model import DelayedNeutrons

    return isinstance(node, DelayedNeutrons)


def _isProduct(node) -> bool:
    from kika.nuclear_data.model import Product

    return isinstance(node, Product)


register("products", _isProduct, deriveProduct)
register("delayedNeutrons", _isDelayedNeutrons, deriveDelayedNeutrons)

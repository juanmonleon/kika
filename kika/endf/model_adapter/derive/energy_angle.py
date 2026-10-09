"""G4c: MF6's bookkeeping from the model -- the ``"mf6"`` block ``encodeMF6MT`` rebuilds from.

**Which reactions.** FUDGE's ``ENDFconversionFlags`` ``MF6`` on any product of
the reaction decide it when the file has them. Otherwise the forms do, by the
rule FUDGE's ``toENDF6`` applies (``reactions/base.py``, ``doMF4AsMF6``): a
product whose distribution only MF6 can state -- ``KalbachMann``,
``energyAngular``, ``angularEnergy``, an ``NBodyPhaseSpace`` -- or a light
charged product with a distribution of its own in a reaction that is not
two-body sends the whole reaction to MF6.

**Which products, in which order.** The reaction's own, as the model lists
them, less what MF6 does not state: photons that MF12-15 carry (a discrete or
primary line, a branching to a level's cascade -- E5e) and a two-body residual
whose distribution is only the recoil and whose decay says nothing more, which
FUDGE marks ``implicitProduct``.

**Each product** (ENDF-6 §6.1): ZAP and LIP from its pid (``Fe56_e3`` is ZAP
26056, LIP 3; LIP is 0 in a level reaction, MT50-91 and 600-850, whose MT names
the level -- FUDGE's rule), AWP from its PoPs mass (the ground state's for an
excited level, atomic to nuclear for H and He), the yield from its multiplicity, and the
LAW from its form:

=================================  =====================================
form                               LAW
=================================  =====================================
``unspecified`` (or none)          0
``KalbachMann``                    1, LANG=2, NA = 1 or 2 (``a`` given)
``uncorrelated`` table             1, LANG=1, NA=0
``energyAngular``                  1, LANG=1, NA = the Legendre order
``angularTwoBody``, a table        2, LANG per node (0 Legendre, 10+INT)
``angularTwoBody``, isotropic      3
``angularTwoBody``, recoil         4
``NBodyPhaseSpace``                6, APSX from its mass
``angularEnergy``                  7
=================================  =====================================

LCT is 1 when every product is in the lab, 2 when every one is in the centre
of mass, and 3 when the light ones (A ≤ 4, a photon included, as
``frameForProduct`` reads it) are in the centre of mass and the heavy ones in
the lab. JP is 0 (the P(ν) subsections of a JP>0 fission section
have no model node). ND is 0 on every node: "the first ND outgoing points are
discrete lines" is a statement the model has no node for, and the ENDF decoder
reports it when it meets one.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["goesToMF6", "mf6Fields"]

_LIGHT = {"n", "H1", "H2", "H3", "He3", "He4"}


def _flagged(reaction, context) -> bool:
    from .reactions import reactionHref

    href = reactionHref(reaction)
    return any(text.split(",")[0].strip() == "MF6"
               for where, text in context.conversionFlags.items()
               if where.startswith(href))


def _form(product):
    from kika.nuclear_data.model import EVAL_LABEL

    distribution = getattr(product, "distribution", None)
    if distribution is None or EVAL_LABEL not in distribution:
        return None
    return distribution[EVAL_LABEL]


def _productFlag(reaction, product, context) -> str:
    """FUDGE's conversion flag on one product of *reaction* ('' when none)."""
    from .reactions import reactionHref

    href = (f"{reactionHref(reaction)}/outputChannel/products/product"
            f"[@label='{product.label}']")
    return context.conversionFlags.get(href, "").split(",")[0].strip()


def _productFlagged(reaction, product, context) -> bool:
    return _productFlag(reaction, product, context) == "MF6"


def _isMF12Photon(product) -> bool:
    """A photon MF12-15 states (FUDGE's rule: every photon of a reaction goes
    there unless its form is an ``energyAngular``): a line, a primary, a
    level's cascade or a continuum."""
    from kika.nuclear_data.model import Branching3d, DiscreteGamma, PrimaryGamma, Uncorrelated

    if product.pid != "photon":
        return False
    form = _form(product)
    if isinstance(form, Branching3d):
        return True
    if isinstance(form, Uncorrelated):
        return True
    return isinstance(getattr(form, "energy", None), (DiscreteGamma, PrimaryGamma))


def _isDiscreteLine(product) -> bool:
    from kika.nuclear_data.model import DiscreteGamma, Uncorrelated

    form = _form(product)
    return (product.pid == "photon" and isinstance(form, Uncorrelated)
            and isinstance(form.energy, DiscreteGamma))


def _onlyMF6(form) -> bool:
    from kika.nuclear_data.model import (AngularEnergy, EnergyAngular, KalbachMann,
                                         NBodyPhaseSpace, Uncorrelated)

    if isinstance(form, (KalbachMann, EnergyAngular, AngularEnergy)):
        return True
    return isinstance(form, Uncorrelated) and isinstance(form.energy, NBodyPhaseSpace)


def goesToMF6(reaction, context) -> bool:
    """Whether *reaction*'s distributions are written in MF6 rather than MF4/MF5."""
    from kika.nuclear_data.model import Unspecified

    if _flagged(reaction, context):
        return True
    channel = getattr(reaction, "outputChannel", None)
    if channel is None:
        return False
    twoBody = channel.genre == "twoBody"
    for product in channel.products:
        if product.pid == "photon":
            continue
        form = _form(product)
        if _onlyMF6(form):
            return True
        if (not twoBody and product.pid not in ("n",) and form is not None
                and not isinstance(form, Unspecified)
                and product.outputChannel is None):
            return True
    return False


def _implicitResidual(product, channel) -> bool:
    """FUDGE's ``implicitProduct``: a two-body recoil that states nothing else."""
    from kika.nuclear_data.model import AngularTwoBody, Branching3d, Unspecified

    if channel.genre != "twoBody" or product.pid in _LIGHT or product.pid == "photon":
        return False
    form = _form(product)
    recoil = isinstance(form, AngularTwoBody) and form.recoilHref is not None
    if not (recoil or isinstance(form, Unspecified) or form is None):
        return False
    decay = product.outputChannel
    return decay is None or all(isinstance(_form(p), (Unspecified, Branching3d)) or _form(p) is None
                                for p in decay.products)


def _zap(pid: str):
    from kika.nuclear_data.model.pops import zaFromPid

    lip = int(pid.split("_e", 1)[1]) if "_e" in pid and pid.split("_e", 1)[1].isdigit() else 0
    return zaFromPid(pid), lip


def _awp(suite, pid: str, report=None) -> Optional[float]:
    """AWP from PoPs: atomic there, nuclear in ENDF for H and He (light_masses)."""
    from kika.nuclear_data.model.pops import zaFromPid

    from ..light_masses import ratioFromAtomic

    if pid == "photon":
        return 0.0
    if pid == "n":
        return 1.0
    # An excited level carries no mass of its own in the distributed files;
    # ENDF's AWP for it is the ground state's (FUDGE's MassTracker keeps the
    # index-0 nuclides only).
    mass = None
    for candidate in (pid, pid.split("_e")[0]):
        mass = getattr(suite.PoPs.particles.get(candidate), "mass", None)
        if mass is not None:
            break
    if mass is None or mass.unit != "amu":
        # The distributed files leave some residuals without a mass (NNDC's
        # Fe-56 has none for Fe57 and Cr55); FUDGE falls back on its AME
        # table there, and kika on its own atomic mass table, said once.
        from kika._constants import ATOMIC_MASS

        amu = ATOMIC_MASS.get(zaFromPid(pid))
        if amu is None:
            return None
        if report is not None:
            report.approximated(f"MF6: {pid} has no mass in PoPs; its AWP is from "
                                f"kika's atomic mass table ({amu} amu)")
        return ratioFromAtomic(zaFromPid(pid), float(amu))
    return ratioFromAtomic(zaFromPid(pid), float(mass.value))


def _yield(multiplicity):
    from kika.nuclear_data.model.functions.simple import Constant1d

    form = getattr(multiplicity, "form", None)
    if isinstance(form, Constant1d):
        lo, hi = float(form.domainMin), float(form.domainMax)
        return [(2, 2)], [lo, hi], [float(form.constant)] * 2
    if form is not None and hasattr(form, "toEndfRegions"):
        x, y, pairs = form.toEndfRegions()
        return ([tuple(map(int, p)) for p in pairs], [float(v) for v in x],
                [float(v) for v in y])
    return None


def _code(function) -> int:
    return int(getattr(function, "endfInterpolationCode", 2) or 2)


def _law(form, frameOf):
    """``(law, law_fields)`` for a product's form, or ``None`` with no MF6 law for it."""
    from kika.nuclear_data.model import (AngularEnergy, AngularTwoBody, EnergyAngular,
                                         Isotropic2d, KalbachMann, Legendre,
                                         NBodyPhaseSpace, Uncorrelated, Unspecified,
                                         toEndfTab2, toEndfTab3)

    if form is None or isinstance(form, Unspecified):
        return 0, {}
    if isinstance(form, KalbachMann):
        functions, _ = toEndfTab2(form.f)
        na = 2 if form.a is not None else 1
        return 1, {"lang": 2, "lep": _code(functions[0]),
                   "nd": [0] * len(functions), "na": [na] * len(functions)}
    if isinstance(form, Uncorrelated) and isinstance(form.energy, NBodyPhaseSpace):
        return 6, {"npsx": int(form.energy.numberOfProducts), "apsx": None}
    from kika.nuclear_data.model import DiscreteGamma

    if isinstance(form, Uncorrelated) and isinstance(form.energy, DiscreteGamma):
        # One outgoing point, discrete, at both ends of the line's domain.
        return 1, {"lang": 1, "lep": 2, "nd": [1, 1], "na": [0, 0]}
    if isinstance(form, Uncorrelated) and form.energy is not None:
        functions, _ = toEndfTab2(form.energy)
        return 1, {"lang": 1, "lep": _code(functions[0]),
                   "nd": [0] * len(functions), "na": [0] * len(functions)}
    if isinstance(form, EnergyAngular):
        nodes, _ = toEndfTab3(form.xys3d)
        na = [max(len(f.coefficients) - 1 for f in node.function1ds) for node in nodes]
        inner = nodes[0].function1ds[0] if nodes else None
        return 1, {"lang": 1, "lep": 2 if inner is None else int(
                       getattr(nodes[0], "interpolationCode", 2) or 2),
                   "nd": [0] * len(nodes), "na": na}
    if isinstance(form, AngularTwoBody):
        if form.recoilHref is not None:
            return 4, {}
        if isinstance(form.angular, Isotropic2d):
            return 3, {}
        functions, _ = toEndfTab2(form.angular)
        return 2, {"lang": [0 if isinstance(f, Legendre) else 10 + _code(f) for f in functions]}
    if isinstance(form, AngularEnergy):
        return 7, {}
    return None


def mf6Fields(reaction, context, report) -> Optional[dict]:
    """The ``"mf6"`` block for *reaction*, or ``None`` with the reason reported."""
    from kika.nuclear_data.model import Frame

    channel = reaction.outputChannel
    mt = int(reaction.ENDF_MT)
    records, frames = [], []
    # FUDGE's `implicitProduct` flag says the same as _implicitResidual, for
    # whatever product the file marks (NNDC's Fe-56 marks the capture residual).
    kept = [p for p in channel.products
            if _productFlag(reaction, p, context) != "implicitProduct"
            and (_productFlagged(reaction, p, context)
                 or not (_isMF12Photon(p) or _implicitResidual(p, channel)))]
    # A residual's de-excitation line goes in MF6 after the products, as FUDGE
    # writes it (``checkDecayProducts``; ENDF/B-VIII.1 Be-9 MT701, the 477 keV
    # line of Li7_e1): LAW=1 with one discrete outgoing point.
    for product in list(channel.products):
        for decayed in getattr(getattr(product, "outputChannel", None), "products", None) or ():
            if _isDiscreteLine(decayed):
                kept.append(decayed)
    for index, product in enumerate(kept):
        form = _form(product)
        law = _law(form, None)
        if law is None:
            report.lost(f"MF6/MT{mt}: product {product.label!r} has a "
                        f"{type(form).__name__}, which no MF6 law states; "
                        f"MF6/MT{mt} is not written")
            return None
        za, lip = _zap(product.pid)
        # FUDGE's rule (gndsToENDF6.toENDF6_MF6): a level reaction's MT already
        # says which level the residual is left in, so LIP is 0 there.
        if 50 <= mt <= 91 or 600 <= mt <= 850:
            lip = 0
        awp = _awp(context.suite, product.pid, report)
        y = _yield(product.multiplicity)
        if awp is None or y is None:
            report.lost(f"MF6/MT{mt}: product {product.label!r} has no "
                        f"{'mass in PoPs' if awp is None else 'tabulated multiplicity'}; "
                        f"MF6/MT{mt} is not written")
            return None
        lawNumber, fields = law
        if lawNumber == 6:
            mass = getattr(form.energy, "mass", None)
            if mass is None:
                report.lost(f"MF6/MT{mt}: the N-body phase space of {product.label!r} "
                            f"states no total mass (APSX); MF6/MT{mt} is not written")
                return None
            from kika._constants import NEUTRON_MASS_AMU

            fields["apsx"] = float(mass.value) / NEUTRON_MASS_AMU
        pairs, ex, ey = y
        records.append({
            "index": index, "zap": float(za), "awp": float(awp), "lip": int(lip),
            "law": int(lawNumber), "y_interp": pairs, "y_energies": ex, "y_values": ey,
            "pid": product.pid, "label": product.label, "modelled": True,
            "law_fields": fields,
        })
        frame = getattr(form, "productFrame", None)
        if frame is not None:
            # ENDF-6's LCT=3 split is on the mass number, A <= 4, a photon
            # included (``energy_angle.frameForProduct``).
            frames.append((int(za) % 1000 <= 4, frame))
    if not records:
        return None
    lab = {f == Frame.lab for _, f in frames}
    if lab == {True}:
        lct = 1
    elif lab <= {False}:
        lct = 2
    elif all((f == Frame.centerOfMass) == light for light, f in frames):
        lct = 3
    else:
        report.lost(f"MF6/MT{mt}: its products' frames are mixed in a way no LCT "
                    f"states; MF6/MT{mt} is not written")
        return None
    return {"mat": context.mat, "za": context.za, "awr": context.awr, "jp": 0,
            "lct": lct, "nk": len(records), "pad": {}, "products": records}

"""E5e: MF12-15 bookkeeping from the model, for a suite read from GNDS.

``photons.encodePhotonSections`` writes MF12-15 from blocks on a reaction's
provenance (``"mf12"``/``"mf13"``, ``"mf14"``, ``"mf15"``). A suite read from
ENDF has them from the file; this module builds them for one that never saw
ENDF, by FUDGE's rules (``toENDF6/gndsToENDF6.py`` ``gammasToENDF6_MF12_13``
and ``toENDF6/reactionSuite.py`` ``addDecayGamma``):

- **a level's cascade** (the residual's photon is a ``branching1d`` and PoPs
  has the level's ``decayData``) is MF12 LO=2: LG = 2 when the transitions
  carry photon-emission probabilities, NS = the level index, LP = 0 (FUDGE does
  not keep LP), the transitions in decreasing final-level energy; MF14 LI=1;
- **every other photon** is MF12 LO=1 -- or MF13 when FUDGE's flag ``MF13``
  says the file stated a production cross section -- with ``ES`` from the
  ``ESk=`` flag, LP = 2 for a primary, 1 for a line with ES ≠ 0, else 0, LF = 1
  for the continuum (whose spectrum is MF15, weight 1 over its incident range)
  and 2 for a line; sorted by decreasing (EG, ES), the total multiplicity first
  when there are several; MF14 is LI=1 when all are isotropic, else LTT=1 with
  the isotropic ones first;
- a photon FUDGE's flag sends to **MF6** is G4c's.

Improvements over FUDGE, declared: the decision is per photon rather than from
the first of the list, and NK of MF14 counts only the photons written.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["photonFields"]


def _flag(context, reaction, product, channelPath) -> str:
    return context.conversionFlags.get(f"{channelPath}/products/product[@label='{product.label}']", "")


def _form(product):
    from kika.nuclear_data.model import EVAL_LABEL

    distribution = getattr(product, "distribution", None)
    if distribution is None or EVAL_LABEL not in distribution:
        return None
    return distribution[EVAL_LABEL]


def _hosts(reaction, reactionPath):
    """``(channel, host label, channel path)``: the reaction's own, then each residual's decay."""
    yield reaction.outputChannel, None, f"{reactionPath}/outputChannel"
    for product in reaction.outputChannel.products:
        if product.pid != "photon" and product.outputChannel is not None:
            yield (product.outputChannel, product.label,
                   f"{reactionPath}/outputChannel/products/product[@label='{product.label}']/outputChannel")


def _base(context) -> dict:
    return {"mat": context.mat, "za": context.za, "awr": context.awr, "pad": {}}


def _cascade(reaction, context, report) -> Optional[dict]:
    """MF12 LO=2 (+MF14) for a level reaction whose residual's decay is a cascade."""
    from kika.nuclear_data.model import Branching1d, pidFromZA
    from kika.nuclear_data.model.pops import zaFromPid

    from ..residuals import levelSeries

    series = levelSeries(int(reaction.ENDF_MT))
    if series is None:
        return None
    for product in reaction.outputChannel.products:
        decay = product.outputChannel
        if product.pid == "photon" or decay is None:
            continue
        photons = [p for p in decay.products if p.pid == "photon"
                   and isinstance(getattr(p.multiplicity, "form", None), Branching1d)]
        if not photons:
            continue
        level = pidFromZA(zaFromPid(product.pid), series[2])
        particle = context.suite.PoPs.particles.get(level) or context.suite.PoPs.particles.get(product.pid)
        data = getattr(particle, "decayData", None)
        if data is None or not len(data.decayModes):
            report.lost(f"MF12/MT{reaction.ENDF_MT}: the residual {product.pid} points "
                        f"at a cascade (branching1d) and PoPs has no decay for it")
            return None
        lg = 2 if any(m.photonEmissionProbabilities is not None for m in data.decayModes) else 1
        finals = []
        for mode in data.decayModes:
            final = mode.finalState()
            energy = getattr(context.suite.PoPs.particles.get(final), "energy", None)
            finals.append((final, float(energy.value) if energy is not None else 0.0))
        return {
            "mf12": {**_base(context), "lo": 2, "lg": lg, "ns": series[2], "n2": 0,
                     "listC2": 0.0, "lp": 0, "listL2": 0, "nt": len(finals),
                     "level": particle.id, "host": product.label, "finals": finals,
                     "lines": [], "digest": "", "sortDescending": True},
            "mf14": {**_base(context), "li": 1, "ltt": 0, "nk": len(finals), "ni": 0,
                     "isotropic": [], "anisotropic": []},
        }
    return None


def _esFlag(flag: str) -> float:
    for item in flag.split(","):
        key, _, value = item.strip().partition("=")
        if key == "ESk":
            return float(value)
    return 0.0


def _sum(multiplicities):
    """The total of several multiplicities, lin-lin on the union of their grids."""
    from kika.algebra.evaluate import sample_on_union
    from kika.algebra.grid import union
    from kika.nuclear_data.model import EVAL_LABEL, XYs1d, multiplicityAxes

    from ..photons import _linlin

    tables = [_linlin(m.form) for m in multiplicities]
    grid = union([x for x, _ in tables])
    total = np.zeros_like(grid)
    for x, y in tables:
        inside = (grid >= x[0]) & (grid <= x[-1])
        total[inside] += sample_on_union(x, y, 2, grid[inside])
    return XYs1d(xs=grid, ys=total, axes=multiplicityAxes(), label=EVAL_LABEL)


def _lineFields(reaction, context, report, reactionPath, inMF6=False) -> Optional[dict]:
    """MF12 LO=1 or MF13 (+MF14, MF15) for the photons that are not a cascade."""
    from kika.nuclear_data.model import (DiscreteGamma, Isotropic2d, Legendre, Multiplicity,
                                         PrimaryGamma, Uncorrelated, toEndfTab2)
    from kika.nuclear_data.model.sums import Add, MultiplicitySum

    mt = int(reaction.ENDF_MT)
    for channel, host, channelPath in _hosts(reaction, reactionPath):
        if inMF6 and host is not None:
            # A residual's decay lines of a reaction written in MF6 are in its
            # MF6 section (G4c), as FUDGE writes them.
            continue
        photons = []
        for product in channel.products:
            form = _form(product)
            if product.pid != "photon" or not isinstance(form, Uncorrelated):
                continue
            flag = _flag(context, reaction, product, channelPath)
            if flag.split(",")[0].strip() == "MF6":
                continue
            photons.append((product, form, flag))
        if not photons:
            continue
        mf = 13 if any(f.split(",")[0].strip() == "MF13" for _, _, f in photons) else 12
        records, continuum = [], None
        for product, form, flag in photons:
            es = _esFlag(flag)
            if isinstance(form.energy, PrimaryGamma):
                eg, lp, lf = float(form.energy.value), 2, 2
            elif isinstance(form.energy, DiscreteGamma):
                eg, lp, lf = float(form.energy.value), (1 if es else 0), 2
            else:
                eg, lp, lf = 0.0, 0, 1
                continuum = product
            records.append({"label": product.label, "eg": eg, "es": es, "lp": lp, "lf": lf,
                            "_form": form, "_product": product})
        records.sort(key=lambda r: (r["eg"], r["es"]), reverse=True)
        out = {}
        total = None
        if len(records) > 1:
            existing = context.suite.sums.multiplicitySums.byENDF_MT(mt)
            if existing is None or existing.multiplicity is None:
                existing = MultiplicitySum(label=f"{reaction.label} total gamma multiplicity",
                                           multiplicity=Multiplicity(form=_sum(
                                               [r["_product"].multiplicity for r in records])),
                                           ENDF_MT=mt)
                for r in records:
                    existing.summands.append(Add(href=f"{channelPath}/products/product"
                                                      f"[@label='{r['label']}']/multiplicity"))
                context.suite.sums.multiplicitySums.append(existing)
                report.approximated(f"MF{mf}/MT{mt}: the total photon multiplicity is not "
                                    f"in the suite; it is the sum of the photons', lin-lin "
                                    f"on the union of their grids (FUDGE recomputes it too)")
            total = [0.0, 0.0, 0, 0]
        block = {**_base(context), "l1": 1 if mf == 12 else 0, "l2": 0, "n2": 0,
                 "host": host, "total": total,
                 "photons": [{k: r[k] for k in ("label", "eg", "es", "lp", "lf")} for r in records]}
        if mf == 13:
            block.update(lines=[], sigma="", digests=[], totalDigest=None)
            block["grids"], block["interps"] = [], []
            from ..photons import _tab1Of

            for r in records:
                pairs, x, _ = _tab1Of(r["_product"].multiplicity.form)
                block["grids"].append([float(v) for v in x])
                block["interps"].append([list(map(int, p)) for p in pairs])
            if total is not None:
                tform = context.suite.sums.multiplicitySums.byENDF_MT(mt).multiplicity.form
                pairs, x, _ = _tab1Of(tform)
                block["totalGrid"] = [float(v) for v in x]
                block["totalInterp"] = [list(map(int, p)) for p in pairs]
            else:
                block["totalGrid"] = block["totalInterp"] = None
        out[f"mf{mf}"] = block

        isotropic = [r for r in records if isinstance(r["_form"].angular, Isotropic2d)]
        anisotropic = [r for r in records if not isinstance(r["_form"].angular, Isotropic2d)]
        mf14 = {**_base(context), "li": 1 if not anisotropic else 0, "ltt": 0 if not anisotropic else 1,
                "nk": len(records), "ni": 0 if not anisotropic else len(isotropic),
                "isotropic": [], "anisotropic": []}
        if anisotropic:
            for r in isotropic:
                mf14["isotropic"].append({"label": r["label"], "eg": r["eg"], "es": r["es"],
                                          "rest": [0, 0, 0, 0]})
            for r in anisotropic:
                functions, _ = toEndfTab2(r["_form"].angular)
                if not all(isinstance(f, Legendre) for f in functions):
                    report.lost(f"MF14/MT{mt}: photon {r['label']!r} has a tabulated angular "
                                f"distribution; FUDGE writes LTT=1 only, and so does this")
                    return None
                mf14["anisotropic"].append({"label": r["label"], "eg": r["eg"], "es": r["es"],
                                            "l1": 0, "l2": 0,
                                            "nodes": [(0.0, 0, 0, 0) for _ in functions]})
        out["mf14"] = mf14

        if continuum is not None:
            energy = _form(continuum).energy
            functions, _ = toEndfTab2(energy)
            lo, hi = float(functions[0].outerDomainValue), float(functions[-1].outerDomainValue)
            out["mf15"] = {**_base(context), "l1": 0, "l2": 0, "n2": 0, "label": continuum.label,
                           "weight": {"c1": 0.0, "c2": 0.0, "l1": 0, "l2": 1,
                                      "interp": [[2, 2]], "x": [lo, hi], "y": [1.0, 1.0]},
                           "tab2": [0.0, 0.0, 0, 0],
                           "nodes": [[0.0, 0, 0] for _ in functions]}
        return out
    return None


def photonFields(reaction, context, report, reactionPath, inMF6=False) -> dict:
    """The MF12-15 blocks for *reaction* (or an orphan product), possibly empty."""
    if getattr(reaction, "ENDF_MT", None) is None or getattr(reaction, "outputChannel", None) is None:
        return {}
    cascade = _cascade(reaction, context, report)
    if cascade is not None:
        return cascade
    return _lineFields(reaction, context, report, reactionPath, inMF6) or {}

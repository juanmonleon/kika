"""MF2/151 for a suite whose resonances came without ENDF bookkeeping.

Three cases are derived, as FUDGE writes them
(``brownies/legacy/toENDF6/resonances``).

**The scattering radius alone** (no ``resolved`` or ``unresolved`` region) is
ENDF's LRU=0 range: LRP=0, one isotope, one range whose only parameters are the
target spin and AP (ENDF-102 §2.1).

**Reich-Moore resolved ranges** are LRF=3. GNDS groups them by (L, J, channel
spin); :func:`~kika.endf.model_adapter.resonances.reichMooreLBlocks` regroups
them by L when the tape is written. Per range:

==============  ===========================================================
``EL``/``EH``   the region's domain
``SPI``         the target's spin: the formalism's PoPs, which must agree
                with the suite's
``AP``          the region's radius, else the formalism's, else the
                evaluation's; a radius that depends on energy is refused
``APL``         the entrance channel's radius where it differs from AP
``AWRI``        the formalism's target mass over the neutron's; MF1's AWR,
                reported, when the formalism states none
``NAPS``        ``calculateChannelRadius``
``AJ``          J, negative for the lower channel spin
``LAD``         0, reported: the model has no node for whether MF4 may be
                computed from the parameters
``NLSC``        ``angularLCount``, else 0, reported
``ZAI``/``ABN`` the target's ZA and 1; ``LFW`` 0, as no unresolved range is
                written
==============  ===========================================================

**Breit-Wigner resolved ranges** are LRF=1/2, already one spin group per L. The
fields are the same, less APL, AJ's sign, LAD and NLSC, which LRF=1/2 does not
have; QX and LRX come from the group's competitive channel, else 0, reported
where GT is not GN+GG+GF.

Not derived, and reported lost: R-Matrix Limited ranges (MF2 is then not
written at all, since a resolved range cannot be skipped) and the unresolved
range (the resolved ones are written without it).
"""
from __future__ import annotations

from typing import Optional

from . import DERIVED, DerivationContext, register

__all__ = ["deriveResonances"]


def _spinOf(particle) -> Optional[float]:
    spin = getattr(particle, "spin", None)
    value = getattr(spin, "value", spin)
    return None if value is None else float(value)


def _targetSpin(suite) -> Optional[float]:
    pops = getattr(suite, "PoPs", None)
    return _spinOf(pops.get(suite.target) if pops is not None else None)


def _domain(radius, suite, report):
    if radius.domainMin is not None and radius.domainMax is not None:
        return float(radius.domainMin), float(radius.domainMax)
    from ..mf1_header import _evaluatedStyle

    style = _evaluatedStyle(suite)
    domain = getattr(style, "projectileEnergyDomain", None)
    if domain is None:
        return None
    report.approximated(
        f"MF2/151: the scattering radius states no energy domain, so its range "
        f"is written over the evaluation's projectile domain "
        f"{domain.min:g}-{domain.max:g} eV")
    return float(domain.min), float(domain.max)


def _provenance(context, regions):
    from kika.nuclear_data.model import EndfProvenance

    header = {"nis": 1,
              "isotopes": [{"za": context.za, "abn": 1.0, "lfw": 0, "ner": len(regions)}],
              "regions": regions}
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header)


def _radiusOnly(resonances, context, report):
    from kika.nuclear_data.model.resonances import radiusToEndf

    radius = resonances.scatteringRadius
    if radius is None:
        return None
    if radius.isEnergyDependent or radius.constant is None:
        report.lost(
            "MF2/151: the scattering radius depends on energy and there is no "
            "resonance range for it to belong to; an LRU=0 range states a "
            "constant AP, so MF2 is not written")
        return None
    suite = context.suite
    spin = _targetSpin(suite)
    if spin is None:
        report.lost(
            f"MF2/151: PoPs states no spin for the target {suite.target!r}, and "
            f"SPI is not assumed, so MF2 is not written")
        return None
    domain = _domain(radius, suite, report)
    if domain is None:
        report.lost("MF2/151: neither the scattering radius nor the evaluation "
                    "states an energy domain, so MF2 is not written")
        return None
    region = {"kind": "radiusOnly", "lru": 0, "lrf": 0, "nro": 0, "naps": 0,
              "el": domain[0], "eh": domain[1], "spi": spin,
              "ap": float(radiusToEndf(radius.constant)), "isotope_index": 0}
    return _provenance(context, [region])


def _rangeSpin(formalism, context, where) -> float:
    suiteSpin = _targetSpin(context.suite)
    pops = formalism.PoPs
    particles = list(pops.particles.values()) if pops is not None else []
    rangeSpin = _spinOf(particles[0]) if len(particles) == 1 else None
    if rangeSpin is not None and suiteSpin is not None and rangeSpin != suiteSpin:
        raise ValueError(f"{where} states target spin {rangeSpin:g} and PoPs {suiteSpin:g}")
    spin = rangeSpin if rangeSpin is not None else suiteSpin
    if spin is None:
        raise ValueError(f"{where}: no target spin is stated, and SPI is not assumed")
    return spin


def _fallbackAwri(formalism, groups, context, report, lrf, where) -> Optional[float]:
    """The range's AWRI: its target's mass over the suite's neutron, MF1's AWR
    when it states no mass, or ``None`` when every block states its own."""
    from ..resonances import formalismAtomicWeightRatio
    from ..mf1_header import _massAmu

    neutron = context.suite.PoPs.particles.get("n") if context.suite.PoPs is not None else None
    stated = formalismAtomicWeightRatio(formalism, _massAmu(neutron))
    if stated is not None:
        return stated
    if all(g.atomicWeightRatio is not None for g in groups):
        return None
    if context.awr is None:
        raise ValueError(f"{where}: no target mass to write AWRI from")
    report.approximated(
        f"MF2/151 LRF={lrf}, {where}: the range states no target mass, so AWRI "
        f"is MF1's AWR {float(context.awr):g}")
    return float(context.awr)


def _breitWignerRange(resonances, region, context, report, where) -> dict:
    """The fields of one LRF=1/2 range; raises ``ValueError`` saying why it cannot be one."""
    import math

    from kika.nuclear_data.model.resonances import BreitWignerApproximation, radiusToEndf
    from ..resonances import resolvedRangeRadius

    formalism = region.formalism
    lrf = {BreitWignerApproximation.singleLevel: 1,
           BreitWignerApproximation.multiLevel: 2}[BreitWignerApproximation(formalism.approximation)]
    spin = _rangeSpin(formalism, context, where)
    radius = resolvedRangeRadius(resonances, region)
    policy = formalism.radiusPolicy
    if radius is None or (policy is not None and policy.phaseRadius is not None):
        raise ValueError(f"{where} states no constant scattering radius for AP")
    if policy is not None and policy.channelMode == "constant":
        raise ValueError(f"{where} states a channel radius of its own (NAPS=2)")
    groups = formalism.resonanceParameters.spinGroups
    if len({g.L for g in groups}) != len(groups):
        raise ValueError(f"{where} has two spin groups of one L")
    for group in groups:
        if group.scatteringRadius is not None and group.scatteringRadius != radius:
            raise ValueError(f"{where}: L={group.L} states its own radius, which "
                             f"LRF={lrf} has no field for")
    awri = _fallbackAwri(formalism, groups, context, report, lrf, where)
    unlisted = sum(1 for g in groups if g.competitiveChannel is None for r in g.resonances
                   if r.totalWidth and not math.isclose(
                       r.totalWidth, r.neutronWidth + r.captureWidth + r.fissionWidth,
                       rel_tol=5e-6, abs_tol=1e-12))
    if unlisted:
        report.approximated(
            f"MF2/151 LRF={lrf}, {where}: {unlisted} resonance(s) have GT unequal "
            f"to GN+GG+GF and the model states no competitive channel, so LRX is "
            f"written 0 and the difference stays in GT as the model has it")
    calculate = (policy.channelMode == "mass" if policy is not None
                 else bool(formalism.calculateChannelRadius))
    return {"kind": "resolved", "lru": 1, "lrf": lrf, "nro": 0, "naps": 0 if calculate else 1,
            "el": float(region.domainMin), "eh": float(region.domainMax),
            "spi": spin, "ap": float(radiusToEndf(radius)), "nls": len(groups),
            "nlsc": 0, "lad": 0, "abn": 1.0, "isotope_index": 0,
            **({} if awri is None else {"awri": awri}),
            "l_blocks": [{"l": g.L,
                          "lrx": 0 if g.competitiveChannel is None else 1,
                          "qx": 0.0 if g.competitiveChannel is None else g.competitiveChannel.Q,
                          "num_resonances": len(g.resonances)} for g in groups]}


def _resolvedRange(resonances, region, context, report) -> dict:
    from kika.nuclear_data.model.resonances import BreitWigner, RMatrix

    where = f"the resolved range {region.domainMin:g}-{region.domainMax:g} eV"
    if isinstance(region.formalism, BreitWigner):
        return _breitWignerRange(resonances, region, context, report, where)
    if isinstance(region.formalism, RMatrix):
        return _reichMooreRange(resonances, region, context, report, where)
    raise ValueError(f"{where} is {type(region.formalism).__name__}, whose ENDF "
                     f"bookkeeping kika does not derive")


def _reichMooreRange(resonances, region, context, report, where) -> dict:
    """The fields of one LRF=3 range; raises ``ValueError`` saying why it cannot be one."""
    from kika.nuclear_data.model.resonances import radiusToEndf
    from ..resonances import (formalismAtomicWeightRatio, reichMooreLBlocks,
                              resolvedRangeRadius)

    formalism = region.formalism
    if formalism.approximation != "ReichMoore":
        raise ValueError(f"{where} is an R-Matrix Limited range (LRF=7), whose ENDF "
                         f"bookkeeping kika does not derive yet")
    spin = _rangeSpin(formalism, context, where)
    radius = resolvedRangeRadius(resonances, region)
    if radius is None:
        raise ValueError(f"{where} states no constant scattering radius for AP")
    awri = _fallbackAwri(formalism, formalism.spinGroups, context, report, 3, where)
    try:
        blocks = reichMooreLBlocks(formalism, radius, spin,
                                   awri if awri is not None else formalismAtomicWeightRatio(formalism))
    except ValueError as why:
        raise ValueError(f"{where} cannot be LRF=3: {why}") from None
    policy = formalism.radiusPolicy
    calculate = (policy.channelMode == "mass" if policy is not None
                 else bool(formalism.calculateChannelRadius))
    nlsc = formalism.angularLCount
    report.approximated(
        f"MF2/151 LRF=3, {where}: LAD is written 0 (the model has no node for "
        f"whether MF4 may be computed from the parameters)"
        + ("" if nlsc is not None else
           " and NLSC 0 (no number of l-values for angular convergence is stated)"),
        unaffectedScopes=("cross-sections",))
    return {"kind": "resolved", "lru": 1, "lrf": 3, "nro": 0, "naps": 0 if calculate else 1,
            "el": float(region.domainMin), "eh": float(region.domainMax),
            "spi": spin, "ap": float(radiusToEndf(radius)), "nls": len(blocks),
            "nlsc": 0 if nlsc is None else int(nlsc), "lad": 0, "abn": 1.0,
            "isotope_index": 0, **({} if awri is None else {"awri": awri}),
            "l_blocks": [{"l": b["l"], "lrx": 0, "qx": 0.0, "num_resonances": len(b["rows"])}
                         for b in blocks]}


def deriveResonances(resonances, path, context: DerivationContext, report):
    """The provenance :func:`encodeMF2MT151` writes MF2/151 from, or ``None``."""
    if not resonances.resolved and resonances.unresolved is None:
        return _radiusOnly(resonances, context, report)
    regions = []
    for region in resonances.resolved:
        try:
            regions.append(_resolvedRange(resonances, region, context, report))
        except ValueError as why:
            report.lost(f"MF2/151: {why}, so MF2 is not written")
            return None
    if resonances.unresolved is not None:
        urr = resonances.unresolved
        selfShielding = getattr(urr.tabulatedWidths, "selfShieldingOnly", False)
        report.lost(
            f"MF2/151: the unresolved range {urr.domainMin:g}-{urr.domainMax:g} eV "
            f"is not written, as kika does not derive its ENDF bookkeeping yet"
            + ("" if selfShielding else
               "; MF3 there is a background without its resonance part"))
    if not regions:
        return None
    return _provenance(context, regions)


def _isResonances(node) -> bool:
    from kika.nuclear_data.model import Resonances

    return isinstance(node, Resonances)


register("resonances", _isResonances, deriveResonances)

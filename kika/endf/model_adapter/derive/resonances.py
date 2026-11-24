"""MF2/151 for a suite whose resonances came without ENDF bookkeeping.

Two cases are derived, both as FUDGE writes them
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

Not derived, and reported lost: Breit-Wigner and R-Matrix Limited ranges (MF2
is then not written at all, since a resolved range cannot be skipped) and the
unresolved range (the resolved ones are written without it).
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


def _reichMooreRange(resonances, region, context, report) -> dict:
    """The fields of one LRF=3 range; raises ``ValueError`` saying why it cannot be one."""
    from kika.nuclear_data.model.resonances import RMatrix, radiusToEndf
    from ..resonances import (formalismAtomicWeightRatio, reichMooreLBlocks,
                              resolvedRangeRadius)

    formalism = region.formalism
    where = f"the resolved range {region.domainMin:g}-{region.domainMax:g} eV"
    if not isinstance(formalism, RMatrix):
        raise ValueError(f"{where} is {type(formalism).__name__}, whose ENDF "
                         f"bookkeeping kika does not derive yet")
    suiteSpin = _targetSpin(context.suite)
    pops = formalism.PoPs
    particles = list(pops.particles.values()) if pops is not None else []
    rangeSpin = _spinOf(particles[0]) if len(particles) == 1 else None
    if rangeSpin is not None and suiteSpin is not None and rangeSpin != suiteSpin:
        raise ValueError(f"{where} states target spin {rangeSpin:g} and PoPs {suiteSpin:g}")
    spin = rangeSpin if rangeSpin is not None else suiteSpin
    if spin is None:
        raise ValueError(f"{where}: no target spin is stated, and SPI is not assumed")
    radius = resolvedRangeRadius(resonances, region)
    if radius is None:
        raise ValueError(f"{where} states no constant scattering radius for AP")
    awri = formalismAtomicWeightRatio(formalism)
    if awri is None and any(g.atomicWeightRatio is None for g in formalism.spinGroups):
        if context.awr is None:
            raise ValueError(f"{where}: no target mass to write AWRI from")
        awri = float(context.awr)
        report.approximated(
            f"MF2/151 LRF=3, {where}: the range states no target mass, so AWRI "
            f"is MF1's AWR {awri:g}")
    try:
        blocks = reichMooreLBlocks(formalism, radius, spin, awri)
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
            "isotope_index": 0, "awri": awri,
            "l_blocks": [{"l": b["l"], "lrx": 0, "qx": 0.0, "num_resonances": len(b["rows"])}
                         for b in blocks]}


def deriveResonances(resonances, path, context: DerivationContext, report):
    """The provenance :func:`encodeMF2MT151` writes MF2/151 from, or ``None``."""
    if not resonances.resolved and resonances.unresolved is None:
        return _radiusOnly(resonances, context, report)
    regions = []
    for region in resonances.resolved:
        try:
            regions.append(_reichMooreRange(resonances, region, context, report))
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

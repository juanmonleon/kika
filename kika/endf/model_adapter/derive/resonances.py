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
``ZAI``/``ABN`` the target's ZA and 1
==============  ===========================================================

**Breit-Wigner resolved ranges** are LRF=1/2, already one spin group per L. The
fields are the same, less APL, AJ's sign, LAD and NLSC, which LRF=1/2 does not
have; QX and LRX come from the group's competitive channel, else 0, reported
where GT is not GN+GG+GF.

**The unresolved range** is LRU=2. LRF is 2 unless FUDGE's ``LRF1`` flag says
otherwise, and the isotope's LFW is 1 when the suite has fission, unless an
``LFW<n>`` flag says otherwise (FUDGE's rule: the GNDS of a non-fissile tape
with LFW=1 carries ``LFW1``). LRF and LFW pick case A, B or C. Widths are
matched to ENDF's columns by their reaction's MT; a column the region does not
state is written 0 with 0 degrees of freedom. INT is the level spacing's
interpolation; AWRI the region's target mass, as for the resolved ranges.

Not derived, and reported lost: R-Matrix Limited ranges (MF2 is then not
written at all, since a resolved range cannot be skipped) and an unresolved
range that its case cannot state (the resolved ones are written without it).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

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


def _provenance(context, regions, lfw: int = 0):
    from kika.nuclear_data.model import EndfProvenance

    header = {"nis": 1,
              "isotopes": [{"za": context.za, "abn": 1.0, "lfw": lfw, "ner": len(regions)}],
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
    """MF1's AWR for the blocks the model gives no mass, or ``None`` when it gives all one."""
    from ..resonances import formalismAtomicWeightRatio

    if formalismAtomicWeightRatio(formalism) is not None             or all(g.atomicWeightRatio is not None for g in groups):
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
                                   formalismAtomicWeightRatio(formalism) or awri)
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


def _hasFission(suite) -> bool:
    from ..mf1_header import _FISSION_MTS

    return (any(getattr(r, "ENDF_MT", None) in _FISSION_MTS for r in suite.reactions)
            or bool(len(getattr(suite, "fissionComponents", ()) or ())))


def _unresolvedFlags(widths, context) -> dict:
    """FUDGE's ``LRF<n>``/``LFW<n>`` notes on the region, as ``{"LRF": n, "LFW": n}``."""
    href = (f"/reactionSuite/resonances/unresolved/tabulatedWidths"
            f"[@label='{widths.label or 'eval'}']")
    flags = {}
    for item in context.conversionFlags.get(href, "").split(","):
        item = item.strip()
        for key in ("LRF", "LFW"):
            if item.startswith(key) and item[len(key):].isdigit():
                flags[key] = int(item[len(key):])
    return flags


def _unresolvedAwri(widths, context, report, where) -> Optional[float]:
    """AWRI as FUDGE writes it: the target's mass over the neutron's, both from PoPs."""
    from kika._constants import NEUTRON_MASS_AMU

    from ..mf1_header import _massAmu

    if all(g.atomicWeightRatio is not None for g in widths.spinGroups):
        return None
    suite = context.suite
    pops = getattr(suite, "PoPs", None)
    own = list(widths.PoPs.particles.values()) if widths.PoPs is not None else []
    target = _massAmu(own[0]) if len(own) == 1 else None
    if target is None and pops is not None:
        target = _massAmu(pops.get(suite.target))
    if target is None:
        if context.awr is None:
            raise ValueError(f"{where}: no target mass to write AWRI from")
        report.approximated(f"MF2/151 LRU=2, {where}: PoPs states no target mass, so "
                            f"AWRI is MF1's AWR {float(context.awr):g}")
        return float(context.awr)
    neutron = _massAmu(pops.get("n")) if pops is not None else None
    return target / (neutron if neutron is not None else NEUTRON_MASS_AMU)


def _isConstant(values) -> bool:
    return values is not None and np.asarray(values).size == 1


def _unresolvedRange(resonances, context, report):
    """The fields of the LRU=2 range and the isotope's LFW; raises ``ValueError`` saying why not."""
    from kika.nuclear_data.model.enums import INTERPOLATION_TO_ENDF_INT, Interpolation
    from ..resonances import unresolvedChannelRoles

    urr = resonances.unresolved
    widths = urr.tabulatedWidths
    where = f"the unresolved range {urr.domainMin:g}-{urr.domainMax:g} eV"
    if widths is None:
        raise ValueError(f"{where} has no tabulatedWidths")
    if widths.scatteringRadius is None:
        raise ValueError(f"{where} states no constant scattering radius for AP")
    flags = _unresolvedFlags(widths, context)
    lrf = flags.get("LRF", 2)
    lfw = flags.get("LFW", int(_hasFission(context.suite)))
    if lrf not in (1, 2):
        raise ValueError(f"{where} is flagged LRF={lrf}, which LRU=2 does not have")
    case = "C" if lrf == 2 else ("B" if lfw else "A")
    spin = _rangeSpin(widths, context, where)
    awri = _unresolvedAwri(widths, context, report, where)
    roles = unresolvedChannelRoles(widths)
    states = []
    for group in widths.spinGroups:
        here = f"{where}, L={group.L} J={group.J:g}"
        present = {roles[c.label] for c in group.channels}
        if case == "C":
            interpolation = getattr(group.levelSpacingFunction, "interpolation", None)
            if group.crossSectionInterpolation is None and interpolation is None:
                raise ValueError(f"{here}: the level spacing is not one table, so it "
                                 f"states no INT for case C")
            states.append({"l": group.L, "j": group.J, "int_code": (
                None if interpolation is None
                else INTERPOLATION_TO_ENDF_INT[Interpolation(interpolation)])})
            continue
        constant = {roles[c.label]: c for c in group.channels if c.constantWidth is not None}
        if not _isConstant(group.levelSpacing) or not {"neutron", "capture"} <= set(constant):
            raise ValueError(f"{here}: case {case} (LRF=1) states one D, GN0 and GG per J, "
                             f"and the model tabulates them")
        if present - {"neutron", "capture", "fission"} or (case == "A" and "fission" in present):
            raise ValueError(f"{here}: case {case} has no column for "
                             f"{sorted(present - {'neutron', 'capture'})}")
        if case == "B" and "fission" not in present:
            raise ValueError(f"{here}: case B (LFW=1) needs a fission width")
        states.append({"l": group.L, "j": group.J})
    region = {"kind": "unresolved", "lru": 2, "lrf": lrf, "nro": 0, "naps": 0,
              "el": float(urr.domainMin), "eh": float(urr.domainMax), "spi": spin,
              "nls": len({g.L for g in widths.spinGroups}), "isotope_index": 0,
              "urr_case": case, "j_states": states,
              **({} if awri is None else {"awri": awri})}
    return region, lfw


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
    lfw = 0
    if resonances.unresolved is not None:
        try:
            region, lfw = _unresolvedRange(resonances, context, report)
            regions.append(region)
        except ValueError as why:
            selfShielding = getattr(resonances.unresolved.tabulatedWidths,
                                    "selfShieldingOnly", False)
            report.lost(f"MF2/151: {why}, so the range is not written"
                        + ("" if selfShielding else
                           "; MF3 there is a background without its resonance part"))
    if not regions:
        return None
    return _provenance(context, regions, lfw)


def _isResonances(node) -> bool:
    from kika.nuclear_data.model import Resonances

    return isinstance(node, Resonances)


register("resonances", _isResonances, deriveResonances)

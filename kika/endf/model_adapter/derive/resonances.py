"""MF2/151 for a suite whose resonances came without ENDF bookkeeping.

Three cases are derived, as FUDGE writes them
(``brownies/legacy/toENDF6/resonances``).

**The scattering radius alone** (no ``resolved`` or ``unresolved`` region) is
ENDF's LRU=0 range: LRP=0, one isotope, one range whose only parameters are the
target spin and AP (ENDF-102 §2.1).

**Reich-Moore resolved ranges flagged** ``LRF3`` are LRF=3; any other
``RMatrix`` range is LRF=7, as FUDGE decides. GNDS groups LRF=3 by (L, J, channel
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

**R-Matrix Limited ranges** (LRF=7) follow FUDGE's ``writeRMatrixParticlePairs``
and ``writeRMatrixSpinGroupHeader``: SPI=AP=0, NAPS=1, KRM from the
approximation. Each pair is the ejectile and the residual of its linked reaction
(the projectile and target for elastic and fission; for a capture that lists only
its photons, the compound nucleus), their masses over the file's own
neutron, Z, and spin and parity in ENDF's sign convention (0, 0 for a capture
residual with none); Q and PNT are 0 for capture and fission; SHF is 0 when the
shift is eliminated, 2 for Brune, else 1. A channel's APT is its radius, else
its reaction's, the range's or the evaluation's; APE its hard-sphere radius,
else its reaction's, else APT; both 0 for capture. BND is the channel's, else 0
or -L by the boundary condition. Unlike FUDGE, Q is the linked channel's as
stated: it already leaves the residual in its level.

Not derived, and reported lost: a resolved range none of these can state (MF2 is
then not written at all, since a resolved range cannot be skipped) and an
unresolved range that its case cannot state (the resolved ones are written
without it).
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
        from kika.gnds.endf_conversion import rMatrixHref

        href = rMatrixHref(context.suite, resonances.resolved.index(region))
        if "LRF3" in context.conversionFlags.get(href, "").split(","):
            return _reichMooreRange(resonances, region, context, report, where)
        return _rMatrixLimitedRange(resonances, region, context, report, where)
    raise ValueError(f"{where} is {type(region.formalism).__name__}, whose ENDF "
                     f"bookkeeping kika does not derive")


def _reichMooreRange(resonances, region, context, report, where) -> dict:
    """The fields of one LRF=3 range; raises ``ValueError`` saying why it cannot be one."""
    from kika.nuclear_data.model.resonances import radiusToEndf
    from ..resonances import (formalismAtomicWeightRatio, reichMooreLBlocks,
                              resolvedRangeRadius)

    formalism = region.formalism
    if formalism.approximation != "ReichMoore":
        raise ValueError(f"{where} is flagged LRF3 and is not Reich-Moore")
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


def _endfSpin(spin, parity, where):
    """ENDF's (spin, parity) pair: the parity rides on a nonzero spin's sign."""
    if parity is None:
        raise ValueError(f"{where} states no parity")
    return (spin * parity, 0) if spin else (spin, parity)


def _linkedReaction(suite, reaction):
    import re

    match = re.search(r"reaction\[@label='(.*)'\]$", reaction.href or "")
    found = [r for r in suite.reactions
             if match is not None and getattr(r, "label", None) == match.group(1)]
    if len(found) != 1:
        raise ValueError(f"the resonance reaction {reaction.label!r} links to no reaction "
                         f"of the suite")
    return found[0]


def _pairParticle(formalism, suite, pid):
    """FUDGE's ``RMatrix.PoPs[pid]``: the range's own PoPs, else the suite's."""
    own = formalism.PoPs.get(pid) if formalism.PoPs is not None else None
    return own if own is not None else suite.PoPs.get(pid)


def _particlePair(reaction, formalism, context, neutron, where) -> dict:
    """One LRF=7 particle pair, as FUDGE's ``writeRMatrixParticlePairs`` writes it."""
    from ..mf1_header import _FISSION_MTS, _massAmu

    suite = context.suite
    linked = _linkedReaction(suite, reaction)
    mt = reaction.reactionMT if reaction.reactionMT is not None else linked.ENDF_MT
    here = f"{where}, pair {reaction.label!r}"
    if mt in _FISSION_MTS or mt == 2:
        ids = (suite.projectile, suite.target)
    else:
        products = [p.pid for p in linked.outputChannel.products.products]
        residual = [pid for pid in products if pid not in (reaction.ejectile, "photon")]
        target = suite.PoPs.get(suite.target)
        if not residual and mt == 102 and getattr(target, "ZA", None) is not None:
            from kika.nuclear_data.model.pops import pidFromZA

            # A capture that lists only its photons: the residual is the compound nucleus.
            residual = [pidFromZA(target.ZA + 1)]
        if reaction.ejectile is None or len(residual) != 1:
            raise ValueError(f"{here}: no single ejectile and residual to write the pair from")
        ids = (reaction.ejectile, residual[0])
    columns = {}
    for suffix, pid in zip("ab", ids):
        particle = _pairParticle(formalism, suite, pid)
        mass = _massAmu(particle)
        level = getattr(particle, "energy", None)
        if mass is None and level is not None and "_e" in pid:
            from kika.gnds.decode import _AMU_EV

            # FUDGE's mass of a level that states none: the ground state's plus E*/c².
            ground = _massAmu(_pairParticle(formalism, suite, pid.split("_e")[0]))
            if ground is not None:
                mass = ground + float(level.convertedTo("eV").value) / _AMU_EV
        if mass is None:
            raise ValueError(f"{here}: PoPs states no mass for {pid!r}")
        spin = getattr(particle.spin, "value", particle.spin)
        if spin is None or particle.parity is None:
            if not (suffix == "b" and mt == 102):
                raise ValueError(f"{here}: PoPs states no spin and parity for {pid!r}")
            i, p = 0, 0
        else:
            i, p = _endfSpin(float(spin), int(particle.parity), f"{here}, {pid!r}")
        columns.update({f"m{suffix}": mass / neutron, f"z{suffix}": getattr(particle, "Z", None) or 0,
                        f"i{suffix}": i, f"p{suffix}": p})
    if mt in (18, 19, 102):
        q = 0.0
    elif reaction.Q is not None:
        q = float(reaction.Q)
    else:
        # FUDGE subtracts the level again: Fe-57 MT51 would be -28826 eV, the tape -14413.
        stated = linked.outputChannel.Q
        if stated is None or stated.value is None or stated.unit != "eV":
            raise ValueError(f"{here}: no Q value in eV")
        q = float(stated.value)
    condition = formalism.boundaryCondition
    shf = (0 if condition == "EliminateShiftFunction" or reaction.eliminated
           else 2 if condition == "Brune" else 1)
    return {**columns, "q": q, "pnt": 0 if mt in (18, 19, 102) else 1, "shf": shf, "mt": mt}


def _radius(value, where) -> float:
    from kika.nuclear_data.model.resonances import ScatteringRadius

    if isinstance(value, ScatteringRadius):
        if value.isEnergyDependent or value.constant is None:
            raise ValueError(f"{where}: an energy-dependent radius has no LRF=7 field")
        return float(value.constant)
    return float(value)


def _rMatrixLimitedRange(resonances, region, context, report, where) -> dict:
    """The fields of one LRF=7 range, as FUDGE's ``toENDF6`` writes them;
    raises ``ValueError`` saying why it cannot be one."""
    from ..resonances import KRM_TO_APPROXIMATION

    formalism = region.formalism
    krm = {v: k for k, v in KRM_TO_APPROXIMATION.items()}.get(formalism.approximation)
    if krm not in (3, 4):
        raise ValueError(f"{where} has approximation {formalism.approximation!r}, "
                         f"which LRF=7 has no KRM for")
    if any("LBK=0" in text and "/channels/channel" in href
           for href, text in context.conversionFlags.items()):
        raise ValueError(f"{where}: a channel is flagged LBK=0, which kika does not write")
    if formalism.radiusPolicy is not None and formalism.radiusPolicy.phaseRadius is not None:
        raise ValueError(f"{where}: an energy-dependent radius has no LRF=7 field")
    neutron = _fileNeutronMass(context)
    reactions = {r.label: r for r in formalism.resonanceReactions}
    pairs = [_particlePair(r, formalism, context, neutron, where)
             for r in formalism.resonanceReactions]
    fallback = next((r for r in (formalism.scatteringRadius, region.scatteringRadius,
                                 resonances.scatteringRadius) if r is not None), None)
    groups = []
    for group in formalism.spinGroups:
        here = f"{where}, spin group {group.label!r}"
        if group.spin is None:
            raise ValueError(f"{here} states no spin")
        aj, pj = _endfSpin(float(group.spin), group.parity, here)
        radii, bnd = [], []
        for channel in group.channels:
            reaction = reactions.get(channel.resonanceReaction)
            if reaction is None:
                raise ValueError(f"{here}: channel {channel.label!r} names no resonance reaction")
            if reaction.reactionMT == 102:
                radii.append((0.0, 0.0))
            else:
                apt = next((r for r in (channel.scatteringRadius, reaction.scatteringRadius,
                                        fallback) if r is not None), None)
                if apt is None:
                    raise ValueError(f"{here}: channel {channel.label!r} has no radius")
                apt = _radius(apt, here)
                ape = next((r for r in (channel.hardSphereRadius, reaction.hardSphereRadius)
                            if r is not None), apt)
                radii.append((_radius(ape, here), apt))
            if channel.boundaryConditionValue is not None:
                bnd.append(channel.boundaryConditionValue)
            elif formalism.boundaryCondition == "EliminateShiftFunction":
                bnd.append(0.0)
            elif formalism.boundaryCondition == "NegativeOrbitalMomentum":
                bnd.append(-float(channel.L))
            else:
                raise ValueError(f"{here}: boundary condition {formalism.boundaryCondition!r} "
                                 f"has no BND")
        groups.append({"aj": aj, "pj": pj, "kps": 0, "radii": radii, "bnd": bnd})
    return {"kind": "resolved", "lru": 1, "lrf": 7, "nro": 0,
            # FUDGE's rule: LRF=7 never computes radii.
            "naps": 1,
            "el": float(region.domainMin), "eh": float(region.domainMax),
            "spi": 0.0, "ap": 0.0, "nls": 0, "nlsc": 0, "lad": 0, "abn": 1.0,
            "isotope_index": 0, "krm": krm, "particle_pairs": pairs, "spin_groups": groups}


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


def _fileNeutronMass(context) -> float:
    """The evaluation's own neutron mass (its PoPs), in amu, else kika's constant."""
    from kika._constants import NEUTRON_MASS_AMU

    from ..mf1_header import _massAmu

    pops = context.suite.PoPs
    neutron = _massAmu(pops.particles.get("n")) if pops is not None else None
    return NEUTRON_MASS_AMU if neutron is None else neutron


def _unresolvedAwri(widths, context, report, where) -> Optional[float]:
    """AWRI as FUDGE writes it: the target's mass over the file's own neutron."""
    from ..mf1_header import _massAmu
    from ..resonances import formalismAtomicWeightRatio

    if all(g.atomicWeightRatio is not None for g in widths.spinGroups):
        return None
    neutron = _fileNeutronMass(context)
    stated = formalismAtomicWeightRatio(widths, neutron)
    if stated is not None:
        return stated
    suite = context.suite
    target = _massAmu(suite.PoPs.get(suite.target)) if suite.PoPs is not None else None
    if target is not None:
        return target / neutron
    if context.awr is None:
        raise ValueError(f"{where}: no target mass to write AWRI from")
    report.approximated(f"MF2/151 LRU=2, {where}: PoPs states no target mass, so "
                        f"AWRI is MF1's AWR {float(context.awr):g}")
    return float(context.awr)


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

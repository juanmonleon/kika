"""§19 ``resonances``, model → XML. The other half of :mod:`kika.gnds.resonances`.

Separate from :mod:`kika.gnds.encode` for the reason the reader is separate: §19
is the one part of a ``reactionSuite`` that is not shaped like the rest, and
putting three formalisms' serialisation in the middle of the reaction writer
would bury both.

**The two things this has to get right, because nothing downstream would catch
them.** The R-Matrix table's ``columnHeaders`` must name the columns in the
order the widths were stored *and* each channel's ``columnIndex`` must point at
its own column — get either wrong and the file holds every number the model held,
in the wrong places, which reads back as a different evaluation. And a
``BreitWigner``'s l-blocks have to be flattened back into one table with ``L``
as a column, in the order the groups are in, so the round trip does not reorder
the resonances.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from kika.nuclear_data.model import (BreitWigner, ConversionReport, RMatrix,
                                     ScatteringRadius)

from .primitives import formatFraction

__all__ = ["writeResonances"]

#: §19.3.6's table columns, in the order FUDGE writes them and in the order the
#: two column sets in ENDF/B-VIII.1-GNDS use. ``fissionWidth`` is written only
#: when some resonance in the range has a non-zero one — 37 of the library's 386
#: ``BreitWigner`` tables carry the column and the rest do not, and adding it
#: everywhere would put a column of zeros into every non-fissile evaluation.
BREIT_WIGNER_COLUMNS = (
    ("energy", "eV", "energy"),
    ("L", "", None),
    ("J", "", "spin"),
    ("totalWidth", "eV", "totalWidth"),
    ("neutronWidth", "eV", "neutronWidth"),
    ("captureWidth", "eV", "captureWidth"),
    ("fissionWidth", "eV", "fissionWidth"),
)


def _number(value) -> str:
    from .encode import _number as formatNumber

    return formatNumber(value)


def _set(element: ET.Element, **attributes) -> ET.Element:
    for name, value in attributes.items():
        if value is not None:
            element.attrib[name] = value
    return element


def _true(value: bool) -> Optional[str]:
    return "true" if value else None


def writeResonances(root: ET.Element, resonances, report: ConversionReport,
                    domain, reactionHrefs: Optional[dict] = None) -> None:
    """``domain`` is ``(min, max)`` as strings — every ``constant1d`` in §19 has
    to declare one and the model's scalars do not carry it, so the evaluation's
    own ``projectileEnergyDomain`` is used, exactly as in the reaction writer.

    ``reactionHrefs`` maps an ENDF MT to the xPath of that reaction in the suite
    being written, for the links an ENDF-decoded region has no list of."""
    element = ET.SubElement(root, "resonances")
    _scatteringRadius(element, resonances.scatteringRadius, report,
                      "resonances", domain=domain)
    for region in resonances.resolved:
        _resolved(element, region, report, domain, resonances.scatteringRadius)
    if resonances.unresolved is not None:
        _unresolved(element, resonances.unresolved, report, domain, reactionHrefs)


def _constant(parent: ET.Element, tag: Optional[str], value: float, domain,
              unit: str = "eV", label: str = "radius") -> ET.Element:
    """One ``<x><constant1d value= domainMin= domainMax=><axes/></constant1d>``.

    ``xData_constant1d`` makes both domain attributes and the ``axes`` child
    mandatory; a scalar written without them is a file no validator accepts.
    """
    holder = parent if tag is None else ET.SubElement(parent, tag)
    node = ET.SubElement(holder, "constant1d")
    _set(node, label="eval", value=_number(value),
         domainMin=domain[0], domainMax=domain[1])
    axes = ET.SubElement(node, "axes")
    _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
    _set(ET.SubElement(axes, "axis"), index="0", label=label, unit=unit)
    return node


def _radiusUnit(unit: Optional[str], report: ConversionReport,
                where: str) -> str:
    """The unit to label a radius axis with, and the warning when there is none.

    **Every radius kika writes goes through here**, and that is still the point,
    though the danger it guards has moved. The rule used to live in
    :func:`_scatteringRadius` alone while two call sites that write a radius
    through :func:`_constant` — a ``resonanceReaction``'s and a ``channel``'s —
    labelled theirs ``fm`` outright; on an ENDF-sourced suite that was a
    factor-of-ten error written into the file as a fact, measured on the Fe-56
    micro-tape as ``0.5002`` labelled ``fm`` where FUDGE emits ``5.002 fm``.

    **Since 2026-08-20 the model is canonically fm** (``MODEL_RADIUS_UNIT``), so
    both readers set the unit and this returns ``"fm"`` for anything that came
    through one. What is left is the case that was always the honest one: a
    model built by hand, or read from a GNDS file whose axis states a unit this
    library does not convert. Neither may be labelled ``fm`` on a guess, and
    neither is.
    """
    if unit is None:
        report.warn(
            f"{where}: the scattering radius carries no unit. Both readers set "
            f"one, so this suite was built by hand — the axis is written with "
            f"an empty unit rather than labelled fm, because nothing has said "
            f"the number is in fm and the model cannot check"
        )
        return ""
    return unit


def _scatteringRadius(parent: ET.Element, radius: Optional[ScatteringRadius],
                      report: ConversionReport, where: str, domain,
                      tag: str = "scatteringRadius") -> None:
    """§19's radius. **The unit is written as the model holds it, or not at all**
    — see :func:`_radiusUnit`, which is where that rule lives."""
    if radius is None:
        return
    element = ET.SubElement(parent, tag)
    unit = _radiusUnit(radius.unit, report, where)
    if radius.isEnergyDependent:
        laws = {law for _, law in (radius.interpolation or [])}
        if len(laws) != 1 or next(iter(laws), None) not in (1,2,3,4,5):
            report.unsupportedNode(f"{where}: GNDS radius XYs1d requires one declared interpolation law")
            parent.remove(element)
            return
        node = ET.SubElement(element, "XYs1d")
        node.attrib["label"] = "eval"
        node.attrib["interpolation"] = {1:'flat',2:'lin-lin',3:'lin-log',4:'log-lin',5:'log-log'}[next(iter(laws))]
        _radiusAxes(node, unit)
        values = ET.SubElement(node, "values")
        values.text = " ".join(
            _number(v) for pair in zip(radius.energies, radius.values) for v in pair
        )
        return
    node = ET.SubElement(element, "constant1d")
    if radius.domainMin is not None and radius.domainMax is not None:
        domain = (_number(radius.domainMin), _number(radius.domainMax))
    _set(node, label="eval", value=_number(radius.constant),
         domainMin=domain[0], domainMax=domain[1])
    _radiusAxes(node, unit)


def _radiusAxes(parent: ET.Element, unit: Optional[str]) -> None:
    axes = ET.SubElement(parent, "axes")
    _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
    _set(ET.SubElement(axes, "axis"), index="0", label="radius", unit=unit or "")


def _resolved(parent: ET.Element, region, report: ConversionReport,
              domain, topRadius: Optional[ScatteringRadius] = None) -> None:
    element = ET.SubElement(parent, "resolved")
    _set(element, domainMin=_number(region.domainMin),
         domainMax=_number(region.domainMax),
         domainUnit=region.domainUnit or "eV")
    formalism = region.formalism
    if isinstance(formalism, RMatrix):
        if region.scatteringRadius is not None:
            report.unsupportedNode('region-local RMatrix radius function is not serialized by this GNDS writer')
        _rMatrix(element, formalism, report, domain, topRadius)
    elif isinstance(formalism, BreitWigner):
        _breitWigner(element, formalism, report, domain, region.scatteringRadius)
    elif formalism is not None:
        report.unsupportedNode(
            f"resolved region: kika's writer has no serialisation for a "
            f"{type(formalism).__name__} formalism; the region is written empty"
        )


def _nestedPoPs(parent, formalism, report: ConversionReport, where: str) -> None:
    """Preserve the modeled local particle database with the shared writer."""
    from .encode_pops import writePoPs
    pops = getattr(formalism, "PoPs", None)
    if pops is not None:
        writePoPs(parent, pops, report)


def _rMatrix(parent: ET.Element, formalism: RMatrix,
             report: ConversionReport, domain,
             topRadius: Optional[ScatteringRadius] = None) -> None:
    groups = _rmGroupsForGNDS(formalism, report)
    reactions, groups = _withoutEmptyFission(formalism, groups, report)
    groups = _withChannelSpins(reactions, groups)
    groups = _rmRadiusOnChannels(formalism, groups, topRadius, report)
    element = ET.SubElement(parent, "RMatrix")
    if formalism.radiusPolicy is not None and (formalism.radiusPolicy.channelMode=='constant' or formalism.radiusPolicy.phaseRadius is not None):
        report.unsupportedNode('independent RM radiusPolicy requires channel-radius normalization before GNDS serialization')
    if formalism.angularLCount not in (None, 0):
        report.lost('RMatrix angularLCount (ENDF NLSC) has no native attribute in this GNDS writer')
    _set(element, label=formalism.label or "eval",
         approximation=formalism.approximation,
         boundaryCondition=formalism.boundaryCondition,
         boundaryConditionValue=None if formalism.boundaryConditionValue is None else _number(formalism.boundaryConditionValue),
         calculateChannelRadius=_true(formalism.calculateChannelRadius if formalism.radiusPolicy is None else formalism.radiusPolicy.channelMode == "mass"),
         relativisticKinematics=_true(formalism.relativisticKinematics),
         reducedWidthAmplitudes=_true(formalism.reducedWidthAmplitudes))
    _nestedPoPs(element, formalism, report, "RMatrix")

    container = ET.SubElement(element, "resonanceReactions")
    for reaction in reactions:
        _resonanceReaction(container, reaction, report, domain)

    container = ET.SubElement(element, "spinGroups")
    for group in groups:
        _spinGroup(container, group, report, domain, formalism.reducedWidthAmplitudes)


def _rmRole(reaction) -> Optional[str]:
    """``neutron``/``capture``/``fission``, by MT first and by the LRF=3
    decoder's labels second: the rule ``prepare_reich_moore`` applies, so the
    writer and the processing code cannot disagree on what a channel is."""
    mt = getattr(reaction, "reactionMT", None)
    if mt == 2 or (mt is None and reaction.label == "elastic"):
        return "neutron"
    if mt == 102 or (mt is None and reaction.label == "capture"):
        return "capture"
    if mt in (18, 19) or (mt is None and reaction.label in ("fission", "fissionA", "fissionB")):
        return "fission"
    return None


def _withoutEmptyFission(formalism, groups, report):
    """D38: a range with no fission width writes no fission channel.

    The LRF=3 decoder gives every Reich-Moore range the four ENDF columns, so a
    non-fissile evaluation carried ``fissionA``/``fissionB`` resonance reactions
    with **no reaction to link to** (there is no MT18), and FUDGE stops on the
    first ``resonanceReaction`` without its mandatory ``link``
    (``resonances/common.py:266``). FUDGE's own ENDF→GNDS writes the fission
    channels only when some resonance has a fission width (``haveFission`` in
    ``ENDF_ITYPE_0_Misc.py``), and that is the rule here. Nothing is lost: an
    all-zero column adds nothing to any cross section, and the model keeps its
    four columns for the ENDF writer.
    """
    roles = {r.label: _rmRole(r) for r in formalism.resonanceReactions}
    empty = {label for label, role in roles.items() if role == "fission"}
    for group in groups:
        for position, channel in enumerate(group.channels):
            if channel.resonanceReaction in empty and any(
                    row[position] != 0 for row in group.widths):
                empty.discard(channel.resonanceReaction)
    if not empty:
        return list(formalism.resonanceReactions), groups
    from copy import deepcopy
    pruned = []
    for source in groups:
        group = deepcopy(source)
        keep = [i for i, channel in enumerate(group.channels)
                if channel.resonanceReaction not in empty]
        group.channels = [group.channels[i] for i in keep]
        group.widths = [[row[i] for i in keep] for row in group.widths]
        # `columnIndex` addresses the table `_rMatrixTable` lays out: dropping a
        # channel must not leave an index pointing past the last column.
        first = min((c.columnIndex for c in group.channels
                     if c.columnIndex is not None), default=1)
        for position, channel in enumerate(group.channels):
            if channel.columnIndex is not None:
                channel.columnIndex = position + first
        pruned.append(group)
    report.warn(
        f"RMatrix: {sorted(empty)} carry no non-zero width in any resonance and "
        f"are not written, as FUDGE does not write them"
    )
    return [r for r in formalism.resonanceReactions if r.label not in empty], pruned


#: The ``channelSpin`` FUDGE gives the channels whose spin no ENDF field states
#: (``ENDF_ITYPE_0_Misc.py``, LRF=3): 1 to the eliminated capture channel and 0
#: to each fission channel. FUDGE's reader requires the attribute on every
#: channel (``resolved.py:716``); the value takes no part in a Reich-Moore cross
#: section, where only the entrance channel's spin enters the statistical factor.
_UNSTATED_CHANNEL_SPIN = {"capture": 1, "fission": 0}


def _withChannelSpins(reactions, groups):
    roles = {r.label: _rmRole(r) for r in reactions}

    def unstated(channel):
        return (channel.channelSpin is None
                and roles.get(channel.resonanceReaction) in _UNSTATED_CHANNEL_SPIN)

    if not any(unstated(c) for g in groups for c in g.channels):
        return groups
    from copy import deepcopy
    groups = [deepcopy(g) for g in groups]
    for group in groups:
        for channel in group.channels:
            if unstated(channel):
                channel.channelSpin = _UNSTATED_CHANNEL_SPIN[roles[channel.resonanceReaction]]
    return groups


def _rmRadiusOnChannels(formalism, groups, topRadius, report):
    """D20b: the range's AP has no slot under ``RMatrix`` — ``RMatrixType``
    (``gnds.xsd:855-868``) admits no ``scatteringRadius`` child. GNDS states a
    radius in two other places, and the writer takes whichever loses nothing:

    * **equal to ``resonances/scatteringRadius``**, as in every single-range
      evaluation (the decoder fills the top one from the first range's AP):
      not written. Reading back, every use falls through to the top radius
      (``prepare_reich_moore.py``: channel, reaction, RMatrix, top).
    * **otherwise**, on every channel that states no radius of its own — the
      only channels the range radius ever reached. Same number, same use; what
      GNDS cannot say is that it was one range-level value, and the report says
      so.
    """
    radius = formalism.scatteringRadius
    if radius is None:
        return groups
    top = topRadius
    if (top is not None and not top.isEnergyDependent and top.constant == radius
            and (top.unit or None) == (formalism.radiusUnit or None)):
        return groups
    if all(c.scatteringRadius is not None for g in groups for c in g.channels):
        report.lost(
            f"RMatrix: range radius {radius} differs from the top-level one and "
            f"every channel states its own, so nothing uses it; GNDS has no slot "
            f"for it under RMatrix and it is not written"
        )
        return groups
    from copy import deepcopy
    groups = [deepcopy(g) for g in groups]
    for group in groups:
        for channel in group.channels:
            if channel.scatteringRadius is None:
                channel.scatteringRadius = radius
                channel.radiusUnit = formalism.radiusUnit
    report.warn(
        f"RMatrix: range radius {radius} differs from the top-level one; GNDS "
        f"has no slot for it under RMatrix, so it is written on the channels "
        f"that use it"
    )
    return groups


def _rmGroupsForGNDS(formalism, report):
    """Project per-row RM J and signed channel spin into true GNDS groups.

    No new widths or parity are inferred. AWRI and angular convergence metadata
    without a native GNDS slot remain explicit conversion losses.
    """
    from copy import deepcopy
    if formalism.approximation != 'ReichMoore':return formalism.spinGroups
    particles=[] if formalism.PoPs is None else list(formalism.PoPs.particles.values())
    target=particles[0] if len(particles)==1 else None
    result=[]
    for source in formalism.spinGroups:
        if not source.spins:
            result.append(source);continue
        if target is None or target.spin is None:
            raise ValueError('serializing legacy RM rows requires modeled target spin')
        if source.atomicWeightRatio is not None:
            report.lost(f'RM {source.label}: per-group atomicWeightRatio has no native GNDS slot; supply physical particle masses for a normalized GNDS evaluation')
        # `spinGroup/@parity` is required (gnds.xsd and FUDGE's reader alike) and
        # LRF=3 states no target parity. FUDGE's ENDF->GNDS assumes a positive
        # one (`targetParity = 1` in ENDF_ITYPE_0_Misc.py) and so does this; the
        # parity takes no part in a Reich-Moore cross section.
        parity=1 if target.parity is None else target.parity
        if target.parity is None:
            report.warn(f'RM {source.label}: ENDF states no target parity; the spin-group parity assumes a positive one, as FUDGE does')
        spin=float(target.spin.value);neutron=next((c for c in source.channels if c.resonanceReaction=='elastic'),None)
        if neutron is None:raise ValueError('legacy RM requires its declared elastic channel')
        l=neutron.L;buckets={}
        for i,aj in enumerate(source.spins):
            j=abs(aj);cs=abs(spin-.5) if aj<0 else spin+.5
            if not abs(l-cs)<=j<=l+cs and aj>=0:cs=abs(spin-.5)
            buckets.setdefault((j,cs),[]).append(i)
        for ordinal,((j,cs),indices) in enumerate(buckets.items()):
            group=deepcopy(source);group.label=f'{source.label}-J{j}-s{cs}-{ordinal}'
            group.spin=j;group.parity=parity*(-1)**l
            group.spins=[];group.energies=[source.energies[i] for i in indices]
            group.widths=[list(source.widths[i]) for i in indices];group.atomicWeightRatio=None
            for position,channel in enumerate(group.channels):
                channel.columnIndex=position+1
                if channel.resonanceReaction=='elastic':channel.channelSpin=cs
            result.append(group)
    return result


def _resonanceReaction(parent: ET.Element, reaction, report: ConversionReport,
                       domain) -> None:
    element = ET.SubElement(parent, "resonanceReaction")
    if reaction.kinematics is not None:
        report.lost(f"resonanceReaction {reaction.label!r}: channel-local particle properties and PNT/SHF require PoPs/link normalization; this writer does not serialize them")
    _set(element, label=reaction.label, ejectile=reaction.ejectile,
         eliminated=_true(reaction.eliminated))
    if reaction.href:
        _set(ET.SubElement(element, "link"), href=reaction.href)
    else:
        # §19.3.3 makes the link mandatory and it is the only formal tie between
        # a resonance channel and a reaction. An ENDF-decoded evaluation has
        # none to give — ENDF states the channels by position — so the file
        # comes out without it and this says so rather than inventing an xPath.
        report.lost(
            f"resonanceReaction {reaction.label!r} has no link to a reaction, "
            f"which §19.3.3 requires; it came from a format that identifies "
            f"resonance channels by position rather than by reference"
        )
    if reaction.Q is not None:
        _constant(element, "Q", reaction.Q, domain, unit="eV", label="Q")
    # Both radii §19.3.3 allows here, in the schema's order. The channel writer
    # below does the same pair for §19.3.4's.
    for tag, value in (("scatteringRadius", reaction.scatteringRadius),
                       ("hardSphereRadius", reaction.hardSphereRadius)):
        if value is not None:
            _constant(element, tag, value, domain,
                      unit=_radiusUnit(reaction.radiusUnit, report,
                                       f"resonanceReaction/{tag}"))


def _externalRMatrix(parent: ET.Element, external) -> None:
    """§19.3.4's ``externalRMatrix``, written back as the model holds it.

    No report and no guard: the model node validates its own ``type`` and its
    two required terms at construction, so anything that reaches here is
    writable. The terms go out **in the model's order**, which is the order the
    source stated them — ``ExternalRMatrixType`` (``gnds.xsd:942-948``) is an
    ``xs:sequence`` of ``double`` with no per-label slot, so any order validates
    and preserving the file's is free.
    """
    if external is None:
        return
    element = ET.SubElement(parent, "externalRMatrix")
    _set(element, type=external.type)
    for term in external.terms:
        _set(ET.SubElement(element, "double"),
             label=term.label, value=_number(term.value),
             unit=term.unit or None)


def _spinGroup(parent: ET.Element, group, report: ConversionReport,
               domain, amplitudes=False) -> None:
    element = ET.SubElement(parent, "spinGroup")
    _set(element, label=group.label,
         spin=None if group.spin is None else formatFraction(group.spin),
         parity=None if group.parity is None else str(group.parity))

    channels = ET.SubElement(element, "channels")
    for channel in group.channels:
        if channel.additionalPhaseShift is not None or channel.phaseShiftMode or channel.phaseAbsorptionReaction is not None:
            report.unsupportedNode("GNDS channel has no supported per-channel KPS phase representation")
        if channel.tabulatedBackground is not None:
            report.unsupportedNode(f"channel {channel.label!r}: GNDS externalRMatrix cannot serialize a tabulated complex background")
        node = ET.SubElement(channels, "channel")
        _set(node, label=channel.label,
             resonanceReaction=channel.resonanceReaction,
             L=None if channel.L is None else str(channel.L),
             channelSpin=(None if channel.channelSpin is None
                          else formatFraction(channel.channelSpin)),
             boundaryConditionValue=(None
                                     if channel.boundaryConditionValue is None
                                     else _number(channel.boundaryConditionValue)),
             columnIndex=(None if channel.columnIndex is None
                          else str(channel.columnIndex)))
        # First child of the channel, ahead of both radii: `RML_ChannelType`
        # (gnds.xsd:915-919) is an `xs:sequence`, so emitting it after them
        # would produce a file that no longer validates. Same lesson §25.3's
        # `parameterCovariances` container taught on 2026-08-19, and the same
        # reason it is written here rather than appended where it was convenient.
        _externalRMatrix(node, channel.externalRMatrix)
        for tag, value in (("scatteringRadius", channel.scatteringRadius),
                           ("hardSphereRadius", channel.hardSphereRadius)):
            if value is not None:
                _constant(node, tag, value, domain,
                          unit=_radiusUnit(channel.radiusUnit, report,
                                           f"channel[@label='{channel.label}']/{tag}"))

    _rMatrixTable(element, group, report, 'eV**0.5' if amplitudes else 'eV')
    if group.additionalPhaseShift is not None or group.phaseShiftMode:
        report.unsupportedNode(f"spinGroup {group.label!r}: additional complex phase shift requires an extension not serialized by this GNDS writer")


def _rMatrixTable(parent: ET.Element, group, report: ConversionReport, widthUnit='eV') -> None:
    """§19.3.5's ``table``, with each channel's width back in its own column.

    The table is laid out by ``columnIndex``, not by channel order: the two are
    the same in every distributed file and need not be, and a writer that
    assumed they were would silently transpose the widths of any file where they
    differ.
    """
    parameters = ET.SubElement(parent, "resonanceParameters")
    columns = 1 + len(group.channels)
    byIndex = {}
    for position, channel in enumerate(group.channels):
        if channel.columnIndex is None:
            report.lost(
                f"spinGroup {group.label!r}: channel {channel.label!r} has no "
                f"columnIndex, so its widths have no column to go in and are "
                f"not written"
            )
            continue
        columns = max(columns, channel.columnIndex + 1)
        byIndex[channel.columnIndex] = (position, channel)

    table = ET.SubElement(parameters, "table")
    _set(table, rows=str(len(group.energies)), columns=str(columns))
    headers = ET.SubElement(table, "columnHeaders")
    _set(ET.SubElement(headers, "column"), index="0", name="energy", unit="eV")
    for index in range(1, columns):
        entry = byIndex.get(index)
        _set(ET.SubElement(headers, "column"), index=str(index),
             name=f"{entry[1].resonanceReaction} width" if entry else f"column{index}",
             unit=widthUnit)

    rows = []
    for position, energy in enumerate(group.energies):
        row = [energy] + [0.0] * (columns - 1)
        for index, (channelPosition, _) in byIndex.items():
            row[index] = group.widths[position][channelPosition]
        rows.append(row)
    data = ET.SubElement(table, "data")
    data.text = " ".join(_number(v) for row in rows for v in row)


def _breitWigner(parent: ET.Element, formalism: BreitWigner,
                 report: ConversionReport, domain, regionRadius=None) -> None:
    element = ET.SubElement(parent, "BreitWigner")
    _set(element, label=formalism.label or "eval",
         approximation=getattr(formalism.approximation, "value", str(formalism.approximation)),
         calculateChannelRadius=_true(formalism.calculateChannelRadius if formalism.radiusPolicy is None
                                      else formalism.radiusPolicy.channelMode == 'mass'))
    _nestedPoPs(element, formalism, report, "BreitWigner")
    if formalism.radiusPolicy is not None and formalism.radiusPolicy.phaseRadius is not None:
        _scatteringRadius(element, formalism.radiusPolicy.phaseRadius, report, "BreitWigner", domain)
    elif regionRadius is not None:
        _scatteringRadius(element, regionRadius, report, "BreitWigner", domain)
    elif formalism.scatteringRadius is not None:
        _scatteringRadius(element, ScatteringRadius(
            constant=formalism.scatteringRadius,
            unit=formalism.radiusUnit), report, "BreitWigner", domain)

    resonances = [(group.L, resonance)
                  for group in formalism.resonanceParameters.spinGroups
                  for resonance in group.resonances]
    if any(group.competitiveChannel is not None for group in formalism.resonanceParameters.spinGroups):
        report.unsupportedNode("GNDS BreitWigner has no native competitive descriptor; exit metadata is not serialized")
    if formalism.radiusPolicy is not None and formalism.radiusPolicy.channelMode == "constant":
        report.unsupportedNode("GNDS BreitWigner cannot serialize an independent constant channel radius (NAPS2)")
    withFission = any(r.fissionWidth for _, r in resonances)
    columns = [c for c in BREIT_WIGNER_COLUMNS
               if c[0] != "fissionWidth" or withFission]

    parameters = ET.SubElement(element, "resonanceParameters")
    table = ET.SubElement(parameters, "table")
    _set(table, rows=str(len(resonances)), columns=str(len(columns)))
    headers = ET.SubElement(table, "columnHeaders")
    for index, (name, unit, _) in enumerate(columns):
        _set(ET.SubElement(headers, "column"), index=str(index), name=name,
             unit=unit)

    numbers = []
    for L, resonance in resonances:
        for name, _, field in columns:
            numbers.append(L if field is None else getattr(resonance, field))
    data = ET.SubElement(table, "data")
    data.text = " ".join(_number(v) for v in numbers)


def _unresolved(parent: ET.Element, region, report: ConversionReport,
                domain, reactionHrefs: Optional[dict] = None) -> None:
    element = ET.SubElement(parent, "unresolved")
    _set(element, domainMin=_number(region.domainMin),
         domainMax=_number(region.domainMax),
         domainUnit=region.domainUnit or "eV")
    widths = region.tabulatedWidths
    if widths is None:
        report.lost("the unresolved region has no tabulatedWidths and is empty")
        return

    node = ET.SubElement(element, "tabulatedWidths")
    policy = getattr(widths, "radiusPolicy", None)
    _set(node, label=widths.label or "eval",
         approximation="SingleLevelBreitWigner",
         calculateChannelRadius=(None if policy is None
                                 else "true" if policy.channelMode == "mass" else "false"),
         useForSelfShieldingOnly=_true(widths.selfShieldingOnly))
    if widths.potentialScatteringInterpolation is not None or widths.potentialScatteringEnergies is not None:
        report.lost('URR independent potential-scattering interpolation requires the KIKA opt-in resonance extension')
    _nestedPoPs(node, widths, report, "tabulatedWidths")
    # `resonanceReactions` before both radii: `TabulatedWidthsType`
    # (gnds.xsd:816-834) is an `xs:sequence`, and written after them it was a
    # schema error that hid the empty list behind it.
    _unresolvedReactions(node, widths, report, domain, reactionHrefs)
    table = None if policy is None else policy.phaseRadius
    # FUDGE's mapping (ENDF_ITYPE_0_Misc.readResonanceSection): AP(E) is the
    # hardSphereRadius when P/S come from the mass, and the scatteringRadius
    # when they use it too. The constant AP is kept beside a hardSphereRadius,
    # where it is unused but is what ENDF writes back.
    if policy is not None and policy.channelMode == "constant":
        _scatteringRadius(node, ScatteringRadius(constant=policy.channelRadius,unit="fm"),
                          report, "tabulatedWidths", domain)
        phase=table if table is not None else ScatteringRadius(
            constant=widths.scatteringRadius,unit=widths.radiusUnit)
        _scatteringRadius(node, phase, report, "tabulatedWidths", domain,tag="hardSphereRadius")
    elif table is not None and table.isEnergyDependent and policy.channelMode == "phase":
        _scatteringRadius(node, table, report, "tabulatedWidths", domain)
    elif widths.scatteringRadius is not None:
        _scatteringRadius(node, ScatteringRadius(
            constant=widths.scatteringRadius,
            unit=widths.radiusUnit), report, "tabulatedWidths", domain)
    if table is not None and table.isEnergyDependent and policy.channelMode == "mass":
        _scatteringRadius(node, table, report, "tabulatedWidths", domain,
                          tag="hardSphereRadius")

    Ls = ET.SubElement(node, "Ls")
    byL = {}
    for group in widths.spinGroups:
        byL.setdefault(group.L, []).append(group)
    for position, (L, groups) in enumerate(byL.items()):
        lNode = ET.SubElement(Ls, "L")
        _set(lNode, label=str(position), value=str(L))
        Js = ET.SubElement(lNode, "Js")
        for index, group in enumerate(groups):
            _unresolvedSpinGroup(Js, group, index, widths.energyGrid, report,
                                 domain)


#: The URR width labels the ENDF decoder writes, and the MT each one is. The
#: competitive width has no single reaction (it is MT4 less the elastic, or
#: MT51 and up), and FUDGE leaves it out of the list too: its ENDF->GNDS writes
#: the competitive width with a label no resonanceReaction carries.
_URR_LABEL_MT = {"neutron": 2, "elastic": 2, "capture": 102, "fission": 18}


def _unresolvedReactions(node: ET.Element, widths, report: ConversionReport,
                         domain, reactionHrefs) -> None:
    """§19.4.1's ``resonanceReactions``, which the schema requires non-empty.

    A GNDS-read region carries its own list and it is written as read. An
    ENDF-decoded one carries none, since ENDF states URR channels by position;
    the list is then built the way FUDGE builds it, one linked entry per width
    whose MT the suite has a reaction for (D38: written empty, the file did not
    validate and FUDGE could not resolve the widths' reactions).
    """
    reactions = ET.SubElement(node, "resonanceReactions")
    if widths.resonanceReactions:
        for reaction in widths.resonanceReactions:
            _resonanceReaction(reactions, reaction, report, domain)
        return
    for label in _channelLabels(widths):
        href = (reactionHrefs or {}).get(_URR_LABEL_MT.get(label))
        if href is not None:
            entry = _set(ET.SubElement(reactions, "resonanceReaction"), label=label)
            _set(ET.SubElement(entry, "link"), href=href)
    if len(reactions) == 0:
        report.lost(
            "the unresolved region names no resonanceReactions, which §19.4.1 "
            "requires, and no width label maps to a reaction of this suite; the "
            "channels are identified only by the labels on their own widths"
        )


def _channelLabels(widths) -> list:
    seen = []
    for group in widths.spinGroups:
        for channel in group.channels:
            if channel.label not in seen:
                seen.append(channel.label)
    return seen


def _unresolvedSpinGroup(parent: ET.Element, group, index: int, blockGrid,
                         report: ConversionReport, domain) -> None:
    element = ET.SubElement(parent, "J")
    _set(element, label=str(index), value=formatFraction(group.J))

    if group.crossSectionInterpolation is not None or group.crossSectionEnergies is not None:
        report.lost(f"URR L={group.L} J={group.J}: cross-section interpolation is distinct from parameter-function interpolation and is not serialized by this GNDS writer")
    grid = group.levelSpacingEnergies if group.levelSpacingEnergies is not None \
        else blockGrid
    spacing, spacingConstant = group.levelSpacing, None
    if (group.levelSpacingFunction is None and spacing is not None
            and np.asarray(spacing).size == 1
            and (grid is None or np.asarray(grid).size != 1)):
        # ENDF cases A and B give one D per J: the model keeps it as a
        # one-element array, which the arrays-against-a-grid path below could
        # only drop. It is a constant, and §19.4.1 writes it as constant1d.
        spacing, spacingConstant = None, float(np.asarray(spacing)[0])
    _average(element, "levelSpacing", grid, spacing, spacingConstant,
             f"spinGroup L={group.L} J={group.J} levelSpacing", report, domain,
             unit="eV", label="levelSpacing", form=group.levelSpacingFunction)

    widths = ET.SubElement(element, "widths")
    for position, channel in enumerate(group.channels):
        if channel.neutronWidthConvention!='reduced':
            report.lost('URR physical neutron widths require the KIKA opt-in resonance extension')
        node = ET.SubElement(widths, "width")
        # An integral count goes out as an integer: FUDGE's `floatOrint`
        # (resonances/common.py:487) does `int(value)` on any integral float,
        # and `int("1.0")` raises, so `1.0` was a file FUDGE could not open.
        dof = channel.degreesOfFreedom
        _set(node, label=str(position), resonanceReaction=channel.label,
             degreesOfFreedom=(None if dof is None else
                               str(int(dof)) if float(dof).is_integer()
                               else _number(dof)))
        grid = channel.energies if channel.energies is not None else blockGrid
        _average(node, None, grid, channel.widths, channel.constantWidth,
                 f"spinGroup L={group.L} J={group.J} width {channel.label!r}",
                 report, domain, unit="eV", label="width", form=channel.averageFunction)


def _average(parent: ET.Element, tag: Optional[str], grid, values,
             constant: Optional[float], where: str,
             report: ConversionReport, domain, unit: str = "eV",
             label: str = "width", form=None) -> None:
    """One §19.4.1 average: an ``XYs1d`` over its grid, or a ``constant1d``."""
    if form is not None:
        from .encode import _function
        holder = parent if tag is None else ET.SubElement(parent, tag)
        if hasattr(form,'constant'):
            if (constant is not None and constant != form.constant) or (
                    values is not None and not np.array_equal(values,[form.constant])):
                report.lost(f"{where}: compatibility scalar disagrees with canonical constant function")
                return
            _function(holder,form,report,where)
            return
        if hasattr(form, 'function1ds'):
            canonical_x, canonical_y, _ = form.toEndfRegions()
        else:
            canonical_x, canonical_y = form.xs, form.ys
        if (values is not None and not np.array_equal(values, canonical_y)) or (
                grid is not None and not np.array_equal(grid, canonical_x)):
            report.lost(f"{where}: compatibility arrays disagree with the canonical function; edit the function and synchronize the arrays")
            return
        _function(holder, form, report, where)
        return
    if constant is not None:
        _constant(parent, tag, constant, domain, unit=unit, label=label)
        return
    holder = parent if tag is None else ET.SubElement(parent, tag)
    if values is None:
        report.lost(f"{where}: no values, so the node is written empty")
        return
    if grid is None or len(grid) != len(values):
        report.lost(
            f"{where}: {len(values)} values and "
            f"{'no' if grid is None else len(grid)} energies to put them on, so "
            f"the average is not written rather than written against a grid it "
            f"does not belong to"
        )
        return
    node = ET.SubElement(holder, "XYs1d")
    node.attrib["label"] = "eval"
    _radiusAxes(node, unit)
    text = ET.SubElement(node, "values")
    text.text = " ".join(_number(v)
                         for pair in zip(np.asarray(grid), np.asarray(values))
                         for v in pair)

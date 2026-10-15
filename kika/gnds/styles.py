"""§9 ``styles``, read and written — shared by both roots.

A ``reactionSuite`` has one and a ``covarianceSuite`` has one, and the schema
makes it **mandatory** on each. They are the same node, so the code for it lives
here rather than in either reader: the first version put it on the reactionSuite
reader alone, and the covariance file kika wrote came out without a ``styles``
and failed validation on the first line after ``externalFiles``.

Evaluated and reconstructed styles describe the distributed evaluations;
``heated`` additionally preserves processed cross sections and their temperature.
Other styles are reported when their reader is not implemented.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Optional

from kika.gnds.nodes import reads, writes
from kika.nuclear_data.model import (ConversionReport, CrossSectionReconstructed,
                                     Evaluated, Heated, PhysicalQuantity, RangeQuantity,
                                     Style, Styles, TargetInfo, TargetInfoElement,
                                     TargetInfoNuclide, URR_probabilityTables)
from kika.nuclear_data.model.styles import MonteCarlo_cdf

__all__ = ["STYLES", "WRITABLE_STYLES", "readStyles", "writeStyles",
           "readPhysicalQuantity", "readRange"]

#: §9 node name → supported reader class.
STYLES = {
    "evaluated": Evaluated,
    "crossSectionReconstructed": CrossSectionReconstructed,
    "heated": Heated,
    # In RS_StylesType of FUDGE's current gnds.xsd (label, date, derivedFrom).
    "MonteCarlo_cdf": MonteCarlo_cdf,
    "URR_probabilityTables": URR_probabilityTables,
}

#: Writer vocabulary derived from the model. Some other styles still have no
#: reader; Heated has an explicit, validated temperature codec in both directions.
WRITABLE_STYLES = tuple(sorted(cls.gndsNodeName for cls in Style.__subclasses__()))


def readPhysicalQuantity(element: Optional[ET.Element]) -> Optional[PhysicalQuantity]:
    if element is None or "value" not in element.attrib:
        return None
    return PhysicalQuantity(value=float(element.attrib["value"]),
                            unit=element.attrib.get("unit", ""))


#: kT units a processed suite states its temperature in, as eV per unit (the
#: ACE, HDF5 and processSuite decoders write kT in eV/k): written back in K.
_KT_UNITS = {"eV/k": 1.0, "MeV/k": 1.0e6}


def _heatedTemperature(style: Heated) -> PhysicalQuantity:
    """Reject incomplete processed metadata instead of losing its meaning.

    A kT (``eV/k``, ``MeV/k``) is converted to K: a suite read from ACE or HDF5,
    or processed by kika, states its temperature so, and was refused here.
    """
    from kika.hdf5.names import K_BOLTZMANN

    temperature = style.temperature
    if (isinstance(temperature, PhysicalQuantity) and temperature.unit in _KT_UNITS
            and isinstance(temperature.value, (int, float))
            and not isinstance(temperature.value, bool)):
        temperature = PhysicalQuantity(
            value=float(temperature.value) * _KT_UNITS[temperature.unit] / K_BOLTZMANN, unit="K")
    if (not isinstance(temperature, PhysicalQuantity)
            or temperature.unit != "K"
            or isinstance(temperature.value, bool)
            or not isinstance(temperature.value, (int, float))
            or not math.isfinite(temperature.value)
            or temperature.value < 0):
        raise ValueError(
            f"heated style {style.label!r} requires a finite non-negative temperature in K"
        )
    if not isinstance(style.derivedFrom, str) or not style.derivedFrom.strip():
        raise ValueError(f"heated style {style.label!r} requires derivedFrom")
    return temperature


def readRange(element: Optional[ET.Element]) -> Optional[RangeQuantity]:
    if element is None or "min" not in element.attrib:
        return None
    return RangeQuantity(min=float(element.attrib["min"]),
                         max=float(element.attrib["max"]),
                         unit=element.attrib.get("unit", ""))


def readTargetInfo(element: ET.Element) -> TargetInfo:
    """``evaluated/targetInfo`` (gnds.xsd TargetInfoType): the target's isotopes."""
    elements = []
    for chemical in element.findall("isotopicAbundances/chemicalElements/chemicalElement"):
        elements.append(TargetInfoElement(
            symbol=chemical.attrib.get("symbol", ""),
            nuclides=[TargetInfoNuclide(pid=n.attrib.get("pid", ""),
                                        atomFraction=float(n.attrib.get("atomFraction", "nan")))
                      for n in chemical.findall("nuclides/nuclide")]))
    return TargetInfo(chemicalElements=elements)


def writeTargetInfo(parent: ET.Element, info: TargetInfo, number) -> ET.Element:
    """The other way. Only a reactionSuite's ``evaluated`` admits it."""
    element = ET.SubElement(parent, "targetInfo")
    container = ET.SubElement(ET.SubElement(element, "isotopicAbundances"),
                              "chemicalElements")
    for chemical in info.chemicalElements:
        node = ET.SubElement(container, "chemicalElement")
        node.attrib["symbol"] = chemical.symbol
        nuclides = ET.SubElement(node, "nuclides")
        for nuclide in chemical.nuclides:
            child = ET.SubElement(nuclides, "nuclide")
            child.attrib["pid"] = nuclide.pid
            child.attrib["atomFraction"] = number(nuclide.atomFraction)
    return element


@reads("style", *STYLES)
def readStyles(element: ET.Element, path: str, report: ConversionReport,
               tally=None) -> Styles:
    """§9.1 ``styles`` → the model.

    ``tally`` is the caller's counter for the losses that would otherwise be
    reported once per style — ``documentation``, which every ``evaluated``
    carries and no model node holds.
    """
    styles = Styles()
    for child in element:
        cls = STYLES.get(child.tag)
        if cls is None:
            report.unsupportedNode(
                f"{path}/{child.tag}: kika models this style but no distributed "
                f"neutron evaluation carries one, so reading it would be untested"
            )
            continue
        style: Style = cls(
            label=child.attrib.get("label", ""),
            derivedFrom=child.attrib.get("derivedFrom"),
            date=child.attrib.get("date"),
        )
        if isinstance(style, Evaluated):
            style.library = child.attrib.get("library")
            style.version = child.attrib.get("version")
            style.temperature = readPhysicalQuantity(child.find("temperature"))
            style.projectileEnergyDomain = readRange(
                child.find("projectileEnergyDomain")
            )
            info = child.find("targetInfo")
            if info is not None:
                style.targetInfo = readTargetInfo(info)
        if isinstance(style, Heated):
            style.temperature = readPhysicalQuantity(child.find("temperature"))
            _heatedTemperature(style)
        document = child.find("documentation")
        if document is not None and (len(document) or document.attrib or (document.text or "").strip()):
            from .documentation import readDocumentation
            style.documentation = readDocumentation(document, report)
        styles.add(style)
    return styles


@writes("style", *WRITABLE_STYLES)
def writeStyles(root: ET.Element, styles: Styles, number,
                documentation: bool = True, report=None) -> ET.Element:
    """The model → ``<styles>``. ``number`` formats a float, from the encoder.

    **The two schemas disagree about ``documentation``, and that is what the
    flag is for.** ``gnds.xsd``'s ``RS_EvaluatedType`` makes the element
    *mandatory* on a reactionSuite's ``evaluated``; ``covariances.xsd``'s own
    ``evaluated`` does not admit it at all. Writing the same style node into
    both roots therefore cannot be right, and the first version that tried
    produced a covariance file rejected on its fifth line.

    What is written is what was read (``documentation.py``: texts, authors,
    dates); a style read with none gets none, and kika invents nothing.
    """
    # Validate mutable Heated metadata before changing the output tree.
    for style in styles:
        if isinstance(style, Heated):
            _heatedTemperature(style)
    container = ET.SubElement(root, "styles")
    for style in styles:
        element = ET.SubElement(container, style.gndsNodeName)
        for name, value in (("label", style.label),
                            ("derivedFrom", style.derivedFrom),
                            ("date", style.date)):
            if value is not None:
                element.attrib[name] = value
        if style.documentation is not None:
            if not documentation:
                if report is None:
                    raise ValueError("this root does not admit style documentation")
                report.lost("covariance style documentation cannot be represented in this schema")
            else:
                from .documentation import writeDocumentation
                writeDocumentation(element, style.documentation)
        if isinstance(style, Heated):
            temperature = _heatedTemperature(style)
            node = ET.SubElement(element, "temperature")
            node.attrib["value"] = number(temperature.value)
            node.attrib["unit"] = temperature.unit
        if not isinstance(style, Evaluated):
            continue
        for name, value in (("library", style.library),
                            ("version", style.version)):
            if value is not None:
                element.attrib[name] = value
        if style.temperature is not None:
            node = ET.SubElement(element, "temperature")
            node.attrib["value"] = number(style.temperature.value)
            node.attrib["unit"] = style.temperature.unit
        if style.projectileEnergyDomain is not None:
            domain = style.projectileEnergyDomain
            node = ET.SubElement(element, "projectileEnergyDomain")
            node.attrib["min"] = number(domain.min)
            node.attrib["max"] = number(domain.max)
            node.attrib["unit"] = domain.unit
        if documentation and style.targetInfo is not None:
            writeTargetInfo(element, style.targetInfo, number)
        if documentation and style.documentation is None:
            ET.SubElement(element, "documentation")
    return container

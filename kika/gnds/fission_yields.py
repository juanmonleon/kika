"""§18.4 ``productYields`` (fission product yields) ↔ the model, roadmap E7c.

Read and written wherever ``fissionFragmentData`` appears: under a fission
reaction's output channel, under a PoPs ``nuclide`` (FUDGE's spontaneous-fission
yields, and kika's neutron-induced ones), and as the root of a file (FUDGE's
neutron-induced yields, ``ENDF_ITYPE_1``).

The ``yields/uncertainty`` FUDGE writes is a ``covariance`` whose ``array`` is
``compression="diagonal"`` -- the variances ENDF's DY squared. That is the form
read; any other uncertainty (a full matrix) is counted and not read.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from kika.nuclear_data.model import (ConversionReport, ElapsedTime, IncidentEnergy,
                                     PhysicalQuantity, ProductYield, Yields)


def _number(value) -> str:
    from .encode import _number as number
    return number(value)


def _floats(text: Optional[str]):
    return [float(v) for v in (text or "").split()]


def _nuclideList(element: Optional[ET.Element]):
    if element is None or "href" in element.attrib:
        return None
    return (element.text or "").split()


def _readYields(element: ET.Element, tally) -> Yields:
    yields = Yields(nuclides=_nuclideList(element.find("nuclides")),
                    values=_floats(element.findtext("values")))
    uncertainty = element.find("uncertainty")
    if uncertainty is not None:
        array = uncertainty.find("covariance/array")
        if array is not None and array.attrib.get("compression") == "diagonal":
            yields.uncertainty = _floats(array.findtext("values"))
        else:
            tally("fissionFragmentData <yields><uncertainty>: only a diagonal covariance is read")
    return yields


def _readTime(element: Optional[ET.Element]):
    if element is None:
        return None
    text = element.find("string")
    if text is not None:
        return text.attrib.get("value")
    double = element.find("double")
    if double is None:
        return None
    return PhysicalQuantity(value=float(double.attrib["value"]), unit=double.attrib.get("unit", "s"),
                            label=double.attrib.get("label"))


def readProductYields(element: Optional[ET.Element], tally):
    """``productYields`` → a list of :class:`ProductYield`."""
    out = []
    if element is None:
        return out
    for node in element.findall("productYield"):
        productYield = ProductYield(label=node.attrib.get("label", "eval"),
                                    nuclides=_nuclideList(node.find("nuclides")))
        for timeNode in node.findall("elapsedTimes/elapsedTime"):
            elapsed = ElapsedTime(label=timeNode.attrib.get("label", ""),
                                  time=_readTime(timeNode.find("time")))
            yieldsNode = timeNode.find("yields")
            if yieldsNode is not None:
                elapsed.yields = _readYields(yieldsNode, tally)
            for energyNode in timeNode.findall("incidentEnergies/incidentEnergy"):
                double = energyNode.find("energy/double")
                elapsed.incidentEnergies.append(IncidentEnergy(
                    label=energyNode.attrib.get("label", ""),
                    energy=PhysicalQuantity(value=float(double.attrib["value"]),
                                            unit=double.attrib.get("unit", "eV"),
                                            label=double.attrib.get("label")),
                    yields=_readYields(energyNode.find("yields"), tally)))
            productYield.elapsedTimes.append(elapsed)
        out.append(productYield)
    return out


def _writeYields(parent: ET.Element, yields: Yields, depth: int) -> None:
    node = ET.SubElement(parent, "yields")
    nuclides = ET.SubElement(node, "nuclides")
    if yields.nuclides is None:
        # The productYield's own list, as FUDGE links it.
        nuclides.set("href", "/".join([".."] * depth) + "/nuclides")
    else:
        nuclides.text = " ".join(yields.nuclides)
    ET.SubElement(node, "values").text = " ".join(_number(v) for v in yields.values)
    if yields.uncertainty is not None:
        covariance = ET.SubElement(ET.SubElement(node, "uncertainty"), "covariance")
        n = len(yields.uncertainty)
        array = ET.SubElement(covariance, "array", shape=f"{n},{n}", compression="diagonal")
        ET.SubElement(array, "values").text = " ".join(_number(v) for v in yields.uncertainty)


def _writeTime(parent: ET.Element, elapsed: ElapsedTime) -> None:
    node = ET.SubElement(parent, "time")
    if isinstance(elapsed.time, str):
        ET.SubElement(node, "string", label=elapsed.label, value=elapsed.time, unit="s")
    elif elapsed.time is not None:
        ET.SubElement(node, "double", label=elapsed.time.label or elapsed.label,
                      value=_number(elapsed.time.value), unit=elapsed.time.unit or "s")


def writeProductYields(parent: ET.Element, productYields,
                       report: Optional[ConversionReport] = None) -> None:
    """The model → ``productYields`` (only when there is one)."""
    if not productYields:
        return
    holder = ET.SubElement(parent, "productYields")
    for productYield in productYields:
        node = ET.SubElement(holder, "productYield", label=productYield.label)
        if productYield.nuclides is not None:
            ET.SubElement(node, "nuclides").text = " ".join(productYield.nuclides)
        times = ET.SubElement(node, "elapsedTimes")
        for elapsed in productYield.elapsedTimes:
            timeNode = ET.SubElement(times, "elapsedTime", label=elapsed.label)
            _writeTime(timeNode, elapsed)
            if elapsed.yields is not None:
                # yields > elapsedTime > elapsedTimes > productYield
                _writeYields(timeNode, elapsed.yields, 3)
            elif elapsed.incidentEnergies:
                energies = ET.SubElement(timeNode, "incidentEnergies")
                for incident in elapsed.incidentEnergies:
                    energyNode = ET.SubElement(energies, "incidentEnergy", label=incident.label)
                    ET.SubElement(ET.SubElement(energyNode, "energy"), "double",
                                  label=incident.energy.label or incident.label,
                                  value=_number(incident.energy.value),
                                  unit=incident.energy.unit or "eV")
                    # yields > incidentEnergy > incidentEnergies > elapsedTime > elapsedTimes > productYield
                    _writeYields(energyNode, incident.yields, 5)


def writeFissionFragmentDataDocument(data, report: Optional[ConversionReport] = None):
    """The model → a GNDS file whose root is ``fissionFragmentData``.

    FUDGE's form for neutron-induced yields (``ENDF_ITYPE_1``), and the one
    FUDGE 6.14 reads them from: its PoPs reader refuses ``incidentEnergies``
    under a nuclide (``PoPs/fissionFragmentData/elapsedTime.py``), though
    ``gnds.xsd`` admits them. The root names no target and has no styles, so
    only the yields go in it.
    """
    report = report if report is not None else ConversionReport()
    root = ET.Element("fissionFragmentData")
    if len(getattr(data, "delayedNeutrons", []) or []) or getattr(data, "fissionEnergyReleases", None):
        report.lost("fissionFragmentData file: delayed neutrons and the energy release belong "
                    "to a reaction and are not written here")
    writeProductYields(root, data.productYields, report)
    return ET.ElementTree(root), report


def readFissionFragmentDataDocument(document):
    """A GNDS file whose root is ``fissionFragmentData`` (FUDGE's NFY) → the model."""
    from kika.nuclear_data.model import FissionFragmentData

    from .decode import _SuiteReader

    report = ConversionReport()
    reader = _SuiteReader(document, {}, report)
    root = document.root
    data = reader.readFissionFragmentData(root, "")
    reader.flushTallies()
    report.warn("a fissionFragmentData file names no target and carries no styles or "
                "documentation (FUDGE's neutron-induced yields); kika writes yields in a "
                "PoPs whose target nuclide carries them")
    data.report = report
    return data, report

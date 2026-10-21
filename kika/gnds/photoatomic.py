"""GNDS photo-atomic forms and PoPs atomic data ↔ the model.

**§14 double-differential forms.** ``coherentPhotonScattering`` (``formFactor``,
optional ``realAnomalousFactor`` and ``imaginaryAnomalousFactor``) and
``incoherentPhotonScattering`` (optional ``scatteringFactor``), each an
``XYs1d`` or a ``regions1d`` (``gnds.xsd`` DDCS_CoherentPhotonScatteringType,
DDCS_IncoherentPhotonScatteringType), into
:class:`~kika.nuclear_data.model.photoatomic.CoherentPhotonScattering` and
:class:`~kika.nuclear_data.model.photoatomic.IncoherentPhotonScattering`. The
dispatch lives in :mod:`kika.gnds.thermal_scattering`, which owns the
``doubleDifferentialCrossSection`` container.

**§12 atomic data.** ``chemicalElement/atomic/configurations/configuration``
(``subshell``, ``electronNumber``, ``bindingEnergy``, ``decayData``; gnds.xsd
PoPs_AtomicType, PoPs_AtomicConfigurationType) into
:class:`~kika.nuclear_data.model.photoatomic.ChemicalElement`. A transition is
a ``decayMode`` of mode ``atomicRelaxation``, read and written by
:mod:`kika.gnds.decay` like any other.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from kika._constants import ATOMIC_NUMBER_TO_NAME
from kika.nuclear_data.model import (Atomic, ChemicalElement, CoherentPhotonScattering, Configuration,
                                     IncoherentPhotonScattering)

__all__ = ["PHOTON_FORMS", "readPhotonForm", "writePhotonForm", "readChemicalElement", "writeAtomic"]

#: The model classes the photon branch of the DDCS writer handles.
PHOTON_FORMS = (CoherentPhotonScattering, IncoherentPhotonScattering)

_COHERENT_PARTS = ("formFactor", "realAnomalousFactor", "imaginaryAnomalousFactor")


def readPhotonForm(element: ET.Element, resolve, label: str):
    """A ``coherentPhotonScattering`` or ``incoherentPhotonScattering`` node → its model form."""
    from .thermal_scattering import _readWrappedXYs1d

    common = dict(label=label, pid=element.attrib.get("pid", "photon"),
                  productFrame=element.attrib.get("productFrame", "lab"))
    if element.tag == "coherentPhotonScattering":
        parts = {name: _readWrappedXYs1d(element.find(name), resolve) for name in _COHERENT_PARTS}
        if parts["formFactor"] is None:
            raise ValueError("coherentPhotonScattering holds no formFactor (gnds.xsd requires one)")
        return CoherentPhotonScattering(**parts, **common)
    return IncoherentPhotonScattering(
        scatteringFactor=_readWrappedXYs1d(element.find("scatteringFactor"), resolve), **common)


def writePhotonForm(container: ET.Element, label: str, form, report, where: str) -> ET.Element:
    """One photon-scattering form under ``doubleDifferentialCrossSection``, children in the schema's order."""
    from .encode import _set
    from .thermal_scattering import _wrappedXYs1d

    element = ET.SubElement(container, form.gndsNodeName)
    _set(element, label=label or form.label, pid=form.pid, productFrame=str(form.productFrame))
    if isinstance(form, CoherentPhotonScattering):
        for name in _COHERENT_PARTS:
            _wrappedXYs1d(element, name, getattr(form, name), report, where)
    else:
        _wrappedXYs1d(element, "scatteringFactor", form.scatteringFactor, report, where)
    return element


def readChemicalElement(element: ET.Element, readDecayData) -> ChemicalElement:
    """A ``chemicalElement`` → the model, with its ``atomic`` data when it has some.

    ``readDecayData`` is the reader's own (``decayData`` node → model), so the
    transitions go through the one §12 decayData reader kika has.
    """
    from .decay import readQuantity

    atomic = None
    node = element.find("atomic")
    if node is not None:
        atomic = Atomic()
        for configuration in node.findall("configurations/configuration"):
            binding = readQuantity(configuration.find("bindingEnergy/double"), "eV")
            if binding is None:
                raise ValueError(f"configuration {configuration.attrib.get('subshell')!r} has no "
                                 f"bindingEnergy (gnds.xsd requires one)")
            atomic.configurations.append(Configuration(
                subshell=configuration.attrib["subshell"],
                electronNumber=float(configuration.attrib["electronNumber"]),
                bindingEnergy=binding,
                decayData=readDecayData(configuration.find("decayData"))))
    Z = int(element.attrib["Z"])
    return ChemicalElement(symbol=element.attrib["symbol"], Z=Z,
                           name=element.attrib.get("name") or ATOMIC_NUMBER_TO_NAME.get(Z), atomic=atomic)


def writeAtomic(chemical: ET.Element, element: ChemicalElement, report) -> None:
    """The element's ``atomic`` node, if it carries atomic data (before ``isotopes``, as the schema orders)."""
    from .decay import writeDecayData, writeQuantity
    from .encode import _number

    if element.atomic is None:
        return
    configurations = ET.SubElement(ET.SubElement(chemical, "atomic"), "configurations")
    for c in element.atomic.configurations:
        node = ET.SubElement(configurations, "configuration")
        node.set("subshell", c.subshell)
        node.set("electronNumber", _number(c.electronNumber))
        writeQuantity(node, "bindingEnergy", c.bindingEnergy)
        writeDecayData(node, c.decayData, report)

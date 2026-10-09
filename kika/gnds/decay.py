"""§12 ``decayData`` in full, PoPs ``aliases``, and a PoPs that stands alone.

Roadmap E5c read and wrote the electromagnetic part of ``decayData`` (a level's
gamma cascade); E7b adds what a radioactive decay sublibrary states and FUDGE
writes (``PoPs/decays``): a mode's ``Q`` and the uncertainty of its
probability, its ``spectra`` -- ``discrete`` lines and ``continuum`` tables --
and the decay's ``averageEnergies``. A decay evaluation is a whole GNDS file
whose root is ``PoPs`` (with its own ``styles``), not a ``reactionSuite``;
:func:`readPoPsDocument` and :func:`writePoPsDocument` are that file's door.

Quantities follow ``PoPs_quantity``/``PQU_double``: ``<double label value
unit>`` with an optional ``<uncertainty><standard><double value/>``. The
discrete line's ``intensity``/``energy`` are the ``Simple`` variant, with the
value on the element itself (``gnds.xsd`` ``PoPs_DiscreteSpectrumType``).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Tuple

from kika.nuclear_data.model import (ConversionReport, Continuum, Decay, DecayData,
                                     DecayMode, DecayModes, DecayPath, Discrete,
                                     MetaStable, PhotonEmissionProbabilities,
                                     PhysicalQuantity, PoPs, Product, Shell, Spectrum)
from kika.nuclear_data.model.quantities import Uncertainty


def _number(value) -> str:
    from .encode import _number as number
    return number(value)


def _set(element, **attributes):
    for name, value in attributes.items():
        if value is not None:
            element.set(name, value)
    return element


# ---------------------------------------------------------------------------
# Quantities with a standard uncertainty
# ---------------------------------------------------------------------------

def _standard(element: Optional[ET.Element]) -> Optional[float]:
    if element is None:
        return None
    node = element.find("uncertainty/standard/double")
    return float(node.attrib["value"]) if node is not None else None


def readQuantity(element: Optional[ET.Element], defaultUnit: str = "") -> Optional[PhysicalQuantity]:
    """A ``<double>`` (or a ``Simple`` quantity) with its standard uncertainty."""
    if element is None or "value" not in element.attrib:
        return None
    sigma = _standard(element)
    return PhysicalQuantity(value=float(element.attrib["value"]),
                            unit=element.attrib.get("unit", defaultUnit),
                            label=element.attrib.get("label"),
                            uncertainty=Uncertainty(sigma) if sigma is not None else None)


def _writeUncertainty(element: ET.Element, sigma: Optional[float]) -> None:
    if sigma is None:
        return
    standard = ET.SubElement(ET.SubElement(element, "uncertainty"), "standard")
    _set(ET.SubElement(standard, "double"), value=_number(sigma))


def _sigma(quantity: Optional[PhysicalQuantity]) -> Optional[float]:
    uncertainty = getattr(quantity, "uncertainty", None)
    return None if uncertainty is None else float(uncertainty.value)


def writeQuantity(parent: ET.Element, tag: str, quantity: PhysicalQuantity,
                  label: Optional[str] = "eval", simple: bool = False) -> ET.Element:
    """``<tag><double label value unit/></tag>``, or ``<tag value unit/>`` when simple."""
    if simple:
        node = _set(ET.SubElement(parent, tag), value=_number(quantity.value),
                    unit=quantity.unit or None)
    else:
        node = _set(ET.SubElement(ET.SubElement(parent, tag), "double"),
                    label=quantity.label or label, value=_number(quantity.value),
                    unit=quantity.unit or None)
    _writeUncertainty(node, _sigma(quantity))
    return node


def _shells(element: Optional[ET.Element]):
    if element is None:
        return []
    return [Shell(label=shell.attrib["label"], value=float(shell.attrib["value"]),
                  uncertainty=_standard(shell)) for shell in element.findall("shell")]


def _writeShells(parent: ET.Element, tag: str, shells) -> None:
    node = ET.SubElement(parent, tag)
    for shell in shells:
        child = _set(ET.SubElement(node, "shell"), label=shell.label,
                     value=_number(shell.value))
        _writeUncertainty(child, getattr(shell, "uncertainty", None))


# ---------------------------------------------------------------------------
# decayData
# ---------------------------------------------------------------------------

def readDecayData(element: Optional[ET.Element], tally) -> Optional[DecayData]:
    """§12 ``decayData`` → the model: modes, their spectra, the average energies."""
    if element is None:
        return None
    modes = DecayModes()
    for mode in element.findall("decayModes/decayMode"):
        if mode.find("internalConversionCoefficients") is not None:
            tally("PoPs <decayMode><internalConversionCoefficients>: a mode-level "
                  "conversion coefficient (no evaluation read so far states one)")
        quantity = mode.find("probability/double")
        if quantity is None:
            tally("PoPs <decayMode><probability>: only a <double> is read")
        emission = mode.find("photonEmissionProbabilities")
        modes.decayModes.append(DecayMode(
            label=mode.attrib["label"], mode=mode.attrib["mode"],
            probability=(float(quantity.attrib.get("value", "nan"))
                         if quantity is not None else float("nan")),
            probabilityUncertainty=_standard(quantity),
            probabilityLabel=(quantity.attrib.get("label") or "eval") if quantity is not None else "eval",
            photonEmissionProbabilities=(PhotonEmissionProbabilities(shells=_shells(emission))
                                         if emission is not None else None),
            Q=readQuantity(mode.find("Q/double"), "eV"),
            decayPath=DecayPath(decays=[_readDecay(decay) for decay in mode.findall("decayPath/decay")]),
            spectra=[_readSpectrum(spectrum, tally) for spectrum in mode.findall("spectra/spectrum")]))
    averages = [readQuantity(node, "eV") for node in element.findall("averageEnergies/averageEnergy")]
    return DecayData(decayModes=modes, averageEnergies=[a for a in averages if a is not None])


def _readDecay(decay: ET.Element) -> Decay:
    complete = decay.attrib.get("complete")
    return Decay(index=int(decay.attrib["index"]), mode=decay.attrib.get("mode"),
                 complete=None if complete is None else complete == "true",
                 products=[Product(pid=product.attrib["pid"], label=product.attrib.get("label"))
                           for product in decay.findall("products/product")])


def _readSpectrum(element: ET.Element, tally) -> Spectrum:
    from .thermal_scattering import _readWrappedXYs1d

    spectrum = Spectrum(label=element.attrib["label"], pid=element.attrib["pid"])
    for child in element:
        if child.tag == "discrete":
            emission = child.find("photonEmissionProbabilities")
            spectrum.emissions.append(Discrete(
                intensity=readQuantity(child.find("intensity")),
                energy=readQuantity(child.find("energy"), "eV"),
                type=child.attrib.get("type"),
                internalConversionCoefficients=_shells(child.find("internalConversionCoefficients")),
                photonEmissionProbabilities=(PhotonEmissionProbabilities(shells=_shells(emission))
                                             if emission is not None else None),
                positronEmissionIntensity=readQuantity(child.find("positronEmissionIntensity")),
                internalPairFormationCoefficient=readQuantity(
                    child.find("internalPairFormationCoefficient"))))
        elif child.tag == "continuum":
            spectrum.emissions.append(Continuum(spectrum=_readWrappedXYs1d(child, None)))
        else:
            tally(f"PoPs <spectrum><{child.tag}>: not a discrete line or a continuum")
    return spectrum


def writeDecayData(node: ET.Element, decayData, report: Optional[ConversionReport] = None) -> None:
    """The model → §12 ``decayData``, children in ``gnds.xsd``'s order."""
    if decayData is None:
        return
    element = ET.SubElement(node, "decayData")
    if len(decayData.decayModes):
        modes = ET.SubElement(element, "decayModes")
        for mode in decayData.decayModes:
            _writeMode(modes, mode, report)
    averages = list(getattr(decayData, "averageEnergies", []) or [])
    if averages:
        holder = ET.SubElement(element, "averageEnergies")
        for average in averages:
            child = _set(ET.SubElement(holder, "averageEnergy"), label=average.label,
                         value=_number(average.value), unit=average.unit or "eV")
            _writeUncertainty(child, _sigma(average))


def _writeMode(parent: ET.Element, mode: DecayMode, report) -> None:
    child = _set(ET.SubElement(parent, "decayMode"), label=mode.label, mode=mode.mode)
    probability = _set(ET.SubElement(ET.SubElement(child, "probability"), "double"),
                       label=getattr(mode, "probabilityLabel", None) or "eval",
                       value=_number(mode.probability))
    _writeUncertainty(probability, getattr(mode, "probabilityUncertainty", None))
    if mode.photonEmissionProbabilities is not None:
        _writeShells(child, "photonEmissionProbabilities", mode.photonEmissionProbabilities.shells)
    if getattr(mode, "Q", None) is not None:
        writeQuantity(child, "Q", mode.Q)
    path = ET.SubElement(child, "decayPath")
    for decay in mode.decayPath:
        complete = getattr(decay, "complete", None)
        step = _set(ET.SubElement(path, "decay"), index=str(decay.index), mode=decay.mode or None,
                    complete=None if complete is None else ("true" if complete else "false"))
        if decay.products:
            products = ET.SubElement(step, "products")
            for product in decay.products:
                _set(ET.SubElement(products, "product"),
                     label=product.label or product.pid, pid=product.pid)
    spectra = list(getattr(mode, "spectra", []) or [])
    if spectra:
        holder = ET.SubElement(child, "spectra")
        for spectrum in spectra:
            _writeSpectrum(holder, spectrum, report)


def _writeSpectrum(parent: ET.Element, spectrum: Spectrum, report) -> None:
    from .encode import _function

    node = _set(ET.SubElement(parent, "spectrum"), label=spectrum.label, pid=spectrum.pid)
    for emission in spectrum.emissions:
        if isinstance(emission, Continuum):
            wrapper = ET.SubElement(node, "continuum")
            _function(wrapper, emission.spectrum, report or ConversionReport(),
                      f"spectrum[@label='{spectrum.label}']/continuum")
            continue
        line = _set(ET.SubElement(node, "discrete"), type=emission.type)
        writeQuantity(line, "intensity", emission.intensity, simple=True)
        writeQuantity(line, "energy", emission.energy, simple=True)
        if emission.internalConversionCoefficients:
            _writeShells(line, "internalConversionCoefficients",
                         emission.internalConversionCoefficients)
        if emission.photonEmissionProbabilities is not None:
            _writeShells(line, "photonEmissionProbabilities",
                         emission.photonEmissionProbabilities.shells)
        if emission.positronEmissionIntensity is not None:
            writeQuantity(line, "positronEmissionIntensity", emission.positronEmissionIntensity,
                          simple=True)
        if emission.internalPairFormationCoefficient is not None:
            writeQuantity(line, "internalPairFormationCoefficient",
                          emission.internalPairFormationCoefficient, simple=True)


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------

def readAliases(element: Optional[ET.Element], tally) -> dict:
    aliases = {}
    if element is None:
        return aliases
    for child in element:
        if child.tag == "metaStable":
            alias = MetaStable(id=child.attrib["id"], pid=child.attrib["pid"],
                               metaStableIndex=int(child.attrib["metaStableIndex"]))
            aliases[alias.id] = alias
        else:
            tally(f"PoPs <aliases><{child.tag}>: only metaStable is read")
    return aliases


def writeAliases(container: ET.Element, aliases) -> None:
    if not aliases:
        return
    node = ET.SubElement(container, "aliases")
    for alias in aliases.values():
        _set(ET.SubElement(node, "metaStable"), id=alias.id, pid=alias.pid,
             metaStableIndex=str(alias.metaStableIndex))


# ---------------------------------------------------------------------------
# A PoPs file
# ---------------------------------------------------------------------------

def readPoPsDocument(document) -> Tuple[PoPs, ConversionReport]:
    """A GNDS file whose root is ``PoPs`` (a decay evaluation) → the model."""
    from .decode import _SuiteReader
    from .styles import readStyles

    report = ConversionReport()
    decoder = _SuiteReader(document, {}, report)
    root = document.root
    if root.tag != "PoPs":
        raise ValueError(f"the root is <{root.tag}>, not <PoPs>")
    pops = decoder.readPoPs(root)
    styles = root.find("styles")
    if styles is not None:
        pops.styles = list(readStyles(styles, "/PoPs/styles", report, decoder.tally))
    decoder.flushTallies()
    pops.report = report
    return pops, report


def writePoPsDocument(pops: PoPs, report: Optional[ConversionReport] = None
                      ) -> Tuple[ET.ElementTree, ConversionReport]:
    """The model → a GNDS ``PoPs`` file, with its ``styles``."""
    from .encode import _number as number
    from .encode_pops import writePoPs
    from .styles import writeStyles
    from kika.nuclear_data.model.styles import Styles

    report = report if report is not None else ConversionReport()
    holder = ET.Element("holder")
    root = writePoPs(holder, pops, report)
    root.set("format", "2.0")
    if pops.styles:
        styles = Styles()
        for style in pops.styles:
            styles.add(style)
        writeStyles(root, styles, number, report=report)
        # styles come first in a PoPs file (gnds.xsd PoPs_Type)
        node = root.find("styles")
        root.remove(node)
        root.insert(0, node)
    return ET.ElementTree(root), report

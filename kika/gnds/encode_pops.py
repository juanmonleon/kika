"""Serialize the same modeled PoPs fields at suite and formalism scope."""
import xml.etree.ElementTree as ET
from typing import Dict, List
from kika.nuclear_data.model import Nuclide, ConversionReport, Unorthodox
from kika._constants import ATOMIC_NUMBER_TO_NAME, ATOMIC_NUMBER_TO_SYMBOL
from .primitives import formatFraction


def _number(value):
    from .encode import _number as number
    return number(value)


def _set(element, **attributes):
    for name, value in attributes.items():
        if value is not None:
            element.set(name, value)
    return element


def writePoPs(root: ET.Element, pops, report: ConversionReport) -> ET.Element:
    """Write represented particle data; source losses stay in the reader report."""
    container = ET.SubElement(root, "PoPs")
    _set(container, name=pops.name or "protare_internal",
         version=pops.version or "1.0", format="2.0")
    nuclides = [p for p in pops.particles.values()
                if isinstance(p, Nuclide)]
    unorthodoxes = [p for p in pops.particles.values()
                    if isinstance(p, Unorthodox)]
    others = [p for p in pops.particles.values()
              if not isinstance(p, (Nuclide, Unorthodox))]

    for particle in others:
        wrapper = ("gaugeBosons" if particle.id == "photon" else "baryons")
        group = container.find(wrapper)
        if group is None:
            group = ET.SubElement(container, wrapper)
        node = ET.SubElement(group, wrapper[:-1])
        node.attrib["id"] = particle.id
        _particleProperties(node, particle)

    if nuclides:
        elements = ET.SubElement(container, "chemicalElements")
        byZ: Dict[int, List[Nuclide]] = {}
        for nuclide in nuclides:
            if nuclide.Z is None or nuclide.A is None:
                report.lost(f"PoPs nuclide {nuclide.id!r}: Z/A is missing; isotope cannot be written")
                continue
            byZ.setdefault(nuclide.Z, []).append(nuclide)
        for Z in sorted(k for k in byZ if k is not None):
            chemical = ET.SubElement(elements, "chemicalElement")
            symbol = ATOMIC_NUMBER_TO_SYMBOL[Z]
            # `name` is the element's name: FUDGE refuses a file whose name and Z
            # disagree, and the symbol written here before was such a file.
            _set(chemical, symbol=symbol, Z=str(Z), name=ATOMIC_NUMBER_TO_NAME[Z])
            isotopes = ET.SubElement(chemical, "isotopes")
            byA: Dict[int, List[Nuclide]] = {}
            for nuclide in byZ[Z]:
                byA.setdefault(nuclide.A, []).append(nuclide)
            for A in sorted(k for k in byA if k is not None):
                isotope = ET.SubElement(isotopes, "isotope")
                _set(isotope, symbol=symbol+str(A), A=str(A))
                holder = ET.SubElement(isotope, "nuclides")
                for nuclide in byA[A]:
                    node = ET.SubElement(holder, "nuclide")
                    node.attrib["id"] = nuclide.id
                    _nuclideProperties(node, nuclide)

    if unorthodoxes:
        # A thermal-scattering target (roadmap E4b). PoPs_UnorthodoxType admits
        # mass, charge and decayData only, so nothing else is written.
        group = ET.SubElement(container, "unorthodoxes")
        for particle in unorthodoxes:
            node = ET.SubElement(group, "unorthodox")
            node.attrib["id"] = particle.id
            if particle.mass is not None:
                _set(ET.SubElement(ET.SubElement(node, "mass"), "double"),
                     label="eval", value=_number(particle.mass.value),
                     unit=particle.mass.unit)
            if particle.charge is not None:
                _set(ET.SubElement(ET.SubElement(node, "charge"), "integer"),
                     label="eval", value=str(particle.charge), unit="e")
            _decayData(node, particle.decayData)

    if len(pops):
        report.warn("PoPs serialization covers all represented particle fields; "
                    "unsupported source particle data remain in the decoding report")
    return container



def _particleProperties(node: ET.Element, particle) -> None:
    if particle.mass is not None:
        _set(ET.SubElement(ET.SubElement(node, "mass"), "double"),
             label="eval", value=_number(particle.mass.value),
             unit=particle.mass.unit)
    if particle.spin is not None:
        _set(ET.SubElement(ET.SubElement(node, "spin"), "fraction"),
             label="eval", value=formatFraction(particle.spin.value),
             unit=particle.spin.unit)
    if particle.parity is not None:
        _set(ET.SubElement(ET.SubElement(node, "parity"), "integer"),
             label="eval", value=str(particle.parity))
    if particle.charge is not None:
        _set(ET.SubElement(ET.SubElement(node, "charge"), "integer"),
             label="eval", value=str(particle.charge), unit="e")
    _halflife(node, particle.halflife)
    _decayData(node, particle.decayData)


def _decayData(node: ET.Element, decayData) -> None:
    """§12 ``decayData/decayModes`` (roadmap E5c), in the schema's order:
    ``probability``, ``photonEmissionProbabilities``, ``decayPath``."""
    if decayData is None:
        return
    element = ET.SubElement(node, "decayData")
    if not len(decayData.decayModes):
        return
    modes = ET.SubElement(element, "decayModes")
    for mode in decayData.decayModes:
        child = _set(ET.SubElement(modes, "decayMode"), label=mode.label, mode=mode.mode)
        _set(ET.SubElement(ET.SubElement(child, "probability"), "double"),
             label="eval", value=_number(mode.probability))
        if mode.photonEmissionProbabilities is not None:
            emission = ET.SubElement(child, "photonEmissionProbabilities")
            for shell in mode.photonEmissionProbabilities.shells:
                _set(ET.SubElement(emission, "shell"), label=shell.label,
                     value=_number(shell.value))
        path = ET.SubElement(child, "decayPath")
        for decay in mode.decayPath:
            step = _set(ET.SubElement(path, "decay"), index=str(decay.index), mode=decay.mode)
            if decay.products:
                products = ET.SubElement(step, "products")
                for product in decay.products:
                    _set(ET.SubElement(products, "product"),
                         label=product.label or product.pid, pid=product.pid)


def _halflife(node: ET.Element, halflife) -> None:
    """§12's ``halflife``, in whichever of its two spellings the model holds.

    Mandatory on a ``baryon`` and a ``gaugeBoson``, so a particle with none
    gets ``<string value="unknown">`` — which is a real §12 value and the
    only honest thing to write: kika does not know, and the alternatives are
    omitting a required element or asserting a number.
    """
    element = ET.SubElement(node, "halflife")
    if halflife is None:
        _set(ET.SubElement(element, "string"), label="eval",
             value="unknown", unit="s")
    elif isinstance(halflife, str):
        _set(ET.SubElement(element, "string"), label="eval",
             value=halflife, unit="s")
    else:
        _set(ET.SubElement(element, "double"), label="eval",
             value=_number(halflife.value), unit=halflife.unit or "s")


def _nuclideProperties(node: ET.Element, nuclide: Nuclide) -> None:
    """The atom's mass and charge on the ``nuclide``, the nucleus's spin and
    parity on the ``nucleus`` — which is where each was read from."""
    if nuclide.mass is not None:
        _set(ET.SubElement(ET.SubElement(node, "mass"), "double"),
             label="eval", value=_number(nuclide.mass.value),
             unit=nuclide.mass.unit)
    if nuclide.charge is not None:
        _set(ET.SubElement(ET.SubElement(node, "charge"), "integer"),
             label="eval", value=str(nuclide.charge), unit="e")
    _decayData(node, nuclide.decayData)
    nucleus = ET.SubElement(node, "nucleus")
    _set(nucleus, id=nuclide.id.lower(), index=str(nuclide.nuclearLevel))
    if nuclide.spin is not None:
        _set(ET.SubElement(ET.SubElement(nucleus, "spin"), "fraction"),
             label="eval", value=formatFraction(nuclide.spin.value),
             unit=nuclide.spin.unit)
    if nuclide.parity is not None:
        _set(ET.SubElement(ET.SubElement(nucleus, "parity"), "integer"),
             label="eval", value=str(nuclide.parity))
    if nuclide.Z is not None:
        _set(ET.SubElement(ET.SubElement(nucleus, "charge"), "integer"),
             label="eval", value=str(nuclide.Z), unit="e")
    if nuclide.halflife is not None:
        _halflife(nucleus, nuclide.halflife)
    if nuclide.energy is not None:
        _set(ET.SubElement(ET.SubElement(nucleus, "energy"), "double"),
             label="eval", value=_number(nuclide.energy.value),
             unit=nuclide.energy.unit or "eV")

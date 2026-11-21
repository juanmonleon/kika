"""OpenMC ``materials.xml``: write a material as a ``<material>`` element and read one back.

Written by hand, so OpenMC need not be installed; a user who has it reads the file with
``openmc.Materials.from_xml``. Nuclides are named as OpenMC names them (``U235``,
``Am242_m1``); OpenMC libraries hold no natural elements, so those are expanded into
their isotopes first.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .material import Material, MaterialCollection, _num

__all__ = ["read_openmc_materials"]

_UNITS_OUT = {"g/cc": "g/cm3", "kg/m3": "kg/m3", "atoms/b-cm": "atom/b-cm"}
_UNITS_IN = {"g/cm3": "g/cc", "g/cc": "g/cc", "kg/m3": "kg/m3", "atom/b-cm": "atoms/b-cm"}


def openmc_name(material: Material, symbol: str) -> str:
    from ..hdf5.names import gndsName

    nuclide = material.nuclide[symbol]
    return gndsName(nuclide.z, nuclide.a, nuclide.isomer)


def expanded(material: Material) -> Material:
    """A copy of *material* with every natural element split into its isotopes."""
    copy = material.copy(material.id)
    if any(n.is_natural for n in copy.nuclide.values()):
        copy.expand_natural_elements()
    return copy


def sab_tables(material: Material) -> Tuple[List[str], List[str]]:
    """The S(alpha,beta) tables named as OpenMC names them (``c_...``), and the others."""
    ours = [t for t in material.thermal_scattering if t.startswith("c_")]
    return ours, [t for t in material.thermal_scattering if not t.startswith("c_")]


def material_element(material: Material) -> ET.Element:
    """The ``<material>`` element of *material*, natural elements expanded."""
    if material.metadata.get("serpent_density_sum"):
        units, value = "sum", None
    elif material.density is None or material.density_unit is None:
        raise ValueError(f"Material {material.id} has no density, which an OpenMC material needs")
    else:
        units, value = _UNITS_OUT[material.density_unit], material.density
    full = expanded(material)
    attrs = {"id": str(material.id)}
    if material.name:
        attrs["name"] = material.name
    if material.temperature is not None:
        attrs["temperature"] = _num(material.temperature)
    if material.metadata.get("vol") is not None:
        attrs["volume"] = _num(material.metadata["vol"])
    if material.metadata.get("depletable") is not None:
        attrs["depletable"] = "true" if material.metadata["depletable"] else "false"
    element = ET.Element("material", attrs)
    density = ET.SubElement(element, "density", {"units": units})
    if value is not None:
        density.set("value", _num(value))
    kind = "wo" if full.is_weight else "ao"
    for symbol, nuclide in sorted(full.nuclide.items(), key=lambda item: (item[1].z, item[1].a, item[1].isomer)):
        ET.SubElement(element, "nuclide", {"name": openmc_name(full, symbol), kind: _num(nuclide.fraction)})
    for table in sab_tables(material)[0]:
        ET.SubElement(element, "sab", {"name": table})
    return element


def to_xml(elements: List[ET.Element], cross_sections: Optional[str] = None) -> str:
    root = ET.Element("materials")
    if cross_sections is not None:
        ET.SubElement(root, "cross_sections").text = str(cross_sections)
    root.extend(elements)
    ET.indent(root, space="  ")
    return "<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(root, encoding="unicode") + "\n"


def _read_material(node: ET.Element) -> Material:
    from ..hdf5.names import parseGndsName

    material = Material(id=int(node.get("id")), name=node.get("name"))
    if node.get("temperature") is not None:
        material.set_temperature(float(node.get("temperature")))
    if node.get("volume") is not None:
        material.metadata["vol"] = float(node.get("volume"))
    if node.get("depletable") is not None:
        material.metadata["depletable"] = node.get("depletable").strip().lower() == "true"
    density = node.find("density")
    units = density.get("units") if density is not None else None
    if units == "sum":
        material.metadata["serpent_density_sum"] = True
    elif units == "atom/cm3":
        material.set_density(float(density.get("value")) * 1e-24, "atoms/b-cm")
    elif units in _UNITS_IN:
        material.set_density(float(density.get("value")), _UNITS_IN[units])
    else:
        raise ValueError(f"OpenMC material {material.id}: density units {units!r} are not supported")
    kinds = set()
    for nuclide in node.findall("nuclide"):
        z, a, m = parseGndsName(nuclide.get("name"))
        kind = "ao" if nuclide.get("ao") is not None else "wo"
        kinds.add(kind)
        if len(kinds) > 1:
            raise ValueError(f"OpenMC material {material.id} has mixed atomic/weight fractions")
        zaid = m * 1000000 + z * 1000 + a
        if zaid in material.nuclide:
            raise ValueError(f"OpenMC material {material.id} lists {nuclide.get('name')} more than once")
        material.add_nuclide(zaid, float(nuclide.get(kind)), kind)
    for unknown in ("element", "macroscopic", "isotropic"):
        if node.find(unknown) is not None:
            raise ValueError(f"OpenMC material {material.id}: <{unknown}> is not supported")
    for sab in node.findall("sab"):
        if float(sab.get("fraction", "1")) != 1.0:
            material.report.lost(f"OpenMC material {material.id}: the fraction {sab.get('fraction')} "
                                 f"of S(alpha,beta) table {sab.get('name')} is not kept")
        material.thermal_scattering.append(sab.get("name"))
    return material


def read_openmc_materials(source: Union[str, Path]) -> MaterialCollection:
    """Read the materials of an OpenMC ``materials.xml``.

    Nuclides become kika ZAIDs (``Am242_m1`` is ``1095242``); ``ao``/``wo`` give the
    fraction type; ``<sab>`` names go to ``thermal_scattering``; ``volume`` and
    ``depletable`` go to ``metadata``; a density in ``atom/cm3`` is stored in
    atoms/b-cm and ``sum`` as for Serpent's.

    Parameters
    ----------
    source
        Path of the file, or its text when it starts with ``<``.

    Returns
    -------
    ~kika.materials.MaterialCollection
        Keyed by the OpenMC material ids.

    Raises
    ------
    ValueError
        On ``macro`` densities, ``<element>``, ``<macroscopic>`` or
        ``<isotropic>`` nodes (kika keeps nuclides only), mixed ``ao``/``wo``,
        or a nuclide listed twice.
    """
    text = str(source)
    if not text.lstrip().startswith("<"):
        text = Path(source).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    collection = MaterialCollection()
    for node in root.findall("material"):
        collection.add_material(_read_material(node))
    return collection

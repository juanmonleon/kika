"""GNDS ``doubleDifferentialCrossSection`` for the thermal scattering law (roadmap E4b).

The model nodes are :mod:`kika.nuclear_data.model.thermal_scattering` (E4a,
built from ENDF MF7). This module reads and writes them, with FUDGE's node
names, attribute rules and child order (``fudge/reactionData/
doubleDifferentialCrossSection/thermalNeutronScatteringLaw/`` and ``gnds.xsd``
TNSL types):

- ``thermalNeutronScatteringLaw_coherentElastic`` → ``S_table/gridded2d``;
- ``thermalNeutronScatteringLaw_incoherentElastic`` → ``boundAtomCrossSection``
  and ``DebyeWallerIntegral/XYs1d``;
- ``thermalNeutronScatteringLaw_incoherentInelastic`` → ``scatteringAtoms``,
  each ``mass``, ``e_critical``?, ``e_max``, ``boundAtomCrossSection``,
  ``boundAtomCrossSectionByNuclide``?, ``coherentAtomCrossSection``?,
  ``distinctScatteringKernel``?, ``selfScatteringKernel``, ``T_effective``?
  -- the schema's sequence, which is also the order FUDGE writes.

Two choices worth stating:

- **``symmetric`` keeps "unsaid".** FUDGE writes it only when true and reads a
  missing attribute as false. kika writes ``true`` *and* ``false`` when the model
  knows, and reads a missing attribute as ``None``, so a file that did not say
  is not made to say no.
- **Arrays are written full.** FUDGE switches S(α,β,T) to ``flattened`` when
  more than 10 % of it is zero; both are GNDS and the reader takes both. Full
  is the one with no rule to reproduce.

``BraggEdges`` (the schema's alternative to ``S_table``) has no ENDF
counterpart and FUDGE neither reads nor writes it; meeting one is reported.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable, Optional

import numpy as np

from kika.nuclear_data.model import (CoherentElastic, ConversionReport,
                                     DistinctScatteringKernel,
                                     DoubleDifferentialCrossSection,
                                     FreeGasApproximation, GaussianApproximation,
                                     Gridded2d, Gridded3d, IncoherentElastic,
                                     IncoherentInelastic, PhysicalQuantity,
                                     Regions1d, ScatteringAtom, SCTApproximation,
                                     SelfScatteringKernel, XYs1d)
from kika.nuclear_data.model.axes import Axes, Axis, Grid

from .nodes import reads, writes

__all__ = ["writeDoubleDifferentialCrossSection", "readDoubleDifferentialCrossSection",
           "writeGridded", "readGridded"]


# ---------------------------------------------------------------------------
# helpers shared by both directions
# ---------------------------------------------------------------------------

def _encode():
    # The encoder imports this module; importing it back lazily keeps one copy
    # of the number formatting and of the functional writer.
    from . import encode
    return encode


def _quantity(parent: ET.Element, tag: str, quantity: Optional[PhysicalQuantity],
              **extra) -> None:
    if quantity is None:
        return
    enc = _encode()
    element = ET.SubElement(parent, tag)
    enc._set(element, **extra)
    element.attrib["value"] = enc._number(quantity.value)
    element.attrib["unit"] = quantity.unit or ""


def _readQuantity(element: Optional[ET.Element]) -> Optional[PhysicalQuantity]:
    if element is None or "value" not in element.attrib:
        return None
    return PhysicalQuantity(float(element.attrib["value"]), element.attrib.get("unit", ""))


def _boolean(element: ET.Element, name: str) -> Optional[bool]:
    value = element.attrib.get(name)
    if value is None:
        return None
    return value.strip().lower() in ("true", "1")


# ---------------------------------------------------------------------------
# gridded2d / gridded3d
# ---------------------------------------------------------------------------

def writeGridded(parent: ET.Element, form) -> ET.Element:
    """§6.2.6: ``axes`` (grids first, then the dependent axis), then a full ``array``."""
    enc = _encode()
    tag = "gridded2d" if isinstance(form, Gridded2d) else "gridded3d"
    element = ET.SubElement(parent, tag)
    axes = ET.SubElement(element, "axes")
    # GridAxesType is a sequence: every <grid> before every <axis>.
    for grid in form.grids:
        node = ET.SubElement(axes, "grid")
        enc._set(node, index=str(grid.index), label=grid.label or "",
                 unit=grid.unit or "", style=str(grid.style.value),
                 interpolation=str(grid.interpolation))
        enc._values(node, grid.values)
    dependent = form.dependentAxis
    enc._set(ET.SubElement(axes, "axis"), index="0", label=dependent.label or "",
             unit=dependent.unit or "")
    values = np.asarray(form.values, dtype=float)
    array = ET.SubElement(element, "array")
    array.attrib["shape"] = ",".join(str(n) for n in values.shape)
    enc._values(array, values)
    return element


def _readArray(element: ET.Element) -> np.ndarray:
    """``array`` of any rank: ``compression`` absent or ``flattened``.

    ``flattened`` is what FUDGE writes for a sparse S(α,β,T): ``starts`` and
    ``lengths`` index runs into the full array, row-major.
    """
    from .primitives import readValues

    shape = tuple(int(part) for part in element.attrib["shape"].split(","))
    compression = element.attrib.get("compression")
    if element.attrib.get("symmetry") not in (None, "none"):
        raise ValueError("a gridded TSL table is not a symmetric matrix")
    labelled = {v.attrib.get("label", "data"): readValues(v) for v in element.findall("values")}
    if compression is None:
        data = labelled.get("data", next(iter(labelled.values()), np.empty(0)))
        return np.asarray(data, dtype=float).reshape(shape)
    if compression == "flattened":
        full = np.zeros(int(np.prod(shape)), dtype=float)
        data = labelled["data"]
        position = 0
        for start, length in zip(labelled["starts"].astype(int).tolist(),
                                 labelled["lengths"].astype(int).tolist()):
            full[start:start + length] = data[position:position + length]
            position += length
        return full.reshape(shape)
    raise ValueError(f"array compression={compression!r} is not read for a gridded table")


def readGridded(element: ET.Element, resolve: Optional[Callable] = None):
    from .primitives import readAxes

    axes = readAxes(element, resolve)
    array = element.find("array")
    if axes is None or array is None:
        raise ValueError(f"<{element.tag}> needs <axes> and <array>")
    values = _readArray(array)
    cls = Gridded2d if element.tag == "gridded2d" else Gridded3d
    return cls(values=values, axes=axes)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _withoutRepeatedPoints(function: XYs1d, report: ConversionReport, where: str) -> XYs1d:
    """Drop a point that repeats the one before it exactly, x and y both.

    ENDF/B-VIII.1 s-CH4 states W'(T) at one temperature as two identical
    points, a legal TAB1. A GNDS ``XYs1d`` must be ascending in x, and FUDGE
    refuses the file ("data not in ascending order") -- so the GNDS writer drops
    the repeat, as FUDGE's own ENDF translator does (``ENDF_ITYPE_2.py:357``).
    The model keeps the tape's two points; only the file loses the copy.
    """
    xs = np.asarray(function.xs, dtype=float)
    ys = np.asarray(function.ys, dtype=float)
    if xs.size < 2:
        return function
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = ~((xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1]))
    if keep.all():
        return function
    report.approximated(f"{where}: {int((~keep).sum())} point(s) repeating the one "
                        f"before exactly are not written; a GNDS XYs1d is ascending in x")
    return XYs1d(xs=xs[keep], ys=ys[keep], interpolation=function.interpolation,
                 axes=function.axes, label=function.label)


def _wrappedXYs1d(parent: ET.Element, tag: str, function, report: ConversionReport,
                  where: str) -> None:
    """``XYs1dWrapperType``: one primary ``XYs1d`` inside *tag*."""
    if function is None:
        return
    enc = _encode()
    wrapper = ET.SubElement(parent, tag)
    if isinstance(function, XYs1d):
        function = _withoutRepeatedPoints(function, report, f"{where}/{tag}")
    if isinstance(function, Regions1d) and len(function.function1ds) > 1:
        report.lost(f"{where}/{tag}: {len(function.function1ds)} interpolation regions; "
                    f"the schema admits a single XYs1d here, so a regions1d is written "
                    f"that a validator will reject")
    enc._function(wrapper, function, report, f"{where}/{tag}")


@writes("scatteringKernelForm", "gridded3d", "SCTApproximation",
        "freeGasApproximation", "GaussianApproximation")
def _writeKernel(parent: ET.Element, kernel, report, where) -> None:
    if isinstance(kernel, (Gridded2d, Gridded3d)):
        writeGridded(parent, kernel)
    elif isinstance(kernel, SCTApproximation):
        ET.SubElement(parent, "SCTApproximation")
    elif isinstance(kernel, FreeGasApproximation):
        ET.SubElement(parent, "freeGasApproximation")
    elif isinstance(kernel, GaussianApproximation):
        node = ET.SubElement(parent, "GaussianApproximation")
        _wrappedXYs1d(node, "phononSpectrum", kernel.phononSpectrum, report, where)
    else:
        report.unsupportedNode(f"{where}: a {type(kernel).__name__} kernel has no GNDS node")


def _writeAtom(parent: ET.Element, atom: ScatteringAtom, report, where) -> None:
    enc = _encode()
    element = ET.SubElement(parent, "scatteringAtom")
    enc._set(element, pid=atom.pid, numberPerMolecule=str(int(atom.numberPerMolecule)),
             primaryScatterer="true" if atom.primaryScatterer else None)
    here = f"{where}/scatteringAtom[@pid='{atom.pid}']"
    _quantity(element, "mass", atom.mass)
    _quantity(element, "e_critical", atom.e_critical)
    _quantity(element, "e_max", atom.e_max)
    _quantity(element, "boundAtomCrossSection", atom.boundAtomCrossSection)
    if atom.boundAtomCrossSectionByNuclide:
        byNuclide = ET.SubElement(element, "boundAtomCrossSectionByNuclide")
        for pid, quantity in atom.boundAtomCrossSectionByNuclide.items():
            _quantity(byNuclide, "boundAtomCrossSection", quantity, pid=pid)
    _quantity(element, "coherentAtomCrossSection", atom.coherentAtomCrossSection)
    if atom.distinctScatteringKernel is not None:
        writeGridded(ET.SubElement(element, "distinctScatteringKernel"),
                     atom.distinctScatteringKernel.kernel)
    self_ = atom.selfScatteringKernel
    kernel = ET.SubElement(element, "selfScatteringKernel")
    if self_.symmetric is not None:
        kernel.attrib["symmetric"] = "true" if self_.symmetric else "false"
    _writeKernel(kernel, self_.kernel, report, here)
    _wrappedXYs1d(element, "T_effective", atom.T_effective, report, here)


@writes("coherentElasticForm", "S_table")
@writes("doubleDifferentialForm", "thermalNeutronScatteringLaw_coherentElastic",
        "thermalNeutronScatteringLaw_incoherentElastic",
        "thermalNeutronScatteringLaw_incoherentInelastic")
def writeDoubleDifferentialCrossSection(parent: ET.Element, ddcs, report: ConversionReport,
                                        where: str) -> Optional[ET.Element]:
    """The model's ``doubleDifferentialCrossSection`` → its node, or ``None`` if empty."""
    if ddcs is None or not len(ddcs):
        return None
    enc = _encode()
    container = ET.SubElement(parent, "doubleDifferentialCrossSection")
    for label, form in ddcs.items():
        here = f"{where} doubleDifferentialCrossSection[{label!r}]"
        if not isinstance(form, (CoherentElastic, IncoherentElastic, IncoherentInelastic)):
            report.unsupportedNode(f"{here}: kika's writer has no serialisation for a "
                                   f"{type(form).__name__}")
            continue
        element = ET.SubElement(container, form.gndsNodeName)
        enc._set(element, label=label or form.label, pid=form.pid,
                 productFrame=str(form.productFrame))
        if isinstance(form, CoherentElastic):
            writeGridded(ET.SubElement(element, "S_table"), form.S_table)
        elif isinstance(form, IncoherentElastic):
            _quantity(element, "boundAtomCrossSection", form.boundAtomCrossSection)
            _wrappedXYs1d(element, "DebyeWallerIntegral", form.DebyeWallerIntegral, report, here)
        else:
            enc._set(element, primaryScatterer=form.primaryScatterer,
                     calculatedAtThermal="true" if form.calculatedAtThermal else None,
                     incoherentApproximation=None if form.incoherentApproximation else "false")
            atoms = ET.SubElement(element, "scatteringAtoms")
            for atom in form.scatteringAtoms:
                _writeAtom(atoms, atom, report, here)
    return container


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _readWrappedXYs1d(element: Optional[ET.Element], resolve):
    from .primitives import readAxes, readForm

    if element is None:
        return None
    children = [c for c in element if c.tag in ("XYs1d", "regions1d")]
    if not children:
        raise ValueError(f"<{element.tag}> holds no XYs1d")
    child = children[0]
    return readForm(child, readAxes(child, resolve))


@reads("scatteringKernelForm", "gridded3d", "SCTApproximation",
       "freeGasApproximation", "GaussianApproximation")
def _readKernel(element: ET.Element, resolve, report, here):
    for child in element:
        if child.tag in ("gridded3d", "gridded2d"):
            return readGridded(child, resolve)
        if child.tag == "SCTApproximation":
            return SCTApproximation()
        if child.tag == "freeGasApproximation":
            return FreeGasApproximation()
        if child.tag == "GaussianApproximation":
            return GaussianApproximation(
                phononSpectrum=_readWrappedXYs1d(child.find("phononSpectrum"), resolve))
    report.unsupportedNode(f"{here}/selfScatteringKernel: no kernel kika reads")
    return None


def _readAtom(element: ET.Element, resolve, report, where) -> Optional[ScatteringAtom]:
    pid = element.attrib.get("pid", "")
    here = f"{where}/scatteringAtom[@pid='{pid}']"
    selfNode = element.find("selfScatteringKernel")
    if selfNode is None:
        report.unsupportedNode(f"{here}: no selfScatteringKernel; the atom is not read")
        return None
    kernel = _readKernel(selfNode, resolve, report, here)
    if kernel is None:
        return None
    byNuclide = {}
    container = element.find("boundAtomCrossSectionByNuclide")
    for child in (container if container is not None else []):
        # FUDGE keys these by `label` in memory and the schema by `pid`.
        key = child.attrib.get("pid") or child.attrib.get("label") or ""
        byNuclide[key] = _readQuantity(child)
    distinct = element.find("distinctScatteringKernel")
    return ScatteringAtom(
        pid=pid,
        numberPerMolecule=int(element.attrib.get("numberPerMolecule", "1")),
        mass=_readQuantity(element.find("mass")),
        e_max=_readQuantity(element.find("e_max")),
        boundAtomCrossSection=_readQuantity(element.find("boundAtomCrossSection")),
        selfScatteringKernel=SelfScatteringKernel(kernel, symmetric=_boolean(selfNode, "symmetric")),
        primaryScatterer=bool(_boolean(element, "primaryScatterer")),
        e_critical=_readQuantity(element.find("e_critical")),
        T_effective=_readWrappedXYs1d(element.find("T_effective"), resolve),
        boundAtomCrossSectionByNuclide=byNuclide,
        coherentAtomCrossSection=_readQuantity(element.find("coherentAtomCrossSection")),
        distinctScatteringKernel=(None if distinct is None else DistinctScatteringKernel(
            kernel=readGridded(distinct.find("gridded3d"), resolve))),
    )


@reads("coherentElasticForm", "S_table")
@reads("doubleDifferentialForm", "thermalNeutronScatteringLaw_coherentElastic",
       "thermalNeutronScatteringLaw_incoherentElastic",
       "thermalNeutronScatteringLaw_incoherentInelastic")
def readDoubleDifferentialCrossSection(element: ET.Element, path: str, resolve,
                                       report: ConversionReport) -> DoubleDifferentialCrossSection:
    here = f"{path}/doubleDifferentialCrossSection"
    ddcs = DoubleDifferentialCrossSection()
    for child in element:
        label = child.attrib.get("label", "")
        common = dict(label=label, pid=child.attrib.get("pid", "n"),
                      productFrame=child.attrib.get("productFrame", "lab"))
        try:
            if child.tag == "thermalNeutronScatteringLaw_coherentElastic":
                table = child.find("S_table")
                if table is None:
                    report.unsupportedNode(
                        f"{here}/{child.tag}: holds {[c.tag for c in child]} and not "
                        f"S_table; BraggEdges has no ENDF counterpart and kika does not "
                        f"read it (nor does FUDGE)")
                    continue
                gridded = table.find("gridded2d")
                form = CoherentElastic(S_table=readGridded(gridded, resolve), **common)
            elif child.tag == "thermalNeutronScatteringLaw_incoherentElastic":
                form = IncoherentElastic(
                    boundAtomCrossSection=_readQuantity(child.find("boundAtomCrossSection")),
                    DebyeWallerIntegral=_readWrappedXYs1d(child.find("DebyeWallerIntegral"),
                                                          resolve),
                    **common)
            elif child.tag == "thermalNeutronScatteringLaw_incoherentInelastic":
                atoms = []
                for node in child.findall("scatteringAtoms/scatteringAtom"):
                    atom = _readAtom(node, resolve, report, f"{here}/{child.tag}")
                    if atom is not None:
                        atoms.append(atom)
                approximation = _boolean(child, "incoherentApproximation")
                form = IncoherentInelastic(
                    scatteringAtoms=atoms,
                    primaryScatterer=child.attrib.get("primaryScatterer", ""),
                    calculatedAtThermal=bool(_boolean(child, "calculatedAtThermal")),
                    incoherentApproximation=True if approximation is None else approximation,
                    **common)
            else:
                report.unsupportedNode(
                    f"{here}/{child.tag}: a double-differential form that is not the "
                    f"thermal scattering law; kika reads only the three TNSL forms")
                continue
        except (ValueError, KeyError) as exc:
            report.unsupportedNode(f"{here}/{child.tag}: {exc}")
            continue
        ddcs[label] = form
    return ddcs

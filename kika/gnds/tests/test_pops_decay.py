"""§12's level energies and electromagnetic decay, read and written (roadmap E5c).

NNDC's S-36 (ENDF/B-VIII.1) has five excited levels, each with a
``decayData/decayModes`` cascade and a ``nucleus/energy``. Until E5c both were
counted and dropped, so the GNDS kika wrote back had ``branching1d`` products
pointing at a cascade that was no longer there. Valid against the schema,
declared in the report, and without the physics.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import ELECTROMAGNETIC, DecayData


def _pops(path):
    return ET.parse(path).getroot().find("PoPs")


def _levels(root):
    out = {}
    for nuclide in root.iter("nuclide"):
        energy = nuclide.find("nucleus/energy/double")
        modes = [
            (mode.attrib["mode"],
             float(mode.find("probability/double").attrib["value"]),
             [float(s.attrib["value"]) for s in mode.findall("photonEmissionProbabilities/shell")],
             [p.attrib["pid"] for p in mode.findall("decayPath/decay/products/product")])
            for mode in nuclide.findall("decayData/decayModes/decayMode")]
        out[nuclide.attrib["id"]] = (
            float(energy.attrib["value"]) if energy is not None else None, modes)
    return out


def test_the_five_cascades_and_their_level_energies_are_read(s36_gnds):
    suite, report = readReactionSuite(Document.parse(s36_gnds))

    level = suite.PoPs["S36_e4"]
    assert level.energy.value == 4522990.0 and level.energy.unit == "eV"
    assert isinstance(level.decayData, DecayData)
    modes = list(level.decayData.decayModes)
    assert [m.probability for m in modes] == [0.248, 0.752]
    assert {m.mode for m in modes} == {ELECTROMAGNETIC}
    assert [m.finalState() for m in modes] == ["S36_e1", "S36"]
    assert modes[0].photonEmissionProbabilities.total() == 1.0

    excited = [p for p in suite.PoPs.particles.values()
               if getattr(p, "decayData", None) is not None]
    assert len(excited) == 5
    assert not any("decayData" in line or "<energy>" in line for line in report.losses)


def test_pops_round_trips_with_its_cascades(s36_gnds, tmp_path):
    first, _ = readReactionSuite(Document.parse(s36_gnds))
    path = tmp_path / "s36.gnds.xml"
    kika.write(first, path)

    assert _levels(_pops(path)) == _levels(_pops(s36_gnds))
    second, _ = readReactionSuite(Document.parse(path))
    assert second.PoPs["S36_e3"].decayData == first.PoPs["S36_e3"].decayData


def test_the_written_pops_validates(s36_gnds, tmp_path):
    """The decayData kika writes is the schema's shape (``gnds.xsd:525-565``)."""
    from kika.gnds.tests.test_encode import _schemaErrors

    first, _ = readReactionSuite(Document.parse(s36_gnds))
    path = tmp_path / "s36.gnds.xml"
    kika.write(first, path)
    errors = [e for e in _schemaErrors(path)
              if "decay" in e or "nucleus" in e or "energy" in e]
    assert errors == []

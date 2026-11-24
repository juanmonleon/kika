"""An ENDF-decoded MF3 is written as ``resonancesWithBackground`` and comes back as the same MF3."""
import xml.etree.ElementTree as ET

import kika
from kika.nuclear_data.model import Regions1d, XYs1d


def _mf3(path, mt):
    with open(path, encoding="ascii", errors="replace") as handle:
        return [line[:66] for line in handle
                if len(line) >= 75 and line[70:72] == " 3" and int(line[72:75]) == mt]


def test_the_resonance_mts_carry_the_resonances_and_the_model_keeps_its_table(micro_tape, tmp_path):
    suite = kika.read(str(micro_tape), covariances=False)
    written = tmp_path / "fe56.gnds.xml"
    kika.write(suite, written, format="gnds")

    root = ET.parse(written).getroot()
    for node in root.iter():
        if node.get("ENDF_MT") in ("1", "2", "102"):
            form = node.find("crossSection/resonancesWithBackground")
            assert form is not None, node.get("ENDF_MT")
            assert form.find("resonances").get("href") == "/reactionSuite/resonances"
            assert [c.tag for c in form.find("background")] == ["resolvedRegion", "fastRegion"]
    assert isinstance(suite.reactions[2].crossSection["eval"], (XYs1d, Regions1d))


def test_an_edge_the_tape_does_not_state_is_cut_flagged_and_deleted_on_the_way_back():
    import numpy as np

    from kika.endf.model_adapter.encode import backgroundTable
    from kika.gnds.resonance_background import Plan, wrap
    from kika.nuclear_data.model import crossSectionAxes

    table = XYs1d(xs=np.array([1.0, 4.0, 8.0, 20.0]), ys=np.array([1.0, 2.0, 3.0, 4.0]),
                  axes=crossSectionAxes())
    how = Plan(mts=frozenset({2}), resolvedMax=5.0, unresolvedMax=10.0, fissionInResolved=False)
    form, flags = wrap(how, 2, table, "eval")

    regions = form.background.regions
    assert list(regions["resolvedRegion"].xs) == [1.0, 4.0, 5.0]
    assert list(regions["unresolvedRegion"].xs) == [5.0, 8.0, 10.0]
    assert list(regions["fastRegion"].xs) == [10.0, 20.0]
    assert flags == {"resolvedRegion": "mergeWithNextRegion,deletePoint",
                     "unresolvedRegion": "mergeWithNextRegion,deletePoint"}
    energies, values, _ = backgroundTable(form, "MT2", joins=flags).toEndfRegions()
    assert list(energies) == [1.0, 4.0, 8.0, 20.0] and list(values) == [1.0, 2.0, 3.0, 4.0]


def test_the_background_goes_back_to_the_tapes_mf3(micro_tape, tmp_path):
    written = tmp_path / "fe56.gnds.xml"
    kika.write(kika.read(str(micro_tape), covariances=False), written, format="gnds")
    back = tmp_path / "fe56.endf"
    kika.write(kika.read(str(written), covariances=False), back, format="endf")
    for mt in (1, 2, 102):
        assert _mf3(back, mt) == _mf3(micro_tape, mt), f"MF3/MT{mt}"

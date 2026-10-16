"""An RML evaluation read from ENDF comes back from the GNDS kika writes of it.

The ENDF decoder numbered each channel's ``columnIndex`` from 0, while the GNDS
width table (§19.3.5) has the energy in column 0 and every distributed file
numbers the widths from 1. Written to GNDS, the first channel (Cl-35's capture)
pointed at the energy column; read back, its widths were zero and
``process_gnds`` refused the suite. Found on 2026-10-16 checking M6 on Cl-35
and Sr-88 of ENDF/B-VIII.1.
"""
from pathlib import Path

import kika
from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_cl35_mf32.endf"


def _groups(suite):
    (region,) = suite.resonances.resolved
    return region.formalism.spinGroups


def test_channels_are_numbered_from_the_first_width_column():
    suite, _ = decodeReactionSuite(read_endf(str(TAPE)))
    for group in _groups(suite):
        assert [c.columnIndex for c in group.channels] == list(range(1, len(group.channels) + 1))


def test_the_widths_come_back_from_gnds(tmp_path):
    suite, _ = decodeReactionSuite(read_endf(str(TAPE)))
    path = tmp_path / "cl35.gnds.xml"
    kika.write(suite, path, format="gnds")
    back = kika.read(str(path), format="gnds", covariances=False)
    assert back.report.isCleanFor("cross-sections"), back.report.losses
    for before, after in zip(_groups(suite), _groups(back)):
        assert [c.resonanceReaction for c in after.channels] == [c.resonanceReaction for c in before.channels]
        assert after.widths == before.widths and after.energies == before.energies

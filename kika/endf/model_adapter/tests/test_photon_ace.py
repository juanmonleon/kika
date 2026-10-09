"""NJOY's ACE from a tape kika wrote is NJOY's ACE from the original (roadmap E5f.2).

The whole chain of ``kika.njoy.templates.NJOY_INPUT_TEMPLATE`` -- RECONR,
BROADR, HEATR, PURR, GASPR and ACER with photon production asked for (iopp=1)
-- run on the evaluation and on ``writeEndfTape(decodeReactionSuite(...))``.
Before E5 the second tape had no MF12-15 and the ACE no photon-production
blocks. The gate is the strongest one there is: the two ACE files are the same
text, line for line, so GPD, MTRP, SIGP, LANDP/ANDP, LDLWP/DLWP and YP are too.

Measured 2026-10-09 on ENDF/B-VIII.1 S-36 (LO=1 + LO=2 + MF6), JENDL-5 Hf-182
(MF12-15 on MT3) and ENDF/B-VIII.1 Fe-56 (66 cascades, 1 452 959 lines): all
three identical. S-36 is the one kept here; it runs in seconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.writers.assemble import writeEndfTape

pytestmark = pytest.mark.njoy


@pytest.fixture(scope="module")
def s36(neutron_libraries):
    path = neutron_libraries["endfb81"] / "n-016_S_036.endf"
    if not path.is_file():
        pytest.skip(f"{path} is not here")
    return path


def test_the_ace_of_kikas_tape_is_the_ace_of_the_original(s36, njoy_exe, tmp_path):
    from kika.ace.parsers.parse_ace import read_ace
    from kika.njoy.run_njoy import run_njoy

    suite, _ = decodeReactionSuite(read_endf(str(s36)))
    written = tmp_path / "kika.endf"
    writeEndfTape(suite, written)

    original = run_njoy(str(njoy_exe), s36, 293.6, "test", tmp_path / "orig", suff=".10")
    mine = run_njoy(str(njoy_exe), written, 293.6, "test", tmp_path / "kika", suff=".10")
    assert original["returncode"] == mine["returncode"] == 0
    theirs = Path(original["ace_file"]).read_text().splitlines()
    ours = Path(mine["ace_file"]).read_text().splitlines()
    assert ours == theirs

    ace = read_ace(mine["ace_file"])
    assert ace.photon_production_xs, "the ACE has no photon production at all"

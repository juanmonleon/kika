"""A processed suite through GNDS and back, with nothing lost.

GNDS states the physics in its own nodes (``xs_pdf_cdf1d``,
``energyAngularMC``, ``angularTwoBody`` for a level, ``discreteGamma`` for a
line, the ``MonteCarlo_cdf`` style); what it has no spelling for -- ACE's LP,
law 3's constants, a table's discrete lines, the URR tables whole, ACE's and
OpenMC's bookkeeping -- goes to ``applicationData`` ``KIKA::processed``
(kika/gnds/processed.py). The gates: an OpenMC file comes back as the same
HDF5 tree, and an ACE table as the same XSS.
"""
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.hdf5 import records as R
from kika.hdf5.neutron_decode import decodeNeutron
from kika.hdf5.neutron_encode import encodeNeutron

DATA = Path(__file__).resolve().parents[2] / "hdf5" / "tests" / "data"


def _through_gnds(suite, tmp_path):
    path = tmp_path / "processed.gnds.xml"
    report = kika.write(suite, path, format="gnds")
    return kika.read(str(path)), report, path


@pytest.mark.parametrize("name", ["Fe56", "U235", "Cu63", "H2"])
def test_an_openmc_file_comes_back_as_the_same_tree(name, tmp_path):
    tree = R.readFile(DATA / f"{name}.h5")
    suite, _ = decodeNeutron(tree)
    back, report, _ = _through_gnds(suite, tmp_path)
    assert not report.unsupported, report.unsupported[:3]
    assert R.compareTrees(tree, encodeNeutron(back)) == []


def test_an_ace_table_comes_back_as_the_same_xss(tmp_path):
    from kika.ace import read_ace
    from kika.ace.encoders.layout import encode_xss
    from kika.ace.model_adapter.encode import encodeAce
    from kika.hdf5.names import K_BOLTZMANN

    h5 = kika.read(str(DATA / "U235.h5"))
    T = [float(s.temperature.value) / K_BOLTZMANN for s in h5.styles
         if type(s).__name__ == "Heated"][0]
    kika.write(h5, tmp_path / "u235.ace", format="ace", temperatures=T, extension=".80c")
    original = read_ace(str(tmp_path / "u235.ace"))
    back, report, _ = _through_gnds(kika.read(str(tmp_path / "u235.ace")), tmp_path)
    written, _ = encodeAce(back)
    a, b = encode_xss(original), encode_xss(written)
    assert list(a.nxs) == list(b.nxs) and list(a.jxs) == list(b.jxs)
    # A4a's tolerance: the model's MeV <-> eV costs an ulp or two.
    np.testing.assert_allclose(b.xss[1:], a.xss[1:], rtol=1e-12, atol=0)


def test_the_physics_is_in_gnds_nodes_and_the_rest_in_the_block(tmp_path):
    """A reader that ignores KIKA::processed still finds the Monte Carlo
    tables and the MonteCarlo_cdf style. (This fixture's photons are
    continuous; the lines-as-discreteGamma case is in the ACE gate.)"""
    suite = kika.read(str(DATA / "U235.h5"))
    _, report, path = _through_gnds(suite, tmp_path)
    text = path.read_text(encoding="utf-8")
    for node in ("<MonteCarlo_cdf ", "<xs_pdf_cdf1d",
                 '<institution label="KIKA::processed">'):
        assert node in text, node
    assert text.count("<applicationData") == 1
    assert any("KIKA::processed" in line for line in report.warnings)


def test_an_evaluation_gets_no_block(tmp_path):
    """Only a processed suite carries KIKA::processed; an ENDF evaluation's
    GNDS is written as before (a yield with a step, MT2's recoil)."""
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite

    tape = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    _, _, path = _through_gnds(suite, tmp_path)
    assert "KIKA::processed" not in path.read_text(encoding="utf-8")

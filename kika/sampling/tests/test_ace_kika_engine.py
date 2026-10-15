"""``AceOptions(engine="kika")``: each realisation processed by kika and written as ACE, no NJOY.

A sample is the unperturbed evaluation's processing with the realisation's
forms in place of the evaluated ones (multiplied at 0 K, then broadened), so
the unperturbed processing written from scratch has to be the table
``EndfProcessing.write_ace`` writes, word for word, and two samples have to be
two different tables of the same layout.
"""
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.sampling.model_perturbation import AceOptions, perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None}

perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("kika-ace")
    return perturbFromModel(TAPE, REQUEST, 2, seed=20261015, outputDir=out, formats=("ace",),
                            ace=AceOptions(engine="kika", temperatures=(293.6,), njoyExe=None))


def _table(record):
    return read_ace(record["ace"], strict=True)


def test_each_sample_is_a_table_and_no_njoy_ran(run):
    for sample in run.samples:
        (record,) = sample["ace"]
        assert record["returncode"] == 0, record["listing"]
        assert Path(record["ace"]).name == "26056.02c"
        assert Path(record["xsdir"]).read_text().split()[0] == "26056.02c"
        assert not (Path(record["ace"]).parent.parent / "njoy").exists()


def test_two_samples_are_two_tables_of_one_layout(run):
    a, b = (_table(s["ace"][0]) for s in run.samples)
    assert list(a.header.nxs_array) == list(b.header.nxs_array)
    ea, eb = np.asarray(a.esz_block.elastic_xs), np.asarray(b.esz_block.elastic_xs)
    assert np.all(np.isfinite(ea)) and np.all(np.isfinite(eb))
    assert not np.array_equal(ea, eb)


def test_the_unperturbed_processing_is_write_aces_table(tmp_path):
    """processSuite + encodeAce from scratch, as a sample is made, against EndfProcessing.write_ace."""
    from kika.ace.model_adapter import encodeAce
    from kika.endf import ProcessingPlan, process_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.hdf5.processing import ACE_PARTICLES, processSuite

    (written,) = process_endf(TAPE, ProcessingPlan(temperatures=(293.6,), workers=1)).write_ace(tmp_path)
    suite, _ = decodeReactionSuite(read_endf(TAPE))
    ace, _ = encodeAce(processSuite(suite, temperatures=[293.6], endf=TAPE, workers=1,
                                    particles=ACE_PARTICLES), 293.6)
    reference = read_ace(str(written), strict=True)
    assert list(ace.header.nxs_array) == list(reference.header.nxs_array)
    assert list(ace.header.jxs_array) == list(reference.header.jxs_array)
    np.testing.assert_allclose(np.asarray(ace.xss_data)[1:], np.asarray(reference.xss_data)[1:],
                               rtol=5e-12, atol=0)


def test_the_kika_engine_needs_no_njoy_and_checks_its_suffix():
    AceOptions(engine="kika", temperatures=(293.6, 600.0), njoyExe=None).validate()
    with pytest.raises(ValueError, match="extension"):
        AceOptions(engine="kika", temperatures=(350.0,), njoyExe=None).validate()
    AceOptions(engine="kika", temperatures=(350.0,), extensions=("12c",), njoyExe=None).validate()
    with pytest.raises(ValueError, match="engine"):
        AceOptions(engine="frendy", temperatures=(293.6,)).validate()

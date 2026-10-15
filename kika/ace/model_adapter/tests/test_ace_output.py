"""Processing → ACE: ``EndfProcessing.write_ace``, ``kika.write(format="ace")`` and the xsdir line.

A table of kika's processing is one temperature of the processed suite the
OpenMC writer takes, written by ``encodeAce`` with ACER's conventions and read
back in strict mode before it replaces a file.
"""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import kika
from kika.ace import read_ace
from kika.endf import PROCESSING_REJECTION_CATEGORIES, ProcessingOutputError, ProcessingPlan, process_endf

TAPE = Path(__file__).resolve().parents[3] / "endf/tests/data/micro_fe56_structural.endf"
TEMPERATURES = (293.6, 900.0)


@pytest.fixture(scope="module")
def run():
    return process_endf(TAPE, ProcessingPlan(temperatures=TEMPERATURES, workers=1))


@pytest.fixture(scope="module")
def written(run, tmp_path_factory):
    directory = tmp_path_factory.mktemp("ace")
    return directory, run.write_ace(directory)


def test_one_table_per_temperature_named_by_zaid_and_suffix(written):
    directory, paths = written
    assert [p.name for p in paths] == ["26056.02c", "26056.09c"]
    for path, T in zip(paths, TEMPERATURES):
        ace = read_ace(str(path), strict=True)
        assert ace.header.zaid == 26056
        assert ace.header.temperature == pytest.approx(T * 8.617333262e-11, rel=1e-6)
        assert ace.header.matid == 2631
        assert [int(m) for m in ace.reaction_mt_data.incident_neutron] == [102, 444]


def test_the_xsdir_names_each_table_once(written, run):
    directory, paths = written
    lines = (directory / "xsdir").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["26056.02c", "26056.09c"]
    assert lines[0].split()[2] == "26056.02c"
    with pytest.raises(ProcessingOutputError) as caught:
        run.write_ace(directory, temperatures=900.0)
    assert caught.value.category == "ace-table-exists"
    run.write_ace(directory, temperatures=900.0, replace=True)
    assert len((directory / "xsdir").read_text().splitlines()) == 2


def test_the_write_stage_reports_the_tables(written, run):
    write = [s for s in run.stages if s.stage == "write" and s.status == "done"][0]
    assert write.details["format"] == "ace"
    assert [Path(p).name for p in write.details["files"]] == ["26056.02c", "26056.09c"]


def test_a_temperature_not_broadened_is_refused(run, tmp_path):
    with pytest.raises(ProcessingOutputError) as caught:
        run.write_ace(tmp_path, temperatures=600.0)
    assert caught.value.category == "ace-temperature-not-processed"


def test_an_extension_names_one_table(run, tmp_path):
    with pytest.raises(ProcessingOutputError):
        run.write_ace(tmp_path, extension=".80c")
    (path,) = run.write_ace(tmp_path, temperatures=293.6, extension=".80c")
    assert path.name == "26056.80c"


def test_ace_needs_a_broadened_temperature(tmp_path):
    result = process_endf(TAPE, ProcessingPlan(temperatures=()))
    with pytest.raises(ProcessingOutputError) as caught:
        result.write_ace(tmp_path)
    assert caught.value.category == "ace-needs-temperature"
    assert set(PROCESSING_REJECTION_CATEGORIES) >= {"ace-needs-temperature", "ace-temperature-not-processed",
                                                   "ace-table-exists", "ace-failed"}


def test_the_door_writes_a_read_table_back_as_it_was(written, tmp_path):
    """ACE → model → ACE through kika.read/kika.write: the same table, header date aside."""
    _, paths = written
    suite = kika.read(str(paths[0]))
    kika.write(suite, tmp_path / "again.02c", format="ace")
    first = paths[0].read_text().splitlines()
    again = (tmp_path / "again.02c").read_text().splitlines()
    assert first[1:] == again[1:]
    assert first[0][:40] == again[0][:40]


def test_the_door_refuses_an_evaluation(tmp_path):
    from kika.ace.model_adapter import AceEncodeError

    suite = kika.read(str(TAPE))
    with pytest.raises(AceEncodeError, match="ace-needs-processed-suite"):
        kika.write(suite, tmp_path / "x.02c", format="ace")


def test_add_xsdir_entry_keeps_the_rest_and_refuses_a_repeat(tmp_path):
    from kika.ace.xsdir import XsdirEntryExists, add_xsdir_entry

    path = tmp_path / "xsdir"
    path.write_text("datapath=/data\natomic weight ratios\n 1001 0.999167\ndirectory\n"
                    "1001.80c 0.999167 h1 0 1 1 100 0 0 2.530E-08 +\n ptable\n")
    add_xsdir_entry(path, "8016.02c 15.857510 8016.02c 0 1 1 9 0 0 2.530E-08")
    with pytest.raises(XsdirEntryExists):
        add_xsdir_entry(path, "8016.02c 15.857510 other 0 1 1 9 0 0 2.530E-08")
    add_xsdir_entry(path, "8016.02c 15.857510 other 0 1 1 9 0 0 2.530E-08", replace=True)
    assert path.read_text().splitlines() == [
        "datapath=/data", "atomic weight ratios", " 1001 0.999167", "directory",
        "1001.80c 0.999167 h1 0 1 1 100 0 0 2.530E-08 +", " ptable",
        "8016.02c 15.857510 other 0 1 1 9 0 0 2.530E-08"]


@pytest.mark.parametrize("interpolation", ["linlin", "flat"])
def test_a_photon_cosine_table_is_cut_into_32_equiprobable_bins(interpolation):
    """ANDP holds isotropic or equiprobable bins only; each bin carries 1/32 of the table."""
    from kika.ace.model_adapter.encode import _equiprobableEdges
    from kika.nuclear_data.model import Interpolation

    mu = np.linspace(-1.0, 1.0, 9)
    pdf = 0.5 + 0.4 * mu
    flat = interpolation == "flat"
    table = NS(xs=mu, pdf=pdf, interpolation=Interpolation.flat if flat else Interpolation.linlin)
    edges = _equiprobableEdges(table)
    assert edges.size == 33 and edges[0] == -1.0 and edges[-1] == 1.0

    def cdf(x):
        if flat:
            i = np.clip(np.searchsorted(mu, x, side="right") - 1, 0, 7)
            mass = np.concatenate([[0.0], np.cumsum(pdf[:-1] * np.diff(mu))])
            return (mass[i] + pdf[i] * (x - mu[i])) / mass[-1]
        return (0.5 * (x + 1.0) + 0.2 * (x * x - 1.0))     # exact for 0.5 + 0.4 mu

    np.testing.assert_allclose(np.diff([cdf(x) for x in edges]), 1.0 / 32, rtol=0, atol=1e-12)


def test_an_isomer_takes_lanls_zaid():
    """Am-242m1: LISO 1 → 95642 (A + 300 + 100 m), as MCNP and OpenMC read it."""
    from kika.hdf5.processing import metastableIndex
    from kika.nuclear_data.model import EndfProvenance
    from kika.nuclear_data.model.pops import MetaStable, PoPs

    endf = NS(provenance=EndfProvenance(headerFields={"liso": 1}), PoPs=PoPs(), target="Am242")
    assert metastableIndex(endf) == 1
    pops = PoPs()
    pops.aliases["Am242_m1"] = MetaStable(id="Am242_m1", pid="Am242_e2", metaStableIndex=1)
    assert metastableIndex(NS(provenance=None, PoPs=pops, target="Am242_e2")) == 1
    assert metastableIndex(NS(provenance=None, PoPs=PoPs(), target="Am242")) == 0


def test_a_photon_keeps_the_file_it_was_stated_in():
    """MF6 photons are MFTYPE 16, MF13 13 (a production cross section), MF12 12."""
    from kika.hdf5.processing import _photonFile

    assert _photonFile(NS(provenance=None), {"label": "photon"}) == 16
    assert _photonFile(NS(provenance=NS(headerFields={"mf13": {}})), None) == 13
    assert _photonFile(NS(provenance=NS(headerFields={"mf12": {}})), None) == 12
    assert _photonFile(NS(provenance=None), None) == 16

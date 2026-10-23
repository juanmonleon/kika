"""``kika.process``: one door for every evaluation kika processes, and the ``kika`` command."""
from pathlib import Path

import pytest

import kika
from kika._process import NSUB_ROUTES, ProcessingRouteError, endf_header

DATA = Path(__file__).resolve().parents[1] / "endf/tests/data"
NEUTRON = DATA / "micro_fe56_structural.endf"
PHOTOATOMIC = DATA / "b81_h_photoatomic.endf"


def test_the_header_says_what_the_tape_is():
    head = endf_header(NEUTRON)
    assert head["nsub"] == 10 and head["materials"] == 1 and head["mat"] == 2631
    assert NSUB_ROUTES[head["nsub"]] == "neutron"
    assert endf_header(PHOTOATOMIC)["nsub"] == 3
    assert endf_header(DATA / "b81_h_relaxation.endf")["nsub"] == 6
    assert endf_header(DATA / "b81_h_electrons.endf")["nsub"] == 113


def test_what_is_not_an_evaluation_to_process_is_refused_with_a_reason(tmp_path):
    table = tmp_path / "rates.csv"
    table.write_text("energy,rate\n1,2\n3,4\n")
    with pytest.raises(ProcessingRouteError, match="not an evaluation to process"):
        kika.process(table)
    with pytest.raises(ProcessingRouteError, match="atomic relaxation"):
        kika.process(DATA / "b81_h_relaxation.endf")
    with pytest.raises(ProcessingRouteError, match="kind must be"):
        kika.process(NEUTRON, kind="decay")
    with pytest.raises(FileNotFoundError):
        kika.process(tmp_path / "missing.endf")


def test_options_of_another_route_are_refused_not_ignored():
    with pytest.raises(ProcessingRouteError, match="heating= is for neutron"):
        kika.process(PHOTOATOMIC, heating=True)
    with pytest.raises(ProcessingRouteError, match="name= is for thermal"):
        kika.process(PHOTOATOMIC, name="c_H_in_H2O")


@pytest.fixture(scope="module")
def neutron(tmp_path_factory):
    out = tmp_path_factory.mktemp("process")
    run = kika.process(NEUTRON, temperatures=[293.6], workers=2)
    written = run.write(out, ["pendf", "hdf5"])
    return run, out, written


def test_a_neutron_tape_goes_through_process_endf(neutron):
    run, out, written = neutron
    assert run.kind == "neutron" and "ace" in run.formats
    assert run.temperatures == (293.6,)                       # reached through the processing result
    assert [p.name for p in written] == ["Fe56.pendf", "Fe56.h5"]
    assert (out / "cross_sections.xml").is_file()             # an HDF5 library is indexed
    with pytest.raises(ValueError, match="written as"):
        run.write(out, "xyz")


def test_the_neutron_report_lists_what_was_written(neutron):
    run, out, _ = neutron
    d = run.report_document().to_dict()
    assert [o["format"] for o in d["outputs"]] == ["pendf", "hdf5"]
    assert all(o["written"] for o in d["outputs"]) and not d["summary"]["incomplete"]
    assert "Files written and read back: PENDF, OpenMC HDF5." in d["summary"]["detail"]


def test_a_photoatomic_tape_finds_its_relaxation_and_electron_tapes(tmp_path):
    run = kika.process(PHOTOATOMIC)
    assert run.kind == "photoatomic" and run.formats == ("ace",)
    assert run.relaxation.name == "b81_h_relaxation.endf" and run.electrons.name == "b81_h_electrons.endf"
    (table,) = run.write(tmp_path)
    assert table.name.startswith("1000.") and (tmp_path / "xsdir").is_file()
    d = run.report_document().to_dict()
    assert d["kind"] == "photoatomic" and d["reconstruction"] is None and d["summary"]["status"] in ("passed", "note")
    html = run.report_document().to_html()
    assert html.startswith("<!doctype html>") and "Photo-atomic" in html
    text = run.report_document().to_text()
    assert text.isascii() and "CHOICES MADE FOR YOU" in text


def test_a_photoatomic_tape_without_its_siblings_says_what_is_missing(tmp_path):
    alone = tmp_path / "lib" / "photoat"
    alone.mkdir(parents=True)
    copy = alone / PHOTOATOMIC.name
    copy.write_bytes(PHOTOATOMIC.read_bytes())
    with pytest.raises(ProcessingRouteError, match="relaxation="):
        kika.process(copy)


def test_the_command_processes_and_reports(tmp_path, capsys):
    from kika._cli import main
    status = main(["process", str(PHOTOATOMIC), "--to", "ace", "hdf5", "-o", str(tmp_path)])
    assert status == 0
    line = capsys.readouterr().out
    assert "b81_h_photoatomic.endf" in line and "not written as hdf5" in line
    assert (tmp_path / "b81_h_photoatomic.report.html").is_file()
    # A folder: what is not an evaluation to process is skipped, not a failure.
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "notes.txt").write_text("nothing to process here\n")
    assert main(["process", str(folder), "-o", str(tmp_path / "out")]) == 0
    assert "skip" in capsys.readouterr().out

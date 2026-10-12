"""The staged ENDF processing entry the app calls (kika.endf.process_endf)."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from kika.endf import (PROCESSING_REJECTION_CATEGORIES, EndfProcessing, ProcessingPlan,
                       inspect_endf_processing, process_endf)
from kika.endf import process as staged

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"
TEMPERATURES = (293.6, 900.0)


@pytest.fixture(scope="module")
def run():
    events = []
    result = process_endf(TAPE, ProcessingPlan(temperatures=TEMPERATURES, workers=2),
                          progress=lambda stage, fraction: events.append((stage, fraction)))
    return result, events


def test_every_stage_is_reported_in_order(run):
    result, _ = run
    assert [s.stage for s in result.stages] == ["read", "decode", "reconstruct", "broaden", "urr", "heating"]
    status = {s.stage: s.status for s in result.stages}
    assert status == dict(read="done", decode="done", reconstruct="done", broaden="done",
                          urr="skipped", heating="skipped")
    assert result.stage("urr").category == "urr-absent"
    assert result.stage("heating").category is None          # not requested
    broaden = result.stage("broaden")
    assert broaden.details["temperatures"] == list(TEMPERATURES)
    assert broaden.details["thnmax_eV"] == result.thnmax == pytest.approx(850000.0)
    assert result.stage("reconstruct").details["n_resonances"] == 320


def test_progress_names_each_temperature(run):
    _, events = run
    stages = {s for s, _ in events}
    assert {"read", "decode", "reconstruct", "broaden:293.6", "broaden:900"} <= stages
    for T in ("293.6", "900"):
        fractions = [f for s, f in events if s == f"broaden:{T}"]
        assert fractions[0] is None and fractions[-1] == pytest.approx(1.0)
        assert all(0 < f <= 1 for f in fractions[1:])


def test_heated_tables_keep_the_reactions_and_the_sum_rule(run):
    result, _ = run
    cold = result.cross_sections()
    assert result.temperatures == TEMPERATURES
    for T in TEMPERATURES:
        hot = result.cross_sections(T)
        assert set(hot) == set(cold) == {1, 2, 102}
        for x, y in hot.values():
            assert x.dtype == y.dtype == np.float64 and x.flags.c_contiguous and y.flags.c_contiguous
            assert np.all(np.diff(x) > 0)
        x1, y1 = hot[1]
        parts = sum(np.interp(x1, *hot[mt], left=0.0) for mt in (2, 102))
        assert np.allclose(y1, parts, rtol=1e-12, atol=0.0)


def test_broadening_stops_at_thnmax(run):
    result, _ = run
    x0, y0 = result.cross_sections()[2]
    x, y = result.cross_sections(900)[2]
    above = x > result.thnmax
    assert above.any()
    # Off the steps, the 0 K values are taken unchanged above thnmax.
    steps = x0[:-1][np.diff(x0) == 0]
    clear = above & ~np.isin(x, steps)
    assert np.allclose(y[clear], np.interp(x[clear], x0, y0), rtol=2e-3)
    # Below it the resonances are broadened: the peak comes down.
    window = (x > 1.0e3) & (x < result.thnmax)
    window0 = (x0 > 1.0e3) & (x0 < result.thnmax)
    assert y[window].max() < y0[window0].max()


def test_unknown_temperature_is_a_key_error(run):
    result, _ = run
    with pytest.raises(KeyError):
        result.cross_sections(600)


def test_to_dict_is_plain_and_picklable(run):
    result, _ = run
    d = result.to_dict()
    assert d["temperatures"] == [0.0, *TEMPERATURES]
    table = d["cross_sections"][900.0][1]
    assert isinstance(table["energies"], np.ndarray) and table["energies"].dtype == np.float64
    assert d["probability_tables"] is None
    pickle.loads(pickle.dumps(d))
    pickle.loads(pickle.dumps(result.stages))


def test_report_has_a_stages_section(run):
    result, _ = run
    report = result.report_document()
    assert "PROCESSING STAGES" in report.to_text()
    assert "Processing stages" in report.to_html()
    assert [s["stage"] for s in report.to_dict()["processing_stages"]][:3] == ["read", "decode", "reconstruct"]


def test_a_doppler_rejection_keeps_the_0K_result(monkeypatch):
    from kika.hdf5 import processing as hp
    from kika.processing.doppler import UnsupportedDopplerError

    def refuse(*args, **kwargs):
        raise UnsupportedDopplerError("bound-target", "refused for the test")
    monkeypatch.setattr(hp, "_broadenAll", refuse)
    result = process_endf(TAPE, ProcessingPlan(temperatures=(600.0,)))
    broaden = result.stage("broaden")
    assert broaden.status == "rejected" and broaden.category == "doppler-unsupported-target"
    assert broaden.details["engine_category"] == "bound-target"
    assert result.temperatures == () and {1, 2, 102} <= set(result.cross_sections())


def test_zero_kelvin_only():
    result = process_endf(TAPE, ProcessingPlan(temperatures=()))
    assert result.stage("broaden").status == "skipped" and result.temperatures == ()
    assert isinstance(result, EndfProcessing)


def test_plan_rejects_non_positive_temperatures():
    with pytest.raises(ValueError):
        ProcessingPlan(temperatures=(0.0,))
    assert ProcessingPlan(temperatures=(900, 293.6, 900)).temperatures == (293.6, 900.0)


def test_inspection_reads_mf1_mf2_only():
    info = inspect_endf_processing(TAPE)
    assert info["resolved"]["formalisms"] == ["Reich-Moore"] and info["resolved"]["resonances"] == 320
    assert info["unresolved"] is None and {1, 2, 102} <= set(info["reactions"])
    assert info["stages"]["reconstruct"]["predicted_rejection"] is None
    assert info["stages"]["urr"] == dict(applicable=False, predicted_rejection="urr-absent",
                                         reason=PROCESSING_REJECTION_CATEGORIES["urr-absent"])


def test_every_category_has_a_sentence():
    assert set(staged._DOPPLER_GROUPS.values()) <= set(PROCESSING_REJECTION_CATEGORIES)
    assert all(isinstance(v, str) and v.endswith(".") for v in PROCESSING_REJECTION_CATEGORIES.values())


def test_urr_tables_on_a_whole_tape(cf252_b81_tape):
    result = process_endf(cf252_b81_tape, ProcessingPlan(temperatures=(293.6,), urr_ladders=16))
    assert result.stage("urr").status == "done", result.stage("urr").summary
    table = result.probability_tables[293.6]
    ne = len(table["energies"])
    assert ne == result.stage("urr").details["n_energies"]
    for row in ("cdf", "total", "elastic", "fission", "capture"):
        assert table[row].shape == (ne, 20) and table[row].dtype == np.float64
    assert np.allclose(table["cdf"][:, -1], 1.0) and np.all(np.diff(table["cdf"], axis=1) >= 0)
    assert table["factors"] is True


def test_a_daemonic_caller_runs_in_process(monkeypatch):
    import multiprocessing
    from kika.hdf5 import processing as hp

    class Daemon:
        daemon = True
    seen = {}
    real = hp._broadenAll

    def spy(suite, temperatures, thnmax, options, workers, progress=None):
        seen["workers"] = workers
        return real(suite, temperatures, thnmax, options, workers, progress=progress)
    monkeypatch.setattr(multiprocessing, "current_process", lambda: Daemon())
    monkeypatch.setattr(hp, "_broadenAll", spy)
    result = process_endf(TAPE, ProcessingPlan(temperatures=(600.0,)))
    assert seen["workers"] == 1 and result.stage("broaden").status == "done"
    assert result.plan.workers == 1


def test_urr_tables_without_broadening(cf252_b81_tape):
    result = process_endf(cf252_b81_tape, ProcessingPlan(temperatures=(293.6,), broaden=False,
                                                         urr_ladders=16))
    broaden = result.stage("broaden")
    assert broaden.status == "skipped" and broaden.category is None
    assert result.temperatures == () and result.stage("urr").status == "done"
    assert set(result.probability_tables) == {293.6}
    with pytest.raises(KeyError):
        result.cross_sections(293.6)


def test_outputs_append_one_write_stage_each(run, tmp_path):
    result, _ = run
    before = len(result.stages)
    result.write_pendf(tmp_path / "fe56.pendf")
    out = result.write_hdf5(tmp_path / "fe56.h5")
    assert out.exists() and not (tmp_path / "cross_sections.xml").exists()
    writes = result.stages[before:]
    assert [(s.stage, s.status, s.details["format"]) for s in writes] == [
        ("write", "done", "pendf"), ("write", "done", "hdf5")]
    assert writes[1].details["temperatures"] == list(TEMPERATURES)
    import kika
    suite = kika.read(out, format="hdf5")
    assert {"heated_294K", "heated_900K"} <= set(suite.styles.labels)


def test_hdf5_needs_a_broadened_temperature(tmp_path):
    from kika.endf import ProcessingOutputError
    result = process_endf(TAPE, ProcessingPlan(temperatures=()))
    with pytest.raises(ProcessingOutputError) as caught:
        result.write_hdf5(tmp_path / "fe56.h5")
    assert caught.value.category == "hdf5-needs-temperature"
    assert caught.value.category in PROCESSING_REJECTION_CATEGORIES
    last = result.stages[-1]
    assert (last.stage, last.status, last.category) == ("write", "rejected", "hdf5-needs-temperature")


def test_hdf5_equals_the_hdf5_writer_from_scratch(cf252_b81_tape, tmp_path):
    """Reusing the broadening and URR tables changes nothing in the file."""
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.hdf5 import records as R
    from kika.hdf5.writer import writeSuite
    result = process_endf(cf252_b81_tape, ProcessingPlan(temperatures=(293.6,)))
    result.write_hdf5(tmp_path / "ours.h5")
    suite, _ = decodeReactionSuite(read_endf(str(cf252_b81_tape)))
    writeSuite(suite, tmp_path, temperatures=[293.6], fileName="ref.h5", index=False, endf=cf252_b81_tape)
    assert R.compareTrees(R.readFile(tmp_path / "ref.h5"), R.readFile(tmp_path / "ours.h5"), limit=5) == []


def _materials(path):
    """``[(TEMP, LRP, {MF2 MTs})]`` per material of a tape, by its records."""
    from kika._records import parse_number
    out = []
    for line in Path(path).read_text().split("\n"):
        # MF1/451: LRP on the first record, TEMP on the fourth.
        if line[70:75] == " 1451" and line[75:80].strip() == "1":
            out.append([None, int(line[22:33]), set()])
        elif line[70:75] == " 1451" and line[75:80].strip() == "4" and out:
            out[-1][0] = parse_number(line[0:11])
        elif line[70:72] == " 2" and line[72:75].strip() not in ("", "0") and out:
            out[-1][2].add(int(line[72:75]))
    return [tuple(m) for m in out]


def test_pendf_has_one_material_per_temperature(run, tmp_path):
    from kika.endf import process as staged_module
    from kika.endf.read_endf import read_endf
    result, _ = run
    assert staged_module.PENDF_TEMPERATURES_SUPPORTED is True
    path = result.write_pendf(tmp_path / "fe56.pendf")
    materials = _materials(path)
    assert [round(m[0], 3) for m in materials] == [0.0, *TEMPERATURES]
    assert all(lrp == 2 for _, lrp, _ in materials)
    # The first material reads back as the 0 K tape kika always wrote.
    tape = read_endf(str(path), mf_numbers=[1, 3])
    assert set(tape.mf[3].mt) >= {1, 2, 102}
    write = result.stages[-1]
    assert write.details["format"] == "pendf" and write.details["temperatures"] == [0.0, *TEMPERATURES]


def test_pendf_refuses_an_unprocessed_temperature(run, tmp_path):
    from kika.endf import ProcessingOutputError
    result, _ = run
    with pytest.raises(ProcessingOutputError) as caught:
        result.write_pendf(tmp_path / "x.pendf", temperatures=[600.0])
    assert caught.value.category == "pendf-temperature-not-processed"
    assert not (tmp_path / "x.pendf").exists()


def test_pendf_carries_mt153_at_each_temperature(cf252_b81_tape, tmp_path):
    result = process_endf(cf252_b81_tape, ProcessingPlan(temperatures=(293.6, 600.0), urr_ladders=16))
    path = result.write_pendf(tmp_path / "cf252.pendf", temperatures=[293.6, 600.0])
    materials = _materials(path)
    assert [round(m[0], 3) for m in materials] == [293.6, 600.0]
    assert all(153 in mf2 and 151 in mf2 for _, _, mf2 in materials)
    assert result.stages[-1].details["mt153"] == [293.6, 600.0]


def test_gnds_has_a_heated_style_per_temperature(run, tmp_path):
    import kika
    result, _ = run
    path = result.write_gnds(tmp_path / "fe56.gnds.xml")
    back = kika.read(path, format="gnds", covariances=False)
    styles = {s.label: s for s in back.styles}
    assert {"eval", "recon", "heated_000", "heated_001"} <= set(styles)
    assert [float(styles[l].temperature.value) for l in ("heated_000", "heated_001")] == list(TEMPERATURES)
    assert styles["heated_001"].derivedFrom == "recon"
    elastic = next(r for r in back.reactions if r.ENDF_MT == 2)
    x, y = result.cross_sections(900)[2]
    from kika.hdf5.processing import _flatten
    xb, yb = _flatten(elastic.crossSection["heated_001"])
    # Off the steps (a nudged pair in memory, a repeated energy in the file).
    step = np.flatnonzero(np.diff(x) <= 2e-12 * x[1:])
    keep = np.ones(len(x), bool)
    keep[step] = keep[step + 1] = False
    assert np.allclose(np.interp(x[keep], xb, yb), y[keep], rtol=1e-12, atol=0)
    write = result.stages[-1]
    assert write.details["format"] == "gnds" and write.details["styles"] == ["recon", "heated_000", "heated_001"]


#: FUDGE's H-2 (ENDF/B-VIII.1), committed whole: no resonance parameters, a
#: production under MT102. The committed resonance fixtures are trimmed and do
#: not span their evaluated domain, so none of them can be reconstructed.
H2_GNDS = Path(__file__).resolve().parents[2] / "gnds/tests/data/n-001_H_002.endf.gnds.xml"


@pytest.fixture(scope="module")
def gnds_run():
    import kika.gnds
    return kika.gnds.process_gnds(H2_GNDS, ProcessingPlan(temperatures=(600.0,), workers=2))


def test_process_gnds_runs_every_stage(gnds_run):
    from kika.endf import EndfProcessing
    assert isinstance(gnds_run, EndfProcessing) and gnds_run.source_format == "gnds"
    status = {s.stage: s.status for s in gnds_run.stages}
    assert status["reconstruct"] == "done" and status["broaden"] == "done"
    assert gnds_run.stage("decode").details["source_format"] == "gnds"
    assert gnds_run.reconstruction.evaluation["mat"] == 128
    assert {1, 2, 102} <= set(gnds_run.cross_sections()) and gnds_run.temperatures == (600.0,)


def test_process_gnds_outputs(gnds_run, tmp_path):
    import kika
    from kika.endf import ProcessingOutputError
    with pytest.raises(ProcessingOutputError) as caught:
        gnds_run.write_pendf(tmp_path / "h2.pendf")
    assert caught.value.category == "pendf-needs-endf-source"
    back = kika.read(gnds_run.write_gnds(tmp_path / "h2.gnds.xml"), format="gnds", covariances=False)
    assert {"eval", "recon", "heated_000"} <= {s.label for s in back.styles}
    assert gnds_run.write_hdf5(tmp_path / "h2.h5").exists()
    formats = [(s.details["format"], s.status) for s in gnds_run.stages if s.stage == "write"]
    assert formats == [("pendf", "rejected"), ("gnds", "done"), ("hdf5", "done")]


def test_process_gnds_sets_aside_the_files_reconstruction(run, tmp_path):
    """kika's own processed GNDS carries recon and heated styles; they are set
    aside. Its PoPs states no target spin (the ENDF decoder does not move MF2's
    SPI there), so the reconstruction is refused by name, not by a ValueError."""
    from kika.endf import process_gnds
    from kika.processing.resonances import UnsupportedResonanceError
    result, _ = run
    source = result.write_gnds(tmp_path / "fe56.gnds.xml")
    with pytest.raises(UnsupportedResonanceError) as caught:
        process_gnds(source, ProcessingPlan(temperatures=()))
    assert caught.value.category == "context-missing"


def test_gnds_and_endf_routes_agree_once_broadened(tmp_path):
    """The same evaluation through both doors: Au-197 B-VIII.1, ENDF and GNDS."""
    from conftest import resolve_library  # noqa: F401  (library roots, as the other library tests)
    import os
    root = Path(os.environ.get("KIKA_LIB_TAPES", "/share_snc/lib/endf"))
    endf = root / "endfb81/n-079_Au_197.endf"
    gnds = root.parent / "gnds/endfb81/n-079_Au_197.endf.gnds.xml"
    if not (endf.exists() and gnds.exists()):
        pytest.skip("Au-197 B-VIII.1 in ENDF and GNDS not on this machine")
    from kika.endf import process_gnds
    plan = ProcessingPlan(temperatures=(293.6,), urr=False)
    a, b = process_endf(endf, plan), process_gnds(gnds, plan)
    assert b.reconstruction.evaluation["mat"] == 7925
    assert b.stage("decode").details["source_reconstruction"] == "recon in the file, ignored"
    for mt in (1, 2, 102):
        xa, ya = a.cross_sections(293.6)[mt]
        xb, yb = b.cross_sections(293.6)[mt]
        rel = np.abs(np.interp(xa, xb, yb) - ya) / np.maximum(np.abs(ya), 1e-8)
        assert np.quantile(rel, 0.999) < 1e-3, mt

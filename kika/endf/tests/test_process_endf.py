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

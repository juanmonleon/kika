"""The end-to-end ENDF entry the app and generated scripts call."""
import re
from pathlib import Path

import numpy as np
import pytest

import kika.processing as processing
from kika.endf import reconstruct_endf
from kika.processing import (ENGINE_VERSION, REJECTION_CATEGORIES, ReconstructionConvergenceError,
                             UnsupportedResonanceError, native_available)

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"


@pytest.fixture(scope="module")
def reconstruction():
    events = []
    rec = reconstruct_endf(TAPE, progress=lambda stage, fraction: events.append((stage, fraction)))
    return rec, events


def test_reconstructs_every_reaction_with_its_q_values(reconstruction):
    rec, _ = reconstruction
    tables = rec.tables_by_mt()
    assert {1, 2, 102} <= set(tables)
    assert rec.engine_version == ENGINE_VERSION and rec.report["engine"] == ENGINE_VERSION
    assert rec.context is not None and rec.context_source.startswith("MF2 AWRI")
    for mt, table in tables.items():
        assert table.energies.shape == table.values.shape and np.all(np.diff(table.energies) >= 0)
        assert table.qm is not None and table.qi is not None and table.lr is not None
    assert tables[102].qm > 0


def test_a_repeated_energy_is_always_a_real_jump(reconstruction):
    rec, _ = reconstruction
    for table in rec.tables_by_mt().values():
        x, y = table.energies, table.values
        i = np.flatnonzero(np.diff(x) == 0)
        assert np.all(np.abs(y[i + 1] - y[i]) > 1e-9 * np.maximum(np.abs(y[i]), np.abs(y[i + 1])))


def test_tables_are_the_reconstructed_forms(reconstruction):
    from kika.algebra import prepare_evaluator
    rec, _ = reconstruction
    table = rec.tables_by_mt()[2]
    mid = 0.5 * (table.energies[1:] + table.energies[:-1])
    mid = mid[np.diff(table.energies) > 0]
    stored = rec.result.evaluate(mid)[rec.result._mt_keys[2]]
    assert np.allclose(prepare_evaluator(table.energies, table.values, 2)(mid), stored, rtol=1e-3, atol=1e-8)


def test_progress_reports_each_stage_in_order(reconstruction):
    _, events = reconstruction
    stages = [s for s, _ in events]
    assert stages[:4] == ["read", "read", "decode", "decode"] and stages[4] == "reconstruct"
    fractions = [f for s, f in events if s == "reconstruct" and f is not None]
    assert fractions and np.all(np.diff(fractions) >= 0) and fractions[-1] == pytest.approx(1.0)
    assert all(type(f) is float for _, f in events if f is not None)


def test_a_failed_write_is_a_typed_publication_failure(reconstruction, monkeypatch, tmp_path):
    import kika.endf.writers.assemble as assemble
    rec, _ = reconstruction

    def broken(*_, **__):
        raise ValueError("reconstructed ENDF conversion is incomplete")
    monkeypatch.setattr(assemble, "writeReconstructedEndfTape", broken)
    with pytest.raises(ReconstructionConvergenceError) as failure:
        rec.write_pendf(tmp_path / "x.pendf")
    assert failure.value.category == "publication-failed"
    assert not (tmp_path / "x.pendf").exists()


@pytest.mark.slow
def test_pendf_is_written_reloaded_and_not_reconstructed_twice(tmp_path):
    rec = reconstruct_endf(TAPE)
    stages = []
    path = tmp_path / "fe56.pendf"
    assert rec.write_pendf(path, progress=lambda s, f: stages.append((s, f))).isClean
    assert stages[0] == ("write", None) and stages[-1] == ("write", 1.0) and path.stat().st_size > 0
    fractions = [f for s, f in stages[1:]]
    assert {s for s, _ in stages} == {"write"} and all(type(f) is float for f in fractions)
    assert len(fractions) > 5 and np.all(np.diff(fractions) >= 0)
    # The report gains the reload check once the PENDF exists.
    report = rec.report_document().to_dict()
    publication = report["verification"]["publication"]
    assert publication["file"] == "fe56.pendf" and publication["mat"] == 2631
    rows = {row["mt"]: row for row in publication["rows"]}
    assert {1, 2, 102} <= set(rows) and all(0 <= row["max_ratio"] <= 1 for row in rows.values())
    assert all(row["at_eV"] is not None for row in rows.values() if row["max_ratio"] > 0)
    assert "write_verify" in report["timings_s"]
    with pytest.raises(UnsupportedResonanceError) as failure:
        reconstruct_endf(path)
    assert failure.value.category == "already-reconstructed"


def test_every_category_the_engine_raises_is_declared():
    root = Path(processing.__file__).resolve().parents[1]
    used = set()
    for path in (list((root / "processing/resonances").glob("*.py"))
                 + [root / "endf/writers/assemble.py", root / "endf/reconstruction.py"]):
        used |= set(re.findall(r"category=['\"]([a-z0-9-]+)['\"]", path.read_text(encoding="utf-8")))
    for cls in (UnsupportedResonanceError, ReconstructionConvergenceError):
        used.add(cls("x").category)
    assert used <= set(REJECTION_CATEGORIES), used - set(REJECTION_CATEGORIES)
    assert len(set(REJECTION_CATEGORIES)) == len(REJECTION_CATEGORIES)


def test_native_kernel_flag_is_a_bool():
    assert isinstance(native_available(), bool)


def test_report_document_without_a_pendf(reconstruction, tmp_path):
    import json
    rec, _ = reconstruction
    report = rec.report_document()
    d = report.to_dict()
    assert d["schema"] == "kika-reconstruction-report/1"
    assert json.loads(json.dumps(d)) == d
    assert d["material"]["mat"] == 2631 and d["material"]["context_source"] == rec.context_source
    assert d["evaluation"]["sha256"] and len(d["evaluation"]["sha256"]) == 64
    assert d["software"]["engine_version"] == ENGINE_VERSION
    assert d["software"]["native_kernel"] is native_available()
    assert d["options"]["rtol"] == rec.result.options.rtol
    (rrr,) = d["resonance_ranges"]
    assert (rrr["lru"], rrr["lrf"], rrr["formalism"]) == (1, 3, "Reich-Moore") and rrr["resonances"] > 0
    rows = {r["mt"]: r for r in d["reactions"]}
    tables = rec.tables_by_mt()
    assert set(rows) == set(tables)
    for mt, row in rows.items():
        assert row["points"] == len(tables[mt].energies)
        assert row["jumps"] == int(np.count_nonzero(np.diff(tables[mt].energies) == 0))
    assert rows[1]["derived_sum"] and not rows[2]["derived_sum"]
    assert d["sums"]["policy"] == "derive-from-leaves"
    assert d["sums"]["graph"] == [dict(mt=1, name="MT1", parts=["MT2", "MT102"], parts_text="MT2 + MT102")]
    assert {x["mt"] for x in d["sums"]["source_discrepancies"]} == {1}
    assert d["verification"]["publication"] is None
    assert all(0 <= v <= 1 for v in d["verification"]["tabulation"].values())

    text = report.to_text()
    assert text.isascii() and "RESONANCE RANGES" in text and "no reload check" in text
    html = report.to_html()
    assert html.startswith("<!doctype html>") and "prefers-color-scheme:dark" in html
    assert "<script" not in html.lower() and "http" not in html
    assert report._repr_html_().startswith("<style>")
    for name in ("r.txt", "r.html", "r.json"):
        assert report.save(tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["schema"] == d["schema"]

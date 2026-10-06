"""What the desktop app needs from layer 1: per-tape MF, progress and stop,
plain-data reports, the exported pages, and a tape's sections without parsing it.

All on the layer-1 micro-tapes (``COV_CHECK_FIXTURES``).
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kika.endf import check_covariance_library, check_covariances, read_endf, tape_inventory
from kika.endf.checks import (CHECKS, LEVELS, CovarianceCheckReport, CovarianceFinding,
                              CovarianceLocation)
from kika.endf.checks.export import jsonable

DATA = Path(__file__).resolve().parent / "data"
CUTS = sorted(DATA.glob("micro_*_covcheck.endf"))
NE20 = DATA / "micro_ne20_covcheck.endf"
W186 = DATA / "micro_w186_covcheck.endf"


@pytest.fixture(scope="module")
def ne20():
    return check_covariances(read_endf(str(NE20)))


def _strictly_plain(value) -> None:
    """Only None, bool, int, finite float, str, list and dict, all the way down."""
    if isinstance(value, dict):
        assert all(type(k) is str for k in value)
        for v in value.values():
            _strictly_plain(v)
    elif isinstance(value, list):
        for v in value:
            _strictly_plain(v)
    else:
        assert value is None or type(value) in (bool, int, float, str), type(value)
        if type(value) is float:
            assert math.isfinite(value)


# ---- 1. per-tape MF -------------------------------------------------------

def test_a_mapping_gives_each_tape_its_own_files():
    report = check_covariance_library({W186: [33], NE20: (34, 33)}, mf=(35,), progress=False)
    w186, ne20 = report.tapes
    assert [w186.name, ne20.name] == [W186.name, NE20.name]  # the mapping's order
    assert w186.mf == (33,) and not w186.has_covariances  # W-186's cut carries MF34 only
    assert ne20.mf == (33, 34) and ne20.report.mf == (33, 34)
    assert ne20.report.counts() == check_covariances(read_endf(str(NE20)), mf=(34,)).counts()


def test_a_mapping_with_no_checked_file_is_refused_before_the_walk():
    seen = []
    with pytest.raises(ValueError, match=f"for {W186.name}"):
        check_covariance_library({NE20: [34], W186: [30]}, progress=False,
                                 on_tape=lambda *a: seen.append(a))
    assert seen == []


def test_tapes_with_the_same_file_name_are_told_apart(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        shutil.copy(NE20, tmp_path / sub / "n.endf")
    report = check_covariance_library(tmp_path, recursive=True, mf=(34,), progress=False)
    assert [t.name for t in report.tapes] == ["a/n.endf", "b/n.endf"]
    assert report.report("b/n.endf").counts() == report.tapes[0].report.counts()


# ---- 2. progress and stop -------------------------------------------------

def test_on_tape_is_called_when_each_tape_is_done():
    seen = []
    check_covariance_library(CUTS[:3], progress=False, on_tape=lambda *a: seen.append(a))
    assert seen == [(i, 3, p.name, True) for i, p in enumerate(CUTS[:3], 1)]


def test_on_tape_reports_a_failed_tape_as_not_ok(tmp_path):
    (tmp_path / "broken.endf").write_text("not ENDF\n")
    seen = []
    check_covariance_library([tmp_path / "broken.endf"], progress=False,
                             on_tape=lambda *a: seen.append(a))
    assert seen == [(1, 1, "broken.endf", False)]


def test_should_stop_returns_the_partial_report():
    seen = []
    report = check_covariance_library(CUTS, progress=False, on_tape=lambda *a: seen.append(a),
                                      should_stop=lambda: len(seen) >= 2)
    assert report.stopped and report.planned == len(CUTS)
    assert [t.name for t in report.tapes] == [p.name for p in CUTS[:2]]
    assert [s[0] for s in seen] == [1, 2]
    assert "stopped" in str(report) and report.to_dict()["stopped"] is True
    full = check_covariance_library(CUTS[:1], progress=False)
    assert not full.stopped and full.planned == 1


# ---- 3. to_dict -----------------------------------------------------------

def test_jsonable_drops_numpy_and_non_finite_numbers():
    out = jsonable({"a": np.float64("nan"), "b": np.int64(3), 4: np.array([1.0, np.inf]),
                    "c": (np.bool_(True), {2, 1}), "d": Path("x")})
    assert out == {"a": None, "b": 3, "4": [1.0, None], "c": [True, [1, 2]], "d": "x"}
    _strictly_plain(out)


def test_report_to_dict_is_plain_data(ne20):
    d = ne20.to_dict()
    _strictly_plain(d)
    json.dumps(d, allow_nan=False)
    assert d["tape"]["name"] == NE20.name and d["tape"]["mat"] == 1025
    assert sum(c["n"] for c in d["counts"]) == len(ne20)
    assert len(d["findings"]) == len(ne20)
    first = d["findings"][0]
    assert first["level"] == "defect" and type(first["location"]["mt"]) is int
    assert first["location_str"].startswith("MF34 MT2xMT2 L")
    # The level floor trims the findings, never the counts.
    d = ne20.to_dict("defect")
    assert {f["level"] for f in d["findings"]} == {"defect"}
    assert sum(c["n"] for c in d["counts"]) == len(ne20)


def test_evidence_with_nan_and_numpy_comes_out_plain():
    finding = CovarianceFinding("x", "warn", "s", CovarianceLocation(mat=np.int64(1), mf=33),
                                {"v": np.float32(np.nan), "arr": np.arange(3)})
    d = CovarianceCheckReport((finding,), source="t.endf", mat=np.int64(1), mf=(33,)).to_dict()
    _strictly_plain(d)
    assert d["findings"][0]["evidence"] == {"v": None, "arr": [0, 1, 2]}


def test_library_to_dict_is_plain_data():
    report = check_covariance_library(CUTS, progress=False)
    d = report.to_dict()
    _strictly_plain(d)
    json.dumps(d, allow_nan=False)
    assert [t["name"] for t in d["tapes"]] == [p.name for p in CUTS]
    assert len(d["findings"]) == sum(1 for _ in report)
    assert sum(r["findings"] for r in d["summary"]) == sum(1 for _ in report)
    assert {f["tape"] for f in d["findings"]} <= {t["name"] for t in d["tapes"]}
    warn = report.to_dict("warn")
    assert len(warn["findings"]) == len(report.at_least("warn"))


def test_write_keeps_the_indices_integers(tmp_path):
    report = check_covariance_library(CUTS, progress=False)
    paths = report.write(tmp_path)
    text = paths["findings"].read_text()
    assert "1025.0" not in text and "\t1025\t" in text
    assert pd.read_csv(paths["files"], sep="\t")["mat"].dtype.kind == "i"


# ---- 4. Markdown and HTML -------------------------------------------------

def test_the_pages_carry_summary_findings_method_and_identity(ne20):
    md = ne20.to_markdown()
    html = ne20.to_html()
    for page in (md, html):
        assert NE20.name in page and "1025" in page
        assert "Method and thresholds" in page and "0.25" not in page  # thresholds as %
        assert "25 %" in page and "2 %" in page
        assert "correlation_out_of_bounds" in page
        assert "11 notes not listed" in page  # notes are counted, not listed
    assert "<link" not in html and "<script" not in html and "http" not in html
    assert "inert_rows" in md.split("## Findings")[0]  # but they are in the summary
    assert md.split("## Findings")[1].count("| note |") == 0
    assert ne20.to_markdown(level="note").split("## Findings")[1].count("| note |") == 11


def test_library_pages_and_write_to_path(tmp_path):
    report = check_covariance_library(CUTS, progress=False, library="cuts")
    out = tmp_path / "r.html"
    html = report.to_html(out)
    assert out.read_text(encoding="utf-8") == html
    md = report.to_markdown()
    for name in (p.name for p in CUTS):
        assert name in md and name in html
    assert "### micro_ne20_covcheck.endf (MAT 1025)" in md


# ---- 6. tape_inventory ----------------------------------------------------

def test_the_inventory_reads_the_directory():
    inv = tape_inventory(NE20)
    assert (inv.mat, inv.za, inv.nsub, inv.source, inv.problem) == (1025, 10020, 10,
                                                                     "directory", None)
    assert inv.mf == (1, 2, 3, 4, 34) and inv.mt(34) == (2,)
    assert inv.library_tag and inv.to_dict()["mf"] == [1, 2, 3, 4, 34]
    _strictly_plain(inv.to_dict())


def test_the_inventory_matches_a_scan_on_every_cut():
    for path in CUTS:
        assert tape_inventory(path).sections == tape_inventory(path, scan=True).sections, path


def test_a_stale_directory_is_what_scan_is_for():
    # This cut kept MF34/MT2 but its directory does not list it.
    path = DATA / "micro_fe56_structural.endf"
    assert 34 not in tape_inventory(path).mf
    assert 34 in tape_inventory(path, scan=True).mf


def test_a_tape_without_a_usable_directory_says_so(tmp_path):
    lines = NE20.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l[70:75] == " 1451")
    # NXC (C6 of the fourth CONT) set to 0.
    cont = lines[head + 3]
    lines[head + 3] = cont[:55] + f"{0:>11}" + cont[66:]
    bad = tmp_path / "nodir.endf"
    bad.write_text("\n".join(lines) + "\n")
    inv = tape_inventory(bad)
    assert inv.sections is None and inv.mf is None and "NXC = 0" in inv.problem
    assert inv.mat == 1025
    assert tape_inventory(bad, scan=True).mf == (1, 2, 3, 4, 34)

    junk = tmp_path / "junk.endf"
    junk.write_text("not an ENDF tape\n")
    inv = tape_inventory(junk)
    assert inv.sections is None and inv.problem == "no MF1/MT451 found"


# ---- the legend -----------------------------------------------------------

def _emitted_checks():
    """Every check name the layer-1 modules can emit, read from their source."""
    import ast

    import kika.endf.checks as pkg

    names = set()
    for path in Path(pkg.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "CovarianceFinding"
                    and node.args):
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
                elif isinstance(first, ast.IfExp):  # "band_gap" if gap else "band_overlap"
                    names |= {first.body.value, first.orelse.value}
    return names


def test_every_check_is_described():
    emitted = _emitted_checks()
    assert len(emitted) > 50
    assert emitted - set(CHECKS) == set(), "describe these in checks/descriptions.py"
    assert set(CHECKS) - emitted == set(), "described but never emitted"
    for d in CHECKS.values():
        assert d.title and d.description and d.mf and set(d.levels) <= set(LEVELS)


def test_the_legend_agrees_with_what_the_cuts_give():
    report = check_covariance_library(CUTS, progress=False)
    for _, f in report:
        d = CHECKS[f.check]
        assert f.location.mf in d.mf, (f.check, f.location.mf)
        assert f.level in d.levels, (f.check, f.level)
    d = report.to_dict("defect")
    assert set(d["checks"]) == {r["check"] for r in d["summary"]}
    entry = d["checks"]["ls1_in_cross_block"]
    assert set(entry) == {"title", "mf", "levels", "description"}
    assert set(entry["levels"]) == {"defect", "note"} and 34 in entry["mf"]
    _strictly_plain(d["checks"])


def test_the_pages_end_with_the_legend_of_their_checks(ne20):
    md, html = ne20.to_markdown(), ne20.to_html()
    legend = md.split("## What each finding means")[1]
    assert "### `ls1_in_cross_block`" in legend and "### `inert_rows`" in legend
    assert "sum_rule_violated" not in legend  # only the checks of this report
    assert '<h3 id="check-ls1_in_cross_block">' in html
    assert set(ne20.to_dict()["checks"]) == {f.check for f in ne20}


def test_library_pages_give_the_full_path_of_each_tape(tmp_path):
    report = check_covariance_library(CUTS, progress=False)
    md, html = report.to_markdown(), report.to_html()
    for p in CUTS:
        assert str(p.resolve()) in md and str(p.resolve()) in html
    # Tapes given as a list: the findings are headed by the path under the
    # directory they share.
    for sub in ("jeff", "endfb"):
        (tmp_path / sub).mkdir()
        shutil.copy(NE20, tmp_path / sub / NE20.name)
    two = check_covariance_library([tmp_path / "jeff" / NE20.name, tmp_path / "endfb" / NE20.name],
                                   progress=False)
    md = two.to_markdown()
    assert f"### jeff/{NE20.name} (MAT 1025)" in md and f"### endfb/{NE20.name}" in md
    assert str(tmp_path.resolve()) in md.split("## Summary")[0]  # the Directory row

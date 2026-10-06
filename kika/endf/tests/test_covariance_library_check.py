"""check_covariance_library: layer 1 over a directory of tapes, one at a time.

The "library" is the six layer-1 micro-tapes (``COV_CHECK_FIXTURES``), whose
findings ``test_covariance_checks_on_real_cuts.py`` pins one by one; here what
is pinned is the walk: every tape gets a row, a broken one does not stop it,
and the tables add up to the per-tape reports.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from kika.endf import check_covariance_library, check_covariances, read_endf

DATA = Path(__file__).resolve().parent / "data"
CUTS = sorted(DATA.glob("micro_*_covcheck.endf"))


@pytest.fixture(scope="module")
def library():
    return check_covariance_library(DATA, patterns=("micro_*_covcheck.endf",),
                                    library="cuts", progress=False)


def test_every_tape_is_checked_and_the_report_matches_one_by_one(library):
    assert [t.name for t in library.tapes] == [p.name for p in CUTS] and len(CUTS) == 6
    assert library.failed == () and len(library.checked) == 6
    for tape in library.tapes:
        alone = check_covariances(read_endf(str(tape.path)))
        assert tape.report.counts() == alone.counts()


def test_the_summary_counts_findings_and_tapes(library):
    summary = library.summary()
    row = summary[(summary.mf == 34) & (summary.check == "correlation_out_of_bounds")]
    assert row[["level", "findings", "tapes"]].values.tolist() == [["defect", 8, 2]]
    assert summary["findings"].sum() == sum(1 for _ in library)
    assert [t.name for t in library.tapes_with("ls1_in_cross_block", "defect")] == [
        "micro_ne20_covcheck.endf"]
    # Defects sort before warnings before notes within an MF.
    mf33 = summary[summary.mf == 33]["level"].tolist()
    assert mf33 == sorted(mf33, key=["defect", "warn", "note"].index)


def test_a_broken_tape_and_a_tape_without_covariances_are_rows_not_errors(tmp_path):
    shutil.copy(DATA / "micro_w186_covcheck.endf", tmp_path / "a_w186.endf")
    (tmp_path / "b_broken.endf").write_text("this is not an ENDF tape\n")
    shutil.copy(DATA / "micro_fe56_structural.endf", tmp_path / "c_no_cov.endf")
    (tmp_path / "README.md").write_text("not a tape")
    report = check_covariance_library(tmp_path, mf=(33,), progress=False)

    assert [t.name for t in report.tapes] == ["a_w186.endf", "b_broken.endf", "c_no_cov.endf"]
    w186, broken, no_cov = report.tapes
    # Only MF33 was asked for, and W-186's cut carries MF34 only.
    assert not w186.has_covariances and w186.ok
    assert not broken.ok and broken.error.startswith("read:")
    assert report.failed == (broken,)
    assert "FAILED b_broken.endf" in str(report)


def test_progress_goes_to_a_callable_one_line_per_tape(tmp_path):
    lines = []
    check_covariance_library(CUTS[:2], progress=lines.append)
    assert lines[0].startswith("[1/2] ") and lines[-1].startswith("2 tapes in ")
    assert sum("defects" in line and line.startswith("[") for line in lines) == 2


def test_progress_true_rewrites_one_line_on_stderr(capsys):
    check_covariance_library(CUTS[:2], progress=True)
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.count("\r") >= 4


def test_write_puts_three_tables_on_disk(library, tmp_path):
    paths = library.write(tmp_path)
    assert sorted(p.name for p in paths.values()) == [
        "covariance_files_cuts.tsv", "covariance_findings_cuts.tsv",
        "covariance_summary_cuts.tsv"]
    findings = pd.read_csv(paths["findings"], sep="\t")
    assert set(findings["level"]) == {"defect", "warn"}  # notes left out by default
    assert len(findings) == len(library.at_least("warn"))
    files = pd.read_csv(paths["files"], sep="\t")
    assert files["n_defect"].sum() == sum(1 for _, f in library if f.level == "defect")


def test_a_path_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_covariance_library(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="31, 33, 34"):
        check_covariance_library(CUTS, mf=(32,))

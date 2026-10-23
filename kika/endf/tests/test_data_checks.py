"""The data checks (everything but covariances), on cuts of real evaluations.

Every micro-tape in ``data/data_checks/`` is a verbatim cut of a published
evaluation -- its MF1/451 and the sections named in the file name -- so a
defect is pinned exactly as the evaluator wrote it. The cut leaves a
directory that lists sections the micro-tape does not have (and an LRP that
promises an MF2 it may not carry): those findings are the cut's, and the tests
below look only at the check each tape is a witness for.

=============================================  =====================================
Micro-tape                                     Witness for
=============================================  =====================================
``micro_pu243_b81_delayed_urr``                delayed nu-bar without MF5/455; a URR
                                               J = I at l = 0 (ENDF/B-VIII.1)
``micro_u241_b81_mf2``                         J = 1/2 on I = 1/2 at l = 0 (B-VIII.1)
``micro_pr141_jendl5_mf2``                     URR parameters stop at 100 of 160 keV
``micro_n14_b81_mt16``                         (n,2n) neutrons beyond E + Q in the CM
``micro_co58_b81_q``                           QM of MT32/MT41 MeV off AME2020
``micro_cm240_b81_delayed``                    a decay constant of 0
``micro_th231_b81_nubar``                      MT452 != MT455 + MT456
``micro_mo100_jeff40_mt102``                   a blank pair at the end of MF3
``micro_cs134_jeff40_mt102``                   a blank pair in the middle of MF3
``micro_h1_b81_clean``                         nothing (MF3, MF4 of H-1)
=============================================  =====================================
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from kika.endf.checks import CheckReport, LibraryReport, check_evaluation, check_library
from kika.endf.checks.data import check_data
from kika.endf.checks.data.descriptions import DATA_CHECKS
from kika.endf.checks.data.legendre import negative_area
from kika.endf.checks.data.reactions import emitted, q_from_masses, residual_za, threshold
from kika.endf.checks.data.resonances import _allowed, _reachable
from kika.endf.checks.findings import DEFECT, LEVELS, NOTE, WARN

DATA = Path(__file__).resolve().parent / "data" / "data_checks"


def _check(name, mf=None):
    return check_data(DATA / f"{name}.endf", mf=mf)


def _levels_of(report, check):
    return sorted({f.level for f in report if f.check == check})


# ---- witnesses ---------------------------------------------------------------

def test_pu243_delayed_neutrons_without_spectra_and_an_unreachable_urr_spin():
    r = _check("micro_pu243_b81_delayed_urr")
    (f,) = r.by_check("delayed_spectra_missing")
    assert f.level == DEFECT and f.location.mt == 455 and f.evidence["nubar"] == pytest.approx(0.0162)
    (s,) = [f for f in r.by_check("resonance_spin") if f.level == DEFECT]
    assert s.location.range_index == 1 and s.evidence["j"] == 3.5 and s.evidence["l_value"] == 0


def test_u241_spin_one_half_at_l0():
    r = _check("micro_u241_b81_mf2", mf=[2])
    (s,) = r.by_check("resonance_spin")
    assert s.level == DEFECT and s.evidence["j"] == 0.5 and s.evidence["spi"] == 0.5


def test_pr141_unresolved_parameters_short_of_their_range():
    r = _check("micro_pr141_jendl5_mf2", mf=[2])
    assert [f.check for f in r] == ["urr_coverage"] and r.findings[0].level == WARN


def test_n14_two_neutrons_beyond_what_n2n_makes_available():
    r = _check("micro_n14_b81_mt16", mf=[6])
    (f,) = [f for f in r.by_check("kinematic_limit") if f.level == DEFECT]
    assert f.location.mt == 16 and f.location.zap == 1
    assert f.evidence["e_out"] > f.evidence["available"] * 1.5


def test_co58_q_values_copied_across_reactions():
    r = _check("micro_co58_b81_q", mf=[3])
    q = {f.location.mt: f for f in r.by_check("q_mass")}
    assert set(q) == {32, 41} and {f.level for f in q.values()} == {WARN}
    assert q[32].evidence["ame"] == pytest.approx(-12.376e6, rel=1e-4)


def test_cm240_zero_decay_constant():
    assert _levels_of(_check("micro_cm240_b81_delayed", mf=[1]), "delayed_constants") == [DEFECT]


def test_th231_total_nubar_is_the_prompt_alone():
    (f,) = _check("micro_th231_b81_nubar", mf=[1]).by_check("nubar_sum")
    assert f.level == WARN and f.evidence["relative"] == pytest.approx(0.017, rel=0.05)


@pytest.mark.parametrize("name, point", [("micro_mo100_jeff40_mt102", 81), ("micro_cs134_jeff40_mt102", 87)])
def test_jeff40_blank_pairs_wherever_they_are(name, point):
    r = _check(name, mf=[3])
    (f,) = r.by_check("table_blank_pairs")
    assert f.level == DEFECT and f.location.mt == 102 and str(point) in f.summary
    assert not r.by_check("table_grid_order")       # the blank, said once, not as a decreasing grid


def test_a_clean_cut_has_nothing_at_mf3_and_mf4():
    r = _check("micro_h1_b81_clean", mf=[3, 4])
    assert [f for f in r if f.level != NOTE] == []


# ---- the physics behind the checks -------------------------------------------

def test_threshold_and_q_from_masses():
    # Fe-56(n,2n): Q = -11.197 MeV (AME2020), threshold -Q (A+1)/A to first order.
    q = q_from_masses(26056, 1, 16)
    assert q == pytest.approx(-11.1975e6, rel=1e-4)
    awr = 55.454
    assert threshold(q, awr, 1.0) == pytest.approx(-q * (awr + 1) / awr, rel=2e-4)
    assert threshold(q, awr, 1.0) > -q * (awr + 1) / awr     # relativistic: a little more
    assert residual_za(26056, 1, 107) == 24053 and emitted(16) == {1: 2} and emitted(5) is None


def test_spin_algebra():
    assert not _reachable(3.5, 3.5, 0) and _reachable(3.0, 3.5, 0) and _reachable(4.0, 3.5, 0)
    assert _allowed(0.0, 1) == [0.5, 1.5] and _allowed(0.5, 0) == [0.0, 1.0]


def test_legendre_negative_area():
    area, fmin = negative_area(np.array([[0.0], [1.0], [1 / 3]]))
    # f = 1/2 + 3/2 a1 mu: a1 = 1 is negative below mu = -1/3, with area 1/3.
    assert area[0] == 0 and area[2] == pytest.approx(0, abs=1e-12)
    assert area[1] == pytest.approx(1 / 3, rel=1e-3) and fmin[1] == pytest.approx(-1.0)


# ---- the framework -----------------------------------------------------------

def _emitted_data_checks():
    import kika.endf.checks.data as pkg

    names = set()
    files = list(Path(pkg.__file__).parent.glob("*.py")) + [Path(pkg.__file__).parents[3] / "_check.py"]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "add") or \
                        (isinstance(func, ast.Name) and func.id == "Finding"):
                    names.add(node.args[0].value)
    return names


def test_every_data_check_is_described_and_ids_are_unique():
    from kika.endf.checks.descriptions import CHECKS

    emitted_ids = _emitted_data_checks()
    assert len(emitted_ids) > 80
    assert emitted_ids - set(DATA_CHECKS) == set(), "describe these in checks/data/"
    assert set(DATA_CHECKS) - emitted_ids == set(), "described but never emitted"
    assert set(DATA_CHECKS) & set(CHECKS) == set()
    for d in DATA_CHECKS.values():
        assert d.title and d.description and d.mf and set(d.levels) <= set(LEVELS)


def test_evaluation_report_joins_both_families_and_each_alone():
    tape = DATA.parent / "micro_fe56_jendl_covcheck.endf"
    both = check_evaluation(tape)
    data = check_evaluation(tape, what="data")
    cov = check_evaluation(tape, what="covariances")
    assert (both.kind, data.kind, cov.kind) == ("evaluation", "data", "covariance")
    assert both.counts() == (data + cov).counts()
    assert cov.findings and all(f.location.mf >= 31 for f in cov)


def test_library_walk_and_pages(tmp_path):
    lib = check_library(DATA, what="data", patterns=("micro_*.endf",), library="witnesses", progress=False)
    assert isinstance(lib, LibraryReport) and lib.kind == "data" and len(lib.tapes) == 10
    assert not lib.failed
    page = lib.to_html(tmp_path / "w.html")
    assert "Data check of witnesses" in page and "delayed_spectra_missing" in page
    assert all(t.has_covariances for t in lib.tapes)    # every tape had data to check


def test_kika_check_front_door(tmp_path):
    import kika

    one = kika.check(DATA / "micro_th231_b81_nubar.endf", what="data", mf=[1])
    assert isinstance(one, CheckReport) and one.by_check("nubar_sum")
    html = one.to_html()
    assert "<title>Data check of micro_th231_b81_nubar.endf</title>" in html
    folder = kika.check(DATA, what="data", mf=[3], progress=False)
    assert isinstance(folder, LibraryReport) and len(folder.tapes) == 10


def test_cli_writes_one_html_page_and_fails_on_defects(tmp_path, capsys):
    from kika._cli import main

    status = main(["check", str(DATA / "micro_cs134_jeff40_mt102.endf"), "--what", "data", "--mf", "3",
                   "-o", str(tmp_path)])
    assert status == 1 and [p.name for p in tmp_path.iterdir()] == ["micro_cs134_jeff40_mt102.check.html"]
    assert "1 defect" in capsys.readouterr().out
    assert main(["check", str(DATA / "micro_h1_b81_clean.endf"), "--what", "data", "--mf", "3", "4",
                 "--no-report"]) == 0

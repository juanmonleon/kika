"""Layer 1 travels with the model covariance, and the pre-flight reads it (phase C6).

``decodeCovarianceSuite`` runs :func:`kika.endf.check_covariances` and leaves
each section's findings on its provenance; :mod:`kika.sampling.section_checks`
traces a pre-flight finding on the assembled joint back to them. The case is
the defect A of JEFF-4.0 MF34: an LB=5 LS=1 triangle in a cross-order block,
harmless as stored and |rho| > 1 once kika mirrors it.
Plan: kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf33 import MF33MT, NISubSubsectionRecord, Subsection
from kika.endf.classes.mf34 import MF34MT
from kika.endf.classes.mf34 import Subsection as Subsection34
from kika.endf.classes.mf34 import SubSubsection, SubSubsectionRecord
from kika.endf.model_adapter.covariances import decodeCovarianceSuite

GRID = [1.0e5, 1.0e6, 1.0e7]


def _lb5(matrix, cls=SubSubsectionRecord, ls=1):
    m = np.asarray(matrix, dtype=float)
    values = list(m[np.triu_indices(m.shape[0])]) if ls == 1 else list(m.ravel())
    return cls(lb=5, ls=ls, ne=len(GRID), nt=len(GRID) + len(values),
               energies=list(GRID), matrix=values)


def _tape(mf_number, *sections):
    endf = ENDF()
    mf = MF(number=mf_number)
    for sec in sections:
        mf.add_section(sec)
    endf.add_file(mf)
    return endf


def _defect_a():
    """MF34 MT2: a_1 and a_2 with small variances in one bin each, and an LS=1
    (1, 2) block whose stored triangle is a valid correlation (0.9) but whose
    mirror meets the two small variances: 0.009 / sqrt(1e-4 * 1e-4) = 90."""
    sec = MF34MT(number=2, _za=26056.0, _awr=55.45, _ltt=1, _nmt1=1, _mat=2631)
    blocks = {(1, 1): [[0.01, 0.0], [0.0, 1e-4]],
              (1, 2): [[5e-4, 0.009], [0.009, 5e-4]],
              (2, 2): [[1e-4, 0.0], [0.0, 0.01]]}
    sec.add_subsection(Subsection34(
        mt1=2, nl=2, nl1=2, mat1=0,
        sub_subsections=[SubSubsection(l=l, l1=l1, lct=1, ni=1, records=[_lb5(m)])
                         for (l, l1), m in blocks.items()]))
    return _tape(34, sec)


def test_the_decode_leaves_each_sections_findings_on_its_provenance():
    rho2 = [[0.01, 0.02], [0.02, 0.01]]  # |rho| = 2 in a self block
    sec = MF33MT(number=1, _za=26056.0, _awr=55.45, _mat=2631, _nl=1)
    sec.add_subsection(Subsection(mt1=1, mat1=0, nc=0, ni=1,
                                  ni_records=[_lb5(rho2, cls=NISubSubsectionRecord)]))
    suite, report = decodeCovarianceSuite(_tape(33, sec))
    found = suite.covarianceSections[0].provenance.covarianceFindings
    assert "correlation_out_of_bounds" in {f.check for f in found}
    assert all(f.location.mf == 33 and f.location.mt == 1 for f in found)
    assert suite.covarianceChecks.defects
    # File findings are not conversion losses: the conversion report does not
    # change because the file is faulty.
    unchecked, unchecked_report = decodeCovarianceSuite(_tape(33, sec), checks=False)
    assert unchecked.covarianceChecks is None
    assert unchecked.covarianceSections[0].provenance.covarianceFindings == ()
    assert report.summary() == unchecked_report.summary()


def test_a_rho_above_one_in_the_joint_is_traced_to_the_ls1_record():
    from kika.cov.conditioning import inspect_blocks
    from kika.sampling.joint_blocks import assembleRequest, collectEntries
    from kika.sampling.section_checks import attribute, relevantFindings, sectionFindings

    suite, _ = decodeCovarianceSuite(_defect_a())
    blocks, index = assembleRequest(collectEntries(suite, {34: None}))
    report = inspect_blocks(blocks, predict=False)
    assert not report.samplable  # layer 2 blocks the draw, as before

    findings = relevantFindings(sectionFindings(suite), index)
    assert "ls1_in_cross_block" in {f.check for f in findings}
    traced = [a for a in attribute(blocks, index, report, findings)
              if a["check"] == "correlation_bound"]
    assert len(traced) == 1
    item = traced[0]
    assert item["measure"] == pytest.approx(90.0)
    assert any("ls1_in_cross_block" in text and "L1xL2" in text for text in item["layer1"])
    assert "L=1" in item["text"] and "L=2" in item["text"] and "<- MF34" in item["text"]


def test_a_block_with_no_file_finding_says_so():
    from kika.cov.conditioning import inspect_blocks
    from kika.sampling.joint_blocks import assembleRequest, collectEntries
    from kika.sampling.section_checks import attribute

    suite, _ = decodeCovarianceSuite(_defect_a(), checks=False)
    blocks, index = assembleRequest(collectEntries(suite, {34: None}))
    report = inspect_blocks(blocks, predict=False)
    traced = attribute(blocks, index, report, ())
    assert traced and all(a["layer1"] == [] for a in traced)
    assert "no layer-1 finding" in traced[0]["text"]


def test_the_sampler_shows_layer1_and_names_the_cause_when_it_refuses(tmp_path, monkeypatch):
    from kika.sampling import model_perturbation as mp

    endf = _defect_a()
    suite, covReport = decodeCovarianceSuite(endf)
    monkeypatch.setattr(mp, "_readSource",
                        lambda source, log, covarianceSource=None:
                        (None, suite, endf, None, "endf", (covReport, None)))
    with pytest.raises(ValueError, match="Traced to the file:.*ls1_in_cross_block"):
        mp.perturbFromModel("tape.endf", {34: None}, 1, dryRun=True)


def test_the_real_ne20_rho_is_traced_to_its_ls1_records():
    """The same on the evaluation it was modelled on: JEFF-4.0 Ne-20 MF34/MT2,
    committed as ``micro_ne20_covcheck.endf``. Each of the seven |rho| > 1 the
    pre-flight finds in the joint is traced to the LS=1 record of its block."""
    from pathlib import Path

    from kika.cov.conditioning import inspect_blocks
    from kika.endf import read_endf
    from kika.sampling.joint_blocks import assembleRequest, collectEntries
    from kika.sampling.section_checks import attribute, relevantFindings, sectionFindings

    tape = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_ne20_covcheck.endf"
    suite, _ = decodeCovarianceSuite(read_endf(str(tape)))
    blocks, index = assembleRequest(collectEntries(suite, {34: [2]}))
    report = inspect_blocks(blocks, predict=False)
    assert not report.samplable

    traced = [a for a in attribute(blocks, index, report,
                                   relevantFindings(sectionFindings(suite), index))
              if a["check"] == "correlation_bound"]
    assert len(traced) == 7
    assert all(any("ls1_in_cross_block" in text for text in a["layer1"]) for a in traced)
    worst = max(traced, key=lambda a: a["measure"])
    assert worst["measure"] == pytest.approx(4.634, abs=1e-3)
    assert "L=1" in worst["text"] and "L=5" in worst["text"] and "L1xL5" in worst["text"]

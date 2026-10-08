"""MF4 of MT600-849 is the outgoing charged particle's, not a neutron's.

ENDF-6 §4 lets the MT name the particle MF4 describes: MT600-649 the proton
that leaves the residual in level ``MT - 600``, then the deuteron, triton, He-3
and alpha. The adapter used to hang every MF4 on an ``n`` product, so a suite
decoded from a tape said ``(n,p_0)`` emits a neutron with the proton's angular
distribution (G4NDL roadmap §5 Fase 10, D10-5).

No committed tape has an MF4 in that range, so this one is made from the W-186
micro-tape by renumbering its MT51 (MF3, MF4 and the MF34 that points at it) to
MT600: the bodies are real, only the label moves.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite
from kika.endf.model_adapter.angular import mf4Ejectile
from kika.endf.model_adapter.covariances import angularDistributionHref
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import writeEndfTape


def _renumbered(source, target, old=51, new=600):
    """*source* with MT *old* relabelled *new*: in MF3/4/34, MF1's directory and MF34's MT1."""
    out, inMF34 = [], False
    for line in source.read_text().splitlines():
        mf, mt = line[70:72].strip(), line[72:75].strip()
        if mt == str(old) and mf in ("3", "4", "34"):
            line = f"{line[:72]}{new:3d}{line[75:]}"
            if mf == "34" and not inMF34:
                inMF34 = True
            elif mf == "34" and line[33:44].strip() == str(old):
                line = f"{line[:33]}{new:11d}{line[44:]}"
        elif mf == "1" and mt == "451" and line[22:33].strip() in ("3", "4", "34") \
                and line[33:44].strip() == str(old):
            line = f"{line[:33]}{new:11d}{line[44:]}"
        out.append(line)
    target.write_text("\n".join(out) + "\n")
    return target


@pytest.fixture(scope="module")
def mt600Tape(tmp_path_factory):
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "tests" / "data" / "micro_w186_covcheck.endf"
    return _renumbered(source, tmp_path_factory.mktemp("mt600") / "w186_mt600.endf")


def _decode(path):
    endf = read_endf(str(path))
    suite, report = decodeReactionSuite(endf)
    suite.covarianceSuite, report = decodeCovarianceSuite(endf, report)
    return endf, suite


@pytest.mark.parametrize("mt, pid", [(2, "n"), (51, "n"), (91, "n"), (600, "H1"),
                                     (649, "H1"), (650, "H2"), (700, "H3"),
                                     (750, "He3"), (800, "He4"), (849, "He4"),
                                     (875, "n")])
def test_the_mt_names_the_particle(mt, pid):
    assert mf4Ejectile(mt) == pid


def test_mt600_mf4_hangs_on_the_proton(mt600Tape):
    _, suite = _decode(mt600Tape)
    products = list(suite.findReactionByENDF_MT(600).outputChannel.products)
    assert [p.pid for p in products] == ["H1"]
    assert products[0].distribution is not None


def test_mt600_mf34_points_at_the_proton(mt600Tape):
    _, suite = _decode(mt600Tape)
    hrefs = {section.rowData.href for section in suite.covarianceSuite.covarianceSections
             if section.rowData.ENDF_MF == 34}
    assert hrefs == {angularDistributionHref(600)}
    assert "product[@label='H1']" in angularDistributionHref(600)


def test_mt600_mf4_is_written_back_unchanged(mt600Tape, tmp_path):
    endf, suite = _decode(mt600Tape)
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out)
    again = read_endf(str(out))
    assert str(again.mf[4].mt[600]) == str(endf.mf[4].mt[600])

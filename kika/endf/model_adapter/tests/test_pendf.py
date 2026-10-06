"""A PENDF's MF3 into the model as the ``recon`` style (G4NDL roadmap Phase 9).

The fast tests use the Fe-56 micro-tape as a stand-in PENDF of itself: it has
the MF1/451 header with ``TEMP = 0`` and lin-lin MF3 tables, which is all
:func:`attachReconstruction` reads. RECONR itself runs in
``kika/g4ndl/tests/test_from_endf.py`` (``njoy``-marked).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.model_adapter.pendf import attachReconstruction, pendfHeader

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"


def _suite():
    return kika.read(str(FE56), format="endf")


def test_header_fields():
    h = pendfHeader(FE56)
    assert (h["mat"], h["za"], h["nfor"]) == (2631, 26056, 6)
    assert h["awr"] == pytest.approx(55.45443) and h["temp"] == 0.0


def test_attach_adds_the_style_and_lin_lin_forms():
    suite = _suite()
    assert "recon" not in suite.styleLabels()
    report = attachReconstruction(suite, FE56)
    assert "recon" in suite.styleLabels()
    xs = suite.reactions[2].crossSection["recon"]
    ev = suite.reactions[2].crossSection["eval"]
    x, y, _ = ev.toEndfRegions()
    assert np.array_equal(xs.xs, x) and np.array_equal(xs.ys, y)
    assert xs.endfInterpolationCode == 2
    assert any("RECONR" in a for a in report.approximations)


def test_a_heated_pendf_is_refused(tmp_path):
    lines = FE56.read_text().splitlines(keepends=True)
    # The fourth MF1/451 record holds TEMP; put 293.6 K there.
    i = [k for k, l in enumerate(lines) if l[70:75] == " 1451"][3]
    lines[i] = " 2.936000+2" + lines[i][11:]
    heated = tmp_path / "heated.pendf"
    heated.write_text("".join(lines))
    assert pendfHeader(heated)["temp"] == pytest.approx(293.6)
    with pytest.raises(ValueError, match="count the Doppler effect twice"):
        attachReconstruction(_suite(), heated)


def test_another_material_is_refused():
    other = kika.read(str(DATA / "micro_u235_nubar.endf"), format="endf")
    with pytest.raises(ValueError, match="MAT"):
        attachReconstruction(other, FE56)


def test_requested_mts_must_exist():
    with pytest.raises(ValueError, match="not attached"):
        attachReconstruction(_suite(), FE56, mts=[2, 999])

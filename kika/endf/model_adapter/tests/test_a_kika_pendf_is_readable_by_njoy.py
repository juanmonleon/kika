"""A PENDF kika writes is one NJOY can read.

Two things stopped NJOY on a kika PENDF of B-VIII.1 Fe-56, each measured:

* **Interpolation regions.** A reconstructed MF3 is a ``regions1d`` with a
  region per discontinuity (73 for MT1; 688 in JEFF-4.0), and NJOY holds 20
  (``nbt(20)`` in ``endf.f90``): MODER stopped with ``illegal TAB1,
  nbt(nr)/=np``. The table is now written one region per law, as RECONR writes
  it, with the jumps as repeated energies -- the same function -- and the
  checks that read it back cut it at their own region edges
  (:func:`curvesOnDomains`).
* **MF1 text.** With all 560 lines of the evaluation's documentation in
  MF1/451, HEATR crashed (0xC0000374); with 300 it ran. RECONR does not carry
  that text, and neither does a kika PENDF now beyond ENDF-6's identification
  block (:func:`trimDescription`).

The NJOY runs themselves need the executable and a full tape, so they are not
here; these pin the two properties that made them pass.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from kika.endf.model_adapter.encode import mergedInterpolationRegions
from kika.endf.model_adapter.pendf import PENDF_DESCRIPTION_LINES, trimDescription
from kika.nuclear_data.model import Regions1d, XYs1d
from kika.nuclear_data.model.functions.regions1d import curvesOnDomains

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _regions():
    """Three lin-lin regions: a jump at 2, a shared point at 3."""
    return Regions1d(function1ds=[
        XYs1d(xs=np.array([1., 1.5, 2.]), ys=np.array([1., 1., 1.])),
        XYs1d(xs=np.array([2., 2.5, 3.]), ys=np.array([5., 5., 4.])),
        XYs1d(xs=np.array([3., 4.]), ys=np.array([4., 2.])),
    ])


def test_neighbouring_regions_of_one_law_are_written_as_one():
    assert mergedInterpolationRegions([(3, 2), (6, 2), (7, 2)]) == [(7, 2)]
    assert mergedInterpolationRegions([(3, 2), (6, 5), (9, 5), (12, 2)]) == \
        [(3, 2), (9, 5), (12, 2)]


def test_the_merged_table_states_the_same_function():
    form = _regions()
    xs, ys, pairs = form.toEndfRegions()
    merged = Regions1d.fromEndfRegions(xs, ys, mergedInterpolationRegions(pairs))
    probes = np.array([1.2, 1.99, 2.0, 2.01, 2.7, 3.0, 3.5, 4.0])
    np.testing.assert_array_equal(form.evaluate(probes), merged.evaluate(probes))


def test_a_merged_table_is_cut_back_at_the_region_edges():
    form = _regions()
    xs, ys, pairs = form.toEndfRegions()
    merged = Regions1d.fromEndfRegions(xs, ys, mergedInterpolationRegions(pairs))
    assert len(merged) == 1
    cut = curvesOnDomains(merged, [(c.domainMin, c.domainMax) for c in form.function1ds])
    for a, b in zip(cut, form.function1ds):
        np.testing.assert_array_equal(a.xs, b.xs)
        np.testing.assert_array_equal(a.ys, b.ys)


def test_an_edge_that_is_not_in_the_table_is_not_invented():
    form = XYs1d(xs=np.array([1., 2., 3.]), ys=np.array([1., 2., 3.]))
    assert curvesOnDomains(form, [(1., 2.5), (2.5, 3.)]) is None
    assert curvesOnDomains(form, [(1., 3.)]) == [form]


def test_a_pendf_keeps_the_identification_block_of_mf1(tmp_path):
    tape = tmp_path / "tape.endf"
    shutil.copyfile(DATA / "micro_fe56_mf33.endf", tape)
    before = tape.read_text().splitlines()
    assert trimDescription(tape) == PENDF_DESCRIPTION_LINES
    after = tape.read_text().splitlines()

    head = [i for i, line in enumerate(after) if line[70:75] == " 1451"]
    assert int(after[head[3]][44:55]) == PENDF_DESCRIPTION_LINES
    text = [line[:66] for line in after[head[3] + 1:head[3] + 1 + PENDF_DESCRIPTION_LINES]]
    assert text == [line[:66] for line in before[head[3] + 1:head[3] + 1 + PENDF_DESCRIPTION_LINES]]
    # Nothing outside MF1/451 moved.
    assert [l for l in after if l[70:75] != " 1451"] == [l for l in before if l[70:75] != " 1451"]
    # And it is idempotent.
    assert trimDescription(tape) == PENDF_DESCRIPTION_LINES
    assert tape.read_text().splitlines() == after

"""LSSF=0 URR tables against an MF3 that already holds the resonances.

An LSSF=0 table is the ladders' cross sections plus MF3 as a background. A
perturbation run's base tape carries RECONR's 0 K reconstruction in MF3
(LRP=2), URR mean included, and the tables shared by its samples counted that
mean twice (Rh-103 ENDF/B-VIII.1: elastic and capture x2.00 of the dilute
cross section). And the samples' MF33 factors never reached those tables,
which OpenMC does not multiply by the smooth cross sections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.hdf5 import processing as P
from kika.nuclear_data import model as m

TAPE = Path(kika.__file__).parent / "endf" / "tests" / "data" / "micro_hf176_covcheck.endf"


def _backgroundOf(monkeypatch, lrp):
    from kika.processing.resonances import probability_tables as T

    seen = {}

    def capture(region, context, energies, temperatures, *, background, **kwargs):
        seen.update(energies=np.asarray(energies), background=background)
        return {}

    monkeypatch.setattr(T, "probabilityTables", capture)
    suite = kika.read(str(TAPE), covariances=False)
    suite.resonances.unresolved.tabulatedWidths.selfShieldingOnly = False
    suite.provenance.headerFields["lrp"] = lrp
    P.urrTablesOf(suite, [293.6])
    return suite, seen


def test_an_lrp2_mf3_gives_the_ladders_its_background_less_the_urr_mean(monkeypatch):
    from kika.processing.resonances.probability_tables import _prepared
    from kika.processing.resonances.unresolved import evaluate_unresolved

    _, stated = _backgroundOf(monkeypatch, 1)
    suite, complete = _backgroundOf(monkeypatch, 2)
    at = stated["energies"].copy()
    at[0] *= 1.0 + 1e-6
    at[-1] *= 1.0 - 1e-6
    mean = evaluate_unresolved(at, _prepared(suite.resonances.unresolved, P._urrContext(suite, None)))
    assert np.any(mean[2] > 0)
    for mt in (1, 2, 18, 102):
        np.testing.assert_allclose(complete["background"][mt], stated["background"][mt] - mean[mt],
                                   rtol=0, atol=1e-12 * max(1.0, float(np.abs(mean[mt]).max())))


def _table():
    E = np.array([1.0e3, 2.0e3])
    rows = np.zeros((2, 6, 2))
    rows[:, 0] = [0.4, 1.0]
    rows[:, 2] = [[4.0, 8.0], [5.0, 9.0]]          # elastic
    rows[:, 4] = [[1.0, 3.0], [2.0, 4.0]]          # capture
    rows[:, 1] = rows[:, 2] + rows[:, 4]           # total
    rows[:, 5] = 1.0e6
    return m.ProbabilityTables(energies=E, table=rows, multiplySmooth=False)


def _xs(ys):
    return m.XYs1d(xs=[1.0e3, 2.0e3], ys=ys)


def test_unit_factors_give_the_table_back():
    table = _table()
    same = P._perturbedTable(table, (True, {2: (_xs([7.0, 7.0]), _xs([7.0, 7.0]))}))
    np.testing.assert_array_equal(same.table, table.table)


def test_a_complete_mf3_scales_the_bands_and_the_total_follows():
    table = _table()
    out = P._perturbedTable(table, (True, {2: (_xs([11.0, 6.0]), _xs([10.0, 6.0])),
                                           102: (_xs([2.0, 3.0]), _xs([2.0, 2.0]))}))
    np.testing.assert_allclose(out.row("elastic"), table.row("elastic") * np.array([[1.1], [1.0]]))
    np.testing.assert_allclose(out.row("capture"), table.row("capture") * np.array([[1.0], [1.5]]))
    np.testing.assert_allclose(out.row("total"), out.row("elastic") + out.row("capture"))
    np.testing.assert_array_equal(out.row("cdf"), table.row("cdf"))
    np.testing.assert_array_equal(table.row("elastic"), [[4.0, 8.0], [5.0, 9.0]])


def test_a_background_mf3_shifts_the_bands_by_what_the_realisation_moved():
    table = _table()
    out = P._perturbedTable(table, (False, {102: (_xs([0.5, 0.0]), _xs([0.25, 0.0]))}))
    np.testing.assert_allclose(out.row("capture"), table.row("capture") + np.array([[0.25], [0.0]]))
    np.testing.assert_allclose(out.row("total"), table.row("total") + np.array([[0.25], [0.0]]))


@pytest.mark.tape
def test_rh103_tables_from_its_lrp2_reconstruction_equal_the_tapes():
    import copy
    import os

    tapes = os.environ.get("KIKA_LIB_TAPES")
    path = Path(tapes) / "n-045_Rh_103.endf" if tapes else None
    if path is None or not path.is_file():
        pytest.skip("ENDF/B-VIII.1 Rh-103 not found (set KIKA_LIB_TAPES)")
    suite = kika.read(str(path), covariances=False)
    stated = P.urrTablesOf(suite, [293.6])[293.6]
    work = P._reconstructed(suite, None, m.ConversionReport())
    complete = copy.deepcopy(work)
    from kika.processing.resonances.suite import _entries

    for reaction in _entries(complete).values():
        reaction.crossSection.forms[P.EVAL_LABEL] = reaction.crossSection.forms[P.RECON_LABEL]
    complete.provenance.headerFields["lrp"] = 2
    np.testing.assert_allclose(P.urrTablesOf(complete, [293.6])[293.6].table, stated.table,
                               rtol=1e-9, atol=1e-12)

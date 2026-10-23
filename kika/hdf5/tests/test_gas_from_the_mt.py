"""Gas production counts the light particles a reaction's MT defines when it states none.

ENDF/B-VIII.1 N-14 gives MT600-810 as MF3 alone: no MF4 or MF6, so no product in
the model. Counting only stated products, N-14's MT203, 204 and 207 were zero
everywhere and MT205 was missing, where GASPR counts one proton per (n,p), one
alpha per (n,alpha) and so on from the MT. GASPR also counts the residual
when it is light itself: Li-7's levels that break up into a triton leave an
alpha, and counting only the triton put Li-7's MT207 at 0.020 b at 14 MeV
where the evaluation states 0.32 b.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kika.hdf5.processing import gasMultiplicities


def _reaction(mt, products=()):
    return SimpleNamespace(ENDF_MT=mt, outputChannel=SimpleNamespace(products=list(products)))


def _counts(reaction, target=None):
    return {gas: form.constant for gas, form in gasMultiplicities(reaction, target).items()}


def test_a_level_reaction_with_no_products_emits_its_series_particle():
    assert _counts(_reaction(600)) == {203: 1.0}
    assert _counts(_reaction(701)) == {205: 1.0}
    assert _counts(_reaction(810)) == {207: 1.0}


def test_a_reaction_with_no_products_emits_what_its_mt_defines():
    assert _counts(_reaction(45)) == {203: 1.0, 207: 1.0}
    assert _counts(_reaction(108)) == {207: 2.0}
    assert _counts(_reaction(102)) == {}
    assert _counts(_reaction(16)) == {}


def test_a_level_that_breaks_up_emits_its_lr_particles():
    # N-14 MT60-77: LR=28, the level breaks up into a proton.
    level = _reaction(60)
    level.provenance = SimpleNamespace(lr=28)
    assert _counts(level) == {203: 1.0}
    level.provenance = SimpleNamespace(lr=0)
    assert _counts(level) == {}
    level.provenance = SimpleNamespace(lr=30)  # n + 2 alpha
    assert _counts(level) == {207: 2.0}


def test_stated_products_are_used_as_stated():
    proton = SimpleNamespace(pid="H1", multiplicity=None)
    assert _counts(_reaction(600, [proton])) == {203: 1.0}
    residual = SimpleNamespace(pid="C13", multiplicity=None)
    assert _counts(_reaction(103, [proton, residual])) == {203: 1.0}


def test_a_light_residual_counts_as_gas():
    # Li-7 + n: a level with LR=33 breaks up into t + alpha.
    level = _reaction(52)
    level.provenance = SimpleNamespace(lr=33)
    assert _counts(level, 3007) == {205: 1.0, 207: 1.0}
    # Li-7 (n,2n alpha) leaves a deuteron; (n,3n alpha) a proton.
    assert _counts(_reaction(24), 3007) == {204: 1.0, 207: 1.0}
    assert _counts(_reaction(25), 3007) == {203: 1.0, 207: 1.0}
    # H-1 capture is a deuteron, H-2 (n,2n) a proton.
    assert _counts(_reaction(102), 1001) == {204: 1.0}
    assert _counts(_reaction(16), 1002) == {203: 1.0}
    # Be-9 (n,2n) leaves Be-8, two alphas.
    assert _counts(_reaction(16), 4009) == {207: 2.0}
    # A heavy residual is not gas.
    assert _counts(_reaction(103), 26056) == {203: 1.0}
    assert _counts(_reaction(102), 26056) == {}


def test_the_mts_beyond_117_emit_what_they_define():
    assert _counts(_reaction(182), 26056) == {204: 1.0, 205: 1.0}
    assert _counts(_reaction(195), 26056) == {207: 2.0}
    assert _counts(_reaction(152), 26056) == {}


# the repository's conftest, by path (see kika/processing/tests/test_t3_tapes.py)
_spec = importlib.util.spec_from_file_location(
    "_kika_root_conftest", Path(__file__).resolve().parents[3] / "conftest.py")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)


@pytest.mark.tape
@pytest.mark.parametrize("name", ["gas_li7_b81", "gas_f19_jendl5"])
def test_a_stated_gas_production_is_replaced_by_the_sum(name):
    # The evaluation's own MT203-207 is a total over its reactions. Summed
    # with them it counted twice (F-19 came out at exactly twice its stated
    # MT203-207), and as a reaction it collided with the sum in the HDF5 file.
    # The sum replaces it, as GASPR replaces it, and the report says by how
    # much they differ.
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.hdf5.processing import _flatten, processSuite

    tape = _root.resolve_tape(name)
    if tape is None:
        pytest.skip(f"{name} is not reachable")
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    stated = {r.ENDF_MT: r for r in suite.reactions if r.ENDF_MT in range(203, 208)}
    out = processSuite(suite, temperatures=[293.6], endf=str(tape))
    assert not any(r.ENDF_MT in stated for r in out.reactions)
    written = {r.ENDF_MT: r for r in out.productions if r.ENDF_MT in range(203, 208)}
    assert set(stated) <= set(written)
    assert all(written[mt].provenance.attributes["redundant"] == 1 for mt in written)
    notes = [w for w in out.report.warnings if "the evaluation states is replaced by the sum" in w]
    assert len(notes) == len(stated)
    # The same, as numbers, for the processing report's gas check.
    assert [row["mt"] for row in out.gasReplaced] == sorted(stated)
    for mt, reaction in stated.items():
        # Where the evaluation's MT203-207 covers what its reactions emit
        # (1-20 MeV), the two agree within the reconstruction tolerance. Li-7's
        # stated MT207 also holds the capture alphas below, JENDL-5 F-19's
        # stops at 20 MeV where its MT5 goes on to 200 MeV.
        x, y = (np.asarray(v, float) for v in _flatten(next(iter(reaction.crossSection.forms.values()))
                                                         if isinstance(reaction.crossSection.forms, dict)
                                                         else list(reaction.crossSection.forms)[-1]))
        f = written[mt].crossSection
        xs, ys = _flatten(f[next(iter(f))])
        keep = (x >= 1e6) & (x < 2e7)     # F-19's stated step to zero sits at 20 MeV
        on = np.interp(x[keep], np.asarray(xs, float), np.asarray(ys, float), left=0.0, right=0.0)
        assert np.max(np.abs(on - y[keep])) < 1e-3 * y.max(), mt


def _write(rows):
    return dict(stage="write", status="done", details=dict(format="hdf5", gas=rows))


def test_the_report_checks_a_replaced_gas_production():
    from kika.endf.processing_report import _gas_check

    row = dict(mt=207, difference_b=1e-6, at_eV=1.4e7, peak_b=0.4, fraction=2.5e-6)
    # Nothing tabulated, or an evaluation that states no MT203-207: no check.
    assert _gas_check([]) is None
    assert _gas_check([_write([])]) is None
    assert _gas_check([_write([row])])["status"] == "passed"
    assert _gas_check([_write([dict(row, fraction=5e-3)])])["status"] == "note"
    # Li-7's capture alphas: the evaluation's MT207 holds 4.6 b the sum leaves out.
    warn = _gas_check([_write([dict(row, difference_b=4.6, at_eV=1e-5, peak_b=4.57, fraction=1.0)])])
    assert warn["status"] == "warning" and "4.6 b at 1e-05 eV" in warn["items"][0]

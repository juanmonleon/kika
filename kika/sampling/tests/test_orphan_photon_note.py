"""Decision J9 of the E5 plan: a run that moves MF3 says the sums' photons stay put."""
from __future__ import annotations

from types import SimpleNamespace

from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.sampling.model_perturbation import _orphanPhotonNote



def _index(mf):
    return {"block": {"components": [SimpleNamespace(mf=mf)]}}


def test_a_run_that_moves_mf3_names_the_sums_whose_photons_it_leaves(n14_b81_tape):
    """The whole N-14 of ENDF/B-VIII.1: its sums are sums there, and carry photons
    (the micro-tape keeps MT4 and MT103-107 without their partials)."""
    suite, _ = decodeReactionSuite(read_endf(str(n14_b81_tape)))
    mts = sorted(int(o.ENDF_MT) for o in suite.orphanProducts)
    assert mts, "N-14's sums carry photons"
    note = _orphanPhotonNote(suite, _index(3))
    assert note is not None and "J9" in note and str(mts) in note


def test_a_run_that_does_not_touch_mf3_says_nothing(n14_b81_tape):
    suite, _ = decodeReactionSuite(read_endf(str(n14_b81_tape)))
    assert _orphanPhotonNote(suite, _index(34)) is None

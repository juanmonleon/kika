"""Gas production counts the light particles a reaction's MT defines when it states none.

ENDF/B-VIII.1 N-14 gives MT600-810 as MF3 alone: no MF4 or MF6, so no product in
the model. Counting only stated products, N-14's MT203, 204 and 207 were zero
everywhere and MT205 was missing, where GASPR counts one proton per (n,p), one
alpha per (n,alpha) and so on from the MT.
"""
from __future__ import annotations

from types import SimpleNamespace

from kika.hdf5.processing import gasMultiplicities


def _reaction(mt, products=()):
    return SimpleNamespace(ENDF_MT=mt, outputChannel=SimpleNamespace(products=list(products)))


def _counts(reaction):
    return {gas: form.constant for gas, form in gasMultiplicities(reaction).items()}


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

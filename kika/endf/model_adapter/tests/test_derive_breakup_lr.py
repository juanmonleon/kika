"""MF3's LR survives the residual: ENDF → model (``residuals.py``) → LR (``derive``).

GNDS has no LR; it states the residual's breakup as a decay channel, and the
ENDF writer has to read the flag back off that channel. These pin the round trip
for every LR of ENDF-102's table, on the layout FUDGE writes — one product per
light particle with its multiplicity, then the leftover nucleus — including the
cases where FUDGE's own rule is wrong or needs a kludge.

No tape needed: none of the committed micro-tapes carries an LR, and the full
evaluations that do (B-10, C-12) are gated by the oracle
(``derive/oracle.py``) instead.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter.derive.reactions import LR_PRODUCTS, _lr
from kika.endf.model_adapter.residuals import _decayChannel
from kika.nuclear_data.model import ConversionReport, Product, pidFromZA

DOMAIN = (1e-5, 2e7)


def _residual(za: int, lr: int):
    report = ConversionReport()
    channel = _decayChannel(za, lr, 1.0e6, report, mt=51, domain=DOMAIN)
    assert channel is not None, report.losses
    return Product(pid=pidFromZA(za, 1), label=pidFromZA(za, 1), outputChannel=channel)


@pytest.mark.parametrize("lr", sorted(LR_PRODUCTS))
def test_every_lr_of_the_table_comes_back_from_a_heavy_residual(lr):
    """A residual heavy enough that the leftover is a nucleus, never a light particle."""
    residual = _residual(26056, lr)
    assert _lr(residual, "MT51", ConversionReport()) == lr


def test_a_breakup_is_one_product_per_light_particle_then_the_leftover():
    """FUDGE's layout, which its ``toENDF6`` reads too: He4 once, multiplicity 2."""
    channel = _residual(26056, 29).outputChannel
    pids = [p.pid for p in channel.products]
    assert pids == ["He4", "Ti48"]
    assert channel.products.products[0].multiplicity.form.constant == 2.0


def test_be8_to_alpha_plus_leftover_alpha_is_lr22():
    """B-10(n,t0) in ENDF/B-VIII.1: LR=22 on Be-8, whose leftover is itself an
    alpha. The second He4 is the leftover, not a second emitted alpha."""
    residual = _residual(4008, 22)
    assert [p.label for p in residual.outputChannel.products] == ["He4", "He4__a"]
    assert _lr(residual, "MT700", ConversionReport()) == 22


@pytest.mark.parametrize("za, lr", [(4008, 29), (6012, 23), (5010, 35), (5011, 36)])
def test_a_breakup_with_nothing_left_over_is_counted_whole(za, lr):
    """Be-8 → 2α, C-12 → 3α, B-10 → d + 2α, B-11 → t + 2α: no leftover nucleus.
    FUDGE gets the first two by kludge, the third by a kludge for "bad data", and
    the fourth wrong (33), because it always drops the last product."""
    residual = _residual(za, lr)
    assert _lr(residual, "MT51", ConversionReport()) == lr


def test_a_gamma_cascade_is_lr0():
    residual = _residual(26056, 0)
    assert [p.pid for p in residual.outputChannel.products] == ["Fe56", "photon"]
    assert _lr(residual, "MT51", ConversionReport()) == 0


def test_a_breakup_outside_the_table_is_refused():
    """D3: no LR is invented. FUDGE writes 0 here."""
    residual = _residual(26056, 22)
    residual.outputChannel.products.products[0].pid = "H3"
    residual.outputChannel.products.products[0].label = "H3"
    residual.outputChannel.products.products[0].multiplicity.form.constant = 3.0
    with pytest.raises(ValueError, match="no LR of ENDF-102's table"):
        _lr(residual, "MT51", ConversionReport())

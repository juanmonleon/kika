"""A reaction producing the same particle twice gives two products and two IXS entries.

n + He-3 -> d + d (MT650): NJOY writes MT650 twice in the deuteron's MTRH,
one entry per deuteron. ``decodeAce`` makes one product per entry (``H2`` and
``H2__a``, as FUDGE names a second product) with that entry's own yield and
distributions, and ``encodeAce`` writes one entry per product, so ACE ->
model -> ACE gives the table back word for word. Checked tape-free on
``_synthetic``'s ``particles=True`` table (MT51 producing two deuterons) and
on ENDF/B-VIII.1 He-3 as NJOY processed it, where the second deuteron used to
overwrite the first and the written table lost one of the two deuterons (6996
of its 36215 words).
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.ixs_keys import ixs_keys
from kika.ace.encoders.layout import encode_xss
from kika.ace.model_adapter import decodeAce
from kika.ace.model_adapter.encode import encodeAce
from kika.ace.tests._synthetic import DEUTERON_ENTRIES, write_table

MC = "MonteCarlo"


def _deuterons(suite, mt):
    return [p for p in suite.reactionByENDF_MT(mt).outputChannel.products if p.pid == "H2"]


def _yields(product):
    form = product.multiplicity[MC]
    return list(getattr(form, "ys", None) if hasattr(form, "ys") else form.coefficients)


def _same_table(original, again):
    a, b = encode_xss(original), encode_xss(again)
    assert list(a.nxs) == list(b.nxs) and list(a.jxs) == list(b.jxs)
    np.testing.assert_allclose(b.xss[1:], a.xss[1:], rtol=1e-12, atol=0)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return read_ace(str(write_table(tmp_path_factory.mktemp("ace") / "d.ace", particles=True)), strict=True)


def test_each_entry_is_a_product_of_its_own(synthetic):
    suite, _ = decodeAce(synthetic)
    first, second = _deuterons(suite, 51)
    assert (first.label, second.label) == ("H2", "H2__a")
    assert [_yields(first), _yields(second)] == [e["yields"] for e in DEUTERON_ENTRIES]
    levels = [p.distribution[MC].weighted[0].functional.energy for p in (first, second)]
    assert [lv.threshold for lv in levels] == pytest.approx(
        [DEUTERON_ENTRIES[0]["law33"][0] * 1e6, DEUTERON_ENTRIES[1]["law33"][0] * 1e6])


def test_the_synthetic_table_comes_back_word_for_word(synthetic):
    suite, _ = decodeAce(synthetic)
    again, _ = encodeAce(suite)
    assert [int(m) for m in again.reaction_mt_data.particle_production[0]] == [51, 51]
    _same_table(synthetic, again)


@pytest.fixture(scope="module")
def he3(he3_ace_b81_tape):
    return read_ace(str(he3_ace_b81_tape), strict=True)


@pytest.mark.tape
def test_he3_reads_both_deuterons_of_mt650(he3):
    i = list(he3.secondary_particle_types.particle_ids).index(31)
    mtrh = he3.reaction_mt_data.particle_production[i]
    assert [int(m) for m in mtrh] == [650, 650]
    keys = ixs_keys(mtrh)
    sigh = he3.particle_production_xs.particle_cross_sections[i + 1]
    andh = he3.angular_distributions.particle_production[i]
    dlwh = he3.energy_distributions.particle_production[i]
    assert list(sigh) == list(andh) == list(dlwh) == keys
    # Each read from its own locators. The evaluation gives both deuterons the
    # same data, so here it is the count that was lost; the synthetic table
    # above is the one whose two entries differ.
    assert len(set(int(v) for v in he3.angular_locators.particle_production[i])) == 2
    assert andh[keys[0]] is not andh[keys[1]] and dlwh[keys[0]] is not dlwh[keys[1]]


@pytest.mark.tape
def test_he3_comes_back_word_for_word(he3):
    suite, _ = decodeAce(he3)
    assert [p.label for p in _deuterons(suite, 650)] == ["H2", "H2__a"]
    again, _ = encodeAce(suite)
    assert encode_xss(again).nxs[1] == he3.header.nxs_array[1] == 36215
    _same_table(he3, again)

"""``InputDeck`` never gives an automatic tape a unit the caller already named.

``moder(nout=-21)`` followed by ``reconr()`` used to give RECONR -21 as both its
input and its output, and ERRORR after it the same tape as ENDF and PENDF.
"""
from kika.njoy.deck import InputDeck


def _cards(deck, module):
    lines = deck.render().splitlines()
    return lines[lines.index(module) + 1].split()


def test_reconr_does_not_reuse_the_unit_moder_was_given():
    deck = InputDeck()
    deck.moder(nin=20, nout=-21)
    deck.reconr(isotope="Fe56")
    deck.errorr(isotope="Fe56", nout=26, egn=[1e-5, 1.0, 2e7])
    assert _cards(deck, "reconr")[:2] == ["-21", "-22"]
    assert _cards(deck, "errorr")[:4] == ["-21", "-22", "0", "26"]


def test_a_named_unit_further_down_is_skipped_too():
    deck = InputDeck()
    deck.moder(nin=20, nout=-22)
    deck.reconr(isotope="Fe56")                 # -21 is free
    deck.broadr(isotope="Fe56", temperatures=[293.6])
    units = {abs(int(u)) for u in _cards(deck, "reconr")[:2] + _cards(deck, "broadr")[:3]}
    assert units == {21, 22, 23}


def test_the_automatic_chain_is_unchanged():
    deck = InputDeck()
    deck.moder(nin=20)
    deck.reconr(isotope="Fe56")
    assert _cards(deck, "moder") == ["20", "-21"]
    assert _cards(deck, "reconr")[:2] == ["-21", "-22"]

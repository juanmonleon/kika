"""THERMR through NJOY for a TSL tape (kika.tsl.thermr).

The deck and the listing check run anywhere; the end-to-end run needs NJOY
and the tapes, and is the ``njoy`` fixture's (skipped, or failed under
``--deep``, where there is none).
"""
from __future__ import annotations

from kika.njoy.locale_guard import parse_broadr_card, parse_reconr_card
from kika.tsl.thermr import _thermr_read, thermr_deck


def test_the_deck_sends_every_tolerance_the_locale_guard_checks():
    deck = thermr_deck(34, 125, [22.0], icoh=1, natom=1)
    assert parse_reconr_card(deck) == (0.001, 0.01, 5e-08)
    assert parse_broadr_card(deck) == (0.001, 0.01, 5e-08)
    assert " 34 125 16 1 2 1 0 1 222 0/" in deck.splitlines()
    assert " 0.001 4.0/" in deck.splitlines()


def test_a_tape_without_elastic_and_two_principal_atoms():
    deck = thermr_deck(1, 125, [293.6, 600.0], icoh=0, natom=2, emax=5.0)
    assert " 1 125 16 2 2 0 0 2 222 0/" in deck.splitlines()
    assert deck.splitlines().count(" 293.6 600/") == 2       # broadr and thermr


def test_a_misread_thermr_tolerance_is_caught():
    listing = ("thermr...compute thermal scattering cross sections and distributions\n"
               " tolerance ............................ {}\n")
    assert _thermr_read(listing.format("1.0000E-03"), 0.001)
    assert not _thermr_read(listing.format("0.0000E+00"), 0.001)

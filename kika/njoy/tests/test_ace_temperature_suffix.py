"""The ZAID suffix of a table follows its temperature, and a temperature with none raises.

``K_TO_SUFFIX`` gives kika's ACE tables their suffix (``.02c`` at 293.6 K),
whether NJOY or kika's own encoder writes them. A temperature that is not in
it used to fall back to ``.00`` in the NJOY runs, which names a table by a
temperature it is not at; now it raises and asks for ``extension=``.
"""
import pytest

from kika._utils import ace_temperature_suffix
from kika.njoy.run_njoy import _get_temperature_info, _render_njoy_input


def test_a_tabulated_temperature_has_its_suffix():
    assert ace_temperature_suffix(293.6) == ".02"
    assert ace_temperature_suffix(600.0) == ".06"


def test_within_one_kelvin_is_the_same_temperature():
    """kT/k_B of an ACE header is 293.59 K, not 293.6 K."""
    assert ace_temperature_suffix(293.59) == ".02"


def test_a_temperature_with_no_suffix_raises():
    with pytest.raises(ValueError, match="extension"):
        ace_temperature_suffix(350.0)


def test_the_njoy_deck_refuses_a_temperature_with_no_suffix():
    with pytest.raises(ValueError, match="extension"):
        _render_njoy_input(2631, 350.0, "title")


def test_the_njoy_deck_takes_any_temperature_with_an_explicit_suffix():
    assert ".12" in _render_njoy_input(2631, 350.0, "title", suff=".12")
    assert _get_temperature_info(350.0) == (350.0, None)

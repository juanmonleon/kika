import pytest

from kika._constants import ATOMIC_MASS, NATURAL_ABUNDANCE
from kika.materials import Material


def test_uranium_masses_are_the_ame2020_values():
    assert ATOMIC_MASS[92235] == pytest.approx(235.0439281, abs=1e-6)
    assert ATOMIC_MASS[92238] == pytest.approx(238.0507869, abs=1e-6)
    assert ATOMIC_MASS[92234] == pytest.approx(234.0409503, abs=1e-6)


def test_every_naturally_abundant_isotope_has_a_mass_and_abundances_sum_to_one():
    for element, isotopes in NATURAL_ABUNDANCE.items():
        assert sum(isotopes.values()) == pytest.approx(1.0, abs=1e-4), element
        for za in isotopes:
            assert za // 1000 == element // 1000, (element, za)
            assert za in ATOMIC_MASS, za


@pytest.mark.parametrize("element, isotopes", [
    ("Rh", ["Rh103"]),
    ("La", ["La138", "La139"]),
    ("Sn", ["Sn112", "Sn114", "Sn115", "Sn116", "Sn117", "Sn118", "Sn119", "Sn120", "Sn122", "Sn124"]),
])
def test_natural_elements_expand_into_their_stable_isotopes(element, isotopes):
    material = Material(id=1)
    material.add_element(element, 1.0)
    material.expand_natural_elements()
    assert sorted(material.nuclide.keys()) == sorted(isotopes)


def test_weight_fractions_of_uranium_use_its_real_mass():
    material = Material(id=1)
    material.add_nuclide("U235", 0.5, "ao")
    material.add_nuclide("U238", 0.5, "ao")
    material.to_weight_fraction()
    expected = 235.0439281 / (235.0439281 + 238.0507869)
    assert material.nuclide["U235"].fraction == pytest.approx(expected, rel=1e-8)

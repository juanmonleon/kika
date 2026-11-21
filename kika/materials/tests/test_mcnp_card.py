import pytest

from kika.materials import Material, read_material


def test_a_nuclide_with_neutron_and_photon_libraries_is_listed_once():
    material = Material(id=1)
    material.add_nuclide("Fe56", 1.0, "ao", library="80c")
    material.nuclide["Fe56"].libs["plib"] = "12p"
    with pytest.warns(UserWarning, match="12p"):
        card = material.to_mcnp()
    entries = [line for line in card.splitlines() if "26056" in line]
    assert len(entries) == 1
    assert "26056.80c" in entries[0]


def test_adding_a_nuclide_of_the_other_fraction_type_is_refused():
    material = Material(id=1)
    material.add_nuclide("H1", 2.0, "ao")
    with pytest.raises(ValueError, match="Convert the material first"):
        material.add_nuclide("O16", 0.888, "wo")
    assert material.fraction_type == "ao"
    assert "O16" not in material.nuclide


def test_the_first_nuclide_sets_the_fraction_type_and_can_be_updated():
    material = Material(id=1)
    material.add_nuclide("O16", 1.0, "wo")
    assert material.fraction_type == "wo"
    material.add_nuclide("O16", 1.0, "ao")
    assert material.fraction_type == "ao"


@pytest.mark.parametrize("suffix, key", [
    ("80c", "nlib"), ("710nc", "nlib"), ("12p", "plib"), ("70u", "pnlib"),
    ("03e", "elib"), ("70h", "hlib"), ("24y", "ylib"),
])
def test_library_suffix_class_picks_its_keyword(suffix, key):
    material = Material(id=1)
    material.add_nuclide("H1", 1.0, "ao", library=suffix)
    assert material.nuclide["H1"].libs == {key: suffix}


def test_an_unknown_library_class_is_refused_not_dropped():
    material = Material(id=1)
    with pytest.raises(ValueError, match="20t"):
        material.add_nuclide("H1", 1.0, "ao", library="20t")


def test_an_uppercase_material_card_is_read():
    material, end = read_material(["M3 26056.80c 1", "     26054.80c 0.06", "NPS 1"], 0)
    assert material is not None and material.id == 3
    assert sorted(material.nuclide.keys()) == ["Fe54", "Fe56"]
    assert end == 2

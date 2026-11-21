import pytest

from kika.materials import Material, read_material


def test_neutron_and_photon_entries_of_a_nuclide_round_trip_as_one_nuclide():
    card = ["m1 nlib=06c plib=02p", "\t6012  9.126127e-03", "\t6013.03c  1.022318e-04",
            "\t6013.04p  1.022318e-04"]
    material, _ = read_material(card, 0)
    assert material.nuclide["C13"].libs == {"nlib": "03c", "plib": "04p"}
    entries = [line.split()[0] for line in material.to_mcnp().splitlines() if "6013" in line]
    assert entries == ["6013.03c", "6013.04p"]


@pytest.mark.parametrize("card, message", [
    ("m1 26056.80c 0.5 26056.80c 0.5", "more than once"),
    ("m1 26056 0.5 26056.12p 0.5", "more than once"),
    ("m1 26056.80c 0.5 26056.12p 0.4", "one amount per nuclide"),
])
def test_a_nuclide_listed_twice_for_the_same_data_is_refused(card, message):
    with pytest.raises(ValueError, match=message):
        read_material([card], 0)


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

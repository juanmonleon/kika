import json

import pytest

from kika._constants import N_AVOGADRO
from kika.materials import Material, MaterialCollection, read_material, read_serpent_materials


def test_a_nuclide_says_its_z_a_and_isomeric_state():
    material, _ = read_material(["m1 95642.80c 1 26000.80c 1"], 0)
    am = material.nuclide["Am242m"]
    assert (am.z, am.a, am.isomer, am.is_natural) == (95, 242, 1, False)
    fe = material.nuclide["Fe"]
    assert (fe.z, fe.a, fe.isomer, fe.is_natural) == (26, 0, 0, True)


def test_reprs_fit_on_one_line():
    material, _ = read_material(["c KIKA_MAT_NAME: water", "m1 1001.80c 2 8016.80c 1"], 1)
    material.set_density(1.0, "g/cc")
    assert repr(material) == "Material(1 'water': 2 nuclides, ao, 1 g/cc)"
    assert repr(material.nuclide["H1"]) == "Nuclide(H1, zaid=1001, fraction=2, libs={'nlib': '80c'})"
    assert "\n" not in repr(MaterialCollection({1: material}))
    assert "<table>" in material._repr_html_()


def test_to_dict_is_json_and_from_dict_gives_the_material_back():
    serpent = read_serpent_materials(
        ["mat fuel -10.4 rgb 255 0 0 tft 600 1200 moder lwtr 1001 burn 1", "92235.09c -0.04 92238.09c -0.96"]
    ).by_name["fuel"]
    mcnp, _ = read_material(["m2 1001.80c 2 8016.80c 1 gas=1 hlib=70h"], 0)
    mcnp.thermal_scattering = ["lwtr.20t"]
    for material in (serpent, mcnp):
        data = json.loads(json.dumps(material.to_dict()))
        again = Material.from_dict(data)
        assert again.to_mcnp() == material.to_mcnp()
        assert again.to_serpent() == material.to_serpent()


def test_a_collection_is_indexed_by_id_or_name():
    collection = read_serpent_materials(["mat fuel -10.0", "92235.09c 1", "mat clad -6.5", "40090.09c 1"])
    assert collection[2] is collection["clad"]
    assert "fuel" in collection and 1 in collection
    assert "water" not in collection and 7 not in collection


def test_avogadro_is_the_si_value_and_drives_the_atomic_density():
    assert N_AVOGADRO == 6.02214076e23
    material = Material(id=1)
    material.add_nuclide("C12", 1.0, "ao")
    material.set_density(12.0, "g/cc")
    assert material.density_in("atoms/b-cm") == pytest.approx(0.602214076, rel=1e-12)


def test_a_natural_element_has_its_mean_mass_in_a_perturbation():
    from kika.mcnp.pert_generator import perturb_material

    carbon, _ = read_material(["m1 6000.80c 1 1001.80c 1"], 0)
    collection = MaterialCollection({1: carbon})
    perturbed = perturb_material(collection, 1, 1001, density=-1.0, pert_mat_id=2, in_place=False)
    info = perturbed.by_id[2].metadata["density_info"]
    mean_mass = (12.0107 + 1.00794) / 2
    assert info["original_atomic_density"] == pytest.approx(N_AVOGADRO / mean_mass * 1e-24, rel=1e-4)

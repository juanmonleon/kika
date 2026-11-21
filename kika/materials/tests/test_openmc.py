import pytest

from kika.materials import Material, MaterialCollection, read_material
from kika.materials.openmc import read_openmc_materials


def steel():
    material, _ = read_material(["m1 26000.80c -0.98 6000.80c -0.02 95642.80c -1e-9"], 0)
    material.name = "steel"
    material.set_density(7.85, "g/cc")
    material.set_temperature(600.0)
    return material


def test_an_openmc_material_names_nuclides_as_openmc_and_splits_natural_elements():
    text = steel().to_openmc()
    assert '<material id="1" name="steel" temperature="600.0">' in text
    assert '<density units="g/cm3" value="7.85" />' in text
    assert 'name="Am242_m1" wo="1e-09"' in text
    for name in ("Fe54", "Fe56", "Fe57", "Fe58", "C12", "C13"):
        assert f'name="{name}"' in text
    assert 'name="Fe"' not in text


def test_a_weight_fraction_splits_by_mass_and_leaves_the_others_alone():
    material = steel()
    material.expand_natural_elements()
    assert material.nuclide["Am242m"].fraction == 1e-9
    iron = sum(material.nuclide[s].fraction for s in ("Fe54", "Fe56", "Fe57", "Fe58"))
    assert iron == pytest.approx(0.98, rel=1e-15)


def test_an_isotope_already_in_the_material_gets_its_share_added():
    material = Material(id=1)
    material.add_nuclide("Fe", 1.0, "ao")
    material.add_nuclide("Fe56", 0.5, "ao")
    material.expand_natural_elements()
    assert material.nuclide["Fe56"].fraction == pytest.approx(0.5 + 0.91754, rel=1e-6)


def test_the_document_reads_back():
    collection = MaterialCollection({1: steel()})
    water = Material(id=2, name="water")
    water.add_nuclide("H1", 2.0, "ao")
    water.add_nuclide("O16", 1.0, "ao")
    water.set_density(0.1, "atoms/b-cm")
    water.thermal_scattering = ["c_H_in_H2O"]
    water.metadata["vol"] = 12.5
    collection.add_material(water)
    document = collection.to_openmc(cross_sections="/data/cross_sections.xml")
    assert "<cross_sections>/data/cross_sections.xml</cross_sections>" in document
    back = read_openmc_materials(document)
    assert back[2].thermal_scattering == ["c_H_in_H2O"]
    assert (back[2].density, back[2].density_unit, back[2].metadata["vol"]) == (0.1, "atoms/b-cm", 12.5)
    assert back["steel"].nuclide["Am242m"].fraction == 1e-9


def test_reading_converts_atom_per_cm3_and_refuses_what_kika_does_not_keep():
    text = ("<materials><material id='3'><density units='atom/cm3' value='1e23'/>"
            "<nuclide name='H1' ao='1'/><sab name='c_H_in_H2O' fraction='0.5'/></material></materials>")
    material = read_openmc_materials(text)[3]
    assert material.density == pytest.approx(0.1) and material.density_unit == "atoms/b-cm"
    assert any("0.5" in loss for loss in material.report.losses)
    for bad in ("<density units='macro' value='1'/>", "<density units='g/cm3' value='1'/><element name='Fe' ao='1'/>"):
        with pytest.raises(ValueError):
            read_openmc_materials(f"<materials><material id='4'>{bad}</material></materials>")


def test_what_openmc_cannot_hold_is_listed_and_the_library_is_checked(tmp_path):
    material = steel()
    material.thermal_scattering = ["lwtr.20t"]
    losses = " | ".join(material.emission_losses("openmc"))
    assert "natural C, Fe" in losses and "library suffixes" in losses and "lwtr.20t" in losses
    index = tmp_path / "cross_sections.xml"
    index.write_text("<?xml version='1.0'?><cross_sections>"
                     "<library materials='Fe54 Fe56 Fe57 Fe58 C12 C13' path='x.h5' type='neutron'/>"
                     "</cross_sections>")
    report = MaterialCollection({1: material}).emission_report("openmc", library=index)
    assert report.materials[0]["missing"] == [["Am242m", "Am242_m1"]]
    assert report.status == "Failed"

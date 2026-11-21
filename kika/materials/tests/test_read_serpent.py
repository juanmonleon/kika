import pytest

from kika.materials import MaterialCollection, read_serpent_materials


def test_a_card_that_follows_a_material_is_not_read_into_it():
    lines = ["mat fuel -10.0", "92235.09c 0.05", "92238.09c 0.95",
             "mflow feed", "94239 1.0e-3", "mat clad -6.5", "40090.09c 1.0"]
    collection = read_serpent_materials(lines)
    assert sorted(collection.by_name["fuel"].nuclide.keys()) == ["U235", "U238"]
    assert list(collection.by_name["clad"].nuclide.keys()) == ["Zr90"]


def test_symbolic_nuclide_names_are_read():
    collection = read_serpent_materials(["mat water -1.0 moder lwtr 1001", "H-1.09c 2 O-16.09c 1"])
    water = collection.by_name["water"]
    assert sorted(water.nuclide.keys()) == ["H1", "O16"]
    assert water.nuclide["H1"].libs == {"nlib": "09c"}
    assert water.metadata["moder"] == [("lwtr", 1001)]


def test_options_and_nuclides_may_span_lines_and_carry_comments():
    lines = ["mat fuel -10.4 % UO2", "   tmp 900 burn 1", "   vol 3.14", "",
             "92235.09c -0.04 92238.09c -0.84 % uranium", "8016.09c -0.12",
             "% end of the fuel", "set pop 1000 10 5"]
    fuel = read_serpent_materials(lines).by_name["fuel"]
    assert fuel.temperature == 900.0
    assert fuel.metadata["burn"] == 1 and fuel.metadata["vol"] == 3.14
    assert fuel.fraction_type == "wo" and len(fuel.nuclide) == 3


def test_a_material_without_nuclides_is_refused():
    with pytest.raises(ValueError, match="no nuclides"):
        read_serpent_materials(["mat fuel -10.0 pdepx 1", "92235.09c 1"])


def test_options_that_are_not_written_back_are_reported():
    fuel = read_serpent_materials(["mat fuel -10.0 coefs 2", "92235.09c 1"]).by_name["fuel"]
    assert any("coefs" in loss for loss in fuel.report.losses)


def test_write_to_serpent_keeps_what_follows_the_material(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"mat fuel -10.0\r\n92235.09c 1\r\n% next: the geometry\r\n\r\nsurf 1 sph 0 0 0 10\r\n")
    collection = MaterialCollection.from_serpent(str(path))
    collection.by_name["fuel"].update_nuclide_fraction("U235", 0.9)
    collection.write_to_serpent(str(path))
    data = path.read_bytes()
    assert b"% next: the geometry\r\n\r\nsurf 1 sph 0 0 0 10\r\n" in data
    assert data.count(b"\n") == data.count(b"\r\n")


def test_block_comments_mass_and_a_card_split_across_lines():
    lines = ["/* fuel pin", "mat fuel -1.0 */", "mat fuel", "  -10.4 mass 12.5 % grams",
             "92235.09c /* enriched */ -0.04", "92238.09c -0.96"]
    fuel = read_serpent_materials(lines).by_name["fuel"]
    assert fuel.metadata["mass"] == 12.5 and fuel.density == 10.4
    assert sorted(fuel.nuclide.keys()) == ["U235", "U238"]
    assert " mass 12.5" in fuel.to_serpent()


def test_write_to_serpent_finds_a_material_by_its_card_name(tmp_path):
    from kika.materials import Material

    path = tmp_path / "input"
    path.write_text("mat cool_ant -1.0\n1001.09c 1\n")
    material = Material(id=1, name="cool ant", density=1.0, density_unit="g/cc")
    material.add_nuclide("H1", 2.0, "ao", library="09c")
    MaterialCollection({1: material}).write_to_serpent(str(path))
    assert "1001.09c" in path.read_text() and "2.0" in path.read_text()


def test_setting_a_density_replaces_sum():
    material = read_serpent_materials(["mat f sum", "1001.09c 1"]).by_name["f"]
    material.set_density(0.5, "g/cc")
    assert material.to_serpent().startswith("mat f -0.5")

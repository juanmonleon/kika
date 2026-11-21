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

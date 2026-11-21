import pytest

from kika.materials import Material, MaterialCollection, read_material, read_serpent_materials

XSDIR = """datapath=/data
atomic weight ratios
1001 0.99916
directory
1001.80c 0.999167 xdata/endf80/H/1001.800nc 0 1 1 12345 0 0 2.5301E-08
8016.80c 15.857510 xdata/endf80/O/8016.800nc 0 1 1 54321 0 0 2.5301E-08 +
 ptable
lwtr.20t 0.999167 xdata/endf80sab/lwtr 0 1 1 999 0 0 2.5301E-08
"""


def water(**changes):
    material, _ = read_material(["m1 1001.80c 0.6666666666666666 8016.80c 0.3333333333333333"], 0)
    material.thermal_scattering = ["lwtr.20t"]
    material.set_density(0.99820, "g/cc")
    for key, value in changes.items():
        setattr(material, key, value)
    return material


def test_cards_carry_every_fraction_exactly():
    material = water()
    material.to_weight_fraction()
    for back in (read_material(material.to_mcnp().splitlines(), 0)[0],
                 read_serpent_materials(material.to_serpent().splitlines()).by_id[1]):
        assert {s: n.fraction for s, n in back.nuclide.items()} == \
               {s: n.fraction for s, n in material.nuclide.items()}


def test_a_card_that_does_not_read_back_is_refused(monkeypatch):
    import kika.materials.parse_mcnp as parse_mcnp

    real = parse_mcnp.read_material

    def lossy(lines, start):
        material, end = real(lines, start)
        del material.nuclide["O16"]
        return material, end

    monkeypatch.setattr(parse_mcnp, "read_material", lossy)
    with pytest.raises(RuntimeError, match="defect in kika"):
        water().to_mcnp()


def test_a_serpent_card_needs_a_density():
    with pytest.raises(ValueError, match="no density"):
        water(density=None, density_unit=None).to_serpent()


def test_what_a_serpent_card_leaves_out_is_listed():
    material = water(keywords={"gas": "1"})
    material.nuclide["H1"].libs["plib"] = "84p"
    losses = " | ".join(material.emission_losses("serpent"))
    assert "gas" in losses and "lwtr.20t" in losses and "plib" in losses
    assert water().emission_losses("mcnp") == []


def test_the_report_finds_the_tables_the_library_lacks(tmp_path):
    xsdir = tmp_path / "xsdir"
    xsdir.write_text(XSDIR)
    collection = MaterialCollection({1: water()})
    steel, _ = read_material(["m2 26056.80c 1 6000 0.01"], 0)
    collection.add_material(steel)
    report = collection.emission_report("mcnp", library=xsdir)
    by_id = {m["id"]: m for m in report.materials}
    assert by_id[1]["status"] == "OK"
    assert by_id[2]["status"] == "Failed"
    assert by_id[2]["missing"] == [["Fe56", "26056.80c"]]
    assert by_id[2]["no_library"] == ["C"]
    assert report.status == "Failed"
    page = report.to_html()
    assert page.startswith("<!DOCTYPE html>") and "26056.80c" in page
    assert report.save(tmp_path / "report.html").read_text(encoding="utf-8") == page


def test_the_report_reads_a_serpent_xsdata_and_finds_name_clashes(tmp_path):
    xsdata = tmp_path / "xsdata"
    xsdata.write_text("1001.09c 1001.09c 1 1001 0 1.0 300 0 /data/h1\n")
    collection = MaterialCollection({1: water(name="cool ant"), 2: water(id=2, name="cool_ant")})
    report = collection.emission_report("serpent", library=xsdata)
    assert report.clashes == ["cool_ant (materials 1, 2)"]
    assert all(m["status"] == "Failed" for m in report.materials)
    with pytest.raises(ValueError, match="cool_ant"):
        collection.to_serpent()

"""``cross_sections.xml`` read and written as OpenMC writes it, and the overlay."""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.hdf5 import library as L
from kika.hdf5.exceptions import HDF5FormatError, NuclideNotFoundError

DATA = Path(__file__).parent / "data"

OFFICIAL_STYLE = """<?xml version='1.0' encoding='UTF-8'?>
<cross_sections>
  <library materials="H2" path="neutron/H2.h5" type="neutron"/>
  <library materials="c_H_in_H2O" path="thermal/c_H_in_H2O.h5" type="thermal"/>
  <library materials="Fe" path="photon/Fe.h5" type="photon"/>
</cross_sections>
"""


def test_read_write_is_the_same_text(tmp_path):
    source = tmp_path / "in" / "cross_sections.xml"
    source.parent.mkdir()
    source.write_text(OFFICIAL_STYLE, newline="\n")
    index = L.CrossSectionsXml.read(source)
    assert [e.type for e in index.entries] == ["neutron", "thermal", "photon"]
    out = index.write(tmp_path / "out.xml")
    assert out.read_bytes() == source.read_bytes()


def test_directory_and_depletion_chain_round_trip(tmp_path):
    index = L.CrossSectionsXml(entries=[L.LibraryEntry(("U235",), "U235.h5", "neutron")],
                               directory="/data/lib", depletionChain="chain.xml")
    index.write(tmp_path / "x.xml")
    again = L.CrossSectionsXml.read(tmp_path / "x.xml")
    assert again.directory == "/data/lib"
    assert again.depletionChain == "chain.xml"
    assert again.resolve(again.entries[0]) == Path("/data/lib/U235.h5")


def test_register_reads_type_and_materials_off_the_file():
    entry = L.registerFile(DATA / "U235.h5", relativeTo=DATA)
    assert entry.materials == ("U235",)
    assert entry.type == "neutron"
    assert entry.path == "U235.h5"
    assert L.registerFile(DATA / "c_H_in_H2O.h5").type == "thermal"
    assert L.registerFile(DATA / "Fe.h5").type == "photon"


def test_lookup_is_exact(tmp_path):
    index = L.CrossSectionsXml(entries=[L.registerFile(p, DATA) for p in sorted(DATA.glob("*.h5"))])
    index.write(DATA.parent / "_tmp_cross_sections.xml")
    try:
        library = L.open(DATA.parent / "_tmp_cross_sections.xml")
        assert library.path("Fe56").name == "Fe56.h5"
        assert library.path("Fe", "photon").name == "Fe.h5"
        with pytest.raises(NuclideNotFoundError):
            library.path("Fe57")
        with pytest.raises(NuclideNotFoundError):
            library.path("Fe56", "thermal")
    finally:
        (DATA.parent / "_tmp_cross_sections.xml").unlink()


def test_overlay_swaps_one_material_and_points_at_the_rest(tmp_path):
    base = tmp_path / "base"
    (base / "neutron").mkdir(parents=True)
    for name in ("Fe56", "U235", "H2"):
        (base / "neutron" / f"{name}.h5").write_bytes((DATA / f"{name}.h5").read_bytes())
    L.CrossSectionsXml(entries=[L.registerFile(p, base) for p in sorted((base / "neutron").glob("*.h5"))]
                       ).write(base / "cross_sections.xml")
    mine = tmp_path / "kika" / "U235.h5"
    mine.parent.mkdir()
    mine.write_bytes((DATA / "U235.h5").read_bytes())

    out = L.overlay(base, {"U235": mine}, tmp_path / "run" / "cross_sections.xml")
    again = L.open(tmp_path / "run")
    assert again.path("U235").resolve() == mine.resolve()
    assert again.path("Fe56").resolve() == (base / "neutron" / "Fe56.h5").resolve()
    assert [e.materials for e in out.entries] == [("Fe56",), ("H2",), ("U235",)]
    assert all(not Path(e.path).is_absolute() for e in out.entries)


def test_overlay_refuses_a_file_whose_group_is_another_material(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    L.CrossSectionsXml(entries=[]).write(base / "cross_sections.xml")
    with pytest.raises(HDF5FormatError):
        L.overlay(base, {"U238": DATA / "U235.h5"}, tmp_path / "o.xml")


def test_describe_lists_nuclides_from_the_index(tmp_path):
    (tmp_path / "neutron").mkdir()
    for name in ("Fe56", "Am242_m1", "C0"):
        (tmp_path / "neutron" / f"{name}.h5").write_bytes(b"x" * 10)
    index = L.CrossSectionsXml(entries=[
        L.LibraryEntry(("Fe56",), "neutron/Fe56.h5", "neutron"),
        L.LibraryEntry(("Am242_m1",), "neutron/Am242_m1.h5", "neutron"),
        L.LibraryEntry(("C0",), "neutron/C0.h5", "neutron"),
        L.LibraryEntry(("U235",), "neutron/U235.h5", "neutron"),          # not on disk
        L.LibraryEntry(("c_H_in_H2O",), "thermal/c_H_in_H2O.h5", "thermal"),
    ])
    index.write(tmp_path / "cross_sections.xml")
    info = L.open(tmp_path).describe()
    assert [(n["target"], n["Z"], n["A"], n["M"]) for n in info["isotopes"]] == [
        ("C0", 6, None, 0), ("Fe56", 26, 56, 0), ("Am242_m1", 95, 242, 1)]
    assert info["missing"] == 1
    assert info["thermal"] == ["c_H_in_H2O"]
    assert info["name"] == tmp_path.name and info["isotopes"][1]["size"] == 10


def test_a_single_h5_is_a_library_of_one():
    pytest.importorskip("h5py")
    library = L.open(DATA / "Fe56.h5")
    assert library.materials() == ["Fe56"]
    assert library.path("Fe56") == DATA / "Fe56.h5"
    info = library.describe()
    assert info["single_file"] and info["name"] == "Fe56"
    assert [n["target"] for n in info["isotopes"]] == ["Fe56"]
    assert library.read("Fe56").target == "Fe56"

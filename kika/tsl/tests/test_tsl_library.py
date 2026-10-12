"""A TSL library indexed by compound, from headers (kika.tsl).

The small library is built in a temporary directory from the micro tapes in
``kika/endf/tests/data`` and the LEAPR / FLASSH inputs in
``kika/njoy/tests/data``, renamed the way NNDC names them. The whole of
ENDF/B-VIII.1's ``thermal_scatt`` is reached under ``--deep``: before this
module, ``index_endf`` dropped 21 of its 114 tapes as duplicates of nonsense
nuclides decoded from their pseudo-ZA.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kika.library_index import index_endf
from kika.tsl import companion_kind, index_tsl, read_tsl_header, tsl_naming

ENDF_DATA = Path(__file__).parents[2] / "endf" / "tests" / "data"
NJOY_DATA = Path(__file__).parents[2] / "njoy" / "tests" / "data"


# ---------------------------------------------------------------- naming

@pytest.mark.parametrize("file_name, principal, compound, variant", [
    ("tsl-HinH2O.endf", "H", "H2O", None),
    ("tsl-OinUO2-HALEU.endf", "O", "UO2", "HALEU"),
    ("tsl-UinUN-100P.endf", "U", "UN", "100P"),
    ("tsl-H1inCaH2.endf", "H1", "CaH2", None),
    ("tsl-7Liin7LiD-mixed.endf", "7Li", "7LiD", "mixed"),
    ("tsl-SiinSiO2-alpha.endf", "Si", "SiO2", "alpha"),
    ("tsl-SiO2-beta.endf", None, "SiO2", "beta"),
    ("tsl-013_Al_027.endf", "Al", "Al metal", None),
    ("tsl-U-metal-HEU.endf", "U", "U metal", "HEU"),
    ("tsl-Be-metal+Sd.endf", "Be", "Be metal", "Sd"),
    ("tsl-reactor-graphite-10P.endf", "C", "Graphite", "reactor 10P"),
    ("tsl-crystalline-graphite.endf", "C", "Graphite", "crystalline"),
    ("tsl-graphiteSd.endf", "C", "Graphite", "Sd"),
    ("tsl-ortho-D.endf", "D", "D2", "ortho"),
    ("tsl-s-CH4.endf", "H", "CH4", "solid"),
    ("tsl_Be_BeO.txt", "Be", "BeO", None),          # JEFF-4.0
    ("tsl_4-Be.txt", "Be", "Be metal", None),       # JEFF-4.0
])
def test_a_tape_is_placed_in_its_compound_by_its_file_name(file_name, principal, compound, variant):
    naming = tsl_naming(file_name)
    assert (naming.principal, naming.compound, naming.variant) == (principal, compound, variant)
    assert naming.source == "filename"


def test_four_enrichments_that_share_one_zsymam_stay_four_tapes():
    # ENDF/B-VIII.1 writes ZSYMAM 'U(UO2)' in natural, -5P, -10P and -100P alike.
    variants = {tsl_naming(f"tsl-UinUO2{s}.endf", "U(UO2)").variant for s in ("", "-5P", "-10P", "-100P")}
    assert variants == {None, "5P", "10P", "100P"}


def test_zsymam_names_a_tape_whose_file_name_says_nothing():
    naming = tsl_naming("material-7.endf", " H(H2O)")
    assert (naming.principal, naming.compound, naming.source) == ("H", "H2O", "zsymam")


@pytest.mark.parametrize("file_name, kind", [
    ("tsl-HinH2O.leapr", "leapr"),
    ("tsl-ZrinZrC_flassh-DOS.txt", "flassh-dos"),
    ("tsl-7Liin7LiD-mixed-DOS.txt", "flassh-dos"),
    ("tsl-OinUO2_flassh-Control.txt", "flassh-control"),
    ("tsl-BeinBe2C-Crystal.txt", "flassh-crystal"),
    ("tsl-HinZrH2_flassh-header.txt", "flassh-header"),
    ("tsl-H2O.readme", "readme"),
    ("tsl-HinH2O-proc.njoy", "njoy"),
    ("tsl-HinH2O.endf", "other"),
])
def test_companion_kinds(file_name, kind):
    assert companion_kind(file_name) == kind


# ---------------------------------------------------------------- one tape

def test_the_header_of_a_coherent_elastic_tape():
    tape = read_tsl_header(ENDF_DATA / "micro_tsl_bemetal_elastic.endf")
    assert (tape.mat, tape.nsub, tape.zsymam, tape.elastic) == (26, 12, "Be-Metal", "coherent")
    assert tape.elastic_temperatures[:3] == [77.0, 100.0, 293.6]
    assert len(tape.composition) == 1 and tape.composition[0]["zai"] == 4009
    assert tape.lab == "LEIP LAB" and tape.problems == []


def test_the_header_of_an_inelastic_tape_stops_after_its_first_beta():
    tape = read_tsl_header(ENDF_DATA / "micro_tsl_sch4.endf")
    assert tape.elastic == "incoherent"
    assert (tape.temperatures, tape.n_alpha, tape.n_beta, tape.lat) == ([22.0], 70, 80, 0)
    assert tape.secondaries == [dict(law="free-gas", free_xs=4.7392, awr=11.898, atoms=1.0)]


def test_mixed_elastic_and_the_jeff_dialect():
    assert read_tsl_header(ENDF_DATA / "micro_tsl_un_elastic.endf").elastic == "both"
    jeff = read_tsl_header(ENDF_DATA / "micro_tsl_jeff_be_elastic.endf")
    assert (jeff.za, jeff.elastic, jeff.problems) == (4000.0, "coherent", [])


# ---------------------------------------------------------------- a library

@pytest.fixture
def small_library(tmp_path):
    copies = {
        "micro_tsl_bemetal_elastic.endf": "tsl-Be-metal.endf",
        "micro_tsl_sch4.endf": "tsl-s-CH4.endf",
        "micro_tsl_un_elastic.endf": "tsl-NinUN.endf",
    }
    for source, name in copies.items():
        shutil.copy(ENDF_DATA / source, tmp_path / name)
    shutil.copy(NJOY_DATA / "tsl-026_Fe_056.leapr", tmp_path / "tsl-Be-metal.leapr")
    shutil.copy(NJOY_DATA / "tsl-ZrinZrC_flassh-DOS.txt", tmp_path / "tsl-NinUN_flassh-DOS.txt")
    (tmp_path / "tsl-NinUN-5P.readme").write_text("a sibling's readme\n")
    (tmp_path / "tsl-CH4.readme").write_text("methane\n")
    (tmp_path / "README.md").write_text("the library\n")
    return tmp_path


def test_a_small_library_groups_by_compound_with_its_inputs(small_library):
    index = index_tsl(small_library)
    assert sorted(t.file_name for t in index.tapes) == [
        "tsl-Be-metal.endf", "tsl-NinUN.endf", "tsl-s-CH4.endf"]
    by_name = {t.name: t for t in index.tapes}
    assert by_name["Be-metal"].companions == {"leapr": [str(small_library / "tsl-Be-metal.leapr")]}
    assert by_name["Be-metal"].has_phonon_spectrum
    assert "flassh-dos" in by_name["NinUN"].companions
    # A readme of a tape that is not here is not this tape's, however close the name.
    assert "readme" not in by_name["NinUN"].companions
    compounds = {c.name: c for c in index.compounds()}
    assert set(compounds) == {"Be metal", "CH4", "UN"}
    assert compounds["CH4"].companions == {"readme": [str(small_library / "tsl-CH4.readme")]}
    assert compounds["CH4"].variants == ["solid"]
    described = index.describe()
    assert described["tapes"] == 3 and described["unreadable"] == 2
    assert [c["name"] for c in described["compounds"]] == ["Be metal", "CH4", "UN"]


def test_index_endf_sets_tsl_tapes_aside(small_library):
    index = index_endf(small_library)
    assert index.files == [] and len(index.thermal) == 3
    assert index.describe()["thermal"] == 3


# ---------------------------------------------------------------- full size

def test_the_whole_of_endf_b_viii_1_is_indexed_without_losing_a_tape(tsl_be_metal_tape):
    root = Path(tsl_be_metal_tape).parent
    index = index_tsl(root)
    tapes = sorted(root.glob("*.endf"))
    assert len(index.tapes) == len(tapes) == 114
    assert len({t.mat for t in index.tapes}) == 114
    assert all(not t.problems for t in index.tapes)
    leaprs = {Path(p).name for t in index.tapes for p in t.companions.get("leapr", [])}
    assert leaprs == {p.name for p in root.glob("*.leapr")}
    compounds = {c.name: c for c in index.compounds()}
    assert compounds["UO2"].variants == ["5P", "10P", "HALEU", "HEU", "100P"]
    assert len(compounds["Graphite"].tapes) == 5
    water = [t for t in index.tapes if t.name == "HinH2O"][0]
    assert (len(water.temperatures), water.n_alpha, water.n_beta, water.elastic) == (94, 222, 317, None)


def test_one_tape_is_indexed_with_the_files_beside_it(small_library):
    index = index_tsl(small_library / "tsl-Be-metal.endf")
    assert index.single_file and [t.name for t in index.tapes] == ["Be-metal"]
    assert set(index.tapes[0].companions) == {"leapr"}
    assert index.unreadable == [] and index.describe()["single_file"] is True

"""Phase 1: names, bytes, tokens and the library index.

Acceptance (roadmap §5 Fase 1): text and zlib give the same tokens; damaged
and truncated input is detected; an isotope is found only under its own name;
``JENDL_HE`` is never mistaken for the elastic data.
"""
from __future__ import annotations

import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

import kika.g4ndl as g4ndl
from kika.g4ndl import (
    G4NDLError, G4NDLFormatError, G4NDLLibrary, IsotopeKey, IsotopeNotFoundError,
)
from kika.g4ndl.names import parse_file_name, parse_target
from kika.g4ndl.stream import read_text
from kika.g4ndl.tokens import TokenStream

DATA = Path(__file__).parent / "data"
JEFF = DATA / "JEFF-4.0"
SYNTH = DATA / "synthetic"


# ------------------------------------------------------------------- names

@pytest.mark.parametrize("stem,key,element", [
    ("26_56_Iron", IsotopeKey(26, 56, 0), "Iron"),
    ("27_58m1_Cobalt", IsotopeKey(27, 58, 1), "Cobalt"),
    ("6_nat_Carbon", IsotopeKey(6, None, 0), "Carbon"),
    ("4_9_Berylium", IsotopeKey(4, 9, 0), "Berylium"),
])
def test_file_names(stem, key, element):
    assert parse_file_name(stem) == (key, element)


@pytest.mark.parametrize("stem", ["README", "26_56", "Iron_26_56", "26_56_Iron.z"])
def test_non_names_are_rejected(stem):
    assert parse_file_name(stem) is None


@pytest.mark.parametrize("target,key", [
    ("Fe56", IsotopeKey(26, 56)), ("fe-56", IsotopeKey(26, 56)),
    ("Am242m", IsotopeKey(95, 242, 1)), ("Am242m1", IsotopeKey(95, 242, 1)),
    ("Co58m1", IsotopeKey(27, 58, 1)), ("Cnat", IsotopeKey(6, None)),
    ("C-nat", IsotopeKey(6, None)), (26056, IsotopeKey(26, 56)),
    ("26056", IsotopeKey(26, 56)), (6000, IsotopeKey(6, None)),
    ((27, 58, 1), IsotopeKey(27, 58, 1)), ((26, 56), IsotopeKey(26, 56)),
])
def test_targets(target, key):
    assert parse_target(target) == key


@pytest.mark.parametrize("target", ["Xx56", "Fe", "95642", 95642, "56Fe"])
def test_bad_targets(target):
    with pytest.raises(ValueError):
        parse_target(target)


# ------------------------------------------------------------------- bytes

def test_z_and_text_give_identical_text(tmp_path):
    src = JEFF / "Elastic/FS/1_1_Hydrogen.z"
    text = read_text(src)
    plain = tmp_path / "1_1_Hydrogen"
    plain.write_text(text, encoding="ascii")
    assert read_text(plain) == text
    a, b = TokenStream(text), TokenStream(read_text(plain))
    assert a.floats(a.remaining, "all").tolist() == b.floats(b.remaining, "all").tolist()


def test_truncated_zlib_is_detected(tmp_path):
    raw = (JEFF / "Elastic/FS/1_1_Hydrogen.z").read_bytes()
    bad = tmp_path / "1_1_Hydrogen.z"
    bad.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(G4NDLFormatError, match="truncated"):
        read_text(bad)


def test_corrupt_zlib_is_detected(tmp_path):
    bad = tmp_path / "1_1_Hydrogen.z"
    bad.write_bytes(b"\x00not zlib at all")
    with pytest.raises(G4NDLFormatError, match="zlib"):
        read_text(bad)


def test_trailing_bytes_after_zlib_are_detected(tmp_path):
    bad = tmp_path / "1_1_Hydrogen.z"
    bad.write_bytes(zlib.compress(b"1 2 3") + b"junk")
    with pytest.raises(G4NDLFormatError, match="after the end"):
        read_text(bad)


def test_size_ceiling(tmp_path):
    big = tmp_path / "1_1_Hydrogen.z"
    big.write_bytes(zlib.compress(b"0 " * 10_000))
    with pytest.raises(G4NDLFormatError, match="max_bytes"):
        read_text(big, max_bytes=1000)


def test_non_ascii_is_detected(tmp_path):
    bad = tmp_path / "1_1_Hydrogen"
    bad.write_bytes("1 2 é".encode("utf-8"))
    with pytest.raises(G4NDLFormatError, match="non-ASCII"):
        read_text(bad)


# ------------------------------------------------------------------ tokens

def test_header_is_two_tokens_and_only_when_first_is_G4NDL():
    t = TokenStream("G4NDL JEFF-4.0\n0 1.0 2\n2\n")
    assert t.header == ("G4NDL", "JEFF-4.0")
    assert t.position == 2 and t.int("repFlag") == 0
    # A file starting with a number keeps its first two tokens.
    t = TokenStream("0 0\n3\n")
    assert t.header is None and (t.int("a"), t.int("b"), t.int("n")) == (0, 0, 3)


def test_lines_mean_nothing():
    a = TokenStream("1 2 3\n4 5 6\n")
    b = TokenStream("1\n2\t3 4\n\n5      6")
    assert a.floats(6, "x").tolist() == b.floats(6, "x").tolist()


def test_integer_written_as_float_is_rejected():
    stream = TokenStream("1 14.8713 2\n2.0\n")
    stream.int("repFlag"), stream.float("mass"), stream.int("frame")
    with pytest.raises(G4NDLFormatError, match=r"integer.*'2\.0'") as exc:
        stream.int("NE")
    assert exc.value.token == 3 and exc.value.record == "NE"


def test_running_off_the_end_names_the_record():
    stream = TokenStream("1 2")
    stream.int("a"), stream.int("b")
    with pytest.raises(G4NDLFormatError, match="end of data") as exc:
        stream.int("NL")
    assert exc.value.record == "NL"
    with pytest.raises(G4NDLFormatError, match="2 values needed"):
        TokenStream("1.0").floats(2, "pairs")


def test_non_numeric_and_non_finite_tokens():
    with pytest.raises(G4NDLFormatError, match="'x'") as exc:
        TokenStream("1.0 2.0 x 4.0").floats(4, "sigma")
    assert exc.value.token == 2
    with pytest.raises(G4NDLFormatError, match="non-finite"):
        TokenStream("1.0 nan").floats(2, "sigma")
    with pytest.raises(G4NDLFormatError, match="non-finite"):
        TokenStream("inf").float("E")


def test_pairs_split_interleaved_values():
    x, y = TokenStream("1 10 2 20 3 30").pairs(3, "cs")
    assert x.tolist() == [1, 2, 3] and y.tolist() == [10, 20, 30]


def test_expect_end():
    s = TokenStream("1 2 3")
    s.int("a")
    with pytest.raises(G4NDLFormatError, match="2 unread tokens"):
        s.expectEnd()
    s.floats(2, "rest")
    s.expectEnd()


# ----------------------------------------------------------------- library

def test_index_of_a_real_mini_library():
    lib = g4ndl.open(JEFF)
    assert isinstance(lib, G4NDLLibrary)
    assert [str(k) for k in lib.isotopes()] == [
        "H1", "He3", "C12", "N14", "Co58m1"]
    assert "Fe56" not in lib and "Co58m1" in lib and "Co58" not in lib
    assert lib.duplicates == [] and lib.unindexed == []
    f = lib.locate("Co58m1", "Elastic/FS")
    assert f.compressed and f.elementName == "Cobalt" and f.path.name == "27_58m1_Cobalt.z"


def test_exact_lookup_never_substitutes(tmp_path):
    lib = g4ndl.open(DATA / "G4NDL-4.7.1")
    # Only natural carbon is there; Geant4 would hand it out for C12.
    assert lib.isotopes() == [IsotopeKey(6, None)]
    with pytest.raises(IsotopeNotFoundError, match="does not substitute"):
        lib.locate("C12", "Elastic/FS")
    assert lib.tokens("Cnat", "Elastic/FS").int("repFlag") in (1, 2, 3)


def test_the_z_wins_and_the_pair_is_reported():
    lib = g4ndl.open(SYNTH)
    f = lib.locate("Be9", "Elastic/FS")
    assert f.path.name == "4_9_Berylium.z" and f.shadowed.name == "4_9_Berylium"
    assert (f.shadowed, f.path) in lib.duplicates
    # The .z says repFlag=0 and the text says 1: the .z is what is read.
    assert lib.tokens("Be9", "Elastic/FS").int("repFlag") == 0


def test_header_survives_through_the_library():
    stream = g4ndl.open(SYNTH).tokens("H1", "Elastic/FS")
    assert stream.header == ("G4NDL", "synthetic")


def test_jendl_he_is_not_indexed(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(JEFF, root)
    he = root / "JENDL_HE/neutron/Elastic/FS"
    he.mkdir(parents=True)
    (he / "26_56_Iron").write_text("1 55.4544 2\n")
    lib = g4ndl.open(root)
    assert "Fe56" not in lib
    assert "JENDL_HE" in lib.presentTopLevel()


def test_a_cross_section_without_a_final_state_is_not_an_isotope(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(JEFF, root)
    (root / "Elastic/CrossSection/0_0_Zero").write_text("0 0\n0\n")
    lib = g4ndl.open(root)
    assert IsotopeKey(0, 0) not in lib.isotopes()


def test_stray_files_are_listed_not_indexed(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(JEFF, root)
    (root / "Elastic/FS/README").write_text("notes")
    assert [p.name for p in g4ndl.open(root).unindexed] == ["README"]


def test_wrong_root_says_where_the_library_is(tmp_path):
    nested = tmp_path / "JEFF-4.0"
    shutil.copytree(JEFF, nested / "JEFF-4.0")
    with pytest.raises(G4NDLError, match="did you mean"):
        g4ndl.open(nested.parent / "JEFF-4.0")
    with pytest.raises(G4NDLError, match="not a directory"):
        g4ndl.open(tmp_path / "missing")


def test_unknown_subdir_and_process():
    lib = g4ndl.open(JEFF)
    with pytest.raises(ValueError, match="not indexed"):
        lib.locate("H1", "Capture/CrossSection")
    with pytest.raises(ValueError, match="not read by kika yet"):
        lib.isotopes("capture")


def test_cross_section_streams_to_its_last_token():
    stream = g4ndl.open(JEFF).tokens("H1", "Elastic/CrossSection")
    assert (stream.int("a"), stream.int("b")) == (0, 0)
    n = stream.int("N")
    e, sigma = stream.pairs(n, "pairs")
    stream.expectEnd()
    assert n == 135 and e[0] == 1e-5 and np.all(sigma > 0)

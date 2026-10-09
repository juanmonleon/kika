"""``kika.library_index``: ENDF and ACE libraries indexed from file headers alone.

The headers are written here, record by record, in the shapes the published
libraries use (ENDF/B-VIII.1 MF1/MT451; LANL Lib81 and NEA JEFF-4.0 ACE first
lines), because the point of the index is that it never parses past them.
"""
from __future__ import annotations

from kika.library_index import _zaid_nuclide, index_ace, index_endf, target_name


def _endf(za, awr, mat, liso=0, nsub=10, temp=0.0):
    def rec(fields, n):
        return "".join(f"{f:>11}" for f in fields) + f"{mat:>4}" + " 1451" + f"{n:>5}"
    return "\n".join([
        " ENDF/B-VIII.1" + " " * 52 + "   0 0  0    0",
        rec([f"{za:.6e}".replace("e+0", "+"), f"{awr:.6e}".replace("e+0", "+"), 1, 1, 0, 1], 1),
        rec(["0.000000+0", "1.000000+0", 0, liso, 0, 6], 2),
        rec(["1.000000+0", "2.000000+7", 1, 0, nsub, 8], 3),
        rec([f"{temp:.6e}".replace("e+0", "+"), "0.000000+0", 0, 0, 10, 20], 4),
    ]) + "\n"


def test_target_names_follow_the_g4ndl_spelling():
    assert target_name(26, 56) == "Fe56"
    assert target_name(95, 242, 1) == "Am242m1"
    assert target_name(6, None) == "Cnat"


def test_endf_headers_give_the_nuclide_isomer_and_sublibrary(tmp_path):
    (tmp_path / "n-026_Fe_056.endf").write_text(_endf(26056, 55.4544, 2631))
    (tmp_path / "n-095_Am_242m1.endf").write_text(_endf(95242, 239.9801, 9547, liso=1))
    (tmp_path / "README.md").write_text("not a tape\n")
    d = index_endf(tmp_path).describe()
    assert [(e["target"], e["mat"], e["nsub"]) for e in d["isotopes"]] == [
        ("Fe56", 2631, 10), ("Am242m1", 9547, 10)]
    assert d["unreadable"] == 1 and d["format"] == "endf"


def test_ace_isomers_decode_in_both_library_conventions():
    # LANL: A + 300 + 100 m.  NEA JEFF-4.0: 300 + (A mod 100).  Ground: A.
    assert _zaid_nuclide("95642.10c", 239.9801) == (95, 242, 1)
    assert _zaid_nuclide("95342.40c", 239.98) == (95, 242, 1)
    assert _zaid_nuclide("27358.40c", 57.43806) == (27, 58, 1)
    assert _zaid_nuclide("47306.40c", 104.9969) == (47, 106, 1)
    assert _zaid_nuclide("26056.10c", 55.454) == (26, 56, 0)
    assert _zaid_nuclide("6000.80c", 11.9078) == (6, None, 0)


def test_ace_first_lines_give_the_table_and_its_temperature(tmp_path):
    (tmp_path / "26056.10c").write_text(" 26056.10c   55.454000  2.5300E-08   08/25/24\nrest\n")
    (tmp_path / "95-Am-242m-600.0").write_text(" 95342.42c  239.980000  5.1704E-08   06/07/25\nrest\n")
    (tmp_path / "1001.10t").write_text(" lwtr.10t  0.999170  2.5300E-08   08/25/24\nrest\n")
    (tmp_path / "xsdir").write_text("datapath\n")
    d = index_ace(tmp_path).describe()
    got = {e["target"]: (e["zaid"], e["temperature"]) for e in d["isotopes"]}
    assert got == {"Fe56": ("26056.10c", 293.6), "Am242m1": ("95342.42c", 600.0)}
    assert d["temperatures"] == [293.6, 600.0]
    assert d["unreadable"] == 1   # the thermal table; xsdir is skipped silently


def test_recursive_walks_the_subdirectories_and_skips_hidden_ones(tmp_path):
    (tmp_path / "Fe").mkdir()
    (tmp_path / "Fe" / "n-026_Fe_056.endf").write_text(_endf(26056, 55.4544, 2631))
    (tmp_path / "n-095_Am_242m1.endf").write_text(_endf(95242, 239.9801, 9547, liso=1))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "n-026_Fe_056.endf").write_text(_endf(26056, 55.4544, 2631))
    flat = index_endf(tmp_path).describe()
    assert [e["target"] for e in flat["isotopes"]] == ["Am242m1"]
    deep = index_endf(tmp_path, recursive=True).describe()
    assert [e["target"] for e in deep["isotopes"]] == ["Fe56", "Am242m1"]
    assert deep["duplicates"] == 0 and deep["unreadable"] == 0

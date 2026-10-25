"""MF2: each resolved resonance's SLBW peak, and the URR parameters as curves.

What the app draws one line per resonance from, so a reader sees where the
strong resonances are without reconstructing the file. Checked against the
textbook form ``sigma_0 = 2.6e6 b eV * ((A+1)/A)^2 / E * g * Gn/G`` and on the
Pr-141 JENDL-5 cut (MLBW, I = 5/2), whose 218.7 eV resonance is hand-worked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf import read_endf

_PR141 = Path(__file__).parent / "data" / "data_checks" / "micro_pr141_jendl5_mf2.endf"


@pytest.fixture(scope="module")
def section():
    return read_endf(str(_PR141)).mf[2].mt[151]


def test_one_peak_per_row_of_the_table_in_its_order(section):
    assert len(section.resonance_peaks()) == len(section.resonance_table())


def test_a_bound_level_has_no_peak(section):
    row, peak = section.resonance_table()[0], section.resonance_peaks()[0]
    assert row["energy"] < 0
    assert peak == {"total": None, "elastic": None, "capture": None, "fission": None}


def test_the_peak_is_the_breit_wigner_one(section):
    rows, peaks = section.resonance_table(), section.resonance_peaks()
    i = next(i for i, r in enumerate(rows) if r["energy"] == pytest.approx(218.7))
    row, peak = rows[i], peaks[i]
    awr, spin = section.atomic_weight_ratio, section.target_spin
    g = (2 * row["spin"] + 1) / (2 * (2 * spin + 1))
    sigma0 = 2.6037e6 * ((awr + 1) / awr) ** 2 / row["energy"] * g * row["GN"] / row["GT"]
    assert peak["total"] == pytest.approx(sigma0, rel=1e-3)
    assert peak["capture"] == pytest.approx(peak["total"] * row["GG"] / row["GT"])
    assert peak["elastic"] == pytest.approx(peak["total"] * row["GN"] / row["GT"])
    assert peak["fission"] == 0.0


def test_the_unresolved_series_cover_the_table(section):
    """Pr-141's URR is Case C: one curve per (l, J), the table's rows in order."""
    series = section.unresolved_parameter_series()
    rows = section.unresolved_parameters_table()
    assert sum(len(s["energies"]) for s in series) == len(rows)
    first = series[0]
    assert first["case"] == "C"
    assert all(len(first[k]) == len(first["energies"]) for k in ("D", "GN0", "GG", "GF", "GX"))
    assert first["energies"][0] == rows[0]["energy"] and first["D"][0] == rows[0]["D"]
    assert all(s["el"] <= e <= s["eh"] for s in series for e in s["energies"])

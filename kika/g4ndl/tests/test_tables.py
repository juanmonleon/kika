"""The viewer's arrays: :meth:`G4NDLLibrary.describe`, :func:`isotopeSummary`
and :func:`angularBulk`.

The test that matters is the last one. :func:`angularBulk` is read by the
kika-app the way it reads an MF4 bulk response (``endfAngularSource`` in
``frontend/src/utils/angularSeries.ts``): coefficients interpolated per region
of the Legendre block and summed, tables lin-lin in mu and in energy above it.
``_client`` mirrors that reading, and it has to land on
:func:`kika.g4ndl.physics.angularPdf`, which is the one checked against Geant4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

import kika.g4ndl as g4ndl
from kika.g4ndl.physics import angularPdf

DATA = Path(__file__).parent / "data"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
MU = np.linspace(-1.0, 1.0, 41)


def _interp(x0, x1, y0, y1, x, law):
    if x1 == x0:
        return y1
    if law == 3:
        t = np.log(x / x0) / np.log(x1 / x0)
    else:
        t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _client(bulk, energy, mu=MU):
    """``endfAngularSource``'s reading of a bulk response, in numpy."""
    e = np.asarray(bulk["energies"])
    rows = bulk["pdf_by_energy"] or [None] * e.size
    leg = [i for i, r in enumerate(rows) if r is None]
    tab = [i for i, r in enumerate(rows) if r is not None]
    last = e[leg[-1]] if leg else -np.inf
    if tab and energy > last:
        te = [e[i] for i in tab]
        tm = [bulk["pdf_mu_by_energy"][i] for i in tab]
        tp = [rows[i] for i in tab]
        if bulk["pdf_boundary_energy"] is not None:
            te.insert(0, bulk["pdf_boundary_energy"])
            tm.insert(0, bulk["pdf_boundary_mu"])
            tp.insert(0, bulk["pdf_boundary"])
        k = max(0, min(int(np.searchsorted(te, energy, side="right")) - 1, len(te) - 2))
        lo = np.interp(mu, tm[k], tp[k])
        hi = np.interp(mu, tm[k + 1], tp[k + 1])
        return _interp(te[k], te[k + 1], lo, hi, energy, 2)
    le = e[leg]
    laws = np.full(max(le.size - 1, 0), 2)
    start = 0
    for nbt, law in bulk["energy_interpolation"] or []:
        laws[start:nbt - 1] = law
        start = nbt - 1
    k = max(0, min(int(np.searchsorted(le, energy, side="right")) - 1, le.size - 2))
    order = bulk["max_order"]
    a = np.array([[bulk["coefficients_by_order"][str(l)][leg[j]] for l in range(order + 1)]
                  for j in (k, k + 1)])
    coefficients = _interp(le[k], le[k + 1], a[0], a[1], energy, laws[k])
    scale = (2 * np.arange(order + 1) + 1) / 2.0
    return npleg.legval(mu, scale * coefficients)


# ---------------------------------------------------------------- describe

def test_describe_lists_the_index_without_reading():
    d = JEFF.describe()
    assert d["name"] == "JEFF-4.0"
    assert [i["target"] for i in d["isotopes"]] == ["H1", "He3", "C12", "N14", "Co58m1"]
    ids = [i["id"] for i in d["isotopes"]]
    assert ids == ["H1", "He3", "C12", "N14", "Co58_m1"]
    co = d["isotopes"][-1]
    assert (co["Z"], co["A"], co["M"], co["element"]) == (27, 58, 1, "Cobalt")
    assert all(i["compressed"] for i in d["isotopes"])
    assert d["processes"] == ["elastic"] and d["unread"] == []


# ----------------------------------------------------------- the summary

def test_summary_of_a_mixed_isotope():
    s = g4ndl.isotopeSummary(JEFF.read("C12"))
    assert s["rep_flag"] == 3 and s["frame"] == "CM"
    assert s["target_mass"] == pytest.approx(11.8969)
    assert s["transition_energy"] == 2.0e7
    assert s["legendre"]["count"] == 702 and s["legendre"]["max_order"] == 8
    assert s["tabulated"]["count"] == 17
    assert s["cross_section"]["count"] == 645
    assert any("does not record which evaluation" in m for m in s["report"]["losses"])


# ------------------------------------------------------- the bulk payload

@pytest.mark.parametrize("target", ["H1", "He3", "C12", "N14", "Co58m1"])
def test_bulk_rows_line_up(target):
    b = g4ndl.angularBulk(JEFF.read(target))
    n = len(b["energies"])
    assert np.all(np.diff(b["energies"]) >= 0)
    assert all(len(v) == n for v in b["coefficients_by_order"].values())
    np.testing.assert_allclose(b["coefficients_by_order"]["0"], 1.0, atol=1e-4)
    if b["pdf_by_energy"] is not None:
        assert len(b["pdf_by_energy"]) == n == len(b["pdf_mu_by_energy"])


def test_regions_are_carried():
    b = g4ndl.angularBulk(JEFF.read("N14"))
    assert b["energy_interpolation"] == [[14, 3], [614, 2]]
    assert b["representation"] == 3 and b["pdf_boundary_energy"] == 2.0e7


@pytest.mark.parametrize("target", ["H1", "He3", "C12", "N14", "Co58m1"])
def test_client_reading_is_angularPdf(target):
    """The arrays, read as the app reads them, are Geant4's p(mu|E)."""
    suite = JEFF.read(target)
    b = g4ndl.angularBulk(suite)
    e = np.asarray(b["energies"])
    # Every stored energy, and the midpoints between them, geometric.
    probes = np.unique(np.concatenate([e, np.sqrt(e[1:] * e[:-1])]))
    for energy in probes[:: max(1, probes.size // 200)]:
        ref = angularPdf(suite, float(energy), MU, side="geant4")
        got = _client(b, float(energy))
        # C-12's tables are LOGLIN in mu: sent lin-lin on 16 points a segment,
        # and interpolated in energy after mu, not before (physics docstring:
        # the two orders differ by up to 8e-5 there). 2.0e-4 measured at 120 MeV.
        np.testing.assert_allclose(got, ref, rtol=5e-4, atol=1e-9 * ref.max(),
                                   err_msg=f"{target} at {energy:g} eV")


def test_table_coefficients_are_moments():
    b = g4ndl.angularBulk(JEFF.read("He3"))
    for i in (0, len(b["energies"]) - 1):
        mu, p = np.asarray(b["pdf_mu_by_energy"][i]), np.asarray(b["pdf_by_energy"][i])
        a1 = b["coefficients_by_order"]["1"][i]
        # p is lin-lin between its cosines, so int p mu dmu is exact on a
        # fine linear resampling of it.
        fine = np.linspace(-1.0, 1.0, 200_001)
        assert a1 == pytest.approx(np.trapezoid(np.interp(fine, mu, p) * fine, fine), abs=1e-6)


# ------------------------------------------------------------ kika.plotting

def test_plottable_g4ndl_suite():
    from kika.plotting import plottable

    suite = JEFF.read("C12")
    xs = plottable(suite, "cross_section", mt=2).data
    assert xs.provenance.format == "g4ndl" and xs.provenance.evaluation == "JEFF-4.0"
    assert "G4NDL" in xs.provenance.describe()
    ad = plottable(suite, "angular_distribution", mt=2, energy=3.0e7).data
    np.testing.assert_allclose(ad.y, angularPdf(suite, 3.0e7, ad.x, side="geant4"))
    assert ad.provenance.frame == "CM"
    ds = plottable(suite, "differential_cross_section", mt=2, energy=3.0e7).data
    assert np.all(ds.y > 0)
    a1 = plottable(suite, "legendre_coefficient", mt=2, order=1).data
    assert a1.x.size == len(g4ndl.angularBulk(suite)["energies"])
    assert a1.provenance.detail == "tables projected"


# ------------------------------------------------------------ the inelastic

INELASTIC = DATA / "inelastic"


def _inelastic(lib, target):
    return g4ndl.open(INELASTIC / lib).read(target)


def test_every_cross_section_is_listed_sums_included():
    suite = _inelastic("G4NDL-4.7.1", "Eu151")
    assert list(g4ndl.crossSections(suite)) == [4, *range(51, 60), 91]
    e, s = g4ndl.crossSections(suite)[4]
    assert e.size == s.size and s.max() > 0
    # Inelastic/CrossSection is not among them: it has no ENDF MT.
    assert g4ndl.inelasticTotal(suite) is None


@pytest.mark.parametrize("lib, target, mts", [
    ("G4NDL-4.7.1", "Eu151", [*range(51, 60), 91]),   # levels + MT91 (MF4 with MF5)
    ("JEFF-4.0", "Au197", [37]),
    ("G4NDL-4.7.1", "Fe58", []),                       # only protons and alphas
])
def test_angular_mts_are_the_neutrons_mf4s(lib, target, mts):
    assert g4ndl.angularMTs(_inelastic(lib, target)) == mts


@pytest.mark.parametrize("lib, target, mt, rep", [
    ("G4NDL-4.7.1", "Eu151", 51, 2),
    ("JEFF-4.0", "Au197", 37, 1),
    ("G4NDL-4.7.1", "Ni64", 51, 0),
])
def test_an_inelastic_bulk_reads_as_a_normalised_pdf(lib, target, mt, rep):
    suite = _inelastic(lib, target)
    b = g4ndl.angularBulk(suite, mt=mt)
    assert b["representation"] == rep and b["frame"] in ("CM", "LAB")
    e = np.asarray(b["energies"])
    assert np.all(np.diff(e) >= 0)
    fine = np.linspace(-1.0, 1.0, 4001)
    for energy in (e[0], 0.5 * (e[0] + e[-1]), e[-1]):
        if rep == 0:
            break
        p = _client(b, energy, fine)
        assert np.trapezoid(p, fine) == pytest.approx(1.0, abs=2e-3)


def test_a_reaction_with_no_neutron_angular_distribution_is_refused():
    suite = _inelastic("G4NDL-4.7.1", "Fe58")
    with pytest.raises(ValueError, match="no neutron"):
        g4ndl.angularBulk(suite, mt=750)

"""kika.nuclear_data.photon_production: the photons a suite emits, from any format.

The micro-tapes are the committed MF12-15 fixtures of ``kika/endf/tests/data``.
The full Fe-56 tape of ENDF/B-VIII.1 and its ACE table are used where they are
reachable and skipped otherwise, like every other tape-backed test in kika:
the physical check there is that the 846.8 keV line of the first excited level
dominates the gammas of inelastic scattering at 2 MeV, and that the ENDF and
the ACE reading of it agree.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kika.nuclear_data.photon_production import (
    PhotonProduction,
    PhotonProductionError,
    resonance_background,
)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
N14 = DATA / "micro_n14_photons.endf"
S36 = DATA / "micro_s36_b81_photons.endf"
U235 = DATA / "micro_u235_photons.endf"
FE56_ENDF = Path(r"C:\Users\Usuario\.kika\endf_cache\endfb8.1\n\26056.endf")
FE56_ACE = Path(r"C:\Users\Usuario\BaradDur\Dev\kika-files\ace\260560_81.02c")
FE56_JENDL5 = Path(r"C:\Users\Usuario\.kika\endf_cache\jendl5\n\26056.endf")

_SUITES: dict = {}


def _endf(path):
    from kika.endf import read_endf
    return read_endf(str(path))


def _suite(path):
    """Decoded once per test session: the micro-tapes are small, Fe-56 is not."""
    if path not in _SUITES:
        from kika.endf.model_adapter import decodeReactionSuite
        endf = _endf(path)
        _SUITES[path] = (endf, decodeReactionSuite(endf)[0])
    return _SUITES[path]


def _sigma(suite, mt, energy):
    reaction = next(r for r in suite.reactions if r.ENDF_MT == mt)
    return float(reaction.crossSection["eval"].evaluate(energy))


# ---------------------------------------------------------------------------
# Cascades (S-36: MF12 LO=2 on the level reactions)
# ---------------------------------------------------------------------------

def test_a_level_reaction_emits_its_whole_cascade_per_reaction():
    from kika.nuclear_data.model import cascadeLines

    _, suite = _suite(S36)
    spectrum = PhotonProduction(suite).spectrum(8.0e6, mt=55, quantity="multiplicity")
    expected = cascadeLines(suite.PoPs, "S36_e5")
    assert [(l["initial"], l["final"]) for l in spectrum["lines"]] == [
        (l.initial, l.final) for l in expected]
    assert [l["value"] for l in spectrum["lines"]] == pytest.approx([l.photons for l in expected])
    assert spectrum["totals"]["line_photons"] == pytest.approx(sum(l.photons for l in expected))
    assert [l["key"] for l in spectrum["lines"]] == [
        f"cascade:{l.initial}:{l.final}" for l in expected]
    # A cascade that reached the model is isotropic (an anisotropic one is refused).
    assert {l["angular"] for l in spectrum["lines"]} == {"isotropic"}


def test_production_weights_each_cascade_by_its_reactions_cross_section():
    _, suite = _suite(S36)
    spectrum = PhotonProduction(suite).spectrum(8.0e6, quantity="production")
    line = next(l for l in spectrum["lines"] if l["key"] == "cascade:S36_e1:S36")
    per_mt = {c["mt"]: c["value"] for c in line["by_mt"]}
    assert per_mt[51] == pytest.approx(_sigma(suite, 51, 8.0e6))
    assert line["value"] == pytest.approx(sum(per_mt.values()))
    assert spectrum["xs_source"] == "file"


def test_below_threshold_a_level_reaction_emits_nothing():
    _, suite = _suite(S36)
    assert PhotonProduction(suite).spectrum(1.0e5, mt=51, quantity="multiplicity")["lines"] == []


def test_the_curves_sum_to_the_total_and_carry_the_gamma_energy():
    _, suite = _suite(S36)
    curves = PhotonProduction(suite).curves(mt=51, quantity="multiplicity")
    x = np.asarray(curves["total"]["energy_ev"])
    total = np.asarray(curves["total"]["value"], dtype=float)
    above = x > 4.0e6  # past MT51's threshold, where its one line is emitted
    assert np.allclose(total[above], 1.0, rtol=1e-6)
    eg = curves["lines"][0]["gamma_energy_ev"]
    energy = np.asarray(curves["energy"]["value"], dtype=float)
    assert np.allclose(energy[above], eg, rtol=1e-5)
    # One reaction: the energy splits into lines and continuum, here lines only.
    assert [p["key"] for p in curves["energy_parts"]] == ["lines"]
    assert curves["energy_parts"][0]["value"] == curves["energy"]["value"]


def test_the_line_keys_agree_between_spectrum_and_curves():
    _, suite = _suite(S36)
    production = PhotonProduction(suite)
    spectrum_keys = {l["key"] for l in production.spectrum(8.0e6)["lines"]}
    curve_keys = {l["key"] for l in production.curves()["lines"]}
    assert spectrum_keys <= curve_keys


def test_energy_parts_split_by_reaction_and_add_up():
    _, suite = _suite(S36)
    curves = PhotonProduction(suite).curves()
    parts = curves["energy_parts"]
    assert all(p["key"].startswith("mt:") or p["key"] == "other" for p in parts)
    total = np.asarray(curves["energy"]["value"], dtype=float)
    # Every part is on the same decimated grid when nothing needs decimating.
    if all(len(p["energy_ev"]) == len(total) for p in parts):
        summed = np.sum([np.asarray(p["value"], dtype=float) for p in parts], axis=0)
        assert np.allclose(summed, total, rtol=1e-4)


def test_the_axis_covers_the_photon_producing_range():
    _, suite = _suite(S36)
    axis = PhotonProduction(suite).incident_axis()
    assert axis["grid"] == sorted(axis["grid"])
    assert axis["min"] >= 1e-5 and axis["max"] > 8.0e6
    # Each level reaction opens at its own threshold, the lowest at the axis start.
    thresholds = axis["thresholds"]
    assert [t["energy_ev"] for t in thresholds] == sorted(t["energy_ev"] for t in thresholds)
    assert thresholds[0]["energy_ev"] == pytest.approx(axis["min"])
    assert all(t["energy_ev"] in axis["grid"] for t in thresholds)


# ---------------------------------------------------------------------------
# Discrete and primary lines, MF14 (N-14)
# ---------------------------------------------------------------------------

def test_a_primary_photon_moves_with_the_incident_energy():
    _, suite = _suite(N14)
    production = PhotonProduction(suite)
    awr = suite.PoPs.particles["N14"].mass.value / 1.00866491595
    low, high = (production.spectrum(e, mt=102, quantity="multiplicity") for e in (1.0, 1.0e6))
    primaries = [[l for l in s["lines"] if l["origin"] == "primary"] for s in (low, high)]
    assert primaries[0][0]["key"] == primaries[1][0]["key"]
    shift = primaries[1][0]["energy_ev"] - primaries[0][0]["energy_ev"]
    assert shift == pytest.approx((1.0e6 - 1.0) * awr / (awr + 1.0))


def test_an_anisotropic_line_gives_its_legendre_distribution():
    _, suite = _suite(N14)
    production = PhotonProduction(suite)
    spectrum = production.spectrum(14.0e6)
    anisotropic = {l["key"] for l in spectrum["lines"] if l["angular"] == "anisotropic"}
    assert "line:4915101" in anisotropic and "primary:102:10829100" in anisotropic
    angular = production.angular("line:4915101", 14.74e6)
    assert angular["form"] == "legendre" and angular["mt"] == 4
    assert angular["node_energy_ev"] == pytest.approx(14.74e6)
    assert angular["coefficients"] == pytest.approx([1.0, 0.0, 0.164])
    mu = np.asarray(angular["mu"])
    pdf = np.asarray(angular["pdf"])
    assert mu.size == 201
    assert np.trapezoid(pdf, mu) == pytest.approx(1.0, rel=1e-4)


def test_an_isotropic_line_is_flat():
    _, suite = _suite(N14)
    production = PhotonProduction(suite)
    line = next(l for l in production.spectrum(14.0e6)["lines"] if l["angular"] == "isotropic")
    angular = production.angular(line["key"], 14.0e6)
    assert angular["form"] == "isotropic"
    assert set(angular["pdf"]) == {0.5}


def test_photons_per_reaction_need_one_reaction():
    _, suite = _suite(N14)
    with pytest.raises(PhotonProductionError):
        PhotonProduction(suite).spectrum(1.0e6, mt=None, quantity="multiplicity")
    with pytest.raises(PhotonProductionError):
        PhotonProduction(suite).curves(mt=2)


def test_one_curve_by_key_matches_the_curves():
    _, suite = _suite(N14)
    production = PhotonProduction(suite)
    x, y, info = production.curve("line:4915101", mt=4, quantity="multiplicity")
    assert x.size == y.size > 0 and info["xs_source"] is None
    with pytest.raises(PhotonProductionError):
        production.curve("line:1", mt=4, quantity="multiplicity")


# ---------------------------------------------------------------------------
# Continua (U-235 fission, MF15)
# ---------------------------------------------------------------------------

def test_a_continuum_is_drawn_at_the_nearest_tabulated_energy():
    _, suite = _suite(U235)
    production = PhotonProduction(suite)
    axis = production.incident_axis(mt=18)
    energy = axis["grid"][len(axis["grid"]) // 2]
    spectrum = production.spectrum(energy, mt=18, quantity="multiplicity")
    assert len(spectrum["continua"]) == 1
    continuum = spectrum["continua"][0]
    assert continuum["key"] == "continuum:18:0"
    nodes = production.continuum_map(18)["continua"][0]["incident_energy_ev"]
    assert continuum["node_energy_ev"] in nodes
    nearest = min(nodes, key=lambda n: abs(math.log(n) - math.log(energy)))
    assert continuum["node_energy_ev"] == nearest


def test_the_continuum_map_holds_every_node_on_one_grid():
    _, suite = _suite(U235)
    production = PhotonProduction(suite)
    plain = production.continuum_map(18)
    scaled = production.continuum_map(18, scaled=True)
    c, s = plain["continua"][0], scaled["continua"][0]
    assert c["grid"] in ("log", "linear")
    assert len(c["value"]) == len(c["incident_energy_ev"])
    assert all(len(row) == len(c["photon_energy_ev"]) <= 300 for row in c["value"])
    k = 0
    assert np.allclose(np.asarray(s["value"][k]),
                       np.asarray(c["value"][k]) * c["multiplicity"][k], rtol=1e-5)
    with pytest.raises(PhotonProductionError):
        production.continuum_map(2)


# ---------------------------------------------------------------------------
# Cross sections: the caller's, and the background left out
# ---------------------------------------------------------------------------

def test_a_callable_overrides_sigma_and_none_falls_back():
    _, suite = _suite(S36)
    calls = []

    def doubled(mt, energies):
        calls.append(mt)
        return None if mt != 51 else 2.0 * np.asarray([_sigma(suite, 51, e) for e in energies])

    base = PhotonProduction(suite).spectrum(8.0e6, mt=51)
    over = PhotonProduction(suite, cross_section=doubled).spectrum(8.0e6, mt=51)
    assert over["xs_source"] == "reconstructed"
    assert over["totals"]["line_photons"] == pytest.approx(2.0 * base["totals"]["line_photons"])
    other = PhotonProduction(suite, cross_section=doubled).spectrum(8.0e6, mt=52)
    assert other["xs_source"] == "file" and 52 in calls


def test_a_mapping_override_puts_its_grid_in_the_curves():
    _, suite = _suite(S36)
    grid = np.geomspace(5.0e6, 2.0e7, 4000)
    table = {51: (grid, np.full_like(grid, 0.25))}
    curves = PhotonProduction(suite, cross_section=table).curves(mt=51, max_points=10 ** 6)
    x = np.asarray(curves["total"]["energy_ev"])
    assert np.isin(np.round(grid[(grid > x[0]) & (grid < x[-1])], -2), np.round(x, -2)).mean() > 0.9
    assert curves["xs_source"] == "reconstructed"


def test_a_background_reaction_is_left_out_below_the_resonance_top():
    _, suite = _suite(S36)
    top = 6.0e6
    production = PhotonProduction(suite, background_mts=[51], resonance_top=top)
    below = production.spectrum(5.0e6, mt=51)
    assert below["lines"] == [] and below["excluded"] == [{"mt": 51, "reason": "background"}]
    assert below["xs_source"] == "background_only" and below["resonance_top_ev"] == top
    curves = production.curves()
    x = np.asarray(curves["total"]["energy_ev"])
    total = curves["total"]["value"]
    assert all(v is None for xi, v in zip(x, total) if xi < top)
    assert any(v is not None for xi, v in zip(x, total) if xi >= top)


def test_resonance_background_reads_mf2():
    endf, _ = _suite(N14)
    mts, top = resonance_background(endf)
    if not mts:
        assert top == 0.0
    else:
        assert {1, 2, 102} <= set(mts) and top > 0


# ---------------------------------------------------------------------------
# Fe-56, ENDF/B-VIII.1 and its ACE table
# ---------------------------------------------------------------------------

FE56_LINE = 0.528  # b: 846.8 keV gammas at 2 MeV, both readings


@pytest.mark.skipif(not FE56_ENDF.exists(), reason=f"{FE56_ENDF} not available")
def test_fe56_first_level_line_dominates_inelastic_at_2_mev():
    endf, suite = _suite(FE56_ENDF)
    mts, top = resonance_background(endf)
    assert 102 in mts and top > 0
    production = PhotonProduction(suite, background_mts=mts, resonance_top=top)
    spectrum = production.spectrum(2.0e6)
    lines = sorted(spectrum["lines"], key=lambda l: -l["value"])
    first = lines[0]
    assert first["key"] == "cascade:Fe56_e1:Fe56"
    assert first["energy_ev"] == pytest.approx(846.78e3, rel=1e-4)
    assert first["value"] == pytest.approx(FE56_LINE, rel=0.01)
    assert first["value"] > 10 * lines[1]["value"]
    assert 51 in {c["mt"] for c in first["by_mt"]}
    # MT102's MF3 is only the background below the resonance top: left out there.
    curves = production.curves()
    assert {"mt": 102, "reason": "background"} in curves["excluded"]


# ---------------------------------------------------------------------------
# MF6 LAW=1 discrete lines (JENDL-5 Fe-56 gives its level gammas there)
# ---------------------------------------------------------------------------

def test_mf6_discrete_points_become_lines_and_the_rest_stays_a_continuum():
    """ND discrete points ahead of a density: lines weighted node by node."""
    from kika.nuclear_data.photon_production import _Node, _Spectrum, _split_discrete

    class _Flat:
        xs = np.array([1e6, 2e7])

        @staticmethod
        def evaluate(e):
            return np.full_like(np.asarray(e, dtype=float), 2.0)

    nodes = [_Node(1e6, [846778.0, 0.0, 1e6], [0.6, 0.0, 8e-7]),
             _Node(2e7, [846778.0, 1238300.0, 0.0, 1e6], [0.4, 0.2, 0.0, 8e-7])]
    item = {"kind": "continuum", "spectrum": _Spectrum(nodes), "mult": _Flat(),
            "angular": None, "product": "photon"}
    parts = _split_discrete(item, {"photon": [1, 2]})
    lines = [p for p in parts if p["kind"] == "line"]
    assert [round(p["energy"]) for p in lines] == [846778, 1238300]
    # y = 2 photons per reaction, the 846.8 keV line carrying 0.6 then 0.4 of it.
    assert lines[0]["mult"].evaluate([1e6, 2e7]) == pytest.approx([1.2, 0.8])
    # A line a node does not list has no weight there.
    assert lines[1]["mult"].evaluate([1e6]) == pytest.approx([0.0])
    (rest,) = [p for p in parts if p["kind"] == "continuum"]
    assert [list(f.xs) for f in rest["spectrum"].function1ds] == [[0.0, 1e6], [0.0, 1e6]]
    # A product with no discrete points is left alone.
    assert _split_discrete(item, {"other": [1, 2]}) == [item]
    # A negative discrete energy is a primary photon of that binding energy.
    primary = {**item, "spectrum": _Spectrum([_Node(1e6, [-7646100.0], [1.0]),
                                             _Node(2e7, [-7646100.0], [1.0])])}
    (part,) = _split_discrete(primary, {"photon": [1, 1]})
    assert part["kind"] == "primary" and part["binding"] == pytest.approx(7646100.0)


@pytest.mark.skipif(not FE56_JENDL5.exists(), reason=f"{FE56_JENDL5} not available")
def test_jendl5_fe56_level_gammas_in_mf6_are_lines():
    """JENDL-5 states MT51's 846.8 keV gamma as MF6 LAW=1 with ND=1, and its
    residual level carries an empty placeholder photon: one line, no warning."""
    _, suite = _suite(FE56_JENDL5)
    spectrum = PhotonProduction(suite).spectrum(2.0e6)
    assert spectrum["warnings"] == []
    first = max(spectrum["lines"], key=lambda l: l["value"])
    assert first["energy_ev"] == pytest.approx(846.778e3, rel=1e-6)
    assert first["value"] == pytest.approx(FE56_LINE, rel=0.02)
    assert 51 in {c["mt"] for c in first["by_mt"]}
    # Capture's primary photons (negative EP in MF6) come out above the binding.
    assert all(l["energy_ev"] > 0 for l in spectrum["lines"])
    assert any(l["origin"] == "primary" for l in spectrum["lines"])


@pytest.mark.skipif(not FE56_ACE.exists(), reason=f"{FE56_ACE} not available")
def test_an_ace_suite_has_no_eval_forms_and_still_answers():
    from kika.ace.model_adapter import decodeAce
    from kika.ace.parsers.parse_ace import read_ace

    decoded = decodeAce(read_ace(str(FE56_ACE)))
    suite = decoded[0] if isinstance(decoded, tuple) else decoded
    mt51 = next(r for r in suite.reactions if r.ENDF_MT == 51)
    assert "eval" not in list(mt51.crossSection.keys())
    production = PhotonProduction(suite)
    spectrum = production.spectrum(2.0e6)
    assert spectrum["xs_source"] == "file" and spectrum["excluded"] == []
    # ACE has no cascades: the level's gamma is a discrete line on MT51.
    line = next(l for l in spectrum["lines"] if l["key"] == "line:846778")
    assert line["origin"] == "line"
    assert line["value"] == pytest.approx(FE56_LINE, rel=0.01)
    assert line["value"] == max(l["value"] for l in spectrum["lines"])


# -- processed tables (HDF5) ---------------------------------------------------

HDF5_FE56 = Path(__file__).resolve().parents[2] / "hdf5" / "tests" / "data" / "Fe56.h5"


def test_hdf5_lines_and_density_tables():
    """An OpenMC suite states lines as ``DiscretePhotonEnergy`` and continua as
    ``Xs_pdf_cdf1d`` densities under ``MonteCarlo``: both are read."""
    pytest.importorskip("h5py")
    from kika.hdf5 import read

    pp = PhotonProduction(read(HDF5_FE56))
    spectrum = pp.spectrum(incident_energy=2.0e6)
    line = next(l for l in spectrum["lines"] if abs(l["energy_ev"] - 846778.0) < 1.0)
    assert line["value"] > 0 and [m["mt"] for m in line["by_mt"]] == [51]
    assert spectrum["continua"]


def test_density_table_steps_and_discrete_split():
    from kika.nuclear_data.model.functions.pdf_cdf import Xs_pdf_cdf1d
    from kika.nuclear_data.model.enums import Interpolation
    from kika.nuclear_data.photon_production import _density_spectrum, _split_discrete

    class _Energy:
        function1ds = [
            Xs_pdf_cdf1d(xs=[5.0e5, 1.0, 2.0, 3.0], pdf=[0.25, 0.5, 0.25, 0.0], cdf=[0.25, 0.25, 0.75, 1.0],
                         interpolation=Interpolation.flat, outerDomainValue=1.0e6, numberOfDiscrete=1),
            Xs_pdf_cdf1d(xs=[5.0e5, 1.0, 2.0, 3.0], pdf=[0.5, 0.25, 0.25, 0.0], cdf=[0.5, 0.5, 0.75, 1.0],
                         interpolation=Interpolation.flat, outerDomainValue=2.0e6, numberOfDiscrete=1),
        ]

    spectrum, nd = _density_spectrum(_Energy())
    assert nd == [1, 1]
    node = spectrum.function1ds[0]
    # The discrete point first, then the histogram as steps.
    assert node.xs.tolist() == [5.0e5, 1.0, 2.0, 2.0, 3.0]
    assert node.ys.tolist() == [0.25, 0.5, 0.5, 0.25, 0.25]
    parts = _split_discrete({"kind": "continuum", "spectrum": spectrum, "mult": None,
                             "angular": None, "product": "photon", "nd": nd}, {})
    line = next(p for p in parts if p["kind"] == "line")
    assert line["energy"] == 5.0e5
    assert line["mult"].evaluate([1.0e6, 2.0e6]).tolist() == [0.25, 0.5]
    assert next(p for p in parts if p["kind"] == "continuum")["spectrum"].function1ds[0].xs[0] == 1.0

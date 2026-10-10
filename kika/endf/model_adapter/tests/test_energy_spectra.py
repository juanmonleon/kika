"""MF5's parametrised spectra ↔ §18.3's nodes, and MT455 ↔ the precursor families.

Roadmap E2 (kika-workspace ``docs/library/endf_coverage_roadmap.md``). Five
committed witnesses, cut verbatim -- ENDF/B-VIII.1 has no MT18 of LF=11 or 12,
so those two come from ENDF/B-VII.1 -- and the same two laws also built by hand,
where the parameters can be planted:

=================================  ==========================================
fixture                            what it carries
=================================  ==========================================
``micro_ra223_mf5_lf7.endf``       MT18 LF=7 -> ``SimpleMaxwellianFission``
``micro_o17_mf5_weighted.endf``    MT16 NK=2 of LF=9 -> ``WeightedFunctionals``
``micro_u235_delayed.endf``        MF1/455 + MF5/455 NK=6 LF=5 -> six families
``micro_u233_mf5_lf11.endf``       MT18 LF=11 -> ``Watt`` (ENDF/B-VII.1)
``micro_am241_mf5_lf12.endf``      MT18 LF=12 -> ``MadlandNix`` (ENDF/B-VII.1)
(hand-built)                       LF=11 ``Watt``, LF=12 ``MadlandNix``
=================================  ==========================================

What is asserted: the model evaluates the same spectrum the ENDF reader does
(one implementation, :mod:`kika.nuclear_data.spectrum_laws`); the tape comes back with the
section byte for byte; GNDS carries every parameter back; an edit to a
parametrised spectrum is refused on the way to ENDF rather than lost; and the
MF35 sampler refuses one by name (PD-3).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.energy import decodeMF5MT
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import writeEndfTape
from kika.nuclear_data.model import (ANALYTIC_SPECTRA, Evaporation,
                                     GeneralEvaporation, MadlandNix,
                                     PhysicalQuantity, SimpleMaxwellianFission,
                                     Watt, WeightedFunctionals, XYs1d)
from kika.nuclear_data.model.functions import Function2d

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
RA223 = DATA / "micro_ra223_mf5_lf7.endf"
O17 = DATA / "micro_o17_mf5_weighted.endf"
U235 = DATA / "micro_u235_delayed.endf"
U233 = DATA / "micro_u233_mf5_lf11.endf"
AM241 = DATA / "micro_am241_mf5_lf12.endf"


def _energy(suite, mt):
    product = suite.findReactionByENDF_MT(mt).outputChannel.products.byPid("n")[0]
    return product.distribution["eval"].energy


def _walk(node):
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return {f.name: _walk(getattr(node, f.name)) for f in dataclasses.fields(node)
                if f.name not in ("axes", "label", "index", "provenance")}
    if isinstance(node, np.ndarray):
        return [round(float(v), 9) for v in node.ravel()]
    if isinstance(node, (list, tuple)):
        return [_walk(v) for v in node]
    if hasattr(node, "value") and hasattr(node, "unit"):
        return (float(node.value), node.unit)
    if hasattr(node, "__iter__") and not isinstance(node, (str, bytes, dict)):
        return [_walk(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# the node, and that it is the reader's spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, mt, cls", [(RA223, 18, SimpleMaxwellianFission),
                                           (O17, 16, WeightedFunctionals),
                                           (U233, 18, Watt), (AM241, 18, MadlandNix)])
def test_a_parametrised_mf5_reaches_the_model_as_its_node(path, mt, cls):
    _endf, suite, report = (lambda e: (e, *decodeReactionSuite(e)))(read_endf(str(path)))
    energy = _energy(suite, mt)
    assert type(energy) is cls
    assert not isinstance(energy, Function2d), "PD-3: never a table to the sampler"
    assert not [m for m in report.unsupported if f"MF5/MT{mt}" in m]


@pytest.mark.parametrize("path, mt", [(RA223, 18), (O17, 16), (U233, 18), (AM241, 18)])
def test_the_model_evaluates_the_readers_spectrum(path, mt):
    """One implementation: the node and the flat partial agree to the last bit."""
    endf = read_endf(str(path))
    suite, _ = decodeReactionSuite(endf)
    energy = _energy(suite, mt)
    section = endf.mf[5].mt[mt]
    for incident in (1.0e6, 7.3e6, 2.0e7):
        grid = np.geomspace(1e3, 3e7, 200)
        expected = sum(
            np.interp(incident, p.p_energies, p.p_values) * p.evaluate_on_grid(incident, grid)
            for p in section.partials)
        assert np.allclose(energy.evaluate(incident, grid), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("path, mt", [(RA223, 18), (O17, 16), (U233, 18), (AM241, 18)])
def test_the_tabulated_spectrum_integrates_to_one(path, mt):
    energy = _energy(decodeReactionSuite(read_endf(str(path)))[0], mt)
    table = energy.toPointwise(incidentEnergies=[1.4e7], points=4000)
    assert np.trapezoid(table[0].ys, table[0].xs) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("path, mt", [(RA223, 18), (O17, 16), (U235, 455),
                                      (U233, 18), (AM241, 18)])
def test_the_tape_comes_back_with_the_section(path, mt, tmp_path):
    endf = read_endf(str(path))
    suite, report = decodeReactionSuite(endf)
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out, report=report)
    assert str(read_endf(str(out)).mf[5].mt[mt]) == str(endf.mf[5].mt[mt])


def test_an_edited_parametrised_spectrum_is_refused_not_lost(tmp_path):
    suite, _ = decodeReactionSuite(read_endf(str(RA223)))
    energy = _energy(suite, 18)
    energy.theta = XYs1d(xs=energy.theta.xs, ys=energy.theta.ys * 1.01,
                         interpolation=energy.theta.interpolation, axes=energy.theta.axes)
    with pytest.raises(ValueError, match="differs from the section it was decoded from"):
        writeEndfTape(suite, tmp_path / "edited.endf")


# ---------------------------------------------------------------------------
# MT455: the families
# ---------------------------------------------------------------------------

def test_mt455_puts_one_spectrum_on_each_family():
    suite, report = decodeReactionSuite(read_endf(str(U235)))
    families = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData.delayedNeutrons
    assert len(families) == 6
    total = suite.sums.multiplicitySums.byENDF_MT(455)
    assert len(total.summands) == 6, "the delayed sum now lists its parts"
    for family in families:
        form = family.product.distribution["eval"]
        assert isinstance(form.energy, GeneralEvaporation)
        assert form.isComplete
    # p_k are constants in ENDF/B-VIII.1, so each family's multiplicity is the
    # delayed nu-bar scaled: the six add back up to it.
    for energy in (1.0e-5, 1.0e6, 1.0e7):
        parts = sum(float(np.asarray(f.product.multiplicity.form.evaluate(energy)))
                    for f in families)
        assert parts == pytest.approx(float(np.asarray(total.multiplicity.form.evaluate(energy))),
                                      rel=1e-6)
    assert not [m for m in report.losses if "MF5/MT455" in m]


def test_a_non_constant_weight_is_sampled_on_the_union_and_said():
    """JEFF-4.0's NK=8: p_k a histogram/lin-lin mix. Exact at every node it shares."""
    from kika.endf.model_adapter.fission_energy import _familyMultiplicity
    from kika.nuclear_data.model import ConversionReport, Regions1d

    nubar = Regions1d.fromEndfRegions([1e-5, 1e6, 2e7], [0.0167, 0.0167, 0.0091], [(3, 2)])
    weight = Regions1d.fromEndfRegions([1e-5, 4e6, 4e6, 2e7], [0.03, 0.03, 0.04, 0.04],
                                       [(2, 1), (4, 2)])
    report = ConversionReport()
    multiplicity = _familyMultiplicity(nubar, weight, "1", report)
    for energy, p in ((1e6, 0.03), (2e7, 0.04)):
        assert float(np.asarray(multiplicity.form.evaluate(energy))) == pytest.approx(
            p * float(np.asarray(nubar.evaluate(energy))), rel=1e-12)
    assert any("union" in line for line in report.approximations)


# ---------------------------------------------------------------------------
# the two laws with no tape here
# ---------------------------------------------------------------------------

def _handBuilt(lf, tabs, headers=None):
    from kika.endf.tests.test_mf5_analytic import one_partial

    from kika.endf.classes.mf5.base import MF5MT

    partial = one_partial(float(lf), 0.0, tabs, headers=headers)
    section = MF5MT(number=18)
    section._za, section._awr, section._mat, section._nk = 92235.0, 233.0248, 9999, 1
    section.partials = [partial]
    return section, partial


def test_watt_and_madland_nix_decode_to_their_nodes_and_agree_with_the_reader():
    from kika.endf.tests.test_mf5_analytic import A_TAB, B_TAB, EFH, EFL, T_M

    for lf, tabs, headers, cls in ((11, [A_TAB, B_TAB], None, Watt),
                                    (12, [T_M], [(EFL, EFH)], MadlandNix)):
        section, partial = _handBuilt(lf, tabs, headers)
        form, _provenance, _report = decodeMF5MT(section)
        assert type(form) is cls
        grid = np.geomspace(1e3, 3e7, 300)
        for energy in (0.0253, 2.0e6):
            assert np.allclose(form.evaluate(energy, grid),
                               partial.evaluate_on_grid(energy, grid), rtol=1e-13, atol=0)
    assert form.EFL.value == EFL and form.EFH.value == EFH
    assert form.averageEnergy(1e-5) == pytest.approx(0.5 * (EFL + EFH) + 4.0 / 3.0 * 1.03e6)


# ---------------------------------------------------------------------------
# GNDS
# ---------------------------------------------------------------------------

def _gndsTrip(suite, tmp_path):
    out = tmp_path / "written.gnds.xml"
    kika.write(suite, str(out), format="gnds")
    return kika.read(str(out)), out


@pytest.mark.parametrize("path, mt", [(RA223, 18), (O17, 16), (U233, 18), (AM241, 18)])
def test_the_spectrum_survives_gnds(path, mt, tmp_path):
    suite = kika.read(str(path))
    back, _out = _gndsTrip(suite, tmp_path)
    assert _walk(_energy(back, mt)) == _walk(_energy(suite, mt))


def test_watt_and_madland_nix_survive_gnds(tmp_path):
    """Planted on Ra-223's MT18 in place of its Maxwellian -- GNDS cares not."""
    suite = kika.read(str(RA223))
    product = suite.findReactionByENDF_MT(18).outputChannel.products.byPid("n")[0]
    eV = lambda v: PhysicalQuantity(v, "eV")
    for form in (
        Watt(U=eV(-3.0e7), a=XYs1d(xs=[1e-5, 2e7], ys=[9.88e5, 1.0e6]),
             b=XYs1d(xs=[1e-5, 2e7], ys=[2.249e-6, 2.3e-6])),
        MadlandNix(EFL=eV(1.06e6), EFH=eV(0.52e6),
                   T_M=XYs1d(xs=[1e-5, 2e7], ys=[1.03e6, 1.2e6])),
    ):
        product.distribution["eval"].energy = form
        back, _out = _gndsTrip(suite, tmp_path)
        assert _walk(_energy(back, 18)) == _walk(form)


def test_the_delayed_families_survive_gnds(tmp_path):
    suite = kika.read(str(U235))
    back, _out = _gndsTrip(suite, tmp_path)
    one = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData.delayedNeutrons
    two = back.findReactionByENDF_MT(18).outputChannel.fissionFragmentData.delayedNeutrons
    for a, b in zip(one, two):
        assert _walk(a.product.distribution["eval"].energy) == \
            _walk(b.product.distribution["eval"].energy)
        assert _walk(a.product.multiplicity.form) == _walk(b.product.multiplicity.form)


@pytest.mark.parametrize("path", [RA223, O17, U235, U233, AM241])
def test_the_written_spectra_are_schema_valid(path, tmp_path):
    from kika.gnds.tests.test_encode import _schemaErrors

    _back, out = _gndsTrip(kika.read(str(path)), tmp_path)
    tags = {"evaporation", "generalEvaporation", "simpleMaxwellianFission", "Watt",
            "MadlandNix", "weightedFunctionals", "weighted", "U", "theta", "g",
            "a", "b", "EFL", "EFH", "T_M", "delayedNeutron", "rate"}
    errors = [e for e in _schemaErrors(out) if e.split("'")[1] in tags]
    assert errors == []


# ---------------------------------------------------------------------------
# PD-3: the sampler refuses a parametrised spectrum by name
# ---------------------------------------------------------------------------

def test_an_mf35_perturbation_of_a_parametrised_spectrum_is_refused_by_name():
    from kika.sampling.perturbation_set import PerturbationSet

    suite = kika.read(str(RA223))
    reaction = suite.findReactionByENDF_MT(18)
    with pytest.raises(ValueError, match=r"SimpleMaxwellianFission.*PD-3"):
        PerturbationSet._energyOf(reaction, 18)


def test_every_parametrised_node_is_outside_function2d():
    assert not any(issubclass(cls, Function2d) for cls in ANALYTIC_SPECTRA)
    assert Evaporation in ANALYTIC_SPECTRA

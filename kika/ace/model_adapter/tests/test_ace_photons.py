"""ACE photon production → photon products.

Two kinds of test. The **synthetic** ones build the parsed ACE objects by hand
and run in the fast lane: one entry per mapping row of
:mod:`kika.ace.model_adapter.photons`, so a regression names the row. The
**tape** ones read Fe-56 and check two things no synthetic table can:

* that Σ_i y_i(E)·σ_i(E) over the decoded products rebuilds the GPD block,
  which NJOY computed independently from the same blocks, to the 7 digits ACE
  prints; and
* that the decoded lines agree with what the ENDF adapter makes of the
  evaluation the table was processed from. ENDF/B-VIII.1 states Fe-56's levels
  as MF12 LO=2 cascades, which the ENDF adapter puts in PoPs; ACE has the same
  cascades already expanded into lines. Walking the PoPs cascade gives the
  line yields back, and they must match.

The ``.80c`` table is ENDF/B-VIII.0 and the tape is VIII.1, so for it only the
yields are compared: the 846.8 keV line production at 2 MeV differs by 11 %
between the two, and all of it is σ(MT51) (0.587 b in VIII.0, 0.528 b in
VIII.1); the yield 0.9997041 is the same in both.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kika.ace.classes.angular_distribution.distributions import (
    EquiprobableAngularDistribution, IsotropicAngularDistribution)
from kika.ace.classes.energy_distribution.distributions.discrete import \
    DiscreteEnergyDistribution
from kika.ace.classes.energy_distribution.distributions.tabular import \
    ContinuousTabularDistribution
from kika.ace.classes.photon_production_xs import (DirectCrossSection,
                                                    PhotonProductionCrossSections,
                                                    YieldBasedCrossSection)
from kika.ace.model_adapter import aceLabels, attachAcePhotons
from kika.nuclear_data.model import (ConversionReport, CrossSection, DiscreteGamma,
                                     Interpolation, Isotropic2d, OutputChannel,
                                     PrimaryGamma, Reaction, ReactionId, ReactionSuite,
                                     XYs1d, XYs2d, crossSectionAxes)


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

GRID_MEV = np.array([1e-11, 1.0, 2.0, 5.0, 20.0])

#: The synthetic table has no header temperature, so its label is the 0 K one.
LABEL = "heated_0K"


def _reaction(mt, ys):
    xs = CrossSection()
    xs[LABEL] = XYs1d(xs=GRID_MEV * 1e6, ys=np.asarray(ys, dtype=float),
                      interpolation=Interpolation.linlin,
                      axes=crossSectionAxes(), label=LABEL)
    return Reaction(id=ReactionId(label=f"MT{mt}", ENDF_MT=mt), crossSection=xs,
                    outputChannel=OutputChannel())


def _yield(mtrp, mtmult, mftype=12, e=(1.0, 20.0), y=(0.5, 0.5)):
    return YieldBasedCrossSection(mt=mtrp, mftype=mftype, mtmult=mtmult,
                                  energies=np.asarray(e), yields=np.asarray(y),
                                  num_energies=len(e))


@pytest.fixture()
def synthetic():
    suite = ReactionSuite(evaluation="synthetic", projectile="n", target="Fe56")
    suite.reactions.append(_reaction(51, [0.0, 0.5, 0.5, 0.4, 0.1]))
    suite.reactions.append(_reaction(102, [1.0, 0.0, 0.002, 0.001, 0.001]))

    sigp = PhotonProductionCrossSections(has_data=True)
    sigp.cross_sections = {
        51001: _yield(51001, 51),                                 # line, LP=0
        102001: _yield(102001, 102, mftype=16, e=(1e-11, 20.0), y=(2.0, 3.0)),
        102002: _yield(102002, 102, e=(1e-11, 20.0), y=(1.0, 1.0)),  # primary
        102003: DirectCrossSection(mt=102003, mftype=13, energy_grid_index=1,
                                   num_entries=5,
                                   cross_sections=np.array([0.5, 0.0, 0.001, 0.0005, 0.0005])),
        4001: DirectCrossSection(mt=4001, mftype=13, energy_grid_index=3,
                                 num_entries=3, cross_sections=np.array([0.1, 0.2, 0.3])),
        16001: _yield(16001, 16),                                 # no MT16 here
        51002: _yield(51002, 51),                                 # unknown law
    }
    law4 = ContinuousTabularDistribution(
        law=4, nbt=[2], interp=[2], n_energies=2, incident_energies=[1e-11, 20.0],
        distributions=[{"intt": 2, "n_discrete": 0, "e_out": np.array([0.0, 2.0]),
                        "pdf": np.array([0.5, 0.5]), "cdf": np.array([0.0, 1.0])}] * 2)
    energy = {
        51001: [DiscreteEnergyDistribution(eg=0.846778, lp=0)],
        102001: [law4],
        102002: [DiscreteEnergyDistribution(eg=7.646, lp=2)],
        102003: [DiscreteEnergyDistribution(eg=0.352, lp=0)],
        4001: [DiscreteEnergyDistribution(eg=1.0, lp=0)],
        16001: [DiscreteEnergyDistribution(eg=1.0, lp=0)],
        51002: [SimpleNamespace(law=9, law_name="Evaporation Spectrum")],
    }
    bins = [np.linspace(-1.0, 1.0, 33), np.linspace(-1.0, 1.0, 33)]
    angular = {mtrp: IsotropicAngularDistribution(mt=mtrp) for mtrp in sigp.cross_sections}
    angular[51001] = EquiprobableAngularDistribution(mt=51001, _energies=np.array([1.0, 20.0]),
                                                     _cosine_bins=bins)
    ace = SimpleNamespace(
        photon_production_xs=sigp,
        angular_distributions=SimpleNamespace(photon_production=angular),
        energy_distributions=SimpleNamespace(photon_production=energy),
        esz_block=SimpleNamespace(energies=GRID_MEV),
        photon_production_data=None,
    )
    report = attachAcePhotons(ace, suite, ConversionReport())
    return suite, report


def _photons(node):
    return [p for p in node.outputChannel.products if p.pid == "photon"]


def test_a_discrete_line_is_a_discrete_gamma_on_its_reaction(synthetic):
    suite, _ = synthetic
    first, second = _photons(suite.findReactionByENDF_MT(51))
    assert (first.label, second.label) == ("photon", "photon__a")
    form = first.distribution[LABEL]
    assert isinstance(form.energy, DiscreteGamma)
    assert form.energy.value == pytest.approx(846778.0)
    # The line's domain is its multiplicity's, as the ENDF adapter writes it.
    assert (form.energy.domainMin, form.energy.domainMax) == (1e6, 20e6)
    np.testing.assert_allclose(first.multiplicity.form.ys, [0.5, 0.5])


def test_equiprobable_bins_become_a_normalised_histogram(synthetic):
    suite, _ = synthetic
    angular = _photons(suite.findReactionByENDF_MT(51))[0].distribution[LABEL].angular
    assert isinstance(angular, XYs2d)
    for function in angular.function1ds:
        assert function.interpolation == Interpolation.flat
        assert np.sum(np.diff(function.xs) * function.ys[:-1]) == pytest.approx(1.0)


def test_lp2_is_a_primary_gamma_and_law4_a_continuum(synthetic):
    suite, _ = synthetic
    photons = {p.label: p.distribution[LABEL] for p in _photons(suite.findReactionByENDF_MT(102))}
    assert isinstance(photons["photon"].energy, XYs2d)
    spectrum = photons["photon"].energy.function1ds[0]
    # MeV and 1/MeV become eV and 1/eV, and the density still integrates to 1.
    assert spectrum.xs[-1] == pytest.approx(2e6)
    assert np.trapezoid(spectrum.ys, spectrum.xs) == pytest.approx(1.0)
    assert isinstance(photons["photon__a"].energy, PrimaryGamma)
    assert photons["photon__a"].energy.value == pytest.approx(7.646e6)
    assert isinstance(photons["photon"].angular, Isotropic2d)


def test_mftype13_is_divided_by_sigma_and_says_so(synthetic):
    suite, report = synthetic
    photon = _photons(suite.findReactionByENDF_MT(102))[2]
    # σ_γ / σ, and 0 (declared) where σ = 0 at 1 MeV.
    np.testing.assert_allclose(photon.multiplicity.form.ys, [0.5, 0.0, 0.5, 0.5, 0.5])
    assert any("MFTYPE 13" in line and "σ_MT = 0 at 1 point" in line
               for line in report.approximations)


def test_mftype13_with_no_reaction_is_an_orphan_carrying_sigma_gamma(synthetic):
    suite, report = synthetic
    (orphan,) = suite.orphanProducts
    assert orphan.ENDF_MT == 4
    np.testing.assert_allclose(orphan.crossSection[LABEL].ys, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(orphan.crossSection[LABEL].xs, [2e6, 5e6, 20e6])
    (photon,) = _photons(orphan)
    assert photon.multiplicity.form.constant == 1.0
    assert any("orphan product" in line for line in report.approximations)


def test_two_photons_on_one_missing_mt_each_keep_their_sigma_gamma():
    """ENDF/B-VIII.1 N-14 states six MFTYPE 13 photons on MT 28, which it has no σ for.

    One orphan per MT carries the sum of their σ_γ and each photon its share,
    so y·Σ gives every photon its own σ_γ back. Before, the orphan held the
    first photon's σ_γ and the others, with y = 1, silently took its numbers.
    """
    suite = ReactionSuite(evaluation="synthetic", projectile="n", target="N14")
    suite.reactions.append(_reaction(51, [0.0, 0.5, 0.5, 0.4, 0.1]))
    sigp = PhotonProductionCrossSections(has_data=True)
    sigp.cross_sections = {
        28001: DirectCrossSection(mt=28001, mftype=13, energy_grid_index=2, num_entries=4,
                                  cross_sections=np.array([0.0, 0.2, 0.4, 0.4])),
        28002: DirectCrossSection(mt=28002, mftype=13, energy_grid_index=3, num_entries=3,
                                  cross_sections=np.array([0.1, 0.1, 0.3])),
    }
    energy = {k: [DiscreteEnergyDistribution(eg=1.0, lp=0)] for k in sigp.cross_sections}
    angular = {k: IsotropicAngularDistribution(mt=k) for k in sigp.cross_sections}
    ace = SimpleNamespace(photon_production_xs=sigp,
                          angular_distributions=SimpleNamespace(photon_production=angular),
                          energy_distributions=SimpleNamespace(photon_production=energy),
                          esz_block=SimpleNamespace(energies=GRID_MEV), photon_production_data=None)
    attachAcePhotons(ace, suite, ConversionReport())
    (orphan,) = suite.orphanProducts
    sigma = orphan.crossSection[LABEL]
    np.testing.assert_allclose(sigma.xs, GRID_MEV[1:] * 1e6)
    np.testing.assert_allclose(sigma.ys, [0.0, 0.3, 0.5, 0.7])
    for photon, expected in zip(_photons(orphan), ([0.0, 0.2, 0.4, 0.4], [0.1, 0.1, 0.3])):
        y = photon.multiplicity.form
        np.testing.assert_allclose(y.ys * np.interp(y.xs, sigma.xs, sigma.ys), expected)


def test_what_cannot_be_decoded_is_reported_and_does_not_raise(synthetic):
    suite, report = synthetic
    assert any("MTRP 16001" in line and "MT16" in line for line in report.unsupported)
    assert any("MTRP 51002" in line and "law 9" in line for line in report.unsupported)
    # The law-9 photon keeps its multiplicity and angle; only the energy half is empty.
    photon = _photons(suite.findReactionByENDF_MT(51))[1]
    assert photon.distribution[LABEL].energy is None
    assert photon.multiplicity is not None


# ---------------------------------------------------------------------------
# Fe-56, real tables
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[4]


def _aceTable(name):
    roots = [Path(os.environ["KIKA_TAPES"]) / "ace"] if os.environ.get("KIKA_TAPES") else []
    roots.append(_REPO.parent / "kika-files" / "ace")
    for root in roots:
        if (root / name).is_file():
            return root / name
    pytest.skip(f"{name} not found under {[str(r) for r in roots]}")


@pytest.fixture(scope="module", params=["260560_80.02c", "260560_81.02c"])
def fe56(request):
    from kika.ace import read_ace
    from kika.ace.model_adapter import decodeAce

    ace = read_ace(str(_aceTable(request.param)))
    suite, report = decodeAce(ace)
    return request.param, ace, suite, report


@pytest.mark.tape
def test_every_mtrp_entry_becomes_a_photon(fe56):
    _, ace, suite, report = fe56
    count = sum(len(_photons(r)) for r in list(suite.reactions) + list(suite.orphanProducts))
    assert count == len(ace.photon_production_xs.cross_sections)
    assert not [line for line in report.unsupported if "MTRP" in line]


@pytest.mark.tape
def test_the_decoded_photons_rebuild_the_gpd_total(fe56):
    """Σ y·σ from the products equals the GPD block NJOY wrote, to ACE's 7 digits."""
    _, ace, suite, _ = fe56
    energies, total = ace.photon_production_total()
    E = energies * 1e6
    rebuilt = np.zeros_like(E)
    for reaction in suite.reactions:
        sigma = reaction.crossSection[aceLabels(ace)[0]]
        for photon in _photons(reaction):
            rebuilt += photon.multiplicity.form.evaluate(E) * np.interp(E, sigma.xs, sigma.ys)
    np.testing.assert_allclose(rebuilt, total, rtol=2e-6, atol=1e-12)
    np.testing.assert_array_equal(suite.photonProductionTotal.ys, total)


@pytest.fixture(scope="module")
def endf_suite(fe56_b81_tape):
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _ = decodeReactionSuite(read_endf(str(fe56_b81_tape)))
    return suite


def _cascadeLines(particles, pid, weight=1.0, out=None):
    """``{E_γ (eV, rounded): yield}`` of a level's whole cascade, from PoPs."""
    out = out if out is not None else defaultdict(float)
    level = particles.get(pid)
    decay = getattr(level, "decayData", None)
    if decay is None:
        return out
    for mode in decay.decayModes:
        final = particles.get(mode.finalState())
        finalEnergy = float(final.energy.value) if getattr(final, "energy", None) is not None else 0.0
        gp = mode.photonEmissionProbabilities.total() if mode.photonEmissionProbabilities else 1.0
        out[round(float(level.energy.value) - finalEnergy)] += weight * float(mode.probability) * gp
        _cascadeLines(particles, mode.finalState(), weight * float(mode.probability), out)
    return out


def _aceLines(reaction, energy, label):
    lines = defaultdict(float)
    for photon in _photons(reaction):
        lines[round(photon.distribution[label].energy.value)] += \
            float(photon.multiplicity.form.evaluate(energy))
    return lines


@pytest.mark.tape
def test_the_8468_kev_line_at_2_mev_agrees_with_the_endf_cascade(fe56, endf_suite):
    name, table, suite, _ = fe56
    label = aceLabels(table)[0]
    E = 2e6
    ace = suite.findReactionByENDF_MT(51)
    endf = endf_suite.findReactionByENDF_MT(51)
    aceYield = _aceLines(ace, E, label)[846778]
    endfYield = _cascadeLines(endf_suite.PoPs.particles, "Fe56_e1")[846778]
    assert aceYield == pytest.approx(endfYield, rel=1e-6)

    if name.endswith("_81.02c"):
        sigmaAce = np.interp(E, ace.crossSection[label].xs, ace.crossSection[label].ys)
        sigmaEndf = float(endf.crossSection["eval"].evaluate(E))
        assert aceYield * sigmaAce == pytest.approx(endfYield * sigmaEndf, rel=1e-3)


@pytest.mark.tape
def test_every_level_line_of_the_b81_table_matches_the_endf_cascade(fe56, endf_suite):
    """Every line of MT51-90 at 5 MeV, against the PoPs cascade walked by hand.

    The tolerance is 1e-3 absolute on a yield: ACER's expansion and the PoPs
    walk differ in how internal conversion is carried down the cascade, and the
    largest gap seen is 2.7e-4 (MT84's 846.8 keV line).
    """
    name, table, suite, _ = fe56
    if not name.endswith("_81.02c"):
        pytest.skip("the .80c table is ENDF/B-VIII.0; the tape is VIII.1")
    label = aceLabels(table)[0]
    E, compared = 5e6, 0
    for mt in range(51, 91):
        ace = suite.findReactionByENDF_MT(mt)
        if ace is None or not _photons(ace):
            continue
        cascade = _cascadeLines(endf_suite.PoPs.particles, f"Fe56_e{mt - 50}")
        for energy, value in _aceLines(ace, E, label).items():
            expected = sum(v for k, v in cascade.items() if abs(k - energy) < 300)
            assert value == pytest.approx(expected, abs=1e-3), (mt, energy)
            compared += 1
    assert compared > 300


@pytest.mark.tape
def test_the_mf6_photons_have_the_endf_multiplicities(fe56, endf_suite):
    name, _, suite, _ = fe56
    if not name.endswith("_81.02c"):
        pytest.skip("the .80c table is ENDF/B-VIII.0; the tape is VIII.1")
    for mt in (5, 16, 91, 102):
        for E in (1e3, 2e6, 14e6):
            ace = sum(float(p.multiplicity.form.evaluate(E)) for p in _photons(suite.findReactionByENDF_MT(mt)))
            endf = sum(float(p.multiplicity.form.evaluate(E))
                       for p in _photons(endf_suite.findReactionByENDF_MT(mt)))
            assert ace == pytest.approx(endf, rel=1e-4, abs=1e-6), (mt, E)

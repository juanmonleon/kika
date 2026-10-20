"""ACE's secondary-neutron blocks reach the model, tape-free.

On the synthetic table of ``kika/ace/tests/_synthetic.py`` (elastic, MT16 with
LAW=44 and TY=-2, MT51 with LAW=3, MT102) and on small stubs for the fission
blocks. The real gate -- every dataset of OpenMC's ``from_ace`` file against
kika's -- runs outside the test suite.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.model_adapter import decodeAce
from kika.ace.model_adapter.secondaries import _delayedNeutrons, _fissionNeutron
from kika.ace.tests._synthetic import AWR, write_table
from kika.nuclear_data import model as m

MC = "MonteCarlo"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    ace = read_ace(str(write_table(tmp_path_factory.mktemp("ace") / "t.ace")))
    return decodeAce(ace)[0]


def _only(reaction):
    (product,) = reaction.outputChannel.products.products
    return product


def test_elastic_is_a_two_body_angular_distribution_in_ev(suite):
    form = _only(suite.reactionByENDF_MT(2)).distribution[MC]
    assert isinstance(form, m.AngularTwoBody)
    assert form.productFrame == m.Frame.centerOfMass
    t0, t1 = form.angular.function1ds
    assert t0.outerDomainValue == pytest.approx(1.0e-11 * 1e6) and t1.outerDomainValue == pytest.approx(20e6)
    np.testing.assert_array_equal(t0.xs, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(t0.pdf, [0.25, 0.5, 0.75])


def test_n2n_has_two_neutrons_in_the_cm_with_its_kalbach_mann_law(suite):
    product = _only(suite.reactionByENDF_MT(16))
    assert product.multiplicity[MC].coefficients.tolist() == [2.0]
    mixture = product.distribution[MC]
    assert isinstance(mixture, m.DistributionMixture) and len(mixture) == 1
    (weighted,) = mixture.weighted
    km = weighted.functional
    assert isinstance(km, m.KalbachMann) and km.productFrame == m.Frame.centerOfMass
    np.testing.assert_allclose(weighted.weight.xs, [1.0e-5, 20.0e6])
    first, second = km.f.function1ds
    np.testing.assert_allclose(second.xs, [0.0, 2.0e6, 8.0e6])
    np.testing.assert_allclose(second.pdf, np.array([0.125, 0.125, 0.125]) / 1e6)
    np.testing.assert_array_equal(km.r.function1ds[1].ys, [0.1, 0.2, 0.3])
    assert product.provenance.attributes["TY"] == -2


def test_a_level_reaction_is_level_energy_with_its_angular_table(suite):
    form = _only(suite.reactionByENDF_MT(51)).distribution[MC].weighted[0].functional
    assert isinstance(form, m.Uncorrelated)
    assert isinstance(form.energy, m.LevelEnergy)
    assert form.energy.threshold == pytest.approx((AWR + 1) / AWR * 0.8467e6)
    assert form.energy.massRatio == pytest.approx((AWR / (AWR + 1)) ** 2)
    (table,) = form.angular.function1ds
    np.testing.assert_array_equal(table.xs, [-1.0, 0.0, 1.0])


def test_summed_reactions_are_sums_not_reactions(suite):
    # kika adds MT 3, 4, 27, 101... as sums of what the file has; left among the
    # reactions they would be counted twice by whatever adds reactions up.
    reactions = {r.ENDF_MT for r in suite.reactions}
    sums = {r.ENDF_MT for r in suite.sums}
    assert reactions >= {2, 16, 51, 102}
    assert not reactions & {1, 3, 4, 27, 101}
    assert {1, 3, 4, 101} <= sums
    assert isinstance(suite.reactionByENDF_MT(4), m.CrossSectionSum)


# --- fission ------------------------------------------------------------------

def _nu(values):
    return types.SimpleNamespace(format="tabulated", polynomial=None, tabulated=types.SimpleNamespace(
        interpolation_regions=[], energies=np.array([1e-11, 20.0]), nubar_values=np.asarray(values, float)))


def _fission_ace(prompt=True, groups=((0.0127, 0.6, 0.6), (0.0317, 0.4, 0.4))):
    nubar = types.SimpleNamespace(has_nubar=True, has_both_nu_types=prompt, prompt=_nu([2.4, 2.6]),
                                  total=_nu([2.416, 2.616]), has_delayed=True, delayed=_nu([0.016, 0.016]))
    bdd = types.SimpleNamespace(precursors=[types.SimpleNamespace(
        decay_constant=lam, interpolation_regions=[], energies=np.array([1e-11, 20.0]),
        probabilities=np.array([p0, p1])) for lam, p0, p1 in groups])
    return types.SimpleNamespace(nubar=nubar, delayed_neutron_data=bdd,
                                 energy_distributions=types.SimpleNamespace(delayed_neutron=[[] for _ in groups]))


def test_prompt_and_total_nu_with_both_given():
    prompt, total, mode = _fissionNeutron(_fission_ace())
    assert mode == "prompt"
    np.testing.assert_array_equal(prompt.ys, [2.4, 2.6])
    np.testing.assert_allclose(total.xs, [1e-5, 20e6])


def test_delayed_groups_carry_their_rate_in_per_second_and_keep_their_probabilities():
    report = m.ConversionReport()
    groups, _ = _delayedNeutrons(_fission_ace(groups=((0.0127, 0.6, 0.6), (0.0317, 0.3, 0.3))), AWR, report)
    assert [g.rate.value for g in groups] == pytest.approx([0.0127e8, 0.0317e8])
    assert groups[0].rate.unit == "1/s"
    # 0.6 + 0.3 != 1: kept as the file states it (OpenMC would renormalise)
    np.testing.assert_allclose(groups[1].product.multiplicity[MC].ys, [0.016 * 0.3] * 2)


def test_an_energy_dependent_group_probability_multiplies_on_the_union_grid():
    ace = _fission_ace(groups=((0.0127, 0.5, 0.7),))
    (group,), _ = _delayedNeutrons(ace, AWR, m.ConversionReport())
    np.testing.assert_allclose(group.product.multiplicity[MC].ys, [0.016 * 0.5, 0.016 * 0.7])

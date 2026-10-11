"""kika's own KERMA and damage (roadmap H8): the pieces against closed forms.

The whole calculation is checked against HEATR on real evaluations in
``kika-workspace/kika_dev/processing/heating`` (Fe-56, ENDF/B-VIII.1: elastic
and capture to < 0.1 %, the total per collision to the medians listed in
``docs/library/heating_kerma_damage_plan.md``).
"""
import numpy as np
import pytest

from kika.nuclear_data import model as m
from kika.processing import heating as H


def _table(xs, pdf, E, flat=False, discrete=0):
    from kika.processing.tabulation.tables import pdfTable

    table, _ = pdfTable(np.asarray(xs, float), np.asarray(pdf, float), histogram=flat,
                        discrete=discrete, outer=E)
    return table


def test_displacement_energy_is_heatrs_table():
    assert H.displacementEnergy(26) == 40.0
    assert H.displacementEnergy(92) == 25.0
    assert H.displacementEnergy(6) == 31.0
    assert H.displacementEnergy(74) == 90.0


def test_lindhard_has_a_hard_step_and_no_damage_without_charge():
    E = np.array([10.0, 39.9, 40.0, 1e5])
    d = H.lindhard(E, 26, 55.85, 26, 55.85, 40.0)
    assert d[0] == 0 and d[1] == 0 and d[2] > 0 and 0 < d[3] < 1e5
    assert np.all(H.lindhard(E, 0, 1.0, 26, 55.85, 40.0) == 0)
    # Robinson's formula by hand at 100 keV, Fe in Fe.
    z, a = 26.0, 55.85
    el = 30.724 * z * z * np.sqrt(2 * z ** (2 / 3)) * 2
    kl = 0.0793 * z ** (2 / 3) * np.sqrt(z) * (2 * a) ** 1.5 / ((2 * z ** (2 / 3)) ** 0.75 * a ** 1.5 * np.sqrt(a))
    eps = 1e5 / el
    assert H.lindhard(1e5, z, a, z, a, 40.0) == pytest.approx(
        1e5 / (1 + kl * (3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps)))


def test_expectations_over_a_table_are_exact_for_polynomials():
    t = _table([0.0, 1.0, 3.0], [1.0, 1.0, 0.0], 1.0)
    # density 1 on [0,1], 1 -> 0 on [1,3]: normalised by 2.
    assert H._expect(t, lambda x: np.ones_like(x)) == pytest.approx(1.0)
    mean = (0.5 + 5.0 / 3.0) / 2.0
    assert H._expect(t, lambda x: x) == pytest.approx(mean)


def test_a_line_counts_with_its_weight():
    # A line of weight 0.5 at 2, and a continuum of integral 1 on [0, 1].
    t = _table([2.0, 0.0, 1.0], [0.5, 1.0, 1.0], 1.0, discrete=1)
    assert H._expect(t, lambda x: x) == pytest.approx((0.5 * 2.0 + 1.0 * 0.5) / 1.5)


def test_isotropic_elastic_deposits_two_a_over_a_plus_one_squared():
    A = 55.454
    mu = _table([-1.0, 1.0], [0.5, 0.5], 1.0)
    angular = m.XYs2d(function1ds=[mu, _table([-1.0, 1.0], [0.5, 0.5], 2e7)])
    form = m.AngularTwoBody(angular=angular, productFrame=m.Frame.centerOfMass)
    E = np.array([1.0, 1e3, 1e6])
    mean = H._meanEnergyOf(form, E, H._Kinematics(A=A, Q=0.0))
    assert np.allclose(E - mean, E * 2 * A / (A + 1) ** 2)


def test_a_level_takes_the_excitation_out_of_the_neutron():
    A, Q = 55.454, -846778.0
    mu = _table([-1.0, 1.0], [0.5, 0.5], 1e6)
    angular = m.XYs2d(function1ds=[mu, _table([-1.0, 1.0], [0.5, 0.5], 2e7)])
    level = m.LevelEnergy(threshold=(A + 1) / A * -Q, massRatio=(A / (A + 1)) ** 2)
    form = m.Uncorrelated(angular=angular, energy=level, productFrame=m.Frame.centerOfMass)
    E = np.array([2e6])
    mean = H._meanEnergyOf(form, E, H._Kinematics(A=A, Q=Q))
    r2 = 1 - (A + 1) / A * -Q / E
    assert mean == pytest.approx(E * (1 + A * A * r2) / (A + 1) ** 2)


def test_photon_lines_ignore_the_frame_and_primaries_take_the_neutron_energy():
    line = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=0, energy=846778.0,
                                                        atomicWeightRatio=55.454),
                          productFrame=m.Frame.centerOfMass)
    E = np.array([1e6, 1e7])
    k = H._Kinematics(A=55.454, Q=0.0, awp=0.0)
    assert np.allclose(H._meanEnergyOf(line, E, k), 846778.0)
    primary = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2, energy=7.6e6,
                                                           atomicWeightRatio=55.454))
    assert np.allclose(H._meanEnergyOf(primary, E, k), 7.6e6 + 55.454 / 56.454 * E)

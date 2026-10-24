"""The native batches over incident energies are the per-energy quadratures, bit for bit.

``twobody_batch``, ``emission_batch`` and ``unitbase_batch`` do in C the ``at(E)`` of
_twoBodyDamage, _tabulatedEmissionDamage and _unitBaseCMMean (and their joint forms)
for many energies at once. Each case here runs the same call twice, with the batches
and with each energy through its Python ``at`` (which calls the chunk kernels), on one
thread and on four, and wants the same array.
"""
import numpy as np
import pytest

from kika.nuclear_data import model as m
from kika.processing import heating as H

pytest.importorskip("kika.processing._heating_native")


def _table(xs, pdf, E, flat=False):
    from kika.processing.tabulation.tables import pdfTable

    table, _ = pdfTable(np.asarray(xs, float), np.asarray(pdf, float), histogram=flat, discrete=0, outer=E)
    return table


def _per_energy(monkeypatch):
    """Every quadrature through its Python at(E), as before the batches."""
    for name in ("_twoBodyBatch", "_emissionBatch", "_unitBaseBatch"):
        monkeypatch.setattr(H, name, lambda at, *args, **kwargs: at)


def _both(monkeypatch, call, threads):
    token = H._energyThreads.set(threads)
    try:
        batched = call()
        with monkeypatch.context() as patch:
            _per_energy(patch)
            reference = call()
    finally:
        H._energyThreads.reset(token)
    return np.asarray(batched), np.asarray(reference)


def _angular(flat=False):
    return m.XYs2d(function1ds=[_table([-1., -.2, .4, 1.], [.2, .5, .7, .6], E, flat) for E in (1e3, 2e5, 2e7)],
                   interpolation=m.Interpolation.flat if flat else m.Interpolation.linlin)


def _spectrum(flat=False, low=0.):
    return m.XYs2d(function1ds=[
        _table([low, 2e4, 3e5, 1e6], [1., 2., .5, 0.], 1e5, flat),
        _table([low, 5e4, 1e6, 4e6], [.5, 1., .7, .1], 5e6, flat),
        _table([low, 1e5, 2e6, 9e6], [.2, .9, .6, 0.], 2e7, flat)],
        interpolation=m.Interpolation.flat if flat else m.Interpolation.linlin,
        interpolationQualifier=m.InterpolationQualifier.unitBase)


GRID = np.unique(np.r_[np.geomspace(1e2, 2e7, 40), 1e5, 5e6])
MODELS = tuple(H.displacementModels(26, 40.0).values())


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("model", [None, "nrt", "arc-dpa"])
@pytest.mark.parametrize("angular", ["isotropic", "linear", "flat"])
@pytest.mark.parametrize("elastic,Q", [(True, 0.), (False, -8.5e5)])
def test_two_body_batch_is_the_per_energy_quadrature(monkeypatch, threads, model, angular, elastic, Q):
    law = m.Isotropic2d() if angular == "isotropic" else _angular(angular == "flat")
    form = m.AngularTwoBody(angular=law, productFrame=m.Frame.centerOfMass)
    token = H._DISPLACEMENTS.set(None if model is None else H.displacementModels(26, 40.0)[model])
    try:
        batched, reference = _both(monkeypatch, lambda: H._twoBodyDamage(form, GRID, 55.4, Q, 26, 40.0, elastic),
                                   threads)
    finally:
        H._DISPLACEMENTS.reset(token)
    np.testing.assert_array_equal(batched, reference)
    assert np.any(batched > 0)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("angular", ["isotropic", "linear"])
def test_joint_two_body_batch_is_the_per_energy_quadrature(monkeypatch, threads, angular):
    law = m.Isotropic2d() if angular == "isotropic" else _angular()
    form = m.AngularTwoBody(angular=law, productFrame=m.Frame.centerOfMass)
    batched, reference = _both(
        monkeypatch, lambda: H._twoBodyDamageJoint(form, GRID, 55.4, 0., 26, 40.0, True, MODELS), threads)
    np.testing.assert_array_equal(batched, reference)
    assert batched.shape == (len(GRID), 3) and np.all(batched[-1] > 0)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("model", [None, "nrt", "arc-dpa"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("particle", [None, (2, 4.0, 1. / 56.), (2, 4.0, 0.), (0, 1.0, 1. / 56.)])
def test_emission_batch_is_the_per_energy_quadrature(monkeypatch, threads, model, flat, particle):
    spectrum = _spectrum(flat)
    token = H._DISPLACEMENTS.set(None if model is None else H.displacementModels(26, 40.0)[model])
    try:
        batched, reference = _both(
            monkeypatch, lambda: H._tabulatedEmissionDamage(spectrum, GRID, 55.4, 26, 40.0, particle=particle),
            threads)
    finally:
        H._DISPLACEMENTS.reset(token)
    np.testing.assert_array_equal(batched, reference)
    if particle is None or particle[0]:
        assert np.any(batched > 0)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("particle", [None, (2, 4.0, 1. / 56.), (2, 4.0, 0.)])
def test_joint_emission_batch_is_the_per_energy_quadrature(monkeypatch, threads, particle):
    spectrum = _spectrum()
    batched, reference = _both(
        monkeypatch,
        lambda: H._tabulatedEmissionDamageJoint(spectrum, GRID, 55.4, 26, 40.0, particle, MODELS), threads)
    np.testing.assert_array_equal(batched, reference)
    assert batched.shape == (len(GRID), 3) and np.all(batched[-1] > 0)


def _energy_angle(low):
    angles = m.XYs3d(function2ds=[
        m.XYs2d(function1ds=[_table([-1., 1.], [.5 - .3 * s * f, .5 + .3 * s * f], x)
                             for x, f in ((low, 1.), (2e5, .5), (4e6, .1))], outerDomainValue=E)
        for E, s in ((1e5, .2), (5e6, .5), (2e7, .9))])
    energy = m.XYs2d(function1ds=[_table([low, 2e5, 4e6], [1., .8, .1], E) for E in (1e5, 5e6, 2e7)],
                     interpolationQualifier=m.InterpolationQualifier.unitBase)
    return m.EnergyAngularMC(energy=energy, energyAngular=angles, productFrame=m.Frame.centerOfMass)


def _kalbach(low):
    energy = m.XYs2d(function1ds=[_table([low, 2e5, 4e6], [1., .8, .1], E) for E in (1e5, 5e6, 2e7)],
                     interpolationQualifier=m.InterpolationQualifier.unitBase)
    rs = m.XYs2d(function1ds=[m.XYs1d(xs=[low, 2e5, 4e6], ys=[.1, .3, .6], outerDomainValue=E) for E in (1e5, 5e6, 2e7)])
    aa = m.XYs2d(function1ds=[m.XYs1d(xs=[low, 2e5, 4e6], ys=[.005, .8, 2.5], outerDomainValue=E)
                              for E in (1e5, 5e6, 2e7)])
    return m.KalbachMann(f=energy, r=rs, a=aa, productFrame=m.Frame.centerOfMass)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("law", ["energy-angle", "kalbach"])
@pytest.mark.parametrize("photon", [False, True])
def test_unit_base_batch_is_the_per_energy_quadrature(monkeypatch, threads, law, photon):
    # A support that does not start at 0 takes the incident-energy mesh (at(E)).
    form = (_energy_angle if law == "energy-angle" else _kalbach)(1e3)
    k = H._Kinematics(A=55.4, Q=0., photon=photon)
    batched, reference = _both(monkeypatch, lambda: H._meanEnergyOf(form, GRID, k), threads)
    np.testing.assert_array_equal(batched, reference)
    assert np.all(np.isfinite(batched)) and np.any(batched > 0)


def test_batch_refuses_inconsistent_buffers():
    native = H._heating_native
    one = np.ones(1)
    with pytest.raises(ValueError, match="inconsistent"):
        native.twobody_batch(one, np.empty(1), np.empty(1), np.array([-1., 1.]), np.array([.5, .5]),
                             np.array([0., 2.]), one, one, one, np.zeros(3), np.zeros(3), one, one, one,
                             55.4, 1., 0., 0., 40., 1., 1., 40., False, False, 0, 0., 0., 1., 0, 0., 0., 1.)
    with pytest.raises((BufferError, ValueError)):
        native.twobody_batch(one, b"12345678", np.empty(1), np.array([-1., 1.]), np.array([.5, .5]),
                             np.array([0., 2.]), one, one, one, np.zeros(120), np.zeros(120), one, one, one,
                             55.4, 1., 0., 0., 40., 1., 1., 40., False, False, 0, 0., 0., 1., 0, 0., 0., 1.)

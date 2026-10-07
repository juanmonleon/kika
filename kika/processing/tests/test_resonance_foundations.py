"""Independent analytic/complex-amplitude checks for the new BW evaluator."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from kika.nuclear_data.model.resonances import (
    BreitWigner, BreitWignerApproximation, Resonance, ResonanceParameters,
    Resonances, ResolvedRegion, ScatteringRadius, SpinGroup, UnresolvedRegion,
)
from kika.processing.resonances import (
    NeutronContext, UnsupportedResonanceError, prepare_resonances,
)
from kika.processing.resonances.channel_functions import neutral_channel_functions


def model(approximation="MultiLevel", l=0, levels=None, radius=5.0):
    if levels is None:
        levels = [Resonance(100., l+.5, .3, .1, .2, 0.)]
    bw = BreitWigner(
        approximation=BreitWignerApproximation(approximation),
        resonanceParameters=ResonanceParameters([SpinGroup(l, levels)]),
        scatteringRadius=radius, radiusUnit="fm")
    return Resonances(resolved=[ResolvedRegion(1e-8, 1e7, formalism=bw)])


def test_lab_kinematics_and_mass_radius():
    ctx = NeutronContext(1., 0.)
    assert ctx.k_squared_per_ev == pytest.approx(2*939.56542052e-6/197.3269804**2/4)
    assert ctx.mass_channel_radius_fm == pytest.approx(1.23*1.00866491595**(1/3)+.8)
    assert NeutronContext(56., 0.).k_squared_per_ev/ctx.k_squared_per_ev == pytest.approx((112/57)**2)


@pytest.mark.parametrize("l", range(4))
def test_ps_against_bessel_wronskian(l):
    x = np.geomspace(1e-7, 100., 120)
    p, s, phi = neutral_channel_functions(l, x)
    f, g = x*spherical_jn(l, x), -x*spherical_yn(l, x)
    fp = spherical_jn(l, x)+x*spherical_jn(l, x, derivative=True)
    gp = -spherical_yn(l, x)-x*spherical_yn(l, x, derivative=True)
    np.testing.assert_allclose(p, x/(f*f+g*g), rtol=2e-13)
    np.testing.assert_allclose(s, x*(f*fp+g*gp)/(f*f+g*g), rtol=2e-12, atol=3e-14)
    np.testing.assert_allclose(np.sin(phi)**2, f*f/(f*f+g*g), rtol=2e-13, atol=1e-28)


def test_pwave_phase_small_argument():
    x = np.array([1e-8, 1e-5, .01])
    phi = neutral_channel_functions(1, x)[2]
    np.testing.assert_allclose(phi, x**3/3-x**5/5+x**7/7, rtol=1e-12)


@pytest.mark.parametrize("approximation", ["SingleLevel", "MultiLevel"])
def test_isolated_swave_partial_widths_and_complex_elastic(approximation):
    ctx = NeutronContext(56., 0.)
    source = model(approximation)
    energy = np.array([25., 99.85, 100., 100.15, 400.])
    result = prepare_resonances(source, ctx).evaluate(energy)
    k2 = ctx.k_squared_per_ev*energy
    gn = .1*np.sqrt(energy/100.)
    width = gn+.2
    denom = (energy-100.)**2+(width/2)**2
    np.testing.assert_allclose(result[102], np.pi*.01/k2*gn*.2/denom, rtol=2e-14)
    phi = np.sqrt(k2)*5.
    collision = np.exp(-2j*phi)*(1-1j*gn/(energy-100.+1j*width/2))
    np.testing.assert_allclose(result[2], np.pi*.01/k2*abs(1-collision)**2, rtol=2e-12)
    assert result[102][2] == pytest.approx(4*np.pi*.01/k2[2]*.1*.2/.3**2)
    np.testing.assert_array_equal(result[18], 0.)
    np.testing.assert_array_equal(result[1], result[2]+result[18]+result[102])


def test_multilevel_interference_and_block_order_invariance():
    levels = [Resonance(100., .5, .3, .1, .2), Resonance(100.1, .5, .4, .3, .1)]
    source = model(levels=levels)
    ctx = NeutronContext(56., 0.)
    prepared = prepare_resonances(source, ctx)
    energies = np.linspace(99., 101., 103)
    reference = prepared.evaluate(energies, block_size=10000)
    gn = np.array([.1, .3])[:, None]*np.sqrt(energies/np.array([100., 100.1])[:, None])
    total = gn+np.array([.2, .1])[:, None]
    amplitude = np.sum(gn/(energies-np.array([100., 100.1])[:, None]+.5j*total), axis=0)
    k2 = ctx.k_squared_per_ev*energies
    u = np.exp(-2j*np.sqrt(k2)*5.)*(1-1j*amplitude)
    np.testing.assert_allclose(reference[2], np.pi*.01/k2*abs(1-u)**2, rtol=2e-13)
    for mt, values in prepared.evaluate(energies[::-1], block_size=7).items():
        np.testing.assert_array_equal(values[::-1], reference[mt])
    source.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances.reverse()
    for mt, values in prepare_resonances(source, ctx).evaluate(energies).items():
        np.testing.assert_allclose(values, reference[mt], rtol=3e-15)
    single = prepare_resonances(model("SingleLevel", levels=levels), ctx).evaluate(energies)
    assert np.max(abs(single[2]-reference[2])) > 100
    np.testing.assert_allclose(single[102], reference[102])


@pytest.mark.parametrize("l", range(4))
def test_potential_only_includes_absent_spin_sectors(l):
    ctx = NeutronContext(56., .5)
    energy = np.array([1e-4, 100., 1e6])
    expected = 4*np.pi*.01/(ctx.k_squared_per_ev*energy)*(2*l+1)*np.sin(
        neutral_channel_functions(l, np.sqrt(ctx.k_squared_per_ev*energy)*5.)[2])**2
    for approximation in ("SingleLevel", "MultiLevel"):
        np.testing.assert_allclose(prepare_resonances(model(approximation, l, []), ctx).evaluate(energy)[2], expected)


def test_mass_radius_and_negative_level_shift_pwave():
    source = model(l=1, levels=[Resonance(-100., 1.5, .3, .1, .2)])
    source.resolved[0].formalism.calculateChannelRadius = True
    ctx = NeutronContext(56., 0.)
    prepared = prepare_resonances(source, ctx)
    assert prepared.regions[0].groups[0].channel_radius.constant == ctx.mass_channel_radius_fm
    energy = np.array([25., 400.])
    x = np.sqrt(ctx.k_squared_per_ev*energy)*ctx.mass_channel_radius_fm
    xr = np.sqrt(ctx.k_squared_per_ev*100.)*ctx.mass_channel_radius_fm
    p, pr = x**3/(1+x*x), xr**3/(1+xr*xr)
    s, sr = -1/(1+x*x), -1/(1+xr*xr)
    gn = .1*p/pr
    delta = energy-(-100.+.1*(sr-s)/(2*pr))
    expected = np.pi*.01/(ctx.k_squared_per_ev*energy)*2*gn*.2/(delta**2+((gn+.2)/2)**2)
    np.testing.assert_allclose(prepared.evaluate(energy)[102], expected, rtol=2e-12)


def test_snapshot_and_shared_boundary_and_gap():
    source = model()
    source.resolved[0].domainMax = 200.
    second = model().resolved[0]
    second.domainMin, second.domainMax = 200., 400.
    second.formalism.scatteringRadius = 7.
    source.resolved.append(second)
    prepared = prepare_resonances(source, NeutronContext(56., 0.))
    before = prepared.evaluate(100.)
    source.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0].captureWidth = 7.
    for mt in before:
        np.testing.assert_array_equal(before[mt], prepared.evaluate(100.)[mt])
    right = prepare_resonances(Resonances(resolved=[second]), prepared.context)
    np.testing.assert_array_equal(prepared.evaluate(200.)[2], right.evaluate(200.)[2])
    source.resolved[1].domainMin = 250.
    source.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0].captureWidth = .2
    with pytest.raises(ValueError, match="gaps"):
        prepare_resonances(source, prepared.context).evaluate(225.)


@pytest.mark.parametrize("mutation, pattern", [
    (lambda m: setattr(m, "unresolved", UnresolvedRegion(1e7, 2e7)), "URR"),
    (lambda m: setattr(m, "scatteringRadius", ScatteringRadius(energies=[1., 2.], values=[5., 6.])), "interpolation"),
    (lambda m: setattr(m.resolved[0], "domainUnit", "keV"), "units"),
    (lambda m: setattr(m.resolved[0], "formalism", None), "only BreitWigner"),
])
def test_unsupported_model_rejected(mutation, pattern):
    source = model()
    mutation(source)
    with pytest.raises(UnsupportedResonanceError, match=pattern):
        prepare_resonances(source, NeutronContext(56., 0.))


@pytest.mark.parametrize("field, value, pattern", [
    ("totalWidth", .5, "GT"), ("energy", 0., "zero-energy"),
    ("neutronWidth", -.1, "negative"), ("spin", 1., "incompatible"),
    ("captureWidth", float("nan"), "nonfinite"),
])
def test_invalid_resonance_rejected(field, value, pattern):
    source = model()
    setattr(source.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0], field, value)
    with pytest.raises(ValueError, match=pattern):
        prepare_resonances(source, NeutronContext(56., 0.))


def test_hidden_competitive_metadata_rejected_even_with_equal_gt():
    source = model()
    source.provenance = SimpleNamespace(headerFields={"regions": [{"l_blocks": [{"lrx": 1, "qx": -1.}]}]})
    with pytest.raises(UnsupportedResonanceError, match="competitive"):
        prepare_resonances(source, NeutronContext(56., 0.))


def test_fission_and_absent_j_sector_against_collision_amplitude():
    ctx = NeutronContext(56., .5)
    source = model(levels=[Resonance(100., 0., .6, .1, .2, .3)])
    energy = np.array([99.7, 100., 100.3])
    result = prepare_resonances(source, ctx).evaluate(energy)
    k2 = ctx.k_squared_per_ev*energy
    gn = .1*np.sqrt(energy/100.)
    width = gn+.5
    phi = np.sqrt(k2)*5.
    u = np.exp(-2j*phi)*(1-1j*gn/(energy-100.+.5j*width))
    expected = np.pi*.01/k2*(.25*abs(1-u)**2+.75*4*np.sin(phi)**2)
    np.testing.assert_allclose(result[2], expected, rtol=2e-13)
    np.testing.assert_allclose(result[18], 1.5*result[102], rtol=2e-15)


def test_rounding_of_total_width_is_reported():
    source = model()
    source.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0].totalWidth = .3000001
    assert "within rounding tolerance" in prepare_resonances(source, NeutronContext(56, 0)).preparation_notes[0]


def test_lossy_input_conversion_is_rejected():
    from kika.nuclear_data.model.conversion import ConversionReport
    report = ConversionReport(losses=["radius unit was not stated"])
    with pytest.raises(UnsupportedResonanceError, match="conversion"):
        prepare_resonances(model(), NeutronContext(56, 0), conversion_report=report)


@pytest.mark.parametrize("header", [
    {"isotopes": [{}, {}]},
    {"regions": [{"kind": "unsupported"}]},
    {"regions": [{"nro": 1, "naps": 2}]},
])
def test_incomplete_source_rejected(header):
    source = model()
    source.provenance = SimpleNamespace(headerFields=header)
    with pytest.raises(UnsupportedResonanceError):
        prepare_resonances(source, NeutronContext(56, 0))


@pytest.mark.parametrize("energy", [0., -1., float("nan"), float("inf"), [[100.]], 1e8])
def test_invalid_evaluation_energy(energy):
    with pytest.raises(ValueError):
        prepare_resonances(model(), NeutronContext(56., 0.)).evaluate(energy)


@pytest.mark.parametrize("awr, spin", [(0., 0.), (float("inf"), 0.), (56., -.5), (56., .3)])
def test_invalid_context(awr, spin):
    with pytest.raises(ValueError):
        NeutronContext(awr, spin)

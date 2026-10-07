"""Independent checks for radius policies, competitive widths and channel limits."""
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn
from kika.nuclear_data.model.resonances import CompetitiveChannel, RadiusPolicy, ScatteringRadius
from kika.processing.resonances import NeutronContext, prepare_resonances, UnsupportedResonanceError
from kika.processing.resonances.radii import prepare_radius
from kika.processing.resonances.channel_functions import neutral_channel_functions
from test_resonance_foundations import model


@pytest.mark.parametrize('law', [1, 2, 3, 4, 5])
def test_radius_interpolation(law):
    source = ScatteringRadius(energies=[1., 9.], values=[2., 8.], interpolation=[(2, law)], unit='fm')
    radius = prepare_radius(source)
    x = 3.
    fraction = np.log(x)/np.log(9.) if law in (3, 5) else (x-1)/8
    expected = 2. if law == 1 else 2.+6*fraction if law in (2,3) else 2.*4**fraction
    assert radius.evaluate(x) == pytest.approx(expected)
    assert radius.evaluate(9.) == 8.
    with pytest.raises(ValueError, match='cover'):
        radius.evaluate(.5)
    source.values[0] = 99.
    assert radius.evaluate(1.) == 2.


def test_mixed_laws_shared_endpoint():
    r = prepare_radius(ScatteringRadius(energies=[1., 2., 4., 8.], values=[2., 3., 6., 12.],
        interpolation=[(2,1),(4,5)], unit='fm'))
    np.testing.assert_allclose(r.evaluate([1.5,2.,3.,6.]), [2.,3.,4.5,9.])


@pytest.mark.parametrize('mode', ['mass','phase','constant'])
def test_independent_radius_policies(mode):
    m = model()
    bw = m.resolved[0].formalism
    table = ScatteringRadius(energies=[1e-8,1e7], values=[3.,7.], interpolation=[(2,2)], unit='fm')
    bw.radiusPolicy = RadiusPolicy(channelMode=mode, channelRadius=4., phaseRadius=table)
    ctx = NeutronContext(56.,0.)
    p = prepare_resonances(m,ctx)
    g = p.regions[0].groups[0]
    e = np.array([25.,100.,400.])
    expected = ctx.mass_channel_radius_fm if mode == 'mass' else 4. if mode == 'constant' else prepare_radius(table).evaluate(e)
    np.testing.assert_allclose(g.channel_radius.evaluate(e), expected)
    np.testing.assert_allclose(g.phase_radius.evaluate(e), prepare_radius(table).evaluate(e))


@pytest.mark.parametrize('approximation', ['SingleLevel','MultiLevel'])
def test_competition_against_complex_swave(approximation):
    m = model(approximation)
    g = m.resolved[0].formalism.resonanceParameters.spinGroups[0]
    g.resonances[0].totalWidth = .7
    g.competitiveChannel = CompetitiveChannel(Q=-20., L=0, channelRadius=4.)
    ctx = NeutronContext(56.,0.)
    threshold = 20.*57/56
    e = np.array([threshold/2,threshold,25.,100.,400.])
    result = prepare_resonances(m,ctx).evaluate(e)
    gn = .1*np.sqrt(e/100.)
    gx = .4*np.sqrt(np.maximum(e-threshold,0)/(100.-threshold))
    width = gn+.2+gx
    denominator = (e-100.)**2+(width/2)**2
    beta = np.pi*.01/(ctx.k_squared_per_ev*e)
    np.testing.assert_allclose(result[51], beta*gn*gx/denominator, rtol=2e-14)
    np.testing.assert_allclose(result[102], beta*gn*.2/denominator, rtol=2e-14)
    u = np.exp(-2j*np.sqrt(ctx.k_squared_per_ev*e)*5.)*(1-1j*gn/(e-100.+.5j*width))
    np.testing.assert_allclose(result[2], beta*abs(1-u)**2, rtol=2e-12)
    np.testing.assert_allclose(result[1], result[2]+result[102]+result[51])
    assert result[51][1] == 0.


def test_missing_exit_l_rejects_and_signed_bw_is_invariant():
    m = model()
    g = m.resolved[0].formalism.resonanceParameters.spinGroups[0]
    ctx = NeutronContext(56.,0.)
    baseline = prepare_resonances(m,ctx).evaluate([99.,100.,101.])
    g.resonances[0].spin *= -1
    signed = prepare_resonances(m,ctx).evaluate([99.,100.,101.])
    for mt in baseline:
        np.testing.assert_array_equal(baseline[mt], signed[mt])
    g.competitiveChannel = CompetitiveChannel(Q=-20.)
    with pytest.raises(UnsupportedResonanceError, match='exit L'):
        prepare_resonances(m,ctx)


@pytest.mark.parametrize('l', [4,8,16,32,64])
def test_high_l_against_bessel_wronskian(l):
    x = np.geomspace(max(1.,l/2), 100., 30)
    p,s,phi = neutral_channel_functions(l,x)
    f,g = x*spherical_jn(l,x), -x*spherical_yn(l,x)
    fp = spherical_jn(l,x)+x*spherical_jn(l,x,derivative=True)
    gp = -spherical_yn(l,x)-x*spherical_yn(l,x,derivative=True)
    np.testing.assert_allclose(p,x/(f*f+g*g),rtol=3e-13)
    np.testing.assert_allclose(s,x*(f*fp+g*gp)/(f*f+g*g),rtol=3e-12,atol=1e-12)
    np.testing.assert_allclose(np.sin(phi)**2,f*f/(f*f+g*g),rtol=3e-13,atol=1e-27)


def test_declared_awri_changes_context_without_mutating_caller():
    m = model()
    m.resolved[0].formalism.resonanceParameters.spinGroups[0].atomicWeightRatio = 12.
    ctx = NeutronContext(56.,0.)
    p = prepare_resonances(m,ctx)
    assert p.regions[0].groups[0].context.atomic_weight_ratio == 12.
    assert ctx.atomic_weight_ratio == 56.
    assert p.preparation_notes


def test_shifted_pwave_against_independent_complex_collision_matrix():
    m = model('SingleLevel',l=1)
    resonance = m.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0]
    resonance.energy,resonance.neutronWidth,resonance.captureWidth,resonance.totalWidth = 1e6,1e5,2e5,3e5
    ctx = NeutronContext(56.,0.)
    e = np.array([.5e6,.95e6,1e6,1.05e6,2e6])
    x = np.sqrt(ctx.k_squared_per_ev*e)*5.
    xr = np.sqrt(ctx.k_squared_per_ev*resonance.energy)*5.
    def ps(x):
        f,g=x*spherical_jn(1,x),-x*spherical_yn(1,x)
        fp=spherical_jn(1,x)+x*spherical_jn(1,x,derivative=True)
        gp=-spherical_yn(1,x)-x*spherical_yn(1,x,derivative=True)
        return x/(f*f+g*g),x*(f*fp+g*gp)/(f*f+g*g),np.arctan2(f,g)
    p,s,phase = ps(x)
    pr,sr,_ = ps(xr)
    gn = 1e5*p/pr
    shifted = 1e6+.5e5*(sr-s)/pr
    width = gn+2e5
    u = np.exp(-2j*phase)*(1-1j*gn/(e-shifted+.5j*width))
    beta = np.pi*.01/(ctx.k_squared_per_ev*e)
    # J=3/2 has g=2; the missing J=1/2 contributes g=1 potential.
    expected = 2*beta*abs(1-u)**2+4*beta*np.sin(phase)**2
    np.testing.assert_allclose(prepare_resonances(m,ctx).evaluate(e)[2],expected,rtol=2e-12)


def test_evaluation_underflow_is_reported_instead_of_losing_resonances():
    m=model(l=2)
    m.resolved[0].domainMin=1e-300
    with pytest.raises(FloatingPointError,match='penetrability underflows'):
        prepare_resonances(m,NeutronContext(56.,0.)).evaluate(1e-250)


def test_local_tables_on_adjacent_regions_have_independent_ownership():
    from copy import deepcopy
    m=model()
    first=m.resolved[0]
    first.domainMin,first.domainMax=1.,200.
    second=deepcopy(first)
    second.domainMin,second.domainMax=200.,1000.
    first.scatteringRadius=ScatteringRadius(energies=[1.,200.],values=[3.,4.],interpolation=[(2,2)],unit='fm')
    # Reference |Er| must be covered even when it lies outside this region.
    second.scatteringRadius=ScatteringRadius(energies=[1.,1000.],values=[6.,7.],interpolation=[(2,2)],unit='fm')
    m.resolved.append(second)
    ctx=NeutronContext(56.,0.)
    full=prepare_resonances(m,ctx).evaluate([199.,200.,201.])
    right=deepcopy(m)
    right.resolved=[second]
    for mt,v in prepare_resonances(right,ctx).evaluate([200.,201.]).items():
        np.testing.assert_array_equal(v,full[mt][1:])

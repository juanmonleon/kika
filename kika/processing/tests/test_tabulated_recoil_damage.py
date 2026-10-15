"""Numerical damage controls: closed moments, held support and thresholds."""
import numpy as np
import pytest
from scipy.integrate import quad
from kika.nuclear_data import model as m
from kika.processing import heating as H


def spectrum(flat=False):
    return m.XYs2d(function1ds=[
        m.Xs_pdf_cdf1d(xs=[0.,1e6],pdf=[1e-6,1e-6],cdf=[0.,1.],outerDomainValue=1e6),
        m.Xs_pdf_cdf1d(xs=[0.,4e6],pdf=[.5e-6,0.],cdf=[0.,1.],outerDomainValue=2e6)],
        interpolation=m.Interpolation.flat if flat else m.Interpolation.linlin,
        interpolationQualifier=m.InterpolationQualifier.unitBase)


@pytest.mark.parametrize("flat",[False,True])
def test_tabulated_recoil_retains_incident_energy_with_held_support(monkeypatch,flat):
    monkeypatch.setattr(H,"lindhard",lambda e,*args:e)
    E=np.array([0.,1.,1e6,1.25e6,1.5e6,np.nextafter(2e6,-np.inf),2e6,3e6,1e8])
    f=np.where(E>=2e6,1.,0.) if flat else np.clip((E-1e6)/1e6,0.,1.)
    mean=(1e6+3e6*f)*((1-f)/2+f/3)
    actual=H._emissionDamage(m.Uncorrelated(energy=spectrum(flat)),E,238.,92,0.)
    np.testing.assert_allclose(actual,(E+mean)/238.,atol=.1,rtol=1.1e-4)


def test_tabulated_recoil_splits_displacement_threshold_against_scalar_quad():
    A,Z,ed=55.,26,40.;law=m.XYs2d(function1ds=[m.Xs_pdf_cdf1d(
        xs=[0.,1e5],pdf=[1e-5,1e-5],cdf=[0.,1.],outerDomainValue=1e5)])
    E=3e4
    # Independent scalar integration, including the threshold in angle.
    def partition(r):
        scale=30.724*Z*Z*(2*Z**(2/3))**.5*2
        factor=.0793*Z**(2/3)*Z**.5*(2*A)**1.5/((2*Z**(2/3))**.75*A**2)
        z=r/scale
        return r/(1+factor*(3.4008*z**(1/6)+.40244*z**.75+z))
    def integrand(ep):
        root=(E*ep)**.5;last=min(1.,(E+ep-A*ed)/(2*root)) if root else 1.
        if last<=-1:return 0.
        return quad(lambda u:partition((E+ep-2*root*u)/A),-1,last,epsabs=1e-7)[0]/2/1e5
    cuts=sorted({0.,1e5,*[v for v in (((E**.5-(A*ed)**.5)**2),((E**.5+(A*ed)**.5)**2)) if 0<v<1e5]})
    expected=sum(quad(integrand,a,b,epsabs=1e-6)[0] for a,b in zip(cuts[:-1],cuts[1:]))
    actual=H._emissionDamage(m.Uncorrelated(energy=law),np.array([E]),A,Z,ed)[0]
    assert actual==pytest.approx(expected,abs=.02,rel=2e-5)


def test_tabulated_recoil_does_not_mutate_source_and_accepts_empty_query():
    law=spectrum();before=[(np.array(t.xs),np.array(t.pdf)) for t in law.function1ds]
    assert H._emissionDamage(m.Uncorrelated(energy=law),np.array([]),238.,92,25.).size==0
    H._emissionDamage(m.Uncorrelated(energy=law),np.array([1e6]),238.,92,25.)
    for t,(x,p) in zip(law.function1ds,before):
        np.testing.assert_array_equal(t.xs,x);np.testing.assert_array_equal(t.pdf,p)


def test_tabulated_recoil_refuses_zero_mass():
    law=m.XYs2d(function1ds=[m.Xs_pdf_cdf1d(xs=[0.,1.],pdf=[0.,0.],cdf=[0.,0.],outerDomainValue=1.)])
    with pytest.raises(ValueError,match="positive width and mass"):
        H._emissionDamage(m.Uncorrelated(energy=law),np.array([1.]),238.,92,25.)

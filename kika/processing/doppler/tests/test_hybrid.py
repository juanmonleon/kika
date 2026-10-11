"""Independent accuracy and conditioning gates of the native hybrid kernel."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from kika.processing.doppler import core

@pytest.mark.parametrize('s,d,z,poly',[(-3.,6.,10.,(1.,.2,.3,.4,.5)),
    (-8.,16.,10.,(1.,.2,.1)),(0.,5.,10.,(1.,)),(-1.,4.,10.,(1.,.2))])
def test_conditioned_closed_moments_against_independent_quadrature(s,d,z,poly):
    result=core._native.closed_moments(s,d,z,poly)
    assert result is not None
    value,error,norm=result
    def integrand(u):
        t=s+d*u
        pair=math.exp(-t*t)*(-math.expm1(-4*z*(z+t)))
        return pair*sum(p*u**j for j,p in enumerate(poly))
    reference,qerror=quad(integrand,0.,1.,epsabs=1e-14,epsrel=1e-13)
    assert abs(value-reference)<=error+qerror+16*np.finfo(float).eps*abs(reference)
    assert norm>0 and error<=3e-13*abs(value)

@pytest.mark.parametrize('s,d,z,poly',[(0.,1e-10,10.,(1.,)),
    (-.001,2.,.01,(1.,)),(2.,.05,10.,(1.,1.,1.,1.,1.))])
def test_ill_conditioned_closed_moments_require_series(s,d,z,poly):
    assert core._native.closed_moments(s,d,z,poly) is None

def test_whole_panel_path_saves_subpanels_without_changing_support():
    panels=((0.,400.,2.,2.,'linlin','source',1.),)
    hybrid=core._native.integrate(panels,0,1,100.,1.,-8.,8.,80,10000)
    series=core._native.integrate(panels,0,1,100.,1.,-8.,8.,80,10000,False)
    assert hybrid[3]<series[3]
    assert abs(hybrid[0]-series[0])<=hybrid[1]+series[1]
    # The resource preflight reserves the potential fallback work.
    with pytest.raises(OverflowError):
        core._native.integrate(panels,0,1,100.,1.,-8.,8.,80,1)

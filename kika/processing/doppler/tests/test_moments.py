import math
import numpy as np
import pytest
from scipy.integrate import quad
from kika.processing.doppler.core import _moments, _MOMENT_WEIGHTS, EPS

@pytest.mark.parametrize('terms',[79,80,81])
@pytest.mark.parametrize('s,d,z',[(-.0001,.2,.001),(.4,.25,.124),(-.3,.2,.125),(-4.,.25,10.),(.001,1e-9,1e6)])
def test_weighted_moments_against_independent_quadrature(s,d,z,terms):
    values,remainder,norm=_moments(s,d,z,terms)
    for j,value in enumerate(values):
        # Stable Gaussian difference; does not use coefficient recurrence.
        def integrand(u):
            t=s+d*u
            return u**j*math.exp(-t*t)*(-math.expm1(-4*z*(z+t)))
        reference,eta=quad(integrand,0.,1.,epsabs=1e-30,epsrel=1e-12)
        assert abs(value-reference)+eta <= remainder+256*EPS*norm

def test_default_moment_weights_cannot_be_modified():
    assert _MOMENT_WEIGHTS.shape==(5,80)
    assert _MOMENT_WEIGHTS.nbytes==3200
    with pytest.raises(ValueError):_MOMENT_WEIGHTS[0,0]=2.

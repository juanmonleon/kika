"""Independent direct gamma and Laplace URR references, 2026-10-08."""
import numpy as np
import pytest
from kika.processing.resonances.fluctuations import width_products

REFERENCES = [([1, 1], [1, 0], 0, 1, 0.34432045758120156), ([1, 1], [1, 0], 0, 0, 0.6556795424187984), ([1, 1], [1.5, 0], 0, 1, 0.3810244194552785), ([1, 1], [1.5, 0], 0, 0, 0.6189755805447215), ([1, 0], [1, 0], 0, 0, 1.0), ([1, 0], [2, 0], 0, 0, 1.0), ([0.001, 0.03], [1, 0], 0, 1, 0.000913641999931093), ([0.001, 0.03], [1, 0], 0, 0, 8.635800006890704e-05), ([0.002, 0.04, 0.1], [1, 0, 2], 0, 1, 0.0007778463019062108), ([0.002, 0.04, 0.1], [1, 0, 2], 0, 2, 0.0011107595683047383), ([0.002, 0.04, 0.1], [1, 0, 2], 0, 0, 0.0001113941297890509), ([0.0005, 0.035, 0.01], [2, 0, 1], 0, 1, 0.00040418214541379234), ([0.0005, 0.035, 0.01], [2, 0, 1], 0, 2, 8.44053821691112e-05), ([0.0005, 0.035, 0.01], [2, 0, 1], 0, 0, 1.1412472417096463e-05), ([0.5, 0.2, 0.3], [1, 0, 1.0123], 0, 1, 0.07661879982186214), ([0.5, 0.2, 0.3], [1, 0, 1.0123], 0, 2, 0.07928597348415459), ([0.5, 0.2, 0.3], [1, 0, 1.0123], 0, 0, 0.3440952266939833), ([0.5, 0.2, 0.3], [2, 0, 1.9478], 0, 1, 0.0866791908261151), ([0.5, 0.2, 0.3], [2, 0, 1.9478], 0, 2, 0.10269560665242984), ([0.5, 0.2, 0.3], [2, 0, 1.9478], 0, 0, 0.31062520252145503), ([1, 1], [1, 20], 0, 1, 0.3374942163548884), ([1, 1], [1, 20], 0, 0, 0.6625057836451116), ([10, 1e-06], [1, 0], 0, 1, 9.99603767250426e-07), ([10, 1e-06], [1, 0], 0, 0, 9.999999000396233), ([1e-08, 0.05], [1, 0], 0, 1, 9.999994000006e-09), ([1e-08, 0.05], [1, 0], 0, 0, 5.9999940000084e-15), ([1, 2], [0, 0], 0, 1, 0.6666666666666666), ([1, 2], [0, 0], 0, 0, 0.3333333333333333)]

@pytest.mark.parametrize('means,degrees,entrance,exit,expected',REFERENCES)
@pytest.mark.parametrize('accelerated',[False,True])
def test_independent_width_products(means,degrees,entrance,exit,expected,accelerated):
    got=width_products(means,degrees,entrance=entrance,accelerated=accelerated)
    assert got[exit] == pytest.approx(expected,rel=2e-12,abs=0.)
    assert sum(got) == pytest.approx(means[entrance],rel=2e-12)


def test_native_quadrature_callback_matches_python_and_unavailable_binary(monkeypatch):
    from kika.processing.resonances import _rm_acceleration
    if _rm_acceleration._native is None or not hasattr(_rm_acceleration._native,'URR_INTEGRAND'):
        pytest.skip('optional quadrature callback unavailable')
    random=np.random.default_rng(391)
    for _ in range(20):
        means=10**random.uniform(-8,2,4);degrees=random.uniform(.2,4,4);degrees[1]=0.
        expected=width_products(means,degrees,accelerated=False)
        actual=width_products(means,degrees)
        np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=0.)
    monkeypatch.setattr(_rm_acceleration,'_native',None)
    np.testing.assert_array_equal(width_products(means,degrees),expected)

@pytest.mark.parametrize('scale',[1e-240,1e-100,1.,1e100,1e240])
def test_scaling_and_permutation(scale):
    means=np.array([1.,.3,.7,.2]);degrees=np.array([1.5,0.,2.4,.75])
    expected=width_products(means,degrees)
    np.testing.assert_allclose(width_products(means*scale,degrees)/scale,expected,rtol=3e-12)
    order=np.array([2,0,3,1])
    np.testing.assert_allclose(width_products(means[order],degrees[order],entrance=1),expected[order],rtol=3e-12)

@pytest.mark.parametrize('degrees',[[0.,0.],[1.,0.],[.5,2.]])
def test_zero_width_and_pure_elastic(degrees):
    np.testing.assert_array_equal(width_products([0.,1.],degrees),[0.,0.])
    np.testing.assert_array_equal(width_products([3.,0.],degrees),[3.,0.])

def test_fixed_width_limit():
    np.testing.assert_allclose(width_products([2.,3.,4.],[0.,0.,0.]),2*np.array([2.,3.,4.])/9,rtol=1e-15)


def test_large_real_nu_converges_to_fixed_widths():
    means=np.array([2.,3.,4.])
    np.testing.assert_allclose(width_products(means,[1e12,1e12,1e12]),2*means/9,rtol=3e-11)

@pytest.mark.parametrize('means,degrees',[([-1.,1.],[1.,1.]),([1.,1.],[1.,-1.]),([np.inf,1.],[1.,1.]),([1.,1.],[1.,np.nan]),([1.,1.],[1.])])
def test_invalid_distribution(means,degrees):
    with pytest.raises(ValueError):width_products(means,degrees)

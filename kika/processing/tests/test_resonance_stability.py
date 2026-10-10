"""Independent high-precision checks of small neutral-channel shifts."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from kika.processing.resonances.channel_functions import neutral_channel_functions, neutral_shift_difference
from kika.processing.resonances.radii import RadiusFunction


def decimal_shift(l, z):
    p, s = z.sqrt(), Decimal(0)
    for order in range(1, l+1):
        d = Decimal(order)-s
        denominator = d*d+p*p
        p, s = z*p/denominator, z*d/denominator-order
    return s


@pytest.mark.parametrize('l',[0,1,2,3,8,16,64])
def test_shift_difference_against_100_digit_recurrence(l):
    zr=1e-6
    z=np.nextafter(zr,0.)
    with localcontext() as context:
        context.prec=100
        expected=float(decimal_shift(l,Decimal.from_float(zr))-decimal_shift(l,Decimal.from_float(z)))
    actual=float(neutral_shift_difference(l,zr,z,zr-z))
    assert actual==pytest.approx(expected,rel=2e-14,abs=0.)
    if l==3:
        assert expected != 0
        assert neutral_channel_functions(l,np.sqrt(zr))[1]-neutral_channel_functions(l,np.sqrt(z))[1]==0


@pytest.mark.parametrize('law',[2,3,4,5])
def test_radius_changes_below_absolute_ulp_are_retained(law):
    radius=RadiusFunction(energies=(1.,1000.),values=(5.,7.),interpolation=((2,law),))
    reference=100.;energy=np.nextafter(reference,np.inf)
    def evaluate(x):
        a,b,ra,rb=Decimal(1),Decimal(1000),Decimal(5),Decimal(7)
        fraction=(x/a).ln()/(b/a).ln() if law in (3,5) else (x-a)/(b-a)
        return ra+fraction*(rb-ra) if law in (2,3) else (ra.ln()+fraction*(rb/ra).ln()).exp()
    with localcontext() as context:
        context.prec=100
        expected=float(evaluate(Decimal.from_float(reference))-evaluate(Decimal.from_float(energy)))
    actual=float(radius.difference(reference,np.array([energy]))[0])
    assert actual==pytest.approx(expected,rel=2e-14,abs=0.)
    assert float(radius.difference(reference,energy))==actual


def test_tabulated_radius_jump_is_not_smoothed_by_difference():
    radius=RadiusFunction(energies=(1.,100.,1000.),values=(5.,7.,9.),interpolation=((3,1),))
    x=np.array([np.nextafter(100.,1.),100.,np.nextafter(100.,1000.),1000.])
    np.testing.assert_array_equal(radius.difference(100.,x),[2.,0.,0.,-2.])


@pytest.mark.parametrize('l',[1,2])
def test_low_orbital_phase_against_independent_bessel_functions(l):
    from scipy.special import spherical_jn, spherical_yn
    x=np.unique(np.r_[np.geomspace(1e-30,1e4,12000),1.,np.nextafter(1.,0.),np.nextafter(1.,2.)])
    actual=neutral_channel_functions(l,x)[2]
    expected=np.arctan2(spherical_jn(l,x),-spherical_yn(l,x))
    np.testing.assert_allclose(np.exp(1j*actual),np.exp(1j*expected),rtol=0.,atol=8e-15)
    np.testing.assert_allclose(actual[x<1],expected[x<1],rtol=8e-14,atol=0.)
    assert neutral_channel_functions(l,.3)[2]==pytest.approx(float(neutral_channel_functions(l,np.array([.3]))[2][0]),rel=2e-15)
    p,s,phase=neutral_channel_functions(l,x,phase=False)
    np.testing.assert_array_equal(p,neutral_channel_functions(l,x)[0])
    assert phase is None


@pytest.mark.parametrize('l',[1,2])
@pytest.mark.parametrize('z',[1e-20,1e-6,.1,1.,10.,1e12])
def test_rational_shift_difference_at_adjacent_energies(l,z):
    zr=np.nextafter(z,np.inf)
    with localcontext() as context:
        context.prec=100
        expected=float(decimal_shift(l,Decimal.from_float(zr))-decimal_shift(l,Decimal.from_float(z)))
    assert float(neutral_shift_difference(l,zr,z,zr-z))==pytest.approx(expected,rel=3e-14,abs=0.)

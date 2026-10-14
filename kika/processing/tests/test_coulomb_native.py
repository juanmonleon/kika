"""Charged-channel functions in the barrier and the closed shift: 40-digit
references and the compiled kernel against the Python reference."""
import numpy as np
import pytest
from kika.processing.resonances import coulomb
from kika.processing.resonances.coulomb import charged_channel_log_functions,closed_charged_shift

# mpmath 1.4.1 at 40 digits: P=rho/(F^2+G^2), S=rho*(FF'+GG')/(F^2+G^2) from
# coulombf/coulombg and their derivatives; phase=atan2(P, rho*F'/F-S).
BARRIER = [
    (0,8.9,0.95,-38.319761251236693325,-3.7568456368451143719,2.8331430430254991398e-18),
    (3,15.0,1.0,-74.835887162495949076,-6.0930702507355980172,2.4568095672743630047e-34),
    (2,4.24,2.0,-11.889161054953055688,-3.9912476957084659667,7.8425392449898082876e-7),
    (0,9.0,16.0,0.62226370125199646454,-3.9406200134310443386,0.16586517509932435665),
    (0,34.0,0.25,-195.81722057685575694,-3.8849288470157749785,1.0958890582984578491e-86),
    (2,85.0,0.1,-518.93905240598273401,-4.5045860967163561469,4.3987718755349711386e-227),
]

# mpmath 1.4.1 at 40 digits: x*W'/W of whitw(-eta, l+1/2, 2x). The last
# column says whether the continued fraction converges within its limit
# (the last point needs about 750 iterations and goes to the integral).
CLOSED = [
    (0,2.27,3.7,-5.3981098627559765455,True),
    (5,14.75,0.57,-6.4799603790475451385,True),
    (2,8.0,1.5,-5.4209928924842423456,True),
    (3,100.0,0.05,-4.3345231749747681544,False),
]


# mpmath 1.3.0 at 40 digits, as BARRIER: points of Ca-40's proton and alpha
# exits (ENDF/B-VIII.1) just above threshold, where the outgoing fraction
# does not converge within its limit and the Laplace integral is used.
LAPLACE = [
    (1,134.86439767244536,0.02380852775070165,-837.86565174994330863,-2.6310109203188288278,2.2340917591681134966e-365),
    (1,213.02295646329114,0.015073130182189524,-1328.9503007058227012,-2.6310824665238452326,1.1850566187753833841e-578),
    (2,212.87156890341933,0.01508384971796251,-1330.7130463400663181,-3.1921340842864301728,1.6804712947618665043e-579),
]


@pytest.mark.parametrize('l,eta,rho,log_p,shift,phase',LAPLACE)
def test_threshold_laplace_against_40_digit_reference(l,eta,rho,log_p,shift,phase):
    with pytest.raises(FloatingPointError):
        coulomb._outgoing(l,eta,rho,coulomb.SMALL_RADIUS_FRACTION_ITERATIONS)
    for actual in (charged_channel_log_functions(l,eta,rho),coulomb._charged_point(l,eta,rho)):
        assert float(actual[0])==pytest.approx(log_p,rel=2e-15)
        assert float(actual[1])==pytest.approx(shift,rel=1e-14)
        assert float(actual[2])==0.  # below the float range: rounds to zero
    if coulomb._native is not None:
        assert coulomb._native_open_rows(l,np.array([eta]),np.array([rho]))[3][0]


@pytest.mark.parametrize('l,eta,rho,log_p,shift,phase',BARRIER)
def test_barrier_functions_against_40_digit_reference(l,eta,rho,log_p,shift,phase):
    for actual in (charged_channel_log_functions(l,eta,rho),coulomb._charged_point(l,eta,rho)):
        assert float(actual[0])==pytest.approx(log_p,rel=2e-14,abs=1e-13)
        assert float(actual[1])==pytest.approx(shift,rel=1e-12)
        assert float(actual[2])==pytest.approx(phase,rel=1e-12)


@pytest.mark.parametrize('l,eta,kappa,shift,fraction',CLOSED)
def test_closed_shift_against_40_digit_reference(monkeypatch,l,eta,kappa,shift,fraction):
    assert float(closed_charged_shift(l,eta,kappa))==pytest.approx(shift,rel=1e-12)
    value = coulomb._closed_fraction(l,eta,kappa)
    if fraction:assert value==pytest.approx(shift,rel=1e-12)
    else:assert value is None
    # The Python reference, where the integral is adaptive quadrature.
    monkeypatch.setattr(coulomb,'_native',None)
    assert float(closed_charged_shift(l,eta,kappa))==pytest.approx(shift,rel=1e-10)


def test_the_barrier_route_needs_no_transport_where_it_is_conditioned(monkeypatch):
    def forbidden(*args):
        raise AssertionError('transport called')
    monkeypatch.setattr(coulomb,'_transport',forbidden)
    for l,eta,rho,*_ in BARRIER:
        coulomb._charged_point(l,eta,rho)


@pytest.mark.skipif(coulomb._native is None,reason='compiled Coulomb kernel not built')
@pytest.mark.parametrize('l',[0,1,2,5,8,20])
def test_native_open_rows_are_the_reference(l):
    rng = np.random.default_rng(l)
    eta = np.concatenate([10**rng.uniform(-3,2.5,300),[1.,3.,8.9,20.]])
    rho = np.concatenate([10**rng.uniform(np.log10(.26),2.3,300),[1.,3.,.95,100.]])
    # and points below rho=0.25 deep in the barrier
    eta = np.concatenate([eta,8.5/np.array([.25,.1,.05,.02])])
    rho = np.concatenate([rho,[.25,.1,.05,.02]])
    actual = charged_channel_log_functions(l,eta,rho)
    reference = np.array([coulomb._charged_point(l,h,r) for h,r in zip(eta,rho)]).T
    np.testing.assert_allclose(actual[0],reference[0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual[1],reference[1],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(np.sin(actual[2]-reference[2]),0.,atol=1e-12)


@pytest.mark.skipif(coulomb._native is None,reason='compiled Coulomb kernel not built')
@pytest.mark.parametrize('l',[0,1,2,5,8,20,64])
def test_native_closed_rows_are_the_reference(l):
    rng = np.random.default_rng(100+l)
    eta = 10**rng.uniform(-2,4,200);kappa = 10**rng.uniform(-3,2,200)
    actual = closed_charged_shift(l,eta,kappa)
    reference = []
    for h,r in zip(eta,kappa):
        value = coulomb._closed_fraction(l,h,r)
        reference.append(coulomb._closed_point(l,h,r) if value is None else value)
    # The quadrature reference is itself accurate to about 1e-10 (40-digit
    # comparison); the fraction to 1e-12.
    np.testing.assert_allclose(actual,reference,rtol=2e-10)


# mpmath 1.3.0 at 40 digits, as BARRIER, for L=0..3 at one (eta, rho): points
# of Ca-40's alpha exit (ENDF/B-VIII.1) and one deeper in the barrier.
LADDER = [
    (0,8.0,3.0,-21.669141874460773723,-5.9475353563141141224,3.103517806286231317e-11),
    (1,8.0,3.0,-22.162116752997418284,-6.0970407412219158193,1.8486869525644235341e-11),
    (2,8.0,3.0,-23.132823986447423525,-6.3869388548247785383,6.6841286097103312141e-12),
    (3,8.0,3.0,-24.553803112942044637,-6.8014460964333058988,1.5161695443494984517e-12),
    (0,6.0,3.6,-11.150417429600682659,-5.1461557080003267716,1.3049233043242580287e-6),
    (1,6.0,3.6,-11.623826421134241088,-5.3176058173130416857,7.8703939193096815054e-7),
    (2,6.0,3.6,-12.553141962411571323,-5.6467369636597551863,2.9302964223764961203e-7),
    (3,6.0,3.6,-13.907763577710485984,-6.1108300876957689782,7.0032208117494628328e-8),
    (0,9.448,2.3155,-31.688379890284707914,-5.9234072882156712068,1.3926007623900238406e-15),
    (1,9.448,2.3155,-32.227150312035695432,-6.0728988258008008496,7.9209774204819391237e-16),
    (2,9.448,2.3155,-33.288820918869686915,-6.3628338414734995015,2.6131821740461415635e-16),
    (3,9.448,2.3155,-34.844564077645658507,-6.7775108451033885146,5.176172257388250859e-17),
    (0,30.0,0.5,-164.98636553262543226,-5.2179451167906813219,2.0332199850374760152e-73),
    (1,30.0,0.5,-165.67884921893559548,-5.3839861136340722304,9.8457486974619758277e-74),
    (2,30.0,0.5,-167.04518033580564685,-5.7039738838344718863,2.366279810021635343e-74),
    (3,30.0,0.5,-169.05172589065821136,-6.1574687594971637702,2.9445309227932905466e-75),
]


@pytest.mark.parametrize('ls',[(0,1,2,3),(1,3),(2,3)])
def test_ladder_by_upward_recurrence_against_40_digit_reference(ls):
    """One evaluation at the lowest L and the outgoing wave's recurrence."""
    points=sorted({(eta,rho) for _,eta,rho,*_ in LADDER})
    eta=np.array([p[0] for p in points]);rho=np.array([p[1] for p in points])
    ladder=coulomb.charged_channel_log_ladder(ls,eta,rho)
    assert sorted(ladder)==sorted(ls)
    for l,e,r,log_p,shift,phase in LADDER:
        if l not in ls:continue
        i=points.index((e,r))
        assert float(ladder[l][0][i])==pytest.approx(log_p,rel=2e-14,abs=1e-13)
        assert float(ladder[l][1][i])==pytest.approx(shift,rel=1e-13)
        assert float(ladder[l][2][i])==pytest.approx(phase,rel=1e-12)
    # The lowest L is the ordinary evaluation, bit for bit.
    direct=charged_channel_log_functions(min(ls),eta,rho)
    for got,expected in zip(ladder[min(ls)],direct):np.testing.assert_array_equal(got,expected)

"""Compiled/reference agreement and independent quadrature in difficult regimes."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from kika.processing.doppler import (DopplerContext,DopplerOptions,SupportPolicy,
    prepare_curve,evaluate_broadened,broaden_curve,DopplerInputError,
    DopplerConvergenceError,UnsupportedDopplerError)
import kika.processing.doppler.core as core


@pytest.fixture
def native():
    # This gate must fail rather than quietly skip an unbuilt extension.
    assert core._native is not None, 'build the native Doppler extension'
    return core._native


@pytest.mark.parametrize('terms',[8,32,79,80,81,160])
@pytest.mark.parametrize('s,d,z',[(-1e-12,.2,1e-12),(-.0001,.2,.001),
    (-.124,.25,.124),(-.125,.25,.125),(.4,.25,.124),
    (-.3,.2,1.),(-4.,.25,10.),(.001,1e-9,1e6),(8.,.25,20.),(-25.,.25,30.)])
def test_native_moments_against_quadrature(native,s,d,z,terms):
    values,remainder,norm,used=native.moments(s,d,z,terms)
    assert 8<=used<=terms
    for j,value in enumerate(values):
        def fn(u):
            t=s+d*u
            return u**j*math.exp(-t*t)*(-math.expm1(-4*z*(z+t)))
        reference,eta=quad(fn,0.,1.,epsabs=1e-300,epsrel=2e-12)
        assert abs(value-reference)+eta<=remainder+256*core.EPS*norm
    if terms==80 and z<20:
        assert used<80


@pytest.mark.parametrize('support',[SupportPolicy('zero','zero'),SupportPolicy('constant','constant'),SupportPolicy('one_over_v','constant')])
@pytest.mark.parametrize('terms',[32,80,81])
def test_native_pointwise_preserves_values_diagnostics_and_jumps(native,support,terms):
    rng=np.random.default_rng(23)
    x=np.geomspace(1e-8,10.,70)
    y=rng.uniform(.1,4.,len(x))
    p=prepare_curve(((x,y),([10.,20.,30.],[1.,-1.,2.])),
        context=DopplerContext(1.),source_temperature=0.,support=support,negative_policy='preserve_signed')
    q=np.geomspace(1e-12,1e5,80)
    a=evaluate_broadened(p,q,temperature=600.,options=DopplerOptions(backend='python',max_series_terms=terms))
    b=evaluate_broadened(p,q,temperature=600.,options=DopplerOptions(backend='native',max_series_terms=terms))
    assert np.all(np.abs(a.values-b.values)<=a.kernel_error_estimates+b.kernel_error_estimates)
    assert np.array_equal(a.tail_bounds,b.tail_bounds)
    assert a.report['analytic_subpanels']>=b.report['analytic_subpanels']
    assert a.report['max_thermal_radius']==b.report['max_thermal_radius']
    for owner,v in a.report['components_sum_over_probes'].items():
        assert b.report['components_sum_over_probes'][owner]==pytest.approx(v,rel=1e-13,abs=1e-12)
    assert a.report['backend']=='python' and b.report['backend']=='native'


def test_native_narrow_peak_and_tabulation(native):
    lo,hi=1.-1e-9,1.+1e-9
    p=prepare_curve((([.8,lo,1.,hi,1.2],[0.,0.,2/(hi-lo),0.,0.]),),
        context=DopplerContext(56.),source_temperature=0.,support=SupportPolicy('zero','zero'))
    a=broaden_curve(p,temperature=293.6,options=DopplerOptions(backend='python'))
    b=broaden_curve(p,temperature=293.6,options=DopplerOptions(backend='native'))
    q=np.geomspace(.8,1.2,151)
    assert np.allclose(np.interp(q,a.form.xs,a.form.ys),np.interp(q,b.form.xs,b.form.ys),rtol=1e-12,atol=1e-20)
    assert b.verify_table()<=1


def test_missing_native_is_explicit_and_auto_uses_python(monkeypatch):
    monkeypatch.setattr(core,'_native',None)
    p=prepare_curve((([0.,1.],[2.,2.]),),context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('zero','constant'))
    with pytest.raises(UnsupportedDopplerError,match='build the native'):
        evaluate_broadened(p,[1.],temperature=600.,options=DopplerOptions(backend='native'))
    assert evaluate_broadened(p,[1.],temperature=600.).report['backend']=='python'
    with pytest.raises(DopplerInputError):DopplerOptions(backend='invalid')


@pytest.mark.parametrize('options',[dict(max_panels=1),dict(max_radius=1),dict(max_series_terms=8,rtol=1e-10),dict(max_work_bytes=100)])
def test_native_limits_keep_failure_categories(native,options):
    p=prepare_curve((([0.,1.],[2.,2.]),),context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('zero','constant'))
    errors=[]
    for backend in ('python','native'):
        with pytest.raises(DopplerConvergenceError) as exc:
            evaluate_broadened(p,[1.],temperature=1/core.KB,options=DopplerOptions(backend=backend,**options))
        errors.append(exc.value.category)
    assert errors[0]==errors[1]


def test_native_argument_guards(native):
    for args in [((),-1,0,1.,1.,-1.,1.,80,10),((),0,1,1.,1.,-1.,1.,80,10),
                 ((),0,0,0.,1.,-1.,1.,80,10),((),0,0,1.,1.,-1.,1.,7,10)]:
        with pytest.raises(ValueError):native.integrate(*args)
    with pytest.raises(ValueError):native.moments(0.,1.,1.,80)


def test_interval_reuse_keeps_the_mesh_and_independent_accuracy():
    from kika.processing.doppler.table import _adaptive
    def function(x):
        return 1.+3.*np.exp(-((x-1.1)/.055)**2)+2.*np.exp(-((x-2.6)/.18)**2)
    def evaluate(x):return {0:function(x)},{0:np.zeros_like(x)}
    seeds=np.array([.1,.8,1.1,1.4,2.,2.6,3.5,4.])
    limited=_adaptive(evaluate,seeds,DopplerOptions(max_work_bytes=4096),200000,(0,))
    full=_adaptive(evaluate,seeds,DopplerOptions(),200000,(0,))
    assert np.array_equal(limited[0],full[0])
    assert np.array_equal(limited[1][0],full[1][0])
    # The cache never changes the mesh and never costs evaluations.
    assert full[2]['evaluations']<=limited[2]['evaluations']
    q=np.linspace(.1,4.,6001)
    assert np.max(np.abs(np.interp(q,full[0],full[1][0])-function(q))/(1e-8+1e-3*np.abs(function(q))))<.45


def test_kernel_special_functions_meet_the_provisions(native):
    """The kernel's own exp, e^-y^2 and erfc against exact references (16 eps is the
    smallest provision they enter)."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494')

    def erfc(y):
        y = Decimal(y)
        if y < 3:
            total, term, n = Decimal(0), y, 0
            while abs(term/(2*n+1)) > Decimal('1e-55'):
                total += term/(2*n+1); n += 1; term = -term*y*y/n
            return 1-2/pi.sqrt()*total
        fraction = y
        for k in range(600, 0, -1): fraction = y+Decimal(k)/2/fraction
        return (-y*y).exp()/pi.sqrt()/fraction

    eps = np.finfo(float).eps
    for y in np.r_[np.linspace(0., 6., 241), np.linspace(6., 26.5, 83), [1e-12, 1e-6, .46875, 4.]]:
        y = float(y)
        e, c, g = native.special(y)
        for got, ref in ((e, (-Decimal(y)).exp()), (g, (-Decimal(y)**2).exp()), (c, erfc(y))):
            assert abs(Decimal(got)/ref-1) <= 6*eps

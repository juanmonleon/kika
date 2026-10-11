"""Public pointwise contracts and analytic numerical regression anchors."""
import math
import numpy as np
import pytest
from scipy.special import erf
from kika.processing.doppler import (
    DopplerContext, DopplerOptions, SupportPolicy, DopplerInputError,
    DopplerConvergenceError, prepare_curve, evaluate_broadened,
)
from kika._constants import BOLTZMANN_CONSTANT

KB = BOLTZMANN_CONSTANT*1e6


def prepared(regions=(([0., 1.], [2., 2.]),), lower="zero", upper="constant", **kw):
    return prepare_curve(regions, context=DopplerContext(1.), source_temperature=0.,
                         support=SupportPolicy(lower, upper), **kw)


@pytest.mark.parametrize("z", [1e-8, 1e-4, .124999, .125001, 1., 10., 1e6])
def test_constant_gaussian_pair_and_limits(z):
    result = evaluate_broadened(prepared(), [z*z], temperature=1/KB)
    expected = 2*((1+1/(2*z*z))*erf(z)+math.exp(-z*z)/(math.sqrt(math.pi)*z))
    assert result.values[0] == pytest.approx(expected, rel=2e-12)
    assert result.kernel_error_estimates[0] < .15*(1e-8+1e-3*abs(expected))


def test_inverse_velocity():
    p = prepared((([1e6, 2e6], [1e-3, 1/math.sqrt(2e6)]),), lower="one_over_v")
    e = np.array([1e-12, .01, 1., 100.])
    assert np.allclose(evaluate_broadened(p, e, temperature=1/KB).values, 1/np.sqrt(e), rtol=1e-12)


def test_identity_discontinuity_and_immutable():
    x = np.array([0., 1.]); y = np.array([1., 1.])
    p = prepared(((x, y), ([1., 2.], [5., 5.])), upper="zero")
    y[:] = 9
    e = [np.nextafter(1., 0.), 1., np.nextafter(1., 2.)]
    r = evaluate_broadened(p, e, temperature=0.)
    assert np.array_equal(r.values, [1, 5, 5])
    assert np.all(r.kernel_error_estimates == 0)
    with pytest.raises(ValueError): p.regions[0][1].setflags(write=True)
    with pytest.raises(ValueError): r.values.setflags(write=True)


@pytest.mark.parametrize("scale", [1e-8, 1e-12, 1e-20])
def test_tiny_increment_on_linear_panel(scale):
    p = prepared((([0., 1., 2., 3.], [1., 2., 3., 4.]),))
    v = evaluate_broadened(p, [1.5], temperature=scale/KB).values[0]
    assert v == pytest.approx(2.5, abs=5*scale+1e-12)


def test_threshold_removal_separate():
    p = prepared((([1., 2., 4.], [0., 3., 1.]),), upper="zero")
    r = evaluate_broadened(p, [.5, 1.], temperature=1/KB,
                          options=DopplerOptions(threshold_policy="zero_below", threshold=1.))
    assert r.values[0] == 0 and r.threshold_removed[0] > 0
    assert r.values[1] > 0 and r.threshold_removed[1] == 0


def test_resources_fail_without_partial_result():
    p = prepared()
    for options in (DopplerOptions(max_work_bytes=100), DopplerOptions(max_panels=1),
                    DopplerOptions(max_series_terms=8, rtol=1e-10), DopplerOptions(max_radius=1)):
        with pytest.raises(DopplerConvergenceError): evaluate_broadened(p, [1.], temperature=1/KB, options=options)


@pytest.mark.parametrize("energies", [[0.], [-1.], [math.nan], [True], [1., True], ['1'], [1+0j], [], [[1.]]])
def test_reject_bad_probe_domain(energies):
    with pytest.raises(DopplerInputError): evaluate_broadened(prepared(), energies, temperature=293.6)


def test_reject_cooling_and_bad_sources():
    with pytest.raises(DopplerInputError): evaluate_broadened(prepared(), [1.], temperature=-1.)
    with pytest.raises(DopplerInputError): prepared((([0., 1.], [-1., 2.]),))
    with pytest.raises(DopplerInputError): prepared((([0., 1.], [1., 2.]), ([2., 3.], [2., 3.])))
    with pytest.raises(DopplerInputError): DopplerContext(True)
    with pytest.raises(DopplerInputError): DopplerOptions(block_size=True)
    hot = prepare_curve((([0., 1.], [1., 1.]),), context=DopplerContext(1.),
                        source_temperature=600., support=SupportPolicy('zero','constant'))
    with pytest.raises(DopplerInputError): evaluate_broadened(hot, [1.], temperature=293.6)


def test_model_conversion_and_rejections():
    from kika.nuclear_data.model import XYs1d, Regions1d, crossSectionAxes, Axes
    form = XYs1d(xs=[0., 1e-6, 2e-6], ys=[1., 2., 3.], axes=Axes.forFunction1d('sigma','b','E','MeV'))
    p = prepare_curve(form, context=DopplerContext(1.), source_temperature=0.,
                      support=SupportPolicy('zero','constant'))
    assert np.array_equal(p.regions[0][0], [0., 1., 2.])
    with pytest.raises(DopplerInputError):
        prepare_curve(XYs1d(xs=[0.,1.],ys=[1.,1.]), context=DopplerContext(1.),
                      source_temperature=0., support=SupportPolicy('zero','zero'))
    with pytest.raises(DopplerInputError):
        prepare_curve(object(), context=DopplerContext(1.), source_temperature=0., support=SupportPolicy('zero','zero'))


def test_signed_linearity_and_block_invariance():
    x = [0., 1., 2., 4.]; a = np.array([1., 2., -1., 1.]); b = np.array([2., 3., 2., 1.])
    ps = [prepared(((x, y),), negative_policy="preserve_signed") for y in (a, b, a+b)]
    energies = np.geomspace(1e-8, 100, 50)
    rows = [evaluate_broadened(p, energies, temperature=1/KB).values for p in ps]
    assert np.allclose(rows[0]+rows[1], rows[2], rtol=1e-12, atol=1e-12)
    block = evaluate_broadened(ps[0], energies, temperature=1/KB, options=DopplerOptions(block_size=1))
    assert np.array_equal(rows[0], block.values)


def test_identically_zero_panels_need_no_gaussian_work(monkeypatch):
    import kika.processing.doppler.core as core
    def forbidden(*args):raise AssertionError('zero polynomial needs no moments')
    monkeypatch.setattr(core,'_moments',forbidden)
    p=prepared((([.1,1.,4.],[0.,0.,0.]),),lower='one_over_v')
    r=evaluate_broadened(p,[.01,1.,10.],temperature=600.)
    assert np.array_equal(r.values,[0.,0.,0.])
    assert r.report['analytic_subpanels']==0

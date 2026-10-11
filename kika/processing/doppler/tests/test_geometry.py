"""Geometry and imposed-zero regressions for the Doppler error estimates."""
import numpy as np
import pytest
from kika.processing.doppler import core, DopplerContext, DopplerOptions, SupportPolicy, prepare_curve, evaluate_broadened

# Gaussian polynomial integrals evaluated with 80 decimal digits; inputs are
# exact float64 numbers, including nearly coincident clipping boundaries.
@pytest.mark.parametrize("lo,hi,v0,v1,energy,lower,upper,reference", [(49.344769574304514, 49.34478669969569, -6.172910366454849, 7.213418599204137, 0.001006497522896499, 6.992857886241395, 6.992859105202657, 5.99534130424976e-24), (34.40083970899687, 34.401027545104974, 8.045458146877065, 5.740154220427454, 0.00016795188962871722, 5.852263278241, 5.852279290919016, 4.467588244035843e-15), (2.9876392254548425, 2.989042051869953, -1327853571.6295884, -837349588.9767413, 1.6362459207679577e-10, 1.7284660846128663, 1.728871834952396, -20166425714.3052)])
def test_narrow_clipping_geometry_is_included(lo,hi,v0,v1,energy,lower,upper,reference):
    assert core._native is not None
    value,error,_,_=core._native.integrate(((lo,hi,v0,v1,'linlin','source',1.),),0,1,energy,1.,lower,upper,80,200000)
    assert abs(value-reference)<=error+128*np.finfo(float).eps*abs(reference)

def test_forced_zero_retains_discarded_uncertainty_without_refinement_noise(monkeypatch):
    p=prepare_curve((([0.,2.],[1.,1.]),),context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('constant','constant'))
    monkeypatch.setattr(core,'_point_block',lambda curve,energies,h,options:[(2.,1e-7,2e-8,{'source':2.},1,8) for e in energies])
    result=evaluate_broadened(p,[.5,1.,1.5],temperature=600.,options=DopplerOptions(threshold_policy='zero_below',threshold=1.))
    assert np.array_equal(result.values,[0.,2.,2.])
    assert np.array_equal(result.kernel_error_estimates,[0.,1e-7,1e-7])
    assert np.array_equal(result.tail_bounds,[0.,2e-8,2e-8])
    assert result.report['max_removed_kernel_error']==1e-7
    assert result.report['max_removed_tail_bound']==2e-8

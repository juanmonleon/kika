"""Independent transport bookkeeping for absolute LSSF=0 and relative LSSF=1."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from kika.processing.resonances import probability_tables as pt
from kika.processing.tests.test_probability_tables import _job


def test_absolute_transport_does_not_rescale_with_smooth_partial_cross_sections():
    C=np.array([1.,3.]);S=np.array([[3.,21.],[1.,9.],[0.,0.],[2.,12.]]);T=S.sum(1)
    observed=pt._estimates(C,S,T,(10.,),(999.,888.,777.,5.),False)
    other=pt._estimates(C,S,T,(10.,),(1.,2.,3.,5.),False)
    np.testing.assert_array_equal(observed['shielding'],other['shielding'])
    probability=np.array([.25,.75]);weight=probability/(np.array([8.,12.])+10.)
    expected=(weight*np.array([2.,4.])).sum()/weight.sum()/3.5
    assert observed['shielding'][3,0]==pytest.approx(expected,abs=1e-15)


def test_lssf_zero_retains_raw_ladder_means_and_one_preserves_legacy_factors():
    job=_job(ladders=8)
    legacy,_=pt._tablesAt(job)
    relative,_=pt._tablesAt(job+([None],True))
    np.testing.assert_array_equal(relative,legacy)
    absolute,_,raw=pt._tablesAt(job+([None],False),keepLadders=True)
    count=raw['counts'].sum(1);sums=raw['sums'].sum(1)
    expected=np.divide(sums,count[:,None,:],out=np.zeros_like(sums),where=count[:,None,:]>0)
    # Per-ladder regrouping and event-wise summation have different rounding.
    np.testing.assert_allclose(absolute[:,1:5],expected,rtol=32*np.finfo(float).eps,atol=0.)
    np.testing.assert_array_equal(absolute[:,0],relative[:,0])
    assert not np.array_equal(absolute[:,1:5],relative[:,1:5])


@pytest.mark.parametrize('relative',[False,True])
def test_public_table_flag_follows_the_source_lssf(monkeypatch,relative):
    job=_job(ladders=4)
    monkeypatch.setattr(pt,'_prepared',lambda region,context:job[0])
    region=NS(tabulatedWidths=NS(selfShieldingOnly=relative))
    result=pt.probabilityTables(region,NS(atomic_weight_ratio=job[1]),[job[2]],[293.6],options=job[7])
    assert result[293.6].multiplySmooth is relative


def test_absolute_band_uncertainty_retains_fluctuating_dilute_normalization():
    # One band: all normalized factors are one, but absolute barns fluctuate.
    counts=np.array([[1.],[1.]])
    sums=np.array([[[3.],[1.],[0.],[2.]],[[7.],[3.],[0.],[4.]]])
    total=sums[:,:,0]
    statistics=pt._ladderStatistics(counts,sums,total,(10.,),None,False)
    np.testing.assert_array_equal(statistics['factorSigma'],np.zeros((4,1)))
    np.testing.assert_array_equal(statistics['crossSectionSigma'],np.array([[2.],[1.],[0.],[1.]]))

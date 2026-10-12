"""Construction certificates preserve checked chords and fresh verification."""
from copy import deepcopy
import numpy as np
import pytest
from kika.processing.doppler import (DopplerContext,DopplerOptions,SupportPolicy,
    prepare_curve,broaden_curve,DopplerInputError,DopplerConvergenceError)
from kika.processing.doppler.table import _verify_construction,_adaptive
from kika.processing.doppler.suite import _source_values

@pytest.mark.parametrize('lower',['zero','constant','one_over_v'])
@pytest.mark.parametrize('upper',['zero','constant'])
def test_vector_source_preserves_jumps_limits_signed_and_unsorted_queries(lower,upper):
    p=prepare_curve((([.1,1.,2.],[3.,-2.,4.]),([2.,3.,10.],[8.,0.,-1.])),
        context=DopplerContext(56.),source_temperature=0.,
        support=SupportPolicy(lower,upper),negative_policy='preserve_signed')
    q=np.r_[np.random.default_rng(4).uniform(.01,20.,500),.1,1.,2.,3.,10.,
        np.nextafter(2.,0.),np.nextafter(2.,np.inf)]
    vector=_source_values(p,q)
    scalar=np.array([p.evaluate_source(float(x)) for x in q])
    assert np.array_equal(vector,scalar)
    assert vector[-3]==-1.  # upper endpoint
    assert _source_values(p,np.array([2.]))[0]==8.  # right owns jump

@pytest.mark.parametrize('temperature',[0.,600.])
def test_certificate_rejects_mutable_publication_and_fresh_check_passes(temperature):
    p=prepare_curve((([.1,.8,1.,1.2,2.],[1.,1.,8.,1.,1.]),),
        context=DopplerContext(56.),source_temperature=0.,support=SupportPolicy('zero','constant'))
    result=broaden_curve(p,temperature=temperature)
    assert result.verify_table()<=1
    assert _verify_construction({0:result.form},result._tables,result.report['regions'],'heated')[0]<=1
    altered=deepcopy(result.form);altered.ys[0]+=1.
    with pytest.raises(DopplerInputError,match='published nodes differ'):
        _verify_construction({0:altered},result._tables,result.report['regions'],'heated')
    altered=deepcopy(result.form);altered.xs[0]*=.9
    with pytest.raises(DopplerInputError,match='published nodes differ'):
        _verify_construction({0:altered},result._tables,result.report['regions'],'heated')
    altered=deepcopy(result.form);altered.label='wrong'
    with pytest.raises(DopplerInputError,match='output form label changed'):
        _verify_construction({0:altered},result._tables,result.report['regions'],'heated')

def test_node_uncertainty_cannot_be_hidden_by_off_grid_probes():
    def evaluate(q):
        eta=np.where(q==1.,1.,0.)
        return {0:np.ones(len(q))},{0:eta}
    with pytest.raises(DopplerConvergenceError,match='node uncertainty exceeds budget'):
        _adaptive(evaluate,np.array([1.,2.]),DopplerOptions(),100,(0,))

def test_reserved_family_finds_feature_missed_by_curvature_prediction():
    options=DopplerOptions(rtol=.001,atol=1e-8)
    def evaluate(q):
        return {0:1.+np.maximum(0.,1.-np.abs(q-1.3300094782075719)/.03)},{0:np.zeros(len(q))}
    x,y,report=_adaptive(evaluate,np.array([1.,2.]),options,10000,(0,))
    q=np.linspace(1.,2.,20001)
    assert np.max(np.abs(np.interp(q,x,y[0])-evaluate(q)[0][0])/(options.atol+options.rtol*evaluate(q)[0][0]))<=1
    assert report['iterations']>1


def test_numeric_certificates_invalidate_split_or_removed_endpoints():
    from kika.processing.doppler.table import _reused_intervals,_retain_intervals
    grid=np.array([1.,2.,3.,4.])
    certificates=_retain_intervals(grid,np.array([True,True,True]),2)
    assert certificates.shape==(2,2) and certificates.nbytes==32
    assert np.array_equal(_reused_intervals(grid,certificates),[True,True,False])
    assert np.array_equal(_reused_intervals(np.array([1.,1.5,2.,3.,4.]),certificates),[False,False,True,False])
    assert not np.any(_reused_intervals(np.array([1.,4.]),certificates))
    assert _retain_intervals(grid,np.ones(3,dtype=bool),0).shape==(0,2)

@pytest.mark.parametrize('block',[1,2,7,23,1000])
def test_chunked_features_match_whole_source_at_every_chunk_edge(block):
    from kika.processing.doppler.table import _source_features
    x=np.arange(257,dtype=float)+1.
    y=np.sin(x);y[50:90]=1.;y[123:150]=0.
    begin,end=3,253
    dy=np.diff(y[begin:end])
    indices=np.flatnonzero((dy[:-1]*dy[1:]<0)|((dy[:-1]==0)!=(dy[1:]==0)))+begin+1
    expected=np.r_[x[0],x[-1],x[indices]]
    assert np.array_equal(list(_source_features(x,y,begin,end,block*128)),expected)

def test_source_feature_scratch_depends_on_chunks_not_cold_table_size():
    from kika.processing.doppler.table import _seeds
    x=np.geomspace(.1,10.,10001)
    p=prepare_curve(((x,1.+np.exp(-((x-2.)/.03)**2)),),context=DopplerContext(56.),
        source_temperature=0.,support=SupportPolicy('zero','constant'))
    small=_seeds((p,),600.,.1,10.,10000,128*17)
    large=_seeds((p,),600.,.1,10.,10000,64*1024*1024)
    assert np.array_equal(small,large)

def test_no_value_cache_allocates_without_reuse():
    from kika.processing.doppler.table import _ConstructionSamples
    calls=[]
    def evaluate(q,left_boundary=None):
        calls.append((q.copy(),left_boundary));return {0:q.copy()},{0:np.zeros(len(q))}
    cache=_ConstructionSamples(evaluate,(0,),DopplerOptions())
    cache(np.array([1.,2.]),left_boundary=2.)
    report=cache.finish()
    assert len(calls)==1 and calls[0][1]==2.
    assert report['cache_allowance_bytes']==0 and report['maximum_entries']==0
    assert report['source_evaluations']==2

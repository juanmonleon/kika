"""Prepared-panel ownership and temporary construction sample reuse."""
import copy
import gc
import pickle
import numpy as np
import pytest
from kika.processing.doppler import (core, DopplerContext, DopplerOptions, SupportPolicy,
    prepare_curve, evaluate_broadened, broaden_curve)
from kika.processing.doppler import table

def prepared():
    return prepare_curve((([.1,.8,1.,1.2,2.],[1.,1.,8.,1.,1.]),),
        context=DopplerContext(56.),source_temperature=0.,support=SupportPolicy('constant','constant'))

def test_numeric_panels_match_tuples_and_own_source_labels():
    native=core._native
    source=((0.,400.,2.,2.,'linlin',''.join(['retained','-label']),1.),)
    packed,storage=native.prepare_panels(source)
    ordinary=native.integrate(source,0,1,100.,1.,-8.,8.,80,10000)
    del source;gc.collect()
    cached=native.integrate(packed,0,1,100.,1.,-8.,8.,80,10000)
    assert ordinary==cached and storage>0
    with pytest.raises(ValueError):native.prepare_panels(((0.,0.,1.,1.,'linlin','x',1.),))

@pytest.mark.parametrize('clone',[copy.deepcopy,lambda p:pickle.loads(pickle.dumps(p))])
def test_prepared_curve_serialization_rebuilds_cache_and_preserves_immutability(clone):
    p=prepared();restored=clone(p)
    assert restored.source_hash==p.source_hash
    assert restored._native_panels is not p._native_panels
    assert restored._native_panel_bytes>0
    with pytest.raises(ValueError):restored.regions[0][0].setflags(write=True)
    a=evaluate_broadened(p,[.2,1.,1.8],temperature=600.)
    b=evaluate_broadened(restored,[.2,1.,1.8],temperature=600.)
    assert np.array_equal(a.values,b.values)

def test_sample_keys_distinguish_left_limits_and_memory_is_bounded():
    calls=[]
    def evaluate(q,left_boundary=None):
        calls.append(q.copy())
        values=q.copy();values[q==left_boundary]=-1.
        return {0:values},{0:np.zeros(len(q))}
    cache=table._ConstructionSamples(evaluate,(0,),DopplerOptions(max_work_bytes=4096,block_size=1))
    assert cache([1.,2.],left_boundary=2.)[0][0][-1]==-1.
    assert cache([2.])[0][0][0]==2.
    for e in range(3,30):cache([float(e)])
    assert len(cache.samples)<=cache.limit
    assert cache.limit*cache.entry_estimate<=4096//8
    assert all(len(q)<=cache.batch for q in calls)
    assert cache.finish()['retained_entries_after_construction']==0
    assert not cache.samples

def test_construction_cache_preserves_grid_and_later_verification_is_fresh(monkeypatch):
    p=prepared();cached=broaden_curve(p,temperature=600.)
    assert cached.report['construction_samples']['source_evaluations']<=sum(r['evaluations'] for r in cached.report['regions'])
    original=table._ConstructionSamples.__init__
    def disabled(self,*args):
        original(self,*args);self.limit=0
    monkeypatch.setattr(table._ConstructionSamples,'__init__',disabled)
    uncached=broaden_curve(p,temperature=600.)
    assert np.array_equal(cached.form.xs,uncached.form.xs)
    assert np.array_equal(cached.form.ys,uncached.form.ys)
    calls=[];evaluate=table._evaluate_curve
    def counting(*args,**kwargs):
        calls.append(len(args[1]));return evaluate(*args,**kwargs)
    monkeypatch.setattr(table,'_evaluate_curve',counting)
    assert cached.verify_table()<=1 and sum(calls)>0

def test_imposed_threshold_tables_converge_with_realistic_kernel_uncertainty(monkeypatch):
    p=prepared();block=core._point_block
    def uncertain(curve,energies,h,options):
        result=[]
        for v,error,tail,component,count,radius in block(curve,energies,h,options):
            error=max(error,.02*options.kernel_share*(options.atol+options.rtol*abs(v)))
            result.append((v,error,tail,component,count,radius))
        return result
    monkeypatch.setattr(core,'_point_block',uncertain)
    result=broaden_curve(p,temperature=600.,options=DopplerOptions(threshold_policy='zero_below',threshold=1.))
    assert result.form.function1ds[0].ys[-1]==0.
    assert result.form.function1ds[1].ys[0]>0.
    assert result.verify_table()<=1

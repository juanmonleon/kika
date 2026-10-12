"""Dense source block moments: bounds, ownership, caches and fallback."""
import gc,pickle,copy
import numpy as np
import pytest
from kika.processing.doppler import (core,DopplerContext,DopplerOptions,SupportPolicy,
    prepare_curve,evaluate_broadened,broaden_curve)
from kika.processing.doppler import _blocks

pytestmark=pytest.mark.skipif(core._native is None or not hasattr(core._native,'prepare_blocks'),reason='native source moments unavailable')

@pytest.fixture
def force(monkeypatch):
    monkeypatch.setattr(_blocks,'_DENSE_SOURCE',0)
    monkeypatch.setattr(_blocks,'_MIN_DIRECT_QUERIES',0)

def prepared(kind=0,lower='zero',upper='constant',signed=False):
    x=np.geomspace(.001,10.,5001)
    if kind==0:y=1.+30*np.exp(-((x-.3)/.002)**2)
    elif kind==1:y=1./np.sqrt(x)
    elif kind==2:y=np.where((x>.1)&(x<.2),1e8,0.)
    else:y=np.sin(50*x)
    return prepare_curve(((x[:2501],y[:2501]),(x[2500:],2*y[2500:])),
        context=DopplerContext(236.),source_temperature=0.,support=SupportPolicy(lower,upper),
        negative_policy='preserve_signed' if signed else 'reject')

@pytest.mark.parametrize('temperature',[1.,293.6,600.,1200.,1e5])
@pytest.mark.parametrize('kind',[0,1,2])
def test_dense_points_inside_error_diagnostics_and_owner_closure(force,temperature,kind,monkeypatch):
    p=prepared(kind);q=np.geomspace(1e-5,100.,401)
    baseline=evaluate_broadened(p,q,temperature=temperature)
    if not (kind==2 and temperature==1.):
        assert baseline.report['moment_blocks']['accepted_blocks']>0
    monkeypatch.setattr(_blocks,'_DENSE_SOURCE',10**9)
    legacy=evaluate_broadened(p,q,temperature=temperature)
    diff=np.abs(baseline.values-legacy.values)
    diagnostic=(baseline.kernel_error_estimates+baseline.tail_bounds+
                legacy.kernel_error_estimates+legacy.tail_bounds)
    assert np.all(diff<=diagnostic+1e-300)
    components=baseline.report['components_sum_over_probes']
    assert sum(components.values())==pytest.approx(float(np.sum(baseline.values)),rel=1e-12)

def test_sparse_signed_identity_and_small_scratch_use_legacy(force):
    p=prepared(3,signed=True)
    r=evaluate_broadened(p,[.1,.3,1.],temperature=600.)
    assert r.report['moment_blocks']['accepted_blocks']==0
    p=prepared();r=evaluate_broadened(p,[.1,.3,1.],temperature=0.)
    assert r.report['moment_blocks']['accepted_blocks']==0
    options=DopplerOptions(max_work_bytes=32*1024*1024)
    r=evaluate_broadened(p,[.1,.3,1.],temperature=600.,options=options)
    assert r.report['moment_blocks']['accepted_blocks']==0

def test_temperatures_replace_cache_and_clone_discards_process_local_storage(force):
    p=prepared();q=np.geomspace(.01,1.,40)
    a=evaluate_broadened(p,q,temperature=600.)
    capsule=next(iter(p._block_moment_cache.values()))[0]
    evaluate_broadened(p,q,temperature=1200.)
    assert len(p._block_moment_cache)==1
    assert next(iter(p._block_moment_cache.values()))[0] is not capsule
    for clone in (copy.deepcopy(p),pickle.loads(pickle.dumps(p))):
        assert not clone._block_moment_cache
        assert np.array_equal(evaluate_broadened(clone,q,temperature=600.).values,a.values)

def test_moment_capsule_owns_source_and_validates_shape_and_indices(force):
    p=prepared();q=np.array([.1,.3,1.]);evaluate_broadened(p,q,temperature=600.)
    capsule,metadata=next(iter(p._block_moment_cache.values()))
    assert metadata['numeric_cache_bytes']<=DopplerOptions().max_work_bytes//16
    expected=core._native.block_values_cached(capsule,q)
    del p;gc.collect()
    assert core._native.block_values_cached(capsule,q)==expected
    with pytest.raises(ValueError):core._native.block_values_cached(capsule,np.array([0.]))
    with pytest.raises(ValueError):core._native.block_values_cached(capsule,q.astype(np.float32))
    p=prepared()
    with pytest.raises(ValueError):core._native.prepare_blocks(p._native_panels,np.zeros((1,23)),.01)
    invalid=np.zeros((1,24));invalid[0,0:6]=[.001,.01,.05,.1,0,999999]
    with pytest.raises(ValueError):core._native.prepare_blocks(p._native_panels,invalid,.01)

def test_unaligned_queries_and_forced_kernel_rejection(force,monkeypatch):
    p=prepared();q=np.array([.1,.3,1.]);a=evaluate_broadened(p,q,temperature=600.)
    storage=bytearray(1+q.nbytes);unaligned=np.ndarray(q.shape,dtype=float,buffer=storage,offset=1);unaligned[:]=q
    b=evaluate_broadened(p,unaligned,temperature=600.)
    assert np.array_equal(a.values,b.values)
    original=core._native.block_values_cached
    def uncertain(*args):
        return [(v,1e100,spans,components,n) for v,e,spans,components,n in original(*args)]
    monkeypatch.setattr(core._native,'block_values_cached',uncertain)
    forced=evaluate_broadened(p,q,temperature=600.)
    assert forced.report['moment_blocks']['legacy_fallbacks']==len(q)
    monkeypatch.setattr(_blocks,'_DENSE_SOURCE',10**9)
    legacy=evaluate_broadened(p,q,temperature=600.)
    assert np.array_equal(forced.values,legacy.values)

def test_threshold_convention_and_fresh_table_verification(force):
    p=prepared();options=DopplerOptions(threshold_policy='zero_below',threshold=.2)
    result=broaden_curve(p,temperature=600.,options=options,output_domain=(.05,.4))
    assert result.verify_table()<=1
    points=result.evaluate([.1,.2,.3])
    assert points.values[0]==0 and points.kernel_error_estimates[0]==0 and points.tail_bounds[0]==0


def test_cached_zero_support_still_rejects_unrepresentable_scaled_energy(force):
    p=prepared(upper='zero');evaluate_broadened(p,[.1,.3,1.],temperature=600.)
    with pytest.raises(core.DopplerConvergenceError) as caught:
        evaluate_broadened(p,[1e308],temperature=600.)
    assert caught.value.category=='unresolvable-scale'

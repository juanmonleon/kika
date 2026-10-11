"""Block control against scalar control, failure attribution and buffer ownership."""
import numpy as np
import pytest
from kika.processing.doppler import core, DopplerOptions, DopplerContext, SupportPolicy, prepare_curve, evaluate_broadened, DopplerConvergenceError

def curve():
    return prepare_curve((([0.,.1,1.,2.,3.],[2.,4.,1.,2.,1.]),),
        context=DopplerContext(56.),source_temperature=0.,support=SupportPolicy('zero','constant'))

@pytest.mark.parametrize('terms',[8,32,79,80,81,160])
def test_block_control_is_identical_to_scalar_native_control(terms):
    p=curve();h=np.sqrt(core.KB*600/56)
    options=DopplerOptions(max_series_terms=terms)
    q=np.geomspace(1e-9,1e4,25)
    try:
        reference=[core._point(p,float(e),h,options) for e in q]
    except DopplerConvergenceError as scalar:
        with pytest.raises(DopplerConvergenceError) as block:
            core._point_block(p,q,h,options)
        assert (block.value.category,block.value.energy,block.value.budget)==(scalar.category,scalar.energy,scalar.budget)
    else:
        assert core._point_block(p,q,h,options)==reference

@pytest.mark.parametrize('layout',['strided','reversed','unaligned'])
def test_query_buffer_layout_preserves_physics(layout):
    q=np.geomspace(1e-6,10.,40)
    if layout=='strided':q=q[::2]
    elif layout=='reversed':q=q[::-1]
    else:
        raw=bytearray(1+q.nbytes);a=np.ndarray(q.shape,dtype='d',buffer=raw,offset=1);a[:]=q;q=a
    p=curve()
    a=evaluate_broadened(p,q,temperature=600.)
    b=evaluate_broadened(p,q.copy(),temperature=600.)
    for name in ['values','kernel_error_estimates','tail_bounds']:
        assert np.array_equal(getattr(a,name),getattr(b,name))

def test_block_memory_allowance_reduces_batch_without_changing_answers():
    p=curve();q=np.geomspace(1e-6,10.,40)
    standard=DopplerOptions(block_size=32)
    base=4096+standard.max_series_terms*128+standard.max_panels*128
    per_point=512+128*len({panel[5] for panel in p._panels})
    constrained=DopplerOptions(block_size=32,max_work_bytes=base+3*per_point)
    a=evaluate_broadened(p,q,temperature=600.,options=constrained)
    b=evaluate_broadened(p,q,temperature=600.,options=standard)
    assert a.report['native_block_size']==3
    assert a.report['scratch_estimate_bytes']<=constrained.max_work_bytes
    assert np.array_equal(a.values,b.values)
    assert a.report['components_sum_over_probes']==b.report['components_sum_over_probes']

def test_late_failure_has_energy_and_no_partial_result():
    with pytest.raises(DopplerConvergenceError) as error:
        evaluate_broadened(curve(),[1.,1e308],temperature=600.)
    assert error.value.category=='unresolvable-scale'
    assert error.value.energy==1e308

@pytest.mark.parametrize('changed',[dict(max_panels=1),dict(max_radius=1),dict(max_series_terms=8,rtol=1e-10)])
def test_block_failure_matches_scalar_category_and_energy(changed):
    p=prepare_curve((([0.,1.],[2.,2.]),),context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('zero','constant'))
    options=DopplerOptions(**changed)
    failures=[]
    for f in [lambda:core._point(p,1.,1.,options),lambda:core._point_block(p,np.array([1.]),1.,options)]:
        with pytest.raises(DopplerConvergenceError) as error:f()
        failures.append((error.value.category,error.value.energy))
    assert failures[0]==failures[1]

def test_invalid_native_buffer_releases_exports():
    p=curve();q=np.array([1.]);table=core._NATIVE_TAIL_TERMS.copy();table[0,0]=np.nan
    with pytest.raises(ValueError):
        core._native.evaluate_block(p._native_panels,q,table,1.,4.,0.,1e-8,1e-3,.1,.15,80,200000,26,False)
    q.resize(2);table.resize((30,3))

def test_preparation_rejects_unordered_panels_for_binary_search():
    with pytest.raises(ValueError,match='unordered'):
        core._native.prepare_panels(((1.,2.,1.,1.,'constant','source',1.),(0.,1.,1.,1.,'constant','source',1.)))

def test_derived_envelope_overflow_is_a_convergence_failure():
    p=prepare_curve((([1e308,1.1e308],[1e308,1e308]),),context=DopplerContext(1.),
        source_temperature=0.,support=SupportPolicy('one_over_v','constant'))
    with pytest.raises(DopplerConvergenceError) as error:
        evaluate_broadened(p,[1.],temperature=600.)
    assert error.value.category=='unresolvable-scale' and error.value.energy==1.

def test_previous_block_is_released_before_next_allocation(monkeypatch):
    import weakref
    class Rows(list):pass
    original=core._point_block;references=[]
    def tracked(*args):
        assert all(ref() is None for ref in references)
        rows=Rows(original(*args));references.append(weakref.ref(rows));return rows
    monkeypatch.setattr(core,'_point_block',tracked)
    result=evaluate_broadened(curve(),np.geomspace(1e-6,10.,20),temperature=600.,options=DopplerOptions(block_size=3))
    assert len(result.values)==20 and len(references)==7
    assert all(ref() is None for ref in references)

def test_immutable_source_hash_is_reused_without_changing_identity(monkeypatch):
    import hashlib
    p=curve();calls=[];original=core.sha256
    def counting(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(core,'sha256',counting)
    first=evaluate_broadened(p,[.1,1.],temperature=600.)
    second=evaluate_broadened(p,[.2,2.],temperature=600.)
    expected=hashlib.sha256(repr((p.context,p.support,p.source_temperature,p.negative_policy,p.source_temperature_origin)).encode())
    for x,y in p.regions:expected.update(x.tobytes());expected.update(y.tobytes())
    assert first.report['source_hash']==second.report['source_hash']==expected.hexdigest()
    assert len(calls)==1

"""Staged rejection and exact reuse preserve all accepted-panel controls."""
import numpy as np
import pytest
from kika.algebra import refine

FRACTIONS=np.array([.25,.5,.75,.1127016653792583,.276393202250021,
                    .723606797749979,.8872983346207417])

@pytest.mark.parametrize('precheck',[0,1,3])
@pytest.mark.parametrize('reuse',[False,True])
def test_every_final_panel_keeps_all_its_actual_probes(precheck,reuse):
    calls=[]
    def physics(q,owner):
        calls.extend(q.tolist())
        return np.column_stack((q*q,np.exp(q)))
    x=np.array([0.,1.]);y=physics(x,None)
    calls.clear()
    result=refine(x,y,physics,lambda a,b:abs(a-b)/1e-4,
        fractions=FRACTIONS,insert='balanced',keep_probes=True,
        precheck=precheck,reuse_probes=reuse)
    assert result.evaluations==len(calls)
    assert result.probe_x.shape==(len(result.x)-1,7)
    np.testing.assert_array_equal(result.probe_y[:,:,0],result.probe_x**2)
    np.testing.assert_array_equal(result.probe_y[:,:,1],np.exp(result.probe_x))
    f=(result.probe_x-result.x[:-1,None])/np.diff(result.x)[:,None]
    chord=result.y[:-1,None,:]+np.diff(result.y,axis=0)[:,None,:]*f[:,:,None]
    assert np.max(abs(result.probe_y-chord))<=1e-4
    if precheck or reuse:
        baseline=refine(x,y,physics,lambda a,b:abs(a-b)/1e-4,
            fractions=FRACTIONS,insert='balanced',keep_probes=True)
        assert result.evaluations<baseline.evaluations

def test_additional_probe_can_reject_a_panel_that_passes_the_prefix():
    center=FRACTIONS[3]
    def physics(q,owner):return np.exp(-((q-center)/.004)**2)
    result=refine([0.,1.],physics(np.array([0.,1.]),None),physics,
        lambda a,b:abs(a-b)/1e-3,fractions=FRACTIONS,insert='balanced',
        precheck=3,reuse_probes=True,keep_probes=True)
    assert center in result.x
    assert np.max(result.y)>.99
    f=(result.probe_x-result.x[:-1,None])/np.diff(result.x)[:,None]
    chord=result.y[:-1,None]+np.diff(result.y)[:,None]*f
    assert np.max(abs(result.probe_y-chord))<=1e-3

def test_cache_uses_starting_interval_identity_at_repeated_boundary():
    x=np.array([0.,1.,1.,2.]);y=np.array([0.,1.,11.,14.])
    def physics(q,owner):return q*q+np.where(owner==2,10.,0.)
    result=refine(x,y,physics,lambda a,b:abs(a-b)/.03,
        snap=lambda q:np.round(q*4)/4,insert='balanced',
        unresolvable='accept',precheck=2,reuse_probes=True)
    expected=result.x**2+np.where(result.x>1.,10.,0.)
    # The two original limits at the repeated boundary retain their values.
    expected[np.flatnonzero(result.x==1.)[1]]=11.
    np.testing.assert_array_equal(result.y,expected)

def test_unresolvable_kept_panel_has_no_missing_staged_probes():
    result=refine([0.,1.],[0.,0.],lambda q,owner:np.ones(len(q)),
        lambda a,b:abs(a-b)*10,fractions=FRACTIONS,insert='balanced',
        min_width_ulps=1e20,unresolvable='accept',keep_probes=True,precheck=3)
    assert result.unresolved==1
    np.testing.assert_array_equal(result.probe_y,np.ones((1,7)))

@pytest.mark.parametrize('precheck',[-1,8,True,1.2])
def test_invalid_precheck_rejected(precheck):
    with pytest.raises(ValueError,match='precheck'):
        refine([0.,1.],[0.,1.],lambda q,o:q,lambda a,b:abs(a-b),precheck=precheck)

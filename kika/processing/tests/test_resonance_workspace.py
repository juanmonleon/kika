"""R8 batching preserves physics, meshes, failure budgets and publication probes."""
from dataclasses import replace
import numpy as np
import pytest
from kika.processing.resonances.grid import (ReconstructionOptions,linearize,
    verification_batches,VERIFICATION_FRACTIONS,ReconstructionConvergenceError)
from kika.processing.resonances.prepare import prepare_resonances,evaluate_region,energy_block_size
from kika.processing.resonances import NeutronContext
from test_r_matrix import rml_model
from test_reich_moore import rm_model
from kika.processing.resonances.reich_moore import solve_collision
from kika.processing.resonances.breit_wigner import Level,Group,evaluate_bw
from kika.processing.resonances.radii import RadiusFunction
from test_resonance_suite import run,suite_model
from kika.processing.resonances import attach_reconstruction


def test_verification_reuses_only_unchanged_numeric_contents(monkeypatch):
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    calls=[];original=type(result)._evaluate_segment
    def counted(self,*args,**kwargs):
        calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(type(result),'_evaluate_segment',counted)
    first=result.verify_suite(suite)
    assert not calls
    first[next(iter(first))]=1e99
    assert max(result.verify_suite(suite).values())<1
    curve=suite.reactions['capture'].crossSection['recon'].function1ds[0]
    curve.ys+=1e-13
    assert max(result.verify_suite(suite).values())<1
    assert calls
    calls.clear()
    curve.ys*=2.
    with pytest.raises(ReconstructionConvergenceError):result.verify_suite(suite)
    assert calls


def test_verified_snapshot_cannot_hide_a_units_or_coverage_edit():
    from copy import deepcopy
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    changed=deepcopy(suite)
    axes=changed.reactions['capture'].crossSection['recon'].function1ds[0].axes
    axes.axes=[replace(axis,unit='mb') if axis.index==0 else axis for axis in axes.axes]
    with pytest.raises(ReconstructionConvergenceError):result.verify_suite(changed)
    changed=deepcopy(suite);changed.reactions.reactions.pop()
    with pytest.raises(ValueError):result.verify_suite(changed)


@pytest.mark.parametrize('approximation',['SingleLevel','MultiLevel'])
def test_bounded_s_wave_matches_ordered_independent_scalar_level_equations(approximation):
    ctx=NeutronContext(56.,1.);radius=RadiusFunction(constant=5.)
    levels=tuple(Level((-10. if i==0 else 10.+i*.7),(.5 if i%3 else 1.5),
        .001*(i+1),.01,.003) for i in range(265))
    group=Group(0,radius,radius,levels)
    e=np.unique(np.r_[np.geomspace(.01,300.,400),[lv.energy for lv in levels if lv.energy>0]])
    p=np.sqrt(ctx.k_squared_per_ev*e)*5.;beta=np.pi*.01/(ctx.k_squared_per_ev*e)
    sin2=np.sin(p)**2;sd=np.sin(2*p);el=np.zeros(len(e));cap=el.copy();fis=el.copy();amplitudes={}
    if approximation=='SingleLevel':el+=beta*4*sin2
    for lv in levels:
        gn=lv.neutron*p/(np.sqrt(ctx.k_squared_per_ev*abs(lv.energy))*5.)
        width=gn+lv.capture+lv.fission+np.zeros(len(e));delta=e-lv.energy
        denominator=delta**2+(width/2)**2;g=(2*lv.spin+1)/(2*(2*ctx.target_spin+1))
        cap+=beta*g*gn*lv.capture/denominator;fis+=beta*g*gn*lv.fission/denominator
        if approximation=='SingleLevel':el+=beta*g*gn*(gn-2*width*sin2+2*delta*sd)/denominator
        else:
            t1,t2=amplitudes.setdefault(lv.spin,(np.zeros(len(e)),np.zeros(len(e))))
            t1+=gn*width/2/denominator;t2+=gn*delta/denominator
    if approximation=='MultiLevel':
        represented=0.
        for spin,(t1,t2) in amplitudes.items():
            g=(2*spin+1)/(2*(2*ctx.target_spin+1));represented+=g
            el+=beta*g*((2*sin2-t1)**2+(sd+t2)**2)
        el+=beta*(1-represented)*4*sin2
    expected={2:el,102:cap,18:fis,1:el+cap+fis}
    for work_bytes in (32768,64*1024**2):
        actual=evaluate_bw(e,(group,),approximation,ctx,work_bytes=work_bytes)
        for mt in expected:np.testing.assert_allclose(actual[mt],expected[mt],rtol=2e-14,atol=1e-12)


def test_dense_pole_workspace_exhaustion_is_an_explanatory_raise():
    with pytest.raises(ReconstructionConvergenceError) as failure:
        solve_collision(np.array([100.]),np.array([100.,100.]),np.zeros(2),
            np.ones((1,2,2)),work_bytes=1)
    assert failure.value.category=='memory-budget-exhausted'


@pytest.mark.parametrize('entrance',[0,1,2])
def test_separable_rm_matches_generic_amplitudes_at_regular_and_undamped_poles(entrance):
    e=np.array([.01,99.9,100.,100.1,100.2,200.])
    levels=np.array([100.,100.,100.2]);gamma=np.array([0.,0.,.02])
    reduced=np.array([[.4,-.2,.3],[.2,.3,-.1],[.3,-.4,.2]])
    factors=np.column_stack((np.sqrt(e),np.ones(len(e)),np.ones(len(e))))
    amplitudes=factors[:,None,:]*reduced[None,:,:]
    expected=solve_collision(e,levels,gamma,amplitudes,entrance=entrance)
    diagnostics={}
    actual=solve_collision(e,levels,gamma,None,diagnostics,entrance=entrance,
        reduced=reduced,channel_factors=factors)
    for value,reference in zip(actual,expected):
        np.testing.assert_allclose(value,reference,rtol=3e-12,atol=2e-12)
    assert diagnostics['rm_max_solver_residual']<1e-11


def test_streamed_verification_has_every_original_probe_in_the_same_order():
    grid=np.r_[1.,np.nextafter(1.,2.),2.,3.,100.]
    expected=np.r_[grid,(grid[:-1,None]+np.diff(grid)[:,None]*VERIFICATION_FRACTIONS).ravel()]
    for batch in (1,3,17,4096):
        blocks=list(verification_batches(grid,batch))
        assert max(map(len,blocks))<=batch
        np.testing.assert_array_equal(np.concatenate(blocks),expected)


def test_refinement_chunks_preserve_grid_values_and_evaluation_count():
    options=ReconstructionOptions(rtol=.005,max_points=20000,max_work_bytes=2**30)
    seeds=np.geomspace(1.,10.,97)
    def physics(e):return {2:1.+np.sin(e)**2,102:np.exp(-e),1:1.+np.sin(e)**2+np.exp(-e)}
    before=linearize(physics,seeds,options,options.max_points)
    after=linearize(physics,seeds,replace(options,max_work_bytes=16384),options.max_points)
    np.testing.assert_array_equal(before[0],after[0])
    for mt in before[1]:
        np.testing.assert_array_equal(before[1][mt],after[1][mt])
        assert before[2]['refinement_maxima'][mt]==after[2]['refinement_maxima'][mt]
        assert before[2]['verification_maxima'][mt]==after[2]['verification_maxima'][mt]
        for weight in ('dE','dE_over_E'):
            for name in before[2]['integrals'][mt][weight]:
                np.testing.assert_allclose(before[2]['integrals'][mt][weight][name],
                    after[2]['integrals'][mt][weight][name],rtol=2e-14,atol=1e-15)
    assert before[2]['evaluations']==after[2]['evaluations']
    assert after[2]['refinement_chunks']>1
    for work_bytes in (16384,2**30):
        with pytest.raises(ReconstructionConvergenceError) as failure:
            linearize(physics,seeds,replace(options,max_work_bytes=work_bytes),100)
        assert failure.value.category=='budget-exhausted'


@pytest.mark.parametrize('kind',['RM','RML'])
def test_exact_physics_stable_between_workspace_sizes_and_at_poles(kind):
    model,ctx=(rm_model(),NeutronContext(56.,0.)) if kind=='RM' else rml_model(multiple=True)
    region=prepare_resonances(model,ctx).regions[0]
    e=np.unique(np.r_[np.linspace(10.,299.,2500),100.,100.2])
    before=evaluate_region(e,region,ctx,work_bytes=4096)
    after=evaluate_region(e,region,ctx,work_bytes=64*1024**2)
    for mt in before:np.testing.assert_allclose(after[mt],before[mt],rtol=2e-12,atol=1e-11)
    assert energy_block_size(region,work_bytes=4096)<energy_block_size(region)
    with pytest.raises(ReconstructionConvergenceError) as failure:
        evaluate_region(e,region,ctx,work_bytes=1)
    assert failure.value.category=='memory-budget-exhausted'


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_workspace_option(value):
    with pytest.raises(ValueError,match='max_work_bytes'):ReconstructionOptions(max_work_bytes=value)


def test_compacted_first_reaction_keeps_independent_master_verification_grid(monkeypatch):
    from test_resonance_suite import suite_model
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.nuclear_data.model import Regions1d
    suite=suite_model()
    suite.reactions.reactions.insert(0,suite.reactions.reactions.pop())  # Constant MT16 first.
    result=reconstruct_suite(suite,NeutronContext(56.,0.))
    first=next(iter(result.forms.values()))
    curves=first.function1ds if isinstance(first,Regions1d) else [first]
    assert all(len(c.xs)==2 for c in curves)
    assert len(result._grids[0])>2
    with pytest.raises(ValueError):result._grids[0].setflags(write=True)
    seen=[];original=type(result)._evaluate_segment
    def record(self,segment,points,*args,**kwargs):
        seen.extend(points)
        return original(self,segment,points,*args,**kwargs)
    monkeypatch.setattr(type(result),'_evaluate_segment',record)
    attach_reconstruction(suite,result)
    assert len(seen)==sum(len(g)+4*(len(g)-1) for g in result._grids)
    assert result.report['stored_points']<sum(len(g) for g in result._grids)*len(result.forms)


def test_explicit_domain_api_also_verifies_master_when_first_sum_is_constant(monkeypatch):
    from test_resonance_tabulation import prepared,constant
    from kika.processing.resonances import tabulate_resonances
    from kika.processing.resonances.tabulate import _Segment
    result=tabulate_resonances(prepared(),backgrounds={16:constant(.5)},sums={1:(16,)})
    assert len(result.forms[1].xs)==2 and len(result._grids[0])>2
    seen=[];original=_Segment.evaluate
    def record(self,points,*args,**kwargs):
        seen.extend(points)
        return original(self,points,*args,**kwargs)
    monkeypatch.setattr(_Segment,'evaluate',record)
    assert max(result.verify_forms(result.forms).values())<=1
    assert len(seen)==sum(len(g)+4*(len(g)-1) for g in result._grids)


def test_physical_witness_rechecks_changed_tables_without_recomputing_kernel(monkeypatch):
    from test_resonance_suite import suite_model
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    import kika.processing.resonances.suite as module
    suite=suite_model();result=reconstruct_suite(suite,NeutronContext(56.,0.))
    attach_reconstruction(suite,result)
    witnesses=result._verification_cache['physical_witnesses'][1]
    frozen=next(value for batch in witnesses.values() for _,partial in batch.values() for value in partial.values())
    with pytest.raises(ValueError):frozen.setflags(write=True)
    def forbidden(*args,**kwargs):raise AssertionError('frozen physical reference was recomputed')
    monkeypatch.setattr(module,'evaluate_region',forbidden)
    curve=suite.reactions['capture'].crossSection['recon'].function1ds[0]
    curve.ys*=1.+1e-7
    assert max(result.verify_suite(suite).values())<=1
    curve.ys*=2.
    with pytest.raises(ReconstructionConvergenceError,match='budget'):
        result.verify_suite(suite)


def test_linear_background_keeps_own_knots_while_master_grid_is_verified():
    from kika.nuclear_data.model import XYs1d
    from test_resonance_tabulation import AXES
    from kika.algebra import evaluate
    suite=suite_model()
    suite.reactions['extra'].crossSection['eval']=XYs1d([10.,42.,1000.],[.5,.9,1.2],axes=AXES,label='eval')
    result=run(suite);form=result.forms_by_mt[16]
    assert sum(len(c.xs) for c in form.function1ds)==5
    assert sum(len(g) for g in result._grids)>100
    attach_reconstruction(suite,result)
    for c in suite.reactions['extra'].crossSection['recon'].function1ds:
        e=np.linspace(c.domainMin,c.domainMax,101)
        np.testing.assert_allclose(evaluate(c.xs,c.ys,2,e),evaluate([10.,42.,1000.],[.5,.9,1.2],2,e),rtol=2e-15)
    assert max(result.verify_suite(suite).values())<=1


@pytest.mark.parametrize('gamma',[.03,1e-30,1e-310,0.])
@pytest.mark.parametrize('roundoff',[1e-11,1e-8])
def test_guarded_gram_capture_matches_direct_levels_and_preserves_poles(gamma,roundoff):
    rng=np.random.default_rng(782)
    reduced=rng.normal(size=(73,3))*.2
    levels=np.linspace(50.,150.,73);widths=np.full(73,gamma)
    e=np.unique(np.r_[np.linspace(20.,180.,101),levels[0],levels[36]])
    factors=np.ones((len(e),3));factors[:,0]=np.sqrt(e/100.)
    w,x=solve_collision(e,levels,widths,None,reduced=reduced,channel_factors=factors)
    diagnostics={}
    fast,capture=solve_collision(e,levels,widths,None,diagnostics,reduced=reduced,
        channel_factors=factors,return_absorption=True,absorption_rtol=roundoff)
    np.testing.assert_array_equal(fast,w)
    np.testing.assert_allclose(capture,2*np.sum(widths[None,:]*abs(x)**2,axis=1),rtol=2e-12,atol=0.)
    assert np.all(capture>=0)
    if gamma==.03:assert diagnostics['rm_gram_capture_energies']>0


def test_gram_capture_falls_back_for_cancellation_and_tighter_accuracy():
    e=np.array([100.,150.]);a=np.array([[.3,.4]])
    factors=np.ones((2,2));widths=np.array([1e-4])
    diagnostics={}
    w,x=solve_collision(e,[100.],widths,None,reduced=a,channel_factors=factors)
    _,capture=solve_collision(e,[100.],widths,None,diagnostics,reduced=a,
        channel_factors=factors,return_absorption=True,absorption_rtol=1e-11)
    assert diagnostics['rm_direct_capture_energies']>=1
    np.testing.assert_allclose(capture,2*np.sum(widths[None,:]*abs(x)**2,axis=1),rtol=2e-12)
    diagnostics={}
    solve_collision(e,[100.],widths,None,diagnostics,reduced=a,channel_factors=factors,
        return_absorption=True,absorption_rtol=1e-20)
    assert diagnostics['rm_direct_capture_energies']==len(e)

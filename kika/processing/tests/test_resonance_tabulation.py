"""R3 checks on independent probes, assembly ownership and lossy serialization."""
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from kika.nuclear_data.model import Axis,Axes,XYs1d,Regions1d,ConversionReport,Background,ResonancesWithBackground
from kika.nuclear_data.model.resonances import CompetitiveChannel,Resonance
from kika.processing.resonances import (NeutronContext,prepare_resonances,tabulate_resonances,
    ReconstructionOptions,ReconstructionConvergenceError,UnsupportedResonanceError)
from test_resonance_foundations import model


AXES=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')])


def prepared(m=None):
    m=model() if m is None else m
    for r in m.resolved:r.domainMin,r.domainMax=10.,300.
    return prepare_resonances(m,NeutronContext(56.,0.))


def constant(value):return XYs1d([10.,300.],[value,value],axes=AXES)


def assert_budget(result,points):
    for mt,value in result.evaluate(points).items():
        actual=result.forms[mt].evaluate(points)
        ratio=abs(actual-value)/(result.options.atol+result.options.rtol*np.maximum(abs(actual),abs(value)))
        assert np.max(ratio)<=1


def test_common_grid_final_background_and_independent_probes():
    p=prepared()
    bg={2:constant(1.),102:constant(.4),16:constant(.6)}
    result=tabulate_resonances(p,backgrounds=bg,sums={1:(2,18,102,16)})
    grid=result.forms[2].xs
    assert all(np.array_equal(f.xs,grid) for f in result.forms.values())
    probes=np.unique(np.r_[np.geomspace(10.,300.,1701),np.linspace(99.,101.,1013)])
    assert_budget(result,probes)
    values=result.evaluate(probes)
    np.testing.assert_allclose(values[1],sum(values[mt] for mt in (2,18,102,16)))
    np.testing.assert_allclose(values[2],p.evaluate(probes)[2]+1.)
    assert result.report['global_error_bound'] is False
    assert max(result.verify_forms(result.forms).values())<=1


def test_narrow_peak_is_seeded_without_absolute_energy_spacing_floor():
    m=model(levels=[Resonance(100.,.5,1e-9,4e-10,6e-10,0.)])
    result=tabulate_resonances(prepared(m),options=ReconstructionOptions(rtol=.01,atol=1e-6))
    assert np.min(np.diff(result.forms[102].xs))<1e-9
    assert_budget(result,100.+np.linspace(-2e-9,2e-9,401))


@pytest.mark.parametrize('law', ['flat','lin-lin','lin-log','log-lin','log-log'])
def test_background_laws_and_snapshots(law):
    bg=XYs1d([10.,300.],[1.,4.],interpolation=law,axes=AXES)
    m=model(levels=[])
    result=tabulate_resonances(prepared(m),backgrounds={16:bg})
    e=np.array([20.,50.,100.,200.])
    expected=bg.evaluate(e)
    np.testing.assert_allclose(result.evaluate(e)[16],expected)
    bg.ys[:]=99.
    np.testing.assert_allclose(result.evaluate(e)[16],expected)


def test_final_cancellation_budget_and_negative_values_are_retained():
    p=prepared()
    knots=np.r_[10.,np.linspace(99.,101.,31),300.]
    bg=XYs1d(knots,-p.evaluate(knots)[102],axes=AXES)
    result=tabulate_resonances(p,backgrounds={102:bg},options=ReconstructionOptions(rtol=.02,atol=1e-3))
    assert_budget(result,np.linspace(99.,101.,1127))
    assert result.report['minima'][102]<0


def test_competitive_mf3_ownership_prevents_double_counting():
    m=model()
    group=m.resolved[0].formalism.resonanceParameters.spinGroups[0]
    group.resonances[0].totalWidth=.7
    group.competitiveChannel=CompetitiveChannel(-20.,L=0,inEvaluatedBackground=True)
    p=prepared(m)
    bg={51:constant(.6),2:constant(.1),102:constant(.2)}
    result=tabulate_resonances(p,backgrounds=bg,sums={1:(2,18,102,51),4:(51,)})
    e=np.array([20.,25.,100.,200.])
    values=result.evaluate(e)
    np.testing.assert_allclose(values[51],.6,rtol=0,atol=1e-12)
    np.testing.assert_allclose(values[1],p.evaluate(e)[1]-p.evaluate(e)[51]+.9)
    np.testing.assert_allclose(values[4],values[51])
    with pytest.raises(ValueError,match='ownership'):
        tabulate_resonances(p)


@pytest.mark.parametrize('graph,message',[
    ({1:(4,),4:(1,)},'cycle'),
    ({1:(2,2)},'duplicate'),
    ({1:(999,)},'missing'),
    ({1:(4,102),4:(102,)},'double counts')])
def test_invalid_sum_graphs_reject_before_result(graph,message):
    with pytest.raises(ValueError,match=message):tabulate_resonances(prepared(),sums=graph)


@pytest.mark.parametrize('options,message',[
    (ReconstructionOptions(max_points=20),'max_points'),
    (ReconstructionOptions(rtol=1e-9,max_iterations=1),'max_iterations')])
def test_exhausted_limits_do_not_return_partial_results(options,message):
    with pytest.raises(ReconstructionConvergenceError,match=message):
        tabulate_resonances(prepared(),options=options)


def test_unsupported_or_missing_background_semantics_rejected():
    with pytest.raises(ValueError,match='cover'):
        tabulate_resonances(prepared(),backgrounds={2:XYs1d([20.,300.],[0.,0.],axes=AXES)})
    with pytest.raises(UnsupportedResonanceError,match='axes'):
        tabulate_resonances(prepared(),backgrounds={2:XYs1d([10.,300.],[0.,0.])})
    with pytest.raises(ValueError,match='rebuilt sum'):
        tabulate_resonances(prepared(),backgrounds={1:constant(0.)},sums={1:(2,18,102)})
    with pytest.raises(UnsupportedResonanceError,match='resolve'):
        tabulate_resonances(prepared(),backgrounds={2:ResonancesWithBackground(
            Background(resolvedRegion=constant(0.)),resonanceRegionHref='external.xml#/resonances')})
    with pytest.raises(TypeError,match='ReconstructionOptions'):
        tabulate_resonances(prepared(),options=0)


def test_background_wrapper_and_histogram_jump_keep_one_sided_values():
    bg=XYs1d([10.,80.,300.],[1.,4.,4.],interpolation='flat',axes=AXES)
    wrapper=ResonancesWithBackground(Background(resolvedRegion=bg))
    result=tabulate_resonances(prepared(),backgrounds={16:wrapper})
    form=result.forms[16]
    assert isinstance(form,Regions1d)
    np.testing.assert_allclose(form.evaluate([np.nextafter(80.,10.),80.,np.nextafter(80.,300.)]),[1.,4.,4.])
    x,y,_=form.toEndfRegions()
    assert list(y[x==80.])==[1.,4.]
    assert max(result.verify_forms(result.forms).values())<=1


def test_endf_duplicate_abscissae_preserve_both_background_limits():
    bg=XYs1d([10.,80.,80.,300.],[1.,1.,4.,4.],axes=AXES)
    result=tabulate_resonances(prepared(),backgrounds={16:bg})
    points=np.array([np.nextafter(80.,10.),80.,np.nextafter(80.,300.)])
    np.testing.assert_allclose(result.evaluate(points)[16],[1.,4.,4.])
    np.testing.assert_allclose(result.forms[16].evaluate(points),[1.,4.,4.])


def test_adjacent_resonance_regions_keep_jump_and_right_ownership():
    m=model()
    first=m.resolved[0]
    first.domainMin,first.domainMax=10.,150.
    second=deepcopy(first)
    second.domainMin,second.domainMax=150.,300.
    second.formalism.scatteringRadius=7.
    m.resolved.append(second)
    p=prepare_resonances(m,NeutronContext(56.,0.))
    result=tabulate_resonances(p)
    e=np.array([np.nextafter(150.,10.),150.,np.nextafter(150.,300.)])
    for mt,v in p.evaluate(e).items():
        np.testing.assert_allclose(result.evaluate(e)[mt],v,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result.forms[mt].evaluate(e),v,rtol=1e-3,atol=1e-8)


def test_gap_never_becomes_an_interpolation_panel():
    m=model()
    m.resolved[0].domainMin,m.resolved[0].domainMax=10.,140.
    second=deepcopy(m.resolved[0])
    second.domainMin,second.domainMax=160.,300.
    m.resolved.append(second)
    result=tabulate_resonances(prepare_resonances(m,NeutronContext(56.,0.)))
    with pytest.raises(ValueError,match='gaps'):result.evaluate(150.)
    with pytest.raises(ValueError,match='gap'):result.forms[2].toEndfRegions()
    with pytest.raises(ValueError,match='gaps'):result.forms[2].evaluate(150.,outOfRange='raise')


def test_gnds_write_reload_and_serialization_failure():
    from kika.gnds.encode import _function
    from kika.gnds.primitives import readAxes,readForm
    result=tabulate_resonances(prepared())
    reloaded={}
    for mt,form in result.forms.items():
        root=ET.Element('crossSection')
        report=ConversionReport()
        node=_function(root,form,report,'R3-test')
        assert report.isClean
        node=ET.fromstring(ET.tostring(node))
        reloaded[mt]=readForm(node,readAxes(node,None))
    assert max(result.verify_forms(reloaded).values())<=1
    corrupted=deepcopy(reloaded)
    corrupted[102].ys*=1.1
    with pytest.raises(ReconstructionConvergenceError,match='budget'):result.verify_forms(corrupted)
    changed_law=deepcopy(reloaded)
    changed_law[102].interpolation=type(changed_law[102].interpolation).loglog
    with pytest.raises(ReconstructionConvergenceError,match='interpolation law'):result.verify_forms(changed_law)


def test_endf_actual_decimal_write_and_parse_are_within_budget():
    from kika.endf.classes.mf3.mf3mt import MF3MT
    from kika.endf.model_adapter import decodeMF3MT,encodeMF3MT
    from kika.endf.parsers.parse_mf3 import parse_mf3_mt
    result=tabulate_resonances(prepared())
    reloaded={}
    for mt,form in result.forms.items():
        source=MF3MT(number=mt,_za=26056.,_awr=56.,_mat=2631,_qm=0.,_qi=0.,_lr=0,
                      _energies=[10.,300.],_cross_sections=[0.,0.],_interpolation=[(2,2)],_nr=1,_np=2)
        reaction,_=decodeMF3MT(source)
        reaction.crossSection['recon']=form
        output,report=encodeMF3MT(reaction,label='recon')
        assert report.isClean
        decoded,_=decodeMF3MT(parse_mf3_mt(str(output).splitlines(),mt))
        reloaded[mt]=decoded.crossSection['eval']
    assert max(result.verify_forms(reloaded).values())<=1


def test_endf_rounding_can_destroy_a_narrow_peak_and_is_rejected():
    from kika.endf.classes.mf3.mf3mt import MF3MT
    from kika.endf.model_adapter import decodeMF3MT
    from kika.endf.parsers.parse_mf3 import parse_mf3_mt
    m=model(levels=[Resonance(100.,.5,1e-9,4e-10,6e-10,0.)])
    result=tabulate_resonances(prepared(m),options=ReconstructionOptions(rtol=.01,atol=1e-6))
    reloaded={}
    for mt,form in result.forms.items():
        source=MF3MT(number=mt,_za=26056.,_awr=56.,_mat=2631,_qm=0.,_qi=0.,_lr=0,
                      _energies=form.xs.tolist(),_cross_sections=form.ys.tolist(),
                      _interpolation=[(len(form.xs),2)],_nr=1,_np=len(form.xs))
        decoded,_=decodeMF3MT(parse_mf3_mt(str(source).splitlines(),mt))
        reloaded[mt]=decoded.crossSection['eval']
    with pytest.raises(ReconstructionConvergenceError,match='budget'):result.verify_forms(reloaded)


def test_tabulated_integral_against_independent_adaptive_quadrature():
    from scipy.integrate import quad
    result=tabulate_resonances(prepared(),backgrounds={16:constant(.6)})
    estimate=result.report['regions'][0]['integrals'][16]
    assert estimate['dE']['reference_estimate']==pytest.approx(.6*290.,rel=1e-13)
    assert estimate['dE_over_E']['reference_estimate']==pytest.approx(.6*np.log(30.),rel=1e-10)
    # Adaptive quadrature locations are independent of the tabulator's probes.
    integral=quad(lambda e:float(result.evaluate(e)[102]),10.,300.,
                  points=[99.,99.9,100.,100.1,101.],epsabs=1e-6,epsrel=1e-9)[0]
    table=result.forms[102].integrate()
    assert abs(integral-table)<=result.options.rtol*abs(integral)+result.options.atol*290.

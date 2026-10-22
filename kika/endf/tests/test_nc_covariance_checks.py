"""Independent linear-algebra oracles for NC diagnostics, including failures."""
from types import SimpleNamespace

import numpy as np
import pytest

from kika.endf import check_covariances
from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf33.mf33 import MF33MT, Subsection, NCSubSubsection, NISubSubsectionRecord
from kika.endf.checks.covariances import _Context
from kika.endf.checks.nc import NCAssembler, NotEvaluable


def ni(matrix, grid=(1.,2.,3.), absolute=True):
    m=np.asarray(matrix,dtype=float)
    if absolute:
        # LB=0 is diagonal; LB=5 stores relative general matrices.
        assert np.array_equal(m,np.diag(np.diag(m)))
        return NISubSubsectionRecord(lb=0,np=len(grid),ne=len(grid),nt=2*len(grid),
            e_table_k=list(grid),f_table_k=list(np.diag(m))+[0.])
    return NISubSubsectionRecord(lb=5,ls=0,ne=len(grid),nt=len(grid)+m.size,
                                energies=list(grid),matrix=list(m.ravel()))


def nc(lo=1.,hi=3.,terms=((1.,2),)):
    return NCSubSubsection(lty=0,e1=lo,e2=hi,nci=len(terms),
                          ci=[c for c,m in terms],xmti=[m for c,m in terms])


def section(mt, records=(), rules=(), cross=(), mtl=0):
    s=MF33MT(number=mt,_mat=2631,_za=26056.,_awr=55.,_mtl=mtl)
    if not mtl:
        s._subsections=[Subsection(mat1=0,mt1=mt,nc=len(rules),ni=len(records),
                                  nc_records=list(rules),ni_records=list(records))]
        for other,rs in cross:
            s._subsections.append(Subsection(mat1=0,mt1=other,ni=len(rs),ni_records=list(rs)))
    s._nl=len(s._subsections)
    return s


def tape(*sections):
    e=ENDF();f=MF(number=33)
    for s in sections:f.add_section(s)
    e.add_file(f)
    return e


def engine(*sections,xs=None,mf=33):
    e=tape(*sections)
    ss={s.number:s for s in sections}
    good={(s.number,0,sub.mt1):list(enumerate(sub.ni_records))
          for s in sections for sub in s.subsections if sub.ni_records}
    return NCAssembler(_Context(e,xs),mf,ss,good)


def constant(value,lo=1.,hi=5.):
    return SimpleNamespace(energies=[lo,hi],values=[value,value])


def test_relative_covariances_use_cross_sections_and_reverse_cross_transpose():
    a=np.array([[.04,.01],[.01,.09]])
    b=np.array([[.16,.02],[.02,.25]])
    ba=np.array([[.005,.009],[-.003,.004]])
    one=section(1,rules=[nc(terms=((2.,2),(-1.,3)))])
    two=section(2,[ni(a,absolute=False)])
    three=section(3,[ni(b,absolute=False)],cross=[(2,[ni(ba,absolute=False)])])
    calc=engine(one,two,three,xs={2:constant(3.),3:constant(5.)})
    g=calc.grid(calc.dependencies(1))
    expected=4*9*a+25*b-2*15*(ba+ba.T)
    np.testing.assert_allclose(calc.block(1,1,g),expected,atol=1e-14)
    np.testing.assert_allclose(calc.block(1,2,g),2*9*a-15*ba,atol=1e-14)


def test_disjoint_rules_mask_both_axes_on_union_grid_and_keep_ni():
    # Explicit hand projection onto [1,2,3,4,5], independent of the assembler.
    a=np.array([[2.,.5],[.5,3.]])
    b=np.array([[5.,1.],[1.,7.]])
    one=section(1,[ni(np.diag([.1,.2,.3,.4]),grid=(1.,2.,3.,4.,5.))],
                [nc(1.,3.,((1.,2),)),nc(3.,5.,((-2.,3),))])
    two=section(2,[ni(a,grid=(1.,3.,5.),absolute=False)])
    three=section(3,[ni(b,grid=(1.,2.,5.),absolute=False)])
    calc=engine(one,two,three,xs={2:constant(1.),3:constant(1.)})
    g=calc.grid(calc.dependencies(1))
    aa=a[np.ix_([0,0,1,1],[0,0,1,1])]
    bb=b[np.ix_([0,1,1,1],[0,1,1,1])]
    ca=np.array([1.,1.,0.,0.]);cb=np.array([0.,0.,-2.,-2.])
    expected=ca[:,None]*aa*ca+cb[:,None]*bb*cb+np.diag([.1,.2,.3,.4])
    np.testing.assert_allclose(calc.block(1,1,g),expected)


def test_public_check_detects_negative_variance_without_clipping():
    # Each constituent self matrix is positive; excessive anticorrelation
    # makes their sum negative. The derived diagonal must remain negative.
    one=section(1,rules=[nc(terms=((1.,2),(1.,3)))])
    two=section(2,[ni(np.eye(2),absolute=False)],cross=[(3,[ni(-2*np.eye(2),absolute=False)])])
    three=section(3,[ni(np.eye(2),absolute=False)])
    r=check_covariances(tape(one,two,three),mf=(33,),xs_sections={2:constant(1.),3:constant(1.)})
    assert any(f.location.mt==1 for f in r.by_check('negative_variance'))
    assert any(f.location.mt==1 for f in r.by_check('nc_matrix_evaluated'))
    assert not r.by_check('nc_matrix_not_evaluated')


def test_derived_self_psd_and_correlations_checked_with_positive_diagonal():
    one=section(1,rules=[nc()])
    two=section(2,[ni([[1.,2.],[2.,1.]],absolute=False)])
    r=check_covariances(tape(one,two),mf=(33,),xs_sections={2:constant(1.)})
    assert any(f.location.mt==1 for f in r.by_check('not_positive_semidefinite'))
    assert any(f.location.mt==1 for f in r.by_check('correlation_out_of_bounds'))


@pytest.mark.parametrize('problem',['missing','cycle','overlap','coefficients','central','unsupported','member','invalid_ni'])
def test_incomplete_inputs_never_pass_as_complete(problem):
    one=section(1,rules=[nc()]);two=section(2,[ni(np.eye(2),absolute=False)])
    sections=[one,two];xs={2:constant(1.)}
    if problem=='missing':sections=[one]
    elif problem=='cycle':two=section(2,rules=[nc(terms=((1.,1),))]);sections=[one,two]
    elif problem=='overlap':one.subsections[0].nc_records.append(nc(2.,3.));one.subsections[0].nc=2
    elif problem=='coefficients':one.subsections[0].nc_records[0].ci=[]
    elif problem=='central':xs=None
    elif problem=='unsupported':two.subsections[0].nc_records=[NCSubSubsection(lty=1,e1=1.,e2=3.)];two.subsections[0].nc=1
    elif problem=='member':two._mtl=851;two._subsections=[];two._nl=0
    elif problem=='invalid_ni':two.subsections[0].ni_records[0].nt=1
    r=check_covariances(tape(*sections),mf=(33,),xs_sections=xs)
    assert any(f.location.mt==1 for f in r.by_check('nc_matrix_not_evaluated'))
    assert not any(f.location.mt==1 for f in r.by_check('nc_matrix_evaluated'))


def test_union_grid_above_the_previous_cutoff_is_accepted():
    grid=list(range(1,1203))
    record=NISubSubsectionRecord(lb=0,np=len(grid),ne=len(grid),nt=2*len(grid),
                                e_table_k=grid,f_table_k=[1.]*1201+[0.])
    calc=engine(section(1,rules=[nc(1.,1202.)]),section(2,[record]))
    np.testing.assert_array_equal(calc.grid(calc.dependencies(1)),grid)


def test_lump_central_values_sum_members_without_distributing_its_covariance():
    one=section(1,rules=[nc(terms=((1.,851),))])
    lump=section(851,[ni(.04*np.eye(2),absolute=False)])
    calc=engine(one,lump,section(52,mtl=851),section(53,mtl=851),
                xs={52:constant(3.),53:constant(7.)})
    g=calc.grid(calc.dependencies(1))
    np.testing.assert_allclose(calc.block(1,1,g),4*np.eye(2))


def test_mf31_uses_nubar_instead_of_cross_sections():
    one=section(452,rules=[nc(terms=((1.,455),))])
    two=section(455,[ni(.04*np.eye(2),absolute=False)])
    calc=engine(one,two,mf=31)
    calc.ctx.mf1={455:SimpleNamespace(lnu=2,energies=[1.,3.],nubar_values=[.2,.2],interpolation=[(2,2)])}
    g=calc.grid(calc.dependencies(452))
    np.testing.assert_allclose(calc.block(452,452,g),.0016*np.eye(2))


def test_zero_covariance_is_evaluated_and_not_an_eigensolver_failure():
    r=check_covariances(tape(section(1,rules=[nc()]),section(2,[ni(np.zeros((2,2)))])),mf=(33,))
    assert r.by_check('nc_matrix_evaluated')
    assert not r.by_check('nc_matrix_not_evaluated')


def test_nonoverlapping_mutual_rules_do_not_create_a_false_cycle():
    one=section(1,[ni(np.diag([0.,3.]))],[nc(1.,2.,((1.,2),))])
    two=section(2,[ni(np.diag([2.,0.]))],[nc(2.,3.,((1.,1),))])
    calc=engine(one,two)
    grid=calc.grid(calc.dependencies(1))
    np.testing.assert_array_equal(calc.block(1,1,grid),np.diag([2.,3.]))
    np.testing.assert_array_equal(calc.block(1,2,grid),np.diag([2.,3.]))


def test_joint_includes_implied_covariances_with_nonconstituents():
    one=section(1,rules=[nc()])
    two=section(2,[ni(np.eye(2),absolute=False)],cross=[
        (3,[ni(-.75*np.eye(2),absolute=False)]),(4,[ni(-.75*np.eye(2),absolute=False)])])
    three=section(3,[ni(np.eye(2),absolute=False)],cross=[(4,[ni(-.75*np.eye(2),absolute=False)])])
    four=section(4,[ni(np.eye(2),absolute=False)])
    xs={m:constant(1.) for m in (2,3,4)}
    calc=engine(one,two,three,four,xs=xs)
    assert calc.connected(1)=={1,2,3,4}
    grid=calc.grid(calc.connected(1))
    np.testing.assert_allclose(calc.block(1,3,grid),-.75*np.eye(2))
    r=check_covariances(tape(one,two,three,four),mf=(33,),xs_sections=xs)
    assert any(f.location.mt==1 for f in r.by_check('joint_not_positive_semidefinite'))
    assert r.by_check('nc_matrix_evaluated')


def test_mixed_absolute_and_relative_components_are_added_in_absolute_units():
    one=section(1,[ni(np.eye(2))],[nc()])
    two=section(2,[ni(2*np.eye(2)),ni(.04*np.eye(2),absolute=False)])
    calc=engine(one,two,xs={2:constant(10.)})
    grid=calc.grid(calc.dependencies(1))
    np.testing.assert_allclose(calc.block(1,1,grid),7*np.eye(2))


@pytest.mark.parametrize('field,value',[('xmti',[float('nan')]),('xmti',[2.5]),
    ('ci',[float('inf')]),('e1',None),('e2',float('inf'))])
def test_invalid_nc_values_are_findings_not_exceptions(field,value):
    rule=nc();setattr(rule,field,value)
    r=check_covariances(tape(section(1,rules=[rule]),section(2,[ni(np.eye(2))])),mf=(33,))
    assert r.by_check('nc_invalid_values')
    assert r.by_check('nc_matrix_not_evaluated')
    assert not r.by_check('nc_matrix_evaluated')


def test_resonance_background_is_never_used_as_reconstructed_cross_section():
    calc=engine(section(1,rules=[nc()]),section(2,[ni(np.eye(2),absolute=False)]))
    calc.ctx.eh=2.
    calc.ctx.mf3={2:constant(10.)}
    grid=calc.grid(calc.dependencies(1))
    with pytest.raises(NotEvaluable,match='reconstructed cross sections'):
        calc.block(1,1,grid)
    # When only above-resonance rows contribute, MF3 is trustworthy there.
    np.testing.assert_allclose(calc.central(2,grid,np.array([False,True])),[10.,10.])


def test_polynomial_nubar_is_averaged_analytically():
    calc=engine(section(452,rules=[nc(terms=((1.,455),))]),
                section(455,[ni(.04*np.eye(2),absolute=False)]),mf=31)
    calc.ctx.mf1={455:SimpleNamespace(lnu=1,coefficients=[2.,3.,1.])}
    grid=np.array([1.,2.,3.])
    mean=np.array([2.+4.5/np.log(2.),2.+5.5/np.log(1.5)])
    np.testing.assert_allclose(calc.block(452,452,grid),np.diag(.04*mean**2))


def test_shared_joint_is_checked_once_and_referenced_by_each_derived_mt(monkeypatch):
    from kika.endf.checks import covariances as module
    original=module._joint_psd
    calls=[]
    def record(*args,**kwargs):
        calls.append(args[0])
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'_joint_psd',record)
    r=check_covariances(tape(section(1,rules=[nc()]),section(2,[ni(np.eye(2))]),
                            section(3,rules=[nc()])),mf=(33,))
    assert calls==[[1,2,3]]
    coverage=r.by_check('nc_matrix_evaluated')
    assert len(coverage)==2
    assert all(f.evidence['joint_evaluated'] and f.evidence['joint_check_mt']==1 for f in coverage)
    assert not r.by_check('nc_matrix_not_evaluated')
    assert not r.by_check('psd_not_evaluated')

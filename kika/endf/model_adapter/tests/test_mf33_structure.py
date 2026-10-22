"""Declared MF33 groups and sums survive conversion without member covariances."""
from types import SimpleNamespace
import numpy as np
import pytest
from kika.endf.classes.mf33.mf33 import MF33MT, Subsection, NCSubSubsection, NISubSubsectionRecord
from kika.endf.model_adapter.covariances import decodeMF33MT, encodeMF33MT, decodeCovarianceSuite
from kika.nuclear_data.model import CovarianceSuite, Sum, Mixed


def section(mt, *, mtl=0, nc=(), ni=()):
    subs=[] if mtl else [Subsection(mat1=0,mt1=mt,nc=len(nc),ni=len(ni),
                                   nc_records=list(nc),ni_records=list(ni))]
    return MF33MT(number=mt,_za=92235.,_awr=233.,_mat=9228,_mtl=mtl,
                  _nl=len(subs),_subsections=subs)


def rule(lo,hi,coefficients=(1.,-1.),mts=(2,102)):
    return NCSubSubsection(lty=0,e1=lo,e2=hi,nci=len(mts),ci=list(coefficients),xmti=list(mts))


def test_disjoint_nc_rules_and_explicit_matrix_survive_endf():
    ni=NISubSubsectionRecord(lb=5,ls=1,ne=2,nt=3,energies=[1.,4.],matrix=[.04])
    original=section(1,nc=[rule(1.,2.),rule(2.,4.,(2.,),(18,))],ni=[ni])
    decoded,report=decodeMF33MT(original)
    assert not report.losses
    assert len(decoded)==1 and isinstance(decoded[0].form,Mixed)
    assert [c.label for c in decoded[0].form.components]==["0","1","2"]
    sums=[c for c in decoded[0].form.components if isinstance(c,Sum)]
    assert [(s.domainMin,s.domainMax,s.coefficients) for s in sums]==[(1.,2.,[1.,-1.]),(2.,4.,[2.])]
    written,_=encodeMF33MT(CovarianceSuite(covarianceSections=decoded),1)
    sub=written.subsections[0]
    assert [(n.e1,n.e2,n.ci,n.xmti) for n in sub.nc_records]==[(1.,2.,[1.,-1.],[2.,102.]),(2.,4.,[2.],[18.])]
    assert len(sub.ni_records)==1
    again,_=decodeMF33MT(written)
    np.testing.assert_array_equal(again[0].form.components[0].matrix,[[.04]])


def test_nc_bad_parallel_arrays_are_not_silently_truncated():
    bad=rule(1.,2.);bad.ci=[1.]
    with pytest.raises(ValueError,match="unequal"):
        decodeMF33MT(section(1,nc=[bad]))


def test_repeated_block_keeps_every_rule_with_unique_component_labels():
    original=section(1,nc=[rule(1.,2.)])
    original.subsections.append(section(1,nc=[rule(2.,4.)]).subsections[0])
    original._nl=2
    decoded,report=decodeMF33MT(original)
    assert not report.losses and len(decoded)==1
    assert [c.label for c in decoded[0].form.components]==["0","1"]
    written,_=encodeMF33MT(CovarianceSuite(covarianceSections=decoded),1)
    assert [(n.e1,n.e2) for n in written.subsections[0].nc_records]==[(1.,2.),(2.,4.)]


def test_unsupported_nc_and_external_blocks_report_losses():
    unsupported=rule(1.,4.);unsupported.lty=1
    decoded,report=decodeMF33MT(section(1,nc=[unsupported]))
    assert not decoded and any("LTY=1" in loss for loss in report.losses)
    external=section(1,nc=[rule(1.,4.)])
    external.subsections[0].mat1=125
    decoded,report=decodeMF33MT(external)
    assert not decoded and any("external MAT1=125" in loss for loss in report.losses)


def test_missing_nc_target_stays_unresolved_and_is_reported():
    endf=SimpleNamespace(mf={33:SimpleNamespace(mt={1:section(1,nc=[rule(1.,4.)])})})
    cov,report=decodeCovarianceSuite(endf,checks=False)
    assert isinstance(cov.byLabel("MF33-MT1").form,Sum)
    assert len([w for w in report.warnings if "unresolved" in w])==2


@pytest.mark.parametrize("law",[2,5])
def test_groups_are_native_reaction_sums_and_not_individual_covariances(law):
    from kika.endf.writers.assemble import _covarianceSections
    from kika.nuclear_data.model import ConversionReport
    from kika.nuclear_data.model import ReactionSuite,Reaction,ReactionId
    explicit=NISubSubsectionRecord(lb=5,ls=1,ne=2,nt=3,energies=[1.,4.],matrix=[.04])
    members={52:section(52,mtl=851),53:section(53,mtl=851),851:section(851,ni=[explicit])}
    tables={mt:SimpleNamespace(energies=[1.,4.],cross_sections=[float(mt),float(mt)],
                              energy_interpolation=[(2,law)]) for mt in [52,53]}
    endf=SimpleNamespace(mf={33:SimpleNamespace(mt=members),3:SimpleNamespace(mt=tables)})
    suite=ReactionSuite(evaluation="test",projectile="n",target="U235")
    for mt in tables:
        suite.reactions.append(Reaction(id=ReactionId(label=f"MT{mt}",ENDF_MT=mt)))
    from kika.nuclear_data.model import EndfProvenance
    suite.provenance=EndfProvenance(za=92235.,awr=233.,mat=9228)
    cov,report=decodeCovarianceSuite(endf,checks=False,reactionSuite=suite)
    suite.covarianceSuite=cov
    assert not report.losses
    assert bool(report.approximations)==(law==5)
    assert [s.rowData.ENDF_MT for s in cov]==[851]
    lump=suite.sums[851]
    assert len(lump.summands)==2
    np.testing.assert_array_equal(lump.crossSection["eval"].ys,[105.,105.])
    written,_=_covarianceSections(suite,9228,ConversionReport())
    actual={mt:s for mf,mt,s in written if mf==33}
    assert set(actual)=={52,53,851}
    assert actual[52]._mtl==actual[53]._mtl==851
    assert len(actual[52].subsections)==0
    decodeCovarianceSuite(endf,checks=False,reactionSuite=suite)
    assert len(suite.sums)==1  # decoding again must not duplicate the group


def test_u235_keeps_all_group_headers_and_nc_rules(u235_b81_tape,tmp_path):
    import kika
    from kika.endf import read_endf
    from kika.endf.model_adapter.decode import decodeReactionSuite
    from kika.endf.writers.assemble import _covarianceSections
    from kika.nuclear_data.model import ConversionReport
    endf=read_endf(str(u235_b81_tape),mf_numbers=[1,3,33])
    suite,_=decodeReactionSuite(endf)
    suite.covarianceSuite,report=decodeCovarianceSuite(endf,checks=False,reactionSuite=suite)
    assert not [x for x in report.losses if "MF33" in x]
    rows,_=_covarianceSections(suite,9228,ConversionReport())
    from kika.endf.parsers.parse_mf33 import parse_mf33_mt
    actual={mt:parse_mf33_mt(str(s).splitlines(),mt) for mf,mt,s in rows if mf==33}
    assert set(actual)==set(endf.mf[33].mt) and len(actual)==87
    for mt,source in endf.mf[33].mt.items():
        assert (actual[mt]._mtl or 0)==(source._mtl or 0)
        before=[(n.e1,n.e2,n.ci,n.xmti) for sub in source.subsections for n in sub.nc_records]
        after=[(n.e1,n.e2,n.ci,n.xmti) for sub in actual[mt].subsections for n in sub.nc_records]
        assert after==before
    assert suite.covarianceSuite.byLabel("MF33-MT1").rowData.href.startswith("/reactionSuite/sums/")
    public=kika.read(u235_b81_tape)
    without=kika.read(u235_b81_tape,covariances=False)
    for reaction in without.reactions:
        for label,form in reaction.crossSection.items():
            actual_form=public.reactions[reaction.ENDF_MT].crossSection[label]
            before=getattr(form,"function1ds",[form])
            after=getattr(actual_form,"function1ds",[actual_form])
            assert len(after)==len(before)
            for a,b in zip(after,before):
                np.testing.assert_array_equal(a.xs,b.xs)
                np.testing.assert_array_equal(a.ys,b.ys)
    assert len(public.sums[851].summands)==39
    assert public.covarianceSuite.byLabel("MF33-MT1").form.summands
    assert any(f.check=="nc_matrix_not_evaluated" for f in public.covarianceSuite.covarianceChecks)
    from kika.gnds.encode import writeCovarianceSuite
    from kika.gnds.xpath import Document,Resolver
    xml,_=writeCovarianceSuite(public.covarianceSuite,"2.1")
    resolver=Resolver(Document(xml.getroot()))
    for term in xml.getroot().iter("summand"):
        result=resolver.resolve(term.attrib["href"])
        assert result, result.problem
        assert result.element.tag in {"mixed","covarianceMatrix","sum"}
    path=tmp_path/"u235.xml"
    kika.write(suite,path)
    reloaded=kika.read(path)
    from kika.endf.writers.assemble import encodeTapeSections
    rows=encodeTapeSections(reloaded,mat=9228,files=(33,))[0]
    after={mt:s for mf,mt,s in rows if mf==33}
    assert set(after)==set(actual)
    for mt in actual:
        assert (after[mt]._mtl or 0)==(actual[mt]._mtl or 0)
        nc=lambda s:[(n.e1,n.e2,n.ci,n.xmti) for sub in s.subsections for n in sub.nc_records]
        assert nc(after[mt])==nc(actual[mt])


def test_checks_distinguish_structural_validation_from_derived_matrix_coverage():
    from kika.endf import check_covariances
    from kika.endf.classes.endf import ENDF
    from kika.endf.classes.mf import MF
    ni=NISubSubsectionRecord(lb=5,ls=1,ne=2,nt=3,energies=[1.,4.],matrix=[.04])
    endf=ENDF();mf=MF(number=33)
    for s in [section(2,ni=[ni]),section(1,nc=[rule(1.,4.,(1.,),(2,))]),
              section(52,mtl=851),section(851,ni=[ni])]:
        mf.add_section(s)
    endf.add_file(mf)
    report=check_covariances(endf,mf=(33,))
    note=report.by_check("nc_matrix_not_evaluated")
    assert len(note)==1 and note[0].location.mt==1 and note[0].level=="note"
    assert not report.by_check("unresolved_reference")
    # A missing group remains a defect; preserving links cannot make it disappear.
    del mf.mt[851]
    report=check_covariances(endf,mf=(33,))
    assert any(f.location.mt==52 for f in report.by_check("unresolved_reference"))

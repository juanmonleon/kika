"""Whole-tape publication verifies representation before replacing a file."""
from copy import deepcopy
from pathlib import Path
import pytest
from kika.endf import read_endf
from kika.endf.model_adapter import decodeReactionSuite, decodeMF2MT151
from kika.endf.classes.mf2.mf2mt151 import (
    MF2MT151,Isotope,EnergyRange,ResolvedResonanceRange,LValueBlock,Resonance)
from kika.endf.writers.assemble import writeReconstructedEndfTape
from kika.processing.resonances import attach_reconstruction, ReconstructionConvergenceError
from test_resonance_suite import suite_model,run


def writable_suite(l=0):
    suite=suite_model()
    source,_=decodeReactionSuite(read_endf(str(Path(__file__).resolve().parents[2]/
        'endf/tests/data/micro_fe56_structural.endf'),mf_numbers=[1,3]))
    suite.provenance=deepcopy(source.provenance)
    for reaction in list(suite.reactions)+list(suite.sums):
        original=source.reactionByENDF_MT(reaction.ENDF_MT if reaction.ENDF_MT!=16 else 2)
        reaction.provenance=deepcopy(original.provenance)
        reaction.outputChannel.Q=deepcopy(original.outputChannel.Q)
    section=MF2MT151(number=151)
    section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
    parameters=ResolvedResonanceRange(0.,.5,0,0,[LValueBlock(56.,l,1,
        [Resonance(100.,l+.5,.3,.1,.2,0.)],0.,0)])
    section._isotopes=[Isotope(26056.,1.,0,1,[EnergyRange(10.,300.,1,2,0,1,parameters)])]
    suite.resonances,provenance,report=decodeMF2MT151(section)
    suite.resonances.provenance=provenance
    suite.report=report
    return suite


def test_successful_publication_is_a_complete_processed_tape(tmp_path):
    suite=writable_suite();result=run(suite);attach_reconstruction(suite,result)
    target=tmp_path/'processed.endf'
    target.write_text('previous file')
    report=writeReconstructedEndfTape(suite,result,target)
    assert report.isClean
    reloaded,report=decodeReactionSuite(read_endf(str(target)))
    assert report.isClean
    assert reloaded.provenance.headerFields['lrp']==2
    assert max(result.verify_suite(reloaded,label='eval').values())<=1
    assert not list(tmp_path.glob('.processed.endf.*'))


def test_unrepresentable_peak_does_not_replace_existing_file(tmp_path):
    suite=writable_suite(l=3);result=run(suite);attach_reconstruction(suite,result)
    target=tmp_path/'processed.endf';target.write_text('previous file')
    with pytest.raises(ReconstructionConvergenceError):
        writeReconstructedEndfTape(suite,result,target)
    assert target.read_text()=='previous file'
    assert not list(tmp_path.glob('.processed.endf.*'))


def test_source_edited_after_attachment_cannot_be_published(tmp_path):
    suite=writable_suite();result=run(suite);attach_reconstruction(suite,result)
    suite.resonances.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0].captureWidth=.25
    target=tmp_path/'processed.endf'
    with pytest.raises(ValueError,match='source changed'):
        writeReconstructedEndfTape(suite,result,target)
    assert not target.exists()

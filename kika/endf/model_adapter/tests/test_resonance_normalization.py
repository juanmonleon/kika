"""Physical model metadata and editable round trips, independent of kernels."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from kika.endf.classes.mf2.mf2mt151 import (
    MF2MT151,Isotope,EnergyRange,RMatrixLimited,RML_ParticlePair,RML_Channel,
    RML_Resonance,RML_SpinGroup,NoBackgroundRMatrix,TabulatedBackgroundRMatrix,
    SammyBackgroundRMatrix,FrohnerBackgroundRMatrix,TabulatedPhaseShift,
    UnresolvedCaseC,URR_LValue_CaseC,URR_JState_CaseC,URR_EnergyPoint)
from kika.endf.model_adapter import decodeMF2MT151,encodeMF2MT151


def section(parameters,lru=1,lrf=7):
    s=MF2MT151(number=151)
    s._za,s._awr,s._mat,s._nis=26056,56.,2631,1
    s._isotopes=[Isotope(26056,1.,0,1,[EnergyRange(1.,1000.,lru,lrf,0,1,parameters)])]
    return s


@pytest.mark.parametrize('background',[
    NoBackgroundRMatrix(1),
    TabulatedBackgroundRMatrix(1,[(2,2)],[1.,1000.],[.1,.3],[(2,5)],[1.,1000.],[.01,.03]),
    SammyBackgroundRMatrix(1,-100.,2000.,.1,.002,.00003,.4,.005),
    FrohnerBackgroundRMatrix(1,-100.,2000.,.1,.4,3.)])
def test_rml_physical_metadata_and_extra_functions_roundtrip(background):
    phase=TabulatedPhaseShift([(2,2)],[1.,1000.],[.1,.2],[(2,2)],[1.,1000.],[.01,.02])
    parameters=RMatrixLimited(1,3,0,.5,.5,
        [RML_ParticlePair(1.,56.,0.,26.,.5,-.5,0.,1,1,2,0.,0.),
         RML_ParticlePair(0.,0.,0.,0.,0.,0.,1e6,-1,0,102,0.,0.)],
        [RML_SpinGroup(-.5,1.,1,1,[RML_Channel(1,0,1.,0.,.5,.6),
          RML_Channel(2,0,0.,0.,0.,0.)],[RML_Resonance(100.,[.1,-.2])],[background],phase)])
    s=section(parameters)
    m,p,report=decodeMF2MT151(s)
    assert report.isClean
    f=m.resolved[0].formalism
    assert f.reducedWidthAmplitudes
    assert f.spinGroups[0].spin == .5 and f.spinGroups[0].parity == -1
    neutron,capture=f.resonanceReactions
    assert neutron.reactionMT == 2
    assert neutron.kinematics.particleB.spin == .5
    assert neutron.kinematics.particleB.parity == -1
    assert neutron.kinematics.particleB.massRatio == 56.
    assert neutron.kinematics.penetrability == 'calculate'
    assert neutron.kinematics.shift == 'calculate'
    assert capture.eliminated and capture.kinematics.effective
    assert capture.kinematics.penetrability == 'unity'
    assert f.spinGroups[0].widths == [[.1,-.2]]  # IFG amplitudes/sign retained
    np.testing.assert_allclose(f.spinGroups[0].additionalPhaseShift.real.evaluate([1.,1000.]),[.1,.2])
    original=copy.deepcopy(p.headerFields)
    encoded=encodeMF2MT151(m,p)
    assert str(encoded) == str(s)
    assert p.headerFields == original
    neutron.kinematics=replace(neutron.kinematics,particleB=replace(neutron.kinematics.particleB,massRatio=55.))
    assert encodeMF2MT151(m,p).isotopes[0].energy_ranges[0].parameters.particle_pairs[0].mb == 55.


def test_urr_case_c_independent_grids_and_cross_section_interpolation():
    def state(j,grid,law):
        return URR_JState_CaseC(j,law,1.,1.,0.,0.,
            [URR_EnergyPoint(e,2.,.1,.2,.3,.4) for e in grid])
    parameters=UnresolvedCaseC(.5,.5,0,1,[URR_LValue_CaseC(56.,0,[state(0.,[1.,1000.],2),state(1.,[1.,10.,1000.],5)])])
    s=section(parameters,2,2)
    m,p,report=decodeMF2MT151(s)
    assert report.isClean
    widths=m.unresolved.tabulatedWidths
    assert widths.energyGrid is None
    assert [len(g.levelSpacingEnergies) for g in widths.spinGroups] == [2,3]
    assert str(widths.spinGroups[1].crossSectionInterpolation) == 'log-log'
    assert str(encodeMF2MT151(m,p)) == str(s)
    widths.spinGroups[1].levelSpacingEnergies[1]=20.
    for channel in widths.spinGroups[1].channels:
        channel.energies[1]=20.
    result=encodeMF2MT151(m,p)
    assert result.isotopes[0].energy_ranges[0].parameters.l_values[0].j_states[1].energy_points[1].es == 20.


def test_multiple_urr_ranges_are_explicitly_rejected_in_report():
    parameters=UnresolvedCaseC(.5,.5,0,0,[])
    s=section(parameters,2,2)
    s._isotopes[0].energy_ranges.append(EnergyRange(1000.,2000.,2,2,0,1,parameters))
    _,p,report=decodeMF2MT151(s)
    assert not report.isClean
    assert p.headerFields['regions'][1]['kind'] == 'unsupported'

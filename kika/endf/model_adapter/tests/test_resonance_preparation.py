"""ENDF boundary normalization checked against an independently built model."""
import numpy as np
import pytest

from kika.endf.classes.mf2.mf2mt151 import (
    MF2MT151, Isotope, EnergyRange, ResolvedResonanceRange, LValueBlock,
    Resonance as EndfResonance,
    EnergyDependentScatteringRadius,
)
from kika.endf.model_adapter import decodeMF2MT151
from kika.nuclear_data.model.resonances import (
    BreitWigner, BreitWignerApproximation, Resonance, ResonanceParameters,
    Resonances, ResolvedRegion, SpinGroup,
)
from kika.processing.resonances import NeutronContext, prepare_resonances


@pytest.mark.parametrize('nro,naps', [(0,0),(0,1),(1,0),(1,1),(1,2)])
def test_all_endf_radius_policies(nro,naps):
    section = MF2MT151(number=151)
    section._za,section._awr,section._mat,section._nis = 26056,56.,2631,1
    table = EnergyDependentScatteringRadius([(2,2)],[1.,1000.],[.3,.7]) if nro else None
    section._isotopes = [Isotope(26056,1.,0,1,[EnergyRange(1.,1000.,1,2,nro,naps,
        ResolvedResonanceRange(0.,.5,1,1,[LValueBlock(56.,1,1,[EndfResonance(100.,1.5,.3,.1,.2,0.)])]),table)])]
    decoded,provenance,report = decodeMF2MT151(section)
    decoded.provenance = provenance
    ctx = NeutronContext(56.,0.)
    g = prepare_resonances(decoded,ctx,conversion_report=report).regions[0].groups[0]
    e = np.array([1.,100.,1000.])
    phase = np.array([3.,3.+99/999*4,7.]) if nro else np.full(3,5.)
    channel = np.full(3,ctx.mass_channel_radius_fm) if naps == 0 else np.full(3,5.) if naps == 2 else phase
    np.testing.assert_allclose(g.phase_radius.evaluate(e),phase)
    np.testing.assert_allclose(g.channel_radius.evaluate(e),channel)


@pytest.mark.parametrize("lrf", [1, 2])
@pytest.mark.parametrize("naps", [0, 1])
@pytest.mark.parametrize("l", [0, 1])
def test_endf_and_manual_model_have_identical_bw_physics(lrf, naps, l):
    section = MF2MT151(number=151)
    section._za, section._awr, section._mat, section._nis = 26056, 56., 2631, 1
    section._isotopes = [Isotope(26056, 1., 0, 1, [EnergyRange(
        1., 1000., 1, lrf, 0, naps,
        ResolvedResonanceRange(0., .5, 1, 1, [LValueBlock(
            56., l, 1, [EndfResonance(100., l+.5, .3, .1, .2, 0.)])]))])]
    decoded, provenance, report = decodeMF2MT151(section)
    decoded.provenance = provenance
    manual = Resonances(resolved=[ResolvedRegion(1., 1000., formalism=BreitWigner(
        approximation=(BreitWignerApproximation.singleLevel if lrf == 1
                       else BreitWignerApproximation.multiLevel),
        resonanceParameters=ResonanceParameters([SpinGroup(
            l, [Resonance(100., l+.5, .3, .1, .2)], atomicWeightRatio=56.)]),
        scatteringRadius=5., radiusUnit="fm", calculateChannelRadius=(naps == 0)))])
    ctx = NeutronContext(56., 0.)
    left, right = prepare_resonances(decoded, ctx, conversion_report=report), prepare_resonances(manual, ctx)
    assert left.regions == right.regions
    energy = np.array([1., 25., 99.9, 100., 100.1, 400., 1000.])
    for mt, values in left.evaluate(energy).items():
        np.testing.assert_array_equal(values, right.evaluate(energy)[mt])
    with pytest.raises(ValueError, match="spin disagrees"):
        prepare_resonances(decoded, NeutronContext(56., .5))

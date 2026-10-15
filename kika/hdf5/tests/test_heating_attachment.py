"""Energy bookkeeping across exclusive fission channels and output temperatures."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from kika.nuclear_data import model as m
from kika.processing import heating as H
from kika.hdf5.processing import _attachHeating,processSuite
from kika.hdf5.names import heatedLabel


def polynomial(value):
    return m.Polynomial1d(coefficients=np.array([value]),domainMin_=0.,domainMax_=1e8)


@pytest.mark.parametrize('carried',[None,9.])
def test_attachment_uses_exclusive_fission_cross_sections_once_at_each_temperature(monkeypatch,carried):
    # Fission's MT301 is HEATR's EFR + EGP less the prompt photons the tables
    # carry for fission (none: all of EGP stays, as Pu-238 of ENDF/B-VIII.1),
    # plus OpenMC's local delayed betas; MT901 and MT442 do not depend on them.
    released=100.+7.+11.-(carried or 0.)
    grid=np.array([1.,2.]); temperatures=[293.6,600.]
    labels=['293K','600K']; grids={name:grid for name in labels}
    def rx(mt,value):
        cs=m.CrossSection()
        for name in labels:
            label=heatedLabel(name)
            cs[label]=m.XYs1d(xs=grid,ys=np.full(2,value),label=label)
        return m.Reaction(id=m.ReactionId(label=str(mt),ENDF_MT=mt),crossSection=cs,
                          outputChannel=m.OutputChannel(Q=m.Q(value=0.)))
    # Total fission is a sum; its exclusive channels are the only channels
    # that may contribute to heating and damage.
    elastic=rx(2,2.); first=rx(19,3.); second=rx(20,5.); total=rx(18,8.)
    release=m.FissionEnergyRelease(label='MonteCarlo',promptProductKE=polynomial(100.),
        delayedBetaEnergy=polynomial(7.),promptGammaEnergy=polynomial(11.),delayedGammaEnergy=polynomial(13.))
    total.outputChannel.fissionFragmentData=m.FissionFragmentData(fissionEnergyReleases=[release])
    out=NS(reactions=[elastic],fissionComponents=[first,second],sums=[total],productions=[])
    work=NS(reactions=[],sums=[],fissionComponents=[])
    def calculation(suite,E,sigma,**kwargs):
        assert np.array_equal(sigma[19],np.full(2,3.))
        assert np.array_equal(sigma[20],np.full(2,5.))
        assert kwargs['qa']=={2:1.}
        return H.HeatingResult(E,np.full(2,4.),np.full(2,4.),np.full(2,38.),
            perReaction={2:(np.full(2,2.),np.full(2,2.),np.full(2,3.)),
                         19:(np.zeros(2),np.zeros(2),np.full(2,4.)),
                         20:(np.zeros(2),np.zeros(2),np.full(2,4.))})
    monkeypatch.setattr(H,'heatingOf',calculation)
    monkeypatch.setattr(H,'fissionPhotonEnergy',
                        lambda suite,E,awr: {} if carried is None else {18:np.full(len(E),carried)})
    _attachHeating(out,work,temperatures,dict.fromkeys(labels,0.),grids,['kerma','damage'],
        235.,92,235,None,m.ConversionReport(),heatingPartials=[302,318,442,445,901],qa={2:1.})
    response={r.ENDF_MT:r for r in out.productions}
    for name in labels:
        label=heatedLabel(name)
        assert np.array_equal(response[301].crossSection[label].ys,np.full(2,4.+8.*released))
        assert np.array_equal(response[901].crossSection[label].ys,np.full(2,4.+8.*131.))
        assert np.array_equal(response[318].crossSection[label].ys,np.full(2,8.*released))
        assert np.array_equal(response[442].crossSection[label].ys,np.full(2,8.*24.))
        assert np.array_equal(response[302].crossSection[label].ys,np.full(2,4.))
        assert np.array_equal(response[445].crossSection[label].ys,np.full(2,6.))
        assert np.array_equal(response[444].crossSection[label].ys,np.full(2,38.))
    assert response[445].provenance.attributes['center_of_mass']==0


@pytest.mark.parametrize('ids',[[302,302],[True],[302.0]])
def test_invalid_request_is_rejected_before_processing(ids):
    with pytest.raises(ValueError,match='distinct integer'):
        processSuite(None,heatingPartials=ids)


def test_undetermined_active_recoil_is_rejected_by_kinematic_response():
    grid=np.array([1.,2.])
    result=H.HeatingResult(grid,np.zeros(2),np.zeros(2),np.zeros(2),
                         perReaction={91:(grid,grid,grid)})
    with pytest.raises(NotImplementedError,match='recoil energy for MT91'):
        result.partialsOf({91:np.ones(2)},[443])


@pytest.mark.parametrize('mt',[302,303,304,318,351,401,402,442,443,445,446,447])
def test_extra_heating_read_from_hdf5_remains_a_production_response(mt):
    from kika.hdf5.neutron_decode import _place
    suite=NS(reactions=[],sums=[],fissionComponents=[],productions=[])
    _place(suite,[(mt,1,(m.ReactionId(label=f'heating-{mt}',ENDF_MT=mt),
                        m.CrossSection(),m.OutputChannel(Q=m.Q(value=0.)),
                        m.Hdf5Provenance(attributes={'redundant':1})))])
    assert len(suite.productions)==1 and suite.productions[0].ENDF_MT==mt
    assert suite.sums==[] and suite.reactions==[]

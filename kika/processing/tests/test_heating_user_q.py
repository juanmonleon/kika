"""Auxiliary-Q contracts against independent two-body energy balance."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from kika.nuclear_data import model as m
from kika.processing import heating as H


def suite(mt=51, Q=-1.):
    level = m.Uncorrelated(angular=m.Isotropic2d(),
        energy=m.LevelEnergy(threshold=1.5, massRatio=4/9),productFrame=m.Frame.centerOfMass)
    neutron = NS(pid='n',distribution=NS(forms={'MonteCarlo':level}),
                 multiplicity=m.Multiplicity(form=m.Polynomial1d(coefficients=np.array([1.]),domainMin_=0.,domainMax_=100.)))
    rx = NS(ENDF_MT=mt, outputChannel=NS(Q=NS(value=Q),products=[neutron]))
    return NS(reactions=[rx])


def test_scalar_qa_changes_balance_but_not_outgoing_kinematics_or_damage():
    E = np.array([10.,20.]); xs={51:np.array([2.,3.])}
    baseline=H.heatingOf(suite(),E,xs,awr=2.,Z=1)
    changed=H.heatingOf(suite(),E,xs,awr=2.,Z=1,qa={51:4.})
    assert np.allclose(changed.kerma-baseline.kerma,5*xs[51])
    assert np.array_equal(changed.damage,baseline.damage)


def test_qbar_changes_balance_and_keeps_evaluated_two_body_kinematics():
    E=np.array([10.,15.,20.]); xs={51:np.ones(3)}
    q=m.XYs1d(xs=[1e-5,10.,20.],ys=[-2.,-2.,-4.])
    got=H.heatingOf(suite(),E,xs,awr=2.,Z=1,qbar={51:q})
    Q=np.array([-2.,-3.,-4.])
    # Isotropic two-body lab mean: (A/(A+1))²(E-threshold)+E/(A+1)².
    mean=4/9*(E-1.5)+E/9
    assert np.allclose(got.kerma,E+Q-mean,rtol=1e-14)


def test_user_table_domain_and_missing_reaction_are_rejected():
    E=np.array([10.,20.]);xs={51:np.ones(2)}
    with pytest.raises(ValueError,match='cover'):
        H.heatingOf(suite(),E,xs,awr=2.,Z=1,qbar={51:m.XYs1d(xs=[10.,15.],ys=[0.,1.])})
    with pytest.raises(ValueError,match='absent'):
        H.heatingOf(suite(),E,xs,awr=2.,Z=1,qa={91:1.})
    with pytest.raises(ValueError,match='both'):
        H.heatingOf(suite(),E,xs,awr=2.,Z=1,qa={51:1.},qbar={51:m.XYs1d(xs=[1.,30.],ys=[0.,1.])})


def test_positive_qbar_does_not_rewrite_the_evaluated_distribution():
    E=np.array([10.,20.]);xs={51:np.ones(2)}
    q=m.XYs1d(xs=[1e-5,20.],ys=[4.,4.])
    got=H.heatingOf(suite(),E,xs,awr=2.,Z=1,qbar={51:q})
    mean=4/9*(E-1.5)+E/9
    assert np.allclose(got.kerma,E+4.-mean)


def test_fission_qa_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError,match='fission-release'):
        H.heatingOf(suite(mt=18),np.array([10.,20.]),{18:np.ones(2)},awr=2.,Z=1,qa={18:1.})

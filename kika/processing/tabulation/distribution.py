"""One product's evaluated distribution → its Monte Carlo form (ACER's job).

:func:`tabulateDistribution` dispatches on the §18 form:

=====================================  =========================================
evaluated                              Monte Carlo (label ``MonteCarlo``)
=====================================  =========================================
angularTwoBody, elastic                angularTwoBody of mu tables
angularTwoBody, a level (MT 51-90,     uncorrelated: mu tables and
600-848, 875-890) for a neutron        :class:`LevelEnergy`
uncorrelated                           uncorrelated of tables (or kept laws);
                                       a ``weightedFunctionals`` energy becomes
                                       a :class:`DistributionMixture`
KalbachMann                            KalbachMann of tables, slope added
energyAngular / angularEnergy          energyAngularMC
=====================================  =========================================

The level kinematics are OpenMC's (``reaction.py:1211-1222``):
``threshold = (A+1)/A |Q|``, ``massRatio = (A/(A+1))**2``, ``A`` the target's
atomic weight ratio. A product kika does not tabulate (``unspecified``, a
recoil link, a branching) is reported and left without a Monte Carlo form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .angular import tabulateAngular
from .correlated import tabulateAngularEnergy, tabulateEnergyAngular, tabulateKalbachMann
from .energy import tabulateEnergy
from .tables import TabulationOptions
from kika._constants import NEUTRON_MASS_MEV

__all__ = ["TabulationContext", "tabulateDistribution", "isLevelMT", "levelEnergy",
           "MONTE_CARLO_LABEL"]

MONTE_CARLO_LABEL = "MonteCarlo"


def _m():
    from kika.nuclear_data import model

    return model


def isLevelMT(mt: int) -> bool:
    """A discrete-level two-body reaction: the outgoing particle's energy follows from Q."""
    return 51 <= mt <= 90 or 600 <= mt <= 648 or 650 <= mt <= 698 or 700 <= mt <= 748 \
        or 750 <= mt <= 798 or 800 <= mt <= 848 or 875 <= mt <= 890


def levelEnergy(Q: float, awr: float):
    """OpenMC's ``LevelInelastic`` from a Q value and the target's atomic weight ratio."""
    A = float(awr)
    return _m().LevelEnergy(threshold=(A + 1.0) / A * abs(float(Q)), massRatio=(A / (A + 1.0)) ** 2)


@dataclass
class TabulationContext:
    """What a tabulation needs beyond the form: the reaction and the nuclide."""

    awr: float
    zaTarget: int
    mt: int
    Q: Optional[float] = None
    pid: str = "n"
    zaEmitted: int = 1
    discrete: Optional[Sequence[int]] = None
    options: TabulationOptions = field(default_factory=TabulationOptions)
    report: Optional[object] = None


def _say(context, message):
    if context.report is not None:
        context.report.approximated(message)


def _uncorrelated(angular, energy, frame, context):
    m = _m()
    tabAngular, clipped = tabulateAngular(angular, context.options)
    tabEnergy, clippedE = tabulateEnergy(energy, context.options, awr=context.awr,
                                         discrete=context.discrete)
    if clipped or clippedE:
        _say(context, f"MT{context.mt} {context.pid}: {clipped + clippedE} tables had negative "
                      f"densities set to zero before normalising")
    return m.Uncorrelated(angular=tabAngular, energy=tabEnergy, label=MONTE_CARLO_LABEL,
                          productFrame=frame)


def tabulateDistribution(form, context: TabulationContext):
    """One evaluated form → its Monte Carlo form, or ``None`` when there is none to give."""
    m = _m()
    options = context.options
    if isinstance(form, m.AngularTwoBody):
        if form.isRecoil:
            return None
        tabAngular, clipped = tabulateAngular(form.angular, options)
        if clipped:
            _say(context, f"MT{context.mt} {context.pid}: {clipped} angular tables had negative "
                          f"densities set to zero before normalising")
        if context.mt == 2:
            if isinstance(tabAngular, m.Isotropic2d):
                tabAngular, _ = tabulateAngular(_isotropicXYs2d(), options)
            return m.AngularTwoBody(angular=tabAngular, productFrame=form.productFrame,
                                    label=MONTE_CARLO_LABEL)
        if isLevelMT(context.mt) and context.Q is not None:
            return m.Uncorrelated(angular=tabAngular, energy=levelEnergy(context.Q, context.awr),
                                  label=MONTE_CARLO_LABEL, productFrame=form.productFrame)
        if context.pid == "photon" and context.Q is not None and context.awr is not None:
            # A two-body photon (H-1 capture, MF6 LAW=2): one line per incident
            # energy, E_gamma = Q - Q^2/(2(A+1) m_n c^2) + A/(A+1) E -- a primary
            # photon, which is what ACER's table holds (2 223 300 eV at thermal).
            if isinstance(tabAngular, m.Isotropic2d):
                tabAngular, _ = tabulateAngular(_isotropicXYs2d(), options)
            q = float(context.Q)
            binding = q - q * q / (2.0 * (float(context.awr) + 1.0) * NEUTRON_MASS_MEV * 1.0e6)
            return m.Uncorrelated(angular=tabAngular,
                                  energy=m.DiscretePhotonEnergy(primaryFlag=2, energy=binding,
                                                                atomicWeightRatio=float(context.awr)),
                                  label=MONTE_CARLO_LABEL, productFrame=m.Frame.lab)
        return None
    if isinstance(form, m.Uncorrelated):
        energy = form.energy
        if isinstance(energy, m.WeightedFunctionals):
            terms = []
            for term in energy.weighted:
                piece = _uncorrelated(form.angular, term.functional, form.productFrame, context)
                terms.append(m.Weighted(weight=term.weight, functional=piece))
            return m.DistributionMixture(weighted=terms, label=MONTE_CARLO_LABEL)
        return _uncorrelated(form.angular, energy, form.productFrame, context)
    if isinstance(form, m.KalbachMann):
        return tabulateKalbachMann(form, options, zaEmitted=context.zaEmitted,
                                   zaTarget=context.zaTarget, discrete=context.discrete)
    if isinstance(form, m.EnergyAngular):
        return tabulateEnergyAngular(form, options, discrete=context.discrete)
    if isinstance(form, m.AngularEnergy):
        return tabulateAngularEnergy(form, options)
    if isinstance(form, (m.Unspecified, m.Branching3d)):
        return None
    raise TypeError(f"no tabulation for a {type(form).__name__}")


def _isotropicXYs2d():
    m = _m()
    return m.XYs2d(function1ds=[
        m.XYs1d(xs=[-1.0, 1.0], ys=[0.5, 0.5], outerDomainValue=1e-5),
        m.XYs1d(xs=[-1.0, 1.0], ys=[0.5, 0.5], outerDomainValue=2e7)])

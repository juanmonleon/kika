"""Kalbach's systematics for the slope ``a`` of a Kalbach-Mann distribution.

ENDF-6 §6.2.3.2 (Kalbach, Phys. Rev. C 37, 2350, 1988): when an evaluator gives
only the pre-compound fraction ``r`` (``NA = 1``, every Kalbach-Mann record of
ENDF/B-VIII.1), the slope is computed from the reaction's masses and energies.
This is ``openmc.data.kalbach_slope`` (``kalbach_mann.py:155-247``), ported
with its two approximations kept so the tables agree with OpenMC's: channel
energies use mass numbers rather than masses, and natural carbon is C-12.

**An incident photon** (``zaProjectile = 0``) takes Chadwick, Young and Chiba's
extension (ENDF-6 §6.2.3.2, Eq. 6.5): the neutron's slope with the photon's
energy put where the neutron's goes (compound = target + neutron), times
``sqrt(E_gamma / 2 m_n) min(4, max(1, 9.3 / sqrt(E_b,cm)))``, energies and the
neutron's mass in MeV. Its angular shape is not Kalbach's either (Eq. 6.6,
:func:`photonKalbachDensity`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["kalbachSlope", "separationEnergy", "photonKalbachDensity"]

#: ENDF-6 Appendix H, Table 3: break-up energies of the light particles, MeV.
_BREAKUP = {1: 0.0, 1001: 0.0, 1002: 2.224566, 1003: 8.481798, 2003: 7.718043, 2004: 28.29566}
_M = {1: 1.0, 1001: 1.0, 1002: 1.0, 2004: 0.0}
_m = {1: 0.5, 1001: 1.0, 1002: 1.0, 1003: 1.0, 2003: 1.0, 2004: 2.0}


def _zan(za):
    z, a = divmod(int(za), 1000)
    return z, a, a - z


def separationEnergy(compound, nucleus, particle) -> float:
    """Eq. 4 of Kalbach 1988, MeV. Arguments are ZA numbers."""
    Zc, Ac, Nc = _zan(compound)
    Za, Aa, Na = _zan(nucleus)
    return (15.68 * (Ac - Aa)
            - 28.07 * ((Nc - Zc) ** 2 / Ac - (Na - Za) ** 2 / Aa)
            - 18.56 * (Ac ** (2. / 3.) - Aa ** (2. / 3.))
            + 33.22 * ((Nc - Zc) ** 2 / Ac ** (4. / 3.) - (Na - Za) ** 2 / Aa ** (4. / 3.))
            - 0.717 * (Zc ** 2 / Ac ** (1. / 3.) - Za ** 2 / Aa ** (1. / 3.))
            + 1.211 * (Zc ** 2 / Ac - Za ** 2 / Aa)
            - _BREAKUP[int(particle)])


def kalbachSlope(energyIncident, energyEmitted, zaEmitted, zaTarget, zaProjectile=1):
    """The slope at incident energy (lab, eV) and emitted energy (CM, eV); vectorised in the latter.

    ``zaProjectile`` is 1 (a neutron) or 0 (a photon, ENDF-6 Eq. 6.5).
    """
    if zaProjectile == 0:
        from kika._constants import NEUTRON_MASS_MEV

        eb = np.asarray(energyEmitted, dtype=float) / 1.0e6
        with np.errstate(divide="ignore"):
            boost = np.where(eb > 0.0, np.minimum(4.0, np.maximum(1.0, 9.3 / np.sqrt(np.where(eb > 0.0, eb, 1.0)))),
                             4.0)
        return (kalbachSlope(energyIncident, energyEmitted, zaEmitted, zaTarget, 1)
                * np.sqrt(float(energyIncident) / 1.0e6 / (2.0 * NEUTRON_MASS_MEV)) * boost)
    if zaProjectile != 1:
        raise NotImplementedError("Kalbach's slope systematics here are for an incident neutron or photon")
    zaEmitted = 6012 if zaEmitted == 6000 else int(zaEmitted)
    zaTarget = 6012 if zaTarget == 6000 else int(zaTarget)
    compound = zaTarget + zaProjectile
    residual = compound - zaEmitted
    _, At, _ = _zan(zaTarget)
    _, Ap, _ = _zan(zaProjectile)
    _, Ar, _ = _zan(residual)
    _, Ab, _ = _zan(zaEmitted)
    epsilonA = energyIncident * At / (At + Ap) / 1.0e6
    epsilonB = np.asarray(energyEmitted, dtype=float) * (Ar + Ab) / (Ar * 1.0e6)
    sA = separationEnergy(compound, zaTarget, zaProjectile)
    sB = separationEnergy(compound, residual, zaEmitted)
    M, m = _M[zaProjectile], _m[zaEmitted]
    eA = epsilonA + sA
    eB = epsilonB + sB
    r1, r3 = min(eA, 130.0), min(eA, 41.0)
    x1, x3 = r1 * eB / eA, r3 * eB / eA
    return 0.04 * x1 + 1.8e-6 * x1 ** 3 + 6.7e-7 * M * m * x3 ** 4


def photonKalbachDensity(mu, r: float, a: float):
    """ENDF-6 Eq. 6.6, normalised over ``mu`` in [-1, 1]: ``[(1 - r) + r a/sinh(a) e^(a mu)] / 2``.

    The angular shape of a Kalbach-Mann product when the projectile is a photon:
    an isotropic part and a forward exponential, where a neutron's is
    ``a/(2 sinh a) [cosh(a mu) + r sinh(a mu)]``. ``a = 0`` is isotropic.
    """
    mu = np.asarray(mu, dtype=float)
    if a == 0.0:
        return np.full(mu.shape, 0.5)
    # a/sinh(a) e^(a mu) = 2a e^(a(mu - 1)) / (1 - e^(-2a)): no overflow at large a
    forward = 2.0 * a * np.exp(a * (mu - 1.0)) / -np.expm1(-2.0 * a) if a > 0.0 else \
        a / np.sinh(a) * np.exp(a * mu)
    return 0.5 * ((1.0 - r) + r * forward)

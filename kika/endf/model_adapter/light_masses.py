"""ENDF's light-particle masses: nuclear in the file, atomic in PoPs (FUDGE's rule).

ENDF-6 states a product's mass (MF6's AWP) and a charged projectile's (MF1/451's
AWI) as a ratio to the neutron, and for the hydrogen and helium isotopes that
ratio is the **nuclear** mass: 0.999167 for a proton, not the atomic 0.99917 +
the electron. A target's AWR is atomic. GNDS has one mass per particle, and the
distributed files (and FUDGE, ``brownies/legacy/converting/massTracker.py``)
put the **atomic** mass in PoPs and convert on the way in and out, adding or
removing Z electron masses and their binding energy. NNDC's Be-9 states H1 as
1.00837326306 amu, which is ENDF's 0.999167 n-masses plus one electron.

The constants are FUDGE's, so a GNDS written by either comes back to the same
ENDF number.
"""
from __future__ import annotations

from kika._constants import NEUTRON_MASS_AMU

__all__ = ["ELECTRON_MASS_AMU", "atomicFromRatio", "ratioFromAtomic", "isLight"]

#: FUDGE's ``MassTracker.electronMass`` (amu).
ELECTRON_MASS_AMU = 0.0005485801

_EV_C2_AMU = 1.07354416620656e-9

#: FUDGE's ``MassTracker.electronBindingEnergiesAmu``: the atom's electron
#: binding energy, negative, in amu.
_BINDING_AMU = {
    1001: -13.8885758218 * _EV_C2_AMU, 1002: -13.9351505263 * _EV_C2_AMU,
    1003: -14.3729527487 * _EV_C2_AMU, 2003: -70.5606714777 * _EV_C2_AMU,
    2004: -75.2181415144 * _EV_C2_AMU, 2005: -78.0 * _EV_C2_AMU,
    2006: -79.94 * _EV_C2_AMU, 2007: -79.99 * _EV_C2_AMU,
}


def isLight(za: int) -> bool:
    """Whether ENDF states this particle's mass ratio as a nuclear mass."""
    return int(za) in _BINDING_AMU


def atomicFromRatio(za: int, ratio: float) -> float:
    """An ENDF AWP/AWI → the atomic mass in amu PoPs carries."""
    amu = float(ratio) * NEUTRON_MASS_AMU
    if isLight(za):
        amu += ELECTRON_MASS_AMU * (int(za) // 1000) + _BINDING_AMU[int(za)]
    return amu


def ratioFromAtomic(za: int, amu: float) -> float:
    """The inverse: PoPs' atomic mass → ENDF's AWP/AWI."""
    amu = float(amu)
    if isLight(za):
        amu -= ELECTRON_MASS_AMU * (int(za) // 1000) + _BINDING_AMU[int(za)]
    return amu / NEUTRON_MASS_AMU

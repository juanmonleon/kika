"""What an MT means: the light particles it emits, its residual, its Q and threshold.

The particles are ENDF-102 Appendix B's (ENDF-6 manual, 2023): MT16 is (z,2n),
MT22 (z,n alpha), MT600-649 (z,p) to the levels of the residual, and so on.
MT5 (anything), MT18-21/38 (fission) and the summation MTs have no fixed
product list and are left out: :func:`emitted` returns ``None`` for them.

Masses are the atomic masses of AME2020 (the table kika ships for the
photonuclear processing). Atomic masses give the right Q of a nuclear
reaction because the electrons balance when charge is conserved.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from .context import AMU_EV, NEUTRON_MASS_U

__all__ = ["emitted", "residual_za", "mass_u", "q_from_masses", "threshold", "LIGHT",
           "SUMMATION_MT", "is_level_mt", "level_family"]

#: Light particles by name: (ZA, symbol in the summaries).
LIGHT = {"n": 1, "p": 1001, "d": 1002, "t": 1003, "h": 2003, "a": 2004}

# The products of each MT, as a string of particle letters (n, p, d, t, h=He-3,
# a=alpha), ENDF-102 Appendix B.
_MT = {
    4: "n", 11: "nnd", 16: "nn", 17: "nnn", 22: "na", 23: "naaa", 24: "nna", 25: "nnna",
    28: "np", 29: "naa", 30: "nnaa", 32: "nd", 33: "nt", 34: "nh", 35: "ndaa", 36: "ntaa",
    37: "nnnn", 41: "nnp", 42: "nnnp", 44: "npp", 45: "npa",
    102: "", 103: "p", 104: "d", 105: "t", 106: "h", 107: "a", 108: "aa", 109: "aaa",
    111: "pp", 112: "pa", 113: "taa", 114: "daa", 115: "pd", 116: "pt", 117: "da",
    152: "nnnnn", 153: "nnnnnn", 154: "nnt", 155: "ta", 156: "nnnnp", 157: "nnnd", 158: "nda",
    159: "nnpa", 160: "nnnnnnn", 161: "nnnnnnnn", 162: "nnnnnp", 163: "nnnnnnp",
    164: "nnnnnnnp", 165: "nnnna", 166: "nnnnna", 167: "nnnnnna", 168: "nnnnnnna",
    169: "nnnnd", 170: "nnnnnd", 171: "nnnnnnd", 172: "nnnt", 173: "nnnnt", 174: "nnnnnt",
    175: "nnnnnnt", 176: "nnh", 177: "nnnh", 178: "nnnnh", 179: "nnnpp", 180: "nnnaa",
    181: "nnnpa", 182: "dt", 183: "npd", 184: "npt", 185: "ndt", 186: "nph", 187: "ndh",
    188: "nth", 189: "nta", 190: "nnpp", 191: "ph", 192: "dh", 193: "ha", 194: "nnnnpp",
    195: "nnnnaa", 196: "nnnnpa", 197: "ppp", 198: "nppp", 199: "nnnppa", 200: "nnnnnpp",
}
for _m in range(50, 92):
    _MT[_m] = "n"
for _lo, _p in ((600, "p"), (650, "d"), (700, "t"), (750, "h"), (800, "a")):
    for _m in range(_lo, _lo + 50):
        _MT[_m] = _p
for _m in range(875, 892):
    _MT[_m] = "nn"

#: Redundant MTs, the sums of others (ENDF-102 Appendix B).
SUMMATION_MT = frozenset({1, 3, 4, 18, 27, 101, 103, 104, 105, 106, 107})

#: The discrete-level families: (first MT, last discrete MT, continuum MT).
_FAMILIES = ((50, 90, 91), (600, 648, 649), (650, 698, 699), (700, 748, 749),
             (750, 798, 799), (800, 848, 849), (875, 890, 891))


def level_family(mt: int) -> Optional[Tuple[int, int, int]]:
    """``(first, last discrete, continuum)`` of the family *mt* belongs to."""
    for fam in _FAMILIES:
        if fam[0] <= mt <= fam[2]:
            return fam
    return None


def is_level_mt(mt: int) -> bool:
    """A discrete level of a family (MT50-90, 600-648, ...), the ground state included.

    MT50 is the ground state of (z,n) for any projectile but the neutron, whose
    (n,n0) is MT2; it simply does not occur on a neutron tape.
    """
    fam = level_family(mt)
    return fam is not None and fam[0] <= mt <= fam[1]


def emitted(mt: int) -> Optional[Dict[int, int]]:
    """``{ZA: count}`` of the light particles *mt* emits; ``None`` when the MT has no fixed list.

    MT2 (elastic) is ``{}`` here too: the projectile comes back, nothing is
    created. MT4 is an (z,n') sum whose partials agree on one neutron.
    """
    if mt == 2:
        return {}
    letters = _MT.get(mt)
    if letters is None:
        return None
    out: Dict[int, int] = {}
    for ch in letters:
        za = LIGHT[ch]
        out[za] = out.get(za, 0) + 1
    return out


def residual_za(target_za: int, projectile_za: int, mt: int) -> Optional[int]:
    """The residual left by *mt*: target + projectile minus what it emits."""
    out = emitted(mt)
    if out is None:
        return None
    z = target_za // 1000 + max(projectile_za, 0) // 1000
    a = target_za % 1000 + max(projectile_za, 0) % 1000
    if mt == 2:
        return target_za
    for za, n in out.items():
        z -= n * (za // 1000)
        a -= n * (za % 1000)
    if z < 0 or a <= 0 or a < z:
        return None
    return 1000 * z + a


@lru_cache(maxsize=1)
def _ame() -> Dict[int, float]:
    import numpy as np

    path = Path(__file__).resolve().parents[3] / "ace" / "processing" / "photonuclear" / "data" / "ame2020_masses.npz"
    d = np.load(path)
    return dict(zip(d["za"].tolist(), d["mass_u"].tolist()))


def mass_u(za: int) -> Optional[float]:
    """The atomic mass of the ground state of *za* in u, AME2020; ``None`` if it is not there."""
    if za == 0:
        return 0.0
    if za == 1:
        return NEUTRON_MASS_U
    return _ame().get(int(za))


def q_from_masses(target_za: int, projectile_za: int, mt: int, *, elis: float = 0.0) -> Optional[float]:
    """The ground-state Q of *mt* (eV) from AME2020; the target's own excitation *elis* added.

    ``None`` when the MT has no fixed products or a mass is missing (an
    elemental target, a nucleus off the table).
    """
    if target_za % 1000 == 0:
        return None
    out = emitted(mt)
    res = residual_za(target_za, projectile_za, mt)
    if out is None or res is None:
        return None
    masses = [mass_u(target_za), mass_u(projectile_za) if projectile_za > 0 else 0.0, mass_u(res)]
    masses += [mass_u(za) for za in out]
    if any(m is None for m in masses):
        return None
    m_in = masses[0] + masses[1]
    m_out = masses[2] + sum(mass_u(za) * n for za, n in out.items())
    return (m_in - m_out) * AMU_EV + elis


def threshold(q: float, awr: float, awi: float) -> float:
    """The lab threshold (eV) of a reaction with Q < 0, relativistic.

    The invariant mass of the final state at rest equals that of target plus
    projectile: ``E_th = ((M_T + m - Q)^2 - (M_T + m)^2) / (2 M_T)`` with the
    masses as energies (``awr``/``awi`` in neutron masses; a photon has
    ``awi = 0``). It is ``-Q (A + a)/A`` to first order in ``Q / M``; the
    difference, ``Q^2 / 2 M_T``, is 0.9 keV for a 10 MeV Q on iron.
    """
    if q >= 0:
        return 0.0
    mn = NEUTRON_MASS_U * AMU_EV
    mt_, mp = awr * mn, awi * mn
    s0 = mt_ + mp
    return ((s0 - q) ** 2 - s0 ** 2) / (2.0 * mt_)

"""The parametrised energy spectra of ENDF-6 §5.1.1 / GNDS §18.3, as formulae.

Five laws are stated by a few energy-dependent parameters instead of a table.
Their *shapes* and the integrals that normalise them live here, once, so that
the ENDF reader (:mod:`kika.endf.classes.mf5.analytic`) and the model's §18.3
nodes (:mod:`kika.nuclear_data.model.energy_spectra`) evaluate the same
spectrum with the same arithmetic. It knows nothing of formats or units:
energies are plain floats in whatever unit the caller keeps consistent (eV
everywhere in kika).

These are physics -- the evaporation, Watt and Madland-Nix fission-spectrum
models -- so they live in the calculation layer, not in :mod:`kika.algebra`,
which holds only the mathematics of tabulated functions. Until October 2026
this module was ``kika.algebra.spectra``.

=====================  ====  ===================================================
law                    LF    unnormalised shape of E'
=====================  ====  ===================================================
simple Maxwellian      7     ``sqrt(E') exp(-E'/theta)``
evaporation            9     ``E' exp(-E'/theta)``
Watt                   11    ``exp(-E'/a) sinh(sqrt(b E'))``
Madland-Nix            12    normalised by construction, see :func:`madland_nix`
general evaporation    5     a table ``g(E'/theta)``; integrated by the table
=====================  ====  ===================================================

The first three are normalised over ``[0, E - U]`` by the closed forms below
(ENDF-6 §5.1.1.2-5). Madland-Nix is the exception: it has no ``U`` and is
normalised over ``[0, inf)`` analytically (§5.1.1.6), which is why truncating
it at ``E - U`` -- the rule of the other laws -- would be wrong. At a thermal
incident energy that bound is 0.0253 eV, and the spectrum would vanish.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

__all__ = [
    "maxwellian", "maxwellian_integral", "evaporation", "evaporation_integral",
    "watt", "watt_integral", "madland_nix", "madland_nix_mean",
    "madland_nix_upper",
]


def maxwellian(e_out, theta: float) -> np.ndarray:
    """LF=7 shape, ``sqrt(E') exp(-E'/theta)``. Zero for ``theta <= 0``."""
    e_out = np.asarray(e_out, dtype=float)
    if theta <= 0.0:
        return np.zeros(e_out.shape)
    return np.sqrt(e_out) * np.exp(-e_out / theta)


def maxwellian_integral(hi: float, theta: float) -> float:
    """``int_0^hi`` of :func:`maxwellian`: ``theta^1.5 [sqrt(pi)/2 erf(sqrt(y)) - sqrt(y) e^-y]``."""
    if theta <= 0.0 or hi <= 0.0:
        return 0.0
    y = hi / theta
    return theta ** 1.5 * (0.5 * math.sqrt(math.pi) * math.erf(math.sqrt(y))
                           - math.sqrt(y) * math.exp(-y))


def evaporation(e_out, theta: float) -> np.ndarray:
    """LF=9 shape, ``E' exp(-E'/theta)``."""
    e_out = np.asarray(e_out, dtype=float)
    if theta <= 0.0:
        return np.zeros(e_out.shape)
    return e_out * np.exp(-e_out / theta)


def evaporation_integral(hi: float, theta: float) -> float:
    """``int_0^hi`` of :func:`evaporation`: ``theta^2 [1 - e^-y (1 + y)]``."""
    if theta <= 0.0 or hi <= 0.0:
        return 0.0
    y = hi / theta
    return theta * theta * (1.0 - math.exp(-y) * (1.0 + y))


def watt(e_out, a: float, b: float) -> np.ndarray:
    """LF=11 shape, ``exp(-E'/a) sinh(sqrt(b E'))``."""
    e_out = np.asarray(e_out, dtype=float)
    if a <= 0.0 or b < 0.0:
        return np.zeros(e_out.shape)
    return np.exp(-e_out / a) * np.sinh(np.sqrt(b * e_out))


def watt_integral(hi: float, a: float, b: float) -> float:
    """``int_0^hi`` of :func:`watt`, ENDF-6 §5.1.1.5's closed form."""
    if a <= 0.0 or b < 0.0 or hi <= 0.0:
        return 0.0
    root = math.sqrt(a * b / 4.0)
    y = math.sqrt(hi / a)
    return (0.5 * math.sqrt(math.pi * b * a ** 3 / 4.0)
            * math.exp(a * b / 4.0)
            * (math.erf(y - root) + math.erf(y + root))
            - a * math.exp(-hi / a) * math.sinh(math.sqrt(b * hi)))


def _madland_nix_one(e_out: np.ndarray, ef: float, tm: float) -> np.ndarray:
    """One fragment's term ``g(E', E_F)`` of §5.1.1.6.

    ``g = [u2^1.5 E1(u2) - u1^1.5 E1(u1) + gamma(3/2, u2) - gamma(3/2, u1)]
    / (3 sqrt(E_F T_M))`` with ``u1,2 = (sqrt(E') -/+ sqrt(E_F))^2 / T_M``,
    ``E1`` the exponential integral and ``gamma`` the *lower* incomplete gamma
    function. Closed, so no quadrature: the docstring that kept LF=12
    undecoded ("needs a numerical double integral") was wrong about that.

    ``u1 = 0`` at ``E' = E_F`` makes ``u1^1.5 E1(u1)`` a ``0 * inf``; its limit
    is 0 and it is taken here rather than left to produce a NaN.
    """
    root_e = np.sqrt(e_out)
    root_f = math.sqrt(ef)
    u1 = (root_e - root_f) ** 2 / tm
    u2 = (root_e + root_f) ** 2 / tm
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        t1 = np.where(u1 > 0.0, u1 ** 1.5 * special.exp1(np.where(u1 > 0.0, u1, 1.0)), 0.0)
        t2 = np.where(u2 > 0.0, u2 ** 1.5 * special.exp1(np.where(u2 > 0.0, u2, 1.0)), 0.0)
    gamma32 = special.gamma(1.5)
    lower = gamma32 * (special.gammainc(1.5, u2) - special.gammainc(1.5, u1))
    return (t2 - t1 + lower) / (3.0 * math.sqrt(ef * tm))


def madland_nix(e_out, efl: float, efh: float, tm: float) -> np.ndarray:
    """LF=12, the Madland-Nix spectrum: normalised to 1 over ``[0, inf)``.

    ``f(E') = [g(E', EFL) + g(E', EFH)] / 2`` -- the light and the heavy
    fragment, each a triangular distribution of temperatures up to ``T_M``.
    Zero for a non-positive ``T_M`` or fragment energy.
    """
    e_out = np.asarray(e_out, dtype=float)
    if tm <= 0.0 or efl <= 0.0 or efh <= 0.0:
        return np.zeros(e_out.shape)
    clipped = np.maximum(e_out, 0.0)
    value = 0.5 * (_madland_nix_one(clipped, efl, tm) + _madland_nix_one(clipped, efh, tm))
    return np.where(e_out >= 0.0, value, 0.0)


def madland_nix_mean(efl: float, efh: float, tm: float) -> float:
    """``<E'>`` of :func:`madland_nix`: ``(EFL + EFH)/2 + 4/3 T_M``.

    The closed form of the first moment, independent of the density above --
    which is what makes it a check on it rather than a restatement.
    """
    return 0.5 * (efl + efh) + 4.0 * tm / 3.0


def madland_nix_upper(efh: float, tm: float, decades: float = 30.0) -> float:
    """An outgoing energy past which :func:`madland_nix` is negligible.

    The tail falls like ``exp(-u1)`` with ``u1 = (sqrt(E') - sqrt(EFH))^2/T_M``,
    so ``u1 = decades * ln 10`` puts the density ~``10**-decades`` below its
    peak scale. Only a grid choice: the law has no upper bound of its own.
    """
    return (math.sqrt(efh) + math.sqrt(decades * math.log(10.0) * tm)) ** 2

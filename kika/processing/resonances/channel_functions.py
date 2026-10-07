"""Neutral-channel functions without a Coulomb barrier.

P/S use the neutral recurrence. The hard-sphere phase is computed from the
Riccati-Bessel ratio (modulo 2*pi); it avoids rho - atan(rho) cancellation.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def neutral_channel_functions(l, rho):
    """Return P_l, S_l, phi_l for finite positive rho, L=0..64.

    Extremely small penetrabilities can underflow; width-reference preparation
    rejects that case rather than silently replacing its denominator.
    """
    if not isinstance(l, (int, np.integer)) or not 0 <= l <= 64:
        raise ValueError("neutral channels support integer L=0..64")
    x = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("rho must be finite and positive")
    p, s = x.copy(), np.zeros_like(x)
    for ll in range(1, l + 1):
        denominator = (ll - s)**2 + p**2
        p, s = x**2 * p / denominator, x**2 * (ll - s) / denominator - ll
    phase = x.copy() if l == 0 else np.arctan2(spherical_jn(l, x), -spherical_yn(l, x))
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(phase)):
        raise FloatingPointError("neutral-channel functions exceeded floating-point range")
    return p, s, phase



def neutral_shift_difference(l, reference_squared, squared, difference):
    """S_l(reference) - S_l(rho), without subtracting two values near -l.

    ``difference`` is the accurately formed reference_squared - squared.
    Propagate divided differences through the neutral recurrence. Tracking
    S_l+l separately avoids losing its small energy dependence for small rho.
    """
    if not isinstance(l, (int, np.integer)) or not 0 <= l <= 64:
        raise ValueError("neutral channels support integer L=0..64")
    zr, z, dz = np.broadcast_arrays(reference_squared, squared, difference)
    if np.any(~np.isfinite(zr+z+dz)) or np.any(zr <= 0) or np.any(z <= 0):
        raise ValueError("finite positive squared channel arguments required")
    pr, p = np.sqrt(zr), np.sqrt(z)
    dp = dz/(pr+p)
    cr, c, dc = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
    for order in range(1, l+1):
        dr, d = 2*order-1-cr, 2*order-1-c
        denominator_r, denominator = dr*dr+pr*pr, d*d+p*p
        dd = -dc
        delta_denominator = dd*(dr+d)+dp*(pr+p)
        next_p = z*p/denominator
        next_c = z*d/denominator
        next_dp = (dz*pr+z*dp-next_p*delta_denominator)/denominator_r
        next_dc = (dz*dr+z*dd-next_c*delta_denominator)/denominator_r
        pr, p = zr*pr/denominator_r, next_p
        cr, c, dp, dc = zr*dr/denominator_r, next_c, next_dp, next_dc
    if np.any(~np.isfinite(dc)):
        raise FloatingPointError("neutral shift difference exceeded floating-point range")
    return dc

"""Neutral-channel functions without a Coulomb barrier.

P/S use the neutral recurrence. The hard-sphere phase is computed from the
Riccati-Bessel ratio (modulo 2*pi); it avoids rho - atan(rho) cancellation.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _low_j_coefficients(l):
    values=[1.]
    for k in range(1,11):values.append(-values[-1]/(2*k*(2*l+2*k+1)))
    return tuple(values)


_LOW_J_COEFFICIENTS={l:_low_j_coefficients(l) for l in (1,2)}


def _low_phase(l, x):
    """Riccati-Bessel ratio for L=1/2, with a regular small-x numerator."""
    z=x*x
    if l==1:
        numerator=np.sin(x)-x*np.cos(x)
        denominator=np.cos(x)+x*np.sin(x)
    else:
        numerator=(3-z)*np.sin(x)-3*x*np.cos(x)
        denominator=(3-z)*np.cos(x)+3*x*np.sin(x)
    numerator=np.asarray(numerator).copy()
    small=x<1.
    zs=z[small]
    coefficients=_LOW_J_COEFFICIENTS[l]
    series=np.full_like(zs,coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        series *= zs
        series += coefficient
    numerator[small] = (x[small]**3/3 if l==1 else x[small]**5/15)*series
    return np.arctan2(numerator,denominator)


def neutral_channel_functions(l, rho, *, phase=True):
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
    phi = (x.copy() if l == 0 else _low_phase(l,x) if l<=2
           else np.arctan2(spherical_jn(l, x), -spherical_yn(l, x))) if phase else None
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(s)) or (phase and np.any(~np.isfinite(phi))):
        raise FloatingPointError("neutral-channel functions exceeded floating-point range")
    return p, s, phi



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
    # Exact divided differences of S_1=-1/(1+z) and
    # S_2=-3*(z+6)/(z*z+3*z+9). No subtraction of near-equal
    # shifts, and no square-root recurrence intermediates.
    if l==1:
        return dz/(1+zr)/(1+z)
    if l==2 and np.all(np.maximum(zr,z)<1e50) and np.all(abs(dz)<1e50):
        return 3*dz*(zr*z+6*(zr+z)+9)/((zr*zr+3*zr+9)*(z*z+3*z+9))
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

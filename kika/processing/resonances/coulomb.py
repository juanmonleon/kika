"""Coulomb logarithmic derivatives from DLMF 33.8, with scaled barrier transport.

The production calculation uses only NumPy/SciPy. Arbitrary precision belongs
to the independent development oracle. Only repulsive charged exits are supported.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import kve


def _fraction(initial, coefficients, max_iterations=20000):
    """Modified Lentz evaluation; require successive converged increments."""
    tiny = 1e-150
    f = complex(initial) if initial != 0 else complex(tiny)
    c = f; d = 0j; stable = 0
    for n in range(1,max_iterations+1):
        a,b = coefficients(n)
        d = b+a*d; c = b+a/c
        if abs(d)<tiny:d = complex(tiny)
        if abs(c)<tiny:c = complex(tiny)
        d = 1/d; change = c*d; f *= change
        stable = stable+1 if abs(change-1)<2e-14 else 0
        if stable >= 3:
            if not np.isfinite(f):break
            return f
    raise FloatingPointError('Coulomb continued fraction did not converge')


def _regular(l,eta,rho):
    first = l+1
    def coefficients(n):
        k = l+n
        return -(1+(eta/k)**2), (2*k+1)/rho+eta/k+eta/(k+1)
    return _fraction(first/rho+eta/first,coefficients).real


def _outgoing(l,eta,rho):
    a = l+1+1j*eta; b = -l+1j*eta
    def coefficients(n):
        return ((1j/rho)*a*b if n==1 else (a+n-1)*(b+n-1)), 2*(rho-eta+1j*n)
    return _fraction(1j*(1-eta/rho),coefficients)


def _transport(l,eta,start,stop,shift,log_p):
    """Transport S=rho*H'/H and ln(P) without large wavefunctions."""
    evaluations = 0
    def equation(t,y):
        nonlocal evaluations
        evaluations += 1
        if evaluations>100000:
            raise FloatingPointError('scaled Coulomb transport exceeded its work budget')
        rho = np.exp(t); s = y[0]
        p = np.exp(y[1])
        return [s-rho*rho+2*eta*rho+l*(l+1)-s*s+p*p,1-2*s]
    y = [shift,log_p]
    result = solve_ivp(equation,(np.log(start),np.log(stop)),y,method='DOP853',rtol=2e-12,atol=2e-13)
    if not result.success or np.any(~np.isfinite(result.y)):
        raise FloatingPointError('scaled Coulomb transport did not converge')
    return result.y[:,-1]


def charged_channel_log_functions(l,eta,rho):
    """Return ln(P),S,hard-sphere phase for finite rho>0, eta>=0.

    Phase is modulo pi, sufficient for neutral-incidence cross sections. A
    ln(P) remains available below the positive float range. A phase below
    that range rounds to zero; this does not change ln(P) or label P zero.
    """
    from .channel_functions import neutral_channel_functions
    if not isinstance(l,(int,np.integer)) or not 0<=l<=64:
        raise ValueError('charged orbital momentum must be L=0..64')
    eta,rho = np.broadcast_arrays(np.asarray(eta,dtype=float),np.asarray(rho,dtype=float))
    if np.any(~np.isfinite(eta+rho)) or np.any(eta<0) or np.any(rho<=0):
        raise ValueError('charged functions require finite eta>=0 and rho>0')
    p = np.empty_like(rho);s = np.empty_like(rho);phase = np.empty_like(rho)
    for index in np.ndindex(rho.shape):
        h,r = float(eta[index]),float(rho[index])
        if h==0:
            p[index],s[index],phase[index] = neutral_channel_functions(l,r)
            if p[index]>0:
                p[index] = np.log(p[index])
            else:
                # Exact positive neutral denominator polynomial, evaluated
                # logarithmically when its penetrability has underflowed.
                from scipy.special import gammaln,logsumexp
                n=np.arange(l+1)
                coefficients=(gammaln(l+n+1)-gammaln(l-n+1)+gammaln(2*n+1)
                    -2*gammaln(n+1)-2*n*np.log(2.))
                p[index]=np.log(r)-logsumexp(coefficients-2*n*np.log(r))
            continue
        turning = h+np.hypot(h,np.sqrt(l*(l+1)))
        start = max(r,turning+8.)
        derivative = _outgoing(l,h,start)
        if derivative.imag<=0:
            raise FloatingPointError('Coulomb outgoing flux lost precision')
        shift = start*derivative.real; lp = np.log(start)+np.log(derivative.imag)
        if start!=r:shift,lp = _transport(l,h,start,r,shift,lp)
        penetration = np.exp(lp)
        if not np.isfinite(lp+shift):
            raise FloatingPointError('nonfinite scaled charged channel functions')
        regular = _regular(l,h,r)
        p[index] = lp;s[index] = shift
        phase[index] = np.arctan2(penetration,r*regular-shift)
    return p,s,phase


def charged_channel_functions(l,eta,rho):
    """Return representable P,S,phase; reject an underflow instead of clipping."""
    log_p,shift,phase = charged_channel_log_functions(l,eta,rho)
    p = np.array(np.exp(log_p),copy=True)
    # Preserve the exact neutral limit, including its rounding. A log/exp
    # round trip needlessly changes the eta=0 branch by a few ulps.
    eta_values,rho_values = np.broadcast_arrays(np.asarray(eta),np.asarray(rho))
    neutral = eta_values==0
    if np.any(neutral):
        from .channel_functions import neutral_channel_functions
        p[neutral] = neutral_channel_functions(l,rho_values[neutral])[0]
    if np.any(~np.isfinite(p)) or np.any(p<=0):
        raise FloatingPointError('charged penetrability is not representable; use logarithmic channel functions')
    return p,shift,phase


def closed_charged_shift(l,eta,kappa):
    """Logarithmic derivative of decaying W(-eta,l+1/2,2*kappa)."""
    from .r_matrix import closed_neutral_shift
    eta,kappa = np.broadcast_arrays(np.asarray(eta,dtype=float),np.asarray(kappa,dtype=float))
    if not isinstance(l,(int,np.integer)) or not 0<=l<=64 or np.any(~np.isfinite(eta+kappa)) or np.any(eta<0) or np.any(kappa<=0):
        raise ValueError('closed charged functions require L=0..64, finite eta>=0, kappa>0')
    out = np.empty_like(kappa)
    for index in np.ndindex(kappa.shape):
        h,r = float(eta[index]),float(kappa[index])
        if h==0:out[index] = closed_neutral_shift(l,r);continue
        # Positive Laplace integral for Tricomi U, with u=kappa*t.
        # The logarithmic derivative is l+1-kappa-2*mean(u). This scaling
        # keeps eta*kappa finite at a threshold; there is no remote stiff ODE.
        mode = .5*(l-r+np.sqrt((l-r)**2+2*(h+l)*r))
        if mode<=0:mode = 1.
        def log_weight(u):
            return 2*l*np.log(u)-(h-l)*np.log1p(r/u)-2*u
        peak = log_weight(mode)
        def weight(v):
            u = mode*v
            return np.exp(log_weight(u)-peak) if v>0 else 0.
        norm,norm_error = quad(weight,0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=200)
        moment,moment_error = quad(lambda v:v*weight(v),0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=200)
        if norm<=0 or not np.isfinite(norm+moment) or max(norm_error/norm,moment_error/moment)>1e-9:
            raise FloatingPointError('closed Coulomb Laplace integral did not converge')
        out[index] = l+1-r-2*mode*moment/norm
    return out


def charged_threshold_shift(l,eta_rho):
    """Exact zero-energy repulsive limit; eta*rho remains finite."""
    value = np.asarray(eta_rho,dtype=float)
    if not isinstance(l,(int,np.integer)) or not 0<=l<=64 or np.any(value<=0) or np.any(~np.isfinite(value)):
        raise ValueError('threshold requires L=0..64 and positive finite eta*rho')
    z = np.sqrt(8*value);order = 2*l+1
    return .5-.25*z*(kve(order-1,z)+kve(order+1,z))/kve(order,z)

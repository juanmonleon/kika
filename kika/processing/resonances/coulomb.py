"""Coulomb logarithmic derivatives from DLMF 33.8, with scaled barrier transport.

The production calculation uses only NumPy/SciPy. Arbitrary precision belongs
to the independent development oracle. Only repulsive charged exits are supported.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.integrate._ivp import dop853_coefficients
from kika._native import load as _load_native
_native = _load_native(__package__ + "._coulomb_native", 1)
from scipy.special import kve, digamma, loggamma, gammaln


def _small_radius_outgoing(l,eta,rho):
    """Integer-b Tricomi series, DLMF 13.2.9, normalized at the origin.

    Products use eta*rho rather than large eta powers. This avoids transporting
    the outgoing wave from its increasingly remote turning point at a threshold.
    The imaginary logarithmic derivative is recovered by normalization, not
    cancellation of complex series components.
    """
    m=2*l+1;a=l+1+1j*eta;z=np.clongdouble(-2j*rho)
    term=np.clongdouble(1.);value=term;derivative=np.clongdouble(0.)
    magnitude=np.longdouble(1.)
    for j in range(1,m):
        term *= -z*(a-m+j-1)/(j*(m-j))
        value+=term;derivative+=j*term;magnitude+=abs(term)
    prefactor=np.clongdouble(np.exp(-np.longdouble(gammaln(m+1)+gammaln(m))))
    for j in range(m):prefactor*=z*(a-m+j)
    coefficient=np.clongdouble(1.);stable=0
    for k in range(10000):
        bracket=np.log(z)+digamma(a+k)-digamma(k+1.)-digamma(m+k+1.)
        piece=prefactor*coefficient*bracket
        value+=piece;derivative+=prefactor*coefficient*((m+k)*bracket+1.)
        magnitude+=abs(piece)
        stable=stable+1 if abs(piece)<2e-18*abs(value) else 0
        if stable==4:break
        coefficient*=z*(a+k)/((k+1)*(m+k+1))
    else:raise FloatingPointError('small-radius Coulomb series did not converge')
    if abs(value)==0 or not np.isfinite(value+derivative) or 32*np.finfo(float).eps*magnitude/abs(value)>2e-10:
        raise FloatingPointError('small-radius Coulomb series lost accuracy through cancellation')
    shift=float((-l+1j*rho+derivative/value).real)
    log_c=l*np.log(2.)-.5*np.pi*eta+loggamma(l+1+1j*eta).real-gammaln(2*l+2)
    log_p=(2*l+1)*np.log(rho)+2*(np.log(2*l+1)+log_c)-2*float(np.log(abs(value)))
    return shift,log_p


def _outgoing_laplace(l,eta,rho):
    """Tricomi's Laplace integral rotated to the positive imaginary axis.

    The scaled integration variable is v=rho*Im(t). Near a threshold the
    dominant weight depends on eta*rho, so the turning point is never visited.
    Both the wave modulus and its derivative come from the same integral.
    """
    product=eta*rho
    mode=.5*(l+np.hypot(l,np.sqrt(2*product)))
    def weight_log(v):
        angle=np.arctan2(rho,v)
        return 2*l*np.log(v)+.5*l*np.log1p((rho/v)**2)-eta*angle-2*v
    peak=weight_log(mode)
    def integrand(t,derivative=False):
        if t==0:return 0j
        v=mode*t;ratio=rho/v
        phase=-l*np.arctan(ratio)-.5*eta*np.log1p(ratio*ratio)
        value=np.exp(weight_log(v)-peak+1j*phase)
        if derivative:
            denom=1+ratio*ratio
            value*=((l*ratio*ratio-eta*ratio)/denom
                    -1j*(l*ratio+eta*ratio*ratio)/denom)
        return value
    values=[];errors=[]
    for derivative in (False,True):
        for imaginary in (False,True):
            fun=lambda t: (integrand(t,derivative).imag if imaginary else integrand(t,derivative).real)
            value,error=quad(fun,0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=300)
            values.append(value);errors.append(error)
    wave=complex(*values[:2]);slope=complex(*values[2:])
    if abs(wave)==0 or not np.isfinite(wave+slope) or sum(errors)>2e-10*abs(wave):
        raise FloatingPointError('outgoing Coulomb Laplace integral did not converge')
    log_wave=(.5*np.pi*eta+(l+1)*np.log(2.)-l*np.log(rho)
              -loggamma(l+1+1j*eta).real+np.log(mode)+peak+np.log(abs(wave)))
    return -l+(slope/wave).real,np.log(rho)-2*log_wave


_TINY = 1e-150


def _fraction(initial, coefficients, max_iterations=20000):
    """Modified Lentz evaluation; require successive converged increments."""
    tiny = _TINY
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


def _outgoing(l,eta,rho,max_iterations=20000):
    a = l+1+1j*eta; b = -l+1j*eta
    def coefficients(n):
        return ((1j/rho)*a*b if n==1 else (a+n-1)*(b+n-1)), 2*(rho-eta+1j*n)
    return _fraction(1j*(1-eta/rho),coefficients,max_iterations)


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


def _barrier_log_p(l,eta,rho,outgoing,regular):
    """ln(P) inside the barrier from F, F'/F and Re(H'/H), by the Wronskian.

    There P=rho*Im(H'/H) is a tiny imaginary part, which the continued
    fraction for H'/H carries only to an absolute accuracy. F is small and its
    power series (DLMF 33.6) has terms of one sign. With F'G-FG'=1 and
    p=Re(H'/H), G solves (p-F'/F)G^2+G/F+(p-F'/F)F^2=0; its dominant root has
    no cancellation, and P=rho/(F^2+G^2). Returns None where the series or
    the root is not well conditioned, for the caller's general route.
    """
    log_c = l*np.log(2.)-.5*np.pi*eta+loggamma(l+1+1j*eta).real-gammaln(2*l+2)
    previous,term,total,magnitude,quiet = 0.,1.,1.,1.,0
    for j in range(1,10000):
        term,previous = (2*eta*rho*term-rho*rho*previous)/(j*(2*l+1+j)),term
        total += term;magnitude += abs(term)
        if not np.isfinite(total) or total>1e280:return None
        quiet = quiet+1 if abs(term)<1e-17*abs(total) else 0
        if quiet>=2:break
    else:return None
    if total<=0 or magnitude>8*total:return None
    log_f = log_c+(l+1)*np.log(rho)+np.log(total)
    a = outgoing.real-regular
    if not a<0:return None
    disc = 1-np.exp(np.log(4*a*a)+4*log_f)
    if not disc>=.25:return None
    log_g = -log_f+np.log1p(np.sqrt(disc))-np.log(2*abs(a))
    return np.log(rho)-2*log_g-np.log1p(np.exp(2*log_f-2*log_g))


#: Iterations of the outgoing fraction allowed below rho=0.25: measured,
#: S keeps a relative accuracy of 1e-12 up to about 3000.
SMALL_RADIUS_FRACTION_ITERATIONS = 3000


def _charged_point(l,h,r):
    """ln(P), S and phase at one point; the reference for every branch."""
    from .channel_functions import neutral_channel_functions
    if h==0:
        p,s,phase = neutral_channel_functions(l,r)
        if p>0:
            p = np.log(p)
        else:
            # Exact positive neutral denominator polynomial, evaluated
            # logarithmically when its penetrability has underflowed.
            from scipy.special import gammaln,logsumexp
            n=np.arange(l+1)
            coefficients=(gammaln(l+n+1)-gammaln(l-n+1)+gammaln(2*n+1)
                -2*gammaln(n+1)-2*n*np.log(2.))
            p=np.log(r)-logsumexp(coefficients-2*n*np.log(r))
        return p,s,phase
    turning = h+np.hypot(h,np.sqrt(l*(l+1)))
    regular = None
    if r<=.25 and h*r<=2.:
        shift,lp=_small_radius_outgoing(l,h,r)
    elif r<=.25:
        # Deep in the barrier: the fraction and the Wronskian while the
        # fraction stays short (its rounding in S grows with the iterations,
        # about 100/rho of them); the Laplace integral nearer the threshold.
        lp = None
        try:
            here = _outgoing(l,h,r,SMALL_RADIUS_FRACTION_ITERATIONS);regular = _regular(l,h,r)
            lp = _barrier_log_p(l,h,r,here,regular);shift = r*here.real
        except FloatingPointError:
            pass
        if lp is None:shift,lp=_outgoing_laplace(l,h,r)
    else:
        start = max(r,turning+8.)
        lp = None
        if start!=r:
            # The fraction at rho carries P=rho*Im(H'/H) to full relative
            # accuracy once P>=|S|/5 (measured: 3e-15 at P/|S|=0.2-0.3, 1e-13
            # at 0.09). Deeper in the barrier S still comes from it and ln(P)
            # from the Wronskian; the transport where that is not conditioned.
            try:
                here = _outgoing(l,h,r)
                if here.imag>0 and r*here.imag>=.2*abs(r*here.real):
                    shift = r*here.real;lp = np.log(r)+np.log(here.imag)
                else:
                    regular = _regular(l,h,r)
                    lp = _barrier_log_p(l,h,r,here,regular)
                    shift = r*here.real
            except FloatingPointError:
                lp = None
        if lp is None:
            derivative = _outgoing(l,h,start)
            if derivative.imag<=0:
                raise FloatingPointError('Coulomb outgoing flux lost precision')
            shift = start*derivative.real; lp = np.log(start)+np.log(derivative.imag)
            if start!=r:shift,lp = _transport(l,h,start,r,shift,lp)
    penetration = np.exp(lp)
    if not np.isfinite(lp+shift):
        raise FloatingPointError('nonfinite scaled charged channel functions')
    if regular is None:regular = _regular(l,h,r)
    return lp,shift,np.arctan2(penetration,r*regular-shift)


def _native_open_rows(l,eta,rho):
    """`_charged_point` in the compiled kernel, outside the small-radius series."""
    n = len(rho)
    log_p,shift,phase = np.empty(n),np.empty(n),np.empty(n);ok = np.zeros(n,np.uint8)
    _native.open_rows(int(l),n,np.ascontiguousarray(eta,dtype=float),np.ascontiguousarray(rho,dtype=float),
                      *_TABLEAU,log_p,shift,phase,ok)
    return log_p,shift,phase,ok.astype(bool)


_STAGES = dop853_coefficients.N_STAGES
_TABLEAU = tuple(np.ascontiguousarray(v,dtype=float).ravel() for v in (
    dop853_coefficients.A[:_STAGES,:_STAGES],dop853_coefficients.B,dop853_coefficients.C[:_STAGES],
    dop853_coefficients.E3,dop853_coefficients.E5))


def charged_channel_log_functions(l,eta,rho):
    """Return ln(P),S,hard-sphere phase for finite rho>0, eta>=0.

    Phase is modulo pi, sufficient for neutral-incidence cross sections. A
    ln(P) remains available below the positive float range. A phase below
    that range rounds to zero; this does not change ln(P) or label P zero.
    Points with eta>0, outside the small-radius series, go to the compiled kernel when it is
    available, which runs the algorithm of `_charged_point`; any point it
    does not complete, and every other point, is computed by
    `_charged_point`, which computes it or raises.
    """
    if not isinstance(l,(int,np.integer)) or not 0<=l<=64:
        raise ValueError('charged orbital momentum must be L=0..64')
    eta,rho = np.broadcast_arrays(np.asarray(eta,dtype=float),np.asarray(rho,dtype=float))
    if np.any(~np.isfinite(eta+rho)) or np.any(eta<0) or np.any(rho<=0):
        raise ValueError('charged functions require finite eta>=0 and rho>0')
    p = np.empty_like(rho);s = np.empty_like(rho);phase = np.empty_like(rho)
    flat_eta,flat_rho = eta.ravel(),rho.ravel()
    pf,sf,phf = p.reshape(-1),s.reshape(-1),phase.reshape(-1)
    batch = np.flatnonzero((flat_eta>0)&((flat_rho>.25)|(flat_eta*flat_rho>2.)))
    single = np.ones(len(flat_rho),bool)
    if _native is not None and len(batch):
        lp,shift,ph,ok = _native_open_rows(l,flat_eta[batch],flat_rho[batch])
        done = batch[ok]
        pf[done],sf[done],phf[done] = lp[ok],shift[ok],ph[ok]
        single[done] = False
    for index in np.flatnonzero(single):
        pf[index],sf[index],phf[index] = _charged_point(l,float(flat_eta[index]),float(flat_rho[index]))
    return p,s,phase


def charged_channel_log_ladder(ls,eta,rho):
    """``{L: (ln(P), S, phase)}`` for every L of *ls* at the same eta, rho.

    One evaluation at the lowest L and the upward recurrence of the outgoing
    wave for the rest (``_coulomb_native.open_ladder``): channels of one pair
    and radius at several L (Ca-40: alpha at L=0,1,2) need one continued
    fraction instead of one each. Against 40-digit mpmath on Ca-40's rows
    the recurrence is within 1.2e-13 in ln(P) and S. Every point the kernel
    does not complete, and every point without it, is
    :func:`charged_channel_log_functions` at its own L.
    """
    ls=sorted({int(l) for l in ls})
    eta,rho=np.broadcast_arrays(np.asarray(eta,dtype=float),np.asarray(rho,dtype=float))
    if (_native is None or not hasattr(_native,'open_ladder') or eta.ndim!=1 or len(ls)<2
            or np.any(~np.isfinite(eta+rho)) or np.any(eta<0) or np.any(rho<=0)):
        return {l:charged_channel_log_functions(l,eta,rho) for l in ls}
    lmin=ls[0];nl=ls[-1]-lmin+1;n=len(rho)
    batch=np.flatnonzero((eta>0)&((rho>.25)|(eta*rho>2.)))
    log_p,shift,phase=(np.empty((nl,len(batch))) for _ in range(3));ok=np.zeros((nl,len(batch)),np.uint8)
    if len(batch):
        _native.open_ladder(lmin,nl,len(batch),np.ascontiguousarray(eta[batch]),np.ascontiguousarray(rho[batch]),
                            *_TABLEAU,log_p.ravel(),shift.ravel(),phase.ravel(),ok.ravel())
    out={}
    for l in ls:
        k=l-lmin;p,s,ph=np.empty(n),np.empty(n),np.empty(n)
        done=batch[ok[k].astype(bool)]
        p[done],s[done],ph[done]=log_p[k][ok[k].astype(bool)],shift[k][ok[k].astype(bool)],phase[k][ok[k].astype(bool)]
        rest=np.setdiff1d(np.arange(n),done,assume_unique=True)
        if len(rest):p[rest],s[rest],ph[rest]=charged_channel_log_functions(l,eta[rest],rho[rest])
        out[l]=(p,s,ph)
    return out


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
    flat_eta,flat_kappa,flat_out = eta.ravel(),kappa.ravel(),out.reshape(-1)
    single = np.ones(len(flat_out),bool)
    charged = np.flatnonzero(flat_eta>0)
    if _native is not None and len(charged):
        shift = np.empty(len(charged));ok = np.zeros(len(charged),np.uint8)
        _native.closed_rows(int(l),len(charged),np.ascontiguousarray(flat_eta[charged]),
                            np.ascontiguousarray(flat_kappa[charged]),shift,ok)
        done = ok.astype(bool)
        flat_out[charged[done]] = shift[done];single[charged[done]] = False
    for index in np.flatnonzero(single):
        h,r = float(flat_eta[index]),float(flat_kappa[index])
        if h==0:flat_out[index] = closed_neutral_shift(l,r);continue
        value = _closed_fraction(l,h,r)
        flat_out[index] = _closed_point(l,h,r) if value is None else value
    return out


#: Iterations beyond which the closed-channel fraction is not used: its
#: rounding grows with them (measured 1e-12 at about 750).
CLOSED_FRACTION_ITERATIONS = 500


def _closed_fraction(l,h,r):
    """The closed shift from Steed's fraction for H+'/H+ at rho=i*kappa, eta=-i*eta_c.

    There H+ is the decaying Whittaker function and every term of the
    fraction is real or imaginary, so it runs in real arithmetic. It
    converges slowly as kappa goes to zero; None beyond
    CLOSED_FRACTION_ITERATIONS, for the Laplace integral.
    """
    a = l+1+h; b = -l+h
    phi = 1+h/r; gamma = phi; delta = 0.; stable = 0
    for n in range(1,CLOSED_FRACTION_ITERATIONS+1):
        an = a*b/r if n==1 else (a+n-1)*(b+n-1)
        beta = 2*(r+h+n)
        delta = beta-an*delta; gamma = beta-an/gamma
        if abs(delta)<_TINY:delta = _TINY
        if abs(gamma)<_TINY:gamma = _TINY
        delta = 1/delta; change = gamma*delta; phi *= change
        stable = stable+1 if abs(change-1)<2e-14 else 0
        if stable>=3:
            return -r*phi if np.isfinite(phi) else None
    return None


def _closed_point(l,h,r):
    """closed_charged_shift at one point with eta>0, by adaptive quadrature."""
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
    return l+1-r-2*mode*moment/norm


def charged_threshold_shift(l,eta_rho):
    """Exact zero-energy repulsive limit; eta*rho remains finite."""
    value = np.asarray(eta_rho,dtype=float)
    if not isinstance(l,(int,np.integer)) or not 0<=l<=64 or np.any(value<=0) or np.any(~np.isfinite(value)):
        raise ValueError('threshold requires L=0..64 and positive finite eta*rho')
    z = np.sqrt(value)*np.sqrt(8.)
    # K_(n+1)/K_n = 2*n/z + K_(n-1)/K_n (DLMF 10.29).
    # Ratios stay finite when individual high-order scaled K values overflow.
    ratio = np.empty_like(z)
    large = z>1e6
    inverse = 1/z[large]
    ratio[large] = 1+.5*inverse-.125*inverse**2
    ratio[~large] = kve(1,z[~large])/kve(0,z[~large])
    for n in range(1,2*l+1):ratio = 2*n/z+1/ratio
    return -l-.5*z/ratio

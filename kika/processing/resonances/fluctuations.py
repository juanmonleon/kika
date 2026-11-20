"""Dilute URR width products from a scaled one-dimensional Laplace integral.

Independent gamma widths have real nu>0; nu=0 is deterministic. These are
Q_nc=<Gamma_n Gamma_c/Gamma_total>, not ratios of mean widths. No historical
ten-point quadrature, arbitrary-precision dependency or integer truncation.
"""
import warnings
import numpy as np
from scipy.integrate import quad,IntegrationWarning


def width_products(means,degrees,*,entrance=0,diagnostics=None,accelerated=True):
    """Return all Q_nc in the same units as finite nonnegative mean widths.

    ``Q_nc`` is the average of ``Gamma_n * Gamma_c / Gamma_total`` over the
    fluctuating widths, for the entrance channel ``n`` and each channel ``c``. The
    products add up to the entrance channel's mean width.

    Parameters
    ----------
    means
        One-dimensional array of mean widths, one per channel, finite and not
        negative, in any one unit (eV in the URR tables). A channel of mean zero is
        inactive and has a zero product.
    degrees
        Real degrees of freedom of each channel's width distribution, finite and
        not negative, same shape as *means*. Zero is a fixed width.
    entrance
        Index of the entrance channel in *means*.
    diagnostics : dict, optional
        If given, the keys ``urr_max_quadrature_relative_error``,
        ``urr_max_width_product_balance_error`` (running maxima) and
        ``urr_laplace_integrals`` (a running count) are updated in it.
    accelerated
        If True, the compiled integrand is used for the quadrature when it is
        available; the integral computed is the same.

    Returns
    -------
    numpy.ndarray
        ``Q_nc`` for every channel, in the unit of *means*, shape of *means*.
        All zeros if the entrance mean is zero; the entrance mean in the entrance
        channel and zero elsewhere if it is the only active channel; the
        proportional split ``means[entrance] * means / means.sum()`` if every
        active width is fixed.

    Raises
    ------
    ValueError
        If *means* is not one-dimensional, *degrees* differs in shape, *entrance*
        is out of range, or a value is not finite and non-negative.
    FloatingPointError
        If the ratio of the widths is too extreme for the quadrature, the
        quadrature does not converge or misses its error check (relative error
        above 1e-10), or the products do not add up to the entrance mean within a
        relative 2e-10.
    """
    m=np.asarray(means,dtype=float);nu=np.asarray(degrees,dtype=float)
    if (m.ndim!=1 or m.shape!=nu.shape or not 0<=entrance<len(m)
            or np.any(~np.isfinite(m)) or np.any(~np.isfinite(nu))
            or np.any(m<0) or np.any(nu<0)):
        raise ValueError('widths and real degrees of freedom must be finite and nonnegative')
    result=np.zeros_like(m)
    if m[entrance]==0:return result
    active=m>0
    if np.count_nonzero(active)==1:
        result[entrance]=m[entrance];return result
    scale=float(np.max(m));values=m/scale
    if np.any(active&(values==0)):
        raise FloatingPointError('width scale ratio is outside the finite quadrature range')
    if np.all(nu[active]==0):
        return m[entrance]*(values/np.sum(values))
    fluctuating=active&(nu>0);fixed=float(np.sum(values[active&~fluctuating]))
    log_theta=np.full(len(m),-np.inf)
    log_theta[fluctuating]=np.log(2.)+np.log(values[fluctuating])-np.log(nu[fluctuating])
    maximum=0.;integrals={}
    for c in np.flatnonzero(active):
        def integrand(s):
            logs=np.logaddexp(0.,s+log_theta[fluctuating])
            deterministic=0. if fixed==0 else np.exp(min(700.,s+np.log(fixed)))
            log_value=s-deterministic-float(np.sum(.5*nu[fluctuating]*logs))
            if fluctuating[entrance]:log_value-=np.logaddexp(0.,s+log_theta[entrance])
            if fluctuating[c]:log_value-=np.logaddexp(0.,s+log_theta[c])
            return np.exp(log_value)
        # Fixed exit widths share exactly the same Laplace integral.
        key=log_theta[c] if fluctuating[c] else None
        if key not in integrals:
            callback=integrand
            if accelerated:
                from ._rm_acceleration import _native
                if _native is not None and hasattr(_native,'URR_INTEGRAND'):
                    import ctypes
                    from scipy import LowLevelCallable
                    # Keep this contiguous owner alive through the synchronous
                    # quad call. Fixed widths have theta=-inf (zero softplus).
                    parameters=np.r_[np.count_nonzero(fluctuating),
                        -np.inf if fixed==0 else np.log(fixed),log_theta[entrance],
                        log_theta[c],np.column_stack((.5*nu[fluctuating],log_theta[fluctuating])).ravel()]
                    callback=LowLevelCallable(_native.URR_INTEGRAND,
                        ctypes.c_void_p(parameters.ctypes.data))
            # A fixed width cuts the integrand off doubly exponentially at
            # s=-log(fixed); across that edge quad's error estimate misses
            # 1e-9 (B-VIII.1 Xe-133: Gn/Gg=5.3, nu=1), so split there.
            cuts=[-np.inf,-np.log(fixed),np.inf] if fixed>0 else [-np.inf,np.inf]
            with warnings.catch_warnings():
                warnings.simplefilter('error',IntegrationWarning)
                try:pieces=[quad(callback,a,b,epsabs=0.,epsrel=2e-12,limit=250) for a,b in zip(cuts[:-1],cuts[1:])]
                except IntegrationWarning as exc:
                    raise FloatingPointError('URR Laplace quadrature did not converge') from exc
                integrals[key]=(sum(p[0] for p in pieces),sum(p[1] for p in pieces))
        value,error=integrals[key]
        if not np.isfinite(value+error) or value<=0 or error>1e-10*value:
            raise FloatingPointError('URR Laplace quadrature failed its error check')
        factor=np.logaddexp(0.,np.log(2.)-np.log(nu[entrance])) if c==entrance and fluctuating[entrance] else 0.
        result[c]=np.exp(np.log(scale)+np.log(values[entrance])+np.log(values[c])+factor+np.log(value))
        maximum=max(maximum,error/value)
    residual=abs(float(np.sum(result))/m[entrance]-1.)
    if np.any(~np.isfinite(result)) or residual>2e-10:
        raise FloatingPointError('URR width-product sum does not conserve the entrance mean')
    if diagnostics is not None:
        diagnostics['urr_max_quadrature_relative_error']=max(diagnostics.get('urr_max_quadrature_relative_error',0.),maximum)
        diagnostics['urr_max_width_product_balance_error']=max(diagnostics.get('urr_max_width_product_balance_error',0.),residual)
        diagnostics['urr_laplace_integrals']=diagnostics.get('urr_laplace_integrals',0)+len(integrals)
    return result

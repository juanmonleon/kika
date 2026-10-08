"""Dilute URR width products from a scaled one-dimensional Laplace integral.

Independent gamma widths have real nu>0; nu=0 is deterministic. These are
Q_nc=<Gamma_n Gamma_c/Gamma_total>, not ratios of mean widths. No historical
ten-point quadrature, arbitrary-precision dependency or integer truncation.
"""
import warnings
import numpy as np
from scipy.integrate import quad,IntegrationWarning


def width_products(means,degrees,*,entrance=0,diagnostics=None):
    """Return all Q_nc in the same units as finite nonnegative mean widths."""
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
            with warnings.catch_warnings():
                warnings.simplefilter('error',IntegrationWarning)
                try:integrals[key]=quad(integrand,-np.inf,np.inf,epsabs=0.,epsrel=2e-12,limit=250)
                except IntegrationWarning as exc:
                    raise FloatingPointError('URR Laplace quadrature did not converge') from exc
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

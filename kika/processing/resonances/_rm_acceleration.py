"""Optional all-level contraction; independent verification stays in NumPy.

No runtime compilation. Unavailable binaries or ineligible numeric regimes
return None to the reference solver. The extension uses bounded stack tiles;
channel solves and positive-capture acceptance checks remain in Python.
"""
import numpy as np
try:
    from . import _rm_native as _native
    if _native.API_VERSION != 1:_native=None
except (ImportError, AttributeError):
    _native=None


def eligible(energies, levels, widths, reduced, factors, work_bytes,
             return_absorption, absorption_rtol):
    if (_native is None or not return_absorption or absorption_rtol != 1e-8
            or reduced is None or work_bytes < 65536):return False
    e,er,g,a,f=(np.asarray(v) for v in (energies,levels,widths,reduced,factors))
    if (e.ndim!=1 or er.ndim!=1 or g.shape!=er.shape or a.ndim!=2
            or a.shape[0]!=len(er) or len(er)<32 or not 1<=a.shape[1]<=3
            or f.shape!=(len(e),a.shape[1]) or not len(e)):return False
    # Restrict acceleration to normal float64 arithmetic. Extreme data use
    # the reference kernel, including its established pole-limit handling.
    for v in (e,er,g,a,f):
        if v.dtype.kind not in 'fi' or not np.all(np.isfinite(v)) or np.any(abs(v)>1e20):return False
    if np.any(g<0) or np.any((g>0)&(g<1e-100)):return False
    # With |E|, |Er|, gamma <= 1e20 and positive gamma >= 1e-100,
    # Im(1/d) >= 1e-141. Nonzero amplitudes and factors >= 1e-40
    # keep every positive diagonal summand (including factor scaling)
    # above 1e-301, safely normal. Extreme products use direct NumPy.
    for v in (a,f):
        if np.any((v!=0)&(abs(v)<1e-40)):return False
    return True


def reference_block_size(n,c,work_bytes):
    """Bound fallback level arrays with the reference workspace estimate."""
    return max(1,int((work_bytes-32*n*c*c)//(128*(n+c*c+1))))


def solve(energies,levels,radiative_widths,amplitudes,diagnostics=None,*,entrance=0,
          reduced=None,channel_factors=None,work_bytes=64*1024**2,
          return_absorption=False,absorption_rtol=0.):
    if not eligible(energies, levels, radiative_widths, reduced, channel_factors,
                    work_bytes, return_absorption, absorption_rtol):
        return None
    target_diagnostics=diagnostics
    if diagnostics is not None:diagnostics=dict(diagnostics)
    from .reich_moore import solve_collision as reference
    e,er,gamma,a,f=(np.require(v,dtype=float,requirements=['C','A']) for v in
                    (energies,levels,radiative_widths,reduced,channel_factors))
    if np.any(gamma<0):raise ValueError('negative radiative width')
    c=a.shape[1];r=np.empty((len(e),c,c),complex);poles=np.zeros(len(e),np.uint8)
    _native.matrices(len(e),len(er),c,e,er,gamma,a.ravel(),f.ravel(),r.view(float).ravel(),poles)
    regular=poles==0
    w=np.zeros((len(e),c),complex);capture=np.zeros(len(e))
    if np.any(regular):
        rr=r[regular];matrix=np.eye(c)[None,:,:]-1j*rr
        rhs=np.zeros((len(rr),c,1),complex);rhs[:,entrance,0]=1.
        try:y=np.ascontiguousarray((1/matrix[:,0,0])[:,None] if c==1 else np.linalg.solve(matrix,rhs)[:,:,0])
        except np.linalg.LinAlgError:return None
        if np.any(np.linalg.norm(y,axis=1)>1+1e-8):return None
        residual=matrix@y[:,:,None]-rhs
        maximum=float(np.max(abs(residual)/(np.linalg.norm(matrix,axis=(1,2))[:,None,None]*np.linalg.norm(y,axis=1)[:,None,None]+1.)))
        if maximum>1e-11:return None
        w[regular]=rr[:,0,:]*y if c==1 else np.einsum('ecd,ed->ec',rr,y)
        b=rr.imag;diag=np.diagonal(b,axis1=1,axis2=2)
        cap=(4*np.real(y[:,0].conj()*(b[:,0,0]*y[:,0])) if c==1 else
             4*np.real(np.sum(y.conj()*np.einsum('eij,ej->ei',b,y),axis=1)))
        majorant=(4*(abs(y[:,0])*np.sqrt(np.maximum(diag[:,0],0.)))**2 if c==1 else
                  4*np.sum(abs(y)*np.sqrt(np.maximum(diag,0.)),axis=1)**2)
        eps=np.finfo(float).eps;tiny=np.finfo(float).tiny
        scale=8*(len(er)+32*c+64)*eps
        bound=scale/(1-scale)*majorant if scale<1 else np.full(len(y),np.inf)
        radiative=gamma>0;active=np.any((a!=0)&radiative[:,None],axis=0)
        normal=np.all((diag>=tiny)|~active[None,:],axis=1)
        normal &= np.all((f[regular]*f[regular]>=tiny)|(f[regular]==0),axis=1)
        normal &= np.all((a*a>=tiny)|(a==0)) and np.all((gamma>=tiny)|~radiative)
        if np.any(radiative):
            distance=np.maximum(abs(e[regular]-er.min()),abs(e[regular]-er.max()))
            with np.errstate(over='ignore',divide='ignore'):
                upper=(distance*distance+(.5*gamma.max())**2)*(1+8*eps)
                lower=np.nextafter(.5*gamma[radiative].min()/upper,0.)
            normal &= lower>=tiny
        accepted=normal&np.isfinite(cap)&(cap>=tiny)&(majorant>=tiny)&(bound<=absorption_rtol*cap)&(bound<=1e-10)
        if not np.any(radiative):accepted[:]=True;cap[:]=0.
        fallback=~accepted
        if np.any(fallback):
            direct=np.zeros(int(fallback.sum()))
            _native.absorption(len(direct),len(er),c,np.ascontiguousarray(e[regular][fallback]),er,gamma,
                a.ravel(),np.ascontiguousarray(f[regular][fallback]).ravel(),np.ascontiguousarray(y[fallback]).view(float).ravel(),direct)
            cap[fallback]=direct
        if diagnostics is not None:
            diagnostics['rm_native_gram_energies']=diagnostics.get('rm_native_gram_energies',0)+int(accepted.sum())
            diagnostics['rm_native_direct_energies']=diagnostics.get('rm_native_direct_energies',0)+int(fallback.sum())
            diagnostics['rm_gram_capture_energies']=diagnostics.get('rm_gram_capture_energies',0)+int(accepted.sum())
            diagnostics['rm_direct_capture_energies']=diagnostics.get('rm_direct_capture_energies',0)+int(fallback.sum())
            if np.any(accepted) and np.any(radiative):
                diagnostics['rm_max_capture_relative_roundoff_estimate']=max(
                    diagnostics.get('rm_max_capture_relative_roundoff_estimate',0.),
                    float(np.max(bound[accepted]/cap[accepted])))
        capture[regular]=cap
        if diagnostics is not None:diagnostics['rm_max_solver_residual']=max(diagnostics.get('rm_max_solver_residual',0.),maximum)
    if np.any(poles):
        indices=np.flatnonzero(~regular)
        block=reference_block_size(len(er),c,work_bytes)
        for start in range(0,len(indices),block):
            sl=indices[start:start+block]
            w[sl],capture[sl]=reference(e[sl],er,gamma,None,diagnostics,entrance=entrance,reduced=a,
                channel_factors=f[sl],work_bytes=work_bytes,return_absorption=True)
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(capture)) or np.any(capture<0):
        return None
    flux=1-abs(1+2j*w[:,entrance])**2-4*(np.sum(abs(w)**2,axis=1)-abs(w[:,entrance])**2)
    if np.any(abs(flux-capture)>1e-9):return None
    if diagnostics is not None:
        diagnostics['rm_native_energies']=diagnostics.get('rm_native_energies',0)+len(e)
        target_diagnostics.update(diagnostics)
    return w,capture

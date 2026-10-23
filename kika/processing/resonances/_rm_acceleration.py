"""Optional compiled Reich-Moore all-level sums.

Production (``absorption_rtol=1e-8``) also accepts the channel-space Gram
capture where its roundoff bound allows. Publication verification
(``absorption_rtol=0`` with ``native=True``) uses the same sums with direct
capture at every energy, the exact quantities of the NumPy reference solver,
which the tests hold this path to.

No runtime compilation. Unavailable binaries or ineligible numeric regimes
return None to the reference solver. The extension uses bounded stack tiles;
Candidate small solves use pivoted C elimination when available; independent
residual/passivity and positive-capture acceptance checks remain in Python.
"""
import contextvars
import numpy as np
from kika._native import load as _load_native
_native = _load_native(__package__ + "._rm_native", 1)

#: Channels per spin group the compiled sums take (``RM_MAX_CHANNELS``).
MAX_CHANNELS=6

#: Threads for the level sums, set by ``reconstruct_suite`` from its
#: ``workers`` for the length of one call. Rows are independent, so the
#: numbers are those of one thread.
level_threads=contextvars.ContextVar('kika_rm_level_threads',default=1)
#: Level x energy x channel-pair products of work per extra thread: below
#: about a millisecond, starting a thread costs more than it saves
#: (measured: c=1, 300 levels, 900 energies is slower on two threads).
THREAD_WORK=2_500_000


def threads_for(energies,levels,channels):
    threads=level_threads.get()
    if threads<=1:return 1
    return max(1,min(threads,energies*levels*channels*(channels+1)//2//THREAD_WORK))


#: Facts about a group's level arrays, which do not change between calls:
#: keyed by the arrays themselves, kept only for read-only arrays (a prepared
#: group's kernel arrays are), so an array edited in place is never served.
_level_facts_cache={}


def _level_facts(er,g,a):
    """Eligibility of the level arrays and the per-group constants of
    :func:`gram_capture` (active channels, normal-range test, width extrema),
    made once per group instead of once per call."""
    cacheable=all(isinstance(v,np.ndarray) and not v.flags.writeable for v in (er,g,a))
    key=(id(er),id(g),id(a))
    if cacheable:
        hit=_level_facts_cache.get(key)
        if hit is not None and hit[0] is er and hit[1] is g and hit[2] is a:return hit[3]
    ok=(er.ndim==1 and g.shape==er.shape and a.ndim==2 and a.shape[0]==len(er) and len(er)>0
        and 1<=a.shape[1]<=MAX_CHANNELS)
    if ok:
        for v in (er,g,a):
            if v.dtype.kind not in 'fi' or not np.all(np.isfinite(v)) or np.any(abs(v)>1e20):ok=False;break
    if ok and (np.any(g<0) or np.any((g>0)&(g<1e-100))):ok=False
    if ok and np.any((a!=0)&(abs(a)<1e-40)):ok=False
    facts=dict(ok=ok)
    if ok:
        tiny=np.finfo(float).tiny;radiative=g>0
        facts.update(radiative=radiative,
            active=np.ascontiguousarray(np.any((a!=0)&radiative[:,None],axis=0),dtype=np.uint8),
            scalar_ok=bool(np.all((a*a>=tiny)|(a==0)) and np.all((g>=tiny)|~radiative)),
            any_radiative=bool(np.any(radiative)),
            halfgmax2=float((.5*g.max())**2) if np.any(radiative) else 0.,
            hgmin=float(.5*g[radiative].min()) if np.any(radiative) else 0.,
            ermin=float(er.min()),ermax=float(er.max()))
    if cacheable:
        if len(_level_facts_cache)>512:_level_facts_cache.clear()
        _level_facts_cache[key]=(er,g,a,facts)
    return facts


def eligible(energies, levels, widths, reduced, factors, work_bytes,
             return_absorption, absorption_rtol):
    # 1e-8: production, Gram capture where its roundoff bound allows.
    # 0: exact, direct capture at every energy (publication verification).
    if (_native is None or not return_absorption or absorption_rtol not in (0., 1e-8)
            or reduced is None or work_bytes < 65536):return False
    e,er,g,a,f=(np.asarray(v) for v in (energies,levels,widths,reduced,factors))
    # Restrict acceleration to normal float64 arithmetic. Extreme data use
    # the reference kernel, including its established pole-limit handling.
    # With |E|, |Er|, gamma <= 1e20 and positive gamma >= 1e-100,
    # Im(1/d) >= 1e-141. Nonzero amplitudes and factors >= 1e-40
    # keep every positive diagonal summand (including factor scaling)
    # above 1e-301, safely normal. Extreme products use direct NumPy.
    if not _level_facts(er,g,a)['ok']:return False
    if e.ndim!=1 or f.shape!=(len(e),a.shape[1]) or not len(e):return False
    for v in (e,f):
        if v.dtype.kind not in 'fi' or not np.all(np.isfinite(v)) or np.any(abs(v)>1e20):return False
    if np.any((f!=0)&(abs(f)<1e-40)):return False
    return True


def reference_block_size(n,c,work_bytes):
    """Bound fallback level arrays with the reference workspace estimate."""
    return max(1,int((work_bytes-32*n*c*c)//(128*(n+c*c+1))))


def channel_solution(matrix,rhs,entrance):
    """Optional pivoted small solve; unsupported/older binaries use LAPACK.

    Callers must still apply their residual, finite-value and physical checks.
    The ordinary independent NumPy solvers never call this function.
    """
    c=matrix.shape[1]
    if c in (2,3) and _native is not None and hasattr(_native,'channel_solve'):
        y=np.empty((len(matrix),c),complex)
        if _native.channel_solve(len(matrix),c,entrance,
                np.ascontiguousarray(matrix).view(float).ravel(),y.view(float).ravel()):return y
    return np.ascontiguousarray(np.linalg.solve(matrix,rhs)[:,:,0])


def gram_capture(r,y,e,er,gamma,a,f,absorption_rtol):
    """Radiative capture 4 Re(y* Im(R) y) and the rows where it is accepted.

    ``r`` is the level sum of :func:`_native.matrices` with the factors ``f``
    applied and nothing else added, so Im(R) is the positive Gram sum of the
    radiative level amplitudes: 4 y* Im(R) y = 2 sum_n gamma_n |X_n|^2, the
    direct capture of ``_native.absorption``. Signed off-diagonal terms can
    cancel, so a row is accepted only under a conservative roundoff estimate
    ``bound``; the caller computes the others directly.
    """
    c=r.shape[1]
    b=r.imag;diag=np.diagonal(b,axis1=1,axis2=2)
    cap=(4*np.real(y[:,0].conj()*(b[:,0,0]*y[:,0])) if c==1 else
         4*np.real(np.sum(y.conj()*np.einsum('eij,ej->ei',b,y),axis=1)))
    majorant=(4*(abs(y[:,0])*np.sqrt(np.maximum(diag[:,0],0.)))**2 if c==1 else
              4*np.sum(abs(y)*np.sqrt(np.maximum(diag,0.)),axis=1)**2)
    eps=np.finfo(float).eps;tiny=np.finfo(float).tiny
    scale=8*(len(er)+32*c+64)*eps
    bound=scale/(1-scale)*majorant if scale<1 else np.full(len(y),np.inf)
    radiative=gamma>0;active=np.any((a!=0)&radiative[:,None],axis=0)
    normal=np.all((diag>=tiny)|~active[None,:],axis=1)
    normal &= np.all((f*f>=tiny)|(f==0),axis=1)
    normal &= np.all((a*a>=tiny)|(a==0)) and np.all((gamma>=tiny)|~radiative)
    if np.any(radiative):
        distance=np.maximum(abs(e-er.min()),abs(e-er.max()))
        with np.errstate(over='ignore',divide='ignore'):
            upper=(distance*distance+(.5*gamma.max())**2)*(1+8*eps)
            lower=np.nextafter(.5*gamma[radiative].min()/upper,0.)
        normal &= lower>=tiny
    accepted=normal&np.isfinite(cap)&(cap>=tiny)&(majorant>=tiny)&(bound<=absorption_rtol*cap)&(bound<=1e-10)
    if absorption_rtol==0.:accepted[:]=False
    if not np.any(radiative):accepted[:]=True;cap[:]=0.
    return cap,accepted,bound


def _fused_rows(rr,f,e,er,gamma,a,c,entrance,absorption_rtol,radiative):
    """The channel solve, residual, W and :func:`gram_capture` of the regular
    rows in one compiled pass (``_native.solve_rows``, c <= 3), with the same
    operations in the same order. None: take the NumPy statements; False: a
    non-passive solution, which the caller rejects as it always has."""
    if c>3 or _native is None or not hasattr(_native,'solve_rows'):return None
    facts=_level_facts(er,gamma,a)
    n=len(rr);y=np.empty((n,c),complex);w=np.empty((n,c),complex)
    cap=np.empty(n);accepted=np.empty(n,np.uint8);bound=np.empty(n)
    status,maximum=_native.solve_rows(n,c,entrance,len(er),rr.view(float).ravel(),f.ravel(),e,facts['active'],
        y.view(float).ravel(),w.view(float).ravel(),cap,accepted,bound,
        float(absorption_rtol),facts['ermin'],facts['ermax'],facts['halfgmax2'],facts['hgmin'],
        int(facts['scalar_ok']),int(facts['any_radiative']))
    if status==2:return False
    if status:return None
    return y,w,cap,accepted.astype(bool),bound,maximum


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
    _native.matrices(len(e),len(er),c,e,er,gamma,a.ravel(),f.ravel(),r.view(float).ravel(),poles,
                     threads_for(len(e),len(er),c))
    regular=poles==0
    w=np.zeros((len(e),c),complex);capture=np.zeros(len(e))
    if np.any(regular):
        rr=np.ascontiguousarray(r[regular]);fr=np.ascontiguousarray(f[regular]);er_rows=np.ascontiguousarray(e[regular])
        radiative=gamma>0
        fused=_fused_rows(rr,fr,er_rows,er,gamma,a,c,entrance,absorption_rtol,radiative)
        if fused is False:return None
        if fused is not None:
            y,w_rows,cap,accepted,bound,maximum=fused
            if maximum>1e-11:return None
            w[regular]=w_rows
        else:
            matrix=np.eye(c)[None,:,:]-1j*rr
            rhs=np.zeros((len(rr),c,1),complex);rhs[:,entrance,0]=1.
            try:
                if c==1:y=np.ascontiguousarray((1/matrix[:,0,0])[:,None])
                else:y=channel_solution(matrix,rhs,entrance)
            except np.linalg.LinAlgError:return None
            if np.any(np.linalg.norm(y,axis=1)>1+1e-8):return None
            residual=matrix@y[:,:,None]-rhs
            maximum=float(np.max(abs(residual)/(np.linalg.norm(matrix,axis=(1,2))[:,None,None]*np.linalg.norm(y,axis=1)[:,None,None]+1.)))
            if maximum>1e-11:return None
            w[regular]=rr[:,0,:]*y if c==1 else np.einsum('ecd,ed->ec',rr,y)
            cap,accepted,bound=gram_capture(rr,y,e[regular],er,gamma,a,f[regular],absorption_rtol)
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

"""Neutral Reich-Moore algebra with eliminated radiative channels.

L-B is zero for the certified RM convention. Width amplitudes include sqrt(P).
Solves channel systems away from naked poles and an augmented channel/level
system at poles. No inverse, epsilon width, energy displacement or clipping.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .channel_functions import neutral_channel_functions
from .breit_wigner import Level, Group
from .grid import check_dense_workspace


def level_matrix(reciprocal,reduced):
    """Exact symmetric channel sum using real BLAS for real amplitudes.

    Only the independent symmetric products are contracted. Splitting the
    complex reciprocal into its two real components avoids promoting the
    real coefficient matrix to complex, and does not omit any level.
    """
    a=np.asarray(reduced);c=a.shape[1]
    row,column=np.triu_indices(c)
    products=a[:,row]*a[:,column]
    values=np.empty((len(reciprocal),len(row)),complex)
    values.real=reciprocal.real@products
    values.imag=reciprocal.imag@products
    result=np.empty((len(reciprocal),c,c),complex)
    result[:,row,column]=values
    result[:,column,row]=values
    return result


def level_excitation(excitation,reduced):
    """Contract complex channel excitation with real level amplitudes."""
    if len(reduced)<32:return excitation@reduced.T
    result=np.empty((len(excitation),len(reduced)),complex)
    result.real=excitation.real@reduced.T
    result.imag=excitation.imag@reduced.T
    return result


def _single_channel_absorption(reciprocal, excitation, amplitudes, gamma):
    """Direct positive all-level sum, factored only for one open channel.

    The common channel excitation factors out of the independent NumPy
    reference's positive sum. No channel Gram cancellation or flux subtraction
    is involved. Conservative normal-range guards retain the level-amplitude
    calculation when regrouping could underflow or overflow.
    """
    # Widths supplied directly to the numerical kernel may be integer arrays.
    # Promote before doubling, rather than overflowing their integer dtype.
    gamma=np.asarray(gamma,dtype=float)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        weights=2*gamma*amplitudes**2
        scale=np.abs(excitation)**2
        squared=reciprocal.real**2
        squared+=reciprocal.imag**2
    active=(gamma>0)&(amplitudes!=0)
    if not np.any(active):return np.zeros(len(reciprocal))
    if (np.any(~np.isfinite(weights)) or np.any(weights[active]<1e-80)
            or np.any(weights>1e80) or np.any(scale<1e-80) or np.any(scale>1e80)
            or np.any(~np.isfinite(scale))):return None
    # The large level array needs only two reductions, without allocating
    # three separate boolean masks. NaN extrema fail the chained comparison.
    if not 1e-80<=np.min(squared)<=np.max(squared)<=1e80:return None
    value=(squared@weights)*scale
    if np.any(~np.isfinite(value)):return None
    return value


@dataclass(frozen=True)
class RMLevel(Level):
    fission_amplitudes: tuple[float, ...] = ()
    neutron_amplitude: float = 0.


@dataclass(frozen=True)
class RMGroup(Group):
    spin: float = 0.
    channel_spin: float = .5

    def __getstate__(self):
        # NumPy's deepcopy/pickle makes read-only arrays writable. Rebuild
        # the private coefficient cache lazily from the immutable levels.
        return {key:value for key,value in self.__dict__.items() if key!='kernel_data'}

    def __setstate__(self,state):
        self.__dict__.update({key:value for key,value in state.items() if key!='kernel_data'})

    @cached_property
    def kernel_data(self):
        """Immutable coefficient arrays from the immutable prepared levels."""
        energies=np.array([level.energy for level in self.levels])
        gamma=np.array([level.capture for level in self.levels])
        neutron=np.array([level.neutron_amplitude for level in self.levels])
        fission=np.array([level.fission_amplitudes for level in self.levels])
        if not self.levels:fission=np.empty((0,0))
        fission=fission[:,np.any(fission!=0,axis=0)]
        amplitudes=np.column_stack((neutron,fission))
        # A read-only flag alone can be reversed by the consumer. Bytes own
        # these snapshots, so callers cannot edit the prepared physics.
        freeze=lambda a:np.frombuffer(np.asarray(a,dtype=float).tobytes(),dtype=float).reshape(a.shape)
        return tuple(freeze(a) for a in (energies,gamma,amplitudes))


def solve_collision(energies, levels, radiative_widths, amplitudes, diagnostics=None, *, entrance=0,
                    reduced=None,channel_factors=None,work_bytes=64*1024**2,
                    return_absorption=False,absorption_rtol=0.):
    """Return W[:, :, entrance] and absorption-level amplitudes.

    amplitudes: (energy, level, open channel), with entrance at index zero.
    The kernel is independent of resonance-model and format classes. Identical
    undamped poles may have dark states: their minimum-norm limit is accepted
    only if the linear-system residual is small.
    With return_absorption, the second output is the positive radiative
    probability instead of level amplitudes. A guarded channel-space Gram
    contraction can avoid materializing the latter; rejected roundoff
    estimates and all naked poles retain the direct level calculation.
    """
    e=np.asarray(energies)
    if not np.isfinite(absorption_rtol) or not 0<=absorption_rtol<=1e-8:
        raise ValueError('absorption_rtol must be finite in [0, 1e-8]')
    if return_absorption and np.any(np.asarray(radiative_widths)<0):
        raise ValueError('absorption requires nonnegative radiative widths')
    separable=reduced is not None
    if separable:
        reduced=np.asarray(reduced,dtype=float);factors=np.asarray(channel_factors,dtype=float)
        n,c=reduced.shape
        if factors.shape!=(len(e),c):raise ValueError('invalid separable RM channel dimensions')
        a=None
    else:
        a=np.asarray(amplitudes);n,c=a.shape[1:]
    if not isinstance(entrance,int) or not 0<=entrance<c:raise ValueError('invalid entrance channel')
    if separable and return_absorption and absorption_rtol == 1e-8:
        from ._rm_acceleration import solve as accelerated_solve,reference_block_size
        accelerated=accelerated_solve(e,levels,radiative_widths,None,diagnostics,
            entrance=entrance,reduced=reduced,channel_factors=factors,
            work_bytes=work_bytes,return_absorption=True,absorption_rtol=absorption_rtol)
        if accelerated is not None:return accelerated
        # Streaming production batches may be larger than reference batches.
        # Ineligible factors or a rejected native solve must not restore a
        # large energy x level temporary when falling back to NumPy.
        block=reference_block_size(n,c,work_bytes)
        if len(e)>block:
            w=np.empty((len(e),c),complex);capture=np.empty(len(e))
            for start in range(0,len(e),block):
                sl=slice(start,start+block)
                w[sl],capture[sl]=solve_collision(e[sl],levels,radiative_widths,None,diagnostics,
                    entrance=entrance,reduced=reduced,channel_factors=factors[sl],
                    work_bytes=work_bytes,return_absorption=True,absorption_rtol=0.)
            return w,capture
    d=np.asarray(levels)[None,:]-e[:,None]-.5j*np.asarray(radiative_widths)[None,:]
    w=np.zeros((len(e),c),complex)
    x=np.zeros(len(e)) if return_absorption else np.zeros((len(e),n),complex)
    if not n or not len(e):return w,x
    # Near naked poles retain level unknowns rather than first dividing by d.
    if separable:
        # Bound the positive pole-test threshold outwards. All levels still
        # enter the collision matrix, including levels excluded from this test.
        upper=(reduced*reduced)@np.max(factors*factors,axis=0)
        upper=np.nextafter(upper*(1+16*c*np.finfo(float).eps),np.inf)
        candidates=np.flatnonzero((.5*abs(np.asarray(radiative_widths))<=1e-6*upper)|
                                  (np.asarray(radiative_widths)==0))
        strength=(factors*factors)@(reduced[candidates]*reduced[candidates]).T
        candidate_poles=(abs(d[:,candidates])<=1e-6*strength)|(d[:,candidates]==0)
        regular=~np.any(candidate_poles,axis=1)
    else:
        strength=np.sum(a*a,axis=2)
        poles=(np.abs(d)<=1e-6*strength) | (d==0)
        regular=~np.any(poles,axis=1)
    maximum=0.;singular=0
    if np.any(regular):
        dr=d if np.all(regular) else d[regular]
        if separable:
            reciprocal=1/dr
            r=level_matrix(reciprocal,reduced)
            f=factors[regular]
            r*=f[:,:,None]*f[:,None,:]
        else:
            ar=a[regular]
            r=np.einsum('enc,en,end->ecd',ar,1/dr,ar,optimize=True)
        matrix=np.eye(c)[None,:,:]-1j*r
        rhs=np.zeros((len(r),c,1),complex);rhs[:,entrance,0]=1.
        y=np.linalg.solve(matrix,rhs)[:,:,0]
        residual=matrix@y[:,:,None]-rhs
        maximum=float(np.max(np.abs(residual)/(np.linalg.norm(matrix,axis=(1,2))[:,None,None]*np.linalg.norm(y,axis=1)[:,None,None]+1.)))
        w[regular]=np.einsum('ecd,ed->ec',r,y)
        if return_absorption and separable and absorption_rtol>0:
            # Im(R) is the positive Gram sum of radiative level amplitudes:
            # 4 y* Im(R) y = 2 sum_n gamma_n |X_n|^2. It is NOT a flux
            # subtraction. Signed off-diagonal terms can cancel, so accept
            # this contraction only under a conservative roundoff estimate.
            gamma=np.asarray(radiative_widths)
            b=r.imag;diag=np.diagonal(b,axis1=1,axis2=2)
            value=4*np.real(np.sum(y.conj()*np.einsum('eij,ej->ei',b,y),axis=1))
            majorant=4*np.sum(abs(y)*np.sqrt(np.maximum(diag,0.)),axis=1)**2
            eps=np.finfo(float).eps;tiny=np.finfo(float).tiny
            scale=8*(n+32*c+64)*eps
            bound=scale/(1-scale)*majorant if scale<1 else np.full(len(y),np.inf)
            radiative=gamma>0
            active=np.any((reduced!=0)&radiative[:,None],axis=0)
            normal=np.all((diag>=tiny)|~active[None,:],axis=1)
            # The usual relative-roundoff model excludes underflowed products.
            normal &= np.all((reciprocal.imag>=tiny)|~radiative[None,:],axis=1)
            normal &= np.all((f*f>=tiny)|(f==0),axis=1)
            squares=reduced*reduced
            normal &= np.all((squares>=tiny)|(reduced==0)) and np.all((gamma>=tiny)|~radiative)
            accepted=normal & np.isfinite(value)&(value>=tiny)&(majorant>=tiny)&(bound<=absorption_rtol*value)&(bound<=1e-10)
            if not np.any(radiative):accepted[:]=True;value[:]=0.
            absorption=value.copy()
            fallback=~accepted
            if np.any(fallback):
                excitation=level_excitation(y[fallback]*f[fallback],reduced)/dr[fallback]
                absorption[fallback]=2*np.sum(gamma[None,:]*abs(excitation)**2,axis=1)
            x[regular]=absorption
            if diagnostics is not None:
                diagnostics['rm_gram_capture_energies']=diagnostics.get('rm_gram_capture_energies',0)+int(accepted.sum())
                diagnostics['rm_direct_capture_energies']=diagnostics.get('rm_direct_capture_energies',0)+int(fallback.sum())
                if np.any(accepted) and np.any(radiative):
                    diagnostics['rm_max_capture_relative_roundoff_estimate']=max(diagnostics.get('rm_max_capture_relative_roundoff_estimate',0.),float(np.max(bound[accepted]/value[accepted])))
        else:
            direct=(_single_channel_absorption(reciprocal,(y*f)[:,0],reduced[:,0],np.asarray(radiative_widths))
                    if return_absorption and separable and c==1 else None)
            if direct is not None:x[regular]=direct
            else:
                excitation=(level_excitation(y*f,reduced) if separable else np.einsum('enc,ec->en',ar,y))/dr
                x[regular]=2*np.sum(np.asarray(radiative_widths)[None,:]*abs(excitation)**2,axis=1) if return_absorption else excitation
    for index in np.flatnonzero(~regular):
        if separable:
            pole=np.zeros(n,bool);pole[candidates]=candidate_poles[index]
        else:pole=poles[index]
        other=~pole
        local=reduced*factors[index] if separable else a[index]
        ap=local[pole];ao=local[other]
        r=(ao.T/d[index,other])@ao
        count=len(ap)
        check_dense_workspace(c+count,work_bytes)
        matrix=np.block([[np.eye(c)-1j*r,-1j*ap.T],[-ap,np.diag(d[index,pole])]])
        rhs=np.zeros(c+count,complex);rhs[entrance]=1.
        try:solution=np.linalg.solve(matrix,rhs)
        except np.linalg.LinAlgError:
            solution=np.linalg.lstsq(matrix,rhs,rcond=None)[0];singular+=1
        residual=np.linalg.norm(matrix@solution-rhs)/(np.linalg.norm(matrix)*np.linalg.norm(solution)+1.)
        maximum=max(maximum,float(residual))
        y=solution[:c];xp=solution[c:]
        w[index]=r@y+ap.T@xp
        if return_absorption:
            excitation=np.empty(n,complex)
            excitation[other]=ao@y/d[index,other];excitation[pole]=xp
            x[index]=2*np.sum(np.asarray(radiative_widths)*abs(excitation)**2)
        else:
            x[index,other]=ao@y/d[index,other];x[index,pole]=xp
    if maximum>1e-11 or np.any(~np.isfinite(w)) or np.any(~np.isfinite(x)):
        raise FloatingPointError('RM linear solve failed residual/finite-value check')
    if diagnostics is not None:
        diagnostics['rm_max_solver_residual']=max(diagnostics.get('rm_max_solver_residual',0.),maximum)
        diagnostics['rm_singular_pole_limits']=diagnostics.get('rm_singular_pole_limits',0)+singular
    return w,x


def evaluate_rm(energies,groups,context,diagnostics=None,*,work_bytes=64*1024**2,absorption_rtol=0.):
    elastic,capture,fission=(np.zeros_like(energies) for _ in range(3))
    for group in groups:
        ctx=group.context or context;k2=ctx.k_squared_per_ev*energies
        beta=np.pi*.01/k2
        phase_functions=None if group.phase_radius.constant==0. else neutral_channel_functions(group.l,np.sqrt(k2)*group.phase_radius.evaluate(energies))
        phi=np.zeros_like(energies) if phase_functions is None else phase_functions[2]
        g=(2*group.spin+1)/(2*(2*ctx.target_spin+1))
        if not group.levels:
            elastic+=beta*g*4*np.sin(phi)**2;continue
        level_energies,gamma,amplitudes=group.kernel_data
        reduced=amplitudes[:,0]
        p=(phase_functions[0] if phase_functions is not None and group.phase_radius==group.channel_radius else
           neutral_channel_functions(group.l,np.sqrt(k2)*group.channel_radius.evaluate(energies),phase=False)[0])
        if np.any(p==0) and np.any(reduced):raise FloatingPointError('RM neutron penetrability underflows')
        factors=np.ones((len(energies),amplitudes.shape[1]));factors[:,0]=np.sqrt(p)
        w,absorption=solve_collision(energies,level_energies,gamma,None,diagnostics,
            reduced=amplitudes,channel_factors=factors,work_bytes=work_bytes,
            return_absorption=True,absorption_rtol=absorption_rtol)
        # Stable 1-U_nn, avoiding the cancellation of hard-sphere phase.
        amplitude=2j*np.exp(-1j*phi)*np.sin(phi)-2j*np.exp(-2j*phi)*w[:,0]
        elastic+=beta*g*np.abs(amplitude)**2
        capture+=beta*g*absorption
        fission+=beta*g*4*np.sum(np.abs(w[:,1:])**2,axis=1)
        flux=1-np.abs(1+2j*w[:,0])**2-4*np.sum(np.abs(w[:,1:])**2,axis=1)
        flux_error=float(np.max(abs(flux-absorption)))
        if flux_error>1e-9:raise FloatingPointError('RM collision matrix fails flux balance')
        if diagnostics is not None:
            diagnostics['rm_max_absolute_flux_error']=max(diagnostics.get('rm_max_absolute_flux_error',0.),flux_error)
    result={2:elastic,102:capture,18:fission,1:elastic+capture+fission}
    if any(np.any(~np.isfinite(v)) for v in result.values()):raise FloatingPointError('nonfinite RM cross sections')
    return result

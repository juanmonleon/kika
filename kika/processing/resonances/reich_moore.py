"""Neutral Reich-Moore algebra with eliminated radiative channels.

L-B is zero for the certified RM convention. Width amplitudes include sqrt(P).
Solves channel systems away from naked poles and an augmented channel/level
system at poles. No inverse, epsilon width, energy displacement or clipping.
"""
from dataclasses import dataclass
import numpy as np
from .channel_functions import neutral_channel_functions
from .breit_wigner import Level, Group


@dataclass(frozen=True)
class RMLevel(Level):
    fission_amplitudes: tuple[float, ...] = ()
    neutron_amplitude: float = 0.


@dataclass(frozen=True)
class RMGroup(Group):
    spin: float = 0.
    channel_spin: float = .5


def solve_collision(energies, levels, radiative_widths, amplitudes, diagnostics=None, *, entrance=0):
    """Return W[:, :, entrance] and absorption-level amplitudes.

    amplitudes: (energy, level, open channel), with entrance at index zero.
    The kernel is independent of resonance-model and format classes. Identical
    undamped poles may have dark states: their minimum-norm limit is accepted
    only if the linear-system residual is small.
    """
    e=np.asarray(energies);a=np.asarray(amplitudes)
    n,c=a.shape[1:]
    if not isinstance(entrance,int) or not 0<=entrance<c:raise ValueError('invalid entrance channel')
    d=np.asarray(levels)[None,:]-e[:,None]-.5j*np.asarray(radiative_widths)[None,:]
    w=np.zeros((len(e),c),complex);x=np.zeros((len(e),n),complex)
    if not n:return w,x
    # Near naked poles retain level unknowns rather than first dividing by d.
    strength=np.sum(a*a,axis=2)
    poles=(np.abs(d)<=1e-6*strength) | (d==0)
    regular=~np.any(poles,axis=1)
    maximum=0.;singular=0
    if np.any(regular):
        ar=a[regular];dr=d[regular]
        r=np.einsum('enc,en,end->ecd',ar,1/dr,ar,optimize=True)
        matrix=np.eye(c)[None,:,:]-1j*r
        rhs=np.zeros((len(r),c,1),complex);rhs[:,entrance,0]=1.
        y=np.linalg.solve(matrix,rhs)[:,:,0]
        residual=matrix@y[:,:,None]-rhs
        maximum=float(np.max(np.abs(residual)/(np.linalg.norm(matrix,axis=(1,2))[:,None,None]*np.linalg.norm(y,axis=1)[:,None,None]+1.)))
        w[regular]=np.einsum('ecd,ed->ec',r,y)
        x[regular]=np.einsum('enc,ec->en',ar,y)/dr
    for index in np.flatnonzero(~regular):
        pole=poles[index];other=~pole;ap=a[index,pole];ao=a[index,other]
        r=(ao.T/d[index,other])@ao
        count=len(ap)
        matrix=np.block([[np.eye(c)-1j*r,-1j*ap.T],[-ap,np.diag(d[index,pole])]])
        rhs=np.zeros(c+count,complex);rhs[entrance]=1.
        try:solution=np.linalg.solve(matrix,rhs)
        except np.linalg.LinAlgError:
            solution=np.linalg.lstsq(matrix,rhs,rcond=None)[0];singular+=1
        residual=np.linalg.norm(matrix@solution-rhs)/(np.linalg.norm(matrix)*np.linalg.norm(solution)+1.)
        maximum=max(maximum,float(residual))
        y=solution[:c];xp=solution[c:]
        w[index]=r@y+ap.T@xp
        x[index,other]=ao@y/d[index,other];x[index,pole]=xp
    if maximum>1e-11 or np.any(~np.isfinite(w)) or np.any(~np.isfinite(x)):
        raise FloatingPointError('RM linear solve failed residual/finite-value check')
    if diagnostics is not None:
        diagnostics['rm_max_solver_residual']=max(diagnostics.get('rm_max_solver_residual',0.),maximum)
        diagnostics['rm_singular_pole_limits']=diagnostics.get('rm_singular_pole_limits',0)+singular
    return w,x


def evaluate_rm(energies,groups,context,diagnostics=None):
    elastic,capture,fission=(np.zeros_like(energies) for _ in range(3))
    for group in groups:
        ctx=group.context or context;k2=ctx.k_squared_per_ev*energies
        beta=np.pi*.01/k2
        phi=np.zeros_like(energies) if group.phase_radius.constant==0. else neutral_channel_functions(group.l,np.sqrt(k2)*group.phase_radius.evaluate(energies))[2]
        g=(2*group.spin+1)/(2*(2*ctx.target_spin+1))
        if not group.levels:
            elastic+=beta*g*4*np.sin(phi)**2;continue
        levels=group.levels;gamma=np.array([level.capture for level in levels])
        reduced=np.array([level.neutron_amplitude for level in levels])
        af=np.array([level.fission_amplitudes for level in levels])
        af=af[:,np.any(af!=0,axis=0)]  # Zero-strength exit channels have no observable effect.
        p=neutral_channel_functions(group.l,np.sqrt(k2)*group.channel_radius.evaluate(energies))[0]
        if np.any(p==0) and np.any(reduced):raise FloatingPointError('RM neutron penetrability underflows')
        a=np.empty((len(energies),len(levels),1+af.shape[1]))
        a[:,:,0]=np.sqrt(p)[:,None]*reduced[None,:]
        a[:,:,1:]=af[None,:,:]
        w,x=solve_collision(energies,[level.energy for level in levels],gamma,a,diagnostics)
        # Stable 1-U_nn, avoiding the cancellation of hard-sphere phase.
        amplitude=2j*np.exp(-1j*phi)*np.sin(phi)-2j*np.exp(-2j*phi)*w[:,0]
        elastic+=beta*g*np.abs(amplitude)**2
        absorption=2*np.sum(gamma[None,:]*np.abs(x)**2,axis=1)
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

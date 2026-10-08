"""Neutron-incidence KRM3/KRM4 algebra, including charged exits and closed shifts.

R is unscaled: reduced amplitudes do not include sqrt(P). A channel/level
augmented system replaces naked divisions near poles. Absorption is evaluated
directly; flux is an independent diagnostic, never an instruction to clip.
"""
from dataclasses import dataclass
import numpy as np
from .breit_wigner import Group
from .channel_functions import neutral_channel_functions
from scipy.special import gammaln, logsumexp


def closed_neutral_shift(l, kappa):
    """Logarithmic derivative of the decaying neutral Riccati solution.

    exp(-kappa) times the finite inverse-power polynomial is evaluated in log
    space. Threshold is the exact limit -l, including l=0.
    """
    x = np.asarray(kappa, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError('closed-channel kappa must be finite and nonnegative')
    if not isinstance(l, (int, np.integer)) or not 0 <= l <= 64:
        raise ValueError('neutral orbital momentum must be L=0..64')
    out = np.full_like(x, -float(l))
    positive = x > 0
    m = np.arange(l+1)
    terms = (gammaln(l+m+1)-gammaln(m+1)-gammaln(l-m+1)
             -m*np.log(2*x[positive])[..., None])
    weights = np.exp(terms-logsumexp(terms, axis=-1)[..., None])
    out[positive] = -x[positive]-np.sum(m*weights, axis=-1)
    return out


@dataclass(frozen=True)
class RMLChannel:
    mt: int
    l: int
    spin: float
    q: float
    cm_ratio: float
    k2_cm: float
    radius: object
    phase_radius: object
    penetrability: str
    shift: str
    boundary: float
    effective: bool = False
    external: object = None
    charge_strength: float = 0.
    phase_function: object = None
    kinematics: object = None
    reaction_label: str = ''
    phase_absorption_mt: object = None

    @property
    def threshold(self):
        return self.kinematics.threshold if self.kinematics is not None else -self.q/self.cm_ratio

    def channel_energy(self,lab):
        return self.kinematics.energy(lab) if self.kinematics is not None else self.cm_ratio*(np.asarray(lab)-self.threshold)

    def k_squared(self,channel_energy):
        return self.kinematics.k_squared(channel_energy) if self.kinematics is not None else self.k2_cm*channel_energy

    def functions(self, energy, *, logarithmic=False):
        e = np.asarray(energy,dtype=float)
        channel_energy = self.channel_energy(e)
        opened = channel_energy > 0
        p = np.zeros_like(e)
        log_p = np.full_like(e,-np.inf)
        s = np.zeros_like(e)
        phase = np.zeros_like(e, dtype=complex)
        if self.effective:
            p.fill(1.)
            log_p.fill(0.)
        else:
            rho = np.sqrt(np.abs(self.k_squared(channel_energy)))*self.radius.evaluate(e)
            if np.any(opened):
                if self.charge_strength:
                    from .coulomb import charged_channel_functions,charged_channel_log_functions
                    # ENDF D.80-D.81 retain the rest reduced mass in eta;
                    # KRL changes k through the two-body invariant.
                    eta = self.charge_strength/np.sqrt(self.k_squared(channel_energy[opened]))
                    if logarithmic:
                        lp,so,_ = charged_channel_log_functions(self.l,eta,rho[opened])
                        po = np.exp(lp)
                    else:
                        po,so,_ = charged_channel_functions(self.l,eta,rho[opened])
                else:
                    po, so, _ = neutral_channel_functions(self.l, rho[opened])
                p[opened] = po if self.penetrability == 'calculate' else 1.
                if self.penetrability != 'calculate':log_p[opened] = 0.
                elif self.charge_strength and logarithmic:log_p[opened] = lp
                else:
                    with np.errstate(divide='ignore'):log_p[opened] = np.log(po)
                s[opened] = so
                if self.phase_radius.constant != 0.:
                    phase_rho = np.sqrt(self.k_squared(channel_energy[opened]))*self.phase_radius.evaluate(e[opened])
                    if self.charge_strength:
                        phase[opened] = (charged_channel_log_functions if logarithmic else charged_channel_functions)(self.l,eta,phase_rho)[2]
                    else:
                        phase[opened] = neutral_channel_functions(self.l,phase_rho)[2]
            if np.any(~opened):
                if self.charge_strength:
                    from .coulomb import closed_charged_shift,charged_threshold_shift
                    closed = channel_energy<0;threshold = channel_energy==0
                    if np.any(closed):
                        eta = self.charge_strength/np.sqrt(-self.k_squared(channel_energy[closed]))
                        s[closed] = closed_charged_shift(self.l,eta,rho[closed])
                    if np.any(threshold):
                        s[threshold] = charged_threshold_shift(self.l,self.charge_strength*self.radius.evaluate(e[threshold]))
                else:
                    s[~opened] = closed_neutral_shift(self.l, rho[~opened])
        if self.phase_function is not None:
            phase[opened] = self.phase_function.evaluate(e[opened])
        real = s-self.boundary if self.shift == 'calculate' else np.full_like(e, -self.boundary)
        values = (p, real+1j*p, phase)
        return (*values,log_p) if logarithmic else values


@dataclass(frozen=True)
class RMLGroup(Group):
    spin: float = 0.
    channels: tuple = ()
    reduced: tuple = ()
    radiation: tuple = ()
    entrances: tuple = ()
    parity: int = 1
    level_metric: tuple = ()
    level_energy: tuple = ()
    level_origin: float = 0.

    @property
    def reaction_mts(self):
        return {c.mt for c in self.channels} | {c.phase_absorption_mt for c in self.channels if c.phase_absorption_mt is not None} | {102}


def solve_rml(energies, levels, radiation, reduced, logarithmic, external=None, *, entrance=0, diagnostics=None,
              log_penetrability=None, error_bounds=None, level_metric=None, level_energy=None,level_origin=0.):
    """Return W_c,n, level absorption X_lambda,n and channel excitation y.

    y=(I-LR)^-1 e_n; W=Ry and X=D^-1 gamma y. Retaining closed
    channels (P=0, real L) preserves their virtual real-shift contribution.
    """
    e = np.asarray(energies)
    a = np.asarray(reduced, dtype=float)
    n, c = a.shape
    d = np.asarray(levels)[None, :]-e[:, None]-.5j*np.asarray(radiation)[None, :]
    z = np.zeros((len(e), c), complex) if external is None else np.asarray(external, complex)
    logarithmic = np.asarray(logarithmic, complex)
    if not 0 <= entrance < c or z.shape != logarithmic.shape or z.shape != (len(e), c):
        raise ValueError('invalid RML channel dimensions/entrance')
    w = np.zeros((len(e), c), complex)
    x = np.zeros((len(e), n), complex)
    excitation = np.zeros_like(w)
    maximum = 0.; singular = 0
    missing = (np.isfinite(log_penetrability)&(logarithmic.imag==0)) if log_penetrability is not None else np.zeros_like(logarithmic,dtype=bool)
    track_missing = np.any(missing)
    w_errors = np.full(w.shape,-np.inf) if track_missing else None
    x_errors = np.full(x.shape,-np.inf) if track_missing else None
    y_errors = np.full(w.shape,-np.inf) if track_missing else None
    def bound(i,matrix,state,channel_map):
        if not np.any(missing[i]):return -np.inf
        try:inverse = np.linalg.solve(matrix,np.eye(len(matrix)))
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError('unrepresentable penetrability has a singular limiting solve') from exc
        epsilon = np.finfo(float).eps
        operations = 8*len(matrix)
        rounding = operations*epsilon/(1-operations*epsilon)
        residual = (np.linalg.norm(np.eye(len(matrix))-matrix@inverse,np.inf)
                    +rounding*(1+np.linalg.norm(abs(matrix)@abs(inverse),np.inf)))
        inverse_norm = np.linalg.norm(inverse,np.inf)*(1+rounding)
        row_norms = np.sum(abs(channel_map),axis=1)*(1+rounding)
        state_norm = np.max(abs(state))*(1+rounding)
        if not np.isfinite(residual+inverse_norm+state_norm) or np.any(~np.isfinite(row_norms)) or residual>=.5:
            raise FloatingPointError('unrepresentable penetrability lacks a verified inverse bound')
        with np.errstate(divide='ignore'):
            log_rows = np.log(row_norms)
            log_delta = np.max(np.asarray(log_penetrability)[i,missing[i]]+log_rows[missing[i]])
            log_q = np.log(inverse_norm)-np.log1p(-residual)+log_delta
            if log_q>=np.log(.5):
                raise FloatingPointError('unrepresentable penetrability changes the RML solve significantly')
            log_state_error = log_q-np.log1p(-np.exp(log_q))+np.log(state_norm)
        w_errors[i] = log_rows+log_state_error
        y_errors[i] = log_state_error
        if diagnostics is not None:
            diagnostics['rml_underflow_bounded_solves'] = diagnostics.get('rml_underflow_bounded_solves',0)+1
            diagnostics['rml_underflow_max_log_perturbation_bound'] = max(diagnostics.get('rml_underflow_max_log_perturbation_bound',-np.inf),float(log_q))
        return log_state_error
    if level_metric is not None:
        metric = np.asarray(level_metric).reshape(n,n)
        eigen_energy = np.asarray(level_energy).reshape(n,n)
        for i,energy in enumerate(e):
            matrix = np.zeros((c+n,c+n),complex)
            matrix[:c,:c] = np.eye(c)-np.diag(logarithmic[i]*z[i])
            matrix[:c,c:] = -logarithmic[i,:,None]*a.T
            matrix[c:,:c] = -a
            matrix[c:,c:] = eigen_energy-(energy-level_origin)*metric-.5j*np.diag(radiation)
            rhs = np.zeros(c+n,complex);rhs[entrance] = 1.
            used_limit=0
            try:solution=np.linalg.solve(matrix,rhs)
            except np.linalg.LinAlgError:
                solution=np.linalg.lstsq(matrix,rhs,rcond=None)[0];used_limit=1
            residual=np.linalg.norm(matrix@solution-rhs,np.inf)/(np.linalg.norm(matrix,np.inf)*np.linalg.norm(solution,np.inf)+1.)
            maximum=max(maximum,residual);singular+=used_limit
            y,x[i] = solution[:c],solution[c:]
            excitation[i]=y;w[i]=z[i]*y+a.T@x[i]
            if np.any(missing[i]):
                channel_map=np.concatenate((np.diag(z[i]),a.T),axis=1)
                delta=bound(i,matrix,solution,channel_map)
                x_errors[i]=delta
        regular = np.zeros(len(e),bool)
        pole_indices = ()
    else:
        poles = (np.abs(d) <= 1e-6*np.sum(a*a, axis=1)[None,:]*np.maximum(1., np.max(abs(logarithmic),axis=1))[:,None]) | (d == 0)
        regular = ~np.any(poles,axis=1)
        pole_indices = np.flatnonzero(~regular)
    if np.any(regular):
        r = np.einsum('nc,en,nd->ecd',a,1/d[regular],a,optimize=True)
        indexes = np.arange(c)
        r[:,indexes,indexes] += z[regular]
        matrix = np.eye(c)[None,:,:]-logarithmic[regular,:,None]*r
        rhs = np.zeros((len(r),c,1),complex);rhs[:,entrance,0] = 1.
        y = np.linalg.solve(matrix,rhs)[:,:,0]
        residual = np.linalg.norm((matrix@y[:,:,None]-rhs).reshape((len(r),c)),axis=1)
        scale = np.linalg.norm(matrix,axis=(1,2))*np.linalg.norm(y,axis=1)+1.
        maximum = float(np.max(residual/scale))
        excitation[regular] = y
        w[regular] = np.einsum('ecd,ed->ec',r,y)
        x[regular] = np.einsum('nc,ec->en',a,y)/d[regular]
        regular_indices = np.flatnonzero(regular)
        for local in np.flatnonzero(np.any(missing[regular],axis=1)):
            i = regular_indices[local]
            delta = bound(i,matrix[local],y[local],r[local])
            with np.errstate(divide='ignore'):
                x_errors[i] = np.log(np.sum(abs(a),axis=1))-np.log(abs(d[i]))+delta
    for i in pole_indices:
        pole = poles[i]
        other = ~pole
        r0 = np.diag(z[i])+(a[other].T/d[i, other])@a[other]
        ap = a[pole]
        matrix = np.block([[np.eye(c)-logarithmic[i, :, None]*r0,
                            -logarithmic[i, :, None]*ap.T],
                           [-ap, np.diag(d[i, pole])]])
        rhs = np.zeros(c+len(ap), complex); rhs[entrance] = 1.
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]; singular += 1
        residual = np.linalg.norm(matrix@solution-rhs)/(np.linalg.norm(matrix)*np.linalg.norm(solution)+1.)
        maximum = max(maximum, float(residual))
        y = solution[:c]; xp = solution[c:]
        excitation[i] = y
        w[i] = r0@y+ap.T@xp
        x[i, other] = a[other]@y/d[i, other]
        x[i, pole] = xp
        if np.any(missing[i]):
            delta = bound(i,matrix,solution,np.concatenate((r0,ap.T),axis=1))
            x_errors[i,pole] = delta
            with np.errstate(divide='ignore'):
                x_errors[i,other] = np.log(np.sum(abs(a[other]),axis=1))-np.log(abs(d[i,other]))+delta
    if maximum > 1e-11 or any(np.any(~np.isfinite(v)) for v in (w, x, excitation)):
        raise FloatingPointError('RML solve failed finite/residual check')
    if diagnostics is not None:
        diagnostics['rml_max_solver_residual'] = max(diagnostics.get('rml_max_solver_residual', 0.), maximum)
        diagnostics['rml_singular_limits'] = diagnostics.get('rml_singular_limits', 0)+singular
    if error_bounds is not None:
        error_bounds.update(w=w_errors,x=x_errors,y=y_errors,missing=missing)
    return w, x, excitation


def _check_underflow_bound(log_value,log_error):
    """Bound the relative perturbation of a squared amplitude, including zero."""
    # A zero may only hide a squared correction below half the least subnormal.
    zero = ~np.isfinite(log_value)
    if np.any(np.isnan(log_value)) or np.any(np.isnan(log_error)):
        raise FloatingPointError('nonfinite underflow accuracy bound')
    limit = np.log(np.nextafter(0.,1.))-np.log(2.)
    if np.any(2*log_error[zero]>=limit):
        raise FloatingPointError('underflow correction to a zero observable is not bounded')
    if np.any(log_error[~zero]-log_value[~zero]>np.log(2e-12)):
        raise FloatingPointError('underflow correction exceeds the amplitude accuracy bound')


def evaluate_rml(energies, groups, context, diagnostics=None):
    e = np.asarray(energies)
    mts = {1, 2, 18, 102} | {mt for g in groups for mt in g.reaction_mts}
    result = {mt: np.zeros_like(e) for mt in mts}
    for group in groups:
        ctx = group.context or context
        incident = group.channels[group.entrances[0]]
        k2 = incident.k_squared(incident.channel_energy(e))
        beta = np.pi*.01/k2*(2*group.spin+1)/(2*(2*ctx.target_spin+1))
        values = [ch.functions(e,logarithmic=True) for ch in group.channels]
        p, log, phase, log_p = (np.stack([v[index] for v in values], axis=1) for index in range(4))
        # Subnormal P can retain only a few significant bits. Bound the
        # complete omitted term, and use ln(P) for charged exit observables.
        charged = np.array([bool(ch.charge_strength) for ch in group.channels])
        limited = charged[None,:]&np.isfinite(log_p)&(log_p<np.log(np.finfo(float).tiny))
        log = log.real+1j*np.where(limited,0.,p)
        if np.any(phase.imag>0):
            raise ValueError('active KPS phase is outside the passive profile')
        attenuation = np.exp(2*phase.imag)
        phase_loss = -np.expm1(2*phase.imag)
        reduced = np.asarray(group.reduced).reshape((len(group.levels),len(group.channels)))
        for index,ch in enumerate(group.channels):
            if (not ch.effective and ch.penetrability == 'calculate' and np.any(reduced[:,index] != 0)
                    and np.any((e > ch.threshold) & (p[:,index] == 0) & ~np.isfinite(log_p[:,index]))):
                raise FloatingPointError('open RML penetrability underflows for an active channel')
        external = np.stack([np.zeros_like(e, dtype=complex) if ch.external is None else ch.external.evaluate(e)
                             for ch in group.channels], axis=1)
        if np.any(external.imag < 0):
            raise ValueError('negative external absorption is outside the passive KRM3 profile')
        for entrance in group.entrances:
            bounds = {}
            w, x, y = solve_rml(e, [lv.energy for lv in group.levels], group.radiation,
                reduced, log,
                external, entrance=entrance, diagnostics=diagnostics,
                log_penetrability=log_p,error_bounds=bounds,
                level_metric=group.level_metric if group.level_metric else None,
                level_energy=group.level_energy if group.level_metric else None,
                level_origin=group.level_origin)
            underflow = np.any(bounds['missing'],axis=1)
            core = 2j*np.sqrt(p)*np.sqrt(p[:, entrance, None])*w
            if np.any(underflow):
                with np.errstate(divide='ignore'):
                    log_core = np.log(2.)+.5*(log_p+log_p[:,entrance,None])+np.log(abs(w))
                    log_core_error = np.log(2.)+.5*(log_p+log_p[:,entrance,None])+bounds['w']
                # Preserve products rescued by a large amplitude or beta.
                mask = bounds['missing'] & (abs(w)>0)
                core[mask] = 1j*np.exp(log_core[mask])*(w[mask]/abs(w[mask]))
            resonant = core*np.exp(-1j*(phase+phase[:, entrance, None]))
            core_collision = core.copy(); core_collision[:,entrance] += 1.
            collision = resonant.copy()
            collision[:, entrance] += np.exp(-2j*phase[:, entrance])
            for out, channel in enumerate(group.channels):
                amplitude = resonant[:, out]
                if out == entrance:
                    amplitude = -np.expm1(-2j*phase[:, entrance])-resonant[:, entrance]
                sigma = beta*abs(amplitude)**2
                if np.any(underflow):
                    with np.errstate(divide='ignore'):
                        log_amplitude = np.log(abs(amplitude))
                    if out != entrance:
                        log_amplitude[underflow] = (log_core[:,out]+phase[:,out].imag+phase[:,entrance].imag)[underflow]
                        sigma[underflow] = np.exp((np.log(beta)+2*log_amplitude)[underflow])
                    _check_underflow_bound((log_amplitude+.5*np.log(beta))[underflow],
                        (log_core_error[:,out]+phase[:,out].imag+phase[:,entrance].imag+.5*np.log(beta))[underflow])
                    with np.errstate(divide='ignore'):
                        _check_underflow_bound((np.log(abs(core_collision[:,out]))+.5*np.log(beta))[underflow],
                            (log_core_error[:,out]+.5*np.log(beta))[underflow])
                result[channel.mt] += sigma
            absorption = (2*p[:, entrance]*np.sum(np.asarray(group.radiation)[None, :]*abs(x)**2, axis=1)
                          +4*p[:, entrance]*np.sum(external.imag*abs(y)**2, axis=1))
            if np.any(underflow):
                with np.errstate(divide='ignore'):
                    level_weights=np.log(2.)+log_p[:,entrance,None]+np.log(np.asarray(group.radiation))[None,:]
                    external_weights=np.log(4.)+log_p[:,entrance,None]+np.log(external.imag)
                    log_x=np.log(abs(x));log_y=np.log(abs(y))
                    active=np.asarray(group.radiation)>0
                    _check_underflow_bound((log_x+.5*(level_weights+np.log(beta)[:,None]))[underflow][:,active],
                        (bounds['x']+.5*(level_weights+np.log(beta)[:,None]))[underflow][:,active])
                    active_external=(external.imag>0)&underflow[:,None]
                    _check_underflow_bound((log_y+.5*(external_weights+np.log(beta)[:,None]))[active_external],
                        (bounds['y']+.5*(external_weights+np.log(beta)[:,None]))[active_external])
                    log_absorption=np.logaddexp(logsumexp(level_weights+2*log_x,axis=1),
                        logsumexp(external_weights+2*log_y,axis=1))
                absorption[underflow]=np.exp(log_absorption[underflow])
            error_core = float(np.max(abs(1-np.sum(abs(core_collision)**2, axis=1)-absorption)))
            absorption *= attenuation[:,entrance]
            capture_sigma=beta*absorption
            if np.any(underflow):capture_sigma[underflow]=np.exp((np.log(beta)+log_absorption+2*phase[:,entrance].imag)[underflow])
            result[102] += capture_sigma
            # Direct positive decomposition of flux lost on entering and
            # leaving the phase layer. Never assign a residual to capture.
            for out,channel in enumerate(group.channels):
                loss = attenuation[:,entrance]*phase_loss[:,out]*abs(core_collision[:,out])**2
                if out == entrance:loss += phase_loss[:,entrance]
                if channel.phase_absorption_mt is not None:
                    loss_sigma=beta*loss
                    if np.any(underflow):
                        with np.errstate(divide='ignore'):
                            log_loss=2*phase[:,entrance].imag+np.log(phase_loss[:,out])+2*np.log(abs(core_collision[:,out]))
                            if out==entrance:log_loss=np.logaddexp(log_loss,np.log(phase_loss[:,entrance]))
                        loss_sigma[underflow]=np.exp((np.log(beta)+log_loss)[underflow])
                    result[channel.phase_absorption_mt] += loss_sigma
                elif np.any(loss != 0):
                    raise ValueError('absorptive phase requires declared reaction ownership')
                absorption += loss
            error = float(np.max(abs(1-np.sum(abs(collision)**2, axis=1)-absorption)))
            error = max(error,error_core)
            if error > 1e-9:
                raise FloatingPointError('RML collision matrix fails flux balance')
            if diagnostics is not None:
                diagnostics['rml_max_absolute_flux_error'] = max(diagnostics.get('rml_max_absolute_flux_error', 0.), error)
    result[1] = sum(v for mt, v in result.items() if mt != 1)
    if any(np.any(~np.isfinite(v)) for v in result.values()):
        raise FloatingPointError('nonfinite RML cross sections')
    return result

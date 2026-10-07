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

    @property
    def threshold(self):
        return self.kinematics.threshold if self.kinematics is not None else -self.q/self.cm_ratio

    def channel_energy(self,lab):
        return self.kinematics.energy(lab) if self.kinematics is not None else self.cm_ratio*(np.asarray(lab)-self.threshold)

    def k_squared(self,channel_energy):
        return self.kinematics.k_squared(channel_energy) if self.kinematics is not None else self.k2_cm*channel_energy

    def functions(self, energy):
        e = np.asarray(energy)
        channel_energy = self.channel_energy(e)
        opened = channel_energy > 0
        p = np.zeros_like(e)
        s = np.zeros_like(e)
        phase = np.zeros_like(e)
        if self.effective:
            p.fill(1.)
        else:
            rho = np.sqrt(np.abs(self.k_squared(channel_energy)))*self.radius.evaluate(e)
            if np.any(opened):
                if self.charge_strength:
                    from .coulomb import charged_channel_functions
                    eta = self.charge_strength/np.sqrt(self.k2_cm*channel_energy[opened])
                    po,so,_ = charged_channel_functions(self.l,eta,rho[opened])
                else:
                    po, so, _ = neutral_channel_functions(self.l, rho[opened])
                p[opened] = po if self.penetrability == 'calculate' else 1.
                s[opened] = so
                if self.phase_radius.constant != 0.:
                    phase_rho = np.sqrt(self.k_squared(channel_energy[opened]))*self.phase_radius.evaluate(e[opened])
                    if self.charge_strength:
                        phase[opened] = charged_channel_functions(self.l,eta,phase_rho)[2]
                    else:
                        phase[opened] = neutral_channel_functions(self.l,phase_rho)[2]
            if np.any(~opened):
                if self.charge_strength:
                    from .coulomb import closed_charged_shift,charged_threshold_shift
                    closed = channel_energy<0;threshold = channel_energy==0
                    if np.any(closed):
                        eta = self.charge_strength/np.sqrt(self.k2_cm*-channel_energy[closed])
                        s[closed] = closed_charged_shift(self.l,eta,rho[closed])
                    if np.any(threshold):
                        s[threshold] = charged_threshold_shift(self.l,self.charge_strength*self.radius.evaluate(e[threshold]))
                else:
                    s[~opened] = closed_neutral_shift(self.l, rho[~opened])
        if self.phase_function is not None:
            phase[opened] = self.phase_function.evaluate(e[opened]).real
        real = s-self.boundary if self.shift == 'calculate' else np.full_like(e, -self.boundary)
        return p, real+1j*p, phase


@dataclass(frozen=True)
class RMLGroup(Group):
    spin: float = 0.
    channels: tuple = ()
    reduced: tuple = ()
    radiation: tuple = ()
    entrances: tuple = ()
    parity: int = 1

    @property
    def reaction_mts(self):
        return {c.mt for c in self.channels} | {102}


def solve_rml(energies, levels, radiation, reduced, logarithmic, external=None, *, entrance=0, diagnostics=None):
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
    poles = (np.abs(d) <= 1e-6*np.sum(a*a, axis=1)[None,:]*np.maximum(1., np.max(abs(logarithmic),axis=1))[:,None]) | (d == 0)
    regular = ~np.any(poles,axis=1)
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
    for i in np.flatnonzero(~regular):
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
    if maximum > 1e-11 or any(np.any(~np.isfinite(v)) for v in (w, x, excitation)):
        raise FloatingPointError('RML solve failed finite/residual check')
    if diagnostics is not None:
        diagnostics['rml_max_solver_residual'] = max(diagnostics.get('rml_max_solver_residual', 0.), maximum)
        diagnostics['rml_singular_limits'] = diagnostics.get('rml_singular_limits', 0)+singular
    return w, x, excitation


def evaluate_rml(energies, groups, context, diagnostics=None):
    e = np.asarray(energies)
    mts = {1, 2, 18, 102} | {mt for g in groups for mt in g.reaction_mts}
    result = {mt: np.zeros_like(e) for mt in mts}
    for group in groups:
        ctx = group.context or context
        incident = group.channels[group.entrances[0]]
        k2 = incident.k_squared(incident.channel_energy(e))
        beta = np.pi*.01/k2*(2*group.spin+1)/(2*(2*ctx.target_spin+1))
        values = [ch.functions(e) for ch in group.channels]
        p, log, phase = (np.stack([v[index] for v in values], axis=1) for index in range(3))
        reduced = np.asarray(group.reduced).reshape((len(group.levels),len(group.channels)))
        for index,ch in enumerate(group.channels):
            if (not ch.effective and ch.penetrability == 'calculate' and np.any(reduced[:,index] != 0)
                    and np.any((e > ch.threshold) & (p[:,index] == 0))):
                raise FloatingPointError('open RML penetrability underflows for an active channel')
        external = np.stack([np.zeros_like(e, dtype=complex) if ch.external is None else ch.external.evaluate(e)
                             for ch in group.channels], axis=1)
        if np.any(external.imag < 0):
            raise ValueError('negative external absorption is outside the passive KRM3 profile')
        for entrance in group.entrances:
            w, x, y = solve_rml(e, [lv.energy for lv in group.levels], group.radiation,
                reduced, log,
                external, entrance=entrance, diagnostics=diagnostics)
            resonant = 2j*np.sqrt(p*p[:, entrance, None])*w*np.exp(-1j*(phase+phase[:, entrance, None]))
            collision = resonant.copy()
            collision[:, entrance] += np.exp(-2j*phase[:, entrance])
            for out, channel in enumerate(group.channels):
                amplitude = resonant[:, out]
                if out == entrance:
                    amplitude = 2j*np.exp(-1j*phase[:, entrance])*np.sin(phase[:, entrance])-resonant[:, entrance]
                result[channel.mt] += beta*abs(amplitude)**2
            absorption = (2*p[:, entrance]*np.sum(np.asarray(group.radiation)[None, :]*abs(x)**2, axis=1)
                          +4*p[:, entrance]*np.sum(external.imag*abs(y)**2, axis=1))
            result[102] += beta*absorption
            error = float(np.max(abs(1-np.sum(abs(collision)**2, axis=1)-absorption)))
            if error > 1e-9:
                raise FloatingPointError('RML collision matrix fails flux balance')
            if diagnostics is not None:
                diagnostics['rml_max_absolute_flux_error'] = max(diagnostics.get('rml_max_absolute_flux_error', 0.), error)
    result[1] = sum(v for mt, v in result.items() if mt != 1)
    if any(np.any(~np.isfinite(v)) for v in result.values()):
        raise FloatingPointError('nonfinite RML cross sections')
    return result

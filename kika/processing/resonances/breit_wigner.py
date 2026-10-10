"""SLBW/MLBW equations (ENDF-102 Appendix D), evaluated in energy blocks.

Capture/fission are partial-width products, with no extra Gamma/2 factor.
Elastic MLBW sums complex amplitudes within J before taking their modulus.
There is no clipping, background, adaptive grid, or width normalization here.
"""
from dataclasses import dataclass
import numpy as np

from .channel_functions import neutral_channel_functions, neutral_shift_difference
from .radii import RadiusFunction
from .context import NeutronContext


@dataclass(frozen=True)
class Level:
    energy: float
    spin: float
    neutron: float
    capture: float
    fission: float
    competitive: float = 0.0


@dataclass(frozen=True)
class Group:
    l: int
    channel_radius: RadiusFunction
    phase_radius: RadiusFunction
    levels: tuple[Level, ...]
    context: NeutronContext | None = None
    competitive_mt: int | None = None
    competitive_l: int | None = None
    competitive_q: float = 0.0
    competitive_awr: float | None = None
    competitive_radius: float | None = None
    competitive_in_background: bool = False


def _accumulate_columns(target, values):
    """Ordered additions, including the existing accumulator's rounding."""
    if not values.shape[1]:return
    values[:,0]+=target
    np.cumsum(values,axis=1,out=values)
    target[:]=values[:,-1]


def _native_levels(energies, group, p, reference_p, beta, ctx, approximation,
                   sin2, sin_double, elastic, capture, fission, work_bytes):
    from ._rm_acceleration import _native
    if (_native is None or not hasattr(_native,'breit_wigner') or group.l>2
            or group.channel_radius.constant is None or len(group.levels)<8
            or any(lv.competitive for lv in group.levels)):
        return None
    spins=list(dict.fromkeys(lv.spin for lv in group.levels))
    if len(spins)>128 or len(energies)*8*(8+2*len(spins))>work_bytes:return None
    levels=np.array([(lv.energy,lv.neutron,lv.capture,lv.fission,
        (2*lv.spin+1)/(2*(2*ctx.target_spin+1)),pr,spins.index(lv.spin))
        for lv,pr in zip(group.levels,reference_p)])
    data=np.column_stack((energies,p,beta,sin2,sin_double))
    if any(np.any(~np.isfinite(a)) or np.any(abs(a)>1e20) for a in (levels,data)):return None
    if np.any(reference_p<=0) or np.any((levels[:,1:4]!=0)&(abs(levels[:,1:4])<1e-40)):return None
    coefficient=ctx.k_squared_per_ev*group.channel_radius.constant**2
    if coefficient<=0 or coefficient*max(np.max(energies),np.max(abs(levels[:,0])))>1e20:return None
    out=np.zeros((len(energies),3+2*len(spins)))
    out[:,0]=capture;out[:,1]=fission;out[:,2]=elastic
    accepted=_native.breit_wigner(len(energies),len(levels),len(spins),group.l,
        int(approximation=='MultiLevel'),coefficient,data.ravel(),levels.ravel(),out.ravel())
    if not accepted or np.any(~np.isfinite(out)):return None
    capture[:]=out[:,0];fission[:]=out[:,1];elastic[:]=out[:,2]
    return {spin:(out[:,3+2*i],out[:,4+2*i]) for i,spin in enumerate(spins)}


def _s_wave(energies,levels,p,reference_p,beta,context,approximation,
            sin2,sin_double,elastic,capture,fission,work_bytes):
    """Exact L=0, no-competition BW; vectorize bounded groups of levels.

    Accumulate in source level order, including each J amplitude. This avoids
    changing interference or summation conventions while removing the massive
    Python loop over individual levels.
    """
    amplitudes={}
    batch=max(1,min(128,work_bytes//max(1,128*len(energies))))
    for start in range(0,len(levels),batch):
        local=levels[start:start+batch]
        gn=p[:,None]*np.array([lv.neutron for lv in local])[None,:]/reference_p[None,start:start+batch]
        if np.any((gn==0)&(np.array([lv.neutron for lv in local])!=0)[None,:]):
            raise FloatingPointError('scaled neutron width underflows at an evaluation energy')
        gc=np.array([lv.capture for lv in local]);gf=np.array([lv.fission for lv in local])
        width=gn+gc[None,:]+gf[None,:]+0.
        delta=energies[:,None]-np.array([lv.energy for lv in local])[None,:]
        denominator=delta**2+(width/2)**2
        weights=np.array([(2*lv.spin+1)/(2*(2*context.target_spin+1)) for lv in local])
        factor=beta[:,None]*weights[None,:]
        _accumulate_columns(capture,factor*gn*gc[None,:]/denominator)
        _accumulate_columns(fission,factor*gn*gf[None,:]/denominator)
        if approximation=='SingleLevel':
            _accumulate_columns(elastic,factor*gn*(gn-2*width*sin2[:,None]+2*delta*sin_double[:,None])/denominator)
        else:
            for spin in dict.fromkeys(lv.spin for lv in local):
                mask=np.array([lv.spin==spin for lv in local])
                t1,t2=amplitudes.setdefault(spin,(np.zeros(len(energies)),np.zeros(len(energies))))
                _accumulate_columns(t1,(gn*width/2/denominator)[:,mask])
                _accumulate_columns(t2,(gn*delta/denominator)[:,mask])
    return amplitudes


def evaluate_bw(energies, groups, approximation, context, *, work_bytes=64*1024**2, accelerated=False):
    elastic, capture, fission = (np.zeros_like(energies) for _ in range(3))
    competitive = {}
    for group in groups:
        ctx = group.context or context
        k2 = ctx.k_squared_per_ev * energies
        beta = np.pi * 0.01 / k2  # fm^2 -> barn
        l = group.l
        radius = group.channel_radius.evaluate(energies)
        same_radius=group.channel_radius==group.phase_radius
        p, s, phi = neutral_channel_functions(l, np.sqrt(k2) * radius, phase=same_radius)
        if any(level.neutron for level in group.levels) and np.any(p == 0):
            raise FloatingPointError('neutron penetrability underflows at an evaluation energy')
        if not same_radius:
            _, _, phi = neutral_channel_functions(l, np.sqrt(k2) * group.phase_radius.evaluate(energies))
        sin2 = np.sin(phi)**2
        sin_double = np.sin(2 * phi)
        potential = 4 * sin2
        if approximation == "SingleLevel":
            elastic += beta * (2*l + 1) * potential
        amplitudes = {}
        # Evaluate level-reference quantities once as a vector, rather than
        # invoking table/channel validation for every level in every block.
        reference_energies=np.asarray([abs(level.energy) for level in group.levels])
        reference_radii=group.channel_radius.evaluate(reference_energies)
        reference_p=neutral_channel_functions(l,np.sqrt(ctx.k_squared_per_ev*reference_energies)*reference_radii,phase=False)[0]
        native=_native_levels(energies,group,p,reference_p,beta,ctx,approximation,
            sin2,sin_double,elastic,capture,fission,work_bytes) if accelerated and len(energies) else None
        if native is not None or (l==0 and len(group.levels)>=32 and not any(level.competitive for level in group.levels)):
            amplitudes=native if native is not None else _s_wave(energies,group.levels,p,reference_p,beta,ctx,approximation,
                sin2,sin_double,elastic,capture,fission,work_bytes)
            if group.competitive_mt is not None:competitive.setdefault(group.competitive_mt,np.zeros_like(energies))
            if approximation=='MultiLevel':
                represented_weight=0.
                for spin,(t1,t2) in amplitudes.items():
                    g=(2*spin+1)/(2*(2*ctx.target_spin+1));represented_weight+=g
                    elastic+=beta*g*((2*sin2-t1)**2+(sin_double+t2)**2)
                elastic+=beta*(2*l+1-represented_weight)*potential
            continue
        for level_index,level in enumerate(group.levels):
            pr=reference_p[level_index]
            gn = level.neutron * p / pr
            if level.neutron and np.any(gn == 0):
                raise FloatingPointError('scaled neutron width underflows at an evaluation energy')
            gx = np.zeros_like(energies)
            if level.competitive:
                entrance_ratio = ctx.atomic_weight_ratio/(1+ctx.atomic_weight_ratio)
                exit_awr = group.competitive_awr or ctx.atomic_weight_ratio
                reduced = exit_awr/(1+exit_awr)
                coefficient = 2*ctx.neutron_mass_mev*1e-6/ctx.hbar_c_mev_fm**2*reduced
                available = entrance_ratio*energies+group.competitive_q
                # ENDF D.123: reference is |Er - E_threshold|, including bound
                # levels. For neutron exit with the same masses this is identical
                # to coefficient * |entrance_ratio*Er + Q|.
                reference = abs(entrance_ratio*level.energy+group.competitive_q)
                pxr = neutral_channel_functions(group.competitive_l,
                    np.sqrt(coefficient*reference)*group.competitive_radius)[0]
                open_mask = available > 0
                if np.any(open_mask):
                    px = neutral_channel_functions(group.competitive_l,
                        np.sqrt(coefficient*available[open_mask])*group.competitive_radius)[0]
                    if np.any(px == 0):
                        raise FloatingPointError('open competitive penetrability underflows')
                    gx[open_mask] = level.competitive*px/pxr
            width = gn + level.capture + level.fission + gx
            if l==0:
                delta=energies-level.energy  # Neutral S_0 is identically zero.
            else:
                reference_energy = reference_energies[level_index]
                reference_radius = reference_radii[level_index]
                radius_delta = group.channel_radius.difference(reference_energy, energies)
                squared_delta = ctx.k_squared_per_ev * (
                    (reference_energy-energies)*reference_radius**2
                    + energies*radius_delta*(reference_radius+radius))
                shift_delta = neutral_shift_difference(l,
                    ctx.k_squared_per_ev*reference_energy*reference_radius**2,
                    k2*radius**2, squared_delta)
                delta = (energies-level.energy) - level.neutron*shift_delta/(2*pr)
            denominator = delta**2 + (width/2)**2
            g = (2*level.spin + 1) / (2*(2*ctx.target_spin + 1))
            capture += beta * g * gn * level.capture / denominator
            fission += beta * g * gn * level.fission / denominator
            if group.competitive_mt is not None:
                partial = competitive.setdefault(group.competitive_mt, np.zeros_like(energies))
                partial += beta*g*gn*gx/denominator
            if approximation == "SingleLevel":
                elastic += beta * g * gn * (
                    gn - 2*width*sin2 + 2*delta*sin_double) / denominator
            else:
                t1, t2 = amplitudes.setdefault(level.spin, (np.zeros_like(energies),
                                                           np.zeros_like(energies)))
                t1 += gn * width/2 / denominator
                t2 += gn * delta / denominator
        if approximation == "MultiLevel":
            represented_weight = 0.0
            for spin, (t1, t2) in amplitudes.items():
                g = (2*spin + 1) / (2*(2*ctx.target_spin + 1))
                represented_weight += g
                elastic += beta * g * ((2*sin2 - t1)**2 + (sin_double + t2)**2)
            # Include hard-sphere scattering from absent J/channel-spin sectors.
            elastic += beta * (2*l + 1 - represented_weight) * potential
    total = elastic + capture + fission + sum(competitive.values(), np.zeros_like(energies))
    result = {1: total, 2: elastic, 18: fission, 102: capture}
    result.update(competitive)
    if any(np.any(~np.isfinite(values)) for values in result.values()):
        raise FloatingPointError("Breit-Wigner evaluation produced nonfinite cross sections")
    return result

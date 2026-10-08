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


def evaluate_bw(energies, groups, approximation, context):
    elastic, capture, fission = (np.zeros_like(energies) for _ in range(3))
    competitive = {}
    for group in groups:
        ctx = group.context or context
        k2 = ctx.k_squared_per_ev * energies
        beta = np.pi * 0.01 / k2  # fm^2 -> barn
        l = group.l
        radius = group.channel_radius.evaluate(energies)
        p, s, _ = neutral_channel_functions(l, np.sqrt(k2) * radius)
        if any(level.neutron for level in group.levels) and np.any(p == 0):
            raise FloatingPointError('neutron penetrability underflows at an evaluation energy')
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
        reference_p=neutral_channel_functions(l,np.sqrt(ctx.k_squared_per_ev*reference_energies)*reference_radii)[0]
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

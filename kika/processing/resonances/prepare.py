"""Immutable preparation of supported neutron resolved and dilute URR models."""
from dataclasses import dataclass, replace
import math
import numpy as np

from .breit_wigner import Group, Level, evaluate_bw
from .context import NeutronContext
from .reich_moore import evaluate_rm
from .radii import prepare_radius
from .channel_functions import neutral_channel_functions


class UnsupportedResonanceError(ValueError):
    """The requested model contains physics not supported by this evaluator."""
    def __init__(self,message,*,category='unsupported-physics'):
        super().__init__(message)
        self.category=category


@dataclass(frozen=True)
class PreparedRegion:
    low: float
    high: float
    approximation: str
    groups: tuple[Group, ...]
    unresolved: object | None = None


@dataclass(frozen=True)
class PreparedResonances:
    """Immutable snapshot. ``evaluate`` returns MT -> array in barns.

    Values are resonance contributions; LSSF=1 contributes zero because its
    dilute cross sections already belong to the evaluated background.
    All energies must be inside one of the prepared regions. At a shared
    boundary the region on the right owns the point; the final high endpoint
    is included. Returned MT1 sums all physical partials, including competition.
    For ENDF BW, competition already belongs to MF3: assembly must honor
    ``competitive_in_background`` and must not add that partial twice.
    """
    context: NeutronContext | None
    regions: tuple[PreparedRegion, ...]
    preparation_notes: tuple[str, ...] = ()

    def evaluate(self, energies, *, block_size=2048, diagnostics=None):
        energy = np.asarray(energies, dtype=float)
        if energy.ndim > 1 or np.any(~np.isfinite(energy)) or np.any(energy <= 0):
            raise ValueError("energies must be a scalar or 1D finite positive eV array")
        if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        flat = energy.reshape(-1)
        owner = np.full(flat.size, -1, dtype=int)
        for i, region in enumerate(self.regions):
            owner[(flat >= region.low) & (flat <= region.high)] = i
        if np.any(owner < 0):
            raise ValueError("energies outside prepared regions (including gaps)")
        mts = {mt for r in self.regions for mt in region_mts(r)}
        output = {mt: np.zeros(flat.size) for mt in sorted(mts)}
        for i, region in enumerate(self.regions):
            indices = np.flatnonzero(owner == i)
            for start in range(0, indices.size, block_size):
                selected = indices[start:start+block_size]
                values = evaluate_region(flat[selected], region, self.context, diagnostics)
                for mt in values:
                    output[mt][selected] = values[mt]
        return {mt: values.reshape(energy.shape) for mt, values in output.items()}


def _positive(value, name):
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def prepare_resonances(resonances, context, *, conversion_report=None, allow_empty=False):
    """Prepare model ``Resonances``; explicit context, no ENDF width positions.

    Neutron BW/RM, scalar/tabulated radii in fm; BW also supports explicit competition.
    Dilute URR supports canonical mean widths; uncertified conventions reject
    the entire request before evaluation.
    Provenance is inspected only to detect data not yet expressible in the
    canonical model; it never supplies coefficients to the kernel.
    """
    from kika.nuclear_data.model.resonances import BreitWigner, Resonances, RMatrix

    empty=isinstance(resonances,Resonances) and not resonances.resolved and resonances.unresolved is None
    if not isinstance(resonances, Resonances) or not (isinstance(context,NeutronContext) or (allow_empty and empty and context is None)):
        raise TypeError("expected model Resonances and NeutronContext")
    if conversion_report is not None and not conversion_report.isCleanFor('cross-sections'):
        raise UnsupportedResonanceError("input conversion reports losses, approximations or unsupported data",category='conversion-not-clean')
    if empty and not allow_empty:
        raise ValueError("no resonance regions")
    header = getattr(resonances.provenance, "headerFields", None) or {}
    if not empty and len(header.get("isotopes", [])) > 1:
        raise UnsupportedResonanceError("isotope mixtures are not implemented")
    for record in header.get("regions", []):
        if record.get("kind") == "unsupported" or (record.get("kind") == "unresolved" and resonances.unresolved is None):
            raise UnsupportedResonanceError("source contains a dropped or unsupported region")
        if ("el" not in record or "eh" not in record) and (
                record.get("nro") or record.get("naps") == 2 or
                any(b.get("lrx") or b.get("qx") for b in record.get("l_blocks", []))):
            raise UnsupportedResonanceError("competitive channel metadata or radius has no region ownership")
    radius = resonances.scatteringRadius
    if radius is not None:
        if radius.unit not in (None, "fm"):
            raise UnsupportedResonanceError("radii must be normalized to fm")
    prepared = []
    notes = []
    for region in sorted(resonances.resolved, key=lambda r: r.domainMin):
        low = float(region.domainMin)
        high = _positive(region.domainMax, "domainMax")
        if not math.isfinite(low) or low < 0 or low >= high:
            raise ValueError("invalid resolved domain")
        if prepared and low < prepared[-1].high:
            raise UnsupportedResonanceError("overlapping resolved regions")
        if region.domainUnit != "eV":
            raise UnsupportedResonanceError("energy units must be normalized to eV")
        bw = region.formalism
        if isinstance(bw, RMatrix):
            from .prepare_reich_moore import prepare_rm
            if bw.resonanceReactions and all(rr.kinematics is not None for rr in bw.resonanceReactions):
                from .prepare_r_matrix import prepare_rml
                groups = prepare_rml(region,resonances,context,notes)
                approximation = 'RMatrixNeutral'
            else:
                groups = prepare_rm(region,resonances,context,notes)
                approximation = 'ReichMoore'
            prepared.append(PreparedRegion(low,high,approximation,groups))
            continue
        if not isinstance(bw, BreitWigner):
            raise UnsupportedResonanceError("only BreitWigner is implemented")
        for record in header.get("regions", []):
            if record.get("el") != low or record.get("eh") != high:
                continue
            if record.get("naps") not in (None, 0, 1, 2):
                raise UnsupportedResonanceError("unknown source radius convention")
            if record.get("naps") == 2 and bw.radiusPolicy is None:
                raise UnsupportedResonanceError("NAPS2 was not normalized into radiusPolicy")
            for block in record.get("l_blocks", []):
                if block.get("lrx") or block.get("qx"):
                    matching = [g for g in bw.resonanceParameters.spinGroups if g.L == block.get("l")]
                    if not matching or matching[0].competitiveChannel is None:
                        raise UnsupportedResonanceError("competitive channel metadata is not modeled")
        if bw.PoPs is not None:
            particles = list(bw.PoPs.particles.values())
            if len(particles) == 1 and particles[0].spin is not None:
                spin = particles[0].spin
                if spin.unit != "hbar" or float(spin.value) != context.target_spin:
                    raise ValueError("model target spin disagrees with context")
        approximation = getattr(bw.approximation, "value", bw.approximation)
        if approximation not in ("SingleLevel", "MultiLevel"):
            raise UnsupportedResonanceError("unknown BreitWigner approximation")
        if bw.radiusUnit not in (None, "fm"):
            raise UnsupportedResonanceError("formalism radii must be normalized to fm")
        groups = []
        seen = set()
        for group in bw.resonanceParameters.spinGroups:
            if not isinstance(group.L, (int, np.integer)) or not 0 <= group.L <= 64:
                raise UnsupportedResonanceError("BW supports integer L=0..64")
            if group.L in seen:
                raise UnsupportedResonanceError("duplicate L blocks must be normalized first")
            seen.add(group.L)
            ctx = context
            if group.atomicWeightRatio is not None:
                ctx = replace(context, atomic_weight_ratio=group.atomicWeightRatio)
                if ctx.atomic_weight_ratio != context.atomic_weight_ratio:
                    notes.append(f"L={group.L}: using declared AWRI={ctx.atomic_weight_ratio:g}")
            phase_radius = group.scatteringRadius
            if phase_radius is None:
                phase_radius = bw.scatteringRadius
            if region.scatteringRadius is not None:
                phase_radius = region.scatteringRadius
            elif radius is not None and radius.isEnergyDependent:
                if len(resonances.resolved) != 1:
                    raise UnsupportedResonanceError("global tabulated radius is ambiguous across regions")
                phase_radius = radius
            elif phase_radius is None and radius is not None:
                phase_radius = radius
            policy = bw.radiusPolicy
            if policy is not None and policy.phaseRadius is not None:
                phase_radius = policy.phaseRadius
            try:
                phase_radius = prepare_radius(phase_radius)
                mode = policy.channelMode if policy is not None else (
                    "mass" if bw.calculateChannelRadius else "phase")
                if mode == "mass":
                    channel_radius = prepare_radius(ctx.mass_channel_radius_fm)
                elif mode == "phase":
                    channel_radius = phase_radius
                elif mode == "constant":
                    channel_radius = prepare_radius(policy.channelRadius)
                else:
                    raise ValueError("unknown channel radius mode")
                # Validate domain coverage before producing a prepared object.
                phase_radius.evaluate(np.array([max(low, np.finfo(float).tiny), high]))
                channel_radius.evaluate(np.array([max(low, np.finfo(float).tiny), high]))
            except ValueError as error:
                raise UnsupportedResonanceError(str(error)) from error
            comp = group.competitiveChannel
            if comp is not None:
                if (comp.L is None or not isinstance(comp.L, (int, np.integer)) or not 0 <= comp.L <= 64):
                    raise UnsupportedResonanceError("competitive exit L is missing or unsupported")
                if not math.isfinite(comp.Q):
                    raise ValueError("competitive Q must be finite [eV]")
                if (not isinstance(comp.reactionMT, int) or not 3 <= comp.reactionMT <= 999
                        or comp.reactionMT in (18, 102)):
                    raise ValueError("competitive reaction must have a distinct partial MT")
                exit_awr = _positive(comp.atomicWeightRatio if comp.atomicWeightRatio is not None
                                     else ctx.atomic_weight_ratio, "competitive AWRI")
                exit_radius = _positive(comp.channelRadius if comp.channelRadius is not None
                                        else replace(ctx,atomic_weight_ratio=exit_awr).mass_channel_radius_fm,
                                        "competitive radius [fm]")
            levels = []
            for res in group.resonances:
                data = (res.energy, res.spin, res.totalWidth, res.neutronWidth,
                        res.captureWidth, res.fissionWidth)
                if not all(math.isfinite(v) for v in data):
                    raise ValueError("nonfinite resonance parameter")
                if res.energy == 0:
                    raise UnsupportedResonanceError("zero-energy width reference is undefined")
                if any(v < 0 for v in data[2:]):
                    raise UnsupportedResonanceError("negative BW widths are invalid")
                spin = abs(res.spin)  # ENDF-102: AJ sign carries no BW channel-spin information.
                # Allowed neutron coupling: s = |I-1/2| or I+1/2, J in |L-s|..L+s.
                allowed = set()
                for channel_spin in {abs(ctx.target_spin-.5), ctx.target_spin+.5}:
                    lower, upper = abs(group.L-channel_spin), group.L+channel_spin
                    allowed.update(lower + n for n in range(round(upper-lower)+1))
                if spin not in allowed:
                    raise ValueError("resonance J is incompatible with L and target spin")
                partial_sum = res.neutronWidth + res.captureWidth + res.fissionWidth
                # Zero is the model's unspecified GT default. Nonzero GT must agree:
                # unresolved excess is competitive physics, never silently dropped.
                gx = 0.0
                agrees = math.isclose(res.totalWidth, partial_sum, rel_tol=5e-6, abs_tol=1e-12)
                if comp is not None:
                    if res.totalWidth == 0 and partial_sum != 0:
                        raise ValueError("competitive BW requires explicit GT")
                    gx = res.totalWidth - partial_sum
                    if gx < 0:
                        if not agrees:
                            raise ValueError("negative competitive width GT-GN-GG-GF")
                        notes.append(f"E={res.energy:g}, L={group.L}: rounded negative GX={gx:g} to zero")
                        gx = 0.0
                elif res.totalWidth and not math.isclose(res.totalWidth, partial_sum,
                                                       rel_tol=5e-6, abs_tol=1e-12):
                    raise UnsupportedResonanceError("GT differs from GN+GG+GF; competitive or invalid widths")
                if comp is None and res.totalWidth and res.totalWidth != partial_sum:
                    notes.append(f"E={res.energy:g}, L={group.L}: GT={res.totalWidth:.17g}; "
                                 f"using GN+GG+GF={partial_sum:.17g} within rounding tolerance")
                if partial_sum + gx == 0:
                    continue  # identically inactive level, no division by zero
                reference_radius = channel_radius.evaluate(abs(res.energy))
                pr = neutral_channel_functions(group.L, np.sqrt(ctx.k_squared_per_ev*abs(res.energy))*reference_radius)[0]
                if pr <= 0:
                    raise UnsupportedResonanceError("neutron reference penetrability underflows")
                if gx:
                    ratio = ctx.atomic_weight_ratio/(1+ctx.atomic_weight_ratio)
                    reference = abs(ratio*res.energy+comp.Q)
                    if reference == 0:
                        raise UnsupportedResonanceError("competitive reference is exactly at threshold")
                    coefficient = 2*ctx.neutron_mass_mev*1e-6/ctx.hbar_c_mev_fm**2*exit_awr/(1+exit_awr)
                    pxr = neutral_channel_functions(comp.L, np.sqrt(coefficient*reference)*exit_radius)[0]
                    if pxr <= 0:
                        raise UnsupportedResonanceError("competitive reference penetrability underflows")
                levels.append(Level(res.energy, spin, res.neutronWidth,
                                    res.captureWidth, res.fissionWidth, gx))
            groups.append(Group(group.L, channel_radius, phase_radius, tuple(levels), context=ctx,
                competitive_mt=None if comp is None else comp.reactionMT,
                competitive_l=None if comp is None else comp.L,
                competitive_q=0. if comp is None else comp.Q,
                competitive_awr=None if comp is None else exit_awr,
                competitive_radius=None if comp is None else exit_radius,
                competitive_in_background=False if comp is None else comp.inEvaluatedBackground))
        if not groups:
            raise ValueError("BW needs at least one L block (may contain no levels)")
        prepared.append(PreparedRegion(low, high, approximation, tuple(groups)))
    if resonances.unresolved is not None:
        from .unresolved import prepare_unresolved
        prepared.append(prepare_unresolved(resonances.unresolved, context, notes))
    prepared.sort(key=lambda r: r.low)
    if any(a.high > b.low for a,b in zip(prepared[:-1],prepared[1:])):
        raise UnsupportedResonanceError("overlapping resonance regions")
    return PreparedResonances(context, tuple(prepared), tuple(notes))


def region_mts(region):
    return {1,2,18,102} | {mt for g in region.groups for mt in getattr(g,'reaction_mts',())} | {
        g.competitive_mt for g in region.groups if g.competitive_mt is not None}


def group_radii(group):
    channels = getattr(group,'channels',())
    return tuple(r for c in channels for r in (c.radius,c.phase_radius)) if channels else (group.channel_radius,group.phase_radius)


def group_knots(group):
    knots = set()
    for average in getattr(group,'averages',()) + ((group.spacing,) if hasattr(group,'spacing') else ()):
        knots.update(average.knots)
    for c in getattr(group,'channels',()):
        if not c.effective:knots.add(c.threshold)
        for table in (c.external,c.phase_function):
            if table is not None and table.kind == 'table':
                for curve in table.real+table.imaginary:knots.update(curve.x)
    return knots


def group_breaks(group):
    breaks = set()
    for average in getattr(group,'averages',()) + ((group.spacing,) if hasattr(group,'spacing') else ()):
        breaks.update(average.breaks)
    for c in getattr(group,'channels',()):
        if not c.effective and c.penetrability == 'unity':breaks.add(c.threshold)
        for table in (c.external,c.phase_function):
            if table is None or table.kind!='table':continue
            for curves in (table.real,table.imaginary):
                for index,curve in enumerate(curves):
                    if curve.law == 1:breaks.update(curve.x[1:])
                    if index and curves[index-1].y[-1] != curve.y[0]:breaks.add(curve.x[0])
    return breaks


def evaluate_region(energies,region,context,diagnostics=None):
    if region.approximation == 'Unresolved':
        from .unresolved import evaluate_unresolved
        return evaluate_unresolved(energies,region,diagnostics)
    if region.approximation == 'RMatrixNeutral':
        from .r_matrix import evaluate_rml
        out = {mt:np.zeros_like(energies) for mt in region_mts(region)}
        for start in range(0,len(energies),128):
            sl = slice(start,start+128)
            for mt,value in evaluate_rml(energies[sl],region.groups,context,diagnostics).items():out[mt][sl] = value
        return out
    if region.approximation=='ReichMoore':
        # Limit temporary level/channel arrays independently of caller block size.
        out={mt:np.zeros_like(energies) for mt in (1,2,18,102)}
        for start in range(0,len(energies),128):
            sl=slice(start,start+128)
            for mt,value in evaluate_rm(energies[sl],region.groups,context,diagnostics).items():out[mt][sl]=value
        return out
    return evaluate_bw(energies,region.groups,region.approximation,context)

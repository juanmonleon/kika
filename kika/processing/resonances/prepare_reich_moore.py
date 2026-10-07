"""Validation and immutable normalization of neutron Reich-Moore models."""
from dataclasses import replace
import math
import numpy as np
from .context import NeutronContext
from .radii import prepare_radius, RadiusFunction
from .channel_functions import neutral_channel_functions
from .reich_moore import RMLevel,RMGroup


def prepare_rm(region,resonances,context,notes):
    from .prepare import UnsupportedResonanceError
    f=region.formalism
    if f.approximation!='ReichMoore':raise UnsupportedResonanceError('R4 supports ReichMoore only; full RML remains deferred')
    if f.angularLCount is not None and (not isinstance(f.angularLCount,int) or f.angularLCount<0):raise ValueError('RM angular L count must be nonnegative integer')
    if f.relativisticKinematics:raise UnsupportedResonanceError('relativistic RM is not implemented')
    if f.boundaryCondition not in (None,'EliminateShiftFunction'):
        raise UnsupportedResonanceError('R4 certifies the eliminated-shift RM convention only')
    if f.radiusUnit not in (None,'fm'):raise UnsupportedResonanceError('RM radii require fm')
    if f.PoPs is not None:
        particles=list(f.PoPs.particles.values())
        if len(particles)==1 and particles[0].spin is not None:
            spin=particles[0].spin
            if spin.unit!='hbar' or float(spin.value)!=context.target_spin:raise ValueError('RM target spin disagrees with context')
    reactions={r.label:r for r in f.resonanceReactions}
    if len(reactions)!=len(f.resonanceReactions):raise ValueError('duplicate resonance reaction labels')
    def kind(r):
        if r.reactionMT==2 or (r.reactionMT is None and r.label=='elastic'):return 'neutron'
        if r.reactionMT==102 or (r.reactionMT is None and r.label=='capture'):return 'capture'
        if r.reactionMT in (18,19) or (r.reactionMT is None and r.label in ('fission','fissionA','fissionB')):return 'fission'
        raise UnsupportedResonanceError('RM channel requires a resolved elastic/capture/fission identity')
    roles={label:kind(r) for label,r in reactions.items()}
    for label,r in reactions.items():
        if r.eliminated!=(roles[label]=='capture'):raise UnsupportedResonanceError('RM eliminates the radiative reaction only')
        if r.kinematics is not None:
            expected='calculate' if roles[label]=='neutron' else 'unity'
            if r.kinematics.penetrability not in (None,'automatic',expected) or r.kinematics.shift not in (None,'zero'):
                raise UnsupportedResonanceError('channel PNT/SHF is outside the eliminated-shift RM convention')
            if roles[label]=='neutron':
                projectile=r.kinematics.particleA;target=r.kinematics.particleB
                if r.kinematics.effective or projectile.massRatio!=1. or projectile.charge!=0. or projectile.spin!=.5:
                    raise UnsupportedResonanceError('R4 entrance must be a physical neutron/target pair')
                if target.spin!=context.target_spin:raise ValueError('declared RM pair target spin disagrees with context')
    prepared=[];covered={};radii_by_l={}
    def allowed(l,s,j):return abs(l-s)<=j<=l+s and float(j-abs(l-s)).is_integer()
    for sg in f.spinGroups:
        if sg.additionalPhaseShift is not None or sg.phaseShiftMode:raise UnsupportedResonanceError('RM additional phase is not implemented')
        if len(sg.widths)!=len(sg.energies) or (sg.spins and len(sg.spins)!=len(sg.energies)):raise ValueError('RM table lengths disagree')
        kinds=[]
        for channel in sg.channels:
            if channel.resonanceReaction not in roles:raise ValueError('RM channel refers to an unknown reaction')
            kinds.append(roles[channel.resonanceReaction])
            if channel.additionalPhaseShift is not None or channel.phaseShiftMode:
                raise UnsupportedResonanceError('RM channel phase requires complete RML pair normalization')
            if channel.externalRMatrix is not None or channel.tabulatedBackground is not None:
                raise UnsupportedResonanceError('RM external R-matrix is deferred to RML')
            if channel.boundaryConditionValue not in (None,0.):raise UnsupportedResonanceError('nonzero RM boundary value is not supported')
            if channel.radiusUnit not in (None,'fm'):raise UnsupportedResonanceError('RM channel radii require fm')
        if kinds.count('neutron')!=1 or kinds.count('capture')!=1:
            raise UnsupportedResonanceError('R4 normalization requires one entrance and one eliminated capture per group')
        ni=kinds.index('neutron');gi=kinds.index('capture');fi=[i for i,k in enumerate(kinds) if k=='fission']
        ch=sg.channels[ni];l=ch.L
        if not isinstance(l,(int,np.integer)) or not 0<=l<=64:raise UnsupportedResonanceError('RM requires integer L=0..64')
        ctx=context if sg.atomicWeightRatio is None else replace(context,atomic_weight_ratio=sg.atomicWeightRatio)
        if ctx!=context:notes.append(f'RM L={l}: declared AWRI={ctx.atomic_weight_ratio:g}')
        rr=reactions[ch.resonanceReaction]
        if rr.kinematics is not None:
            declared=rr.kinematics.particleB.massRatio
            if sg.atomicWeightRatio is not None and sg.atomicWeightRatio!=declared:
                raise ValueError('RM group AWRI and physical pair mass disagree')
            if ctx.atomic_weight_ratio!=declared:
                ctx=replace(ctx,atomic_weight_ratio=declared)
                notes.append(f'RM {sg.label}: using declared neutron-pair target mass ratio={declared:g}')
            if sg.parity is not None and sg.parity!=rr.kinematics.particleA.parity*rr.kinematics.particleB.parity*(-1)**l:
                raise ValueError('RM spin-group parity is incompatible with its neutron entrance')
        phase=ch.hardSphereRadius
        if phase is None:phase=ch.scatteringRadius
        if phase is None:phase=rr.hardSphereRadius if rr.hardSphereRadius is not None else rr.scatteringRadius
        if phase is None:phase=f.scatteringRadius
        if phase is None:phase=resonances.scatteringRadius
        # NRO overrides global AP, but a declared APL remains local.
        if ch.hardSphereRadius is None and ch.scatteringRadius is None and region.scatteringRadius is not None:phase=region.scatteringRadius
        policy=f.radiusPolicy
        if policy is not None and policy.phaseRadius is not None:phase=policy.phaseRadius
        mode=policy.channelMode if policy is not None else ('mass' if f.calculateChannelRadius else 'phase')
        true_radius=(ctx.mass_channel_radius_fm if mode=='mass' else policy.channelRadius if mode=='constant' else ch.scatteringRadius if ch.scatteringRadius is not None else phase)
        if mode not in ('mass','phase','constant'):raise UnsupportedResonanceError('unknown RM radius policy')
        phase_zero = (isinstance(phase,(int,float,np.number)) and phase==0.) or (
            phase is not None and not getattr(phase,'isEnergyDependent',True)
            and getattr(phase,'constant',None)==0. and getattr(phase,'unit',None) in (None,'fm'))
        phase=RadiusFunction(constant=0.) if phase_zero else prepare_radius(phase)
        radius=prepare_radius(true_radius)
        phase.evaluate(np.array([max(region.domainMin,np.finfo(float).tiny),region.domainMax]))
        radius.evaluate(np.array([max(region.domainMin,np.finfo(float).tiny),region.domainMax]))
        old=radii_by_l.get(l)
        if old is not None and old!=(phase,radius,ctx):raise UnsupportedResonanceError('inconsistent neutron radii/AWRI within one L')
        radii_by_l[l]=(phase,radius,ctx)
        buckets={}
        for i,(energy,widths) in enumerate(zip(sg.energies,sg.widths)):
            if len(widths)!=len(kinds) or not all(math.isfinite(v) for v in [energy,*widths]):raise ValueError('nonfinite or malformed RM row')
            signed=sg.spins[i] if sg.spins else sg.spin
            if signed is None or not math.isfinite(signed):raise ValueError('RM requires a declared J')
            j=abs(signed)
            high=context.target_spin+.5;low=abs(context.target_spin-.5)
            if sg.spins:
                s=low if signed<0 else high
                if not allowed(l,s,j) and signed>=0 and allowed(l,low,j):s=low
            else:
                s=ch.channelSpin
                if s is None:
                    choices={s0 for s0 in (low,high) if allowed(l,s0,j)}
                    if len(choices)!=1:raise UnsupportedResonanceError('ambiguous RM entrance channel spin')
                    s=choices.pop()
            if s not in (low,high) or not allowed(l,s,j):raise ValueError('RM J/L/channel-spin coupling is invalid')
            if widths[gi]<0 and not f.reducedWidthAmplitudes:raise ValueError('negative radiative width')
            if not f.reducedWidthAmplitudes and widths[ni]<0 and sg.spins:raise ValueError('negative LRF3 neutron width')
            if energy==0 and widths[ni]!=0 and not f.reducedWidthAmplitudes:raise UnsupportedResonanceError('zero-energy RM neutron width reference')
            pr=1. if f.reducedWidthAmplitudes or widths[ni]==0 else float(neutral_channel_functions(l,np.sqrt(ctx.k_squared_per_ev*abs(energy))*radius.evaluate(abs(energy)))[0])
            if pr<=0:raise UnsupportedResonanceError('RM reference penetrability underflows')
            an=widths[ni] if f.reducedWidthAmplitudes else math.copysign(math.sqrt(abs(widths[ni])/(2*pr)),widths[ni])
            af=tuple(widths[k] if f.reducedWidthAmplitudes else math.copysign(math.sqrt(abs(widths[k])/2),widths[k]) for k in fi)
            capture=2*widths[gi]**2 if f.reducedWidthAmplitudes else widths[gi]
            gn=2*pr*an**2 if not f.reducedWidthAmplitudes else (2*float(neutral_channel_functions(l,np.sqrt(ctx.k_squared_per_ev*abs(energy))*radius.evaluate(abs(energy)))[0])*an**2 if energy else 0.)
            level=RMLevel(energy,j,gn,capture,2*sum(a*a for a in af),fission_amplitudes=af,neutron_amplitude=an)
            buckets.setdefault((j,s),[]).append(level)
        # An empty block still declares L coverage and its hard-sphere radius.
        for (j,s),levels in buckets.items():
            key=(l,j,s)
            if key in covered:raise UnsupportedResonanceError('duplicate RM entrance sector; normalize its levels into one group')
            group=RMGroup(l,radius,phase,tuple(levels),context=ctx,spin=j,channel_spin=s)
            covered[key]=group;prepared.append(group)
    if not radii_by_l:raise ValueError('RM requires at least one declared neutron L')
    # Complete exactly the declared orbital sectors. NLSC is angular metadata,
    # not permission to invent additional cross-section L values.
    for l,(phase,radius,ctx) in radii_by_l.items():
        for s in {abs(ctx.target_spin-.5),ctx.target_spin+.5}:
            first=abs(l-s)
            for index in range(round(l+s-first)+1):
                j=first+index
                if (l,j,s) not in covered:
                    prepared.append(RMGroup(l,radius,phase,(),context=ctx,spin=j,channel_spin=s))
    return tuple(prepared)

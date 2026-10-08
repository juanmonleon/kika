"""Immutable normalization of neutron-incidence KRM3/KRM4 channels."""
from dataclasses import dataclass
import math
import numpy as np
from .breit_wigner import Level
from .r_matrix import RMLChannel, RMLGroup
from .radii import prepare_radius, RadiusFunction


@dataclass(frozen=True)
class ExternalFunction:
    kind: str
    terms: tuple = ()
    real: tuple = ()
    imaginary: tuple = ()

    def evaluate(self, energy):
        e = np.asarray(energy)
        if self.kind == 'table':
            def evaluate(curves):
                out = np.zeros_like(e); covered = np.zeros(len(e), bool)
                for curve in curves:
                    mask = (e >= curve.x[0]) & (e <= curve.x[-1])
                    out[mask] = curve.evaluate(e[mask]); covered |= mask
                if not np.all(covered):
                    raise ValueError('external function does not cover evaluation energies')
                return out
            return evaluate(self.real)+1j*evaluate(self.imaginary)
        t = dict(self.terms); down = t['singularityEnergyBelow']; up = t['singularityEnergyAbove']
        if np.any(e <= down) or np.any(e >= up):
            raise ValueError('external statistical function outside its singularities')
        if self.kind == 'SAMMY':
            return (t['constantExternalR']+t['linearExternalR']*e+t['quadraticExternalR']*e**2
                    -t['linearLogarithmicCoefficient']*(up-down)
                    -(t['constantLogarithmicCoefficient']+t['linearLogarithmicCoefficient']*e)*np.log((up-e)/(e-down)))
        # ENDF D.1.7.5-3 uses inverse hyperbolic tangent, not ordinary arctangent.
        u = (2*e-up-down)/(up-down)
        return (t['constantExternalR']+2*t['poleStrength']*np.arctanh(u)
                +1j*t['averageRadiationWidth']/(up-down)/(1-u*u))


def prepare_external(channel, low, high):
    from .prepare import UnsupportedResonanceError
    from .assemble import BackgroundCurve
    from kika.nuclear_data.model.functions import XYs1d, Regions1d
    if channel.externalRMatrix is not None and channel.tabulatedBackground is not None:
        raise ValueError('channel declares two external backgrounds')
    if channel.tabulatedBackground is not None:
        def snapshot(function):
            curves = function.function1ds if isinstance(function, Regions1d) else [function]
            out = []
            for curve in curves:
                if not isinstance(curve, XYs1d) or curve.domainUnit != 'eV' or curve.rangeUnit not in ('', None):
                    raise UnsupportedResonanceError('external tables require dimensionless XYs/Regions with eV domain')
                x, y = np.asarray(curve.xs), np.asarray(curve.ys); law = curve.endfInterpolationCode
                from kika.algebra import split_at_discontinuities
                # The shared algebra validates laws and retains both sides of
                # each repeated-energy step; no interpolation crosses it.
                for xx,yy,_ in split_at_discontinuities(x,y,law):
                    if out and out[-1].x[-1] != xx[0]:
                        raise UnsupportedResonanceError('external table regions must be adjacent')
                    out.append(BackgroundCurve(tuple(xx), tuple(yy), law))
            if not out or out[0].x[0] > low or out[-1].x[-1] < high:
                raise ValueError('external table must cover the resolved domain')
            return tuple(out)
        return ExternalFunction('table', real=snapshot(channel.tabulatedBackground.real),
                                imaginary=snapshot(channel.tabulatedBackground.imaginary))
    external = channel.externalRMatrix
    if external is None:
        return None
    units = dict(singularityEnergyBelow='eV', singularityEnergyAbove='eV', constantExternalR='',
                 linearExternalR='1/eV', quadraticExternalR='1/eV**2', constantLogarithmicCoefficient='',
                 linearLogarithmicCoefficient='1/eV', poleStrength='', averageRadiationWidth='eV')
    permitted = ({'constantExternalR', 'linearExternalR', 'quadraticExternalR',
                  'constantLogarithmicCoefficient', 'linearLogarithmicCoefficient'} if external.type == 'SAMMY'
                 else {'constantExternalR', 'poleStrength', 'averageRadiationWidth'}) | {'singularityEnergyBelow','singularityEnergyAbove'}
    if any(t.label not in permitted for t in external.terms):
        raise UnsupportedResonanceError('unknown external R-matrix coefficient')
    terms = {key:0. for key in permitted}
    for term in external.terms:
        terms[term.label] = term.convertedTo(units[term.label]).value
    if not all(math.isfinite(v) for v in terms.values()) or not terms['singularityEnergyBelow'] < low < high < terms['singularityEnergyAbove']:
        raise ValueError('external singularities must bracket the resolved domain')
    if terms.get('averageRadiationWidth', 0.) < 0:
        raise ValueError('negative external radiation width')
    return ExternalFunction(external.type, tuple(sorted(terms.items())))


def prepare_rml(region, resonances, context, notes):
    from .prepare import UnsupportedResonanceError
    f = region.formalism
    if f.approximation not in ('ReichMoore','RMatrixLimited'):
        raise UnsupportedResonanceError('RML requires KRM3 or KRM4')
    eliminated_capture = f.approximation == 'ReichMoore'
    if f.radiusPolicy is not None:
        raise UnsupportedResonanceError('RML uses explicit per-channel APT/APE, not an unnormalized global radius policy')
    if f.boundaryCondition not in (None, 'Given', 'NegativeOrbitalMomentum', 'EliminateShiftFunction', 'Brune'):
        raise UnsupportedResonanceError('Brune/unknown boundary convention requires a separate certification')
    reactions = {r.label:r for r in f.resonanceReactions}
    if len(reactions) != len(f.resonanceReactions):
        raise ValueError('duplicate resonance reaction labels')
    cm = context.atomic_weight_ratio/(1+context.atomic_weight_ratio)
    prepared = []; coverage = {}
    def allowed(l, s, j):
        return abs(l-s) <= j <= l+s and float(j-abs(l-s)).is_integer()
    for sg in f.spinGroups:
        if sg.spins:
            raise UnsupportedResonanceError('RML requires one J per spin group, not a legacy per-row AJ table')
        if sg.spin is None or not math.isfinite(sg.spin) or sg.spin < 0 or not float(2*sg.spin).is_integer():
            raise ValueError('invalid RML group spin')
        if sg.parity not in (-1,1):raise ValueError('RML group requires declared parity +1 or -1')
        if sg.additionalPhaseShift is not None or sg.phaseShiftMode not in (None,0,1) or (sg.phaseShiftMode and any(ch.phaseShiftMode is None for ch in sg.channels)):
            raise UnsupportedResonanceError('KPS requires explicit per-channel LPS; legacy group phases are ambiguous')
        if sg.atomicWeightRatio not in (None, context.atomic_weight_ratio):
            raise ValueError('RML group AWRI disagrees with incident context')
        channels = []; indexes = []; capture = []; entrances = []; identities = set()
        for index, ch in enumerate(sg.channels):
            if ch.resonanceReaction not in reactions:
                raise ValueError('unknown RML resonance reaction')
            rr = reactions[ch.resonanceReaction]; pair = rr.kinematics
            phase_function = None; phase_absorption_mt = None
            if ch.phaseShiftMode not in (None,0,1) or (ch.phaseShiftMode==1 and ch.additionalPhaseShift is None) or (ch.phaseShiftMode==0 and ch.additionalPhaseShift is not None):
                raise ValueError('invalid per-channel LPS declaration')
            if ch.additionalPhaseShift is not None:
                from types import SimpleNamespace
                phase_function = prepare_external(SimpleNamespace(externalRMatrix=None,tabulatedBackground=ch.additionalPhaseShift),region.domainMin,region.domainMax)
                absorptive = any(any(v!=0 for v in curve.y) for curve in phase_function.imaginary)
                if ch.phaseAbsorptionReaction is not None or absorptive:
                    owner = reactions.get(ch.phaseAbsorptionReaction)
                    mt_owner = None if owner is None else owner.reactionMT
                    allowed_owner = isinstance(mt_owner,int) and (
                        mt_owner in (18,19,102,103,104,105,106,107)
                        or 51<=mt_owner<=91 or 600<=mt_owner<=849)
                    if not allowed_owner:
                        raise UnsupportedResonanceError('absorptive phase requires declared reaction ownership with a supported nonelastic MT')
                    if any(any(v>0 for v in curve.y) for curve in phase_function.imaginary):
                        raise ValueError('passive KPS requires imaginary phase <= 0 for exp(-i phase)')
                    phase_absorption_mt = 18 if mt_owner==19 else mt_owner
                if rr.eliminated:raise UnsupportedResonanceError('eliminated channel cannot declare an external phase')
            if ch.phaseAbsorptionReaction is not None and phase_function is None:
                raise ValueError('phase absorption ownership requires a phase function')
            if rr.reactionMT is None or pair is None:
                raise UnsupportedResonanceError('RML requires normalized reaction MT and physical/effective pair data')
            if not isinstance(rr.reactionMT,int) or rr.reactionMT not in (2,18,19,102,103,104,105,106,107) and not (51<=rr.reactionMT<=91 or 600<=rr.reactionMT<=849):
                raise UnsupportedResonanceError('unsupported two-body RML reaction MT')
            if rr.eliminated:
                if not eliminated_capture or rr.reactionMT != 102 or ch.externalRMatrix is not None or ch.tabulatedBackground is not None:
                    raise UnsupportedResonanceError('KRM3 eliminates capture without an external capture channel')
                if not pair.effective or pair.penetrability not in ('automatic','unity') or pair.shift != 'zero':
                    raise UnsupportedResonanceError('eliminated capture requires effective unity P and zero shift')
                capture.append(index); continue
            if rr.reactionMT == 102 and eliminated_capture:
                raise UnsupportedResonanceError('KRM3 requires eliminated capture')
            if not eliminated_capture and (ch.externalRMatrix is not None or ch.tabulatedBackground is not None):
                raise UnsupportedResonanceError('ENDF background R-matrix is defined for KRM3 only')
            if pair.penetrability not in ('automatic', 'calculate', 'unity') or pair.shift not in ('zero', 'calculate', 'brune'):
                raise UnsupportedResonanceError('unknown PNT/SHF or Brune shift')
            if pair.shift == 'brune' and f.boundaryCondition != 'Brune':
                raise ValueError('Brune shift requires the Brune boundary convention')
            if ch.radiusUnit not in (None, 'fm') or f.radiusUnit not in (None, 'fm'):
                raise UnsupportedResonanceError('RML radii must be normalized to fm')
            l, s = ch.L, ch.channelSpin
            if not isinstance(l, (int, np.integer)) or not 0 <= l <= 64 or s is None or not math.isfinite(s):
                raise ValueError('invalid RML L/s/J coupling')
            mt = 18 if rr.reactionMT == 19 else rr.reactionMT
            effective = mt in (18,102)
            strength = 0.
            kinematics = None
            if effective:
                if not pair.effective or pair.penetrability not in ('automatic','unity') or pair.shift != 'zero':
                    raise UnsupportedResonanceError('effective fission/capture requires unity P and zero shift')
                k2 = 0.; q = 0.; mode = 'unity'; radius = RadiusFunction(constant=1.); phase = RadiusFunction(constant=0.)
            else:
                a, b = pair.particleA, pair.particleB
                if not allowed(l,s,sg.spin):raise ValueError('invalid physical channel L/s/J coupling')
                if pair.effective or any(not math.isfinite(v) or v<=0 for v in (a.massRatio,b.massRatio)):
                    raise UnsupportedResonanceError('physical RML requires positive finite two-body masses')
                if any(p.parity not in (-1,1) or not math.isfinite(p.charge) or p.charge<0
                       or not math.isfinite(p.spin) or p.spin<0 or not float(2*p.spin).is_integer() for p in (a,b)):
                    raise ValueError('invalid physical pair parity/charge/spin')
                if not abs(a.spin-b.spin)<=s<=a.spin+b.spin or not float(s-abs(a.spin-b.spin)).is_integer():
                    raise ValueError('channel spin disagrees with pair spins')
                if mt==2 or 51<=mt<=91:
                    if a.massRatio!=1. or a.charge!=0 or a.spin!=.5 or a.parity!=1:
                        raise UnsupportedResonanceError('neutron channels require a neutral neutron ejectile')
                elif a.charge*b.charge<=0:
                    raise UnsupportedResonanceError('charged exit MT requires a repulsive charged pair')
                if sg.parity is not None and sg.parity != a.parity*b.parity*(-1)**l:
                    raise ValueError('group parity disagrees with physical channel')
                if rr.Q is None or not math.isfinite(rr.Q):
                    raise ValueError('RML physical pair needs a finite declared Q [eV]')
                q = rr.Q
                if mt == 2:
                    if q != 0. or b.massRatio != context.atomic_weight_ratio or b.spin != context.target_spin:
                        raise ValueError('elastic pair disagrees with incident context/Q')
                    entrances.append(len(channels))
                mu = context.neutron_mass_mev*a.massRatio*b.massRatio/(a.massRatio+b.massRatio)
                k2 = 2*mu*1e-6/context.hbar_c_mev_fm**2
                from scipy.constants import alpha
                strength = alpha*a.charge*b.charge*mu/context.hbar_c_mev_fm
                if f.relativisticKinematics:
                    from .kinematics import RelativisticPair
                    kinematics = RelativisticPair(context.neutron_mass_mev,context.neutron_mass_mev*context.atomic_weight_ratio,
                        a.massRatio*context.neutron_mass_mev,b.massRatio*context.neutron_mass_mev,q,context.hbar_c_mev_fm)
                mode = 'calculate' if pair.penetrability == 'automatic' else pair.penetrability
                declared = ch.scatteringRadius if ch.scatteringRadius is not None else rr.scatteringRadius
                if declared is None:
                    declared = region.scatteringRadius or f.scatteringRadius or resonances.scatteringRadius
                radius = prepare_radius(declared)
                declared_phase = ch.hardSphereRadius if ch.hardSphereRadius is not None else rr.hardSphereRadius
                if declared_phase is None:declared_phase = declared
                phase_zero = isinstance(declared_phase,(int,float)) and declared_phase == 0.
                phase = RadiusFunction(constant=0.) if phase_zero else prepare_radius(declared_phase)
                for r in (radius,phase):r.evaluate(np.array([max(region.domainMin,np.finfo(float).tiny),region.domainMax]))
            shift = pair.shift
            if f.boundaryCondition == 'Brune':
                if ch.boundaryConditionValue is not None or f.boundaryConditionValue is not None:
                    raise ValueError('Brune parameters do not have boundary constants')
                shift = 'calculate' if pair.shift == 'brune' else pair.shift; boundary = 0.
            elif f.boundaryCondition == 'EliminateShiftFunction':
                if ch.boundaryConditionValue not in (None,0.):
                    raise ValueError('explicit boundary conflicts with eliminated-shift convention')
                shift = 'zero'; boundary = 0.
            elif f.boundaryCondition == 'NegativeOrbitalMomentum':
                boundary = -float(l) if ch.boundaryConditionValue is None else ch.boundaryConditionValue
            else:
                boundary = ch.boundaryConditionValue
                if boundary is None:boundary = f.boundaryConditionValue
                if boundary is None:
                    if f.boundaryCondition == 'Given':raise ValueError('Given boundary requires a declared channel/global value')
                    boundary = 0.
            if not math.isfinite(boundary):raise ValueError('nonfinite boundary constant')
            identity = (ch.resonanceReaction,l,s)
            if identity in identities:raise ValueError('duplicate coherent channel')
            identities.add(identity)
            channel = RMLChannel(mt,l,s,q,cm,k2,radius,phase,mode,shift,boundary,effective,
                                 prepare_external(ch,region.domainMin,region.domainMax),strength,phase_function,kinematics,rr.label,phase_absorption_mt)
            channels.append(channel); indexes.append(index)
            if mt == 2:
                previous = coverage.get(l)
                signature = (radius, phase)
                if previous is not None and previous != signature:
                    raise UnsupportedResonanceError('missing-sector completion requires consistent entrance radii per L')
                coverage[l] = signature
        if not entrances or len(capture) != int(eliminated_capture):
            raise UnsupportedResonanceError('RML requires elastic entrance and the capture channels of its formalism')
        if len(sg.energies) != len(sg.widths):raise ValueError('RML table lengths disagree')
        reduced = []; radiation = []; levels = []
        for er, row in zip(sg.energies, sg.widths):
            if len(row) != len(sg.channels) or not all(math.isfinite(v) for v in [er,*row]):
                raise ValueError('nonfinite/malformed RML row')
            gg = (2*row[capture[0]]**2 if f.reducedWidthAmplitudes else abs(row[capture[0]])) if eliminated_capture else 0.
            amplitudes = []; physical = []
            for i,ch in zip(indexes,channels):
                width = row[i]
                if f.reducedWidthAmplitudes:
                    amplitude = width
                    pr = ch.functions(np.array([max(abs(er),np.finfo(float).tiny)]),logarithmic=True)[0][0]
                else:
                    if ch.effective or ch.penetrability == 'unity':pr = 1.
                    else:
                        # ENDF negative dummy levels use the absolute LAB
                        # resonance energy before the channel Q is applied.
                        reference = abs(float(ch.channel_energy(abs(er))))
                        if reference == 0 and width != 0:raise UnsupportedResonanceError('physical width reference lies at channel threshold')
                        if reference == 0:pr = 1.
                        else:
                            from .channel_functions import neutral_channel_functions
                            rho = np.sqrt(ch.k_squared(reference))*ch.radius.evaluate(abs(er))
                            if ch.charge_strength:
                                from .coulomb import charged_channel_log_functions
                                eta = ch.charge_strength/np.sqrt(ch.k_squared(reference))
                                log_pr = float(charged_channel_log_functions(ch.l,eta,rho)[0])
                                pr = math.exp(log_pr)
                            else:
                                pr = float(neutral_channel_functions(ch.l,rho)[0])
                    if pr < np.finfo(float).tiny and width != 0:
                        if not ch.charge_strength:raise UnsupportedResonanceError('RML reference penetrability underflows')
                        log_amplitude = .5*(math.log(abs(width))-math.log(2.)-log_pr)
                        if log_amplitude > .5*math.log(np.finfo(float).max):
                            raise FloatingPointError('IFG0 reduced amplitude squared exceeds the finite solver range')
                        amplitude = math.copysign(math.exp(log_amplitude),width)
                    else:
                        amplitude = 0. if width == 0 else math.copysign(math.sqrt(abs(width)/(2*pr)),width)
                if not math.isfinite(amplitude) or abs(amplitude)>math.sqrt(np.finfo(float).max):
                    raise FloatingPointError('reduced amplitude squared exceeds the finite solver range')
                amplitudes.append(amplitude)
                physical.append(2*pr*amplitude**2 if f.reducedWidthAmplitudes else abs(width))
            reduced.append(tuple(amplitudes)); radiation.append(gg)
            levels.append(Level(er,sg.spin,sum(v for v,c in zip(physical,channels) if c.mt==2),gg,
                                sum(v for v,c in zip(physical,channels) if c.mt==18),
                                sum(v for v,c in zip(physical,channels) if c.mt not in (2,18))))
        first = channels[entrances[0]]
        prepared.append(RMLGroup(first.l, first.radius, first.phase_radius, tuple(levels), context=context,
            spin=sg.spin, channels=tuple(channels), reduced=tuple(reduced), radiation=tuple(radiation), entrances=tuple(entrances),parity=sg.parity))
    if not prepared:raise ValueError('RML requires groups')
    # ENDF may split the same J/parity across channel-spin records. Rebuild
    # one sector, zero-extending each row to its union of declared channels.
    # Shared channels sum coherently through R; hard-sphere scattering is
    # counted once. Incompatible radius/phase/boundary definitions reject.
    buckets = {}
    for group in prepared:buckets.setdefault((group.spin,group.parity),[]).append(group)
    merged = []
    for sector,groups in buckets.items():
        if len(groups)==1:merged.extend(groups);continue
        columns = [];positions = {};maps = []
        for group in groups:
            mapping = []
            for channel in group.channels:
                key=(channel.reaction_label,channel.l,channel.spin)
                if key in positions:
                    position=positions[key]
                    if columns[position]!=channel:
                        raise UnsupportedResonanceError('split coherent sector has incompatible channel definitions')
                else:
                    position=len(columns);positions[key]=position;columns.append(channel)
                mapping.append(position)
            maps.append(mapping)
        rows=[];levels=[];radiation=[]
        for group,mapping in zip(groups,maps):
            for row in group.reduced:
                extended=np.zeros(len(columns));extended[mapping]=row;rows.append(tuple(extended))
            levels.extend(group.levels);radiation.extend(group.radiation)
        first=groups[0]
        merged.append(RMLGroup(first.l,first.channel_radius,first.phase_radius,tuple(levels),context=context,
            spin=sector[0],parity=sector[1],channels=tuple(columns),reduced=tuple(rows),radiation=tuple(radiation),
            entrances=tuple(i for i,c in enumerate(columns) if c.mt==2)))
    prepared=merged
    represented = {(g.spin,c.l,c.spin) for g in prepared for c in g.channels if c.mt==2}
    for l,(radius,phase) in coverage.items():
        for s in {abs(context.target_spin-.5),context.target_spin+.5}:
            first = abs(l-s)
            for index in range(round(l+s-first)+1):
                j = first+index
                if (j,l,s) in represented:continue
                if any(c.phase_function is not None for g in prepared for c in g.channels if c.mt==2 and c.l==l):
                    raise UnsupportedResonanceError('external entrance phases require explicitly complete potential sectors')
                k2 = context.k_squared_per_ev/cm
                kin = next(c.kinematics for g in prepared for c in g.channels if c.mt==2)
                c = RMLChannel(2,l,s,0.,cm,k2,radius,phase,'calculate','zero',0.,kinematics=kin)
                prepared.append(RMLGroup(l,radius,phase,(),context=context,spin=j,channels=(c,),entrances=(0,)))
    if f.boundaryCondition == 'Brune':
        from .brune import prepare_brune
        prepared = [prepare_brune(g) for g in prepared]
        notes.append('Brune alternative parameters; generalized level metric retained, including eliminated absorption')
    notes.append('RML KRM3/KRM4, neutral incidence and repulsive charged exits; closed-channel shift retained')
    return tuple(prepared)


def normalize_suite_pairs(suite, context):
    """Resolve native model links and PoPs for RML, without reading a format.

    Returns a copy only when normalization is needed. It never fills a missing
    physical mass, spin, parity or charge. ENDF-only PNT/SHF remain explicit in
    existing ChannelKinematics rather than being reinterpreted through GNDS.
    """
    from copy import deepcopy
    from .prepare import UnsupportedResonanceError
    from .suite import _entries, _target
    from kika.nuclear_data.model.resonances import RMatrix, ChannelParticle, ChannelKinematics
    resolved = suite.resonances
    if resolved is None:return resolved
    candidates = [r for r in resolved.resolved if isinstance(r.formalism,RMatrix)
                  and (r.formalism.approximation=='RMatrixLimited' or r.formalism.relativisticKinematics or r.formalism.boundaryCondition in ('Given','NegativeOrbitalMomentum','Brune')
                       or any(ch.additionalPhaseShift is not None or ch.phaseShiftMode or ch.phaseAbsorptionReaction is not None for g in r.formalism.spinGroups for ch in g.channels)
                       or any(rr.reactionMT not in (None,2,18,19,102) for rr in r.formalism.resonanceReactions)
                       or any(sum(ch.resonanceReaction==rr.label for ch in g.channels)>1
                              for rr in r.formalism.resonanceReactions if rr.reactionMT==2 for g in r.formalism.spinGroups))
                  and any(rr.kinematics is None for rr in r.formalism.resonanceReactions)]
    if not candidates:return resolved
    result = deepcopy(resolved);entries = _entries(suite)
    neutron = suite.PoPs[suite.projectile]
    if neutron.mass is None:raise ValueError('native RML requires declared neutron mass')
    mn = neutron.mass.convertedTo('amu').value
    for region in result.resolved:
        if region.domainMin not in {c.domainMin for c in candidates}:continue
        f = region.formalism
        def particle(pid):
            local = f.PoPs.particles if f.PoPs is not None else {}
            lp = local.get(pid);gp = suite.PoPs.particles.get(pid)
            def declared(name):
                value = getattr(lp,name,None)
                return value if value is not None else getattr(gp,name,None)
            mass,spin,parity,charge = (declared(name) for name in ('mass','spin','parity','charge'))
            if any(v is None for v in (mass,spin,parity,charge)):
                raise ValueError(f'native RML particle {pid} lacks mass/spin/parity/charge')
            return ChannelParticle(mass.convertedTo('amu').value/mn,charge,spin.convertedTo('hbar').value,parity)
        for rr in f.resonanceReactions:
            if rr.kinematics is not None:continue
            if rr.reactionMT in (18,19,102):
                zero = ChannelParticle(0.,0.,0.,1)
                rr.kinematics = ChannelKinematics(zero,zero,'unity','zero',effective=True)
                continue
            if rr.href is None:raise UnsupportedResonanceError('native RML requires a local reaction link')
            key,_ = _target(rr.href.rstrip('/')+'/crossSection','/reactionSuite/resonances',entries,'eval')
            reaction = entries[key]
            if rr.reactionMT != reaction.ENDF_MT:raise ValueError('native RML reaction MT/link mismatch')
            if rr.reactionMT == 2:
                residual = suite.target
                if rr.Q is None:rr.Q = 0.  # Elastic identity fixes Q, unlike an unknown inelastic Q.
            else:
                ejectile = rr.ejectile or (suite.projectile if 51<=rr.reactionMT<=91 else None)
                if ejectile is None:raise ValueError('native charged pair requires a declared ejectile')
                products = [p.pid for p in reaction.outputChannel.products if p.pid != ejectile]
                if len(products) != 1:raise UnsupportedResonanceError('native RML requires one declared residual product')
                residual = products[0]
                if rr.Q is None:
                    q = reaction.outputChannel.Q
                    if q is None or not q.isKnown:raise ValueError('native RML reaction Q is absent')
                    from kika.nuclear_data.model.quantities import PhysicalQuantity
                    rr.Q = PhysicalQuantity(q.value,q.unit).convertedTo('eV').value
            shift = 'zero' if f.boundaryCondition in (None,'EliminateShiftFunction') else 'calculate'
            ejectile = suite.projectile if rr.reactionMT==2 else ejectile
            rr.kinematics = ChannelKinematics(particle(ejectile),particle(residual),'calculate',shift)
    return result

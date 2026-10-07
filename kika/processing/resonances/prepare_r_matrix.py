"""Immutable normalization of explicitly declared neutral KRM3 channels."""
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
                if len(x) < 2 or np.any(np.diff(x) <= 0) or np.any(~np.isfinite(x+y)) or law not in (1,2,3,4,5):
                    raise ValueError('invalid external table')
                if law in (3,5) and np.any(x <= 0) or law in (4,5) and np.any(y <= 0):
                    raise ValueError('invalid logarithmic external table')
                if out and out[-1].x[-1] != x[0]:
                    raise UnsupportedResonanceError('external table regions must be adjacent')
                out.append(BackgroundCurve(tuple(x), tuple(y), law))
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
    if f.approximation != 'ReichMoore' or f.relativisticKinematics:
        raise UnsupportedResonanceError('R5a certifies nonrelativistic KRM3 only')
    if f.radiusPolicy is not None:
        raise UnsupportedResonanceError('RML uses explicit per-channel APT/APE, not an unnormalized global radius policy')
    if f.boundaryCondition not in (None, 'Given', 'NegativeOrbitalMomentum', 'EliminateShiftFunction'):
        raise UnsupportedResonanceError('Brune/unknown boundary convention requires a separate certification')
    reactions = {r.label:r for r in f.resonanceReactions}
    if len(reactions) != len(f.resonanceReactions):
        raise ValueError('duplicate resonance reaction labels')
    cm = context.atomic_weight_ratio/(1+context.atomic_weight_ratio)
    prepared = []; sectors = set(); coverage = {}
    def allowed(l, s, j):
        return abs(l-s) <= j <= l+s and float(j-abs(l-s)).is_integer()
    for sg in f.spinGroups:
        if sg.spins:
            raise UnsupportedResonanceError('RML requires one J per spin group, not a legacy per-row AJ table')
        if sg.spin is None or not math.isfinite(sg.spin) or sg.spin < 0 or not float(2*sg.spin).is_integer():
            raise ValueError('invalid RML group spin')
        key = (sg.spin, sg.parity)
        if key in sectors:
            raise UnsupportedResonanceError('duplicate coherent J/parity group; merge its levels/channels explicitly')
        sectors.add(key)
        if sg.additionalPhaseShift is not None or sg.phaseShiftMode:
            raise UnsupportedResonanceError('KPS phase functions require a separate R5b certification')
        if sg.atomicWeightRatio not in (None, context.atomic_weight_ratio):
            raise ValueError('RML group AWRI disagrees with incident context')
        channels = []; indexes = []; capture = []; entrances = []; identities = set()
        for index, ch in enumerate(sg.channels):
            if ch.resonanceReaction not in reactions:
                raise ValueError('unknown RML resonance reaction')
            rr = reactions[ch.resonanceReaction]; pair = rr.kinematics
            if rr.reactionMT is None or pair is None:
                raise UnsupportedResonanceError('RML requires normalized reaction MT and physical/effective pair data')
            if not isinstance(rr.reactionMT, int) or rr.reactionMT not in (2,18,19,102) and not 51 <= rr.reactionMT <= 91:
                raise UnsupportedResonanceError('R5a supports elastic, neutron-inelastic, fission and eliminated capture')
            if rr.eliminated:
                if rr.reactionMT != 102 or ch.externalRMatrix is not None or ch.tabulatedBackground is not None:
                    raise UnsupportedResonanceError('KRM3 eliminates capture without an external capture channel')
                if not pair.effective or pair.penetrability not in ('automatic','unity') or pair.shift != 'zero':
                    raise UnsupportedResonanceError('eliminated capture requires effective unity P and zero shift')
                capture.append(index); continue
            if rr.reactionMT == 102:
                raise UnsupportedResonanceError('explicit capture requires KRM4')
            if pair.penetrability not in ('automatic', 'calculate', 'unity') or pair.shift not in ('zero', 'calculate'):
                raise UnsupportedResonanceError('unknown PNT/SHF or Brune shift')
            if ch.radiusUnit not in (None, 'fm') or f.radiusUnit not in (None, 'fm'):
                raise UnsupportedResonanceError('RML radii must be normalized to fm')
            l, s = ch.L, ch.channelSpin
            if not isinstance(l, (int, np.integer)) or not 0 <= l <= 64 or s is None or not math.isfinite(s):
                raise ValueError('invalid RML L/s/J coupling')
            mt = 18 if rr.reactionMT == 19 else rr.reactionMT
            effective = mt == 18
            if effective:
                if not pair.effective or pair.penetrability not in ('automatic','unity') or pair.shift != 'zero':
                    raise UnsupportedResonanceError('effective fission requires unity P and zero shift')
                k2 = 0.; q = 0.; mode = 'unity'; radius = RadiusFunction(constant=1.); phase = RadiusFunction(constant=0.)
            else:
                a, b = pair.particleA, pair.particleB
                if not allowed(l,s,sg.spin):raise ValueError('invalid physical channel L/s/J coupling')
                if pair.effective or a.massRatio != 1. or a.charge != 0 or a.spin != .5 or not math.isfinite(b.massRatio) or b.massRatio <= 0:
                    raise UnsupportedResonanceError('R5a physical channels require a neutral neutron and finite positive target mass')
                if s not in {abs(b.spin-.5), b.spin+.5} or b.spin < 0 or not float(2*b.spin).is_integer():
                    raise ValueError('channel spin disagrees with pair spins')
                if sg.parity is not None and sg.parity != a.parity*b.parity*(-1)**l:
                    raise ValueError('group parity disagrees with physical channel')
                if rr.Q is None or not math.isfinite(rr.Q):
                    raise ValueError('RML physical pair needs a finite declared Q [eV]')
                q = rr.Q
                if mt == 2:
                    if q != 0. or b.massRatio != context.atomic_weight_ratio or b.spin != context.target_spin:
                        raise ValueError('elastic pair disagrees with incident context/Q')
                    entrances.append(len(channels))
                k2 = 2*context.neutron_mass_mev*1e-6/context.hbar_c_mev_fm**2*b.massRatio/(1+b.massRatio)
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
            if f.boundaryCondition == 'EliminateShiftFunction':
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
                                 prepare_external(ch,region.domainMin,region.domainMax))
            channels.append(channel); indexes.append(index)
            if mt == 2:
                previous = coverage.get(l)
                signature = (radius, phase)
                if previous is not None and previous != signature:
                    raise UnsupportedResonanceError('missing-sector completion requires consistent entrance radii per L')
                coverage[l] = signature
        if not entrances or len(capture) != 1:
            raise UnsupportedResonanceError('KRM3 group requires an elastic entrance and one eliminated capture channel')
        if len(sg.energies) != len(sg.widths):raise ValueError('RML table lengths disagree')
        reduced = []; radiation = []; levels = []
        for er, row in zip(sg.energies, sg.widths):
            if len(row) != len(sg.channels) or not all(math.isfinite(v) for v in [er,*row]):
                raise ValueError('nonfinite/malformed RML row')
            gg = 2*row[capture[0]]**2 if f.reducedWidthAmplitudes else abs(row[capture[0]])
            amplitudes = []; physical = []
            for i,ch in zip(indexes,channels):
                width = row[i]
                if f.reducedWidthAmplitudes:
                    amplitude = width
                    pr = ch.functions(np.array([max(abs(er),np.finfo(float).tiny)]))[0][0]
                else:
                    if ch.effective or ch.penetrability == 'unity':pr = 1.
                    else:
                        # ENDF negative dummy levels use the absolute LAB
                        # resonance energy before the channel Q is applied.
                        reference = abs(cm*abs(er)+ch.q)
                        if reference == 0 and width != 0:raise UnsupportedResonanceError('physical width reference lies at channel threshold')
                        if reference == 0:pr = 1.
                        else:
                            from .channel_functions import neutral_channel_functions
                            pr = float(neutral_channel_functions(ch.l,np.sqrt(ch.k2_cm*reference)*ch.radius.evaluate(abs(er)))[0])
                    if pr <= 0 and width != 0:raise UnsupportedResonanceError('RML reference penetrability underflows')
                    amplitude = 0. if width == 0 else math.copysign(math.sqrt(abs(width)/(2*pr)),width)
                amplitudes.append(amplitude); physical.append(2*pr*amplitude**2)
            reduced.append(tuple(amplitudes)); radiation.append(gg)
            levels.append(Level(er,sg.spin,sum(v for v,c in zip(physical,channels) if c.mt==2),gg,
                                sum(v for v,c in zip(physical,channels) if c.mt==18),
                                sum(v for v,c in zip(physical,channels) if c.mt not in (2,18))))
        first = channels[entrances[0]]
        prepared.append(RMLGroup(first.l, first.radius, first.phase_radius, tuple(levels), context=context,
            spin=sg.spin, channels=tuple(channels), reduced=tuple(reduced), radiation=tuple(radiation), entrances=tuple(entrances)))
    if not prepared:raise ValueError('RML requires groups')
    represented = {(g.spin,c.l,c.spin) for g in prepared for c in g.channels if c.mt==2}
    for l,(radius,phase) in coverage.items():
        for s in {abs(context.target_spin-.5),context.target_spin+.5}:
            first = abs(l-s)
            for index in range(round(l+s-first)+1):
                j = first+index
                if (j,l,s) in represented:continue
                k2 = context.k_squared_per_ev/cm
                c = RMLChannel(2,l,s,0.,cm,k2,radius,phase,'calculate','zero',0.)
                prepared.append(RMLGroup(l,radius,phase,(),context=context,spin=j,channels=(c,),entrances=(0,)))
    notes.append('R5a: neutral KRM3, closed-channel real shift retained; no Coulomb/Brune/KPS')
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
                  and (r.formalism.boundaryCondition in ('Given','NegativeOrbitalMomentum','Brune')
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
                products = [p.pid for p in reaction.outputChannel.products if p.pid != suite.projectile]
                if len(products) != 1:raise UnsupportedResonanceError('native RML requires one declared residual product')
                residual = products[0]
                if rr.Q is None:
                    q = reaction.outputChannel.Q
                    if q is None or not q.isKnown:raise ValueError('native RML reaction Q is absent')
                    from kika.nuclear_data.model.quantities import PhysicalQuantity
                    rr.Q = PhysicalQuantity(q.value,q.unit).convertedTo('eV').value
            shift = 'zero' if f.boundaryCondition in (None,'EliminateShiftFunction') else 'calculate'
            rr.kinematics = ChannelKinematics(particle(suite.projectile),particle(residual),'calculate',shift)
    return result

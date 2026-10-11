"""Adaptive linear tables with thermal seeds and separate validation probes.

All finite-probe claims are empirical. No point removal is attempted: thinning
can be added only with a fresh verification. The table engine is independent
of the resonance linearizer's max(actual, chord) error scaling.
"""
from dataclasses import dataclass, asdict
import math
from types import MappingProxyType
import numpy as np

from .core import (PreparedDopplerCurve, DopplerOptions, DopplerInputError,
                   DopplerConvergenceError, evaluate_broadened, _immutable,
                   _quantity, KB, ENGINE_VERSION, _selected_backend)

REFINE = (.25, .5, .75)
CHECK = (.1127016653792583, .276393202250021, .723606797749979, .8872983346207417)
VERIFY = (.0694318442029737, .3300094782075719, .6699905217924281, .9305681557970262)


def _options(options):
    options = DopplerOptions() if options is None else options
    if not isinstance(options, DopplerOptions):
        raise DopplerInputError('invalid-options', 'DopplerOptions required')
    return options


def _domain(curves, domain):
    low = max(float(c.regions[0][0][0]) for c in curves)
    high = min(float(c.regions[-1][0][-1]) for c in curves)
    if domain is None:
        if low == 0:
            raise DopplerInputError('missing-output-domain', 'positive lower output bound required')
        domain = (low, high)
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise DopplerInputError('invalid-domain', 'pair of eV bounds required')
    lo, hi = (_quantity(v, 'output_domain') for v in domain)
    if lo >= hi or lo < low or hi > high:
        raise DopplerInputError('invalid-domain', 'domain must restrict all source tables')
    return lo, hi


def _breaks(curves, per_options, temperature, domain):
    lo, hi = domain
    points = {lo, hi}
    for c, opt in zip(curves, per_options):
        if temperature == c.source_temperature:
            points.update(float(x[0]) for x, _ in c.regions if lo < x[0] < hi)
        if opt.threshold_policy == 'zero_below' and lo < opt.threshold < hi:
            points.add(float(opt.threshold))
    return tuple(sorted(points))


def _seeds(curves, temperature, lo, hi, point_budget, work_bytes):
    seeds = {lo, hi}
    # A coarse lethargy guard plus all source knots inside this output segment.
    # Thermal flanks of extrema/boundaries are included even when feature lies
    # outside output domain. This prevents a midpoint-only missed narrow peak.
    n = max(2, math.ceil(math.log(hi/lo)/math.log(2))+1)
    if n > point_budget:
        raise DopplerConvergenceError('resources', 'lethargy seeds exceed max_points')
    seeds.update(np.geomspace(lo, hi, n).tolist())
    for c in curves:
        h = math.sqrt(KB*(temperature-c.source_temperature)/c.context.atomic_weight_ratio)
        for xs, ys in c.regions:
            first, last = np.searchsorted(xs, [lo, hi])
            if last-first > point_budget:
                raise DopplerConvergenceError('resources', 'source knots exceed seed budget')
            seeds.update(xs[first:last].tolist())
            # Input extrema include peaks thinner than any output probe spacing.
            window_lo=max(0.,math.sqrt(lo)-26*h)**2
            window_hi=(math.sqrt(hi)+26*h)**2
            begin=max(0,int(np.searchsorted(xs,window_lo))-1)
            end=min(len(xs),int(np.searchsorted(xs,window_hi,side='right'))+1)
            local_x,local_y=xs[begin:end],ys[begin:end]
            if len(local_x)*64>work_bytes:
                raise DopplerConvergenceError('resources','feature-seeding scratch exceeds max_work_bytes')
            dy = np.diff(local_y)
            extrema = np.flatnonzero((dy[:-1]*dy[1:] < 0) | ((dy[:-1]==0) != (dy[1:]==0)))+1
            features = np.r_[xs[0], xs[-1], local_x[extrema]]
            relevant = features[(features >= window_lo) & (features <= window_hi)]
            for knot in relevant:
                y = math.sqrt(float(knot))
                for shift in (-8., -4., -2., -1., -.5, -.25, 0., .25, .5, 1., 2., 4., 8.):
                    shifted = y+shift*h
                    if shifted > 0:
                        energy = shifted*shifted
                        if lo < energy < hi: seeds.add(energy)
            if len(seeds) > point_budget:
                raise DopplerConvergenceError('resources', 'thermal seeds exceed max_points')
    return np.array(sorted(seeds))


def _ratio(reference, candidate, eta, options, share):
    available = share*(options.atol + options.rtol*np.maximum(np.abs(reference)-eta, 0.))
    return (np.abs(candidate-reference)+eta)/available


def _adaptive(evaluate, seeds, options, budget, keys):
    if len(seeds)*len(keys) > budget:
        raise DopplerConvergenceError('resources', 'initial table exceeds max_points')
    values, errors = evaluate(seeds)
    # Retain only table nodes; successful off-grid probes are not cached.
    nodes = {float(x): (tuple(values[k][i] for k in keys), tuple(errors[k][i] for k in keys))
             for i, x in enumerate(seeds)}
    batch = max(1, min(options.block_size//7, options.max_work_bytes//(2048*(len(keys)+1))))
    if 2048*(len(keys)+1) > options.max_work_bytes:
        raise DopplerConvergenceError('resources', 'table scratch exceeds max_work_bytes')
    # An accepted interval remains exactly the same function/chord until a
    # node is inserted inside it. Keep only acceptance ratios, never probe
    # arrays. Independent reserved verification still evaluates every interval.
    accepted_intervals = {}
    cache_limit = options.max_work_bytes // 256
    maxrefine, maxcheck, evaluations = 0., 0., len(seeds)
    for iteration in range(options.max_iterations):
        grid = np.array(sorted(nodes))
        data = {k:np.array([nodes[float(x)][0][j] for x in grid]) for j,k in enumerate(keys)}
        additions = {}
        iteration_refine = max((v[0] for v in accepted_intervals.values()), default=0.)
        iteration_check = max((v[1] for v in accepted_intervals.values()), default=0.)
        for start in range(0, len(grid)-1, batch):
            left, right = grid[start:start+batch], grid[start+1:start+batch+1]
            # Equal-sized arrays at final chunk.
            left = left[:len(right)]
            selected = np.array([i for i, (a,b) in enumerate(zip(left,right))
                                 if (float(a),float(b)) not in accepted_intervals], dtype=int)
            if not len(selected):
                continue
            left, right = left[selected], right[selected]
            fractions = np.array(REFINE+CHECK)
            q = left[:,None]+(right-left)[:,None]*fractions
            actual, eta = evaluate(q.ravel()); evaluations += q.size
            ratios = np.zeros(q.shape)
            for k in keys:
                chord = data[k][start+selected,None]+(data[k][start+selected+1]-data[k][start+selected])[:,None]*fractions
                ratios = np.maximum(ratios, _ratio(actual[k].reshape(q.shape), chord,
                                                 eta[k].reshape(q.shape), options, options.table_share))
            iteration_refine = max(iteration_refine, float(np.max(ratios[:,:3])))
            iteration_check = max(iteration_check, float(np.max(ratios[:,3:])))
            for row in np.flatnonzero(np.max(ratios,axis=1)<=1):
                if len(accepted_intervals) >= cache_limit:
                    break
                accepted_intervals[(float(left[row]),float(right[row]))] = (
                    float(np.max(ratios[row,:3])), float(np.max(ratios[row,3:])))
            for row in np.flatnonzero(np.max(ratios,axis=1)>1):
                worst = int(np.argmax(ratios[row]))
                for column in {1, worst}:
                    point = float(q[row,column])
                    if point <= left[row] or point >= right[row]:
                        raise DopplerConvergenceError('unresolvable-grid', 'cannot represent refinement point', energy=point)
                    index = row*7+column
                    additions[point] = (tuple(actual[k][index] for k in keys), tuple(eta[k][index] for k in keys))
            if (len(nodes)+len(additions))*len(keys) > budget:
                raise DopplerConvergenceError('resources', 'refinement exceeds max_points')
        maxrefine, maxcheck = iteration_refine, iteration_check
        if not additions:
            reserved_maximum=0.
            # A separate probe family must pass before accepting. Failed
            # verification refines the mesh; it never widens the tolerance.
            for start in range(0,len(grid)-1,batch):
                left,right=grid[start:start+batch],grid[start+1:start+batch+1]
                left=left[:len(right)]
                fractions=np.array(VERIFY)
                q=left[:,None]+(right-left)[:,None]*fractions
                actual,eta=evaluate(q.ravel());evaluations+=q.size
                ratios=np.zeros(q.shape)
                for k in keys:
                    chord=data[k][start:start+len(left),None]+(data[k][start+1:start+len(left)+1]-data[k][start:start+len(left)])[:,None]*fractions
                    ratios=np.maximum(ratios,_ratio(actual[k].reshape(q.shape),chord,eta[k].reshape(q.shape),options,options.table_share))
                reserved_maximum=max(reserved_maximum,float(np.max(ratios)))
                for row in np.flatnonzero(np.max(ratios,axis=1)>1):
                    accepted_intervals.pop((float(left[row]),float(right[row])), None)
                    column=int(np.argmax(ratios[row]));point=float(q[row,column]);index=row*len(VERIFY)+column
                    if not left[row]<point<right[row]:
                        raise DopplerConvergenceError('unresolvable-grid','verification cannot refine')
                    additions[point]=(tuple(actual[k][index] for k in keys),tuple(eta[k][index] for k in keys))
                if (len(nodes)+len(additions))*len(keys)>budget:
                    raise DopplerConvergenceError('resources','verification refinement exceeds max_points')
            if not additions:
                return grid,data,dict(iterations=iteration+1,evaluations=evaluations,
                                      refinement_ratio=maxrefine,check_ratio=maxcheck,
                                      reserved_verification_ratio=reserved_maximum)
        nodes.update(additions)
    raise DopplerConvergenceError('tabulation-budget-exhausted', 'max_iterations reached')


def _forms(tables, keys, label):
    from kika.nuclear_data.model import XYs1d, Regions1d, crossSectionAxes
    result = {}
    for key in keys:
        regions = [XYs1d(x.copy(), y[key].copy(), axes=crossSectionAxes(), label=label) for x,y in tables]
        result[key] = regions[0] if len(regions)==1 else Regions1d(regions, axes=crossSectionAxes(), label=label)
    return result


def _evaluate_curve(curve, q, temperature, options, left_boundary=None):
    q=np.asarray(q,dtype=float)
    actual_q=q.copy()
    if left_boundary is not None and temperature != curve.source_temperature:
        actual_q[actual_q==left_boundary]=np.nextafter(left_boundary,0.)
    result=evaluate_broadened(curve,actual_q,temperature=temperature,options=options)
    values=result.values.copy()
    if left_boundary is not None and temperature == curve.source_temperature:
        indices=np.flatnonzero(q==left_boundary)
        for x,y in curve.regions:
            if x[0]<left_boundary<=x[-1]:
                values[indices]=np.interp(left_boundary,x,y)
                break
        if options.threshold_policy=='zero_below' and left_boundary<=options.threshold:
            values[indices]=0.
    return values,result.kernel_error_estimates+result.tail_bounds


def _tabulate(curves, per_options, temperature, domain, options, evaluate, keys, label, progress=None):
    temperature = _quantity(temperature, 'temperature', True)
    if any(temperature < c.source_temperature for c in curves):
        raise DopplerInputError('cooling', 'T1<T0')
    bounds = _breaks(curves, per_options, temperature, domain)
    tables, diagnostics = [], []
    used = 0
    for i, (low,high) in enumerate(zip(bounds[:-1], bounds[1:])):
        # A semantic boundary belongs to the right. The preceding region stores
        # the left limit independently, including zero_below at the threshold.
        left_high = i < len(bounds)-2
        def segment(q):
            return evaluate(q,left_boundary=high if left_high else None)
        # No heating/linearization when deltaT=0: canonical source knots suffice.
        identity = all(temperature==c.source_temperature for c in curves)
        if identity:
            seeds = np.unique(np.r_[low,high,*[x[(x>low)&(x<high)] for c in curves for x,_ in c.regions]])
            y, _ = segment(seeds)
            check = dict(iterations=0, evaluations=len(seeds), refinement_ratio=0., check_ratio=0., identity=True)
            if (used+len(seeds)*len(keys))>options.max_points:
                raise DopplerConvergenceError('resources', 'identity exceeds max_points')
        else:
            seeds = _seeds(curves, temperature, low, high, options.max_points//len(keys),options.max_work_bytes)
            seeds, y, check = _adaptive(segment, seeds, options, options.max_points-used, keys)
        tables.append((seeds, y)); diagnostics.append(check)
        used += len(seeds)*len(keys)
        if progress is not None: progress((i+1)/(len(bounds)-1))
    forms = _forms(tables, keys, label)
    frozen = tuple((_immutable(x), MappingProxyType({k:_immutable(y[k]) for k in keys})) for x,y in tables)
    identity=all(temperature==c.source_temperature for c in curves)
    conventions=[(domain[0],min(domain[1],o.threshold)) for o in per_options
                 if o.threshold_policy=='zero_below' and o.threshold>domain[0]]
    return forms, frozen, dict(points=used, regions=diagnostics, empirical_verification=False,
                              global_error_bound=False, table_share=options.table_share,
                              serialization_share=options.serialization_share,
                              region_domains=[(a,b) for a,b in zip(bounds[:-1],bounds[1:])],
                              heated_domains=[] if all(temperature==c.source_temperature for c in curves) else [domain],
                              conserved_domains=[domain] if identity and not conventions else [],
                              convention_modified_domains=conventions,
                              unsupported_domains=[], thinning=False)


def _verify(forms, tables, evaluate, options, *, share, label=None):
    from .core import prepare_curve, DopplerContext, SupportPolicy
    from kika.nuclear_data.model import XYs1d, Regions1d
    if set(forms) != set(tables[0][1]):
        raise DopplerInputError('coverage-changed', 'result reactions differ')
    maximum = {k:0. for k in forms}
    for index,(grid, _) in enumerate(tables):
        lo, hi = float(grid[0]), float(grid[-1])
        readers = {}
        knots = {float(v) for v in grid}
        for key,form in forms.items():
            if label is not None and form.label != label:
                raise DopplerInputError('label-conflict', 'output form label changed')
            children = form.function1ds if isinstance(form, Regions1d) else [form]
            if len(children)!=len(tables): raise DopplerInputError('region-structure-changed', 'regions differ')
            child = children[index]
            if not isinstance(child, XYs1d) or child.domainMin != lo or child.domainMax != hi:
                raise DopplerInputError('domain-changed', 'boundary moved')
            p = prepare_curve(child, context=DopplerContext(1.), source_temperature=0.,
                              support=SupportPolicy('zero','zero'), negative_policy='preserve_signed')
            if child.domainUnit != 'eV' or child.rangeUnit != 'b':
                raise DopplerInputError('invalid-units', 'canonical eV/b output required')
            readers[key] = p.regions[0]
            knots.update(p.regions[0][0].tolist())
        mesh = np.array(sorted(knots))
        batch = max(1, min(options.block_size//5, options.max_work_bytes//(2048*(len(forms)+1))))
        for start in range(0,len(mesh)-1,batch):
            left,right=mesh[start:start+batch],mesh[start+1:start+batch+1]
            left=left[:len(right)]
            q=np.unique(np.r_[left,right,(left[:,None]+(right-left)[:,None]*np.array(VERIFY)).ravel()])
            actual,eta=evaluate(q,left_boundary=hi if index<len(tables)-1 else None)
            for key,(x,y) in readers.items():
                ratio=_ratio(actual[key],np.interp(q,x,y),eta[key],options,share)
                maximum[key]=max(maximum[key],float(np.max(ratio)))
    if any(v>1 for v in maximum.values()):
        raise DopplerConvergenceError('table-verification-failed', f'table exceeds budget: {maximum}')
    return maximum


def _read_form(form,energies):
    from kika.nuclear_data.model import Regions1d
    energies=np.asarray(energies,dtype=float)
    children=form.function1ds if isinstance(form,Regions1d) else [form]
    values=np.empty_like(energies); owner=np.full(len(energies),-1)
    for i,c in enumerate(children):owner[(energies>=c.domainMin)&(energies<=c.domainMax)]=i
    if np.any(owner<0):raise DopplerInputError('invalid-domain','reference probes outside table')
    for i,c in enumerate(children):
        selected=owner==i
        values[selected]=np.interp(energies[selected],c.xs,c.ys)
    return values


def _external_check(forms,energies,reference,uncertainty,options):
    energies=np.asarray(energies,dtype=float)
    if energies.ndim!=1 or not len(energies) or np.any(~np.isfinite(energies)) or np.any(energies<=0):
        raise DopplerInputError('invalid-energy','positive reference probes')
    if set(reference)!=set(forms):raise DopplerInputError('coverage-changed','reference reaction set differs')
    maxima={}
    for key,form in forms.items():
        value=np.asarray(reference[key],dtype=float)
        eta=np.asarray(uncertainty[key],dtype=float)
        if value.shape!=energies.shape or eta.shape!=energies.shape or np.any(~np.isfinite(value)) or np.any(~np.isfinite(eta)) or np.any(eta<0):
            raise DopplerInputError('invalid-reference','finite value/uncertainty arrays required')
        maxima[key]=float(np.max(_ratio(value,_read_form(form,energies),eta,options,1.)))
    if any(v>1 for v in maxima.values()):
        raise DopplerConvergenceError('reference-verification-failed',f'external reference exceeds budget: {maxima}')
    return dict(maximum_ratios=maxima,probes=len(energies),method='external-reference',global_error_bound=False)


@dataclass(frozen=True)
class DopplerResult:
    form: object
    report: object
    prepared: PreparedDopplerCurve
    temperature: float
    options: DopplerOptions
    _tables: tuple

    def evaluate(self, energies):
        return evaluate_broadened(self.prepared, energies, temperature=self.temperature, options=self.options)

    def verify_table(self, form=None, *, total_budget=False):
        def evaluate(q,left_boundary=None):
            v,e=_evaluate_curve(self.prepared,q,self.temperature,self.options,left_boundary)
            return {0:v},{0:e}
        share=1. if total_budget else self.options.table_share
        return _verify({0:self.form if form is None else form},self._tables,evaluate,self.options,share=share)[0]

    def verify_reference(self,energies,values,*,uncertainty=None):
        uncertainty=np.zeros(len(energies)) if uncertainty is None else uncertainty
        return _external_check({0:self.form},energies,{0:values},{0:uncertainty},self.options)


def broaden_curve(prepared, *, temperature, options=None, output_domain=None, label='heated', progress=None):
    if not isinstance(prepared, PreparedDopplerCurve):
        raise DopplerInputError('invalid-prepared', 'prepare_curve required')
    options=_options(options)
    temperature=_quantity(temperature,'temperature',True)
    if not isinstance(label,str) or not label: raise DopplerInputError('invalid-label','label required')
    domain=_domain((prepared,),output_domain)
    def evaluate(q,left_boundary=None):
        v,e=_evaluate_curve(prepared,q,temperature,options,left_boundary)
        return {0:v},{0:e}
    forms,tables,report=_tabulate((prepared,),(options,),temperature,domain,options,evaluate,(0,),label,progress)
    report.update(engine='doppler-table-1',core_engine=ENGINE_VERSION,backend=_selected_backend(options),source_hash=prepared.source_hash,temperature_source=prepared.source_temperature,
                  temperature_final=temperature,options=asdict(options),accuracy_scope='input_table',
                  delta_temperature=temperature-prepared.source_temperature,mass=asdict(prepared.context),
                  source_temperature_origin=prepared.source_temperature_origin,kB_eV_per_K=KB,
                  units=dict(energy='eV',cross_section='b',temperature='K'),support=asdict(prepared.support),
                  negative_policy=prepared.negative_policy,threshold_policy=options.threshold_policy,
                  threshold=options.threshold,retained_table_numeric_bytes=sum(x.nbytes+sum(v.nbytes for v in y.values()) for x,y in tables),
                  source_minimum=min(float(np.min(y)) for _,y in prepared.regions),
                  result_minimum=min(float(np.min(y[0])) for _,y in tables),
                  distribution_temperature_treatment='not-processed',covariance_temperature_treatment='not-processed')
    result=DopplerResult(forms[0],MappingProxyType(report),prepared,temperature,options,tables)
    ratio=result.verify_table()
    report['verification_ratio']=ratio
    report['table_verification']='separate-fractions-against-production'
    return result

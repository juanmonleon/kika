"""Scalar free-gas suites: explicit sources, exclusive sums and atomic attachment.

Local graph/link utilities are shared with reconstruction. Nuclear kernels,
source preparation and numerical tolerances remain Doppler-specific.
"""
from copy import deepcopy, copy
from collections.abc import Mapping
from dataclasses import dataclass, asdict
import json
import math
from types import MappingProxyType
import numpy as np

from .core import (DopplerContext, DopplerInputError, UnsupportedDopplerError, KB,
                   DopplerConvergenceError, prepare_curve, evaluate_broadened,
                   _quantity, DopplerOptions)
from .table import _domain, _options, _tabulate, _verify, _evaluate_curve, _external_check

# Reuse model identity/link grammar, never the resonance physics evaluator.
from ..resonances.suite import ReactionKey, _entries, _source_forms, _graph, _fingerprint
from ..resonances.prepare import UnsupportedResonanceError


@dataclass(frozen=True)
class DopplerInputAccuracy:
    """Structured assertions, not a numerical certificate of source physics."""
    accuracy_scope: str = 'input_table'
    source_temperature: float | None = None
    infinite_dilution: bool | None = None
    temperature_model: str = 'convolution'

    def __post_init__(self):
        if self.accuracy_scope != 'input_table':
            raise UnsupportedDopplerError('source-accuracy-not-supported','source envelope required for physical_input')
        if self.source_temperature is not None:
            object.__setattr__(self,'source_temperature',_quantity(self.source_temperature,'source_temperature',True))
        if self.infinite_dilution is not None and not isinstance(self.infinite_dilution,bool):
            raise DopplerInputError('invalid-dilution-declaration','boolean declaration required')
        if self.temperature_model not in ('convolution','free-gas-plus-threshold-convention'):
            raise DopplerInputError('invalid-temperature-model','temperature_model')


def _model_graph(suite, source_style):
    try:
        entries=_entries(suite)
        forms=_source_forms(entries,source_style)
        graph,order=_graph(entries,source_style)
    except UnsupportedResonanceError as exc:
        raise UnsupportedDopplerError(getattr(exc,'category','unsupported-graph'),str(exc)) from exc
    except (ValueError,KeyError,TypeError) as exc:
        raise DopplerInputError('invalid-graph',str(exc)) from exc
    if not entries: raise DopplerInputError('empty-suite','no cross sections')
    return entries,forms,graph,order


def _temperature(suite, style, assertion, declaration):
    from kika.nuclear_data.model import Heated, Evaluated, CrossSectionReconstructed, PhysicalQuantity
    try: chain=suite.styles.chain(style)
    except (ValueError,KeyError) as exc: raise DopplerInputError('invalid-style',str(exc)) from exc
    if not isinstance(chain[-1],Evaluated):
        raise UnsupportedDopplerError('unsupported-style','source must derive from evaluated')
    values=[]
    if assertion is not None: values.append(('explicit',_quantity(assertion,'source_temperature',True)))
    if declaration.source_temperature is not None:
        values.append(('declaration',declaration.source_temperature))
    nearest=next((s for s in chain if isinstance(s,(Heated,CrossSectionReconstructed,Evaluated))),None)
    if isinstance(nearest,(Heated,Evaluated)):
        quantity=nearest.temperature
        if not isinstance(quantity,PhysicalQuantity):
            raise DopplerInputError('missing-temperature','PhysicalQuantity source temperature required')
        try:
            _quantity(quantity.value,'style_temperature',True)
            value=_quantity(quantity.convertedTo('K').value,'style_temperature',True)
        except ValueError as exc: raise DopplerInputError('invalid-temperature',str(exc)) from exc
        values.insert(0,('style',value))
    else:
        # A reconstruction label does not establish its temperature. A heated
        # evaluated background combined with 0 K resonance parameters is mixed.
        root=chain[-1]
        if root.temperature is not None and root.temperature.convertedTo('K').value != 0:
            raise UnsupportedDopplerError('mixed-source-temperature','reconstructed source has nonzero evaluated background temperature')
    if not values: raise DopplerInputError('missing-temperature','explicit reconstructed source temperature required')
    temperature=values[0][1]
    if any(v!=temperature for _,v in values[1:]):
        raise DopplerInputError('temperature-conflict','style and assertions disagree')
    source_model=declaration.temperature_model
    known_model=not isinstance(nearest,Heated)
    if isinstance(nearest,Heated) and nearest.documentation and nearest.documentation.body:
        try:
            doc=json.loads(nearest.documentation.body.text)
            if doc.get('schema')=='kika-doppler-suite-1':
                if doc['temperature_final'] != temperature:
                    raise DopplerInputError('temperature-conflict','processing report and style disagree')
                source_model=doc['temperature_model']
                known_model=True
        except KeyError as exc:raise DopplerInputError('provenance-conflict','incomplete Doppler source report') from exc
        except (json.JSONDecodeError,TypeError): pass
    return temperature,values[0][0],source_model,known_model


def _mass(suite, context):
    from kika.nuclear_data.model.pops import zaFromPid, Unorthodox
    try:
        za=zaFromPid(suite.target)
        if za<1000 or za%1000==0: raise ValueError('not a single isotope')
    except ValueError as exc:
        raise UnsupportedDopplerError('unsupported-target','single identified isotope required') from exc
    target=suite.PoPs.particles.get(suite.target)
    if isinstance(target,Unorthodox):
        raise UnsupportedDopplerError('bound-target','thermal scattering target')
    neutron=suite.PoPs.particles.get('n')
    ratio=None
    if target is not None and neutron is not None and target.mass is not None and neutron.mass is not None:
        try:
            _quantity(target.mass.value,'target_mass');_quantity(neutron.mass.value,'neutron_mass')
            ratio=_quantity(target.mass.convertedTo('amu').value,'target_mass')/_quantity(neutron.mass.convertedTo('amu').value,'neutron_mass')
        except ValueError as exc:raise DopplerInputError('invalid-mass',str(exc)) from exc
    if context is not None and not isinstance(context,DopplerContext):
        raise DopplerInputError('invalid-context','DopplerContext required')
    if context is None:
        if ratio is None: raise DopplerInputError('missing-mass','explicit DopplerContext required')
        return DopplerContext(ratio,mass_origin='PoPs')
    if ratio is not None and not math.isclose(ratio,context.atomic_weight_ratio,rel_tol=1e-10,abs_tol=0):
        raise DopplerInputError('mass-conflict','explicit context disagrees with PoPs')
    return context


def _sum(values, errors, graph, order):
    for key in order:
        values[key]=np.array([math.fsum(row) for row in zip(*(values[c] for c in graph[key]))])
        errors[key]=np.array([math.fsum(row) for row in zip(*(errors[c] for c in graph[key]))])
    return values,errors


@dataclass(frozen=True)
class SuiteDopplerResult:
    forms: object
    report: object
    source_style: str
    label: str
    temperature: float
    options: DopplerOptions
    _prepared: object
    _per_options: object
    _graph: object
    _order: tuple
    _tables: tuple
    _source_hash: str
    _mt_keys: object
    _declaration: DopplerInputAccuracy
    _provenance_json: str = ''

    @property
    def forms_by_mt(self):
        return MappingProxyType({mt:self.forms[k] for mt,k in self._mt_keys.items()})

    def evaluate(self,energies):
        values,errors={},{}
        for key,p in self._prepared.items():
            r=evaluate_broadened(p,energies,temperature=self.temperature,options=self._per_options[key])
            values[key]=r.values; errors[key]=r.kernel_error_estimates+r.tail_bounds
        return _sum(values,errors,self._graph,self._order)[0]

    def _evaluate(self,energies,left_boundary=None):
        values,errors={},{}
        for key,p in self._prepared.items():
            values[key],errors[key]=_evaluate_curve(p,energies,self.temperature,self._per_options[key],left_boundary)
        return _sum(values,errors,self._graph,self._order)

    def verify_table(self,forms=None,*,total_budget=False):
        return self._verify_table(forms,total_budget=total_budget)

    def _verify_table(self,forms=None,*,total_budget=False,evaluator=None):
        forms=self.forms if forms is None else forms
        result=_verify(forms,self._tables,self._evaluate if evaluator is None else evaluator,self.options,
                       share=1. if total_budget else self.options.table_share,label=self.label)
        # Shared output grid gives exact linear closure apart from roundoff.
        # For caller/reloaded grids also check all union knots; piecewise linear
        # difference achieves its extrema at those knots.
        from kika.nuclear_data.model import Regions1d
        for index in range(len(self._tables)):
            curves={k:(f.function1ds[index] if isinstance(f,Regions1d) else f) for k,f in forms.items()}
            grid=np.unique(np.concatenate([c.xs for c in curves.values()]))
            v={k:np.interp(grid,c.xs,c.ys) for k,c in curves.items()}
            for key in self._order:
                summed=np.array([math.fsum(row) for row in zip(*(v[c] for c in self._graph[key]))])
                ratio=np.max(np.abs(v[key]-summed)/(self.options.atol+self.options.rtol*np.abs(summed)))
                if ratio>self.options.table_share:
                    raise DopplerConvergenceError('sum-not-closed','output sums changed')
        return result

    def verify_reference(self,energies,values,*,uncertainty=None):
        if uncertainty is None:uncertainty={k:np.zeros(len(energies)) for k in values}
        return _external_check(self.forms,energies,values,uncertainty,self.options)

    def verify_suite(self,suite,*,total_budget=False):
        from kika.nuclear_data.model import Heated,PhysicalQuantity,Styles
        shadow=copy(suite)
        shadow.styles=Styles([s for s in suite.styles if s.label!=self.label])
        if _fingerprint(shadow,self.source_style)!=self._source_hash:
            raise DopplerInputError('stale-source','source/graph changed after processing')
        style=suite.styles[self.label]
        if not isinstance(style,Heated) or style.derivedFrom!=self.source_style or not isinstance(style.temperature,PhysicalQuantity) or style.temperature.unit!='K' or style.temperature.value!=self.temperature:
            raise DopplerInputError('style-conflict','Heated temperature/derivation changed')
        try:
            metadata=json.loads(style.documentation.body.text)
            if metadata!=json.loads(self._provenance_json):raise ValueError('report changed')
        except (AttributeError,ValueError,TypeError) as exc:
            raise DopplerInputError('provenance-conflict','Doppler declaration changed') from exc
        entries=_entries(suite)
        if set(entries)!=set(self.forms): raise DopplerInputError('coverage-changed','reaction coverage changed')
        return self.verify_table({k:r.crossSection[self.label] for k,r in entries.items()},total_budget=total_budget)


def broaden_suite(suite, context=None, *, source_style='recon', label='heated', temperature,
                  support, source_temperature=None, input_accuracy=None, urr_policy=None,
                  options=None, output_domain=None, negative_policy='reject',
                  thresholds_by_reaction=None, progress=None):
    """Prepare complete scalar neutron/lab source, tabulate and derive all sums.

    No mutation. URR requires explicit dilute-mean semantics; distributions and
    covariances keep their original styles and are not processed.
    """
    from dataclasses import replace
    from kika.nuclear_data.model import ReactionSuite, Evaluated, CrossSectionReconstructed
    if not isinstance(suite,ReactionSuite):raise DopplerInputError('invalid-suite','ReactionSuite required')
    options=_options(options)
    declaration=DopplerInputAccuracy() if input_accuracy is None else input_accuracy
    if not isinstance(declaration,DopplerInputAccuracy):raise DopplerInputError('invalid-input-accuracy','DopplerInputAccuracy required')
    if not isinstance(label,str) or not label or label==source_style:
        raise DopplerInputError('invalid-label','distinct output style required')
    if label in suite.styles.labels:raise DopplerInputError('label-conflict','style already exists')
    if len(suite.styles.labels)!=len(set(suite.styles.labels)):
        raise DopplerInputError('invalid-style','duplicate source style labels')
    if suite.projectile!='n' or str(suite.projectileFrame)!='lab' or suite.interaction!='nuclear':
        raise UnsupportedDopplerError('unsupported-projectile-frame','only neutron/lab nuclear suites')
    for container in ('orphanProducts','productions','incompleteReactions'):
        if len(getattr(suite,container)):
            raise UnsupportedDopplerError('incomplete-coverage',f'{container} cannot be certified as exclusive scalar leaves')
    if suite.report is not None and not suite.report.isCleanFor('cross-sections'):
        raise UnsupportedDopplerError('conversion-not-clean','cross-section conversion losses')
    t0,origin,source_model,known_model=_temperature(suite,source_style,source_temperature,declaration)
    evaluated=suite.styles.evaluatedFor(source_style)
    if evaluated.targetInfo is not None:
        raise UnsupportedDopplerError('bound-or-mixture-target','targetInfo requires separate isotope/thermal treatment')
    temperature=_quantity(temperature,'temperature',True)
    if temperature<t0:raise DopplerInputError('cooling','T1<T0')
    if not known_model and input_accuracy is None and temperature>t0:
        raise UnsupportedDopplerError('unknown-source-temperature-model','declare convolution semantics for external Heated source')
    if source_model!='convolution' and temperature>t0:
        raise UnsupportedDopplerError('nonconvolution-source','threshold-truncated heated source cannot claim free-gas semigroup')
    context=_mass(suite,context)
    if declaration.infinite_dilution is False:
        raise UnsupportedDopplerError('finite-dilution-source','finite-dilution sources are unsupported')
    urr=suite.resonances is not None and suite.resonances.unresolved is not None
    if urr and (urr_policy!='dilute_mean' or declaration.infinite_dilution is not True):
        raise UnsupportedDopplerError('unknown-urr-semantics','declare complete infinite-dilution mean source and dilute_mean')
    if urr_policy not in (None,'dilute_mean'):
        raise UnsupportedDopplerError('unsupported-urr-policy','only dilute_mean supported')
    entries,forms,graph,order=_model_graph(suite,source_style)
    if any(label in r.crossSection for r in entries.values()):raise DopplerInputError('label-conflict','form already exists')
    mt_keys={}
    for key,r in entries.items():
        if r.ENDF_MT is not None:
            if r.ENDF_MT in mt_keys:raise UnsupportedDopplerError('ambiguous-reaction','duplicate MT projection')
            mt_keys[r.ENDF_MT]=key
    from kika._constants import MF3_SUM_RULES
    def present_components(mt):
        out=set()
        for item in MF3_SUM_RULES.get(mt,((),''))[0]:
            if isinstance(item,str):
                target=int(item[1:])
                out.update({target} if target in mt_keys else present_components(target))
            elif item in mt_keys:out.add(item)
        return out
    for mt in mt_keys:
        if (mt==1 or present_components(mt)) and mt_keys[mt] not in graph:
            raise UnsupportedDopplerError('aggregate-without-graph','overlapping aggregate needs exclusive summands')
    leaves=tuple(k for k in entries if k not in graph)
    def descendants(k):return [k] if k not in graph else sum((descendants(c) for c in graph[k]),[])
    if 1 in mt_keys and set(descendants(mt_keys[1]))!=set(leaves):
        raise UnsupportedDopplerError('incomplete-total-graph','total does not cover all exclusive leaves')
    if not isinstance(support,Mapping) or set(support)!=set(leaves):raise DopplerInputError('support-coverage','support keyed by every leaf ReactionKey')
    thresholds={} if thresholds_by_reaction is None else dict(thresholds_by_reaction)
    if not set(thresholds)<=set(leaves):raise DopplerInputError('threshold-coverage','thresholds indexed by leaves')
    prepared,per_options={},{}
    for key in leaves:
        prepared[key]=prepare_curve(forms[key],context=context,source_temperature=t0,support=support[key],
                                    negative_policy=negative_policy,source_temperature_origin=origin)
        per_options[key]=replace(options,threshold_policy='zero_below',threshold=thresholds[key]) if key in thresholds else options
    # Source References may share bytes, but aliases as exclusive graph leaves
    # would count the same physical source twice. Reject that ambiguity.
    from kika.nuclear_data.model import Reference
    for key in order:
        descendants_keys=descendants(key)
        owners=[id(forms[k]) for k in descendants_keys]
        if len(owners)!=len(set(owners)) and any(isinstance(entries[k].crossSection[source_style],Reference) for k in descendants_keys):
            raise UnsupportedDopplerError('duplicate-source-leaf','sum contains aliased source twice')
    endpoints={(float(p.regions[0][0][0]),float(p.regions[-1][0][-1])) for p in prepared.values()}
    if len(endpoints)!=1:raise UnsupportedDopplerError('domain-not-covered','suite leaves require a complete common source domain')
    domain=_domain(tuple(prepared.values()),output_domain)
    declared=evaluated.projectileEnergyDomain
    if declared is not None:
        from kika.nuclear_data.model.units import conversion_factor
        factor=conversion_factor(declared.unit,'eV')
        bounds=(declared.min*factor,declared.max*factor)
        if next(iter(endpoints))!=bounds:
            raise UnsupportedDopplerError('domain-not-covered','source leaves do not cover evaluated domain')
    source_hash=_fingerprint(suite,source_style)
    source_balance={}
    if graph:
        # Validate the redundant source aggregates before replacing them by sums.
        source_curves={k:prepare_curve(forms[k],context=context,source_temperature=t0,
                       support=next(iter(support.values())),negative_policy=negative_policy) for k in graph}
        if any((float(p.regions[0][0][0]),float(p.regions[-1][0][-1]))!=next(iter(endpoints)) for p in source_curves.values()):
            raise UnsupportedDopplerError('domain-not-covered','source sum domain differs from leaves')
        source_grid=np.unique(np.concatenate([x for p in (*prepared.values(),*source_curves.values()) for x,_ in p.regions]))
        for start in range(0,len(source_grid),options.block_size):
            base=source_grid[start:start+options.block_size]
            right=source_grid[start+1:start+len(base)+1]
            mid=(base[:len(right)]+right)*.5
            left=np.nextafter(base[base>source_grid[0]],source_grid[0])
            q=np.unique(np.r_[base,mid,left])
            v={k:np.array([p.evaluate_source(float(e)) for e in q]) for k,p in prepared.items()}
            for key in order:
                summed=np.array([math.fsum(row) for row in zip(*(v[c] for c in graph[key]))])
                original=np.array([source_curves[key].evaluate_source(float(e)) for e in q])
                ratio=float(np.max(np.abs(original-summed)/(options.atol+options.rtol*np.abs(summed))))
                source_balance[key]=max(source_balance.get(key,0.),ratio)
                if ratio>1:raise DopplerInputError('source-sum-not-closed','source aggregate and leaves disagree')
                v[key]=summed
    # Result with frozen source mappings is the single evaluator for every table.
    result=SuiteDopplerResult({}, {},source_style,label,temperature,options,MappingProxyType(prepared),
                             MappingProxyType(per_options),MappingProxyType(graph),order,(),source_hash,
                             MappingProxyType(mt_keys),declaration)
    keys=tuple(entries)
    from .table import _ConstructionSamples
    construction=_ConstructionSamples(result._evaluate,keys,options)
    tables_forms,tables,tab_report=_tabulate(tuple(prepared.values()),tuple(per_options.values()),temperature,
                                             domain,options,construction,keys,label,progress)
    if any(o.threshold_policy=='zero_below' for o in per_options.values()):
        model='free-gas-plus-threshold-convention'
    elif source_model!='convolution':model=source_model
    elif not known_model and input_accuracy is None:model='unknown'
    else:model='convolution'
    from .core import ENGINE_VERSION, _selected_backend
    report=dict(core_engine=ENGINE_VERSION,backend=_selected_backend(options),schema='kika-doppler-suite-1',engine='doppler-suite-1',source_style=source_style,label=label,
                source_hash=source_hash,temperature_source=t0,temperature_final=temperature,temperature_model=model,
                delta_temperature=temperature-t0,source_temperature_origin=origin,kB_eV_per_K=KB,
                units=dict(energy='eV',cross_section='b',temperature='K'),output_domain=domain,
                inherited_source_error='excluded-by-input_table-scope',
                negative_policy=negative_policy,
                mass=asdict(context),input_accuracy=asdict(declaration),accuracy_scope='input_table',options=asdict(options),
                source_balance={f'{k.container}/{k.label}':v for k,v in source_balance.items()},
                reactions={f'{k.container}/{k.label}':dict(source_hash=p.source_hash,support=asdict(p.support),
                   threshold_policy=per_options[k].threshold_policy,threshold=per_options[k].threshold,
                   source_minimum=min(float(np.min(y)) for _,y in p.regions),
                   result_minimum=min(float(np.min(y[k])) for _,y in tables)) for k,p in prepared.items()},
                graph={f'{k.container}/{k.label}':[f'{c.container}/{c.label}' for c in graph[k]] for k in graph},
                urr_treatment='broadened-dilute-mean' if urr else 'absent',finite_dilution_supported=False,
                distribution_temperature_treatment='not-processed',covariance_temperature_treatment='not-processed',
                tabulation=tab_report,empirical_verification=False,global_error_bound=False)
    result=SuiteDopplerResult(MappingProxyType(tables_forms),MappingProxyType(report),source_style,label,temperature,
                options,result._prepared,result._per_options,result._graph,order,tables,source_hash,result._mt_keys,declaration)
    checks=result._verify_table(evaluator=construction)
    report['tabulation']['construction_samples']=construction.finish()
    report['verification_ratios']={f'{k.container}/{k.label}':v for k,v in checks.items()}
    if _fingerprint(suite,source_style)!=source_hash:raise DopplerInputError('stale-source','suite changed during processing')
    # JSON is a canonical, independent provenance snapshot used during attachment.
    report=json.loads(json.dumps(report))
    return SuiteDopplerResult(result.forms,MappingProxyType(report),source_style,label,temperature,options,
                             result._prepared,result._per_options,result._graph,order,tables,source_hash,result._mt_keys,declaration,
                             json.dumps(report,sort_keys=True))


def attach_broadening(suite,result,*,date=None):
    """Recheck mutable output, then publish prebuilt cross-section/style containers."""
    from kika.nuclear_data.model import Heated,PhysicalQuantity,Styles
    from kika.nuclear_data.model.documentation import Documentation,DocumentationText
    from datetime import date as calendar_date
    if date is None:
        date=calendar_date.today().isoformat()
    try:
        if not isinstance(date,str):raise ValueError('string required')
        date=calendar_date.fromisoformat(date).isoformat()
    except ValueError as exc:
        raise DopplerInputError('invalid-date','ISO calendar date required') from exc
    if not isinstance(result,SuiteDopplerResult):raise DopplerInputError('invalid-result','SuiteDopplerResult required')
    if json.dumps(dict(result.report),sort_keys=True)!=result._provenance_json:
        raise DopplerInputError('provenance-conflict','result report was edited')
    if _fingerprint(suite,result.source_style)!=result._source_hash:
        raise DopplerInputError('stale-source','suite changed; recompute')
    entries=_entries(suite)
    if set(entries)!=set(result.forms):raise DopplerInputError('coverage-changed','reaction set changed')
    if result.label in suite.styles.labels or any(result.label in r.crossSection for r in entries.values()):
        raise DopplerInputError('label-conflict','style/form label exists')
    candidate=deepcopy(suite)
    candidates=_entries(candidate)
    for key,r in candidates.items():r.crossSection[result.label]=deepcopy(result.forms[key])
    metadata=result._provenance_json
    heated=Heated(result.label,derivedFrom=result.source_style,date=date,temperature=PhysicalQuantity(result.temperature,'K'),
                  documentation=Documentation(body=DocumentationText(metadata)))
    candidate.styles.add(heated)
    result.verify_suite(candidate)
    containers={}
    for key,r in entries.items():
        container=deepcopy(r.crossSection)
        for label,form in r.crossSection.items():container[label]=form
        container[result.label]=candidates[key].crossSection[result.label]
        containers[key]=container
    styles=Styles(list(suite.styles.styles)+[heated])
    # No operation capable of failing remains after this publication boundary.
    for key,r in entries.items():r.crossSection=containers[key]
    suite.styles=styles
    return suite

"""Complete model-suite BW assembly, local links and atomic style attachment.

All format imports are deferred to callers. Source data are copied, hash-guarded
and never edited by reconstruction. The supported material is a neutron/lab
suite whose resonance ranges are all BW and whose additive sums close.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import re
from types import MappingProxyType
import numpy as np

from .assemble import prepare_backgrounds
from .breit_wigner import evaluate_bw
from .prepare import evaluate_region, group_radii, group_knots, group_breaks
from .context import NeutronContext
from .grid import ReconstructionOptions, ReconstructionConvergenceError, linearize, error_ratio
from .prepare import prepare_resonances, UnsupportedResonanceError
from .tabulate import _Segment, _seeds


CONTAINERS=('reactions','sums','fissionComponents','productions','orphanProducts','incompleteReactions')


@dataclass(frozen=True,order=True)
class ReactionKey:
    """Model identity; MT is used only to project BW partials and ENDF output."""
    container: str
    label: str


def _entries(suite):
    entries={}
    for container in CONTAINERS:
        for reaction in getattr(suite,container):
            key=ReactionKey(container,reaction.label)
            if key in entries:raise ValueError(f'duplicate reaction identity {key}')
            entries[key]=reaction
    return entries


def _fingerprint(suite,style):
    h=hashlib.sha256()
    def add(value):
        h.update(type(value).__qualname__.encode()+b':')
        if isinstance(value,np.ndarray):
            h.update(str(value.dtype).encode()+repr(value.shape).encode()+value.tobytes())
        elif is_dataclass(value):
            for f in fields(value):h.update(f.name.encode()+b'=');add(getattr(value,f.name))
        elif isinstance(value,dict):
            for key in sorted(value,key=repr):add(key);add(value[key])
        elif isinstance(value,(tuple,list)):
            for child in value:add(child)
        elif isinstance(value,Enum):add(value.value)
        elif isinstance(value,np.generic):add(value.item())
        elif isinstance(value,(str,int,float,bool,type(None))):h.update(repr(value).encode())
        else:raise TypeError(f'cannot hash reconstruction input {type(value).__name__}')
        h.update(b';')
    add((suite.projectile,suite.target,suite.projectileFrame,suite.resonances,suite.PoPs,suite.styles.styles,
         getattr(suite.provenance,'headerFields',None),suite.report))
    for key,reaction in _entries(suite).items():
        add((key,reaction.id,reaction.crossSection.get(style),getattr(reaction,'summands',None),reaction.outputChannel.Q))
    return h.hexdigest()


def _path(key):
    tags={'reactions':'reaction','orphanProducts':'orphanProduct',
          'fissionComponents':'fissionComponent','productions':'production','incompleteReactions':'reaction'}
    branch='sums/crossSectionSums/crossSectionSum' if key.container=='sums' else f'{key.container}/{tags[key.container]}'
    if "'" in key.label and '"' in key.label:raise UnsupportedResonanceError('link labels containing both quote types require XPath escaping')
    quote='"' if "'" in key.label else "'"
    return f"/reactionSuite/{branch}[@label={quote}{key.label}{quote}]/crossSection"


def _split_path(path):
    out=[];start=0;quote=None;brackets=0
    for i,c in enumerate(path):
        if quote:
            if c==quote:quote=None
        elif c in ('"',"'"):quote=c
        elif c=='[':brackets+=1
        elif c==']':brackets-=1
        elif c=='/' and brackets==0:
            if i>start:out.append(path[start:i])
            start=i+1
    if start<len(path):out.append(path[start:])
    if quote or brackets:raise ValueError('invalid local cross-section link')
    return out


def _normalize_link(href,base):
    if href.startswith('#'):href=href[1:]
    if '#' in href or '://' in href:
        raise UnsupportedResonanceError('external cross-section links must be normalized into the model first')
    parts=[] if href.startswith('/') else _split_path(base)
    for step in _split_path(href):
        if step=='..':
            if not parts:raise ValueError('link escapes reactionSuite')
            parts.pop()
        elif step!='.':parts.append(step)
    return '/'+ '/'.join(parts)


def _target(href,base,entries,style):
    path=_normalize_link(href,base)
    def semantic(path):
        out=[]
        for step in _split_path(path):
            match=re.fullmatch(r"([^\[]+)(?:\[@label=([\"'])(.*?)\2\])?",step)
            if match is None:raise UnsupportedResonanceError('only local attribute-label links are implemented')
            out.append((match[1],match[3]))
        return out
    steps=semantic(path)
    for key in entries:
        prefix=semantic(_path(key))
        if steps==prefix:return key,style
        if len(steps)==len(prefix)+1 and steps[:-1]==prefix and steps[-1][1] is not None:
            return key,steps[-1][1]
    raise ValueError(f'cross-section link has no model target: {href}')


def _source_forms(entries,style):
    from kika.nuclear_data.model import Reference
    resolved={}
    def visit(key,label,active):
        identity=(key,label)
        if identity in active:raise ValueError('cycle in cross-section References')
        form=entries[key].crossSection.get(label)
        if isinstance(form,Reference):
            other,target_label=_target(form.href,_path(key)+f"/reference[@label='{label}']",entries,label)
            if target_label != style:
                raise UnsupportedResonanceError("References must select the requested source style")
            return visit(other,target_label,active|{identity})
        if form is None:raise ValueError(f'no source form {label!r} for {key}')
        return form
    for key in entries:resolved[key]=visit(key,style,set())
    return resolved


def _graph(entries,style):
    graph={}
    for key,reaction in entries.items():
        if key.container!='sums':continue
        summands=getattr(reaction,'summands',None)
        if not summands:raise UnsupportedResonanceError(f'sum {key.label} has no component graph; completeness cannot be certified')
        targets=[_target(part.href,_path(key).rsplit('/crossSection',1)[0]+'/summands/add',entries,style) for part in summands]
        if any(label!=style for _,label in targets):raise UnsupportedResonanceError('sum links must select the requested source style')
        graph[key]=tuple(target for target,_ in targets)
    order=[];leaves={}
    def visit(key,active):
        if key in active:raise ValueError('cycle in reaction sum graph')
        if key in leaves:return leaves[key]
        if key not in graph:return {key}
        covered=set()
        for child in graph[key]:
            sub=visit(child,active|{key})
            if sub&covered:raise ValueError('reaction sum double counts a leaf through redundant branches')
            covered|=sub
        leaves[key]=covered;order.append(key)
        return covered
    for key in graph:visit(key,set())
    return graph,tuple(order)


def _curves(forms):
    from kika.nuclear_data.model import ResonancesWithBackground,Background
    out={}
    for key,form in forms.items():
        if isinstance(form,ResonancesWithBackground):
            if form.resonanceRegionHref:
                target=_normalize_link(form.resonanceRegionHref,_path(key)+"/resonancesWithBackground/resonances")
                if target not in ('/reactionSuite/resonances','/reactionSuite/resonances/resolved'):
                    raise ValueError('resonance link does not identify this suite\'s resolved parameters')
            if form.background is None:raise ValueError('missing resonance background')
            form=form.background
        if isinstance(form,Background):
            if form.unresolvedRegion is not None:raise UnsupportedResonanceError('URR backgrounds are not implemented')
            pieces=[v for v in (form.resolvedRegion,form.fastRegion) if v is not None]
            if not pieces:raise ValueError('empty resonance background')
        else:pieces=[form]
        curves=[]
        for piece in pieces:curves.extend(prepare_backgrounds({2:piece})[2])
        curves.sort(key=lambda c:c.x[0])
        if any(a.x[-1]>b.x[0] for a,b in zip(curves[:-1],curves[1:])):
            raise ValueError('overlapping resolved/fast background domains')
        out[key]=tuple(curves)
    return out


@dataclass(frozen=True)
class _SuiteSegment:
    low: float
    high: float
    curves: object
    region: object
    left_high: bool


@dataclass(frozen=True)
class SuiteReconstructionResult:
    forms: object
    report: object
    source_style: str
    label: str
    options: ReconstructionOptions
    _prepared: object
    _segments: tuple
    _graph: object
    _order: tuple
    _mt_keys: object
    _source_hash: str

    def _evaluate_segment(self,s,points,include_resonances=True):
        e=np.asarray(points,dtype=float).copy()
        if s.left_high:e[e==s.high]=np.nextafter(s.high,s.low)
        values={key:(curve.evaluate(e) if curve is not None else np.zeros(len(e))) for key,curve in s.curves.items()}
        if include_resonances and s.region is not None:
            for start in range(0,len(e),self.options.block_size):
                sl=slice(start,start+self.options.block_size)
                physical=evaluate_region(e[sl],s.region,self._prepared.context)
                for g in s.region.groups:
                    if g.competitive_in_background:
                        owner = self._mt_keys.get(g.competitive_mt)
                        if owner is None or owner in self._graph:
                            raise UnsupportedResonanceError("competitive background requires an exclusive model reaction")
                        physical[g.competitive_mt]-=evaluate_bw(e[sl],(g,),s.region.approximation,self._prepared.context)[g.competitive_mt]
                for mt,value in physical.items():
                    if mt==1:continue
                    key=self._mt_keys.get(mt)
                    if key is not None and key not in self._graph:values[key][sl]+=value
                    elif np.any(value!=0):raise UnsupportedResonanceError(f'MF2 partial MT{mt} has no exclusive model reaction')
        if include_resonances:
            for key in self._order:values[key]=sum((values[part] for part in self._graph[key]),np.zeros(len(e)))
        if any(np.any(~np.isfinite(v)) for v in values.values()):raise FloatingPointError('nonfinite suite cross section')
        return values

    def evaluate(self,energies):
        e=np.asarray(energies,dtype=float)
        if e.ndim>1 or np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError('finite positive scalar or 1D energies required')
        flat=e.reshape(-1);owner=np.full(len(flat),-1)
        for i,s in enumerate(self._segments):owner[(flat>=s.low)&(flat<=s.high)]=i
        if np.any(owner<0):raise ValueError('energy outside complete reconstructed domain')
        values={key:np.zeros(len(flat)) for key in self.forms}
        for i,s in enumerate(self._segments):
            indices=np.flatnonzero(owner==i)
            if len(indices):
                for key,v in self._evaluate_segment(s,flat[indices]).items():values[key][indices]=v
        return {key:v.reshape(e.shape) for key,v in values.items()}

    @property
    def forms_by_mt(self):
        return {mt:self.forms[key] for mt,key in self._mt_keys.items()}

    def verify_source(self, suite):
        """Reject stale evaluated data, including edits made after attachment."""
        candidate = deepcopy(suite)
        try:
            candidate.styles.styles = [candidate.styles[label]
                for label in self.report['source_style_labels']]
        except KeyError as error:
            raise ValueError('source styles changed since reconstruction') from error
        if _fingerprint(candidate, self.source_style) != self._source_hash:
            raise ValueError('evaluated source changed since reconstruction')

    def verify_suite(self,suite,*,label=None):
        """Verify a complete model reloaded from GNDS or a processed ENDF tape."""
        from kika.nuclear_data.model import Regions1d
        for name in ('projectile','target','projectileFrame'):
            if str(getattr(suite,name))!=self.report[name]:
                raise ReconstructionConvergenceError(f'serialized material identity changed: {name}')
        entries=_entries(suite);chosen=self.label if label is None else label
        projected={r.ENDF_MT:key for key,r in entries.items() if r.ENDF_MT is not None}
        maxima={key:0. for key in self.forms}
        checked={}
        worst={}
        for key,original in self.forms.items():
            target=key if key in entries else next((projected[mt] for mt,k in self._mt_keys.items() if k==key and mt in projected),None)
            if target is None:raise ValueError(f'reloaded suite lost reaction {key}')
            form=entries[target].crossSection.get(chosen)
            if form is None:raise ValueError(f'reloaded suite lacks selected form for {key}')
            curves=form.function1ds if isinstance(form,Regions1d) else [form]
            if len(curves)!=len(self._segments):raise ReconstructionConvergenceError('serialized region structure changed')
            checked[key]=curves
        first=next(iter(self.forms.values()))
        originals=first.function1ds if isinstance(first,Regions1d) else [first]
        for i,(segment,source_curve) in enumerate(zip(self._segments,originals)):
            grid=source_curve.xs
            fractions=np.array([.1732050807568877,.3819660112501051,.6180339887498949,.8267949192431123])
            points=np.r_[grid,(grid[:-1,None]+np.diff(grid)[:,None]*fractions).ravel()]
            reference=self._evaluate_segment(segment,points)
            serialized={}
            for key,curves in checked.items():
                curve=curves[i]
                if curve.domainMin!=segment.low or curve.domainMax!=segment.high:raise ReconstructionConvergenceError('serialization moved a domain boundary')
                if curve.domainUnit!='eV' or curve.rangeUnit!='b' or curve.endfInterpolationCode!=2:
                    raise ReconstructionConvergenceError('serialized units or interpolation changed')
                if np.any(np.diff(curve.xs)<0) or np.any(~np.isfinite(curve.xs)) or np.any(~np.isfinite(curve.ys)):
                    raise ReconstructionConvergenceError('invalid serialized table')
                values=np.asarray(curve.evaluate(points))
                if values.shape!=points.shape or np.any(~np.isfinite(values)):
                    raise ReconstructionConvergenceError('invalid serialized values')
                serialized[key]=values
                ratio=error_ratio(reference[key],values,self.options)
                maximum=float(np.max(ratio))
                if maximum>maxima[key]:
                    index=int(np.argmax(ratio))
                    worst[key]=(float(points[index]),float(reference[key][index]),float(values[index]))
                maxima[key]=max(maxima[key],maximum)
            for key in self._order:
                components=sum((serialized[part] for part in self._graph[key]),np.zeros(len(points)))
                if np.any(error_ratio(serialized[key],components,self.options)>1):
                    raise ReconstructionConvergenceError(f'serialized sum does not close within the total budget: {key}')
        if any(not np.isfinite(v) or v>1 for v in maxima.values()):raise ReconstructionConvergenceError(f'reloaded suite exceeds total budget: {maxima}; worst (eV, reference b, serialized b): {worst}')
        return maxima


def reconstruct_suite(suite,context=None,*,source_style='eval',label='recon',options=None):
    """Reconstruct a complete supported BW/RM model suite without mutating it.

    Local References and native summand links are resolved. All sums must have
    a graph and their evaluated values must close at the requested accuracy.
    Existing partials are preserved outside supported resonance domains; output aggregates are
    rebuilt once from exclusive leaves. No partial success or automatic fallback.
    """
    from kika.nuclear_data.model import ReactionSuite,CrossSectionReconstructed,Evaluated,Axis,Axes,XYs1d,Regions1d
    if not isinstance(suite,ReactionSuite):raise TypeError('expected model ReactionSuite')
    options=ReconstructionOptions() if options is None else options
    if not isinstance(options,ReconstructionOptions):raise TypeError('expected ReconstructionOptions')
    if not label or label==source_style:raise ValueError('reconstructed label must differ from the source style')
    if label in suite.styles.labels:raise ValueError('output style already exists')
    chain=suite.styles.chain(source_style)
    if any(isinstance(s,CrossSectionReconstructed) for s in chain):raise ValueError('source style already includes resonance reconstruction')
    if not isinstance(chain[-1],Evaluated):raise ValueError('source style must derive from an evaluated style')
    if getattr(suite.provenance,'headerFields',{}).get('lrp')==2:
        raise ValueError('source ENDF already contains resonance contributions (LRP=2)')
    if suite.report is not None and not suite.report.isClean:raise UnsupportedResonanceError('suite conversion reports losses or unsupported data')
    if suite.projectile!='n' or str(suite.projectileFrame)!='lab':raise UnsupportedResonanceError('only incident neutron laboratory suites are implemented')
    if len(suite.incompleteReactions):raise UnsupportedResonanceError('incomplete reactions prevent material coverage certification')
    if context is None:
        try:target=suite.PoPs[suite.target];neutron=suite.PoPs[suite.projectile]
        except KeyError as exc:raise ValueError('supply explicit NeutronContext when model particles are incomplete') from exc
        if target.mass is None or neutron.mass is None or target.spin is None:
            raise ValueError('supply explicit NeutronContext when model mass/spin are incomplete')
        context=NeutronContext(target.mass.convertedTo('amu').value/neutron.mass.convertedTo('amu').value,
                               target.spin.convertedTo('hbar').value)
    from .prepare_r_matrix import normalize_suite_pairs
    prepared=prepare_resonances(normalize_suite_pairs(suite,context),context,conversion_report=suite.report)
    source_hash=_fingerprint(suite,source_style)
    snapshot=deepcopy(suite)
    entries=_entries(snapshot)
    if not entries:raise ValueError('no model cross sections')
    if any(label in r.crossSection for r in entries.values()):raise ValueError('output form label already exists')
    forms=_source_forms(entries,source_style)
    graph,order=_graph(entries,source_style)
    curves=_curves(forms)
    mt_keys={}
    for key,r in entries.items():
        if r.ENDF_MT is not None:
            if r.ENDF_MT in mt_keys:raise UnsupportedResonanceError('ambiguous MT projection for model reactions')
            mt_keys[r.ENDF_MT]=key
    if 1 in mt_keys and mt_keys[1] not in graph:raise UnsupportedResonanceError('total must have an explicit additive model graph')
    low=min(c.x[0] for cs in curves.values() for c in cs)
    high=max(c.x[-1] for cs in curves.values() for c in cs)
    declared=chain[-1].projectileEnergyDomain
    if declared is not None:
        if declared.unit!='eV' or declared.min!=low or declared.max!=high:
            raise UnsupportedResonanceError('modeled cross sections do not cover the evaluated projectile energy domain')
    if any(r.low<low or r.high>high for r in prepared.regions):raise ValueError('resonance region outside modeled material domain')
    cuts={low,high}
    for r in prepared.regions:cuts.update((r.low,r.high))
    for cs in curves.values():
        for c in cs:
            cuts.update((c.x[0],c.x[-1]))
            if c.law==1:cuts.update(c.x[1:])
    for r in prepared.regions:
        for g in r.groups:
            cuts.update(x for x in group_breaks(g) if r.low<x<r.high)
            for radius in group_radii(g):
                previous=1
                for nbt,law in radius.interpolation:
                    if law==1:cuts.update(x for x in radius.energies[previous:nbt] if r.low<x<r.high)
                    previous=nbt
    edges=sorted(cuts);segments=[]
    for a,b in zip(edges[:-1],edges[1:]):
        region=next((r for r in prepared.regions if r.low<=a and r.high>=b),None)
        local={}
        for key,cs in curves.items():
            candidates=[c for c in cs if c.x[0]<=a and c.x[-1]>=b]
            if len(candidates)>1:raise ValueError('ambiguous source region ownership')
            if not candidates and not (b<=cs[0].x[0] or a>=cs[-1].x[-1]):
                raise UnsupportedResonanceError(f'source function has an uncovered internal gap: {key}')
            local[key]=candidates[0] if candidates else None
        segments.append(_SuiteSegment(a,b,MappingProxyType(local),region,b<high))
    result=SuiteReconstructionResult(MappingProxyType({}),MappingProxyType({}),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash)
    tables=[];checks=[];points=0;source_balance={key:0. for key in graph}
    for s in segments:
        seed_curves={i:(c,) for i,c in enumerate(s.curves.values()) if c is not None}
        if s.region is not None:
            seeds=_seeds(_Segment(s.low,s.high,s.region,seed_curves,s.left_high),context)
        else:
            # Original knots suffice for linear fast-region data. Inserting
            # unnecessary near-threshold points amplifies ENDF x-rounding.
            seeds=sorted({s.low,s.high}|{x for c in seed_curves.values() for x in c[0].x if s.low<=x<=s.high})
            for cs in seed_curves.values():
                c=cs[0]
                if c.law==1 and c.x[-1]==s.high and c.y[-1]!=c.y[-2]:
                    seeds.append(float(np.nextafter(s.high,s.low)))
            seeds=sorted(set(seeds))
        # Before changing an aggregate, prove that its stated components cover
        # it at source knots and interior points. No invented residual channel.
        grid=np.asarray(seeds)
        probe=np.r_[grid,(grid[:-1]+grid[1:])*.5]
        original=result._evaluate_segment(s,probe,False)
        for key in order:
            summed=sum((original[part] for part in graph[key]),np.zeros(len(probe)))
            ratio=error_ratio(original[key],summed,options)
            source_balance[key]=max(source_balance[key],float(np.max(ratio)))
            if np.any(ratio>1):
                index=int(np.argmax(ratio))
                raise UnsupportedResonanceError(f'evaluated sum {key} does not close; missing components or inconsistent source; eV={probe[index]}, stated b={original[key][index]}, components b={summed[index]}, ratio={ratio[index]}')
            original[key]=summed
        x,y,check=linearize(lambda e:result._evaluate_segment(s,e),seeds,options,options.max_points-points)
        points+=len(x);tables.append((x,y));checks.append(dict(domain=(s.low,s.high),points=len(x),**check))
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')]);output={}
    for key in entries:
        children=[XYs1d(x.copy(),y[key].copy(),axes=axes,index=i) for i,(x,y) in enumerate(tables)]
        output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
        output[key].label=label
    normalized_hash=hashlib.sha256((source_hash+repr(context)+repr(options)).encode()).hexdigest()
    minima={key:min(float(np.min(y[key])) for _,y in tables) for key in entries}
    report=dict(engine='kika-bw-suite-1',scope='all modeled cross sections across evaluated domain',
        source_sha256=source_hash,normalized_sha256=normalized_hash,minima=minima,
        projectile=str(suite.projectile),target=str(suite.target),projectileFrame=str(suite.projectileFrame),
        domain=(low,high),points=points,regions=checks,source_sum_error_ratios=source_balance,
        source_style=source_style,source_style_labels=tuple(suite.styles.labels),
        label=label,options=options,context=context,empirical_verification=True,global_error_bound=False)
    return SuiteReconstructionResult(MappingProxyType(output),MappingProxyType(report),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash)


def attach_reconstruction(suite,result):
    """Validate and attach a complete result atomically, preserving evaluated forms."""
    from kika.nuclear_data.model import CrossSectionReconstructed
    if not isinstance(result,SuiteReconstructionResult):raise TypeError('expected SuiteReconstructionResult')
    if _fingerprint(suite,result.source_style)!=result._source_hash:
        raise ValueError('suite changed since reconstruction; recompute before attaching')
    entries=_entries(suite)
    if set(entries)!=set(result.forms):raise ValueError('reaction coverage changed')
    if result.label in suite.styles.labels or any(result.label in r.crossSection for r in entries.values()):
        raise ValueError('reconstruction label already exists')
    # Verify even if the caller edited the mutable canonical output forms.
    candidate=deepcopy(suite)
    for key,reaction in _entries(candidate).items():reaction.crossSection[result.label]=deepcopy(result.forms[key])
    candidate.styles.add(CrossSectionReconstructed(result.label,derivedFrom=result.source_style))
    result.verify_suite(candidate)
    # Publish prebuilt containers only after every operation that can fail.
    candidates=_entries(candidate)
    containers={}
    for key,reaction in entries.items():
        container=deepcopy(reaction.crossSection)
        for existing,form in reaction.crossSection.items():container[existing]=form
        container[result.label]=candidates[key].crossSection[result.label]
        containers[key]=container
    for key,reaction in entries.items():reaction.crossSection=containers[key]
    suite.styles=candidate.styles
    return suite


from .prepare import group_radii, group_knots, group_breaks

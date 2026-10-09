"""Complete model-suite resonance assembly, local links and atomic style attachment.

All format imports are deferred to callers. Source data are copied, hash-guarded
and never edited by reconstruction. The supported material is a neutron/lab
suite whose resonance ranges are supported and whose additive sums close.
"""
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import re
from types import MappingProxyType
import numpy as np

from .assemble import prepare_backgrounds
from .breit_wigner import evaluate_bw
from .prepare import evaluate_region, group_radii, group_knots, group_breaks, region_mts
from .context import NeutronContext
from .grid import ReconstructionOptions, ReconstructionConvergenceError, linearize, error_ratio, verification_batches
from .prepare import PreparedResonances, prepare_resonances, UnsupportedResonanceError
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
        if not summands:raise UnsupportedResonanceError(f'sum {key.label} has no component graph; completeness cannot be certified',category='sum-without-graph')
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


def _sum_leaves(key,graph):
    return {key} if key not in graph else set().union(*(_sum_leaves(part,graph) for part in graph[key]))


def _curves(forms):
    from kika.nuclear_data.model import ResonancesWithBackground,Background
    out={}
    for key,form in forms.items():
        if isinstance(form,ResonancesWithBackground):
            if form.resonanceRegionHref:
                target=_normalize_link(form.resonanceRegionHref,_path(key)+"/resonancesWithBackground/resonances")
                if target not in ('/reactionSuite/resonances','/reactionSuite/resonances/resolved','/reactionSuite/resonances/unresolved'):
                    raise ValueError('resonance link does not identify this suite\'s resolved parameters')
            if form.background is None:raise ValueError('missing resonance background')
            form=form.background
        if isinstance(form,Background):
            out[key]=prepare_backgrounds({2:form})[2]
            continue
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
    _shared_linear: object=field(init=False,repr=False,compare=False)

    def __post_init__(self):
        from kika.algebra.prepared import _prepare_shared_linear_evaluator
        groups={}
        for key,curve in self.curves.items():
            if curve is None or len(curve.x)<2:continue
            laws=tuple(np.broadcast_to(curve._laws,(len(curve.x)-1,)))
            # Uniform lin-lin tables already use NumPy's efficient interp.
            if all(law==2 for law in laws):continue
            groups.setdefault((tuple(curve.x),laws),[]).append((key,curve))
        shared=[]
        for (x,laws),members in groups.items():
            if len(members)<2:continue
            ys=np.column_stack([curve.y for _,curve in members])
            shared.append((tuple(key for key,_ in members),
                           _prepare_shared_linear_evaluator(x,ys,laws)))
        object.__setattr__(self,'_shared_linear',tuple(shared))


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
    _physical_owners: object
    _grids: tuple=()
    _verification_cache: object=field(default_factory=dict,init=False,repr=False,compare=False)

    def _evaluate_segment(self,s,points,include_resonances=True,*,physical_witness=None,constant_values=None,shared_linear=True):
        e=np.asarray(points,dtype=float).copy()
        if s.left_high:e[e==s.high]=np.nextafter(s.high,s.low)
        constant_values={} if constant_values is None else constant_values
        grouped={}
        # Independent verification keeps the ordinary background readers as
        # well as the direct NumPy physical kernel.
        if shared_linear and physical_witness is None:
            for keys,reader in s._shared_linear:
                needed=[(i,key) for i,key in enumerate(keys) if key not in constant_values
                        and (not include_resonances or key not in self._graph)]
                if len(needed)<2:continue
                batch=reader(e,max_cells=max(1,min(65536,self.options.max_work_bytes//64)))
                if batch is not None:grouped.update((key,batch[:,i]) for i,key in needed)
        values={}
        for key,curve in s.curves.items():
            if key in constant_values:values[key]=np.full(len(e),constant_values[key])
            elif key in grouped:values[key]=grouped[key]
            elif curve is not None and (not include_resonances or key not in self._graph):
                values[key]=curve.evaluate(e)
            else:values[key]=np.zeros(len(e))
        total=self._mt_keys.get(1)
        total_parts=_sum_leaves(total,self._graph) if total is not None else None
        if include_resonances and s.region is not None:
            for start in range(0,len(e),self.options.block_size):
                sl=slice(start,start+self.options.block_size)
                if physical_witness is not None and start in physical_witness:
                    saved_e,physical=physical_witness[start]
                    if not np.array_equal(saved_e,e[sl]):raise ValueError('physical witness energy mismatch')
                else:
                    physical=evaluate_region(e[sl],s.region,self._prepared.context,work_bytes=self.options.max_work_bytes,
                        absorption_rtol=0. if physical_witness is not None else min(1e-8,self.options.rtol*1e-5))
                    for g in s.region.groups:
                        if not g.competitive_in_background:continue
                        owner = self._mt_keys.get(g.competitive_mt)
                        if owner is None or owner in self._graph:
                            raise UnsupportedResonanceError("competitive background requires an exclusive model reaction")
                        physical[g.competitive_mt]-=evaluate_bw(e[sl],(g,),s.region.approximation,self._prepared.context)[g.competitive_mt]
                    if physical_witness is not None:
                        # Store the physical partials, never the candidate table
                        # or the assembled sums. Bytes own an immutable snapshot.
                        freeze=lambda a:np.frombuffer(np.ascontiguousarray(a,dtype=float).tobytes(),dtype=float)
                        physical_witness[start]=(freeze(e[sl]),{mt:freeze(v) for mt,v in physical.items() if mt!=1})
                for mt,value in physical.items():
                    if mt==1:continue
                    key=self._physical_owners.get(mt)
                    if key is not None and key not in self._graph:
                        if total_parts is not None and key not in total_parts and np.any(value!=0):
                            raise UnsupportedResonanceError(f'total graph omits resonance partial MT{mt}',category='incomplete-total-graph')
                        values[key][sl]+=value
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
        from kika.algebra import prepare_evaluator
        for name in ('projectile','target','projectileFrame'):
            if str(getattr(suite,name))!=self.report[name]:
                raise ReconstructionConvergenceError(f'serialized material identity changed: {name}')
        entries=_entries(suite);chosen=self.label if label is None else label
        projected={r.ENDF_MT:key for key,r in entries.items() if r.ENDF_MT is not None}
        maxima={key:0. for key in self.forms}
        checked={};covered=set()
        worst={}
        for key,original in self.forms.items():
            target=key if key in entries else next((projected[mt] for mt,k in self._mt_keys.items() if k==key and mt in projected),None)
            if target is None:raise ValueError(f'reloaded suite lost reaction {key}')
            covered.add(target)
            form=entries[target].crossSection.get(chosen)
            if form is None:raise ValueError(f'reloaded suite lacks selected form for {key}')
            curves=form.function1ds if isinstance(form,Regions1d) else [form]
            if len(curves)!=len(self._segments):raise ReconstructionConvergenceError('serialized region structure changed')
            checked[key]=curves
        if covered!=set(entries):raise ReconstructionConvergenceError('serialized reaction coverage changed')
        first=next(iter(self.forms.values()))
        originals=first.function1ds if isinstance(first,Regions1d) else [first]
        grids=self._grids or tuple(c.xs for c in originals)
        # A verified numeric snapshot can be reused only when every selected
        # table, its interpretation, and the frozen reference are identical.
        # Hash contents on each call, never object identity or a style label.
        digest=hashlib.sha256(repr((self._prepared,self.options,self._graph,self._order)).encode())
        def array(value):
            value=np.ascontiguousarray(value)
            digest.update(repr((value.dtype.str,value.shape)).encode())
            digest.update(memoryview(value).cast('B'))
        for grid in grids:array(grid)
        reference_signature=digest.digest()
        witnesses=self._verification_cache.get('physical_witnesses')
        if witnesses is None or witnesses[0]!=reference_signature:
            witnesses=(reference_signature,{})
            self._verification_cache['physical_witnesses']=witnesses
        for key,curves in checked.items():
            digest.update(repr(key).encode())
            for curve in curves:
                digest.update(repr((type(curve).__qualname__,curve.domainUnit,curve.rangeUnit,
                    curve.endfInterpolationCode,curve.domainMin,curve.domainMax)).encode())
                array(curve.xs);array(curve.ys)
        signature=digest.digest()
        cached=self._verification_cache.get('entry')
        if cached is not None and cached[0]==signature:
            return dict(cached[1])
        for i,(segment,grid) in enumerate(zip(self._segments,grids)):
            fractions=np.array([.1732050807568877,.3819660112501051,.6180339887498949,.8267949192431123])
            evaluators={}
            for key,curves in checked.items():
                curve=curves[i]
                if curve.domainMin!=segment.low or curve.domainMax!=segment.high:raise ReconstructionConvergenceError('serialization moved a domain boundary')
                if curve.domainUnit!='eV' or curve.rangeUnit!='b' or curve.endfInterpolationCode!=2:
                    raise ReconstructionConvergenceError('serialized units or interpolation changed')
                try:evaluators[key]=prepare_evaluator(curve.xs,curve.ys,2)
                except ValueError as exc:raise ReconstructionConvergenceError('invalid serialized table') from exc
            verification_batch=max(1,min(32768,self.options.max_work_bytes//(64*(len(checked)+1))))
            for batch,points in enumerate(verification_batches(grid,verification_batch,fractions)):
                # Fresh/edited/rounded output is checked again at every original
                # node and probe. Only the immutable physical reference is reused.
                witness=witnesses[1].setdefault((i,batch),{})
                reference=self._evaluate_segment(segment,points,physical_witness=witness)
                serialized={}
                for key,evaluator in evaluators.items():
                    values=np.asarray(evaluator(points))
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
        self._verification_cache['entry']=(signature,dict(maxima))
        return maxima


def reconstruct_suite(suite,context=None,*,source_style='eval',label='recon',options=None):
    """Reconstruct a complete supported resonance model suite without mutating it.

    Local References and native summand links are resolved. All sums must have
    a graph. Evaluated aggregate discrepancies are recorded; output aggregates
    are derived from their exclusive leaves and must close after publication.
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
        raise UnsupportedResonanceError('source ENDF already contains resonance contributions (LRP=2)',category='already-reconstructed')
    if getattr(suite.provenance,'headerFields',{}).get('lrp')==1 and suite.resonances is None:
        raise UnsupportedResonanceError('source declares resonance parameters but MF2 is absent from the model',category='missing-resonance-regions')
    if suite.report is not None and not suite.report.isCleanFor('cross-sections'):raise UnsupportedResonanceError('suite conversion reports losses or unsupported data affecting cross sections',category='conversion-not-clean')
    if suite.projectile!='n' or str(suite.projectileFrame)!='lab':raise UnsupportedResonanceError('only incident neutron laboratory suites are implemented')
    if len(suite.incompleteReactions):raise UnsupportedResonanceError('incomplete reactions prevent material coverage certification',category='incomplete-reactions')
    has_resonances=suite.resonances is not None and (suite.resonances.resolved or suite.resonances.unresolved is not None)
    if context is not None and not isinstance(context,NeutronContext):raise TypeError('expected NeutronContext')
    if context is None and has_resonances:
        try:target=suite.PoPs[suite.target];neutron=suite.PoPs[suite.projectile]
        except KeyError as exc:raise ValueError('supply explicit NeutronContext when model particles are incomplete') from exc
        if target.mass is None or neutron.mass is None or target.spin is None:
            raise ValueError('supply explicit NeutronContext when model mass/spin are incomplete')
        context=NeutronContext(target.mass.convertedTo('amu').value/neutron.mass.convertedTo('amu').value,
                               target.spin.convertedTo('hbar').value)
    from .prepare_r_matrix import normalize_suite_pairs
    from .unresolved import normalize_unresolved_links
    resonances=normalize_unresolved_links(suite,normalize_suite_pairs(suite,context),source_style)
    prepared=(PreparedResonances(context,()) if resonances is None else
              prepare_resonances(resonances,context,conversion_report=suite.report,allow_empty=True))
    source_hash=_fingerprint(suite,source_style)
    snapshot=deepcopy(suite)
    entries=_entries(snapshot)
    if not entries:raise ValueError('no model cross sections')
    if any(label in r.crossSection for r in entries.values()):raise ValueError('output form label already exists')
    forms=_source_forms(entries,source_style)
    if not prepared.regions:
        from kika.nuclear_data.model import ResonancesWithBackground
        if any(isinstance(f,ResonancesWithBackground) for f in forms.values()):
            raise UnsupportedResonanceError('resonance backgrounds require modeled resonance regions',category='missing-resonance-regions')
    graph,order=_graph(entries,source_style)
    curves=_curves(forms)
    mt_keys={}
    for key,r in entries.items():
        if r.ENDF_MT is not None:
            if r.ENDF_MT in mt_keys:raise UnsupportedResonanceError('ambiguous MT projection for model reactions')
            mt_keys[r.ENDF_MT]=key
    physical_owners=dict(mt_keys)
    # ENDF-102 §3.4.5 assigns the MF2 fission contribution to MT19 when
    # MT18 is represented by its chance partials. Rebuild MT18 through the
    # model graph afterwards, so that contribution is included only once.
    if mt_keys.get(18) in graph:
        parts=_sum_leaves(mt_keys[18],graph);first_chance=mt_keys.get(19)
        allowed={mt_keys[mt] for mt in (19,20,21,38) if mt in mt_keys}
        if first_chance in parts and first_chance not in graph and parts<=allowed:
            physical_owners[18]=first_chance
    elif 18 not in mt_keys and 19 in mt_keys and mt_keys[19] not in graph:
        physical_owners[18]=mt_keys[19]
    if 1 in mt_keys and mt_keys[1] not in graph:raise UnsupportedResonanceError('total must have an explicit additive model graph',category='total-without-graph')
    low=min(c.x[0] for cs in curves.values() for c in cs)
    high=max(c.x[-1] for cs in curves.values() for c in cs)
    declared=chain[-1].projectileEnergyDomain
    if declared is not None:
        if declared.unit!='eV' or declared.min!=low or declared.max!=high:
            raise UnsupportedResonanceError('modeled cross sections do not cover the evaluated projectile energy domain',category='domain-not-covered')
    if any(r.low<low or r.high>high for r in prepared.regions):raise ValueError('resonance region outside modeled material domain')
    cuts={low,high}
    for r in prepared.regions:cuts.update((r.low,r.high))
    for cs in curves.values():
        for c in cs:
            cuts.update((c.x[0],c.x[-1]))
            cuts.update(c.breaks)
    for r in prepared.regions:
        if r.unresolved is not None:cuts.update(r.unresolved.breaks)
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
                raise UnsupportedResonanceError(f'source function has an uncovered internal gap: {key}',category='internal-gap')
            local[key]=candidates[0] if candidates else None
        segments.append(_SuiteSegment(a,b,MappingProxyType(local),region,b<high))
    result=SuiteReconstructionResult(MappingProxyType({}),MappingProxyType({}),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash,MappingProxyType(physical_owners))
    tables=[];checks=[];points=0;source_balance={key:0. for key in graph};source_worst={}
    for s in segments:
        seed_curves={i:(c,) for i,(key,c) in enumerate(s.curves.items()) if c is not None and key not in graph}
        if s.region is not None:
            seeds=_seeds(_Segment(s.low,s.high,s.region,seed_curves,s.left_high),context)
        else:
            # Original knots suffice for linear fast-region data. Inserting
            # unnecessary near-threshold points amplifies ENDF x-rounding.
            seeds=sorted({s.low,s.high}|{x for c in seed_curves.values() for x in c[0].x if s.low<=x<=s.high})
            for cs in seed_curves.values():
                c=cs[0]
                if c.x[-1]==s.high and c.endpoint_jump:
                    seeds.append(float(np.nextafter(s.high,s.low)))
            seeds=sorted(set(seeds))
        # Diagnose the evaluated aggregate against exclusive leaves. Output
        # sums are derived even if the redundant source aggregate disagrees.
        # Redundant aggregates inform diagnostics, not the reconstructed mesh.
        # Keep their original knots in this separate source-comparison grid.
        diagnostic_seeds=set(seeds)|{x for c in s.curves.values() if c is not None for x in c.x if s.low<=x<=s.high}
        grid=np.asarray(sorted(diagnostic_seeds))
        probe=np.r_[grid,(grid[:-1]+grid[1:])*.5]
        original=result._evaluate_segment(s,probe,False)
        for key in order:
            summed=sum((original[part] for part in graph[key]),np.zeros(len(probe)))
            ratio=error_ratio(original[key],summed,options)
            maximum=float(np.max(ratio))
            if maximum>source_balance[key]:
                index=int(np.argmax(ratio))
                source_worst[key]=dict(energy_eV=float(probe[index]),side='left-limit' if s.left_high and probe[index]==s.high else 'point',stated_b=float(original[key][index]),components_b=float(summed[index]),error_ratio=maximum)
            source_balance[key]=max(source_balance[key],maximum)
            original[key]=summed
        affected=({physical_owners.get(mt) for mt in region_mts(s.region)}
                  if s.region is not None else set())
        constants={}
        high=np.nextafter(s.high,s.low) if s.left_high else s.high
        for key,curve in s.curves.items():
            if key in graph or key in affected:continue
            value=(0. if curve is None else curve.constant_on(s.low,high)) if high>s.low else None
            if value is not None:constants[key]=value
        for key in order:
            if all(part in constants for part in graph[key]):
                constants[key]=sum((constants[part] for part in graph[key]),0.)
        x,y,check=linearize(lambda e:result._evaluate_segment(s,e,constant_values=constants),
            seeds,options,options.max_points-points,constants=constants)
        points+=len(x);tables.append((x,y));checks.append(dict(domain=(s.low,s.high),points=len(x),**check))
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')]);output={}
    from kika.algebra import compress_flat
    for key in entries:
        children=[]
        for i,(x,y) in enumerate(tables):
            s=segments[i];curve=s.curves[key]
            affected=s.region is not None and key in {physical_owners.get(mt) for mt in region_mts(s.region)}
            if curve is not None and key not in graph and not affected:
                original_x=np.asarray(curve.x)
                laws=np.broadcast_to(curve.law,(len(original_x)-1,))
                relevant=(original_x[:-1]<s.high)&(original_x[1:]>s.low)
                if np.all(laws[relevant]==2):
                    # An untouched lin-lin background already has an exact
                    # representation on its own knots. It need not inherit
                    # every other reaction's resonance or fast-region knot.
                    x=np.unique(np.r_[s.low,original_x[(original_x>s.low)&(original_x<s.high)],s.high])
                    if not s.left_high and original_x[-1]==s.high and curve.endpoint_jump:
                        x=np.unique(np.r_[x,np.nextafter(s.high,s.low)])
                    query=x.copy()
                    if s.left_high:query[query==s.high]=np.nextafter(s.high,s.low)
                    values=curve.evaluate(query)
                else:values=y[key]
            else:values=y[key]
            cx,cy,_=compress_flat(x,values,2)
            children.append(XYs1d(cx,cy,axes=axes,index=i))
        output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
        output[key].label=label
    normalized_hash=hashlib.sha256((source_hash+repr(context)+repr(options)).encode()).hexdigest()
    minima={key:min(float(np.min(y[key])) for _,y in tables) for key in entries}
    report=dict(engine='kika-native-suite-1',scope='all modeled cross sections across evaluated domain',
        source_sha256=source_hash,normalized_sha256=normalized_hash,minima=minima,
        projectile=str(suite.projectile),target=str(suite.target),projectileFrame=str(suite.projectileFrame),
        domain=(low,high),points=points,regions=checks,source_sum_error_ratios=source_balance,
        stored_points=sum(len(c.xs) for f in output.values() for c in (f.function1ds if isinstance(f,Regions1d) else [f])),
        source_sum_policy='derive-from-leaves',source_sum_discrepancies=source_worst,
        source_conversion_report=deepcopy(suite.report),
        resonance_partial_owners=MappingProxyType(physical_owners),
        source_style=source_style,source_style_labels=tuple(suite.styles.labels),
        label=label,options=options,context=context,preparation_notes=prepared.preparation_notes,
        empirical_verification=True,global_error_bound=False)
    return SuiteReconstructionResult(MappingProxyType(output),MappingProxyType(report),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash,MappingProxyType(physical_owners),
        tuple(np.frombuffer(x.tobytes(),dtype=x.dtype) for x,_ in tables))


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

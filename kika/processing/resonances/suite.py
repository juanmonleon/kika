"""Complete model-suite resonance assembly, local links and atomic style attachment.

All format imports are deferred to callers. Source data are copied, hash-guarded
and never edited by reconstruction. The supported material is a neutron/lab
suite whose resonance ranges are supported and whose additive sums close.
"""
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import re
from types import MappingProxyType
from typing import NamedTuple
import numpy as np

from .assemble import BackgroundCurve, prepare_backgrounds
from .breit_wigner import evaluate_bw
from .prepare import evaluate_region, group_radii, group_knots, group_breaks, region_mts
from .context import NeutronContext
from .grid import ReconstructionOptions, ReconstructionConvergenceError, linearize_steps, linearize_group_steps, error_ratio, verification_batches
from .prepare import PreparedResonances, prepare_resonances, UnsupportedResonanceError
from .tabulate import _Segment, _seeds


CONTAINERS=('reactions','sums','fissionComponents','productions','orphanProducts','incompleteReactions')


class ReactionKey(NamedTuple):
    """Model identity; MT is used only to project BW partials and ENDF output.

    A tuple, so that hashing and comparing it is done in C: keys are hashed
    tens of millions of times tabulating a material cut into thousands of
    segments (JENDL-5 Cl-35: 42 M), where a dataclass's __hash__ cost 4 s.
    """
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


#: The constants of a request that declares none: one object, so that the
#: plans keyed on it are made once.
_NO_CONSTANTS=MappingProxyType({})


def _compress_and_join(pieces):
    """``_join_continuous(compress_flat_pieces(pieces))`` in one vectorised pass.

    The arithmetic is theirs: each piece keeps the nodes compress_flat keeps,
    and a piece that starts on the last node of the one before (abscissa,
    value and sign) continues it without that node. Only the per-piece
    Python work goes: JENDL-5 Cl-35 has 3 376 pieces for each of 234
    reactions.
    """
    from kika.algebra import compress_flat_pieces
    pieces=[(np.asarray(x,dtype=float),np.asarray(y,dtype=float)) for x,y in pieces]
    lengths=np.array([len(x) for x,_ in pieces])
    if len(pieces)<2 or np.any(lengths<2):
        return _join_continuous(compress_flat_pieces(pieces))
    x=np.concatenate([p[0] for p in pieces]);y=np.concatenate([p[1] for p in pieces])
    ends=np.cumsum(lengths);starts=ends-lengths
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return _join_continuous(compress_flat_pieces(pieces))
    inside=np.ones(len(x)-1,dtype=bool);inside[ends[:-1]-1]=False
    if np.any(np.diff(x)[inside]<0):
        return _join_continuous(compress_flat_pieces(pieces))
    negative=(y==0)&np.signbit(y)
    keep=np.ones(len(x),dtype=bool)
    keep[1:-1]=~((y[:-2]==y[1:-1])&(y[1:-1]==y[2:])&(x[:-2]<x[1:-1])&(x[1:-1]<x[2:])
                 &~negative[:-2]&~negative[1:-1]&~negative[2:])
    keep[starts]=True;keep[ends-1]=True
    with np.errstate(over='ignore'):
        kept=x[keep]
        gaps=~np.isfinite(np.diff(kept))
    if np.any(gaps):
        return _join_continuous(compress_flat_pieces(pieces))
    last,first=ends[:-1]-1,starts[1:]
    joined=(x[last]==x[first])&(y[last]==y[first])&(np.signbit(y[last])==np.signbit(y[first]))
    keep[first[joined]]=False
    region=np.cumsum(np.r_[0,~joined])
    owner=np.repeat(region,lengths)[keep]
    xs,ys=x[keep],y[keep]
    cut=np.flatnonzero(np.diff(owner))+1
    return list(zip(np.split(xs,cut),np.split(ys,cut)))


def _join_continuous(pieces):
    """Consecutive lin-lin pieces joined where they share their end node exactly.

    A reaction is cut wherever *any* reaction jumps (JENDL-5 Cl-35: 3 376
    segments for 234 reactions, 790 000 pieces). Where this one does not --
    the last node of a piece and the first of the next are the same abscissa
    and the same value, bit for bit -- the two are one table through that
    node. Values are untouched and every segment edge stays a node, so
    ``curvesOnDomains`` cuts the table back at the segment edges.
    """
    out=[];xs=[];ys=[]
    for x,y,_ in pieces:
        if xs and xs[-1][-1]==x[0] and ys[-1][-1]==y[0] and np.signbit(ys[-1][-1])==np.signbit(y[0]):
            xs.append(x[1:]);ys.append(y[1:])
            continue
        if xs:out.append((np.concatenate(xs),np.concatenate(ys)))
        xs=[x];ys=[y]
    if xs:out.append((np.concatenate(xs),np.concatenate(ys)))
    return out


def _knots_in(curve,low,high):
    """The knots of *curve* in [low, high], without reading the rest of it."""
    x=curve._x
    return x[np.searchsorted(x,low,'left'):np.searchsorted(x,high,'right')]


def _laws_over(curve,low,high):
    """The laws of the intervals of *curve* that overlap (low, high)."""
    x=curve._x
    laws=np.broadcast_to(curve._laws,(len(x)-1,))
    return laws[max(int(np.searchsorted(x,low,'right'))-1,0):int(np.searchsorted(x,high,'left'))]


@dataclass(frozen=True)
class _SuiteSegment:
    low: float
    high: float
    curves: object
    region: object
    left_high: bool
    _shared_linear: object=field(init=False,repr=False,compare=False)
    shared_cache: InitVar[object]=None

    def __post_init__(self,shared_cache):
        # Segments of one material share their curves; a tape whose
        # backgrounds step every few hundred eV has thousands of segments,
        # and reading every whole curve again for each made this quadratic.
        identity=tuple((key,id(curve)) for key,curve in self.curves.items())
        if shared_cache is not None and identity in shared_cache:
            object.__setattr__(self,'_shared_linear',shared_cache[identity][1])
            return
        from kika.algebra.prepared import _prepare_shared_linear_evaluator
        # Backgrounds are split at their jumps, so the set of curves changes
        # from segment to segment; what each curve contributes (its grid and
        # laws, and whether it is uniformly lin-lin) is worked out once.
        per_curve,grids,readers=({},{},{}) if shared_cache is None else (
            shared_cache.setdefault('curves',{}),shared_cache.setdefault('grids',{}),
            shared_cache.setdefault('readers',{}))
        groups={}
        for key,curve in self.curves.items():
            if curve is None or len(curve.x)<2:continue
            known=per_curve.get(id(curve))
            if known is None:
                laws=tuple(np.broadcast_to(curve._laws,(len(curve.x)-1,)))
                # Uniform lin-lin tables already use NumPy's efficient interp.
                grid=None if all(law==2 for law in laws) else grids.setdefault((tuple(curve.x),laws),len(grids))
                known=per_curve[id(curve)]=(curve,grid)
            if known[1] is not None:groups.setdefault(known[1],[]).append((key,curve))
        table={grid:shape for shape,grid in grids.items()} if groups else {}
        shared=[]
        for grid,members in groups.items():
            if len(members)<2:continue
            token=(grid,)+tuple((key,id(curve)) for key,curve in members)
            if token not in readers:
                x,laws=table[grid]
                ys=np.column_stack([curve.y for _,curve in members])
                readers[token]=(tuple(curve for _,curve in members),_prepare_shared_linear_evaluator(x,ys,laws))
            shared.append((tuple(key for key,_ in members),readers[token][1]))
        object.__setattr__(self,'_shared_linear',tuple(shared))
        if shared_cache is not None:
            # The curves are held so their ids stay theirs.
            shared_cache[identity]=(tuple(self.curves.values()),self._shared_linear)


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
    _columns_cache: object=field(default_factory=dict,init=False,repr=False,compare=False)

    def _wanted(self,only):
        """*only* and every reaction their sums are made of."""
        wanted=self._columns_cache.get(only)
        if wanted is None:
            wanted=set();stack=list(only)
            while stack:
                key=stack.pop()
                if key in wanted:continue
                wanted.add(key);stack.extend(self._graph.get(key,()))
            self._columns_cache[only]=wanted
        return wanted

    def _evaluate_segment(self,s,points,include_resonances=True,*,physical_witness=None,constant_values=None,shared_linear=True,physics=None,only=None,backgrounds=None):
        """Every reaction at *points*, or, with *only*, at least those (and
        what their sums need); each is computed as it is for the full set."""
        e=np.asarray(points,dtype=float).copy()
        if s.left_high:e[e==s.high]=np.nextafter(s.high,s.low)
        constant_values=_NO_CONSTANTS if constant_values is None else constant_values
        zeros,parts=self._sum_plan(constant_values)
        # A request for some columns leaves out the reactions that are the
        # constant zero here: they are none of its columns and add nothing.
        wanted=None if only is None else self._wanted(only)-(zeros-set(only))
        grouped={}
        # Verification keeps the ordinary background readers and evaluates the
        # physics exactly (direct capture, no Gram contraction).
        if shared_linear and physical_witness is None:
            for keys,reader in s._shared_linear:
                needed=[(i,key) for i,key in enumerate(keys) if key not in constant_values
                        and (not include_resonances or key not in self._graph)]
                if len(needed)<2:continue
                if wanted is not None and not any(key in wanted for _,key in needed):continue
                batch=reader(e,max_cells=max(1,min(65536,self.options.max_work_bytes//64)))
                if batch is not None:grouped.update((key,batch[:,i]) for i,key in needed)
        values={};shared={}
        for key,curve in s.curves.items():
            if wanted is not None and key not in wanted:continue
            # One read-only array per constant: constants are only read, by
            # sums and columns, and most of them are the same zero (keyed by
            # its bits, so that -0.0 keeps its sign).
            if key in constant_values:
                value=constant_values[key];token=float(value).hex()
                if token not in shared:
                    shared[token]=np.full(len(e),value);shared[token].setflags(write=False)
                values[key]=shared[token]
            elif key in grouped:values[key]=grouped[key]
            elif curve is not None and (not include_resonances or key not in self._graph):
                values[key]=backgrounds[key] if backgrounds is not None and key in backgrounds else curve.evaluate(e)
            else:values[key]=np.zeros(len(e))
        if 'total_parts' not in self._columns_cache:
            total=self._mt_keys.get(1)
            self._columns_cache['total_parts']=_sum_leaves(total,self._graph) if total is not None else None
        total_parts=self._columns_cache['total_parts']
        if include_resonances and s.region is not None:
            # Precomputed physics (``_answer``) is one block over every energy.
            blocks=[0] if physics is not None else range(0,len(e),self.options.block_size)
            for start in blocks:
                sl=slice(None) if physics is not None else slice(start,start+self.options.block_size)
                if physics is not None:
                    physical=physics
                elif physical_witness is not None and start in physical_witness:
                    saved_e,physical=physical_witness[start]
                    if not np.array_equal(saved_e,e[sl]):raise ValueError('physical witness energy mismatch')
                else:
                    self._check_competitive_owners(s.region)
                    if physical_witness is not None:
                        # The verification witness is exact (no Gram capture), computed
                        # by the compiled kernel; tests hold that to the NumPy solver.
                        physical=_exact_physical_block(s.region,self._prepared.context,e[sl],self.options.max_work_bytes)
                    else:
                        physical=evaluate_region(e[sl],s.region,self._prepared.context,work_bytes=self.options.max_work_bytes,
                            absorption_rtol=min(1e-8,self.options.rtol*1e-5))
                        _subtract_competitive(physical,s.region,self._prepared.context,e[sl])
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
                        if key in values:values[key][sl]+=value
                    elif np.any(value!=0):raise UnsupportedResonanceError(f'MF2 partial MT{mt} has no exclusive model reaction')
        checked=values
        if include_resonances:
            for key in self._order:
                if wanted is None or key in wanted:
                    values[key]=sum((values[part] for part in parts[key]),np.zeros(len(e)))
            checked=self._finite_witnesses((only,id(constant_values)),values,constant_values)
        # One pass over every checked value: a reaction apiece cost more than
        # the values on a segment of a few points.
        if checked and not np.all(np.isfinite(np.concatenate([np.ravel(values[key]) for key in checked]))):
            raise FloatingPointError('nonfinite suite cross section')
        return values

    def _own(self,token,owner):
        """File a request plan under the segment or constants it is for."""
        self._columns_cache.setdefault(('owner',owner),[]).append(token)

    def _forget(self,segment,constants):
        """Drop the request plans of a finished segment."""
        for owner in (segment,constants):
            for token in self._columns_cache.pop(('owner',owner),()):
                self._columns_cache.pop(token,None)

    def _sum_plan(self,constant_values):
        """The reactions that are the constant zero on a segment, and each
        sum's parts without them.

        Leaving them out is exact, bit for bit: a sum starts at +0.0, so it is
        never -0.0, and x+(+-0.0) is x. A material has hundreds of partials
        that are zero across a resonance span, and every sum read them all.
        """
        token=('sums',id(constant_values))
        plan=self._columns_cache.get(token)
        if plan is None:
            zeros=frozenset(key for key,value in constant_values.items() if value==0 and key not in self._graph)
            parts={key:tuple(part for part in self._graph[key] if part not in zeros) for key in self._order}
            # The constants are held so their id stays theirs.
            plan=self._columns_cache[token]=(zeros,parts,constant_values);self._own(token,id(constant_values))
        return plan[0],plan[1]

    def _finite_witnesses(self,only,values,constants=_NO_CONSTANTS):
        """The reactions whose finiteness implies that of all *values*.

        A sum is not finite when any of its parts is not (a NaN propagates;
        an infinity stays one or, against an opposite one, becomes NaN), so
        the sums just formed and the reactions in none of them are enough;
        and a declared constant is a finite float by construction.
        """
        token=('finite',only)
        keys=self._columns_cache.get(token)
        if keys is None:
            sums=[key for key in self._order if key in values]
            covered=set()
            for key in sums:covered|=self._wanted((key,))
            keys=tuple(key for key in values if key in sums or (key not in covered and key not in constants))
            self._columns_cache[token]=keys
            if isinstance(only,tuple) and len(only)==2:self._own(token,only[1])
        return keys

    def _answer(self,requests):
        """Values for many ``(segment, (energies, mts), constants)`` requests at once.

        The resonance physics of every request on one region is one batch,
        in request order and blocks of ``block_size``; the rest is
        :meth:`_evaluate_segment`, request by request.
        """
        physics=[None]*len(requests);regions={};adjusted=[];curves={}
        for n,(s,(points,only),constants) in enumerate(requests):
            e=np.asarray(points,dtype=float).copy()
            if s.left_high:e[e==s.high]=np.nextafter(s.high,s.low)
            adjusted.append(e)
            if s.region is not None and only!='diagnostic':regions.setdefault(id(s.region),(s.region,[]))[1].append(n)
            for key,curve in self._plain_curves(s,constants,only):
                curves.setdefault(id(curve),(curve,[]))[1].append((n,key))
        # A background curve is read elementwise and one curve spans many
        # segments: read it once over every request that needs it.
        backgrounds=[{} for _ in requests]
        for curve,members in curves.values():
            values=curve.evaluate(np.concatenate([adjusted[n] for n,_ in members]));end=0
            for n,key in members:
                backgrounds[n][key]=values[end:end+len(adjusted[n])];end+=len(adjusted[n])
        context=self._prepared.context;size=self.options.block_size
        for region,members in regions.values():
            self._check_competitive_owners(region)
            energies=[adjusted[n] for n in members]
            flat=np.concatenate(energies);blocks=[]
            for start in range(0,len(flat),size):
                e=flat[start:start+size]
                physical=evaluate_region(e,region,context,work_bytes=self.options.max_work_bytes,
                    absorption_rtol=min(1e-8,self.options.rtol*1e-5))
                blocks.append(_subtract_competitive(physical,region,context,e))
            keys=list(blocks[0])
            if any(list(b)!=keys for b in blocks):raise ValueError('resonance blocks disagree on their partials')
            joined={mt:np.concatenate([b[mt] for b in blocks]) for mt in keys}
            end=0
            for n,e in zip(members,energies):
                physics[n]={mt:v[end:end+len(e)] for mt,v in joined.items()};end+=len(e)
        return [self._evaluate_segment(s,e,False,constant_values=self._source_constants(c),
                    only=self._diagnostic_keys(s,c),backgrounds=b) if only=='diagnostic' else
                self._evaluate_segment(s,e,constant_values=c,physics=p,only=only,backgrounds=b)
                for (s,(e,only),c),p,b in zip(requests,physics,backgrounds)]

    def _source_constants(self,constants):
        """A segment's constant leaves: as the source states them, exactly
        (constant_on certifies the value); sums keep their stated forms."""
        token=('source',id(constants))
        plan=self._columns_cache.get(token)
        if plan is None:
            plan=self._columns_cache[token]=(MappingProxyType({k:v for k,v in constants.items() if k not in self._graph}),constants)
            self._own(token,id(constants))
        return plan[0]

    def _diagnostic_keys(self,s,constants):
        """What a source diagnostic reads: every reaction but the leaves that
        are the constant zero on the segment."""
        token=('diagnostic',id(constants))
        plan=self._columns_cache.get(token)
        if plan is None:
            plan=self._columns_cache[token]=(tuple(k for k in s.curves if k in self._graph or not (k in constants and constants[k]==0)),constants)
            self._own(token,id(constants))
        return plan[0]

    def _plain_curves(self,s,constants,only):
        """The ``(key, curve)`` that :meth:`_evaluate_segment` may read one by
        one for this request: not constant and not summed."""
        token=('plain',id(s),id(constants),only)
        plan=self._columns_cache.get(token)
        if plan is None:
            # A diagnostic reads the source forms as stated, sums included,
            # but for the constant leaves (see _source_constants).
            source=only=='diagnostic'
            if source:constants=self._source_constants(constants)
            wanted=None if only is None or source else self._wanted(only)
            # Keys a shared reader serves are read here too: the reader takes
            # precedence, and declines a query that crosses a log-law panel,
            # when the curve is read alone.
            plain=tuple((key,curve) for key,curve in s.curves.items()
                        if (wanted is None or key in wanted) and key not in constants
                        and curve is not None and (source or key not in self._graph))
            # The segment and constants are held so their ids stay theirs.
            plan=self._columns_cache[token]=(s,constants,plain);self._own(token,id(s))
        return plan[2]

    def _check_competitive_owners(self,region):
        for g in region.groups:
            if not g.competitive_in_background:continue
            owner=self._mt_keys.get(g.competitive_mt)
            if owner is None or owner in self._graph:
                raise UnsupportedResonanceError("competitive background requires an exclusive model reaction")

    def _precompute_witnesses(self,witnesses,grids,verification_batch,fractions,workers,progress):
        """Fill the exact physical witnesses on worker threads (see ``parallel``).

        Returns False, leaving the serial path to compute them, when the work
        is too small to be worth it. The blocks are the serial path's blocks
        and the numbers are bit-identical.
        """
        from . import parallel
        items=[];work=0.
        for i,(segment,grid) in enumerate(zip(self._segments,grids)):
            if segment.region is None:continue
            self._check_competitive_owners(segment.region)
            levels=sum(len(getattr(g,'levels',()) or ()) for g in segment.region.groups) or 1
            for batch,points in enumerate(verification_batches(grid,verification_batch,fractions)):
                if (i,batch) in witnesses[1]:continue
                e=np.asarray(points,dtype=float).copy()
                if segment.left_high:e[e==segment.high]=np.nextafter(segment.high,segment.low)
                size=self.options.block_size
                for start in range(0,len(e),size):
                    items.append((i,batch,start,e[start:start+size]))
                work+=len(e)*levels
        if not items or work<parallel.PARALLEL_WORK_THRESHOLD:return False
        freeze=lambda a:np.frombuffer(np.ascontiguousarray(a,dtype=float).tobytes(),dtype=float)
        context=self._prepared.context;work_bytes=self.options.max_work_bytes
        def compute(item):
            i,_,_,e=item
            return _exact_physical_block(self._segments[i].region,context,e,work_bytes)
        def store(item,physical):
            i,batch,start,e=item
            witnesses[1].setdefault((i,batch),{})[start]=(freeze(e),{mt:freeze(v) for mt,v in physical.items() if mt!=1})
        try:
            parallel.compute_blocks(items,compute,workers,store,progress)
        except BaseException:
            # Never leave a partly filled batch behind: the serial path would
            # treat its missing blocks as computed.
            for i,batch,_,_ in items:witnesses[1].pop((i,batch),None)
            raise
        return True

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

    def verify_suite(self,suite,*,label=None,progress=None,workers=None):
        """Verify a complete model reloaded from GNDS or a processed ENDF tape.

        ``progress(fraction)`` is called after each batch with the share of
        checked energies, a plain float in (0, 1]. ``workers`` threads share
        the exact physics of large materials (default
        ``parallel.default_workers()``, at most four, low priority); 1 keeps
        everything on this thread. The numbers are bit-identical either way.
        """
        from kika.nuclear_data.model import Regions1d
        from kika.nuclear_data.model.functions.regions1d import curvesOnDomains
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
            # A TAB1 may state the segments as one region per law (NJOY reads
            # at most 20); cut it back at the segment edges, or fail as before.
            curves=curvesOnDomains(form,[(s.low,s.high) for s in self._segments])
            if curves is None:raise ReconstructionConvergenceError('serialized region structure changed')
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
            self._verification_cache['last']=(dict(cached[1]),dict(cached[2]))
            return dict(cached[1])
        total=sum(len(g)+4*max(0,len(g)-1) for g in grids);done=0
        fractions=np.array([.1732050807568877,.3819660112501051,.6180339887498949,.8267949192431123])
        verification_batch=max(1,min(32768,self.options.max_work_bytes//(64*(len(checked)+1))))
        if workers is None:
            from .parallel import default_workers
            workers=default_workers()
        # Parallel physics first (0-90 % of the bar), then the comparison.
        share=1.
        if workers>1 and self._precompute_witnesses(witnesses,grids,verification_batch,fractions,workers,
                (lambda f:progress(float(.9*f))) if progress is not None else None):
            share=.1
        report=(lambda f:progress(float(1-share+share*f))) if progress is not None else None
        for i,(segment,grid) in enumerate(zip(self._segments,grids)):
            evaluators={}
            for key,curves in checked.items():
                curve=curves[i]
                if curve.domainMin!=segment.low or curve.domainMax!=segment.high:raise ReconstructionConvergenceError('serialization moved a domain boundary')
                if curve.domainUnit!='eV' or curve.rangeUnit!='b' or curve.endfInterpolationCode!=2:
                    raise ReconstructionConvergenceError('serialized units or interpolation changed')
                try:evaluators[key]=prepare_evaluator(curve.xs,curve.ys,2)
                except ValueError as exc:raise ReconstructionConvergenceError('invalid serialized table') from exc
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
                done+=len(points)
                if report is not None and total:report(float(min(1.,done/total)))
        if any(not np.isfinite(v) or v>1 for v in maxima.values()):raise ReconstructionConvergenceError(f'reloaded suite exceeds total budget: {maxima}; worst (eV, reference b, serialized b): {worst}')
        self._verification_cache['entry']=(signature,dict(maxima),dict(worst))
        self._verification_cache['last']=(dict(maxima),dict(worst))
        return maxima

    @property
    def last_verification(self):
        """``(maxima, worst)`` of the last :meth:`verify_suite` that passed, or None.

        ``maxima[key]`` is the largest error over the budget ``atol+rtol*|sigma|``;
        ``worst[key]`` is ``(energy eV, reference b, serialized b)`` at that point.
        """
        last=self._verification_cache.get('last')
        return None if last is None else (dict(last[0]),dict(last[1]))


def _subtract_competitive(physical,region,context,e):
    for g in region.groups:
        if g.competitive_in_background:
            physical[g.competitive_mt]-=evaluate_bw(e,(g,),region.approximation,context)[g.competitive_mt]
    return physical


def _exact_physical_block(region,context,e,work_bytes):
    """Exact physical partials of one verification block (serial and workers)."""
    physical=evaluate_region(e,region,context,work_bytes=work_bytes,absorption_rtol=0.,native=True)
    return _subtract_competitive(physical,region,context,e)


#: Identifies the reconstruction physics, not the commit. Change it when a
#: kernel, grid or publication change alters any reconstructed value, so
#: caches keyed on it are invalidated exactly then.
#: 2: thresholds below the kinematic one are raised as RECONR does (D37).
#: 3: an RML width of a bound level is referred to its channel energy, not to
#: |E_lambda| (Ca-40 thermal (n,alpha) x6.4); charged-channel functions and
#: groups of up to six channels in C; geometric seeds by the decade.
ENGINE_VERSION='kika-native-suite-3'


def _target_awr(suite,context):
    """Target/neutron mass ratio for kinematic thresholds: PoPs's (MF1's AWR), else the context's."""
    try:
        target=suite.PoPs[suite.target];neutron=suite.PoPs[suite.projectile]
        if target.mass is not None and neutron.mass is not None:
            return target.mass.convertedTo('amu').value/neutron.mass.convertedTo('amu').value
    except KeyError:pass
    return None if context is None else context.atomic_weight_ratio


def _raise_thresholds(curves,entries,graph,awr):
    """RECONR's threshold rule (``reconr.f90`` lunion), applied to exclusive leaves.

    A reaction with ``thrx = -Q (A+1)/A > 0`` whose table starts below
    ``sigfig(thrx, 7, +1)`` has its first energy raised to that value, and any
    following energies the move overtakes are pushed up one place in the
    seventh figure each. Evaluations round the threshold to seven figures and
    sometimes round it down, which leaves the reaction open a few ppm below
    the kinematic limit; ACER flags that as a threshold error. Values are kept,
    only abscissae move. Sums are rebuilt from their leaves, so they follow.
    The moved energies are the seven-figure numbers RECONR writes, without
    ``sigfig``'s 1e-13 bias, so a written tape reloads onto the same segment
    boundaries. Returns ``{key: (old, new)}`` for the report.
    """
    from .probability_tables import sigfig
    def figures(x,shade):return float(f'{sigfig(x,7,shade):.7g}')
    changed={}
    for key,cs in curves.items():
        if key in graph:continue
        channel=entries[key].outputChannel
        q=None if channel is None else channel.Q
        if q is None or q.value is None or q.value>=0:continue
        if q.unit!='eV':raise UnsupportedResonanceError(f'reaction Q of {key.label} is not in eV')
        if awr is None:raise ValueError('a threshold reaction needs the target mass (PoPs or NeutronContext)')
        threshold=figures(-q.value*(awr+1)/awr,+1)
        first=cs[0];x=list(first.x)
        if x[0]>=threshold:continue
        old=x[0];x[0]=threshold;i=0
        while i+1<len(x) and x[i+1]<=x[i]:
            if i>=10:raise ValueError(f'ill-behaved threshold for {key.label}')
            x[i+1]=figures(x[i],+1);i+=1
        curves[key]=(BackgroundCurve(tuple(x),first.y,first.law),)+tuple(cs[1:])
        changed[key]=(float(old),float(threshold))
    return changed


def _resonance_mass(resonances,target_id):
    """The target mass a formalism's own PoPs states, first region first, else ``None``.

    FUDGE writes MF2's AWRI there and MF1's AWR on the top-level target; the
    resonance parameters were fitted with AWRI, which is also what the ENDF
    route (``context_from_endf``) reconstructs with. So it wins over PoPs's.
    """
    sources=[r.formalism for r in resonances.resolved or []]
    if resonances.unresolved is not None:sources.append(resonances.unresolved.tabulatedWidths)
    for source in sources:
        local=getattr(source,'PoPs',None)
        if local is None:continue
        particles=local.particles
        particle=particles.get(target_id) or (next(iter(particles.values())) if len(particles)==1 else None)
        if particle is not None and particle.mass is not None:return particle.mass
    return None


def reconstruct_suite(suite,context=None,*,source_style='eval',label='recon',options=None,progress=None):
    """Reconstruct a complete supported resonance model suite without mutating it.

    Local References and native summand links are resolved. All sums must have
    a graph. Evaluated aggregate discrepancies are recorded; output aggregates
    are derived from their exclusive leaves and must close after publication.
    Existing partials are preserved outside supported resonance domains; output aggregates are
    rebuilt once from exclusive leaves. No partial success or automatic fallback.
    ``progress(fraction)``, if given, is called after each energy segment with
    the completed fraction of the material domain in lethargy (segments are
    uneven in cost, so it is an indication, not an estimate of time left).
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
        mass=_resonance_mass(suite.resonances,suite.target) or target.mass
        context=NeutronContext(mass.convertedTo('amu').value/neutron.mass.convertedTo('amu').value,
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
    raised=_raise_thresholds(curves,entries,graph,_target_awr(suite,context))
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
    edges=sorted(cuts);segments=[];shared_cache={}
    # The pieces of a curve are sorted and do not overlap, so their starts and
    # ends are both non-decreasing: the pieces with start <= a and end >= b
    # are the run [bisect_left(ends, b), bisect_right(starts, a)), found by
    # bisection rather than by reading every piece for every segment.
    import bisect
    spans_of={key:([c.x[0] for c in cs],[c.x[-1] for c in cs]) for key,cs in curves.items()}
    for a,b in zip(edges[:-1],edges[1:]):
        region=next((r for r in prepared.regions if r.low<=a and r.high>=b),None)
        local={}
        for key,cs in curves.items():
            starts,ends=spans_of[key]
            candidates=cs[bisect.bisect_left(ends,b):bisect.bisect_right(starts,a)]
            if len(candidates)>1:raise ValueError('ambiguous source region ownership')
            if not candidates and not (b<=cs[0].x[0] or a>=cs[-1].x[-1]):
                raise UnsupportedResonanceError(f'source function has an uncovered internal gap: {key}',category='internal-gap')
            local[key]=candidates[0] if candidates else None
        segments.append(_SuiteSegment(a,b,MappingProxyType(local),region,b<high,shared_cache))
    result=SuiteReconstructionResult(MappingProxyType({}),MappingProxyType({}),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash,MappingProxyType(physical_owners))
    tables=[None]*len(segments);checks=[None]*len(segments);points=0
    source_balance={key:0. for key in graph};source_worst={}
    span=np.log(high/low) if low>0 else None
    # Segments are tabulated together, a window at a time, so that the
    # resonance physics of all their refinement passes is evaluated in a few
    # large batches (``_answer``) rather than one small one per segment and
    # pass: JENDL-5 Cl-35 cuts its RRR into 3 376 segments. The window holds
    # at most ``max_work_bytes`` of starting panels, by linearize's own
    # estimate. Segments start in order, so the source diagnostics are those
    # of a serial run; each refines under the budget left by the finished
    # ones, and the total is checked in order at the end.
    pending={}
    # Constancy and linearity of every source curve on every segment that
    # reads it, decided per curve for all its segments at once (the same
    # answers constant_on and _laws_over give one segment at a time).
    span_high=[np.nextafter(s.high,s.low) if s.left_high else s.high for s in segments]
    readers={}
    for i,s in enumerate(segments):
        for key,curve in s.curves.items():
            if curve is not None and key not in graph:readers.setdefault(id(curve),(curve,[]))[1].append(i)
    constant_at={};linear_at={}
    for curve,where in readers.values():
        where=np.asarray(where);lows=np.array([segments[i].low for i in where]);highs=np.array([span_high[i] for i in where])
        values=np.full(len(where),np.nan);wide=highs>lows
        if np.any(wide):values[wide]=curve.constant_on_many(lows[wide],highs[wide])
        linear=curve.linear_over_many(lows,highs)
        for i,v,l in zip(where.tolist(),values.tolist(),linear.tolist()):
            constant_at[(id(curve),i)]=v;linear_at[(id(curve),i)]=l
    def prepare(position):
        """Seeds, constants, deferred columns and source-diagnostic grid of
        one segment; none of it depends on another segment."""
        s=segments[position]
        seed_curves={i:(c,) for i,(key,c) in enumerate(s.curves.items()) if c is not None and key not in graph}
        if s.region is not None:
            seeds=_seeds(_Segment(s.low,s.high,s.region,seed_curves,s.left_high),context)
        else:
            # Original knots suffice for linear fast-region data. Inserting
            # unnecessary near-threshold points amplifies ENDF x-rounding.
            seeds=sorted({s.low,s.high}|{float(x) for c in seed_curves.values() for x in _knots_in(c[0],s.low,s.high)})
            for cs in seed_curves.values():
                c=cs[0]
                if c.x[-1]==s.high and c.endpoint_jump:
                    seeds.append(float(np.nextafter(s.high,s.low)))
            seeds=sorted(set(seeds))
        affected=({physical_owners.get(mt) for mt in region_mts(s.region)}
                  if s.region is not None else set())
        constants={}
        high=np.nextafter(s.high,s.low) if s.left_high else s.high
        for key,curve in s.curves.items():
            if key in graph or key in affected:continue
            if high<=s.low:value=None
            elif curve is None:value=0.
            else:
                value=constant_at[(id(curve),position)]
                if value!=value:value=None
            if value is not None:constants[key]=value
        for key in order:
            if all(part in constants for part in graph[key]):
                constants[key]=sum((constants[part] for part in graph[key]),0.)
        deferred={}
        for key,curve in s.curves.items():
            if key in graph or key in affected or key in constants or curve is None:continue
            linear=linear_at[(id(curve),position)]
            # Source knots are included in seeds, so these untouched columns
            # are affine inside each seed panel. Still check all seven final
            # probes; a failure falls back to the full adaptive column set.
            if linear:
                def read(e,curve=curve,s=s):
                    e=np.asarray(e,dtype=float).copy()
                    if s.left_high:e[e==s.high]=np.nextafter(s.high,s.low)
                    return curve.evaluate(e)
                deferred[key]=read
        work=min(len(seeds),max(1,options.max_work_bytes//1024))*(len(s.curves)-len(constants)-len(deferred)+1)*1024
        pending[position]=(constants,work)
        # Diagnose the evaluated aggregate against exclusive leaves. Output
        # sums are derived even if the redundant source aggregate disagrees.
        # Redundant aggregates inform diagnostics, not the reconstructed mesh.
        # Keep their original knots in this separate source-comparison grid.
        diagnostic_seeds=set(seeds)|{float(x) for c in s.curves.values() if c is not None for x in _knots_in(c,s.low,s.high)}
        grid=np.asarray(sorted(diagnostic_seeds))
        probe=np.r_[grid,(grid[:-1]+grid[1:])*.5]
        return seeds,constants,deferred,probe
    def diagnose(position,probe,original):
        s=segments[position];constants=pending[position][0]
        for key in order:
            # A part that is the constant zero here adds nothing, exactly (a
            # sum starts at +0.0), and the diagnostic request leaves it out.
            parts=[part for part in graph[key] if part in graph or not (part in constants and constants[part]==0)]
            summed=sum((original[part] for part in parts),np.zeros(len(probe)))
            ratio=error_ratio(original[key],summed,options)
            maximum=float(np.max(ratio))
            if maximum>source_balance[key]:
                index=int(np.argmax(ratio))
                source_worst[key]=dict(energy_eV=float(probe[index]),side='left-limit' if s.left_high and probe[index]==s.high else 'point',stated_b=float(original[key][index]),components_b=float(summed[index]),error_ratio=maximum)
            source_balance[key]=max(source_balance[key],maximum)
            original[key]=summed
    def tag(steps,position):
        """One segment's linearize_steps requests, marked with the segment."""
        try:
            request=next(steps)
            while True:
                request=steps.send((yield (*request,position)))
        except StopIteration as stop:
            return stop.value
    keys=tuple(segments[0].curves) if segments else ()
    def run(positions,items,budget):
        """Tabulate consecutive segments: their source diagnostics in one
        request, answered in segment order, then one refinement for all of
        them (linearize_group_steps) or, for one segment, its own."""
        probes=[item[3] for item in items]
        values=yield (np.concatenate(probes),'diagnostic',np.repeat(positions,[len(v) for v in probes]))
        end=0
        for position,probe in zip(positions,probes):
            diagnose(position,probe,{key:v[end:end+len(probe)] for key,v in values.items()});end+=len(probe)
        if len(positions)==1:
            seeds,constants,deferred,_=items[0]
            return [(yield from tag(linearize_steps(seeds,options,budget,constants=constants,deferred=deferred,keys=keys),positions[0]))],None
        spans=[item[:3] for item in items]
        steps=linearize_group_steps(spans,options,budget,keys);owners=np.asarray(positions)
        try:
            request=next(steps)
            while True:
                e,only,local=request
                request=steps.send((yield (e,only,owners[local])))
        except StopIteration as stop:
            results,group=stop.value
        if group is not None:
            # One check for the group; each segment keeps its own tables.
            return results,group
        for k,item in enumerate(results):
            if item is None:
                # A deferred column failed a probe: this segment alone, with
                # the ordinary retry of every column.
                seeds,constants,deferred=spans[k]
                results[k]=yield from tag(linearize_steps(seeds,options,budget,constants=constants,deferred=deferred,keys=keys),positions[k])
        return results,None
    # Consecutive segments of one region, each of a few seeds, are refined
    # together (JENDL-5 Cl-35: 3 376 segments, a median of two seeds each);
    # a group stays within the seed count linearize gives one chunk.
    GROUP_SEEDS=2048
    live={};started=0;finished=0.;done=0;working=0
    def finish(index,value):
        nonlocal finished,done
        x,y,*check=value;s=segments[index]
        tables[index]=(x,y)
        if check:checks[index]=dict(domain=(s.low,s.high),points=len(x),**check[0])
        done+=len(x)
        if index in pending:result._forget(id(segments[index]),id(pending[index][0]))
        pending.pop(index,None)
        if progress is not None:
            finished=max(finished,float(min(1.,np.log(s.high/low)/span)) if span else
                         sum(t is not None for t in tables)/len(segments))
            progress(finished)
    def complete(positions,results,group):
        for i,value in zip(positions,results):finish(i,value)
        if group is not None:
            # A group's segments report as one region, as the chunks of one
            # segment do.
            first,last=segments[positions[0]],segments[positions[-1]]
            checks[positions[0]]=dict(domain=(first.low,last.high),points=sum(len(tables[i][0]) for i in positions),**group)
    def answer(requests):
        """Values for requests ``(energies, only, segment)``, *segment* one
        index or one per energy: each split into one request per segment."""
        flat=[];shapes=[]
        for e,only,owner in requests:
            if np.ndim(owner)==0:
                shapes.append(None);flat.append((segments[int(owner)],(e,only),pending[int(owner)][0]));continue
            ordered=np.argsort(owner,kind='stable')
            parts=np.split(ordered,np.flatnonzero(np.diff(owner[ordered]))+1);shapes.append((len(e),parts))
            for idx in parts:
                i=int(owner[idx[0]]);flat.append((segments[i],(e[idx],only),pending[i][0]))
        values=result._answer(flat);out=[];at=0
        for (e,only,owner),shape in zip(requests,shapes):
            if shape is None:
                out.append(values[at]);at+=1;continue
            n,parts=shape;pieces=values[at:at+len(parts)];at+=len(parts)
            if isinstance(only,tuple):keys=only
            else:
                # Each segment's diagnostic leaves out its own zero leaves:
                # the union, zero where a segment has none (it never reads it).
                keys=tuple(dict.fromkeys(key for piece in pieces for key in piece))
            joined={}
            for key in keys:
                column=np.zeros(n)
                for idx,piece in zip(parts,pieces):
                    if key in piece:column[idx]=piece[key]
                joined[key]=column
            out.append(joined)
        return out
    while started<len(segments) or live:
        while started<len(segments) and (not live or working<options.max_work_bytes):
            positions=[started];items=[prepare(started)];started+=1
            region=segments[positions[0]].region
            total=len(items[0][0])
            while (region is not None and started<len(segments) and segments[started].region is region
                   and total<GROUP_SEEDS):
                item=prepare(started)
                if total+len(item[0])>GROUP_SEEDS:
                    # Too many seeds for this group: it starts the next one,
                    # already prepared.
                    break
                positions.append(started);items.append(item);total+=len(item[0]);started+=1
            work=sum(pending[i][1] for i in positions)
            generator=run(positions,items,options.max_points-done)
            try:request=next(generator)
            except StopIteration as stop:
                complete(positions,*stop.value)
            else:
                live[positions[0]]=[generator,request,positions,work];working+=work
        order_live=sorted(live)
        answers=answer([live[i][1] for i in order_live])
        for i,values in zip(order_live,answers):
            try:live[i][1]=live[i][0].send(values)
            except StopIteration as stop:
                complete(live[i][2],*stop.value)
                working-=live[i][3];del live[i]
    # The window's request plans hold every segment's constants; the tables
    # are all that the output needs.
    result._columns_cache.clear();pending.clear()
    for x,_ in tables:
        points+=len(x)
        if points>options.max_points:
            raise ReconstructionConvergenceError('refinement exceeds max_points',category='budget-exhausted')
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')]);output={};minima={}
    from kika.algebra import compress_flat_pieces
    owned=[{physical_owners.get(mt) for mt in region_mts(s.region)} if s.region is not None else set()
           for s in segments]
    lows=np.array([x[0] for x,_ in tables]);highs=np.array([x[-1] for x,_ in tables])
    for key in entries:
        tabs=[y[key] for _,y in tables]
        if key not in graph and all(not t.flags.writeable and len(t)>=2 for t in tabs):
            values=np.array([t[0] for t in tabs])
            if not np.any((values==0)&np.signbit(values)):
                # The constant of every segment (read-only tables, see below):
                # compression leaves each segment its two ends, and segments
                # with one value join through their shared edge. Written
                # directly, as _compress_and_join would make it.
                cut=np.flatnonzero(values[1:]!=values[:-1])+1
                children=[]
                for a,b in zip(np.r_[0,cut],np.r_[cut,len(values)]):
                    cx=np.r_[lows[a],highs[a:b]];children.append(XYs1d(cx,np.full(len(cx),values[a]),axes=axes,index=len(children)))
                output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
                output[key].label=label
                minima[key]=float(np.min(values))
                for _,y in tables:del y[key]
                continue
        pieces=[];summed=key in graph
        for i,(x,y) in enumerate(tables):
            if summed or not y[key].flags.writeable:
                pieces.append((x,y[key]));continue
            s=segments[i];curve=s.curves[key]
            if curve is not None and key not in owned[i]:
                # (A read-only table is a constant of the segment, shared by
                # _linearize_chunk; on its own knots the curve is that same
                # constant, and compression leaves the segment's two ends.)
                original_x=curve._x
                if np.all(_laws_over(curve,s.low,s.high)==2):
                    # An untouched lin-lin background already has an exact
                    # representation on its own knots. It need not inherit
                    # every other reaction's resonance or fast-region knot.
                    x=np.unique(np.r_[s.low,original_x[np.searchsorted(original_x,s.low,'right'):np.searchsorted(original_x,s.high,'left')],s.high])
                    if not s.left_high and original_x[-1]==s.high and curve.endpoint_jump:
                        x=np.unique(np.r_[x,np.nextafter(s.high,s.low)])
                    query=x.copy()
                    if s.left_high:query[query==s.high]=np.nextafter(s.high,s.low)
                    values=curve.evaluate(query)
                else:values=y[key]
            else:values=y[key]
            pieces.append((x,values))
        children=[XYs1d(cx,cy,axes=axes,index=i) for i,(cx,cy) in enumerate(_compress_and_join(pieces))]
        output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
        output[key].label=label
        # A reaction's tables are released once its output exists, so the
        # output and the tables are never all held at once.
        minima[key]=float(np.min(np.concatenate([y[key] for _,y in tables])))
        for _,y in tables:del y[key]
    normalized_hash=hashlib.sha256((source_hash+repr(context)+repr(options)).encode()).hexdigest()
    report=dict(engine=ENGINE_VERSION,scope='all modeled cross sections across evaluated domain',
        source_sha256=source_hash,normalized_sha256=normalized_hash,minima=minima,
        projectile=str(suite.projectile),target=str(suite.target),projectileFrame=str(suite.projectileFrame),
        domain=(low,high),points=points,regions=[c for c in checks if c is not None],source_sum_error_ratios=source_balance,
        stored_points=sum(len(c.xs) for f in output.values() for c in (f.function1ds if isinstance(f,Regions1d) else [f])),
        source_sum_policy='derive-from-leaves',source_sum_discrepancies=source_worst,
        raised_thresholds=MappingProxyType({key.label:v for key,v in raised.items()}),
        source_conversion_report=deepcopy(suite.report),
        resonance_partial_owners=MappingProxyType(physical_owners),
        source_style=source_style,source_style_labels=tuple(suite.styles.labels),
        label=label,options=options,context=context,preparation_notes=prepared.preparation_notes,
        empirical_verification=True,global_error_bound=False)
    return SuiteReconstructionResult(MappingProxyType(output),MappingProxyType(report),source_style,label,options,prepared,
        tuple(segments),MappingProxyType(graph),order,MappingProxyType(mt_keys),source_hash,MappingProxyType(physical_owners),
        tuple(np.frombuffer(x.tobytes(),dtype=x.dtype) for x,_ in tables))


def attach_reconstruction(suite,result,*,progress=None,workers=None):
    """Validate and attach a complete result atomically, preserving evaluated forms.

    ``progress(fraction)`` follows the verification, as in ``verify_suite``.
    """
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
    from datetime import date as calendar_date
    candidate.styles.add(CrossSectionReconstructed(result.label,derivedFrom=result.source_style,date=calendar_date.today().isoformat()))
    result.verify_suite(candidate,progress=progress,workers=workers)
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

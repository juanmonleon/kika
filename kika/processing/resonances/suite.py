"""Complete model-suite resonance assembly, local links and atomic style attachment.

All format imports are deferred to callers. Source data are copied, hash-guarded
and never edited by reconstruction. The supported material is a neutron/lab
suite whose resonance ranges are supported and whose additive sums close.
"""
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from enum import Enum
from collections.abc import Mapping
import functools
import hashlib
import operator
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
    # A radionuclide production (MF8-10) shares its reaction's MT, so it cannot
    # be projected by MT; it is published on its own (kika.processing.productions).
    from kika.nuclear_data.model import isRadionuclideProduction
    entries={}
    for container in CONTAINERS:
        for reaction in getattr(suite,container):
            if container=='productions' and isRadionuclideProduction(reaction):continue
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


@functools.lru_cache(maxsize=65536)
def _semantic(path):
    """A link path as its ``(element, label)`` steps: a pure function of the
    string, and every part of every sum reads every reaction's path."""
    out=[]
    for step in _split_path(path):
        match=re.fullmatch(r"([^\[]+)(?:\[@label=([\"'])(.*?)\2\])?",step)
        if match is None:raise UnsupportedResonanceError('only local attribute-label links are implemented')
        out.append((match[1],match[3]))
    return tuple(out)


def _target(href,base,entries,style):
    path=_normalize_link(href,base)
    steps=_semantic(path)
    for key in entries:
        prefix=_semantic(_path(key))
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


#: Marks a request for energies of several segments (see ``_answer``).
_GROUP=object()

#: What a mapping lookup returns for a missing key.
_ABSENT=object()


class _Curves(Mapping):
    """A segment's curve per reaction, read off the material's table of
    curve indices (*column*, -1 for none) instead of held as a dict."""
    __slots__=('_keys','_rows','_choices','_column')

    def __init__(self,keys,rows,choices,column):
        self._keys=keys;self._rows=rows;self._choices=choices;self._column=column

    def __getitem__(self,key):
        row=self._rows[key];return self._choices[row][self._column[row]]

    def __contains__(self,key):
        return key in self._rows

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def items(self):
        return zip(self._keys,map(operator.getitem,self._choices,self._column.tolist()))

    def values(self):
        return map(operator.getitem,self._choices,self._column.tolist())


class _Constants(dict):
    """A segment's constants, with their reactions' positions in *order*
    (``columns``) and their values (``column_values``) as arrays, for a caller that
    lays out many segments' constants at once."""
    __slots__=('order','columns','column_values')


class _DeferredRead:
    """A deferred column's reader on one segment: its curve, at the left
    neighbour of the segment's high where the segment is closed on the
    left there (*left_high* is that ``(high, low)``, or None). The curve
    and the edge are what a caller reading many segments at once needs."""
    __slots__=('curve','left_high')

    def __init__(self,curve,left_high):
        self.curve=curve;self.left_high=left_high

    def __call__(self,e):
        e=np.asarray(e,dtype=float).copy()
        if self.left_high is not None:
            high,low=self.left_high;e[e==high]=np.nextafter(high,low)
        return self.curve.evaluate(e)

#: The constants of a request that declares none: one object, so that the
#: plans keyed on it are made once.
_NO_CONSTANTS=MappingProxyType({})


def _compress_and_join_flat(x,y,lengths):
    """``_join_continuous(compress_flat_pieces(pieces))`` in one vectorised
    pass, for the pieces of *lengths* laid end to end in *x*, *y*.

    The arithmetic is theirs: each piece keeps the nodes compress_flat keeps,
    and a piece that starts on the last node of the one before (abscissa,
    value and sign) continues it without that node. Only the per-piece
    Python work goes, and no piece objects are made: JENDL-5 Cl-35 has 3 376
    pieces for each of 234 reactions.
    """
    from kika.algebra import compress_flat_pieces
    lengths=np.asarray(lengths)
    ends=np.cumsum(lengths);starts=ends-lengths
    def pieces():
        return list(zip(np.split(x,ends[:-1]),np.split(y,ends[:-1])))
    if len(lengths)<2 or np.any(lengths<2):
        return _join_continuous(compress_flat_pieces(pieces()))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return _join_continuous(compress_flat_pieces(pieces()))
    inside=np.ones(len(x)-1,dtype=bool);inside[ends[:-1]-1]=False
    if np.any(np.diff(x)[inside]<0):
        return _join_continuous(compress_flat_pieces(pieces()))
    negative=(y==0)&np.signbit(y)
    keep=np.ones(len(x),dtype=bool)
    keep[1:-1]=~((y[:-2]==y[1:-1])&(y[1:-1]==y[2:])&(x[:-2]<x[1:-1])&(x[1:-1]<x[2:])
                 &~negative[:-2]&~negative[1:-1]&~negative[2:])
    keep[starts]=True;keep[ends-1]=True
    with np.errstate(over='ignore'):
        kept=x[keep]
        gaps=~np.isfinite(np.diff(kept))
    if np.any(gaps):
        return _join_continuous(compress_flat_pieces(pieces()))
    last,first=ends[:-1]-1,starts[1:]
    joined=(x[last]==x[first])&(y[last]==y[first])&(np.signbit(y[last])==np.signbit(y[first]))
    keep[first[joined]]=False
    region=np.cumsum(np.r_[0,~joined])
    owner=np.repeat(region,lengths)[keep]
    xs,ys=x[keep],y[keep]
    cut=np.flatnonzero(np.diff(owner))+1
    bounds=np.r_[0,cut,len(xs)].tolist()
    return [(xs[a:b],ys[a:b]) for a,b in zip(bounds[:-1],bounds[1:])]


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


def _own_knots(curve,segments,positions):
    """``(positions, lengths, x, values)``: each segment's table on the
    curve's own knots (the segment's two ends and the knots strictly
    inside), for the segments over which the curve is lin-lin, laid end to
    end in segment order. What the per-segment
    ``np.unique(np.r_[low, inside, high])`` and one read each give, laid out
    by index arithmetic and read once.
    """
    x=curve._x
    lows=np.array([s.low for s in segments]);highs=np.array([s.high for s in segments])
    linear=curve.linear_over_many(lows,highs)
    keep=np.flatnonzero(linear)
    positions=np.asarray(positions)
    if not len(keep):return positions[:0],np.zeros(0,dtype=np.int64),np.zeros(0),np.zeros(0)
    lows,highs=lows[keep],highs[keep]
    left=np.array([segments[k].left_high for k in keep])
    a=np.searchsorted(x,lows,'right');b=np.searchsorted(x,highs,'left')
    inside=np.maximum(b-a,0)
    # Knots strictly inside are increasing and distinct from the ends, so
    # np.unique is the sorted concatenation itself; a curve ending at a
    # segment's high with a jump there adds that high's left neighbour.
    jump=(~left)&(x[-1]==highs)&curve.endpoint_jump
    last_inside=np.where(inside>0,x[np.maximum(b-1,0)],-np.inf)
    if np.any(jump&(last_inside==np.nextafter(highs,lows))):
        # The added left neighbour is a knot already: np.unique would merge it.
        pieces=[_own_knots_one(curve,segments[k]) for k in keep.tolist()]
        return (positions[keep],np.array([len(p[0]) for p in pieces],dtype=np.int64),
                np.concatenate([p[0] for p in pieces]),np.concatenate([np.asarray(p[1],dtype=float) for p in pieces]))
    count=inside+2+jump
    ends=np.cumsum(count);starts=ends-count
    nodes=np.empty(int(ends[-1]) if len(ends) else 0)
    nodes[starts]=lows
    positions_inside=np.repeat(starts+1,inside)+(np.arange(int(inside.sum()))-np.repeat(np.cumsum(inside)-inside,inside))
    nodes[positions_inside]=x[positions_inside+np.repeat(a-starts-1,inside)]
    nodes[ends[jump]-2]=np.nextafter(highs[jump],lows[jump])
    nodes[ends-1]=highs
    query=nodes.copy()
    at_high=np.repeat(left,count)&(query==np.repeat(highs,count))
    query[at_high]=np.nextafter(np.repeat(highs,count)[at_high],np.repeat(lows,count)[at_high])
    read=np.asarray(curve.evaluate(query),dtype=float)
    return positions[keep],count.astype(np.int64),nodes,read


def _own_knots_one(curve,s):
    original_x=curve._x
    x=np.unique(np.r_[s.low,original_x[np.searchsorted(original_x,s.low,'right'):np.searchsorted(original_x,s.high,'left')],s.high])
    if not s.left_high and original_x[-1]==s.high and curve.endpoint_jump:
        x=np.unique(np.r_[x,np.nextafter(s.high,s.low)])
    query=x.copy()
    if s.left_high:query[query==s.high]=np.nextafter(s.high,s.low)
    return x,curve.evaluate(query)


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
    _linear: object=field(default=None,init=False,repr=False,compare=False)
    _cache: object=field(default=None,init=False,repr=False,compare=False)
    _identity: object=field(default=None,init=False,repr=False,compare=False)
    shared_cache: InitVar[object]=None
    identity: InitVar[object]=None

    def __post_init__(self,shared_cache,identity):
        # (*identity*, when given, names the curves as the default does.)
        object.__setattr__(self,'_cache',shared_cache);object.__setattr__(self,'_identity',identity)

    @property
    def _shared_linear(self):
        """The segment's shared lin-lin readers, made when first asked: only
        a segment read on its own asks (a group reads its curves itself).

        Segments of one material share their curves; a tape whose
        backgrounds step every few hundred eV has thousands of segments,
        and reading every whole curve again for each made this quadratic.
        """
        if self._linear is not None:return self._linear
        shared_cache=self._cache;identity=self._identity
        if identity is None:identity=tuple((key,id(curve)) for key,curve in self.curves.items())
        if shared_cache is not None and identity in shared_cache:
            object.__setattr__(self,'_linear',shared_cache[identity][1])
            return self._linear
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
        object.__setattr__(self,'_linear',tuple(shared))
        if shared_cache is not None:
            # The curves are held so their ids stay theirs.
            shared_cache[identity]=(tuple(self.curves.values()),self._linear)
        return self._linear


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
            zeros=self._zeros(constant_values)
            # Segments share their zero sets (JENDL-5 Cl-35: 212 for 3 376
            # segments), and the parts depend on nothing else.
            parts=self._columns_cache.get(('parts',zeros))
            if parts is None:
                parts=self._columns_cache[('parts',zeros)]={key:tuple(part for part in self._graph[key] if part not in zeros) for key in self._order}
            # The constants are held so their id stays theirs.
            plan=self._columns_cache[token]=(zeros,parts,constant_values);self._own(token,id(constant_values))
        return plan[0],plan[1]

    def _zeros(self,constant_values,zeros=None):
        """The leaves that are the constant zero on a segment (*zeros*, when
        the caller has them already, is filed as that answer)."""
        token=('zeros',id(constant_values))
        plan=self._columns_cache.get(token)
        if plan is None:
            if zeros is None:zeros=frozenset(key for key,value in constant_values.items() if value==0 and key not in self._graph)
            plan=self._columns_cache[token]=(zeros,constant_values);self._own(token,id(constant_values))
        return plan[0]

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
        physics=[None]*len(requests);regions={};adjusted=[];curves={};orders={}
        for n,item in enumerate(requests):
            if item[0] is _GROUP:
                # Energies of several segments: each segment's left edge rule,
                # and its physics in segment order, as if asked one by one.
                _,segs,consts,local,points,only,_=item
                e=np.asarray(points,dtype=float).copy()
                highs=np.array([s.high for s in segs]);lows_=np.array([s.low for s in segs])
                left=np.array([s.left_high for s in segs])
                at=left[local]&(e==highs[local])
                e[at]=np.nextafter(highs[local][at],lows_[local][at])
                adjusted.append(e);orders[n]=np.argsort(local,kind='stable')
                if segs[0].region is not None:regions.setdefault(id(segs[0].region),(segs[0].region,[]))[1].append(n)
                continue
            s,(points,only),constants=item
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
            energies=[adjusted[n][orders[n]] if n in orders else adjusted[n] for n in members]
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
                part={mt:v[end:end+len(e)] for mt,v in joined.items()};end+=len(e)
                if n in orders:
                    back=np.empty_like(orders[n]);back[orders[n]]=np.arange(len(back))
                    part={mt:v[back] for mt,v in part.items()}
                physics[n]=part
        # The group requests' columns, and each curve they read, read once
        # over all of them (the read is elementwise).
        grouped={};group_reads={}
        for n,item in enumerate(requests):
            if item[0] is not _GROUP:continue
            values,reads,everywhere=self._group_columns(item,adjusted[n]);grouped[n]=(values,everywhere)
            for curve,column,at in reads:group_reads.setdefault(id(curve),(curve,[]))[1].append((n,column,at))
        for curve,uses in group_reads.values():
            if len(uses)==1:
                n,column,at=uses[0];column[at]=curve.evaluate(adjusted[n][at]);continue
            read=curve.evaluate(np.concatenate([adjusted[n][at] for n,_,at in uses]));end=0
            for n,column,at in uses:column[at]=read[end:end+len(at)];end+=len(at)
        out=[]
        for n,(item,p,b) in enumerate(zip(requests,physics,backgrounds)):
            if item[0] is _GROUP:out.append(self._group_sums(*grouped[n],len(adjusted[n]),p));continue
            s,(e,only),c=item
            out.append(self._evaluate_segment(s,e,False,constant_values=self._source_constants(c),
                    only=self._diagnostic_keys(s,c),backgrounds=b) if only=='diagnostic' else
                self._evaluate_segment(s,e,constant_values=c,physics=p,only=only,backgrounds=b))
        return out

    def _evaluate_group(self,item,e,physics):
        """:meth:`_evaluate_segment` for energies of several segments of one
        region at once: every value is the one its own segment gives (a
        constant, its curve, the physics), and every sum
        adds the parts any of them needs (a part that is zero on a segment
        adds nothing there, exactly: a sum starts at +0.0)."""
        values,reads,everywhere=self._group_columns(item,e)
        for curve,column,at in reads:column[at]=curve.evaluate(e[at])
        return self._group_sums(values,everywhere,len(e),physics)

    def _group_columns(self,item,e):
        """A group request's columns but for its curve reads, which come
        back as ``(curve, column, at)`` for the caller to fill (a curve is
        read elementwise, so the caller may read it for several requests at
        once), with the zero leaves of every segment."""
        _,segs,consts,local,points,only,members=item
        n=len(e);order=np.argsort(local,kind='stable')
        cut=np.flatnonzero(np.diff(local[order]))+1
        position={int(local[p[0]]):p for p in np.split(order,cut)}
        zero_sets=[self._zeros(c) for c in consts]
        closure=self._wanted(only);asked=set(only)
        # Segments share their zero sets (the same object, see
        # reconstruct_suite): what each set wants is worked out once.
        distinct={id(zeros):zeros for zeros in zero_sets}
        everywhere=frozenset.intersection(*distinct.values())
        union_=set().union(*(closure-(zeros-asked) for zeros in distinct.values()))
        keys=[key for key in segs[0].curves if key in union_]
        # The segments' constant leaves and curve indices, as reconstruct_suite
        # laid them out for every segment (a group's constants are those).
        rows,held,constants,indices,choices=self._columns_cache['segment_tables']
        values={};reads=[]
        for key in keys:
            # Each segment's value: its constant or its curve's, a curve read
            # elementwise once over all the segments that read it. (Not the
            # segments' shared readers: each segment has its own set of
            # curves, so a reader serves one segment; their lin-lin arithmetic
            # and node values are the curve's, see panel_value.) A sum's
            # column is remade from its parts below.
            row=rows[key];here=held[row,members]
            column=np.where(here,constants[row,members],0.)[local]
            if key not in self._graph:
                index=indices[row,members];read=np.flatnonzero(~here&(index>=0))
                for j in np.unique(index[read]).tolist():
                    gs=read[index[read]==j].tolist()
                    at=position[gs[0]] if len(gs)==1 else np.concatenate([position[g] for g in gs])
                    reads.append((choices[row][j],column,at))
            values[key]=column
        return values,reads,everywhere

    def _group_sums(self,values,everywhere,n,physics):
        """A group request's physics added and sums made (see
        :meth:`_evaluate_group`)."""
        if 'total_parts' not in self._columns_cache:
            total=self._mt_keys.get(1)
            self._columns_cache['total_parts']=_sum_leaves(total,self._graph) if total is not None else None
        total_parts=self._columns_cache['total_parts']
        if physics is not None:
            for mt,value in physics.items():
                if mt==1:continue
                key=self._physical_owners.get(mt)
                if key is not None and key not in self._graph:
                    if total_parts is not None and key not in total_parts and np.any(value!=0):
                        raise UnsupportedResonanceError(f'total graph omits resonance partial MT{mt}',category='incomplete-total-graph')
                    if key in values:values[key]+=value
                elif np.any(value!=0):raise UnsupportedResonanceError(f'MF2 partial MT{mt} has no exclusive model reaction')
        for key in self._order:
            if key in values:
                values[key]=sum((values[part] for part in self._graph[key] if part not in everywhere),np.zeros(n))
        if values and not np.all(np.isfinite(np.concatenate(list(values.values())))):
            raise FloatingPointError('nonfinite suite cross section')
        return values

    def _source_constants(self,constants):
        """A segment's constant leaves: as the source states them, exactly
        (constant_on certifies the value); sums keep their stated forms."""
        token=('source',id(constants))
        plan=self._columns_cache.get(token)
        if plan is None:
            plan=self._columns_cache[token]=(MappingProxyType({k:v for k,v in constants.items() if k not in self._graph}),constants)
            self._own(token,id(constants))
            # Its zero leaves are the segment's.
            self._zeros(plan[0],self._zeros(constants))
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
    lows_,highs_=np.array(edges[:-1],dtype=float),np.array(edges[1:],dtype=float)
    owner_of={}
    for key,cs in curves.items():
        starts=np.array([c.x[0] for c in cs],dtype=float);ends=np.array([c.x[-1] for c in cs],dtype=float)
        first,stop=np.searchsorted(ends,highs_,'left'),np.searchsorted(starts,lows_,'right')
        count=stop-first
        if np.any(count>1):raise ValueError('ambiguous source region ownership')
        if np.any((count<1)&~((highs_<=cs[0].x[0])|(lows_>=cs[-1].x[-1]))):
            raise UnsupportedResonanceError(f'source function has an uncovered internal gap: {key}',category='internal-gap')
        owner_of[key]=(cs,np.where(count==1,first,-1).tolist())
    # Each segment's curves: one index per reaction (-1: none, which the
    # trailing None of each list reads), and the column of indices names them.
    keys_=tuple(owner_of);choices=[list(cs)+[None] for cs,_ in owner_of.values()]
    indices=np.array([at for _,at in owner_of.values()],dtype=np.int64).reshape(len(owner_of),len(edges)-1)
    row_of={key:i for i,key in enumerate(keys_)};columns_=np.ascontiguousarray(indices.T)
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        region=next((r for r in prepared.regions if r.low<=a and r.high>=b),None)
        column=columns_[i]
        segments.append(_SuiteSegment(a,b,_Curves(keys_,row_of,choices,column),region,b<high,shared_cache,column.tobytes()))
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
    span_high=[np.nextafter(s.high,s.low) if s.left_high else s.high for s in segments];level_cache={}
    # The knots of the source curves a segment reads: strictly inside (seeds)
    # and in the closed span (fast-region seeds, source diagnostics). The
    # segment edges include every piece's ends, so no piece overlaps a
    # segment in part: the knots of all the curves in a segment's span are
    # those of the curves it reads, but for its own ends, which are seeds
    # anyway; and seeds and grids are sets. One sorted array of every knot
    # (and one of the leaves') serves every segment.
    all_knots=np.unique(np.concatenate([c._x for cs,_ in owner_of.values() for c in cs])) if owner_of else np.zeros(0)
    leaf_curves=[c._x for key,(cs,_) in owner_of.items() if key not in graph for c in cs]
    leaf_knots=np.unique(np.concatenate(leaf_curves)) if leaf_curves else np.zeros(0)
    lows_of=np.array([s.low for s in segments]);highs_of=np.array([s.high for s in segments]);bounds_of={}
    def knots_of(knots,position,closed):
        # (Each segment's run of *knots*, found for all segments at once.)
        bounds=bounds_of.get((id(knots),closed))
        if bounds is None:
            bounds=bounds_of[(id(knots),closed)]=(
                np.searchsorted(knots,lows_of,'left' if closed else 'right').tolist(),
                np.searchsorted(knots,highs_of,'right' if closed else 'left').tolist())
        return knots[bounds[0][position]:bounds[1][position]]
    # Whether a leaf curve ends at the segment's high with a jump there.
    ends_jump=[False]*len(segments)
    for key,(cs,at) in owner_of.items():
        if key in graph:continue
        at=np.asarray(at)
        for j in np.unique(at[at>=0]).tolist():
            curve=cs[j]
            if not bool(curve.endpoint_jump):continue
            where=np.flatnonzero(at==j);hi=np.array([segments[i].high for i in where])
            for k in np.flatnonzero(curve._x[-1]==hi).tolist():ends_jump[where[k]]=True
    # Each reaction's constant (NaN: none) and linearity on each segment, as
    # (reaction, segment) arrays: a segment reads its column.
    names=tuple(owner_of);rows={key:i for i,key in enumerate(names)}
    keyed=np.empty(len(names),dtype=object)
    for i,key in enumerate(names):keyed[i]=key
    constant_of=np.full((len(names),len(segments)),np.nan);linear_of=np.zeros((len(names),len(segments)),dtype=bool)
    present_of=np.zeros((len(names),len(segments)),dtype=bool);summed_row=np.array([key in graph for key in names],dtype=bool)
    readers={}
    for row,(key,(cs,at)) in enumerate(owner_of.items()):
        at=np.asarray(at);present_of[row]=at>=0
        if key in graph:continue
        order_=np.argsort(at,kind='stable');at_sorted=at[order_]
        cut=np.flatnonzero(np.diff(at_sorted))+1
        for run in np.split(order_,cut):
            if len(run) and at[run[0]]>=0:
                curve=cs[int(at[run[0]])];readers.setdefault(id(curve),(curve,[]))[1].append((row,run))
    for curve,members in readers.values():
        where=np.concatenate([run for _,run in members]);lows=np.array([segments[i].low for i in where]);highs=np.array([span_high[i] for i in where])
        values=np.full(len(where),np.nan);wide=highs>lows
        if np.any(wide):values[wide]=curve.constant_on_many(lows[wide],highs[wide])
        linear=curve.linear_over_many(lows,highs);end=0
        for row,run in members:
            constant_of[row,run]=values[end:end+len(run)];linear_of[row,run]=linear[end:end+len(run)];end+=len(run)
    # Each sum's leaf parts as rows (None: a part with no curve, never a
    # constant) and its parts that are sums.
    sum_parts={}
    for key in order:
        leaves=[rows.get(part) for part in graph[key] if part not in graph]
        sum_parts[key]=(None if None in leaves else np.array(leaves,dtype=np.int64),
                        [part for part in graph[key] if part in graph])
    blocked_in={};zero_sets={};sum_values={}
    def blocked(region):
        """The reactions a segment of *region* never holds constant or
        defers: the sums, and the partials its physics owns."""
        out=blocked_in.get(id(region))
        if out is None:
            out=summed_row.copy()
            if region is not None:
                for mt in region_mts(region):
                    row=rows.get(physical_owners.get(mt))
                    if row is not None:out[row]=True
            blocked_in[id(region)]=out=(region,out)
        return out[1]
    # Each segment's constant leaves (a missing curve is the constant zero;
    # none on a span that closes to nothing) and deferred columns, for every
    # segment at once; prepare reads its row, and the group requests (whose
    # constants are these) read them by index.
    free_of=np.stack([~blocked(s.region) for s in segments],axis=1) if segments else np.zeros((len(names),0),dtype=bool)
    closed=np.array([(np.nextafter(s.high,s.low) if s.left_high else s.high)<=s.low for s in segments],dtype=bool)
    value_any=np.where(present_of,constant_of,0.)
    held_of=free_of&(value_any==value_any)&~closed
    value_of=np.where(held_of,value_any,0.)
    deferred_of=free_of&present_of&~held_of&linear_of
    held_rows,value_rows,deferred_rows=(np.ascontiguousarray(a.T) for a in (held_of,value_of,deferred_of))
    result._columns_cache['segment_tables']=(rows,held_of,value_of,indices,choices)
    def prepare(position):
        """Seeds, constants, deferred columns and source-diagnostic grid of
        one segment; none of it depends on another segment."""
        s=segments[position]
        if s.region is not None:
            # (The curves' knots come precomputed; _seeds reads no curve.)
            seeds=_seeds(_Segment(s.low,s.high,s.region,_NO_CONSTANTS,s.left_high),context,level_cache,
                         knots=([knots_of(leaf_knots,position,False)],ends_jump[position]))
        else:
            # Original knots suffice for linear fast-region data. Inserting
            # unnecessary near-threshold points amplifies ENDF x-rounding.
            seeds={s.low,s.high}|set(knots_of(leaf_knots,position,True).tolist())
            if ends_jump[position]:seeds.add(float(np.nextafter(s.high,s.low)))
            seeds=sorted(seeds)
        # The leaves constant here (a missing curve is the constant zero),
        # in the order of the segment's curves, then the sums of constants.
        held=held_rows[position];value=value_rows[position]
        at=np.flatnonzero(held)
        constants=_Constants(zip(keyed[at].tolist(),value[at].tolist()));added=[]
        for key in order:
            # Constant when every part is: its leaves held here, its sums
            # constant already (the order puts parts first). A sum of leaves
            # only is the same for the same values, added in the same order.
            leaves,sums=sum_parts[key]
            if leaves is not None and held[leaves].all() and all(part in constants for part in sums):
                if sums:total=sum((constants[part] for part in graph[key]),0.)
                else:
                    token=(key,value[leaves].tobytes());total=sum_values.get(token)
                    if total is None:total=sum_values[token]=sum(value[leaves].tolist(),0.)
                constants[key]=total;added.append((key,total))
        constants.order=names
        if all(rows.get(key) is not None for key,_ in added):
            constants.columns=np.concatenate([at,np.array([rows[key] for key,_ in added],dtype=at.dtype)])
            constants.column_values=np.concatenate([value[at],np.array([total for _,total in added],dtype=float)])
        else:constants.columns=constants.column_values=None
        # One object per zero set (JENDL-5 Cl-35: 212 for 3 376 segments), so
        # that what is worked out per set is worked out once.
        zeros=frozenset(keyed[at[value[at]==0]].tolist())
        result._zeros(constants,zero_sets.setdefault(zeros,zeros))
        deferred={}
        # Source knots are included in seeds, so these untouched columns
        # are affine inside each seed panel. Still check all seven final
        # probes; a failure falls back to the full adaptive column set.
        closed=(s.high,s.low) if s.left_high else None
        for row in np.flatnonzero(deferred_rows[position]).tolist():
            key=names[row];deferred[key]=_DeferredRead(s.curves[key],closed)
        work=min(len(seeds),max(1,options.max_work_bytes//1024))*(len(s.curves)-len(constants)-len(deferred)+1)*1024
        pending[position]=(constants,work)
        # Diagnose the evaluated aggregate against exclusive leaves. Output
        # sums are derived even if the redundant source aggregate disagrees.
        # Redundant aggregates inform diagnostics, not the reconstructed mesh.
        # Keep their original knots in this separate source-comparison grid.
        grid=np.unique(np.concatenate([np.asarray(seeds,dtype=float),knots_of(all_knots,position,True)]))
        probe=np.concatenate([grid,(grid[:-1]+grid[1:])*.5])
        return seeds,constants,deferred,probe
    def diagnose(positions,probes,original):
        """Each stated sum against the sum of its parts, on the probes of
        consecutive segments laid end to end, reported segment by segment in
        order. A part that is the constant zero on a segment is left out of
        its request and reads +0.0 there; adding it changes nothing, exactly
        (a sum starts at +0.0, so it is never -0.0, and x+0.0 is x)."""
        ends=np.cumsum([len(p) for p in probes]);starts=np.r_[0,ends[:-1]]
        for key in order:
            summed=sum((original[part] for part in graph[key] if part in original),np.zeros(int(ends[-1])))
            ratio=error_ratio(original[key],summed,options)
            maxima=np.maximum.reduceat(ratio,starts).tolist()
            for position,probe,a,b,maximum in zip(positions,probes,starts.tolist(),ends.tolist(),maxima):
                if maximum>source_balance[key]:
                    s=segments[position];index=int(np.argmax(ratio[a:b]))
                    source_worst[key]=dict(energy_eV=float(probe[index]),side='left-limit' if s.left_high and probe[index]==s.high else 'point',stated_b=float(original[key][a+index]),components_b=float(summed[a+index]),error_ratio=maximum)
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
        diagnose(positions,probes,dict(values))
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
    units={}
    def finish(index,value):
        nonlocal finished,done
        x,y,*check=value;s=segments[index]
        tables[index]=x;units[index]=dict(x=x,lengths=np.array([len(x)]),y=y,constant=None)
        if check:checks[index]=dict(domain=(s.low,s.high),points=len(x),**check[0])
        done+=len(x)
        if index in pending:result._forget(id(segments[index]),id(pending[index][0]))
        pending.pop(index,None)
        if progress is not None:
            finished=max(finished,float(min(1.,np.log(s.high/low)/span)) if span else
                         sum(t is not None for t in tables)/len(segments))
            progress(finished)
    def complete(positions,results,group):
        nonlocal done
        if group is not None and isinstance(results,dict):
            # One table for the group; each segment keeps its grid.
            start=0
            for i,n in zip(positions,results['lengths'].tolist()):
                tables[i]=results['x'][start:start+n];start+=n
                result._forget(id(segments[i]),id(pending[i][0]));pending.pop(i,None)
            units[positions[0]]=results;done+=int(results['lengths'].sum())
            if progress is not None:
                s=segments[positions[-1]]
                progress(float(min(1.,np.log(s.high/low)/span)) if span else
                         sum(t is not None for t in tables)/len(segments))
        else:
            for i,value in zip(positions,results):finish(i,value)
        if group is not None:
            # A group's segments report as one region, as the chunks of one
            # segment do.
            first,last=segments[positions[0]],segments[positions[-1]]
            checks[positions[0]]=dict(domain=(first.low,last.high),points=sum(len(tables[i]) for i in positions),**group)
    def source_values(e,owner):
        """A source-diagnostic request over the energies of several
        segments: every reaction as the source states it, sums included,
        each segment reading its constant leaves as their constants and the
        rest from their curves (one read per curve over all the segments
        that read it; elementwise, the left edge rule per segment). A leaf
        that is the constant zero on a segment is left out there: +0.0, as
        the segments' requests joined it."""
        members,local=np.unique(owner,return_inverse=True)
        e=np.asarray(e,dtype=float).copy()
        highs=np.array([segments[i].high for i in members]);lows=np.array([segments[i].low for i in members])
        left=np.array([segments[i].left_high for i in members])
        at=left[local]&(e==highs[local])
        e[at]=np.nextafter(highs[local][at],lows[local][at])
        order_=np.argsort(local,kind='stable')
        position=np.split(order_,np.flatnonzero(np.diff(local[order_]))+1)
        held,value,index=held_of[:,members],value_of[:,members],indices[:,members]
        asked=summed_row[:,None]|~(held&(value==0))
        out={}
        for row in np.flatnonzero(asked.any(axis=1)).tolist():
            ask=asked[row];constant=ask&held[row]
            column=np.where(constant[local],value[row][local],0.)
            read=np.flatnonzero(ask&~constant&(index[row]>=0))
            for j in np.unique(index[row][read]).tolist():
                gs=read[index[row][read]==j].tolist()
                at=position[gs[0]] if len(gs)==1 else np.concatenate([position[g] for g in gs])
                column[at]=choices[row][j].evaluate(e[at])
                if not np.all(np.isfinite(column[at])):raise FloatingPointError('nonfinite suite cross section')
            out[names[row]]=column
        return out
    def answer(requests):
        """Values for requests ``(energies, only, segment)``, *segment* one
        index or one per energy: each split into one request per segment."""
        flat=[];shapes=[]
        for e,only,owner in requests:
            if np.ndim(owner)==0:
                shapes.append(None);flat.append((segments[int(owner)],(e,only),pending[int(owner)][0]));continue
            if only=='diagnostic':
                shapes.append(('ready',source_values(e,owner)));continue
            if isinstance(only,tuple):
                members,local=np.unique(owner,return_inverse=True)
                shapes.append('group')
                flat.append((_GROUP,[segments[i] for i in members],[pending[i][0] for i in members],local,e,only,members))
                continue
            ordered=np.argsort(owner,kind='stable')
            parts=np.split(ordered,np.flatnonzero(np.diff(owner[ordered]))+1);shapes.append((len(e),parts))
            for idx in parts:
                i=int(owner[idx[0]]);flat.append((segments[i],(e[idx],only),pending[i][0]))
        values=result._answer(flat);out=[];at=0
        for (e,only,owner),shape in zip(requests,shapes):
            if isinstance(shape,tuple) and shape[0]=='ready':
                out.append(shape[1]);continue
            if shape is None or shape=='group':
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
            # (The fast region groups too: it has no physics, and its many
            # lin-lin backgrounds are deferred columns read once per group.)
            while (started<len(segments) and segments[started].region is region
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
    for x in tables:
        points+=len(x)
        if points>options.max_points:
            raise ReconstructionConvergenceError('refinement exceeds max_points',category='budget-exhausted')
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')]);output={};minima={}
    from kika.algebra import compress_flat_pieces
    lows=np.array([x[0] for x in tables]);highs=np.array([x[-1] for x in tables])
    ordered=[units[i] for i in sorted(units)]
    grid=np.concatenate([u['x'] for u in ordered]);lengths=np.concatenate([u['lengths'] for u in ordered])
    ends=np.cumsum(lengths);starts=ends-lengths
    names=tuple(segments[0].curves) if segments else ()
    column_of={key:j for j,key in enumerate(names)}
    def constants_of(key):
        # Each segment's constant for *key* (NaN: not a constant table there).
        out=[]
        for u in ordered:
            if u['constant'] is not None:out.append(u['constant'][:,column_of[key]])
            else:
                t=u['y'][key]
                # (A read-only table is a constant of the segment, shared by
                # _linearize_chunk.)
                out.append(np.array([np.nan if t.flags.writeable or len(t)<1 else t[0]]))
        return np.concatenate(out)
    for key in entries:
        values=constants_of(key);summed=key in graph
        if not summed and not np.any(np.isnan(values)) and np.all(lengths>=2) and not np.any((values==0)&np.signbit(values)):
            # The constant of every segment: compression leaves each segment
            # its two ends, and segments with one value join through their
            # shared edge. Written directly, as _compress_and_join_flat would make it.
            cut=np.flatnonzero(values[1:]!=values[:-1])+1
            children=[]
            for a,b in zip(np.r_[0,cut],np.r_[cut,len(values)]):
                cx=np.r_[lows[a],highs[a:b]];children.append(XYs1d(cx,np.full(len(cx),values[a]),axes=axes,index=len(children)))
            output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
            output[key].label=label
            minima[key]=float(np.min(values))
            for u in ordered:u['y'].pop(key,None)
            continue
        # (A group's column constant on all its segments is its constants
        # repeated over each segment's nodes; see _group_tables.)
        column=np.concatenate([u['y'][key] if key in u['y'] else np.repeat(u['constant'][:,column_of[key]],u['lengths'])
                               for u in ordered])
        own=[]
        if not summed:
            # An untouched lin-lin background already has an exact
            # representation on its own knots. It need not inherit every
            # other reaction's resonance or fast-region knot. The segments
            # that read one curve are done together.
            # (Its curve on each segment where it is not a constant and its
            # physics does not own it, grouped by curve, segments in order.)
            row=rows[key];index=indices[row]
            pick=np.flatnonzero(np.isnan(values)&(index>=0)&free_of[row])
            order_=pick[np.argsort(index[pick],kind='stable')]
            for where in np.split(order_,np.flatnonzero(np.diff(index[order_]))+1):
                if not len(where):continue
                where=where.tolist()
                table=_own_knots(choices[row][index[where[0]]],[segments[i] for i in where],where)
                if len(table[0]):own.append(table)
        if own:
            # Those segments' tables in place of theirs on the common grid:
            # each segment's nodes are a run of grid, or of its own table.
            at=np.concatenate([t[0] for t in own]);count=np.concatenate([t[1] for t in own])
            source=np.r_[grid,np.concatenate([t[2] for t in own])]
            read=np.r_[column,np.concatenate([t[3] for t in own])]
            sized=lengths.copy();sized[at]=count
            base=starts.copy();base[at]=len(grid)+np.cumsum(count)-count
            stop=np.cumsum(sized)
            take=np.repeat(base-(stop-sized),sized)+np.arange(int(stop[-1]))
            joined=_compress_and_join_flat(source[take],read[take],sized)
        else:
            joined=_compress_and_join_flat(grid,column,lengths)
        children=[XYs1d(cx,cy,axes=axes,index=i) for i,(cx,cy) in enumerate(joined)]
        output[key]=children[0] if len(children)==1 else Regions1d(children,axes=axes,label=label)
        output[key].label=label
        # A reaction's tables are released once its output exists, so the
        # output and the tables are never all held at once.
        minima[key]=float(np.min(column))
        for u in ordered:u['y'].pop(key,None)
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
        tuple(np.frombuffer(np.ascontiguousarray(x).tobytes(),dtype=x.dtype) for x in tables))


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
    # Radionuclide productions follow their reactions (MF10 linearised, as RECONR does).
    from ..productions import production_forms,publish_production_forms
    productions=production_forms(suite,result.source_style,result.label,result.forms,
                                 linearise_to=result.options.rtol)
    # Publish prebuilt containers only after every operation that can fail.
    candidates=_entries(candidate)
    containers={}
    for key,reaction in entries.items():
        container=deepcopy(reaction.crossSection)
        for existing,form in reaction.crossSection.items():container[existing]=form
        container[result.label]=candidates[key].crossSection[result.label]
        containers[key]=container
    for key,reaction in entries.items():reaction.crossSection=containers[key]
    publish_production_forms(productions)
    suite.styles=candidate.styles
    return suite


from .prepare import group_radii, group_knots, group_breaks

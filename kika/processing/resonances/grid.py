"""Batched multi-reaction linearization with explicit empirical error checks."""
from collections.abc import Mapping
from dataclasses import dataclass
import functools
import math
import numpy as np


@functools.lru_cache(maxsize=64)
def _key_index(keys):
    return {key:i for i,key in enumerate(keys)}


@functools.lru_cache(maxsize=64)
def _shared_keys(keys):
    """One tuple object for every segment with the same reactions."""
    return keys


class ReactionColumns(Mapping):
    """A read-only ``{reaction: value}`` of one segment's checks.

    A segment's report holds a value for every reaction, and a material
    with thousands of segments made them most of what a reconstruction kept
    (JENDL-5 Cl-35: 3 376 segments x 234 reactions, 173 MB as dicts). The
    keys are shared between segments; floats are stored as one array and
    read back as Python floats. It reads, compares and copies like the dict
    it replaces; ``dict(columns)`` gives that dict.
    """
    __slots__=('_keys','_values')

    def __init__(self,items):
        keys,values=zip(*items) if items else ((),())
        self._keys=_shared_keys(tuple(keys))
        floats=all(type(v) is float for v in values)
        self._values=np.array(values,dtype=float) if floats and values else tuple(values)

    @classmethod
    def of(cls,keys,values):
        """Columns from a key tuple and its values (an array or a tuple)."""
        self=cls.__new__(cls)
        self._keys=_shared_keys(tuple(keys));self._values=values
        return self

    def __getitem__(self,key):
        value=self._values[_key_index(self._keys)[key]]
        return float(value) if isinstance(self._values,np.ndarray) else value

    def __iter__(self):return iter(self._keys)
    def __len__(self):return len(self._keys)
    def __contains__(self,key):return key in _key_index(self._keys)
    def __repr__(self):return repr(dict(self))
    def __reduce__(self):return (ReactionColumns,(list(self.items()),))


class ReconstructionConvergenceError(RuntimeError):
    """No result is returned when a requested numerical budget is exhausted."""
    def __init__(self,message,*,category='verification-failed'):
        super().__init__(message)
        self.category=category


@dataclass(frozen=True)
class ReconstructionOptions:
    """Strict mixed error in barns; limits never relax tolerances.

    Half the requested budget is used for tabulation, leaving headroom for
    serialization. Finite interior probes are empirical checks, not a global
    mathematical bound. Negative values are retained and reported.
    ``max_work_bytes`` targets temporary batches and seed-panel chunks; retained
    input/output, adaptive chunk growth and backend allocations are additional.
    A dense exceptional solve exceeding its estimated target raises explicitly.
    """
    rtol: float = 1e-3
    atol: float = 1e-8
    max_points: int = 200000
    max_iterations: int = 40
    block_size: int = 2048
    max_work_bytes: int = 64*1024**2

    def __post_init__(self):
        if not math.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError('rtol must be finite and between zero and one')
        if not math.isfinite(self.atol) or self.atol <= 0:
            raise ValueError('atol must be finite and positive [barn]')
        for name in ('max_points','max_iterations','block_size','max_work_bytes'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_points < 2:
            raise ValueError('max_points must be at least two')


# Refinement and the second check use distinct interior locations. They both
# influence the adaptive mesh; external V3 probes must be independently chosen.
REFINEMENT_FRACTIONS=np.array([.25,.5,.75])
VERIFICATION_FRACTIONS=np.array([.1127016653792583,.276393202250021,
                                  .723606797749979,.8872983346207417])


def error_ratio(actual, linear, options):
    difference=np.abs(actual-linear)
    scale=np.maximum(np.abs(actual),np.abs(linear))
    # Reuse owned float64 temporaries on the production path, keeping the
    # expression's arithmetic order. Other dtypes retain ordinary promotion.
    if (isinstance(difference,np.ndarray) and isinstance(scale,np.ndarray)
            and difference.dtype==np.float64 and scale.dtype==np.float64
            and difference.shape==scale.shape
            and np.result_type(scale,options.rtol,options.atol)==np.float64):
        np.multiply(options.rtol,scale,out=scale)
        np.add(options.atol,scale,out=scale)
        np.divide(difference,scale,out=difference)
        return difference
    return difference/(options.atol+options.rtol*scale)


def check_dense_workspace(size,work_bytes):
    """Reject an exceptional dense system before allocating its workspace."""
    if 64*size*size>work_bytes:
        raise ReconstructionConvergenceError(
            f'dense resonance system of order {size} exceeds max_work_bytes={work_bytes}',
            category='memory-budget-exhausted')


def linearize(evaluate, seeds, options, point_budget, *, constants=None, deferred=None):
    """:func:`linearize_steps` driven by *evaluate* (energies -> {MT: values})."""
    from kika.algebra import drive
    return drive(linearize_steps(seeds,options,point_budget,constants=constants,deferred=deferred),
                 lambda request:evaluate(request[0]))


def _requests_of(steps, columns, mts):
    """Pass a refinement's requests up as energies, its answers back as columns."""
    try:
        request=next(steps)
        while True:
            request=steps.send(columns((yield request[0],mts)))
    except StopIteration as stop:
        return stop.value


def linearize_steps(seeds, options, point_budget, *, constants=None, deferred=None, keys=None):
    """Refine independent seed-panel chunks on the same shared reaction mesh.

    A generator: it yields ``(energies, mts)`` and is sent ``{MT: values}``
    there, for every MT or, when *mts* is a tuple, at least for those; it
    returns ``(grid, values, check)``. :func:`linearize` drives it
    with one function; the suite advances many segments together so that
    their resonance physics is evaluated in one batch.

    The workspace target sizes starting chunks; final tables and an adaptive
    chunk's growth are additional storage. It is not a bound on process RSS.
    Accepted panels retain all probes and the same error budgets. A failing panel gains its
    worst probe and the midpoint; adding all seven probes multiplies output size.
    ``constants`` declares functions proven constant by their source model,
    never inferred from equal samples. They retain output and diagnostics but
    need no adaptive column. Workspace sizing uses the active column count;
    retained complete output tables remain additional storage.
    ``deferred`` maps untouched piecewise-linear source columns to their own
    readers. They are checked at all seven final probes after refining the
    active columns; any failure repeats the chunk with every column active.
    Source knots must be seeds. Deferral never skips accepted-panel checks.
    With *keys* (every reaction, in output order) the seed request asks for
    the refined columns only: constants and deferred columns take their
    tables from their values and readers, never from these.
    """
    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    first=yield from _seed_values(x,keys,constants,deferred)
    count=max(1,sum(mt not in (constants or {}) and mt not in (deferred or {}) for mt in first))
    # Conservative live-array estimate for seven probes, sorting, chords,
    # ratios, kept probes, and child panels. Adaptive growth remains explicit.
    panels=max(1,min(4096,options.max_work_bytes//(1024*(count+1))))
    if len(x)-1<=panels:
        return (yield from _linearize_chunk(x,options,point_budget,first=first,constants=constants,deferred=deferred))
    chunks=[];checks=[];points=0
    for start in range(0,len(x)-1,panels):
        end=min(start+panels,len(x)-1)
        future_seeds=len(x)-end-1
        budget=point_budget-points-future_seeds+(1 if start else 0)
        grid,values,check=yield from _linearize_chunk(x[start:end+1],options,budget,
            first={mt:None if v is None else v[start:end+1] for mt,v in first.items()},constants=constants,deferred=deferred)
        keep=slice(1,None) if start else slice(None)
        chunks.append((grid[keep],{mt:v[keep] for mt,v in values.items()}))
        checks.append(check);points+=len(grid[keep])
    grid=np.concatenate([chunk[0] for chunk in chunks])
    values={mt:np.concatenate([chunk[1][mt] for chunk in chunks]) for mt in first}
    check=dict(iterations=max(c['iterations'] for c in checks),
        evaluations=sum(c['evaluations'] for c in checks)-(len(checks)-1),
        refinement_maxima=ReactionColumns([(mt,max(c['refinement_maxima'][mt] for c in checks)) for mt in first]),
        verification_maxima=ReactionColumns([(mt,max(c['verification_maxima'][mt] for c in checks)) for mt in first]),
        verification_worst={mt:max(((c['verification_maxima'][mt],c['verification_worst'][mt]) for c in checks
                                    if mt in c.get('verification_worst',{})),key=lambda t:t[0])[1]
                            for mt in first if any(mt in c.get('verification_worst',{}) for c in checks)},
        integrals=ReactionColumns([(mt,{weight:{name:sum(c['integrals'][mt][weight][name] for c in checks)
            for name in checks[0]['integrals'][mt][weight]} for weight in ('dE','dE_over_E')}) for mt in first]),
        refinement_chunks=len(chunks))
    return grid,values,check


def _linearize_chunk(seeds, options, point_budget, *, first=None, constants=None, deferred=None):
    """Return a common grid, values and checks; retain node evaluations.

    The refinement itself is :func:`kika.algebra.refine`, the one adaptive
    engine in kika: the three primary fractions reject failing panels first;
    a candidate accepted panel also passes all four verification fractions.
    A panel whose mixed error ratio exceeds one half anywhere gains
    its worst tested probe and midpoint. Exact previous-pass probes are reused
    on new panels; only missing energies are evaluated.
    """
    from kika.algebra import RefinementError, refine_steps

    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    if first is None:first=yield x,None
    seeded=len(x)
    constants={} if constants is None else constants
    deferred={} if deferred is None else deferred
    mts=[mt for mt in first if mt not in constants and mt not in deferred]
    # Keep one real column even on an entirely constant span, preserving the
    # ordinary probe/budget/unresolvable checks and reporting conventions.
    if not mts:mts=[next(iter(first))]
    columns=lambda values:np.column_stack([np.asarray(values[mt],dtype=float) for mt in mts])
    fractions=np.r_[REFINEMENT_FRACTIONS,VERIFICATION_FRACTIONS]
    try:
        result=yield from _requests_of(refine_steps(x,columns(first),
                      lambda actual,linear:error_ratio(actual,linear,options)/.5,
                      fractions=fractions,insert='balanced',max_passes=options.max_iterations,
                      max_points=point_budget,keep_probes=True,
                      precheck=3,reuse_probes=True),columns,tuple(mts))
    except RefinementError as exc:
        message={'points':'refinement exceeds max_points',
                 'passes':'refinement exceeds max_iterations',
                 'unresolvable':'required separation is not representable in float64'}[exc.reason]
        category={'points':'budget-exhausted','passes':'iterations-exhausted','unresolvable':'float64-separation'}[exc.reason]
        raise ReconstructionConvergenceError(message,category=category) from exc
    done=_chunk_tables(result.x,{mt:result.y[:,j] for j,mt in enumerate(mts)},result.probe_x,result.probe_y,
                       first,constants,deferred,options,result.passes,seeded+result.evaluations)
    if done is None:
        # Deferral is only scheduling: a column that fails any ordinary
        # accepted-panel probe must take part in a fresh complete refine.
        extra_evaluations=result.evaluations
        # Release retained candidate probes before allocating the retry.
        result=None
        if any(v is None for v in first.values()):first=None
        grid,values,check=yield from _linearize_chunk(seeds,options,point_budget,
            first=first,constants=constants)
        check['evaluations']+=extra_evaluations
        return grid,values,check
    return done


def verification_batches(grid,batch_size,fractions=VERIFICATION_FRACTIONS):
    """Original nodes, then every panel's probes, without a full probe array."""
    grid=np.asarray(grid,dtype=float)
    for start in range(0,len(grid),batch_size):
        yield grid[start:start+batch_size]
    fractions=np.asarray(fractions)
    # Flat indexing preserves the former panel-major order even for tiny batches.
    for start in range(0,(len(grid)-1)*len(fractions),batch_size):
        index=np.arange(start,min(start+batch_size,(len(grid)-1)*len(fractions)))
        panel,probe=np.divmod(index,len(fractions))
        yield grid[panel]+(grid[panel+1]-grid[panel])*fractions[probe]

def _refined(keys,constants,deferred):
    return [mt for mt in keys if mt not in constants and mt not in deferred] or [keys[0]]


def _seed_values(x,keys,constants,deferred):
    """The seed request: every reaction, or with *keys* the refined ones."""
    if keys is None:
        return (yield x,None)
    keys=tuple(keys);constants=constants or {};deferred=deferred or {}
    values=yield x,tuple(_refined(keys,constants,deferred))
    return {mt:values.get(mt) for mt in keys}


def linearize_group_steps(spans, options, point_budget, keys):
    """Refine consecutive segments of one region in one refinement.

    *spans* is a list of ``(seeds, constants, deferred)``, one per segment,
    in order; consecutive segments share their edge, which becomes a
    zero-width interval: a step, never refined, valued by each side's own
    segment. Panels are refined independently, so each segment's grid and
    checks are those it gets refined alone (the columns of the others are
    constant or affine on its panels and never fail there); what is shared
    is the cost per refinement, which a material cut into thousands of
    one-panel segments (JENDL-5 Cl-35) paid thousands of times.

    A generator: it yields ``(energies, mts, segment)`` -- *segment* the
    span index of each energy -- and is sent ``{MT: values}`` there. It
    returns ``(tables, check)``: per span ``(grid, values)`` and one
    check for the group; or, where a deferred column fails, per span
    ``(grid, values, check)`` or None for the caller to refine alone, and
    None.
    """
    from kika.algebra import RefinementError, refine_steps
    xs=[np.unique(np.asarray(seeds,dtype=float)) for seeds,_,_ in spans]
    if sum(len(x) for x in xs)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    x=np.concatenate(xs);node=np.repeat(np.arange(len(xs)),[len(v) for v in xs])
    keys=tuple(keys);union=[];owned=[]
    for _,constants,deferred in spans:
        mts=_refined(keys,constants,deferred);owned.append(mts)
        for mt in mts:
            if mt not in union:union.append(mt)
    # The seed request asks for the refined columns only (see linearize_steps).
    values=yield x,tuple(union),node
    first={mt:values.get(mt) for mt in keys}
    starts=np.r_[0,np.cumsum([len(v) for v in xs])]
    firsts=[{mt:None if v is None else v[a:b] for mt,v in first.items()} for a,b in zip(starts[:-1],starts[1:])]
    index={mt:j for j,mt in enumerate(union)}
    columns=lambda values:np.column_stack([np.asarray(values[mt],dtype=float) for mt in union])
    # The span of each starting interval; a boundary step has none of its own.
    interval=np.repeat(np.arange(len(xs)),[len(v) for v in xs])[:-1]
    evaluations=np.zeros(len(xs),dtype=np.int64)
    steps=refine_steps(x,columns(first),lambda actual,linear:error_ratio(actual,linear,options)/.5,
        fractions=np.r_[REFINEMENT_FRACTIONS,VERIFICATION_FRACTIONS],insert='balanced',
        max_passes=options.max_iterations,max_points=point_budget,keep_probes=True,precheck=3,reuse_probes=True)
    try:
        try:
            q,owner=next(steps)
            while True:
                segment=interval[owner]
                evaluations+=np.bincount(segment,minlength=len(xs))
                q,owner=steps.send(columns((yield q,tuple(union),segment)))
        except StopIteration as stop:
            result=stop.value
    except RefinementError as exc:
        message={'points':'refinement exceeds max_points',
                 'passes':'refinement exceeds max_iterations',
                 'unresolvable':'required separation is not representable in float64'}[exc.reason]
        category={'points':'budget-exhausted','passes':'iterations-exhausted','unresolvable':'float64-separation'}[exc.reason]
        raise ReconstructionConvergenceError(message,category=category) from exc
    # Each edge is twice in the grid: the first copy ends the span below, the
    # second starts the one above. Added nodes are strictly inside a span.
    bounds=[]
    for k in range(len(xs)):
        a=int(np.searchsorted(result.x,xs[k][0],'left'))+(1 if k else 0)
        b=int(np.searchsorted(result.x,xs[k][-1],'left'))+1
        bounds.append((a,b))
    panels=np.cumsum([0]+[b-a-1 for a,b in bounds])
    def alone():
        # Each span's own checks, as if refined alone (a deferred column of
        # some span failed: _chunk_tables marks it for its own retry).
        out=[]
        for k,(f,(_,constants,deferred),mts) in enumerate(zip(firsts,spans,owned)):
            (a,b),p0,p1=bounds[k],panels[k],panels[k+1]
            take=[index[mt] for mt in mts]
            passes=int(result.probe_pass[p0:p1].max()) if p1>p0 else 0
            out.append(_chunk_tables(result.x[a:b],{mt:result.y[a:b,index[mt]] for mt in mts},result.probe_x[p0:p1],
                                     result.probe_y[p0:p1][:,:,take],f,constants,deferred,options,passes,
                                     len(xs[k])+int(evaluations[k])))
        return out,None
    return _group_tables(result,xs,bounds,panels,firsts,spans,union,index,evaluations,options,alone)


def _group_tables(result,xs,bounds,panels,firsts,spans,union,index,evaluations,options,alone):
    """Per-span tables and one check for the whole group.

    The tables are those of each span refined alone: its refined columns
    from the common refinement, its constants as their shared table and its
    deferred columns from their readers. The check is the group's, as
    linearize aggregates the chunks of one segment: maxima are the maxima
    over every panel, evaluations add, iterations are the deepest pass.
    """
    keys=tuple(firsts[0])
    pos=np.flatnonzero(np.diff(result.x)>0)
    probes=result.probe_x
    if len(pos)!=len(probes):return alone()
    span_of_panel=np.repeat(np.arange(len(xs)),np.diff(panels))
    x1,x2=result.x[pos],result.x[pos+1]
    fractions=(probes-x1[:,None])/(x2-x1)[:,None]
    weights=np.array([5/18,4/9,5/18]);gauss=[3,1,6]
    factor=(x2-x1)[:,None]*weights
    factors=(('dE',factor),('dE_over_E',factor*(1/probes[:,gauss])))
    # Columns no span refines: deferred in some spans, constant in the rest.
    late=[mt for mt in keys if mt not in index and any(mt in d for _,_,d in spans)]
    tables=[];late_samples={mt:np.empty(probes.shape) for mt in late};late_nodes={mt:np.empty((len(pos),2)) for mt in late}
    for k,(f,(_,constants,deferred)) in enumerate(zip(firsts,spans)):
        (a,b),p0,p1=bounds[k],panels[k],panels[k+1]
        grid=result.x[a:b];by_value={};y={}
        for mt in keys:
            if mt in constants:
                value=constants[mt];token=float(value).hex()
                if token not in by_value:
                    table=np.full(len(grid),value);table.setflags(write=False);by_value[token]=table
                y[mt]=by_value[token]
            elif mt in index:y[mt]=result.y[a:b,index[mt]]
            else:y[mt]=None
        tables.append((grid,y))
    # Deferred columns that no span refines: their readers at every span's
    # nodes and probes, one curve read for all the spans that share it.
    for mt in late:
        users=[k for k in range(len(xs)) if mt in spans[k][2]]
        readers=[spans[k][2][mt] for k in users]
        curve=getattr(readers[0],'curve',None)
        if curve is not None and all(getattr(r,'curve',None) is curve for r in readers):
            pieces=[];sizes=[]
            for k,r in zip(users,readers):
                p0,p1=panels[k],panels[k+1]
                e=np.r_[tables[k][0],probes[p0:p1].ravel()]
                if r.left_high is not None:e[e==r.left_high[0]]=np.nextafter(*r.left_high)
                pieces.append(e);sizes.append(len(e))
            values=np.asarray(curve.evaluate(np.concatenate(pieces)),dtype=float)
            parts=np.split(values,np.cumsum(sizes)[:-1])
        else:
            parts=[np.asarray(r(np.r_[tables[k][0],probes[panels[k]:panels[k+1]].ravel()]),dtype=float)
                   for k,r in zip(users,readers)]
        for k,v in zip(users,parts):
            p0,p1=panels[k],panels[k+1];n=len(tables[k][0])
            tables[k][1][mt]=v[:n];late_samples[mt][p0:p1]=v[n:].reshape(probes[p0:p1].shape)
        for k in range(len(xs)):
            p0,p1=panels[k],panels[k+1];y=tables[k][1]
            if mt not in spans[k][2]:late_samples[mt][p0:p1]=spans[k][1][mt]
            late_nodes[mt][p0:p1,0]=y[mt][:-1];late_nodes[mt][p0:p1,1]=y[mt][1:]
    checked=list(union)+late
    samples=result.probe_y
    nodes=np.stack([result.y[pos],result.y[pos+1]],axis=1)
    if late:
        samples=np.concatenate([samples,np.stack([late_samples[mt] for mt in late],axis=-1)],axis=-1)
        nodes=np.concatenate([nodes,np.stack([late_nodes[mt] for mt in late],axis=-1)],axis=-1)
    # A column constant on a span is its constant there, on the nodes and
    # the probes alike, so it never counts against that span.
    for j,mt in enumerate(checked):
        for k,(_,constants,_) in enumerate(spans):
            if mt in constants:
                p0,p1=panels[k],panels[k+1]
                samples[p0:p1,:,j]=constants[mt];nodes[p0:p1,:,j]=constants[mt]
    linear=nodes[:,0,None,:]+(nodes[:,1]-nodes[:,0])[:,None,:]*fractions[:,:,None]
    ratio=error_ratio(samples,linear,options)
    for j,mt in enumerate(late):
        r=ratio[:,:,len(union)+j]
        deferred_here=np.array([mt in spans[k][2] for k in range(len(xs))])[span_of_panel]
        if np.any(~np.isfinite(r[deferred_here])) or np.any(r[deferred_here]>.5):return alone()
    refinement=np.zeros(len(keys));verification=np.zeros(len(keys));integrals=[None]*len(keys);worst={}
    where={mt:i for i,mt in enumerate(keys)}
    for j,mt in enumerate(checked):
        r=ratio[:,:,j];i=where[mt]
        refinement[i]=float(np.max(r[:,:3]));verification[i]=float(np.max(r[:,3:]))
        panel,probe=np.unravel_index(int(np.argmax(r[:,3:])),r[:,3:].shape);probe+=3
        worst[mt]=(float(probes[panel,probe]),float(samples[panel,probe,j]),float(linear[panel,probe,j]))
        a=np.ascontiguousarray(samples[:,gauss,j]);l=np.ascontiguousarray(linear[:,gauss,j])
        integrals[i]={name:{'reference_estimate':float(np.sum(f*a)),'linear_estimate':float(np.sum(f*l)),
                            'absolute_difference_estimate':float(np.sum(f*np.abs(a-l)))} for name,f in factors}
    for i,mt in enumerate(keys):
        if integrals[i] is not None:continue
        # Constant on every span: its quadrature is the constant's.
        value=np.array([spans[k][1][mt] for k in range(len(xs))])[span_of_panel][:,None]
        estimates={name:float(np.sum(f*value)) for name,f in factors}
        integrals[i]={name:dict(reference_estimate=estimates[name],linear_estimate=estimates[name],
                                absolute_difference_estimate=0.) for name,f in factors}
    check=dict(iterations=int(result.probe_pass.max()) if len(result.probe_pass) else 0,
               evaluations=int(sum(len(x) for x in xs)+evaluations.sum()),
               refinement_maxima=ReactionColumns.of(keys,refinement),
               verification_maxima=ReactionColumns.of(keys,verification),
               verification_worst=worst,integrals=ReactionColumns.of(keys,tuple(integrals)))
    return tables,check


def _chunk_tables(x, y, probes, actual, first, constants, deferred, options, iterations, evaluations):
    """Tables and checks of one refined span, or None when a deferred column
    fails a probe (the caller then refines with every column active).

    *y* maps the refined columns, in the order of *actual*'s last axis, to
    their node values; every other reaction of *first* is deferred or
    constant.
    """
    y=dict(y)
    column={mt:j for j,mt in enumerate(y)}
    # On very narrow panels a nominal fraction may round onto an endpoint.
    # Compare the line at the *representable* queried abscissa.
    actual_fractions=(probes-x[:-1,None])/np.diff(x)[:,None]
    weights=np.array([5/18,4/9,5/18])
    gauss_indices=[3,1,6]
    energy=probes[:,gauss_indices]
    factor=np.diff(x)[:,None]*weights
    factors=(('dE',factor),('dE_over_E',factor*(1/energy)))
    keys=tuple(first)
    # The checked columns -- refined and deferred -- in one stacked array; a
    # material cut into thousands of segments has hundreds of constant
    # partials in each, which only need their shared table and quadrature.
    late=[mt for mt in keys if mt in deferred and mt not in column]
    for mt in late:y[mt]=np.asarray(deferred[mt](x),dtype=float)
    checked=list(column)+late
    samples=actual
    if late:
        extra=np.stack([np.asarray(deferred[mt](probes.ravel()),dtype=float).reshape(probes.shape) for mt in late],axis=-1)
        samples=np.concatenate([actual,extra],axis=-1)
    tables=np.stack([y[mt] for mt in checked],axis=-1)
    linear=tables[:-1,None,:]+(tables[1:]-tables[:-1])[:,None,:]*actual_fractions[:,:,None]
    ratio=error_ratio(samples,linear,options)
    if late:
        bad=np.any(~np.isfinite(ratio[:,:,len(column):])) or np.any(ratio[:,:,len(column):]>.5)
        if bad:return None
    index=_key_index(_shared_keys(keys))
    refinement=np.zeros(len(keys));verification=np.zeros(len(keys));integrals=[None]*len(keys)
    worst={}
    a_all=np.ascontiguousarray(np.moveaxis(samples[:,gauss_indices,:],-1,0))
    l_all=np.ascontiguousarray(np.moveaxis(linear[:,gauss_indices,:],-1,0))
    for k,mt in enumerate(checked):
        r=ratio[:,:,k];i=index[mt]
        refinement[i]=float(np.max(r[:,:3]));verification[i]=float(np.max(r[:,3:]))
        # Where the verification maximum sits: (energy, sampled, tabulated).
        panel,probe=np.unravel_index(int(np.argmax(r[:,3:])),r[:,3:].shape);probe+=3
        worst[mt]=(float(probes[panel,probe]),float(samples[panel,probe,k]),float(linear[panel,probe,k]))
        a,b=a_all[k],l_all[k]
        integrals[i]={name:{'reference_estimate':float(np.sum(f*a)),'linear_estimate':float(np.sum(f*b)),
                            'absolute_difference_estimate':float(np.sum(f*np.abs(a-b)))} for name,f in factors}
    by_value={}
    def constant(value):
        # Many modeled partials are identically zero on a resonance span.
        # Their table and quadrature are made, and reported, once per value.
        token=float(value).hex()
        if token not in by_value:
            table=np.full(len(x),value);table.setflags(write=False)
            quadrature={}
            for name,f in factors:
                estimate,difference=float(np.sum(f*value)),float(np.sum(f*0.))
                quadrature[name]=dict(reference_estimate=estimate,linear_estimate=estimate,
                                      absolute_difference_estimate=difference)
            by_value[token]=(table,quadrature)
        return by_value[token]
    out={}
    for i,mt in enumerate(keys):
        if mt in y:out[mt]=y[mt]
        else:out[mt],integrals[i]=constant(constants[mt])
    return x,out,dict(iterations=iterations,evaluations=evaluations,
                   refinement_maxima=ReactionColumns.of(keys,refinement),
                   verification_maxima=ReactionColumns.of(keys,verification),
                   verification_worst=worst,integrals=ReactionColumns.of(keys,tuple(integrals)))

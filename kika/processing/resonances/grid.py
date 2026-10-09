"""Batched multi-reaction linearization with explicit empirical error checks."""
from dataclasses import dataclass
import math
import numpy as np


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
    """Refine independent seed-panel chunks on the same shared reaction mesh.

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
    """
    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    first=evaluate(x)
    count=max(1,sum(mt not in (constants or {}) and mt not in (deferred or {}) for mt in first))
    # Conservative live-array estimate for seven probes, sorting, chords,
    # ratios, kept probes, and child panels. Adaptive growth remains explicit.
    panels=max(1,min(4096,options.max_work_bytes//(1024*(count+1))))
    if len(x)-1<=panels:
        return _linearize_chunk(evaluate,x,options,point_budget,first=first,constants=constants,deferred=deferred)
    chunks=[];checks=[];points=0
    for start in range(0,len(x)-1,panels):
        end=min(start+panels,len(x)-1)
        future_seeds=len(x)-end-1
        budget=point_budget-points-future_seeds+(1 if start else 0)
        grid,values,check=_linearize_chunk(evaluate,x[start:end+1],options,budget,
            first={mt:v[start:end+1] for mt,v in first.items()},constants=constants,deferred=deferred)
        keep=slice(1,None) if start else slice(None)
        chunks.append((grid[keep],{mt:v[keep] for mt,v in values.items()}))
        checks.append(check);points+=len(grid[keep])
    grid=np.concatenate([chunk[0] for chunk in chunks])
    values={mt:np.concatenate([chunk[1][mt] for chunk in chunks]) for mt in first}
    check=dict(iterations=max(c['iterations'] for c in checks),
        evaluations=sum(c['evaluations'] for c in checks)-(len(checks)-1),
        refinement_maxima={mt:max(c['refinement_maxima'][mt] for c in checks) for mt in first},
        verification_maxima={mt:max(c['verification_maxima'][mt] for c in checks) for mt in first},
        integrals={mt:{weight:{name:sum(c['integrals'][mt][weight][name] for c in checks)
            for name in checks[0]['integrals'][mt][weight]} for weight in ('dE','dE_over_E')} for mt in first},
        refinement_chunks=len(chunks))
    return grid,values,check


def _linearize_chunk(evaluate, seeds, options, point_budget, *, first=None, constants=None, deferred=None):
    """Return a common grid, values and checks; retain node evaluations.

    The refinement itself is :func:`kika.algebra.refine`, the one adaptive
    engine in kika: the three primary fractions reject failing panels first;
    a candidate accepted panel also passes all four verification fractions.
    A panel whose mixed error ratio exceeds one half anywhere gains
    its worst tested probe and midpoint. Exact previous-pass probes are reused
    on new panels; only missing energies are evaluated.
    (Until October 2026 every pass re-probed every panel, converged ones
    included, which asked the physics the same question once per pass.)
    """
    from kika.algebra import RefinementError, refine

    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    first=evaluate(x) if first is None else first
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
        result=refine(x,columns(first),lambda q,owner:columns(evaluate(q)),
                      lambda actual,linear:error_ratio(actual,linear,options)/.5,
                      fractions=fractions,insert='balanced',max_passes=options.max_iterations,
                      max_points=point_budget,keep_probes=True,
                      precheck=3,reuse_probes=True)
    except RefinementError as exc:
        message={'points':'refinement exceeds max_points',
                 'passes':'refinement exceeds max_iterations',
                 'unresolvable':'required separation is not representable in float64'}[exc.reason]
        category={'points':'budget-exhausted','passes':'iterations-exhausted','unresolvable':'float64-separation'}[exc.reason]
        raise ReconstructionConvergenceError(message,category=category) from exc
    x=result.x
    y={mt:result.y[:,j] for j,mt in enumerate(mts)}
    probes,actual=result.probe_x,result.probe_y
    # On very narrow panels a nominal fraction may round onto an endpoint.
    # Compare the line at the *representable* queried abscissa.
    actual_fractions=(probes-x[:-1,None])/np.diff(x)[:,None]
    weights=np.array([5/18,4/9,5/18])
    gauss_indices=[3,1,6]
    energy=probes[:,gauss_indices]
    factor=np.diff(x)[:,None]*weights
    factors=(('dE',factor),('dE_over_E',factor*(1/energy)))
    maxima={}
    integrals={}
    for mt in first:
        if mt not in mts and mt not in deferred:continue
        if mt in mts:
            samples=actual[:,:,mts.index(mt)]
        else:
            y[mt]=np.asarray(deferred[mt](x),dtype=float)
            samples=np.asarray(deferred[mt](probes.ravel()),dtype=float).reshape(probes.shape)
        linear=y[mt][:-1,None]+(y[mt][1:]-y[mt][:-1])[:,None]*actual_fractions
        ratio=error_ratio(samples,linear,options)
        if mt in deferred and (np.any(~np.isfinite(ratio)) or np.any(ratio>.5)):
            # Deferral is only scheduling: a column that fails any ordinary
            # accepted-panel probe must take part in a fresh complete refine.
            extra_evaluations=result.evaluations
            # Release retained candidate probes before allocating the retry.
            result=x=y=probes=actual=actual_fractions=samples=linear=ratio=None
            factors=factor=energy=a=None
            grid,values,check=_linearize_chunk(evaluate,seeds,options,point_budget,
                first=first,constants=constants)
            check['evaluations']+=extra_evaluations
            return grid,values,check
        maxima[mt]=(float(np.max(ratio[:,:3])),float(np.max(ratio[:,3:])))
        a=samples[:,gauss_indices]
        integrals[mt]={}
        for name,factor in factors:
            integrals[mt][name]={'reference_estimate':float(np.sum(factor*a)),
                                'linear_estimate':float(np.sum(factor*linear[:,gauss_indices])),
                                'absolute_difference_estimate':float(np.sum(factor*np.abs(a-linear[:,gauss_indices])))}
    for mt in first:
        if mt in y:continue
        value=constants[mt]
        y[mt]=np.full(len(x),value)
        maxima[mt]=(0.,0.)
        integrals[mt]={}
        for name,factor in factors:
            estimate=float(np.sum(factor*value))
            integrals[mt][name]=dict(reference_estimate=estimate,
                linear_estimate=estimate,absolute_difference_estimate=float(np.sum(factor*0.)))
    y={mt:y[mt] for mt in first}
    return x,y,dict(iterations=result.passes,evaluations=seeded+result.evaluations,
                   refinement_maxima={mt:v[0] for mt,v in maxima.items()},
                   verification_maxima={mt:v[1] for mt,v in maxima.items()},
                   integrals=integrals)


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

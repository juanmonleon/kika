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
    """
    rtol: float = 1e-3
    atol: float = 1e-8
    max_points: int = 200000
    max_iterations: int = 40
    block_size: int = 2048

    def __post_init__(self):
        if not math.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError('rtol must be finite and between zero and one')
        if not math.isfinite(self.atol) or self.atol <= 0:
            raise ValueError('atol must be finite and positive [barn]')
        for name in ('max_points','max_iterations','block_size'):
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
    return np.abs(actual-linear)/(options.atol+options.rtol*np.maximum(np.abs(actual),np.abs(linear)))


def linearize(evaluate, seeds, options, point_budget):
    """Return a common grid, values and checks; retain node evaluations.

    The refinement itself is :func:`kika.algebra.refine`, the one adaptive
    engine in kika: every panel is probed at the refinement and verification
    fractions, a panel whose mixed error ratio exceeds one half anywhere gains
    all of its probes, and only the panels a pass creates are probed again.
    (Until October 2026 every pass re-probed every panel, converged ones
    included, which asked the physics the same question once per pass.)
    """
    from kika.algebra import RefinementError, refine

    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points',category='budget-exhausted')
    first=evaluate(x)
    seeded=len(x)
    mts=list(first)
    columns=lambda values:np.column_stack([np.asarray(values[mt],dtype=float) for mt in mts])
    fractions=np.r_[REFINEMENT_FRACTIONS,VERIFICATION_FRACTIONS]
    try:
        result=refine(x,columns(first),lambda q,owner:columns(evaluate(q)),
                      lambda actual,linear:error_ratio(actual,linear,options)/.5,
                      fractions=fractions,insert='all',max_passes=options.max_iterations,
                      max_points=point_budget,keep_probes=True)
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
    maxima={}
    integrals={}
    for j,mt in enumerate(mts):
        linear=y[mt][:-1,None]+(y[mt][1:]-y[mt][:-1])[:,None]*actual_fractions
        ratio=error_ratio(actual[:,:,j],linear,options)
        maxima[mt]=(float(np.max(ratio[:,:3])),float(np.max(ratio[:,3:])))
        a=actual[:,gauss_indices,j]
        energy=probes[:,gauss_indices]
        integrals[mt]={}
        for name,weight in (('dE',1.),('dE_over_E',1/energy)):
            factor=np.diff(x)[:,None]*weights*weight
            integrals[mt][name]={'reference_estimate':float(np.sum(factor*a)),
                                'linear_estimate':float(np.sum(factor*linear[:,gauss_indices])),
                                'absolute_difference_estimate':float(np.sum(factor*np.abs(a-linear[:,gauss_indices])))}
    return x,y,dict(iterations=result.passes,evaluations=seeded+result.evaluations,
                   refinement_maxima={mt:v[0] for mt,v in maxima.items()},
                   verification_maxima={mt:v[1] for mt,v in maxima.items()},
                   integrals=integrals)

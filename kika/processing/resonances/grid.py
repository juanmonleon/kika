"""Batched multi-reaction linearization with explicit empirical error checks."""
from dataclasses import dataclass
import math
import numpy as np


class ReconstructionConvergenceError(RuntimeError):
    """No result is returned when a requested numerical budget is exhausted."""


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
    """Return a common grid, values and checks; retain node evaluations."""
    x=np.unique(np.asarray(seeds,dtype=float))
    if len(x)>point_budget:
        raise ReconstructionConvergenceError('seed grid exceeds max_points')
    y=evaluate(x)
    evaluations=len(x)
    for iteration in range(options.max_iterations):
        fractions=np.r_[REFINEMENT_FRACTIONS,VERIFICATION_FRACTIONS]
        probes=x[:-1,None]+np.diff(x)[:,None]*fractions
        if np.any(probes <= x[:-1,None]) or np.any(probes >= x[1:,None]):
            # Only irreducible panels need special handling; larger panels
            # may still be refined. Test their endpoint-scale error below.
            probes=np.maximum(x[:-1,None],np.minimum(x[1:,None],probes))
        actual=evaluate(probes.ravel())
        # On very narrow panels a nominal fraction may round onto an endpoint.
        # Compare the line at the *representable* queried abscissa.
        actual_fractions=(probes-x[:-1,None])/np.diff(x)[:,None]
        evaluations+=probes.size
        bad=np.zeros(len(x)-1,dtype=bool)
        maxima={}
        for mt in y:
            linear=y[mt][:-1,None]+(y[mt][1:]-y[mt][:-1])[:,None]*actual_fractions
            ratio=error_ratio(actual[mt].reshape(probes.shape),linear,options)
            bad |= np.any(ratio>.5,axis=1)
            maxima[mt]=(float(np.max(ratio[:,:3])),float(np.max(ratio[:,3:])))
        if not np.any(bad):
            weights=np.array([5/18,4/9,5/18])
            gauss_indices=[3,1,6]
            integrals={}
            for mt in y:
                a=actual[mt].reshape(probes.shape)[:,gauss_indices]
                linear=y[mt][:-1,None]+(y[mt][1:]-y[mt][:-1])[:,None]*actual_fractions[:,gauss_indices]
                energy=probes[:,gauss_indices]
                integrals[mt]={}
                for name,weight in (('dE',1.),('dE_over_E',1/energy)):
                    factor=np.diff(x)[:,None]*weights*weight
                    integrals[mt][name]={'reference_estimate':float(np.sum(factor*a)),
                                        'linear_estimate':float(np.sum(factor*linear)),
                                        'absolute_difference_estimate':float(np.sum(factor*np.abs(a-linear)))}
            return x,y,dict(iterations=iteration+1,evaluations=evaluations,
                           refinement_maxima={mt:v[0] for mt,v in maxima.items()},
                           verification_maxima={mt:v[1] for mt,v in maxima.items()},
                           integrals=integrals)
        additions=probes[bad].ravel()
        merged=np.unique(np.r_[x,additions])
        if len(merged)==len(x):
            raise ReconstructionConvergenceError('required separation is not representable in float64')
        if len(merged)>point_budget:
            raise ReconstructionConvergenceError('refinement exceeds max_points')
        # Cache both old endpoints and all new samples; no endpoint recomputation.
        order=np.argsort(np.r_[x,additions],kind='stable')
        all_x=np.r_[x,additions][order]
        unique=np.r_[True,np.diff(all_x)>0]
        y={mt:np.r_[v,actual[mt].reshape(probes.shape)[bad].ravel()][order][unique]
           for mt,v in y.items()}
        x=merged
    raise ReconstructionConvergenceError('refinement exceeds max_iterations')

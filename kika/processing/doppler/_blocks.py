"""Temperature-local moments for dense nonnegative linlin sources.

Gauss-Legendre integrates the source polynomials exactly in exact arithmetic;
the Gaussian expansion remainder and float diagnostics share the kernel budget.
The original point evaluator remains the fallback for all rejected queries.
"""
import math
import numpy as np
from . import core

_ORDER=16
_MIN_PANELS=8
_DENSE_SOURCE=250000
_MIN_DIRECT_QUERIES=16384


class _BlockRows(list):
    pass


def _prepare(curve,h,options):
    nodes,weights=np.polynomial.legendre.leggauss(11)
    rows=[];offset=int(curve.regions[0][0][0]>0 and curve.support.lower!='zero')
    limit=options.max_work_bytes//16//(8*(_ORDER+8))
    for x,y in curve.regions:
        if len(x)*8>options.max_work_bytes//4 or np.any(y<0):return None
        velocity=np.sqrt(x)
        groups=[];start=0
        while start<len(x)-1:
            stop=min(len(x)-1,max(start+1,int(np.searchsorted(velocity,velocity[start]+h,side='right'))-1))
            if stop-start>=_MIN_PANELS:
                groups.append((start,stop))
                if len(rows)+len(groups)>limit:return None
            start=stop
        if groups:
            starts=np.array([a for a,b in groups]);ends=np.array([b for a,b in groups])
            sums=np.zeros((len(groups),_ORDER+1))
            for chunk in range(0,len(x)-1,8192):
                stop=min(len(x)-1,chunk+8192)
                indices=np.arange(chunk,stop)
                ids=np.searchsorted(starts,indices,side='right')-1
                good=(ids>=0)&(indices<ends[np.maximum(ids,0)])
                indices=indices[good];ids=ids[good]
                if not len(indices):continue
                lo,hi=x[indices],x[indices+1]
                yl=velocity[indices];ds=(hi-lo)/(velocity[indices+1]+yl)
                fraction=(nodes+1.)*.5
                displacement=ds[:,None]*fraction
                energy_fraction=fraction*(2*yl[:,None]+displacement)/(2*yl[:,None]+ds[:,None])
                sigma=y[indices,None]+(y[indices+1]-y[indices])[:,None]*energy_fraction
                v=yl[:,None]+displacement
                centre=(velocity[starts[ids]]+velocity[ends[ids]])*.5
                half=(x[ends[ids]]-x[starts[ids]])/(velocity[ends[ids]]+velocity[starts[ids]])*.5
                t=(((lo-centre*centre)/(yl+centre))[:,None]+displacement)/half[:,None]
                mass=v*v*sigma*(ds[:,None]*weights*.5)
                power=np.ones_like(t)
                for j in range(_ORDER+1):
                    contribution=np.sum(mass*power,axis=1)
                    sums[:,j]+=np.bincount(ids,weights=contribution,minlength=len(groups))
                    power*=t
            for i,(first,last) in enumerate(groups):
                centre=(velocity[first]+velocity[last])*.5
                half=(x[last]-x[first])/(velocity[last]+velocity[first])*.5
                condition=1.+abs(centre)/max(half,np.finfo(float).tiny)
                provision=sums[i,0]*(1024.+8*(last-first))*core.EPS*condition
                rows.append([x[first],x[last],centre,half/h,offset+first,offset+last,provision,*sums[i]])
        offset+=len(x)-1
    if not rows:return None
    table=np.ascontiguousarray(rows,dtype=float)
    if not np.all(np.isfinite(table)):return None
    try:
        capsule,nbytes=core._native.prepare_blocks(curve._native_panels,table,h)
    except ValueError:
        return None
    if nbytes>options.max_work_bytes//16:return None
    return capsule,dict(source_blocks=len(rows),numeric_cache_bytes=nbytes)


def _eligible(curve,options):
    return (options.backend!='python' and core._native is not None and
        hasattr(core._native,'prepare_blocks') and curve._native_panels is not None and
        len(curve._panels)>=_DENSE_SOURCE and options.max_series_terms==80 and
        options.max_radius>=6 and options.max_work_bytes>=64*1024*1024)


def warm(curve,temperature,options,domain=None):
    """Prepare once for a dense construction; keep at most one thermal scale."""
    if temperature<=curve.source_temperature or not _eligible(curve,options):return
    scale=core.KB*(temperature-curve.source_temperature)/curve.context.atomic_weight_ratio
    if scale<=0 or not math.isfinite(scale):return
    h=math.sqrt(scale)
    if domain is not None:
        low=max(0.,math.sqrt(domain[0])-6*h)**2
        high=(math.sqrt(domain[1])+6*h)**2
        covered=np.searchsorted(curve._starts,high,side='right')-np.searchsorted(curve._ends,low,side='left')
        if covered<_DENSE_SOURCE:return
    _cache(curve,h,options,True)


def _cache(curve,h,options,create):
    key=(h,options.max_work_bytes)
    cache=curve._block_moment_cache
    if key in cache:return cache[key]
    if not create:return None
    with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
        prepared=_prepare(curve,h,options)
    cache.clear();cache[key]=prepared
    return prepared


def try_evaluate(curve,energies,h,options):
    if not _eligible(curve,options):return None
    cache=_cache(curve,h,options,len(energies)>=_MIN_DIRECT_QUERIES)
    if cache is None:return None
    capsule,metadata=cache
    if any(not math.isfinite(v) for v in curve._envelope):
        raise core.DopplerConvergenceError('unresolvable-scale','tail envelope overflow',energy=float(energies[0]))
    rows=core._native.block_values_cached(capsule,np.ascontiguousarray(energies))
    result=_BlockRows();computed=accepted=fallbacks=0
    for energy,(value,error,spans,components,used) in zip(energies,rows):
        computed+=used
        x=math.sqrt(energy);z=x/h
        if z<1 or not math.isfinite(z*z) or not math.isfinite(value) or not math.isfinite(error):
            result.append(core._point(curve,float(energy),h,options));fallbacks+=1;continue
        tail=core._tail(z,h,*curve._envelope,6)
        low,high=max(0.,x-6*h)**2,(x+6*h)**2
        first=int(np.searchsorted(curve._ends,low,side='left'))
        last=int(np.searchsorted(curve._starts,high,side='right'))
        parts=[value];estimates=[error];count=used;cursor=first
        try:
            for a,b in [*spans,(last,last)]:
                if a>cursor:
                    v,eta,component,n=core._native.integrate(curve._native_panels,cursor,a,
                        float(energy),h,max(-z,-6.),6.,options.max_series_terms,options.max_panels)
                    count+=n;parts.append(v);estimates.append(eta)
                    for owner,contribution in component.items():components[owner]=components.get(owner,0.)+contribution
                cursor=b
        except (ArithmeticError,OverflowError):
            result.append(core._point(curve,float(energy),h,options));fallbacks+=1;continue
        value=math.fsum(parts)
        error=math.fsum(estimates)+64*core.EPS*math.fsum(map(abs,parts))
        if (count>options.max_panels or value<0 or not math.isfinite(value) or not math.isfinite(error) or
            error>options.kernel_share*(options.atol+options.rtol*abs(value)) or
            tail>options.tail_share*(options.atol+options.rtol*max(abs(value)-tail,0.))):
            result.append(core._point(curve,float(energy),h,options));fallbacks+=1;continue
        result.append((value,error,tail,components,count,6));accepted+=used
    result.moment_diagnostics=dict(metadata,computed_blocks=computed,accepted_blocks=accepted,
                                  queries=len(energies),legacy_fallbacks=fallbacks)
    return result

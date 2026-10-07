"""Immutable RRR backgrounds and explicitly declared additive reaction graphs.

No ENDF/GNDS imports and no implicit sums inferred from numerical MT ranges.
The first tabulator covers resolved domains only; it does not claim material
coverage outside them.
"""
from dataclasses import dataclass
import numpy as np
from kika.algebra import discontinuities, evaluate
from .prepare import UnsupportedResonanceError, evaluate_region, region_mts
from .breit_wigner import evaluate_bw


@dataclass(frozen=True)
class BackgroundCurve:
    x: tuple[float,...]
    y: tuple[float,...]
    law: int

    def evaluate(self,e):
        """The curve under its law (:func:`kika.algebra.evaluate`), zero off it."""
        return evaluate(self.x,self.y,self.law,np.asarray(e,dtype=float))


def prepare_backgrounds(backgrounds):
    from kika.nuclear_data.model import XYs1d,Regions1d,Background,ResonancesWithBackground
    out={}
    for mt,form in (backgrounds or {}).items():
        if not isinstance(mt,int) or isinstance(mt,bool) or mt<=0:
            raise ValueError('background keys must be positive reaction MT integers')
        if isinstance(form,ResonancesWithBackground):
            if form.resonanceRegionHref is not None:
                raise UnsupportedResonanceError('resolve the resonance link before supplying its Background to the explicit RRR API')
            form=form.background
        if isinstance(form,Background):
            if form.unresolvedRegion is not None:
                raise UnsupportedResonanceError('URR background is outside this RRR tabulator')
            form=form.resolvedRegion
        if isinstance(form,XYs1d):curves=[form]
        elif isinstance(form,Regions1d):curves=form.function1ds
        else:raise UnsupportedResonanceError('background must contain XYs1d or Regions1d resolved data')
        if not curves:raise ValueError('empty background')
        snapshots=[]
        for curve in curves:
            if not isinstance(curve,XYs1d):raise UnsupportedResonanceError('background region must be XYs1d')
            if curve.axes is None or curve.domainUnit!='eV' or curve.rangeUnit!='b':
                raise UnsupportedResonanceError('background requires explicit eV/b axes')
            x,y=curve.xs,curve.ys
            law=curve.endfInterpolationCode
            if (len(x)<2 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))
                    or np.any(np.diff(x)<0) or np.any(x<=0)):
                raise ValueError('background requires finite nondecreasing positive energies')
            if law not in (1,2,3,4,5):raise UnsupportedResonanceError('unsupported background interpolation law')
            if law in (4,5) and np.any(y<=0):raise ValueError('log-value background requires positive values')
            # ENDF uses duplicate abscissae for one-sided values at a jump.
            # Split them into independently owned regions, never deduplicate
            # a value or interpolate across that zero-width transition.
            cuts=np.r_[0,discontinuities(x)+1,len(x)]
            for start,stop in zip(cuts[:-1],cuts[1:]):
                xx,yy=x[start:stop],y[start:stop]
                if len(xx)<2:raise UnsupportedResonanceError('isolated repeated endpoint has no background interval')
                if snapshots and xx[0]<snapshots[-1].x[-1]:raise ValueError('overlapping background regions')
                snapshots.append(BackgroundCurve(tuple(xx),tuple(yy),law))
        out[mt]=tuple(snapshots)
    return out


def prepare_sums(sums,available):
    graph={mt:tuple(parts) for mt,parts in (sums or {}).items()}
    for mt,parts in graph.items():
        if not isinstance(mt,int) or isinstance(mt,bool) or mt<=0 or not parts:
            raise ValueError('sums require positive MT and nonempty components')
        if len(parts)!=len(set(parts)):raise ValueError('duplicate sum components')
        if any(not isinstance(p,int) or isinstance(p,bool) or p<=0 for p in parts):
            raise ValueError('sum components must be positive MT integers')
    order=[]
    leaves={}
    def visit(mt,active):
        if mt in active:raise ValueError('cycle in reaction sums')
        if mt in leaves:return leaves[mt]
        if mt not in graph:
            if mt not in available:raise ValueError(f'missing sum component MT{mt}')
            return {mt}
        covered=set()
        for part in graph[mt]:
            sub=visit(part,active|{mt})
            if covered & sub:raise ValueError('sum double counts a component through redundant branches')
            covered |= sub
        leaves[mt]=covered
        order.append(mt)
        return covered
    for mt in graph:visit(mt,set())
    return graph,tuple(order)


def evaluate_assembled(region,context,e,backgrounds,graph,order,block_size):
    e=np.asarray(e,dtype=float)
    mts=region_mts(region)|set(backgrounds)|set(graph)
    out={mt:np.zeros(len(e)) for mt in mts}
    for start in range(0,len(e),block_size):
        sl=slice(start,start+block_size)
        values=evaluate_region(e[sl],region,context)
        # Subtract each owned group's contribution separately: the same MT
        # can occur in other groups whose competition is not already in MF3.
        owned=[g for g in region.groups if g.competitive_in_background]
        for g in owned:
            if g.competitive_mt not in backgrounds or (1 not in backgrounds and 1 not in graph):
                raise ValueError('competitive background ownership requires its partial and a total background or explicit total sum')
            extra=evaluate_bw(e[sl],(g,),region.approximation,context)[g.competitive_mt]
            values[g.competitive_mt]-=extra
            values[1]-=extra
        for mt,value in values.items():out[mt][sl]=value
    for mt,curves in backgrounds.items():
        found=np.zeros(len(e),dtype=bool)
        for curve in curves:
            select=(e>=curve.x[0])&(e<=curve.x[-1])
            # Shared boundaries are handled before this function by selecting
            # a single curve per segment. Overlapping ownership is an error.
            if np.any(found&select):raise ValueError('ambiguous background boundary ownership')
            out[mt][select]+=curve.evaluate(e[select])
            found |= select
        if not np.all(found):raise ValueError(f'background MT{mt} does not cover requested RRR segment')
    for mt in order:out[mt]=sum((out[p] for p in graph[mt]),np.zeros(len(e)))
    if any(np.any(~np.isfinite(v)) for v in out.values()):raise FloatingPointError('nonfinite assembled cross section')
    return out

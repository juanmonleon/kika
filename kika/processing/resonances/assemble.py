"""Immutable resonance backgrounds and explicitly declared additive reaction graphs.

No ENDF/GNDS imports and no implicit sums inferred from numerical MT ranges.
The low-level tabulator covers prepared resonance domains; the suite facade
checks coverage of the complete modeled material.
"""
from dataclasses import dataclass,field
import numpy as np
from kika.algebra import prepare_evaluator,join_pieces,discontinuities,validate
from .prepare import UnsupportedResonanceError, evaluate_region, region_mts
from .breit_wigner import evaluate_bw


@dataclass(frozen=True)
class BackgroundCurve:
    x: tuple[float,...]
    y: tuple[float,...]
    law: int | tuple[int,...]
    _x: object=field(init=False,repr=False,compare=False)
    _y: object=field(init=False,repr=False,compare=False)
    _laws: object=field(init=False,repr=False,compare=False)
    _evaluate: object=field(init=False,repr=False,compare=False)

    def __post_init__(self):
        for name,value,dtype in (('_x',self.x,float),('_y',self.y,float),('_laws',self.law,int)):
            array=np.asarray(value,dtype=dtype).copy()
            array.setflags(write=False)
            object.__setattr__(self,name,array)
        object.__setattr__(self,'_evaluate',prepare_evaluator(self._x,self._y,self._laws))

    @property
    def breaks(self):
        x=self._x;laws=np.broadcast_to(self._laws,(len(x)-1,))
        return tuple(np.unique(np.r_[x[discontinuities(x)],x[1:][laws==1]]))

    @property
    def endpoint_jump(self):
        first=int(np.searchsorted(self._x,self.x[-1],side='left'))
        laws=np.broadcast_to(self._laws,(len(self.x)-1,))
        left=self.y[first-1] if laws[first-1]==1 else self.y[first]
        return left!=self.y[-1]

    def evaluate(self,e):
        """The curve under its law (:func:`kika.algebra.evaluate`), zero off it."""
        return self._evaluate(np.asarray(e,dtype=float))


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
            pieces=[v for v in (form.resolvedRegion,form.unresolvedRegion,form.fastRegion) if v is not None]
            children=[c for piece in pieces for c in (piece.function1ds if isinstance(piece,Regions1d) else [piece])]
            if not children:raise ValueError('empty background')
            children.sort(key=lambda c:c.domainMin)
            # A repeated right endpoint in resolvedRegion may own its next
            # interval in fastRegion. Join the whole background first.
            out[mt]=prepare_backgrounds({mt:Regions1d(children)})[mt]
            continue
        if isinstance(form,XYs1d):curves=[form]
        elif isinstance(form,Regions1d):curves=form.function1ds
        else:raise UnsupportedResonanceError('background must contain XYs1d or Regions1d resolved data')
        if not curves:raise ValueError('empty background')
        pieces=[]
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
            validate(x,y,law)
            if pieces and x[0]<pieces[-1][0][-1]:raise ValueError('overlapping background regions')
            pieces.append((x,y,law))
        # Join adjacent interpolation regions before locating steps. A step
        # at an INT boundary can end the preceding region with an otherwise
        # isolated point whose interval belongs to the following region.
        # Algebra retains both values and removes only identical shared points.
        clusters=[]
        for piece in pieces:
            if not clusters or clusters[-1][-1][0][-1]!=piece[0][0]:clusters.append([])
            clusters[-1].append(piece)
        snapshots=[]
        for cluster in clusters:
            x,y,laws=join_pieces(cluster)
            if x[-1]<=x[0]:raise ValueError('background must contain a positive-width domain')
            # Preserve the raw function and its interval laws. Algebra owns
            # repeated-node semantics, including first/final endpoint values;
            # a strict continuous-piece representation would discard them.
            snapshots.append(BackgroundCurve(tuple(x),tuple(y),tuple(int(v) for v in laws)))
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


def evaluate_assembled(region,context,e,backgrounds,graph,order,block_size,work_bytes=64*1024**2,*,absorption_rtol=0.):
    e=np.asarray(e,dtype=float)
    mts=region_mts(region)|set(backgrounds)|set(graph)
    out={mt:np.zeros(len(e)) for mt in mts}
    for start in range(0,len(e),block_size):
        sl=slice(start,start+block_size)
        values=evaluate_region(e[sl],region,context,work_bytes=work_bytes,absorption_rtol=absorption_rtol)
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

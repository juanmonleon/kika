"""Prepared-domain resonance tabulation, background assembly and V3 checks."""
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
import numpy as np

from .prepare import PreparedResonances, group_radii, group_knots, group_breaks, region_mts
from .assemble import prepare_backgrounds,prepare_sums,evaluate_assembled
from .grid import ReconstructionOptions,ReconstructionConvergenceError,linearize,error_ratio,VERIFICATION_FRACTIONS,verification_batches
from .channel_functions import neutral_channel_functions


@dataclass(frozen=True)
class _Segment:
    low: float
    high: float
    region: object
    backgrounds: object
    left_high: bool

    def evaluate(self,energy,context,graph,order,block_size,work_bytes=64*1024**2,*,absorption_rtol=0.):
        e=np.asarray(energy,dtype=float).copy()
        if self.left_high:e[e==self.high]=np.nextafter(self.high,self.low)
        return evaluate_assembled(self.region,context,e,self.backgrounds,graph,order,block_size,work_bytes,absorption_rtol=absorption_rtol)


@dataclass(frozen=True)
class ReconstructionResult:
    """RRR-only forms and a frozen reference; no automatic suite mutation.

    MT1 includes exactly the declared inputs or explicitly supplied sum graph.
    Coverage is restricted to the prepared resonance domains (RRR and/or URR).
    ``verify_forms`` must be called after serialization/reloading; rounding is
    part of the requested total budget and may cause an explicit failure.
    """
    forms: object
    report: object
    options: ReconstructionOptions
    _prepared: PreparedResonances
    _segments: tuple
    _graph: object
    _order: tuple
    _grids: tuple=()

    def evaluate(self,energies):
        """Assembled reference callable, preserving scalar/1D shape and order."""
        e=np.asarray(energies,dtype=float)
        if e.ndim>1 or np.any(~np.isfinite(e)) or np.any(e<=0):
            raise ValueError('energies must be finite positive scalar or 1D eV array')
        flat=e.reshape(-1)
        owner=np.full(len(flat),-1)
        for i,s in enumerate(self._segments):owner[(flat>=s.low)&(flat<=s.high)]=i
        if np.any(owner<0):raise ValueError('energies outside reconstructed domains, including gaps')
        out={mt:np.zeros(len(flat)) for mt in self.forms}
        for i,s in enumerate(self._segments):
            indices=np.flatnonzero(owner==i)
            if len(indices):
                values=s.evaluate(flat[indices],self._prepared.context,self._graph,self._order,self.options.block_size,self.options.max_work_bytes)
                for mt in out:out[mt][indices]=values.get(mt,np.zeros(len(indices)))
        return {mt:v.reshape(e.shape) for mt,v in out.items()}

    def verify_forms(self,forms):
        """Check reloaded model forms at original nodes and interior probes.

        Returns per-MT maximum mixed-error ratios (one means the full budget).
        Finite probes cannot establish a mathematical all-energy guarantee.
        """
        from kika.nuclear_data.model import Regions1d
        from kika.nuclear_data.model.functions.regions1d import curvesOnDomains
        from kika.algebra import prepare_evaluator
        if set(forms)!=set(self.forms):raise ValueError('reloaded reactions differ from reconstructed output')
        maxima={mt:0. for mt in forms}
        for index,s in enumerate(self._segments):
            exemplar=next(iter(self.forms.values()))
            grid=self._grids[index] if self._grids else (exemplar.function1ds[index].xs if isinstance(exemplar,Regions1d) else exemplar.xs)
            evaluators={}
            for mt,form in forms.items():
                curves=curvesOnDomains(form,[(seg.low,seg.high) for seg in self._segments])
                if curves is None:raise ValueError('serialized region structure changed')
                curve=curves[index]
                if curve.domainMin!=s.low or curve.domainMax!=s.high:
                    raise ReconstructionConvergenceError('serialization moved a region boundary')
                if curve.domainUnit!='eV' or curve.rangeUnit!='b':raise ValueError('reloaded form units must be eV/b')
                if curve.endfInterpolationCode!=2:
                    raise ReconstructionConvergenceError('serialization changed the linear interpolation law')
                if (np.any(~np.isfinite(curve.xs)) or np.any(~np.isfinite(curve.ys))
                        or np.any(np.diff(curve.xs)<0)):
                    raise ReconstructionConvergenceError('invalid reloaded grid or values')
                evaluators[mt]=prepare_evaluator(curve.xs,curve.ys,2)
            batch=max(1,min(32768,self.options.max_work_bytes//(64*(len(forms)+1))))
            for points in verification_batches(grid,batch):
                actual=s.evaluate(points,self._prepared.context,self._graph,self._order,
                    self.options.block_size,self.options.max_work_bytes)
                for mt,evaluator in evaluators.items():
                    value=np.asarray(evaluator(points))
                    if np.any(~np.isfinite(value)):raise ReconstructionConvergenceError('nonfinite reloaded form')
                    ratio=error_ratio(actual.get(mt,np.zeros(len(points))),value,self.options)
                    maxima[mt]=max(maxima[mt],float(np.max(ratio)))
        if any(v>1 for v in maxima.values()):
            raise ReconstructionConvergenceError(f'reloaded output exceeds total mixed-error budget: {maxima}')
        return maxima


def _segments(prepared,backgrounds):
    out=[]
    for region in prepared.regions:
        if region.low<=0:raise ValueError('tabulation requires strictly positive lower energy bounds')
        cuts={region.low,region.high}
        if region.unresolved is not None:cuts.update(region.unresolved.breaks)
        for curves in backgrounds.values():
            for curve in curves:
                cuts.update(x for x in (curve.x[0],curve.x[-1]) if region.low<x<region.high)
                cuts.update(x for x in curve.breaks if region.low<x<region.high)
        for group in region.groups:
            cuts.update(x for x in group_breaks(group) if region.low<x<region.high)
            for radius in group_radii(group):
                previous=1
                for nbt,law in radius.interpolation:
                    if law==1:cuts.update(x for x in radius.energies[previous:nbt] if region.low<x<region.high)
                    previous=nbt
        edges=sorted(cuts)
        for low,high in zip(edges[:-1],edges[1:]):
            local={}
            for mt,curves in backgrounds.items():
                candidates=[c for c in curves if c.x[0]<=low and c.x[-1]>=high]
                if len(candidates)!=1:raise ValueError(f'background MT{mt} does not uniquely cover RRR [{low}, {high}]')
                local[mt]=(candidates[0],)
            out.append(_Segment(low,high,region,MappingProxyType(local),high<region.high or
                any(r.low==high for r in prepared.regions if r is not region)))
    return tuple(out)


#: Geometric seeds of one segment: GEOMETRIC_SEEDS across it, but never
#: denser than GEOMETRIC_SEEDS per decade. A tape whose background changes
#: interpolation law every few hundred eV cuts the range into thousands of
#: segments (JENDL-5 Cl-35: 3 291 narrower than 1 %); a fixed count per
#: segment seeded each 100 eV one with 65 points, 205 000 seeds that the
#: refinement then verified panel by panel. The net only starts refinement:
#: levels, thresholds and source knots are seeded on their own, and every
#: final panel passes its probes.
GEOMETRIC_SEEDS=65


def _level_energies(group,cache):
    """A group's level energies, once per reconstruction when *cache* is given."""
    if cache is None:return np.array([lv.energy for lv in group.levels],dtype=float)
    entry=cache.get(id(group))
    if entry is None:entry=cache[id(group)]=(group,np.array([lv.energy for lv in group.levels],dtype=float))
    return entry[1]


#: Peak widths seeded either side of a level.
_WIDTH_MULTIPLES=np.geomspace(1/64,128,35)


def _cached(cache,key,make):
    """``make()``, once per reconstruction when *cache* is given."""
    if cache is None:return make()
    entry=cache.get(key)
    if entry is None:entry=cache[key]=make()
    return entry


def _center_iterates(group,region,cache):
    """The fixed-point centre estimates of a group's levels, every step.

    Row ``i`` is the estimate after ``i`` steps; a level steps while it stays
    inside *region* (and from a positive energy), and holds after. A segment
    stops a level at its first estimate outside the segment, which lies
    inside the region's run of estimates, so each segment reads its centres
    off these rows with the update it would have made itself.
    """
    def make():
        er=_level_energies(group,cache)
        reduced=np.asarray(group.reduced).reshape(len(er),len(group.channels))
        rows=[er.copy()];centers=er.copy()
        active=(er>0)&(centers>=region.low)&(centers<=region.high)
        for _ in range(8):
            if np.any(active):
                real=np.stack([c.functions(centers[active],logarithmic=True)[1].real for c in group.channels],axis=1)
                centers[active]=er[active]-np.sum(reduced[active]*reduced[active]*real,axis=1)
                active&=(centers>=region.low)&(centers<=region.high)
            rows.append(centers.copy())
        return np.array(rows)
    return _cached(cache,('centers',id(group),region.low,region.high),make)


def _fixed_seeds(group,context):
    """A group's seeds that no level places: its radius energies, its knots
    and its competitive threshold, sorted and finite. (A segment reads those
    in its closed span; its own ends are seeds anyway.)"""
    values=[float(x) for radius in group_radii(group) for x in radius.energies]+[float(x) for x in group_knots(group)]
    if group.competitive_mt is not None and group.competitive_q<0:
        values.append(-group.competitive_q*(1+context.atomic_weight_ratio)/context.atomic_weight_ratio)
    values=np.array(values,dtype=float)
    return np.unique(values[np.isfinite(values)])


def _level_seeds(group,region,cache):
    """Each level's seeds in a fixed-point group: the peak and the peak
    plus and minus its width times each multiple, at its energy
    (``peaks``) and at its last centre estimate (``centres``), with the
    lowest and highest of its estimates."""
    rows=_center_iterates(group,region,cache)
    energy=_level_energies(group,cache)
    width=np.array([lv.neutron+lv.capture+lv.fission+lv.competitive for lv in group.levels],dtype=float)[:,None]
    def around(peak):
        peak=peak[:,None]
        return np.concatenate([peak,peak-width*_WIDTH_MULTIPLES,peak+width*_WIDTH_MULTIPLES],axis=1)
    return dict(energy=energy,peaks=around(energy),centres=around(rows[-1]),
                lowest=rows.min(axis=0),highest=rows.max(axis=0))


def _region_seeds(region,context,cache):
    """The seed tables of every group of *region* joined, when every group
    seeds as a fixed-point group or has no levels; None otherwise."""
    fixed=[];tables=[]
    for g in region.groups:
        ctx=g.context or context
        fixed.append(_cached(cache,('fixed',id(g),id(ctx)),lambda:_fixed_seeds(g,ctx)))
        if not g.levels:continue
        if region.approximation!='RMatrixNeutral' or g.level_metric:return None
        tables.append(_cached(cache,('levels',id(g),region.low,region.high),lambda:_level_seeds(g,region,cache)))
    width=1+2*len(_WIDTH_MULTIPLES)
    joined={name:np.concatenate([t[name] for t in tables]) if tables else np.zeros((0,width) if name in ('peaks','centres') else 0)
            for name in ('energy','peaks','centres','lowest','highest')}
    joined['fixed']=np.unique(np.concatenate(fixed)) if fixed else np.zeros(0)
    return joined


def _seeds(segment,context,cache=None,knots=None):
    lo,hi=segment.low,segment.high
    count=GEOMETRIC_SEEDS
    if lo>0 and hi>lo:
        count=int(min(GEOMETRIC_SEEDS,max(2,1+math.ceil((GEOMETRIC_SEEDS-1)*math.log10(hi/lo)))))
    seeds=[lo,hi];parts=[np.geomspace(lo,hi,count)]
    if segment.region.unresolved is not None:
        seeds.extend(x for x in segment.region.unresolved.grid if lo<=x<=hi)
    if knots is not None:
        # Precomputed by the caller: the curves' knots strictly inside, and
        # whether one ends at hi with a jump there.
        inside,jump=knots
        parts.extend(inside)
        if not segment.left_high and jump:seeds.append(np.nextafter(hi,lo))
    else:
        for curves in segment.backgrounds.values():
            for c in curves:
                points=c._x if hasattr(c,'_x') else np.asarray(c.x,dtype=float)
                seeds.extend(points[np.searchsorted(points,lo,'right'):np.searchsorted(points,hi,'left')])
            if not segment.left_high and any(c.x[-1]==hi and c.endpoint_jump for c in curves):
                seeds.append(np.nextafter(hi,lo))
    table=_cached(cache,('region',id(segment.region),id(context)),lambda:_region_seeds(segment.region,context,cache))
    if table is not None:
        # Every group's fixed seeds and level seeds in one table (see the
        # loop below, which this is group by group).
        fixed,energy=table['fixed'],table['energy']
        parts.append(fixed[np.searchsorted(fixed,lo,'left'):np.searchsorted(fixed,hi,'right')])
        peak=(energy>0)&(energy>=lo)&(energy<=hi)
        parts.append(table['peaks'][peak].ravel())
        parts.append(table['centres'][peak&(table['lowest']>=lo)&(table['highest']<=hi)].ravel())
        seeds=np.concatenate([np.array(seeds,dtype=float)]+parts)
        return np.unique(seeds[np.isfinite(seeds)&(seeds>=lo)&(seeds<=hi)]).tolist()
    for g in segment.region.groups:
        ctx=g.context or context
        # The group's radius energies, knots and competitive threshold.
        fixed=_cached(cache,('fixed',id(g),id(ctx)),lambda:_fixed_seeds(g,ctx))
        parts.append(fixed[np.searchsorted(fixed,lo,'left'):np.searchsorted(fixed,hi,'right')])
        if segment.region.approximation=='RMatrixNeutral' and not g.level_metric and g.levels:
            # A level seeds through its peak inside [lo, hi], and through its
            # fixed-point centre (eight steps, stopped at the first estimate
            # outside [lo, hi]) when that centre is inside: when every
            # estimate is, and the centre is the last.
            table=_cached(cache,('levels',id(g),segment.region.low,segment.region.high),
                          lambda:_level_seeds(g,segment.region,cache))
            energy=table['energy']
            peak=(energy>0)&(energy>=lo)&(energy<=hi)
            parts.append(table['peaks'][peak].ravel())
            parts.append(table['centres'][peak&(table['lowest']>=lo)&(table['highest']<=hi)].ravel())
            continue
        iterations=0 if segment.region.approximation in ("ReichMoore","RMatrixNeutral") else 4
        energies=_level_energies(g,cache)
        # A level seeds a segment only through a peak inside it.
        inside=(energies>0)&(energies>=lo)&(energies<=hi)
        for level_index in np.flatnonzero(inside):
            level=g.levels[level_index]
            center=level.energy
            if iterations:
                pr,sr,_=neutral_channel_functions(g.l,np.sqrt(ctx.k_squared_per_ev*level.energy)*g.channel_radius.evaluate(level.energy))
            # Fixed-point seeds isolate shifted peaks; correctness still comes
            # from subsequent reference evaluations, not this estimate.
            for _ in range(iterations):
                if not lo<=center<=hi:break
                _,shift,_=neutral_channel_functions(g.l,np.sqrt(ctx.k_squared_per_ev*center)*g.channel_radius.evaluate(center))
                center=level.energy+.5*level.neutron*(sr-shift)/pr
            width=level.neutron+level.capture+level.fission+level.competitive
            for peak in (level.energy,center):
                if lo<=peak<=hi:
                    seeds.append(peak)
                    seeds.extend((peak-width*_WIDTH_MULTIPLES).tolist());seeds.extend((peak+width*_WIDTH_MULTIPLES).tolist())
    seeds=np.concatenate([np.array(seeds,dtype=float)]+parts)
    return np.unique(seeds[np.isfinite(seeds)&(seeds>=lo)&(seeds<=hi)]).tolist()


def tabulate_resonances(prepared,*,backgrounds=None,sums=None,options=None,label='recon'):
    """Tabulate supported resolved/dilute URR blocks on shared per-segment grids.

    Backgrounds map MT to canonical model XYs1d/Regions1d, Background or
    ResonancesWithBackground. All supplied functions must cover the requested
    resonance domains. Fast-region data are untouched and outside this result.
    ``sums`` is an explicit additive MT graph; aggregates replace their kernel
    values, and background values for a rebuilt sum are rejected as ambiguous.
    Only exactly constant spans are compacted; no approximate thinning,
    clipping, automatic tolerance relaxation or implicit attachment.
    """
    from kika.nuclear_data.model import Axis,Axes,XYs1d,Regions1d
    from kika.algebra import compress_flat
    if not isinstance(prepared,PreparedResonances):raise TypeError('expected PreparedResonances')
    if not prepared.regions:raise ValueError('no prepared regions to tabulate')
    options=ReconstructionOptions() if options is None else options
    if not isinstance(options,ReconstructionOptions):raise TypeError('expected ReconstructionOptions')
    if not isinstance(label,str) or not label:raise ValueError('nonempty output label required')
    background=prepare_backgrounds(backgrounds)
    available={mt for r in prepared.regions for mt in region_mts(r)}|set(background)
    graph,order=prepare_sums(sums,available)
    if set(graph)&set(background):raise ValueError('a rebuilt sum cannot also have an evaluated background')
    segments=_segments(prepared,background)
    tables=[]
    checks=[]
    points=0
    for s in segments:
        def evaluate(e):return s.evaluate(e,prepared.context,graph,order,options.block_size,options.max_work_bytes,
            absorption_rtol=min(1e-8,options.rtol*1e-5))
        x,y,check=linearize(evaluate,_seeds(s,prepared.context),options,options.max_points-points)
        points+=len(x)
        tables.append((x,y))
        checks.append(dict(domain=(s.low,s.high),points=len(x),**check))
    mts=sorted(available|set(graph))
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'crossSection','b')])
    forms={}
    for mt in mts:
        curves=[]
        for i,(x,y) in enumerate(tables):
            cx,cy,_=compress_flat(x,y.get(mt,np.zeros(len(x))),2)
            curves.append(XYs1d(cx,cy,axes=axes,index=i))
        forms[mt]=curves[0] if len(curves)==1 else Regions1d(curves,axes=axes,label=label)
        forms[mt].label=label
    fingerprint=hashlib.sha256(repr((prepared,background,graph,options)).encode()).hexdigest()
    report=dict(engine='kika-resonance-tabulator-1',scope='prepared resonance domains',normalized_sha256=fingerprint,
                context=prepared.context,options=options,regions=checks,points=points,
                stored_points=sum(len(c.xs) for f in forms.values() for c in (f.function1ds if isinstance(f,Regions1d) else [f])),
                units=dict(energy='eV',cross_section='b'),background_mts=tuple(sorted(background)),
                sum_graph=dict(graph),preparation_notes=prepared.preparation_notes,
                empirical_verification=True,global_error_bound=False,negative_policy='retain and report',
                minima={mt:min(float(np.min(t[1].get(mt,np.zeros(len(t[0]))))) for t in tables) for mt in mts})
    grids=tuple(np.frombuffer(x.tobytes(),dtype=x.dtype) for x,_ in tables)
    return ReconstructionResult(MappingProxyType(forms),MappingProxyType(report),options,prepared,segments,MappingProxyType(graph),order,grids)

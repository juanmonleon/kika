"""Immutable dilute URR preparation and evaluation on the canonical model.

ENDF cross-section interpolation is kept separate from parameter functions.
Wide panels (>3 in energy) receive at least ten logarithmic nodes per decade.
Competition affects the width denominator only; its cross section is in MF3.
"""
from dataclasses import dataclass,replace
import numpy as np
from kika.algebra import evaluate,discontinuities
from .breit_wigner import Group
from .radii import prepare_radius
from .channel_functions import neutral_channel_functions
from .fluctuations import width_products


@dataclass(frozen=True)
class Average:
    constant: float | None = None
    curves: tuple = ()

    @property
    def knots(self):return tuple(x for curve in self.curves for x in curve[0])

    @property
    def breaks(self):
        result=set()
        for i,(x,y,law) in enumerate(self.curves):
            if law==1:result.update(x[1:])
            if i and self.curves[i-1][1][-1]!=y[0]:result.add(x[0])
        return result

    def evaluate(self,energy):
        e=np.asarray(energy,dtype=float)
        if self.constant is not None:return np.full_like(e,self.constant)
        values=np.zeros_like(e);covered=np.zeros(e.shape,dtype=bool)
        for x,y,law in self.curves:
            mask=(e>=x[0])&(e<=x[-1]);values[mask]=evaluate(x,y,law,e[mask]);covered|=mask
        if not np.all(covered):raise ValueError('URR parameter function does not cover the evaluation domain')
        return values


def average(function,values,grid,constant,low,high,*,positive=False,reduced=False):
    from kika.nuclear_data.model import Constant1d,XYs1d,Regions1d
    from .prepare import UnsupportedResonanceError
    def check(v):
        a=np.asarray(v,dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a<=0 if positive else a<0):
            raise ValueError('URR level spacings must be positive and mean widths nonnegative')
    if function is not None:
        allowed_units=('eV','eV**(1/2)','eV**0.5') if reduced else ('eV',)
        if function.domainUnit!='eV' or function.rangeUnit not in allowed_units:
            raise UnsupportedResonanceError('URR parameter functions require normalized eV energy and width/spacing units')
        if isinstance(function,Constant1d):
            if function.domainMin>low or function.domainMax<high:raise ValueError('URR constant domain does not cover region')
            constant=float(function.constant)
        else:
            curves=function.function1ds if isinstance(function,Regions1d) else [function]
            result=[]
            for curve in curves:
                if not isinstance(curve,XYs1d):raise UnsupportedResonanceError('URR averages require constant1d, XYs1d or regions1d')
                if curve.domainUnit!='eV':raise ValueError('URR average energies must be eV')
                x=np.asarray(curve.xs);y=np.asarray(curve.ys);law=curve.endfInterpolationCode
                check(y)
                if len(x)<2 or np.any(~np.isfinite(x)) or np.any(x<=0) or np.any(np.diff(x)<0):raise ValueError('invalid URR parameter grid')
                if law not in (1,2,3,4,5):raise UnsupportedResonanceError('unknown URR parameter interpolation')
                if law in (4,5) and np.any(y<=0):raise ValueError('URR logarithmic parameters must be positive')
                cuts=np.r_[0,discontinuities(x)+1,len(x)]
                for a,b in zip(cuts[:-1],cuts[1:]):
                    if b-a<2:raise ValueError('isolated URR repeated endpoint')
                    result.append((tuple(x[a:b]),tuple(y[a:b]),law))
            if not result or result[0][0][0]>low or result[-1][0][-1]<high:raise ValueError('URR parameter grid does not cover region')
            if any(a[0][-1]!=b[0][0] for a,b in zip(result[:-1],result[1:])):raise ValueError('URR parameter regions overlap or have gaps')
            return Average(curves=tuple(result))
    if constant is None:
        if values is None:raise ValueError('missing URR average')
        values=np.asarray(values,dtype=float)
        if values.ndim!=1 or not len(values):raise ValueError('invalid URR average values')
        if len(values)==1:constant=float(values[0])
        else:
            x=np.asarray(grid,dtype=float)
            if x.ndim!=1 or x.shape!=values.shape or np.any(~np.isfinite(x)) or np.any(x<=0) or np.any(np.diff(x)<=0):raise ValueError('invalid URR average grid')
            if x[0]>low or x[-1]<high:raise ValueError('URR parameter grid does not cover region')
            check(values)
            return Average(curves=((tuple(x),tuple(values),2),))
    check(constant)
    return Average(constant=float(constant))


@dataclass(frozen=True)
class UnresolvedGroup(Group):
    spin: float = 0.
    spacing: Average | None = None
    averages: tuple = ()
    degrees: tuple = ()


@dataclass(frozen=True)
class UnresolvedData:
    self_shielding_only: bool = False
    grid: tuple = ()
    law: int | None = None
    quadrature: tuple = ()
    sigma_curves: tuple = ()


def _physics(energy,groups,diagnostics=None):
    e=np.asarray(energy,dtype=float);out={mt:np.zeros_like(e) for mt in (1,2,18,102)};potential=set()
    for group in groups:
        ctx=group.context;k2=ctx.k_squared_per_ev*e
        rho=np.sqrt(k2)*group.channel_radius.evaluate(e)
        p,_,_=neutral_channel_functions(group.l,rho)
        _,_,phi=neutral_channel_functions(group.l,np.sqrt(k2)*group.phase_radius.evaluate(e))
        spacing=group.spacing.evaluate(e)
        widths=np.stack([v.evaluate(e) for v in group.averages],axis=1)
        # ENDF GN0 is reduced, including the explicitly stated neutron nu.
        active=widths[:,0]>0
        if np.any(active&(p==0)):
            raise FloatingPointError('URR neutron penetrability underflows; no silent zero-width substitution')
        if np.any(active):
            widths[active,0]=np.exp(np.log(widths[active,0])+.5*np.log(e[active])
                +np.log(group.degrees[0])+np.log(p[active])-np.log(rho[active]))
        if np.any(active&((widths[:,0]==0)|~np.isfinite(widths[:,0]))):
            raise FloatingPointError('URR physical neutron width is outside floating-point range')
        products=np.stack([width_products(row,group.degrees,diagnostics=diagnostics) for row in widths])
        factor=2*np.pi**2*.01/k2*(2*group.spin+1)/(2*(2*ctx.target_spin+1))/spacing
        out[2]+=factor*(products[:,0]-2*widths[:,0]*np.sin(phi)**2)
        out[102]+=factor*products[:,1];out[18]+=factor*products[:,2]
        if group.l not in potential:
            out[2]+=4*np.pi*.01/k2*(2*group.l+1)*np.sin(phi)**2;potential.add(group.l)
    if any(np.any(~np.isfinite(v)) for v in out.values()):raise FloatingPointError('nonfinite dilute URR cross section')
    out[1]=out[2]+out[18]+out[102]
    return out


def prepare_unresolved(region,context,notes):
    from .prepare import PreparedRegion,UnsupportedResonanceError
    from kika.nuclear_data.model.enums import INTERPOLATION_TO_ENDF_INT
    low,high=float(region.domainMin),float(region.domainMax);source=region.tabulatedWidths
    if not np.isfinite(low+high) or not 0<low<high or region.domainUnit!='eV':raise ValueError('invalid URR eV domain')
    if source is None:raise UnsupportedResonanceError('missing tabulated URR widths')
    if source.selfShieldingOnly:
        notes.append('URR LSSF=1: dilute cross sections already in MF3; no additional mean or probability tables')
        return PreparedRegion(low,high,'Unresolved',(),UnresolvedData(True))
    if source.radiusUnit not in (None,'fm'):raise ValueError('URR radii must be normalized to fm')
    if source.PoPs is not None:
        particles=list(source.PoPs.particles.values())
        if len(particles)==1 and particles[0].spin is not None:
            spin=particles[0].spin
            if spin.unit!='hbar' or float(spin.value)!=context.target_spin:
                raise ValueError('URR target spin disagrees with the explicit context')
    policy=source.radiusPolicy
    if policy is None:raise UnsupportedResonanceError('URR radius policy is absent; declare radiusPolicy explicitly before calculating')
    phase=prepare_radius(policy.phaseRadius if policy.phaseRadius is not None else source.scatteringRadius)
    groups=[];seen=set();laws=set();grids=[];lcontexts={}
    for sg in source.spinGroups:
        l,j=sg.L,float(sg.J)
        if not isinstance(l,(int,np.integer)) or not 0<=l<=64 or not np.isfinite(j) or j<0 or not float(2*j).is_integer():raise ValueError('invalid URR L/J')
        allowed=any(abs(l-s)<=j<=l+s and float(l+s-j).is_integer() for s in {abs(context.target_spin-.5),context.target_spin+.5})
        if not allowed:raise ValueError(f'URR J={j:g} is incompatible with I={context.target_spin:g}, L={l}')
        if (l,j) in seen:raise ValueError('duplicate URR L/J group')
        seen.add((l,j));ctx=context if sg.atomicWeightRatio is None else replace(context,atomic_weight_ratio=sg.atomicWeightRatio)
        if l in lcontexts and lcontexts[l]!=ctx:raise UnsupportedResonanceError('URR groups of one L disagree on mass')
        lcontexts[l]=ctx
        channel=prepare_radius(ctx.mass_channel_radius_fm) if policy.channelMode=='mass' else phase if policy.channelMode=='phase' else prepare_radius(policy.channelRadius) if policy.channelMode=='constant' else None
        if channel is None:raise ValueError('unknown URR channel radius policy')
        for r in (phase,channel):r.evaluate([low,high])
        spacing=average(sg.levelSpacingFunction,sg.levelSpacing,sg.levelSpacingEnergies if sg.levelSpacingEnergies is not None else source.energyGrid,None,low,high,positive=True)
        by_label={}
        labels={'neutron':'neutron','elastic':'neutron','capture':'capture','fission':'fission','competitive':'competitive'}
        links={rr.label:rr.reactionMT for rr in source.resonanceReactions}
        mt_labels={2:'neutron',102:'capture',18:'fission',19:'fission'}
        for c in sg.channels:
            mt=links.get(c.label)
            name=labels.get(c.label,mt_labels.get(mt,'competitive' if mt is not None and mt>0 else None))
            if name is None:raise UnsupportedResonanceError(f'unidentified URR width channel {c.label!r}')
            if mt is not None and name!=mt_labels.get(mt,'competitive'):
                raise ValueError('URR width label/MT mismatch')
            if name in by_label:raise ValueError('duplicate URR width channel')
            if not np.isfinite(c.degreesOfFreedom) or c.degreesOfFreedom<0:raise ValueError('URR degrees of freedom must be real and nonnegative')
            f=average(c.averageFunction,c.widths,c.energies if c.energies is not None else source.energyGrid,c.constantWidth,low,high,reduced=name=='neutron')
            by_label[name]=(f,float(c.degreesOfFreedom))
        if 'neutron' not in by_label:raise ValueError('URR group needs a reduced neutron width')
        pairs=[by_label.get(name,(Average(0.),0.)) for name in ('neutron','capture','fission','competitive')]
        if pairs[0][1]==0 and any(v>0 for v in (pairs[0][0].constant,) if v is not None):
            raise UnsupportedResonanceError('nonzero reduced GN0 requires positive neutron nu; fixed physical widths are not reduced GN0')
        if pairs[0][1]==0 and any(v>0 for curve in pairs[0][0].curves for v in curve[1]):
            raise UnsupportedResonanceError('nonzero reduced GN0 requires positive neutron nu; fixed physical widths are not reduced GN0')
        groups.append(UnresolvedGroup(l,channel,phase,(),context=ctx,spin=j,spacing=spacing,averages=tuple(p[0] for p in pairs),degrees=tuple(p[1] for p in pairs)))
        law=None if sg.crossSectionInterpolation is None else INTERPOLATION_TO_ENDF_INT.get(sg.crossSectionInterpolation)
        if law not in (None,2,5):raise UnsupportedResonanceError('URR cross-section interpolation supports INT2/5')
        laws.add(law)
        if law is not None:
            grid=sg.levelSpacingEnergies if sg.levelSpacingEnergies is not None else source.energyGrid
            if grid is None:raise ValueError('URR cross-section interpolation requires a declared energy grid')
            grids.append(tuple(map(float,grid)))
    if not groups:raise ValueError('URR needs spin groups')
    if len(laws)!=1 or grids and len(set(grids))!=1:raise UnsupportedResonanceError('mixed URR sigma laws or grids need an explicit aggregate interpolation convention')
    law=next(iter(laws));data=UnresolvedData()
    if law is not None:
        grid=np.asarray(grids[0]);
        if len(grid)<2 or grid[0]!=low or grid[-1]!=high or np.any(np.diff(grid)<=0):raise ValueError('URR sigma grid must span its domain in increasing order')
        nodes=set(grid)
        for a,b in zip(grid[:-1],grid[1:]):
            if b/a>3:nodes.update(np.geomspace(a,b,int(np.ceil(10*np.log10(b/a)))+1))
        # An explicit function step retains both one-sided values. Sigma
        # interpolation never bridges a jump in widths or phase radius.
        breaks=set()
        for group in groups:
            for value in (group.spacing,)+group.averages:breaks.update(value.breaks)
            for radius in (group.channel_radius,group.phase_radius):
                previous=1
                for nbt,int_code in radius.interpolation:
                    if int_code==1:breaks.update(radius.energies[previous:nbt])
                    previous=nbt
        breaks={x for x in breaks if low<x<high}
        nodes.update(breaks);grid=np.asarray(sorted(nodes));diagnostics={}
        edges=sorted({low,high}|breaks);curves={mt:[] for mt in (2,18,102)}
        for a,b in zip(edges[:-1],edges[1:]):
            x=grid[(grid>=a)&(grid<=b)];energies=x.copy()
            if b<high:energies[-1]=np.nextafter(b,a)
            values=_physics(energies,groups,diagnostics)
            for mt in curves:
                y=values[mt]
                if law==5 and np.any(y<=0) and not np.all(y==0):raise ValueError('logarithmic URR sigma law requires positive nonzero partials')
                curves[mt].append((tuple(x),tuple(y),law))
        data=UnresolvedData(grid=tuple(grid),law=law,quadrature=tuple(diagnostics.items()),
            sigma_curves=tuple(tuple(curves[mt]) for mt in (2,18,102)))
        notes.append(f'URR sigma INT={law}; {len(grid)} nodes; wide-panel parameter interpolation then sigma interpolation')
    else:notes.append('URR evaluates physical energy dependence using each canonical parameter function')
    notes.append('URR dilute mean; real nu; fixed nu=0; competition only in total width')
    return PreparedRegion(low,high,'Unresolved',tuple(groups),data)


def evaluate_unresolved(energies,region,diagnostics=None):
    data=region.unresolved
    if diagnostics is not None:
        for key,value in data.quadrature:
            diagnostics[key]=max(diagnostics.get(key,0),value)
    if data.self_shielding_only:return {mt:np.zeros_like(energies) for mt in (1,2,18,102)}
    if data.law is None:return _physics(energies,region.groups,diagnostics)
    out={}
    for mt,curves in zip((2,18,102),data.sigma_curves):
        values=np.zeros_like(energies)
        for x,y,law in curves:
            mask=(energies>=x[0])&(energies<=x[-1])
            if not all(v==0 for v in y):values[mask]=evaluate(x,y,law,energies[mask])
            else:values[mask]=0.
        out[mt]=values
    out[1]=out[2]+out[18]+out[102];return out


def normalize_unresolved_links(suite,resonances,source_style):
    """Resolve canonical local reaction identities on a copy, never from a format."""
    from copy import deepcopy
    from .suite import _entries,_target
    if resonances is None or resonances.unresolved is None:return resonances
    source=resonances.unresolved.tabulatedWidths
    if source is None or source.selfShieldingOnly or not source.resonanceReactions:return resonances
    result=deepcopy(resonances);entries=_entries(suite)
    for rr in result.unresolved.tabulatedWidths.resonanceReactions:
        if rr.href is None:
            if rr.reactionMT is None:raise ValueError('URR reaction requires an MT or a local reaction link')
            continue
        href=rr.href.rstrip('/')
        if not href.endswith('/crossSection'):href+='/crossSection'
        key,_=_target(href,'/reactionSuite/resonances/unresolved/tabulatedWidths',entries,source_style)
        mt=entries[key].ENDF_MT
        if mt is None:raise ValueError('URR linked reaction requires an explicit reaction MT')
        if rr.reactionMT is not None and rr.reactionMT!=mt:raise ValueError('URR reaction MT/link mismatch')
        rr.reactionMT=mt
    return result

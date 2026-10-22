"""Numerical LTY=0 diagnostics: energy-dependent linear covariance propagation.

ENDF-102 sections 33.2.2.1 and 33.3.2: C_derived = A C_NI A.T, in
absolute units, including cross-reaction blocks. NI components of a derived
quantity remain additive residual components. No clipping or PSD repair occurs.
"""
from __future__ import annotations

import numpy as np

from ..classes.mf33.mf33 import MF33MT, bin_average_xs
from .findings import NOTE, CovarianceFinding, CovarianceLocation

class NotEvaluable(ValueError):
    """Missing or invalid input prevents a complete NC calculation."""


def valid_lty0(rule):
    """Parallel lists, finite scalars and integral positive MTs before any traversal."""
    try:
        numbers = np.asarray([rule.e1,rule.e2,*rule.ci,*rule.xmti],dtype=float)
        return (len(rule.ci)>0 and len(rule.ci)==len(rule.xmti)==rule.nci
                and bool(np.all(np.isfinite(numbers))) and 0<=rule.e1<rule.e2
                and all(float(m)==int(m) and m>0 for m in rule.xmti))
    except (ValueError,TypeError,OverflowError):
        return False


class NCAssembler:
    def __init__(self, ctx, mf, sections, usable):
        from .covariances import _mat1

        self.ctx, self.mf = ctx, mf
        self.sections, self.usable = sections, usable
        self._central_cache = {}
        self.rules = {}
        for mt, sec in sections.items():
            self.rules[mt] = [n for s in sec.subsections
                              if _mat1(s, sec._mat) == 0 and int(s.mt1) == mt
                              for n in s.nc_records]

    def dependencies(self, mt, seen=None):
        from .covariances import _mat1, _overlap

        seen = set() if seen is None else seen
        if mt in seen:
            return seen
        seen.add(mt)
        sec = self.sections.get(mt)
        if sec is None or sec._mtl:
            raise NotEvaluable(f"MT{mt} has no independent covariance section")
        subs = [s for s in sec.subsections if _mat1(s, sec._mat) == 0 and int(s.mt1) == mt]
        if len(subs) != 1:
            raise NotEvaluable(f"MT{mt} needs exactly one self subsection")
        if any(s.nc_records for s in sec.subsections if s is not subs[0]):
            raise NotEvaluable(f"MT{mt} has NC outside its local self subsection")
        rules = self.rules[mt]
        if not rules and not subs[0].ni_records:
            raise NotEvaluable(f"MT{mt} has no self covariance data")
        for i, rule in enumerate(rules):
            if rule.lty != 0:
                raise NotEvaluable(f"MT{mt}: LTY={rule.lty} needs another-material standard")
            if not valid_lty0(rule):
                raise NotEvaluable(f"MT{mt} has an invalid NC range or coefficient/target list")
            if any(_overlap(rule.e1, rule.e2, n.e1, n.e2) for n in rules[:i]):
                raise NotEvaluable(f"MT{mt} has overlapping NC rules")
            for target in rule.xmti:
                self.dependencies(int(target), seen)
        return seen

    def grid(self, nodes):
        from .covariances import _mat1, _grids

        edges = set()
        for mt in nodes:
            for n in self.rules[mt]:
                edges.update((n.e1, n.e2))
            sec = self.sections[mt]
            for s in sec.subsections:
                if _mat1(s, sec._mat) == 0 and int(s.mt1) in nodes:
                    for r in s.ni_records:
                        for g in _grids(r):
                            edges.update(g)
        grid = np.array(sorted(edges), dtype=float)
        if grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise NotEvaluable("invalid or empty union grid")
        return grid

    def connected(self, mt):
        """Include implied correlations with evaluated and other derived quantities."""
        from .covariances import _mat1

        nodes = self.dependencies(mt)
        while True:
            added = set()
            for candidate,sec in self.sections.items():
                if candidate in nodes or sec._mtl:
                    continue
                outgoing = {int(s.mt1) for s in sec.subsections
                            if _mat1(s,sec._mat)==0 and s.ni_records}
                outgoing.update(int(t) for r in self.rules[candidate] if valid_lty0(r)
                                for t in r.xmti)
                incoming = any(int(s.mt1)==candidate and s.ni_records
                               and _mat1(s,self.sections[m]._mat)==0
                               for m in nodes for s in self.sections[m].subsections)
                if outgoing & nodes or incoming:
                    added.update(self.dependencies(candidate))
            if added <= nodes:
                return nodes
            nodes |= added

    def weights(self, mt, grid):
        """Residual NI identities plus NC terms, restricted on each energy axis."""
        result = {}

        def visit(m, weight, stack):
            if not np.any(weight):
                return
            if m in stack:
                raise NotEvaluable(f"NC cycle involving MT{m} in an active energy range")
            result[m] = result.get(m, np.zeros(len(grid)-1)) + weight
            for rule in self.rules[m]:
                mask = (grid[:-1] >= rule.e1) & (grid[1:] <= rule.e2)
                for c, target in zip(rule.ci, rule.xmti):
                    visit(int(target), weight * mask * c, stack | {m})

        visit(mt, np.ones(len(grid)-1), set())
        return {m:w for m,w in result.items() if np.any(w)}

    def central(self, mt, grid, active):
        """Use reconstructed MF3 or trustworthy above-resonance data; never backgrounds."""
        key = (mt,grid.tobytes(),active.tobytes())
        if key in self._central_cache:
            return self._central_cache[key]
        if self.mf == 31:
            source = self.ctx.mf1.get(mt)
            if source is None:
                raise NotEvaluable(f"MF1 MT{mt} nu-bar is missing")
            if source.lnu == 2:
                from ...algebra import interval_laws
                y = bin_average_xs(source.energies, source.nubar_values, grid,
                                   interval_laws(len(source.energies), source.interpolation))
            elif source.lnu == 1:
                lo, hi = grid[:-1], grid[1:]
                y = np.full(len(lo), float(source.coefficients[0]))
                for k,c in enumerate(source.coefficients[1:],1):
                    log = np.log(np.divide(hi,lo,out=np.ones_like(hi),where=lo>0))
                    term = np.divide(hi**k-lo**k,k*log,out=(hi**(k+1)-lo**(k+1))/((k+1)*(hi-lo)),where=lo>0)
                    y += c*term
            else:
                raise NotEvaluable(f"MF1 MT{mt} nu-bar representation is unavailable")
        else:
            members = [m for m,s in self.sections.items() if s._mtl == mt]
            if 851 <= mt <= 870:
                if not members:
                    raise NotEvaluable(f"lump MT{mt} has no declared members")
                y = sum(self.central(m,grid,active) for m in members)
            else:
                entry = self.ctx.cross_section(mt)
                if entry is None:
                    raise NotEvaluable(f"MT{mt} central cross section unavailable; supply xs_sections")
                source, floor = entry
                if np.any(active & (grid[:-1] < floor)):
                    raise NotEvaluable(f"MT{mt} needs reconstructed cross sections below {floor:g} eV; supply xs_sections")
                y = MF33MT._bin_average_xs(source,list(grid))
        if not np.all(np.isfinite(y)):
            raise NotEvaluable(f"MT{mt} has nonfinite central values")
        if len(self._central_cache) >= 128:
            self._central_cache.clear()
        self._central_cache[key] = y
        return y

    def primitive(self, a, b, grid, active_a, active_b):
        from .covariances import _mat1, _expected_nt

        # Prefer a declared orientation; an absent cross block is zero by
        # ENDF-102 33.3.2 a.1. A missing self section is never treated as zero.
        chosen = None
        for row,col,transpose in ((a,b,False),(b,a,True)):
            sec = self.sections[row]
            subs = [s for s in sec.subsections if _mat1(s,sec._mat)==0 and int(s.mt1)==col]
            if subs:
                if len(subs) != 1:
                    raise NotEvaluable(f"duplicate subsection MT{row}xMT{col}")
                chosen = (sec,subs[0],transpose,row,col)
                break
        if chosen is None:
            return None
        sec,sub,transpose,row,col = chosen
        good = self.usable.get((row,0,col),[])
        if len(good) != len(sub.ni_records):
            raise NotEvaluable(f"MT{row}xMT{col} contains an invalid NI component")
        total = None
        for _,rec in good:
            if _expected_nt(rec) != rec.nt:
                raise NotEvaluable(f"MT{row}xMT{col} has an inconsistent NI count")
            bundle = sec._process_ni_records_to_matrix([rec],f"MT{row}xMT{col}",target_grid=list(grid))
            if bundle is None:
                raise NotEvaluable(f"MT{row}xMT{col} NI could not be assembled")
            matrix,_,relative = bundle
            if transpose:
                matrix = matrix.T
            if not np.all(np.isfinite(matrix)):
                raise NotEvaluable(f"MT{row}xMT{col} has nonfinite covariances")
            if relative and np.any(matrix):
                # Only rows/columns contributing to this block require sigma.
                ar = active_a & np.any(matrix[:,active_b] != 0,axis=1)
                ac = active_b & np.any(matrix[active_a,:] != 0,axis=0)
                if not np.any(ar) or not np.any(ac):
                    continue
                sr,sc = self.central(a,grid,ar),self.central(b,grid,ac)
                matrix = matrix * sr[:,None] * sc[None,:]
            total = matrix.copy() if total is None else total + matrix
        return total

    def block(self, a, b, grid):
        wa,wb = self.weights(a,grid),self.weights(b,grid)
        total = np.zeros((len(grid)-1,len(grid)-1))
        for i,ci in wa.items():
            for j,cj in wb.items():
                matrix = self.primitive(i,j,grid,ci!=0,cj!=0)
                if matrix is not None:
                    total += ci[:,None] * matrix * cj[None,:]
        if not np.all(np.isfinite(total)):
            raise NotEvaluable("nonfinite propagated covariance")
        return total


def check_nc(ctx, mf, sections, usable, out):
    """Run ordinary numerical checks on propagated self and implied cross blocks."""
    from .covariances import _check_self_block, _check_cross_block, _joint_psd, _mf33_magnitude

    engine = NCAssembler(ctx,mf,sections,usable)
    checked_joint = {}
    for mt in sorted(sections):
        if not any(n.lty==0 for s in sections[mt].subsections for n in s.nc_records):
            continue
        loc = CovarianceLocation(mat=sections[mt]._mat,mf=mf,mt=mt,mat1=0,mt1=mt)
        try:
            nodes = sorted(engine.connected(mt))
            grid = engine.grid(nodes)
            matrix = engine.block(mt,mt,grid)
            _check_self_block(matrix,grid,loc,out,[],None)
            _mf33_magnitude(ctx,mt,matrix,grid,False,loc,out)
            n = len(grid)-1
            joint_key = (tuple(nodes),grid.tobytes())
            new_joint = joint_key not in checked_joint
            selfs = {mt:matrix} if new_joint else {}
            cross = {}
            for other in nodes:
                if other == mt:
                    continue
                own = engine.block(other,other,grid)
                pair = engine.block(mt,other,grid)
                crossloc = CovarianceLocation(mat=loc.mat,mf=mf,mt=mt,mat1=0,mt1=other)
                _check_cross_block(pair,np.diag(matrix),np.diag(own),grid,crossloc,out,None)
                if new_joint:
                    selfs[other] = own
                    cross[min(mt,other),max(mt,other)] = pair if mt<other else pair.T
            if new_joint:
                # Fill every constituent cross block: omitting them would invent
                # a joint matrix inconsistent with the propagated derived row.
                for i,a in enumerate(nodes):
                    for b in nodes[i+1:]:
                        if (a,b) not in cross:
                            cross[a,b] = engine.block(a,b,grid)
                if any(np.any(v) for v in [*selfs.values(),*cross.values()]):
                    _joint_psd(nodes,{m:grid for m in nodes},selfs,cross,
                               lambda ms:", ".join(f"MT{m}" for m in ms),lambda ms:loc,out)
                checked_joint[joint_key] = mt
            out.append(CovarianceFinding("nc_matrix_evaluated",NOTE,
                "LTY=0 covariance assembled in absolute units, including NI and cross-reaction terms; "
                "variance, symmetry, correlation, self PSD and joint PSD checks executed without clipping",
                loc,{"n_bins":n,"energy_range":[float(grid[0]),float(grid[-1])],
                     "constituents":nodes,"joint_evaluated":True,
                     "joint_check_mt":checked_joint[joint_key],"joint_n_rows":n*len(nodes)}))
        except (ValueError,IndexError,np.linalg.LinAlgError) as exc:
            out.append(CovarianceFinding("nc_matrix_not_evaluated",NOTE,
                f"NC numerical coverage is incomplete: {exc}. Explicit NI checks remain separate.",
                loc,{"reason":str(exc)}))

"""Immutable Brune level metric, PRC 66 044611 (2002), equations 20–21.

The alternative level-space absorption is retained in that basis. In
particular, diagonal eliminated widths must not be silently diagonalized
again after a nonorthogonal change to formal parameters.
"""
from dataclasses import replace
import numpy as np


def prepare_brune(group):
    energies=np.asarray([level.energy for level in group.levels],dtype=float)
    n=len(energies)
    if not n:return group
    gamma=np.asarray(group.reduced)
    origin=float(energies[0]);centered=energies-origin
    shifts=np.stack([ch.functions(energies,logarithmic=True)[1].real for ch in group.channels],axis=1)
    metric=np.eye(n);energy_matrix=np.diag(centered+np.sum(gamma*gamma*shifts,axis=1))
    for i in range(n):
        for j in range(i):
            gap=energies[i]-energies[j]
            if gap==0:
                raise ValueError('degenerate Brune alternative energies do not define independent level coordinates')
            # Difference quotients belong to the coordinate transformation,
            # not a finite-difference approximation to the channel physics.
            m=-np.sum(gamma[i]*gamma[j]*(shifts[i]-shifts[j]))/gap
            value=.5*(centered[i]+centered[j])*m+.5*np.sum(gamma[i]*gamma[j]*(shifts[i]+shifts[j]))
            metric[i,j]=metric[j,i]=m
            energy_matrix[i,j]=energy_matrix[j,i]=value
    if np.any(~np.isfinite(metric)) or np.any(~np.isfinite(energy_matrix)):
        raise FloatingPointError('nonfinite Brune level transformation')
    try:np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Brune level metric must be positive definite') from exc
    return replace(group,level_metric=tuple(map(tuple,metric)),level_energy=tuple(map(tuple,energy_matrix)),level_origin=origin)

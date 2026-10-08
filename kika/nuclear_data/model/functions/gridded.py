"""GNDS-2.1 §6.2.6: ``gridded2d`` and ``gridded3d``, a full array on explicit grids.

Added for the thermal scattering law (ENDF-coverage roadmap E4), whose two
tables are exactly this shape: the coherent-elastic ``S_table`` is S(E, T) on
the Bragg edges x the temperatures, and the incoherent-inelastic kernel is
S(alpha, beta, T). Neither is a stack of independent 1-d functions -- every
temperature shares the same Bragg edges, every beta the same alpha grid -- so
an ``XYs2d``/``XYs3d`` would repeat a grid per row and lose the statement that
it is shared. ENDF-6 itself writes it once (``TemperatureTable``).

**Axis order is GNDS's**: ``axes`` index 0 is the dependent quantity and the
highest index the outermost grid, and ``values`` is indexed outermost first,
so ``values[t, e]`` for a ``gridded2d`` over (temperature, energy_in) and
``values[t, b, a]`` for a ``gridded3d`` over (temperature, beta, alpha). That
is FUDGE's layout (``thermalNeutronScatteringLaw/coherentElastic.py``,
``incoherentInelastic.py``) and the schema's (``xData.xsd`` gridded*d: axes,
then one ``array``).

**No evaluation.** Each grid carries its own interpolation law, and what a
point between two temperatures *means* differs between the two TSL tables
(ENDF's LI, the Bragg staircase). Evaluating belongs to the physics that
reads them, not to the container.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..axes import Axes, Grid

__all__ = ["Gridded2d", "Gridded3d"]


class _Gridded:
    """What ``gridded2d`` and ``gridded3d`` share: the array, the axes, the grids."""

    ndim: int = 0

    def _check(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != self.ndim:
            raise ValueError(
                f"a gridded{self.ndim}d holds a {self.ndim}-d array, got shape "
                f"{self.values.shape}"
            )
        shape = tuple(int(np.asarray(g.values).size) for g in self.grids)
        if shape != self.values.shape:
            raise ValueError(
                f"the grids give shape {shape} (outermost first) and the array "
                f"is {self.values.shape}"
            )

    @property
    def grids(self) -> List[Grid]:
        """The independent grids, **outermost first** -- the order ``values`` is indexed in."""
        byIndex = {axis.index: axis for axis in self.axes.axes}
        grids = []
        for index in range(self.ndim, 0, -1):
            axis = byIndex.get(index)
            if not isinstance(axis, Grid) or axis.values is None:
                raise ValueError(
                    f"a gridded{self.ndim}d needs a grid with values at axis "
                    f"index {index}; got {axis!r}"
                )
            grids.append(axis)
        return grids

    @property
    def dependentAxis(self):
        return next(a for a in self.axes.axes if a.index == 0)


@dataclass
class Gridded2d(_Gridded):
    """§6.2.6. ``values[outer, inner]`` on two grids (axis indices 2 and 1)."""

    values: np.ndarray
    axes: Axes
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    ndim = 2

    def __post_init__(self) -> None:
        self._check()

    def __repr__(self) -> str:
        outer, inner = self.grids
        return (f"Gridded2d({outer.label} x {inner.label}: "
                f"{self.values.shape[0]} x {self.values.shape[1]})")


@dataclass
class Gridded3d(_Gridded):
    """§6.2.6. ``values[outer, middle, inner]`` on three grids (indices 3, 2, 1)."""

    values: np.ndarray
    axes: Axes
    label: Optional[str] = None
    ndim = 3

    def __post_init__(self) -> None:
        self._check()

    def __repr__(self) -> str:
        labels = " x ".join(g.label for g in self.grids)
        return f"Gridded3d({labels}: {'x'.join(str(n) for n in self.values.shape)})"

"""Data model for MCNP FMESH mesh tally results.

FMESH results never go to MCTAL. MCNP writes them either to the MESHTAL text
file (MCNP 5 to 6.3, ``out=col``/``cf``/``colsci``/``cfsci``) or, with
``out=xdmf`` in MCNP 6.3+, to ``/results/mesh_tally`` in the HDF5 runtape.
:func:`kika.mcnp.read_meshtal` reads both into the same :class:`MeshTally`.

A mesh tally is an array over ``(energy, time, i, j, k)``. ``i j k`` are
``x y z`` for a rectangular mesh and ``r z theta`` for a cylindrical one, as on
the FMESH card. Like :class:`kika.mcnp.mctal.Tally`, :attr:`MeshTally.result`
keeps the energy and time totals MCNP writes as an extra last bin, and the
labelled views separate them:

* :meth:`MeshTally.to_xarray` -- the binned values;
* :meth:`MeshTally.total` -- the total over energy and/or time.

Theta is in revolutions (0 to 1), the unit of the FMESH card and the MESHTAL
file, in both readers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: Spatial dimension names per FMESH geometry, in ``i j k`` order.
SPATIAL_DIMS: Dict[str, Tuple[str, str, str]] = {
    "rectangular": ("x", "y", "z"),
    "cylindrical": ("r", "z", "theta"),
}


@dataclass
class MeshTally:
    """One FMESH tally.

    :ivar id: Tally number.
    :ivar particle: Particle scored, as MCNP names it (``"neutron"``...).
    :ivar geometry: ``"rectangular"`` or ``"cylindrical"``.
    :ivar comment: FC card text (lines joined with newlines), ``""`` if none.
    :ivar notes: Modifier notes MCNP prints for the tally (e.g. that a dose
        function applies), verbatim.
    :ivar bounds: Bin edges of the three spatial axes, keyed by the names in
        :data:`SPATIAL_DIMS`, in ``i j k`` order. Theta in revolutions.
    :ivar energy_bounds: Energy bin edges (MeV), ``n_energy + 1`` values.
    :ivar time_bounds: Time bin edges (shakes), ``n_time + 1`` values.
    :ivar result: Values, shape ``(n_e, n_t, n_i, n_j, n_k)`` where ``n_e``
        and ``n_t`` include the total bin when :attr:`has_energy_total` /
        :attr:`has_time_total`.
    :ivar error: Relative errors, same shape.
    :ivar has_energy_total: The last energy index holds the energy total.
    :ivar has_time_total: The last time index holds the time total.
    :ivar origin: Mesh origin (cylindrical meshes; ``None`` when unknown).
    :ivar axis: Cylinder axis direction (cylindrical meshes).
    :ivar vec: Direction of theta = 0 (cylindrical meshes).
    :ivar attrs: Extra metadata the source provides (the HDF5 attributes and
        small datasets: multiplier, dose function, transformation...).
    """

    id: int
    particle: str = ""
    geometry: str = "rectangular"
    comment: str = ""
    notes: List[str] = field(default_factory=list)
    bounds: Dict[str, np.ndarray] = field(default_factory=dict)
    energy_bounds: np.ndarray = field(default_factory=lambda: np.array([0.0, 1e36]))
    time_bounds: np.ndarray = field(default_factory=lambda: np.array([-1e36, 1e36]))
    result: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0, 0, 0)))
    error: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0, 0, 0)))
    has_energy_total: bool = False
    has_time_total: bool = False
    origin: Optional[np.ndarray] = None
    axis: Optional[np.ndarray] = None
    vec: Optional[np.ndarray] = None
    attrs: Dict[str, object] = field(default_factory=dict)

    # -- shape --------------------------------------------------------------

    @property
    def spatial_dims(self) -> Tuple[str, str, str]:
        return SPATIAL_DIMS[self.geometry]

    @property
    def n_energy(self) -> int:
        """Number of energy bins, without the total."""
        return len(self.energy_bounds) - 1

    @property
    def n_time(self) -> int:
        """Number of time bins, without the total."""
        return len(self.time_bounds) - 1

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Number of mesh cells along ``i j k``."""
        return tuple(len(self.bounds[d]) - 1 for d in self.spatial_dims)

    def centers(self, dim: str) -> np.ndarray:
        e = self.bounds[dim]
        return 0.5 * (e[1:] + e[:-1])

    @property
    def volumes(self) -> np.ndarray:
        """Volume of each mesh cell (cm3), shape :attr:`shape`.

        Computed from the bounds, so it exists for every source; it is the
        ``Volume`` column MCNP writes with ``out=cf``.
        """
        a, b, c = (np.diff(self.bounds[d]) for d in self.spatial_dims)
        if self.geometry == "cylindrical":
            r = self.bounds["r"]
            a = np.pi * (r[1:] ** 2 - r[:-1] ** 2)  # full annulus; theta (rev) gives the fraction
        return a[:, None, None] * b[None, :, None] * c[None, None, :]

    # -- labelled views ---------------------------------------------------

    def to_xarray(self) -> xr.Dataset:
        """Binned values as a Dataset with ``result`` and ``error``.

        Dimensions are ``energy`` and ``time`` (only when there is more than
        one bin) followed by the three spatial ones. Spatial coordinates are
        the bin centres, with ``<dim>_low``/``<dim>_high`` edges; ``energy`` and
        ``time`` carry the upper bound, as MCNP prints them, plus ``_low``.
        Totals are left out; see :meth:`total`.
        """
        return self._view(())

    def total(self, *axes: str) -> xr.Dataset:
        """Total over ``"energy"`` and/or ``"time"``, as a Dataset.

        A single-bin axis is its own total, so ``total("energy")`` also works
        on a tally without energy bins.

        :raises ValueError: For an axis other than energy/time.
        """
        for a in axes:
            if a not in ("energy", "time"):
                raise ValueError(f"mesh tally {self.id}: totals exist for energy and time, not {a!r}")
        return self._view(tuple(axes))

    def to_dataframe(self) -> pd.DataFrame:
        """:meth:`to_xarray` flattened to one row per bin."""
        return self.to_xarray().to_dataframe().reset_index()

    def _view(self, totals: Tuple[str, ...]) -> xr.Dataset:
        index = []
        dims: List[str] = []
        coords: Dict[str, Tuple[str, np.ndarray]] = {}
        for name, bounds, has_total in (
            ("energy", self.energy_bounds, self.has_energy_total),
            ("time", self.time_bounds, self.has_time_total),
        ):
            n = len(bounds) - 1
            if name in totals:
                index.append(n if has_total else 0)
            elif n > 1:
                index.append(slice(0, n))
                dims.append(name)
                coords[name] = (name, np.asarray(bounds[1:], dtype=float))
                coords[f"{name}_low"] = (name, np.asarray(bounds[:-1], dtype=float))
            else:
                index.append(0)
        index.extend([slice(None)] * 3)
        for name in self.spatial_dims:
            dims.append(name)
            e = self.bounds[name]
            coords[name] = (name, self.centers(name))
            coords[f"{name}_low"] = (name, e[:-1])
            coords[f"{name}_high"] = (name, e[1:])
        data = {
            "result": (dims, self.result[tuple(index)]),
            "error": (dims, self.error[tuple(index)]),
        }
        attrs = {"tally": self.id, "particle": self.particle, "geometry": self.geometry,
                 "comment": self.comment}
        if totals:
            attrs["total_over"] = list(totals)
        return xr.Dataset(data, coords=coords, attrs=attrs)

    def __repr__(self) -> str:
        grid = "x".join(str(n) for n in self.shape)
        extra = []
        if self.n_energy > 1:
            extra.append(f"energy={self.n_energy}" + ("+total" if self.has_energy_total else ""))
        if self.n_time > 1:
            extra.append(f"time={self.n_time}" + ("+total" if self.has_time_total else ""))
        label = f" {self.comment.splitlines()[0]!r}" if self.comment else ""
        dims = ",".join(self.spatial_dims)
        return (f"<MeshTally {self.id} ({self.particle}, {self.geometry}){label}: "
                f"{dims}={grid}" + "".join(f", {e}" for e in extra) + ">")


@dataclass
class Meshtal:
    """FMESH results of one MCNP run.

    :ivar code: Code name as written (``mcnp``, ``mcnp6``...).
    :ivar version: Code version.
    :ivar build_date: Build date (``ld=`` of the MESHTAL header).
    :ivar probid: Date and time of the run (MESHTAL text only).
    :ivar title: Problem title (MESHTAL text only).
    :ivar nps: Histories used to normalise the tallies, if the source gives it.
    :ivar source: ``"text"`` (MESHTAL file) or ``"hdf5"`` (runtape).
    :ivar tallies: Mesh tallies keyed by number.
    """

    code: str = ""
    version: str = ""
    build_date: str = ""
    probid: str = ""
    title: str = ""
    nps: Optional[float] = None
    source: str = "text"
    tallies: Dict[int, MeshTally] = field(default_factory=dict)

    def __getitem__(self, number: int) -> MeshTally:
        try:
            return self.tallies[number]
        except KeyError:
            raise KeyError(f"no mesh tally {number}; available: {list(self.tallies)}") from None

    def __iter__(self) -> Iterator[MeshTally]:
        return iter(self.tallies.values())

    def __len__(self) -> int:
        return len(self.tallies)

    def __repr__(self) -> str:
        head = f"<Meshtal {self.code} {self.version} ({self.source}): {len(self.tallies)} mesh tallies"
        if self.nps is not None:
            head += f", nps={self.nps:g}"
        lines = [head + ">"]
        if self.title:
            lines.append(f"  {self.title}")
        lines.extend(f"  {t!r}" for t in self.tallies.values())
        return "\n".join(lines)

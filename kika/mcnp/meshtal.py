"""Data model for MCNP FMESH mesh tally results.

FMESH results never go to MCTAL. MCNP writes them either to the MESHTAL text
file (MCNP 5 to 6.3, ``out=col``/``cf``/``colsci``/``cfsci``) or, with
``out=xdmf`` in MCNP 6.3+, to ``/results/mesh_tally`` in the HDF5 runtape.
:func:`~kika.mcnp.parse_meshtal.read_meshtal` reads both into the same :class:`MeshTally`.

A mesh tally is an array over ``(energy, time, i, j, k)``. ``i j k`` are
``x y z`` for a rectangular mesh and ``r z theta`` for a cylindrical one, as on
the FMESH card. Like :class:`kika.mcnp.mctal.Tally`, :attr:`MeshTally.result`
keeps the energy and time totals MCNP writes as an extra last bin, and the
labelled views separate them:

* :meth:`MeshTally.to_xarray` -- the binned values;
* :meth:`MeshTally.total` -- the total over energy and/or time.

Theta is in revolutions, the unit of the FMESH card and the MESHTAL file. The
HDF5 reader converts a theta grid written in radians (maximum above 1) to
revolutions.
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

    :func:`~kika.mcnp.parse_meshtal.read_meshtal` returns them in a
    :class:`Meshtal`, keyed by number. Values and errors are stored as the
    file has them: MCNP writes a relative error (a fraction of the value, not an
    absolute error) and kika does not convert it.

    Attributes
    ----------
    id : int
        Tally number.
    particle : str
        Particle scored, as MCNP names it (``"neutron"``...).
    geometry : str
        ``"rectangular"`` or ``"cylindrical"``.
    comment : str
        FC card text (lines joined with newlines), ``""`` if none.
    notes : list of str
        Modifier notes MCNP prints for the tally (e.g. that a dose function
        applies), verbatim.
    bounds : dict of str to numpy.ndarray
        Bin edges of the three spatial axes, keyed by the names in
        :data:`SPATIAL_DIMS`, in ``i j k`` order. MCNP writes lengths in cm and
        theta in revolutions; kika does not convert them, except the theta grid
        of an HDF5 runtape, taken from radians to revolutions when its maximum
        exceeds 1.
    energy_bounds : numpy.ndarray
        Energy bin edges, ``n_energy + 1`` values. MCNP writes MeV; kika does
        not convert.
    time_bounds : numpy.ndarray
        Time bin edges, ``n_time + 1`` values. MCNP writes shakes; kika does
        not convert.
    result : numpy.ndarray
        Values, shape ``(n_e, n_t, n_i, n_j, n_k)``: energy, time, then the mesh
        cells along ``i j k`` (the order of :attr:`spatial_dims`, ``k``
        last). ``n_e`` and ``n_t`` include the total bin as one more last index
        when :attr:`has_energy_total` or :attr:`has_time_total` is set.
    error : numpy.ndarray
        Relative errors, same shape as *result*.
    has_energy_total, has_time_total : bool
        The last energy (time) index holds the total over the energy (time) bins.
    origin : numpy.ndarray or None
        Mesh origin (cylindrical meshes; None when the source does not give it).
    axis : numpy.ndarray or None
        Cylinder axis direction (cylindrical meshes).
    vec : numpy.ndarray or None
        Direction of theta = 0 (cylindrical meshes).
    attrs : dict
        Extra metadata the source provides. From an HDF5 runtape: the group's
        attributes and every dataset of the group that is not otherwise read. From a
        text file with a ``Volume`` column: ``"volume"``, the volume of each mesh
        cell with the shape of the mesh.
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
        """Names of the ``i j k`` axes: ``x y z`` or ``r z theta``, by geometry."""
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
        """Midpoints of the bins along one spatial axis.

        Parameters
        ----------
        dim
            One of :attr:`spatial_dims`.

        Returns
        -------
        numpy.ndarray
            One midpoint per mesh cell along *dim*, in the units of
            :attr:`bounds`.

        Raises
        ------
        KeyError
            If *dim* is not a key of :attr:`bounds`.
        """
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
        one bin) followed by the three spatial ones, in ``i j k`` order.
        Spatial coordinates are the bin centres, with ``<dim>_low``/``<dim>_high``
        edges; ``energy`` and ``time`` carry the upper bound, as MCNP prints them,
        plus ``_low``. Totals are left out; see :meth:`total`.
        """
        return self._view(())

    def total(self, *axes: str) -> xr.Dataset:
        """Total over ``"energy"`` and/or ``"time"``, as a Dataset.

        A single-bin axis is its own total, so ``total("energy")`` also works
        on a tally without energy bins. With several bins and no total bin
        written, it returns the first bin, not a sum.

        Parameters
        ----------
        *axes
            ``"energy"`` and/or ``"time"``. Each named axis is removed from the
            result; the other axes stay as in :meth:`to_xarray`. With no
            argument the result equals :meth:`to_xarray`.

        Returns
        -------
        xarray.Dataset
            ``result`` and ``error`` (relative) over the axes that remain, with the
            coordinates of :meth:`to_xarray` and the attribute ``total_over`` (the
            list of axes given) when there is one.

        Raises
        ------
        ValueError
            For an axis other than ``"energy"`` or ``"time"``.
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
    """FMESH results of one MCNP run, as returned by
    :func:`~kika.mcnp.parse_meshtal.read_meshtal`.

    Indexing with a tally number returns its :class:`MeshTally` (``KeyError`` if
    absent); iterating yields the mesh tallies and ``len`` counts them.

    Attributes
    ----------
    code : str
        Code name as written (``mcnp``, ``mcnp6``...).
    version : str
        Code version.
    build_date : str
        Build date (``ld=`` of the MESHTAL header).
    probid : str
        Date and time of the run (MESHTAL text only).
    title : str
        Problem title (MESHTAL text only).
    nps : float or None
        Histories used to normalise the tallies; None if the source does not
        give it.
    source : str
        ``"text"`` (MESHTAL file) or ``"hdf5"`` (runtape).
    tallies : dict of int to MeshTally
        Mesh tallies keyed by number.
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

"""Data model for MCNP MCTAL files.

A MCTAL tally is an eight-dimensional array. MCNP writes the bins in the fixed
order ``f d u s m c e t`` (cells/surfaces/detectors, flagged/direct, user,
segment, multiplier, cosine, energy, time), with ``t`` varying fastest, and a
``(result, relative error)`` pair per bin. :class:`Tally` keeps exactly that
array, **including** the "total" bins MCNP appends when a bin line carries the
``t`` suffix (``et``, ``ut``, ...), so nothing written in the file is lost.

The labelled views drop the totals and the unbinned axes:

* :meth:`Tally.to_xarray` returns the binned values, one named dimension per
  axis that has bins;
* :meth:`Tally.total` returns the total bin of one or more axes.

The layout follows the MCTAL description in the MCNP manuals and the reader in
LANL's ``mcnptools`` (BSD-3-Clause, Copyright 2022 Triad National Security,
LLC), which this module and :mod:`kika.mcnp.parse_mctal` port to Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: MCTAL bin axes in file order (``t`` varies fastest).
AXIS_KEYS: Tuple[str, ...] = ("f", "d", "u", "s", "m", "c", "e", "t")

#: Dimension name used for each axis in the labelled views. ``f`` is renamed
#: per tally type by :attr:`Tally.region_name`.
AXIS_NAMES: Dict[str, str] = {
    "f": "cell",
    "d": "flag",
    "u": "user",
    "s": "segment",
    "m": "multiplier",
    "c": "cosine",
    "r": "radiography",
    "e": "energy",
    "t": "time",
}

#: Lower edge of the first bin of the bounded axes. MCNP prints only upper
#: bounds; the first bin starts at these values.
_FIRST_LOWER = {"c": -1.0, "e": 0.0, "t": -np.inf}

#: MCNP6 particle designators in the order of the particle flag list that
#: follows a ``tally`` line with a negative particle type.
MCNP6_PARTICLES: Tuple[str, ...] = (
    "n", "p", "e", "|", "q", "u", "v", "f", "h", "l", "+", "-", "x", "y",
    "o", "!", "<", ">", "g", "/", "z", "k", "%", "^", "b", "_", "~", "c",
    "w", "@", "d", "t", "s", "a", "*", "?", "#",
)

#: Bit flags of a positive particle type (MCNP5 and earlier).
_MCNP5_PARTICLE_BITS = ((1, "n"), (2, "p"), (4, "e"))

TALLY_KINDS: Dict[int, str] = {
    1: "surface current",
    2: "surface flux",
    4: "cell flux",
    5: "point detector",
    6: "energy deposition",
    7: "fission energy deposition",
    8: "pulse height",
}

#: Column names of a KCODE cycle record, in file order (MCNP manual, MCTAL
#: format; same enumeration as ``mcnptools.MctalKcode``). Records longer than
#: this keep their extra columns under positional names.
KCODE_COLUMNS: Tuple[str, ...] = (
    "keff_collision",
    "keff_absorption",
    "keff_track_length",
    "prompt_removal_lifetime_collision",
    "prompt_removal_lifetime_absorption",
    "avg_keff_collision",
    "avg_keff_collision_std",
    "avg_keff_absorption",
    "avg_keff_absorption_std",
    "avg_keff_track_length",
    "avg_keff_track_length_std",
    "avg_keff_combined",
    "avg_keff_combined_std",
    "avg_keff_combined_bias_corrected",
    "avg_keff_combined_bias_corrected_std",
    "prompt_removal_lifetime_combined",
    "prompt_removal_lifetime_combined_std",
    "cycle_histories",
    "avg_keff_combined_fom",
)


@dataclass
class BinAxis:
    """One of the eight MCTAL bin axes of a tally.

    Attributes
    ----------
    key : str
        MCTAL letter (``f d u s m c e t``; ``r`` replaces ``c`` for
        radiography tallies).
    n : int
        Bin count as written in the file, **including** the total bin.
        ``0`` means the axis is not binned; it then still holds one value.
    values : numpy.ndarray or None
        Numbers listed under the bin line (cell/surface numbers for
        ``f``, upper bin bounds for ``c``/``e``/``t``), or None when MCNP
        lists none (e.g. point detectors, user bins). MCNP writes energies in
        MeV and times in shakes; kika does not convert them.
    has_total : bool
        The last bin is the total over the others (``t`` suffix).
    cumulative : bool
        The bins are cumulative (``c`` suffix).
    flags : tuple of int
        Extra integers on the bin line after the count, verbatim.
    """

    key: str
    n: int = 0
    values: Optional[np.ndarray] = None
    has_total: bool = False
    cumulative: bool = False
    flags: Tuple[int, ...] = ()

    @property
    def size(self) -> int:
        """Length of this axis in :attr:`Tally.result` (``max(n, 1)``)."""
        return max(self.n, 1)

    @property
    def n_bins(self) -> int:
        """Number of real bins, i.e. without the total bin."""
        return self.size - int(self.has_total)

    @property
    def is_binned(self) -> bool:
        """``True`` if the tally card asked for bins on this axis."""
        return self.n > 0

    @property
    def has_edges(self) -> bool:
        """The listed values are all ``n_bins + 1`` bin edges (image grids)."""
        return self.values is not None and len(self.values) == self.n_bins + 1

    @property
    def is_bounded(self) -> bool:
        """The listed values are bin bounds (cosine, energy, time, image grids)."""
        if self.values is None or self.key == "f":
            return False
        return self.has_edges or (self.key in _FIRST_LOWER and len(self.values) >= self.n_bins)

    @property
    def upper(self) -> Optional[np.ndarray]:
        """Upper bounds of the real bins, or ``None`` if the axis has no bounds."""
        if not self.is_bounded:
            return None
        if self.has_edges:
            return np.asarray(self.values[1:], dtype=float)
        return np.asarray(self.values[: self.n_bins], dtype=float)

    @property
    def lower(self) -> Optional[np.ndarray]:
        """Lower bounds of the real bins, or ``None`` if the axis has no bounds.

        MCNP lists only upper bounds for cosine, energy and time; the first
        lower bound is then ``-1``, ``0`` and ``-inf`` respectively, as MCNP
        defines those bins.
        """
        if not self.is_bounded:
            return None
        if self.has_edges:
            return np.asarray(self.values[:-1], dtype=float)
        return np.concatenate(([_FIRST_LOWER[self.key]], self.upper[:-1]))

    @property
    def is_informative(self) -> bool:
        """The axis is worth a dimension: binned, and more than one bin or labelled."""
        return self.is_binned and (self.n_bins > 1 or self.values is not None or self.key == "f")

    def labels(self) -> np.ndarray:
        """Coordinate values for the real bins of this axis.

        Upper bounds for bounded axes, the listed numbers for ``f`` (a ``0``
        there is a union of cells or surfaces), and the 1-based bin index
        otherwise.
        """
        if self.is_bounded:
            return self.upper
        if self.values is not None and len(self.values) == self.n_bins:
            vals = np.asarray(self.values)
            if np.all(vals == np.round(vals)):
                return vals.astype(np.int64)
            return vals
        return np.arange(1, self.n_bins + 1)


@dataclass
class Tfc:
    """Tally fluctuation chart of one tally (or one perturbation of it).

    The chart is the history of one bin of the tally: its value, relative error and
    figure of merit at each of the prints MCNP makes as the run goes on. The
    arrays have one entry per print, in the order of the file.

    Attributes
    ----------
    bins : dict of str to int
        0-based index, per axis key, of the bin MCNP tracks, as an index into
        :attr:`Tally.result` (total bins included).
    nps : numpy.ndarray
        Histories at each print (integers).
    result : numpy.ndarray
        Tally value at each print.
    error : numpy.ndarray
        Relative error at each print (a fraction of the value, as MCNP writes it).
    fom : numpy.ndarray
        Figure of merit at each print (``0`` when MCNP omits it).
    """

    bins: Dict[str, int]
    nps: np.ndarray
    result: np.ndarray
    error: np.ndarray
    fom: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """The chart as a table with the columns ``nps``, ``result``, ``error``, ``fom``.

        One row per print, with a default integer index.
        """
        return pd.DataFrame(
            {"nps": self.nps, "result": self.result, "error": self.error, "fom": self.fom}
        )

    def plot(self, figsize=(15, 5), show_error_bars: bool = True):
        """Plot result, relative error and figure of merit against NPS.

        Parameters
        ----------
        figsize
            Size of the figure in inches, ``(width, height)``. The figure has three
            panels side by side.
        show_error_bars
            If True, the result panel has error bars of half-length
            ``error * abs(result)``, the relative error made absolute.

        Returns
        -------
        matplotlib.figure.Figure
            The figure; it is not shown or saved.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        if show_error_bars:
            ax1.errorbar(self.nps, self.result, yerr=self.error * np.abs(self.result),
                         fmt="b.-", capsize=3, ecolor="lightblue")
        else:
            ax1.plot(self.nps, self.result, "b.-")
        ax1.set(xlabel="NPS", ylabel="Result", title="Result vs NPS")
        ax2.plot(self.nps, self.error, "r.-")
        ax2.set(xlabel="NPS", ylabel="Relative error", title="Error vs NPS")
        ax3.plot(self.nps, self.fom, "g.-")
        ax3.set(xlabel="NPS", ylabel="Figure of merit", title="FOM vs NPS")
        for ax in (ax1, ax2, ax3):
            ax.grid(True)
        fig.tight_layout()
        return fig


@dataclass
class Perturbation:
    """Results of one PERT card for a tally; same shape as the unperturbed one.

    Attributes
    ----------
    number : int
        PERT card number.
    result : numpy.ndarray
        The ``vals pert`` block as MCNP writes it, with the shape of
        :attr:`Tally.result`.
    error : numpy.ndarray
        Relative errors (as MCNP writes them), same shape.
    tfc : Tfc or None
        Fluctuation chart of the perturbation, if written.
    """

    number: int
    result: np.ndarray
    error: np.ndarray
    tfc: Optional[Tfc] = None


@dataclass
class Tally:
    """One tally of a MCTAL file.

    :func:`~kika.mcnp.parse_mctal.read_mctal` returns them in an :class:`Mctal`. ``result`` and
    ``error`` keep the file's eight-dimensional array, with the total bins;
    :meth:`to_xarray`, :meth:`total` and :meth:`to_dataframe` give labelled
    views. MCNP writes a relative error with every value (a fraction of the
    value) and kika stores it as written.

    Attributes
    ----------
    id : int
        Tally number.
    particles : list of str
        Particle designators the tally scores (``["n"]``, ...).
    detector_type : int
        Detector type from the ``tally`` line, copied as MCNP writes it. kika
        takes 3 or more as a radiograph and a negative value as a TMESH tally
        (-1 rectangular, -2 cylindrical, -3 spherical).
    modifier : int
        Tally modifier from the ``tally`` line, copied as MCNP writes it.
    header : tuple of int
        All integers of the ``tally`` line after the number, verbatim.
    comment : str
        FC card text (lines joined with newlines), ``""`` if none.
    axes : dict of str to BinAxis
        The eight bin axes, keyed by MCTAL letter in file order
        (``f d u s m c e t``; the key ``r`` replaces ``c`` for radiography tallies).
    result : numpy.ndarray
        Values, shape ``tuple(a.size for a in axes.values())``, one axis per
        MCTAL letter in the order of *axes*, with ``t`` varying fastest. An
        unbinned axis has length 1 and a binned one its bin count, total bin
        included as the last index.
    error : numpy.ndarray
        Relative errors, same shape.
    tfc : Tfc or None
        Fluctuation chart, if written.
    perturbations : dict of int to Perturbation
        PERT results keyed by PERT card number; each has arrays of the shape of
        *result*.
    """

    id: int
    particles: List[str] = field(default_factory=list)
    detector_type: int = 0
    modifier: int = 0
    header: Tuple[int, ...] = ()
    comment: str = ""
    axes: Dict[str, BinAxis] = field(default_factory=dict)
    result: np.ndarray = field(default_factory=lambda: np.zeros(0))
    error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tfc: Optional[Tfc] = None
    perturbations: Dict[int, Perturbation] = field(default_factory=dict)

    # -- descriptive -----------------------------------------------------

    @property
    def kind(self) -> str:
        """Tally type from the last digit of the number (``"cell flux"``...).

        A TMESH tally is ``"mesh"``: its number does not follow the F-card
        convention.
        """
        if self.is_mesh:
            return "mesh"
        return TALLY_KINDS.get(self.id % 10, "other")

    @property
    def region_name(self) -> str:
        """Name of the ``f`` dimension: ``surface``, ``detector`` or ``cell``."""
        return {1: "surface", 2: "surface", 5: "detector"}.get(self.id % 10, "cell")

    @property
    def is_radiograph(self) -> bool:
        """FIP/FIR/FIC image tally: ``s`` and ``c`` are the image grid axes."""
        return self.detector_type >= 3

    @property
    def is_mesh(self) -> bool:
        """TMESH tally written to MCTAL: the ``f`` axis is a mesh."""
        return self.detector_type < 0

    @property
    def mesh_geometry(self) -> Optional[str]:
        """``"rectangular"``, ``"cylindrical"`` or ``"spherical"`` for a TMESH tally."""
        if not self.is_mesh:
            return None
        return {1: "rectangular", 2: "cylindrical", 3: "spherical"}.get(-self.detector_type, "unknown")

    @property
    def mesh(self) -> Optional[Dict[str, np.ndarray]]:
        """Bin edges of a TMESH tally, keyed ``cora``/``corb``/``corc``.

        The ``f`` line of a mesh tally carries the three bin counts after the
        total and its listed values are the concatenated edges of the three
        coordinates. ``None`` for any other tally.
        """
        if not self.is_mesh:
            return None
        f = self.axes["f"]
        counts = f.flags[-3:]
        if len(f.flags) < 3 or f.values is None or int(np.prod(counts)) != f.n \
                or len(f.values) != sum(counts) + 3:
            raise ValueError(f"tally {self.id}: f line {f.n} {f.flags} is not a mesh description")
        edges, start = {}, 0
        for name, count in zip(("cora", "corb", "corc"), counts):
            edges[name] = np.asarray(f.values[start:start + count + 1], dtype=float)
            start += count + 1
        return edges

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of :attr:`result`: the length of each axis, totals included.

        One entry per axis of :attr:`axes`, in that order; an unbinned axis has
        length 1.
        """
        return tuple(a.size for a in self.axes.values())

    def dim_name(self, key: str) -> str:
        """Dimension name of axis *key* in the labelled views.

        Parameters
        ----------
        key
            MCTAL letter of the axis (``"f"``, ``"e"``...).

        Returns
        -------
        str
            ``"f"`` gives :attr:`region_name` (``"surface"``, ``"detector"`` or
            ``"cell"``). For a radiograph, ``"s"`` is ``"image_s"`` and ``"c"`` or
            ``"r"`` is ``"image_t"``; for a TMESH tally, ``"s"`` is ``"quantity"``.
            Otherwise ``d`` is ``"flag"``, ``u`` ``"user"``, ``s`` ``"segment"``,
            ``m`` ``"multiplier"``, ``c`` ``"cosine"``, ``e`` ``"energy"`` and
            ``t`` ``"time"``.

        Raises
        ------
        KeyError
            For a letter that is not an MCTAL axis letter.
        """
        if key == "f":
            return self.region_name
        if self.is_radiograph and key in ("s", "c", "r"):
            return {"s": "image_s"}.get(key, "image_t")
        if self.is_mesh and key == "s":
            return "quantity"  # one S bin per RMESH/CMESH/SMESH keyword
        return AXIS_NAMES[key]

    @property
    def regions(self) -> np.ndarray:
        """Cell, surface or detector labels of the ``f`` axis."""
        return self.axes["f"].labels()

    @property
    def energies(self) -> Optional[np.ndarray]:
        """Upper energy bounds of the energy bins, or ``None``.

        MCNP writes MeV; kika does not convert.
        """
        return self.axes["e"].upper

    @property
    def times(self) -> Optional[np.ndarray]:
        """Upper time bounds of the time bins, or ``None``.

        MCNP writes shakes; kika does not convert.
        """
        return self.axes["t"].upper

    def _labelled_keys(self) -> List[str]:
        # An axis is a dimension when the tally card binned it and the bins say
        # something: one unlabelled bin (the flag axis without a split, a single
        # FM bin, the unit axes of a TMESH) is not a dimension.
        return [k for k, a in self.axes.items() if a.is_informative and not (k == "d" and a.n == 1)]

    # -- labelled views ---------------------------------------------------

    def to_xarray(self, perturbation: Optional[int] = None) -> xr.Dataset:
        """Binned values as a Dataset with ``result`` and ``error``.

        Total bins are left out (see :meth:`total`), and so is every axis the
        tally did not bin. Bounded axes carry their upper bounds as the
        coordinate and a ``<name>_low`` coordinate with the lower bounds. When
        the cell/surface numbers repeat (MCNP writes ``0`` for each union bin
        and for the total of an F card ending in ``T``), the dimension is the
        1-based bin index and the numbers are in ``<name>_id``.

        The dimensions are the binned axes that carry information, in the order
        of :attr:`axes` (``f d u s m c e t``) and named by :meth:`dim_name`; a
        flag axis of one bin is not a dimension. A tally with no such axis gives
        zero-dimensional ``result`` and ``error``. For a TMESH tally the ``f``
        axis becomes the dimensions ``cora``, ``corb`` and ``corc``, with the bin
        midpoints as coordinates and ``_low`` and ``_high`` edges. MCNP writes
        energies in MeV and times in shakes; kika does not convert them. ``error``
        is relative.

        Parameters
        ----------
        perturbation
            PERT card number to return instead of the unperturbed tally. None
            gives the unperturbed tally.

        Returns
        -------
        xarray.Dataset
            The variables ``result`` and ``error``, with the attributes ``tally``
            (number), ``kind``, ``comment`` and, when set, ``perturbation``.

        Raises
        ------
        KeyError
            If the tally has no perturbation *perturbation*.
        """
        return self._view(perturbation, total_keys=())

    def total(self, *axes: str, perturbation: Optional[int] = None) -> xr.Dataset:
        """Total bin(s) of one or more axes, as a Dataset over the others.

        ``tally.total("energy")`` is the energy-integrated tally per cell (and
        per any other binned axis).

        Parameters
        ----------
        *axes
            Dimension names (``"energy"``) or MCTAL letters (``"e"``). Each
            named axis is replaced by its total bin and is no longer a
            dimension; the other dimensions are those of :meth:`to_xarray`.
        perturbation
            PERT card number, as in :meth:`to_xarray`.

        Returns
        -------
        xarray.Dataset
            ``result`` and ``error`` (relative) over the remaining dimensions,
            with the attributes of :meth:`to_xarray` and ``total_over``, the
            dimension names of the totals. With no *axes* it equals
            :meth:`to_xarray`.

        Raises
        ------
        ValueError
            If an axis has no total bin.
        KeyError
            If a name is not an axis of the tally, or the tally has no
            perturbation *perturbation*.
        """
        keys = []
        for name in axes:
            key = self._key(name)
            if not self.axes[key].has_total:
                raise ValueError(f"tally {self.id}: axis {name!r} has no total bin")
            keys.append(key)
        return self._view(perturbation, total_keys=tuple(keys))

    def to_dataframe(self, perturbation: Optional[int] = None) -> pd.DataFrame:
        """:meth:`to_xarray` flattened to one row per bin.

        Parameters
        ----------
        perturbation
            PERT card number, as in :meth:`to_xarray`.

        Returns
        -------
        pandas.DataFrame
            The xarray conversion of :meth:`to_xarray`, with a default integer
            index: one column per dimension and per coordinate (``<name>_low``,
            ``<name>_id``, and for a TMESH tally ``_high`` too), plus ``result`` and
            ``error`` (relative). A tally with no dimensions gives one row with
            ``result`` and ``error`` only.
        """
        ds = self.to_xarray(perturbation)
        if not ds.result.dims:
            return pd.DataFrame({"result": [float(ds.result)], "error": [float(ds.error)]})
        return ds.to_dataframe().reset_index()

    def _key(self, name: str) -> str:
        if name in self.axes:
            return name
        for key in self.axes:
            if self.dim_name(key) == name:
                return key
        raise KeyError(f"tally {self.id} has no axis {name!r}")

    def _arrays(self, perturbation: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        if perturbation is None:
            return self.result, self.error
        try:
            p = self.perturbations[perturbation]
        except KeyError:
            raise KeyError(f"tally {self.id} has no perturbation {perturbation}") from None
        return p.result, p.error

    def _view(self, perturbation: Optional[int], total_keys: Tuple[str, ...]) -> xr.Dataset:
        result, error = self._arrays(perturbation)
        index = []
        dims: List[str] = []
        coords: Dict[str, Tuple[str, np.ndarray]] = {}
        labelled = self._labelled_keys()
        for key, axis in self.axes.items():
            if key in total_keys:
                index.append(axis.size - 1)
            elif key == "f" and self.is_mesh:
                index.append(slice(None))
                dims.append("f")
            elif key in labelled:
                index.append(slice(0, axis.n_bins))
                name = self.dim_name(key)
                dims.append(name)
                labels = axis.labels()
                if len(np.unique(labels)) < len(labels):
                    # MCNP writes 0 for every union bin and for the total over
                    # cells of an F card with T, so the numbers can repeat: index
                    # the bins instead and keep the numbers alongside.
                    coords[name] = (name, np.arange(1, len(labels) + 1))
                    coords[f"{name}_id"] = (name, labels)
                else:
                    coords[name] = (name, labels)
                if axis.lower is not None:
                    coords[f"{name}_low"] = (name, axis.lower)
            else:
                index.append(0)
        result, error = result[tuple(index)], error[tuple(index)]
        if self.is_mesh:
            result, error, dims = self._unfold_mesh(result, error, dims, coords)
        data = {"result": (dims, result), "error": (dims, error)}
        attrs = {"tally": self.id, "kind": self.kind, "comment": self.comment}
        if self.is_mesh:
            attrs["mesh_geometry"] = self.mesh_geometry
        if perturbation is not None:
            attrs["perturbation"] = perturbation
        if total_keys:
            attrs["total_over"] = [self.dim_name(k) for k in total_keys]
        return xr.Dataset(data, coords=coords, attrs=attrs)

    def _unfold_mesh(self, result, error, dims, coords):
        # The f index of a TMESH runs with CORA fastest and CORC slowest
        # (MCNP 6.3.1 manual, section 6.3.4.1.3).
        edges = self.mesh
        axis = dims.index("f")
        shape = tuple(len(edges[n]) - 1 for n in ("corc", "corb", "cora"))
        new_dims = dims[:axis] + ["corc", "corb", "cora"] + dims[axis + 1:]
        unfold = lambda a: a.reshape(a.shape[:axis] + shape + a.shape[axis + 1:])
        order = list(range(len(new_dims)))
        order[axis:axis + 3] = [axis + 2, axis + 1, axis]
        new_dims = [new_dims[i] for i in order]
        for name, e in edges.items():
            coords[name] = (name, 0.5 * (e[1:] + e[:-1]))
            coords[f"{name}_low"] = (name, e[:-1])
            coords[f"{name}_high"] = (name, e[1:])
        return (unfold(result).transpose(order), unfold(error).transpose(order), new_dims)

    def __repr__(self) -> str:
        parts = [f"{self.dim_name(k)}={self.axes[k].n_bins}" + ("+total" if self.axes[k].has_total else "")
                 for k in self._labelled_keys()]
        if self.is_mesh:
            parts[0] = "mesh=" + "x".join(str(len(e) - 1) for e in self.mesh.values())
        dims = ", ".join(parts) or "scalar"
        extra = f", {len(self.perturbations)} perturbations" if self.perturbations else ""
        label = f" {self.comment.splitlines()[0]!r}" if self.comment else ""
        return f"<Tally {self.id} ({self.kind}, {''.join(self.particles)}){label}: {dims}{extra}>"


@dataclass
class Kcode:
    """KCODE cycle records of a MCTAL file.

    Each row of :attr:`data` is one cycle, in the file's order. The columns are the
    ``keff`` of the cycle by the collision, absorption and track-length estimators,
    the prompt removal lifetimes, the running averages of those estimators with their
    standard deviations (names ending in ``_std``), the combined ``keff`` with and
    without bias correction, the number of histories of the cycle and the figure of
    merit of the combined ``keff``. Their names are in :attr:`columns`. kika stores
    the numbers as MCNP writes them and does not convert them.

    Attributes
    ----------
    cycles : int
        Number of cycles written.
    settle : int
        Inactive (settle) cycles.
    data : numpy.ndarray
        Records, shape ``(cycles, n_columns)``; columns in
        :data:`KCODE_COLUMNS` order.
    """

    cycles: int
    settle: int
    data: np.ndarray

    @property
    def columns(self) -> List[str]:
        """Column names of :attr:`data`, one per column.

        The names of :data:`KCODE_COLUMNS` for as many columns as the records
        have; further columns are named ``column_<i>``, from the first unnamed
        index.
        """
        n = self.data.shape[1]
        names = list(KCODE_COLUMNS[:n])
        return names + [f"column_{i}" for i in range(len(names), n)]

    def to_dataframe(self) -> pd.DataFrame:
        """The records as a table with the columns of :attr:`columns`.

        The index is the row number, from 1 to ``cycles``, named ``cycle``.
        """
        df = pd.DataFrame(self.data, columns=self.columns)
        df.index = pd.RangeIndex(1, self.cycles + 1, name="cycle")
        return df

    @property
    def keff(self) -> Tuple[float, float]:
        """Final combined keff and its standard deviation (last cycle)."""
        last = self.data[-1]
        i = KCODE_COLUMNS.index("avg_keff_combined")
        return float(last[i]), float(last[i + 1])


@dataclass
class Mctal:
    """Contents of an MCNP MCTAL file, as :func:`~kika.mcnp.parse_mctal.read_mctal` returns it.

    Indexing with a tally number returns its :class:`Tally` (``KeyError`` if
    absent); iterating yields the tallies and ``len`` counts them.

    Attributes
    ----------
    code : str
        Code name (``mcnp6``...).
    version : str
        Code version string.
    probid : str
        Problem identification (date and time of the run).
    dump : int
        Dump number.
    nps : int
        Histories run.
    random_numbers : int
        Pseudorandom numbers used.
    title : str
        Problem title (first line of the input).
    npert : int
        Number of PERT cards (0 if none).
    tally_numbers : list of int
        Tally numbers in the order of the ``ntal`` list.
    tallies : dict of int to Tally
        Tallies keyed by number.
    kcode : Kcode or None
        KCODE records, if the file has them (a criticality calculation).
    """

    code: str = ""
    version: str = ""
    probid: str = ""
    dump: int = 0
    nps: int = 0
    random_numbers: int = 0
    title: str = ""
    npert: int = 0
    tally_numbers: List[int] = field(default_factory=list)
    tallies: Dict[int, Tally] = field(default_factory=dict)
    kcode: Optional[Kcode] = None

    def __getitem__(self, number: int) -> Tally:
        try:
            return self.tallies[number]
        except KeyError:
            raise KeyError(f"no tally {number}; available: {self.tally_numbers}") from None

    def __iter__(self) -> Iterator[Tally]:
        return iter(self.tallies.values())

    def __len__(self) -> int:
        return len(self.tallies)

    def __repr__(self) -> str:
        lines = [f"<Mctal {self.code} {self.version} {self.probid}: nps={self.nps}, "
                 f"{len(self.tallies)} tallies"
                 + (f", {self.npert} perturbations" if self.npert else "")
                 + (", kcode" if self.kcode is not None else "") + ">"]
        if self.title:
            lines.append(f"  {self.title}")
        lines.extend(f"  {t!r}" for t in self.tallies.values())
        return "\n".join(lines)


def particles_from_header(ptype: int, flags: Sequence[int]) -> List[str]:
    """Particle designators of a tally from its ``tally``-line particle type.

    Parameters
    ----------
    ptype
        Particle type on the ``tally`` line. Negative: *flags* lists
        one 0/1 entry per MCNP6 particle. Positive: MCNP5 bit flags (1 n, 2 p,
        4 e).
    flags
        The particle flag list (used when *ptype* is negative); entry ``i``
        switches on the ``i``-th designator of ``MCNP6_PARTICLES``.

    Returns
    -------
    list of str
        The designators of the particles switched on (``["n", "p"]``), in the
        order of the flag list, or ``n``, ``p``, ``e`` for MCNP5 bits. A
        *ptype* of 0 gives an empty list.
    """
    if ptype < 0:
        return [p for p, on in zip(MCNP6_PARTICLES, flags) if on]
    return [p for bit, p in _MCNP5_PARTICLE_BITS if ptype & bit]

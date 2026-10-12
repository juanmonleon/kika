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

    :ivar key: MCTAL letter (``f d u s m c e t``; ``r`` replaces ``c`` for
        radiography tallies).
    :ivar n: Bin count as written in the file, **including** the total bin.
        ``0`` means the axis is not binned; it then still holds one value.
    :ivar values: Numbers listed under the bin line (cell/surface numbers for
        ``f``, upper bin bounds for ``c``/``e``/``t``), or ``None`` when MCNP
        lists none (e.g. point detectors, user bins).
    :ivar has_total: The last bin is the total over the others (``t`` suffix).
    :ivar cumulative: The bins are cumulative (``c`` suffix).
    :ivar flags: Extra integers on the bin line after the count, verbatim.
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

    :ivar bins: 0-based index, per axis key, of the bin MCNP tracks.
    :ivar nps: Histories at each print.
    :ivar result: Tally value at each print.
    :ivar error: Relative error at each print.
    :ivar fom: Figure of merit at each print (``0`` when MCNP omits it).
    """

    bins: Dict[str, int]
    nps: np.ndarray
    result: np.ndarray
    error: np.ndarray
    fom: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"nps": self.nps, "result": self.result, "error": self.error, "fom": self.fom}
        )

    def plot(self, figsize=(15, 5), show_error_bars: bool = True):
        """Plot result, relative error and figure of merit against NPS.

        :returns: The matplotlib figure.
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

    :ivar number: PERT card number.
    :ivar result: Values as MCNP writes them under ``vals pert`` (the
        perturbed difference, or a Taylor coefficient with METHOD=2/3), with
        the shape of :attr:`Tally.result`.
    :ivar error: Relative errors, same shape.
    :ivar tfc: Fluctuation chart of the perturbation, if written.
    """

    number: int
    result: np.ndarray
    error: np.ndarray
    tfc: Optional[Tfc] = None


@dataclass
class Tally:
    """One tally of a MCTAL file.

    :ivar id: Tally number.
    :ivar particles: Particle designators the tally scores (``["n"]``, ...).
    :ivar detector_type: Detector type from the ``tally`` line: 0 none,
        1 point, 2 ring, 3 pinhole / 4 rectangular / 5 cylindrical radiograph;
        for a TMESH tally minus the mesh type (-1 rectangular, -2 cylindrical,
        -3 spherical).
    :ivar modifier: Tally modifier from the ``tally`` line: 0 none, 1 for an
        ``*F`` tally, 2 for a ``+F`` tally.
    :ivar header: All integers of the ``tally`` line after the number, verbatim.
    :ivar comment: FC card text (lines joined with newlines), ``""`` if none.
    :ivar axes: The eight bin axes, keyed by MCTAL letter in file order.
    :ivar result: Values, shape ``tuple(a.size for a in axes.values())``,
        totals included.
    :ivar error: Relative errors, same shape.
    :ivar tfc: Fluctuation chart, if written.
    :ivar perturbations: PERT results keyed by PERT card number.
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
        return tuple(a.size for a in self.axes.values())

    def dim_name(self, key: str) -> str:
        """Dimension name of axis *key* in the labelled views."""
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
        """Upper energy bounds of the energy bins (MeV), or ``None``."""
        return self.axes["e"].upper

    @property
    def times(self) -> Optional[np.ndarray]:
        """Upper time bounds of the time bins (shakes), or ``None``."""
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

        :param perturbation: PERT card number to return instead of the
            unperturbed tally.
        """
        return self._view(perturbation, total_keys=())

    def total(self, *axes: str, perturbation: Optional[int] = None) -> xr.Dataset:
        """Total bin(s) of one or more axes, as a Dataset over the others.

        ``tally.total("energy")`` is the energy-integrated tally per cell (and
        per any other binned axis).

        :param axes: Dimension names (``"energy"``) or MCTAL letters (``"e"``).
        :param perturbation: PERT card number, as in :meth:`to_xarray`.
        :raises ValueError: If an axis has no total bin.
        """
        keys = []
        for name in axes:
            key = self._key(name)
            if not self.axes[key].has_total:
                raise ValueError(f"tally {self.id}: axis {name!r} has no total bin")
            keys.append(key)
        return self._view(perturbation, total_keys=tuple(keys))

    def to_dataframe(self, perturbation: Optional[int] = None) -> pd.DataFrame:
        """:meth:`to_xarray` flattened to one row per bin."""
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

    :ivar cycles: Number of cycles written.
    :ivar settle: Inactive (settle) cycles.
    :ivar data: Records, shape ``(cycles, n_columns)``; columns in
        :data:`KCODE_COLUMNS` order.
    """

    cycles: int
    settle: int
    data: np.ndarray

    @property
    def columns(self) -> List[str]:
        n = self.data.shape[1]
        names = list(KCODE_COLUMNS[:n])
        return names + [f"column_{i}" for i in range(len(names), n)]

    def to_dataframe(self) -> pd.DataFrame:
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
    """Contents of an MCNP MCTAL file.

    :ivar code: Code name (``mcnp6``...).
    :ivar version: Code version string.
    :ivar probid: Problem identification (date and time of the run).
    :ivar dump: Dump number.
    :ivar nps: Histories run.
    :ivar random_numbers: Pseudorandom numbers used.
    :ivar title: Problem title (first line of the input).
    :ivar npert: Number of PERT cards (0 if none).
    :ivar tally_numbers: Tally numbers in the order of the ``ntal`` list.
    :ivar tallies: Tallies keyed by number.
    :ivar kcode: KCODE records, if the run was a criticality calculation.
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

    :param ptype: Particle type on the ``tally`` line. Negative: *flags* lists
        one 0/1 entry per MCNP6 particle. Positive: MCNP5 bit flags (1 n, 2 p,
        4 e).
    :param flags: The particle flag list (used when *ptype* is negative).
    """
    if ptype < 0:
        return [p for p, on in zip(MCNP6_PARTICLES, flags) if on]
    return [p for bit, p in _MCNP5_PARTICLE_BITS if ptype & bit]

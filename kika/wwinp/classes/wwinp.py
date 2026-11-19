"""Main WWINP container class."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .header import Header
from .geometry import GeometryData
from .weight_windows import WeightWindowValues
from .query_result import QueryResult


@dataclass
class WWINP:
    """Top-level container for a WWINP file.

    Parameters
    ----------
    filename : str or None
        Source file path (informational).
    header : Header
        File header metadata.
    geometry : GeometryData
        Geometry axis definitions and mesh data.
    values : WeightWindowValues
        Weight window value arrays and operations.
    time_bins : dict[int, np.ndarray]
        Time bin boundaries per particle index.
    energy_bins : dict[int, np.ndarray]
        Energy bin boundaries per particle index.
    """

    filename: Optional[str] = None
    header: Header = field(default_factory=Header)
    geometry: GeometryData = field(default_factory=GeometryData)
    values: WeightWindowValues = field(default_factory=WeightWindowValues)
    time_bins: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    energy_bins: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Delegated operations
    # ------------------------------------------------------------------

    def copy(self) -> WWINP:
        """Return a deep copy."""
        return deepcopy(self)

    def multiply(
        self,
        factor: float = 1.2,
        particles: Optional[list[int] | int] = None,
    ) -> None:
        """Scale weight window values by *factor*, in place.

        Parameters
        ----------
        factor
            Multiplier applied to every value.
        particles
            Particle indices (0-based, as in ``values.ww_values``) to scale. None
            scales all of them.

        Raises
        ------
        ValueError
            If a requested particle index is not in ``values.ww_values``.
        """
        self.values.multiply(factor, particles)

    def soften(
        self,
        power: float = 0.6,
        particles: Optional[list[int] | int] = None,
    ) -> None:
        """Raise values to *power* (< 1 softens, > 1 hardens), in place.

        Parameters
        ----------
        power
            Exponent applied to every value.
        particles
            Particle indices (0-based, as in ``values.ww_values``) to modify. None
            modifies all of them.

        Raises
        ------
        ValueError
            If a requested particle index is not in ``values.ww_values``.
        """
        self.values.soften(power, particles)

    def apply_ratio_threshold(
        self,
        threshold: float = 10.0,
        particles: Optional[list[int] | int] = None,
    ) -> int:
        """Zero out cells whose neighbor ratio exceeds *threshold*, in place.

        The ratio of a cell is 1 or the largest ``neighbour / cell`` over its face
        neighbours along the three spatial axes, whichever is greater, taken within
        each time and energy slice. A cell with value 0 has ratio 1.

        Parameters
        ----------
        threshold
            Cells with a ratio strictly greater than this are set to 0.
        particles
            Particle indices (0-based, as in ``values.ww_values``) to modify. None
            modifies all of them.

        Returns
        -------
        int
            Number of cells set to 0, summed over particles, times and energies.

        Raises
        ------
        ValueError
            If a requested particle index is not in ``values.ww_values``.
        """
        return self.values.apply_ratio_threshold(threshold, particles)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        particle: Optional[int] = None,
        energy: Optional[Union[float, tuple[float, float]]] = None,
        time: Optional[Union[float, tuple[float, float]]] = None,
        **spatial_kwargs: Optional[Union[float, tuple[float, float]]],
    ) -> QueryResult:
        """Query weight window values by particle, energy, time, and spatial coordinates.

        Spatial coordinates are passed as keyword arguments matching the
        geometry axis labels (e.g. ``x=5.0``, ``y=(0, 10)``, ``z=(-5, 5)``
        for cartesian geometry).

        Parameters
        ----------
        particle
            Particle index (0-based). None returns every particle, in ascending
            order of index.
        energy
            A value, or a ``(low, high)`` range, in the units of ``energy_bins``.
            The entries of ``energy_bins`` are taken as upper boundaries, with 0.0
            as the lower edge of the first bin. A value selects the bin containing
            it (both neighbours when it lies on an interior boundary; a value below
            the first edge or above the last is clamped to the first or last bin).
            A range selects every bin it overlaps, including a bin that starts
            exactly at *high*. None selects all energy bins.
        time
            A value or a ``(low, high)`` range, in the units of ``time_bins``,
            selected with the same rules as *energy*, but against the values of
            ``time_bins`` as they are, with no 0.0 edge added: the addressable bins
            are the intervals between consecutive values of ``time_bins``. None
            selects all time bins. It does nothing when the file is not time
            dependent or the particle has no ``time_bins``.
        **spatial_kwargs
            One optional value or ``(low, high)`` range per geometry axis label
            (``x, y, z`` for cartesian, ``r, z, theta`` for cylindrical,
            ``r, phi, theta`` for spherical), against the fine mesh boundaries of
            that axis and with the same rules as *energy*. An axis that is not given
            is kept whole. A keyword that is not an axis label is ignored.

        Returns
        -------
        QueryResult
            One entry per selected particle. Each ``ww_values`` array has shape
            ``(nt, ne, nfz, nfy, nfx)`` restricted to the selection: the third,
            fourth and fifth axes follow the third, second and first geometry axis
            labels. The intervals hold the ``(starts, ends)`` of the selected bins.

        Raises
        ------
        KeyError
            If *particle* is not in ``values.ww_values``.
        ValueError
            If a ``(low, high)`` range has low greater than high.
        """
        from kika.wwinp.utils import get_closest_indices, get_range_indices

        labels = self.geometry.axis_labels
        meshes = self.geometry.fine_mesh

        # Determine particle list
        if particle is not None:
            ptypes = [particle]
        else:
            ptypes = sorted(self.values.ww_values.keys())

        # Resolve spatial slicing for each axis
        spatial_slices: list[np.ndarray] = []  # index arrays for axes 0,1,2
        spatial_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for label in labels:
            grid = meshes[label]
            coord = spatial_kwargs.get(label)
            if coord is not None:
                if isinstance(coord, tuple):
                    indices = get_range_indices(grid, coord)
                else:
                    indices = get_closest_indices(grid, coord)
            else:
                indices = np.arange(len(grid) - 1)
            spatial_slices.append(indices)
            starts = grid[indices]
            ends = grid[np.minimum(indices + 1, len(grid) - 1)]
            spatial_intervals[label] = (starts, ends)

        # Build results per particle
        result_values: list[np.ndarray] = []
        result_energy: list[tuple[np.ndarray, np.ndarray]] = []
        result_time: list[tuple[np.ndarray, np.ndarray]] = []

        for p in ptypes:
            ww = self.values.ww_values[p]  # (nt, ne, nfz, nfy, nfx)
            e_bins = self.energy_bins[p]

            # Energy slicing
            if energy is not None:
                if isinstance(energy, tuple):
                    e_idx = get_range_indices(
                        np.concatenate(([0.0], e_bins)), energy
                    )
                    # Convert from boundary indices to bin indices
                    e_idx = e_idx[e_idx < len(e_bins)]
                else:
                    e_idx = get_closest_indices(
                        np.concatenate(([0.0], e_bins)), energy
                    )
                    e_idx = e_idx[e_idx < len(e_bins)]
            else:
                e_idx = np.arange(len(e_bins))

            e_grid = np.concatenate(([0.0], e_bins))
            e_starts = e_grid[e_idx]
            e_ends = e_grid[np.minimum(e_idx + 1, len(e_grid) - 1)]
            result_energy.append((e_starts, e_ends))

            # Time slicing
            if self.header.has_time_dependency and p in self.time_bins:
                t_bins = self.time_bins[p]
                if time is not None:
                    if isinstance(time, tuple):
                        t_idx = get_range_indices(t_bins, time)
                    else:
                        t_idx = get_closest_indices(t_bins, time)
                else:
                    t_idx = np.arange(ww.shape[0])
                t_starts = t_bins[t_idx]
                t_ends = t_bins[np.minimum(t_idx + 1, len(t_bins) - 1)]
                result_time.append((t_starts, t_ends))
            else:
                t_idx = np.arange(ww.shape[0])
                result_time.append((np.array([]), np.array([])))

            # Slice: ww[t_idx][:, e_idx][:, :, z_idx][:, :, :, y_idx][..., x_idx]
            sliced = ww[np.ix_(t_idx, e_idx)]
            sliced = sliced[:, :, spatial_slices[2][:, None, None],
                            spatial_slices[1][None, :, None],
                            spatial_slices[0][None, None, :]]
            result_values.append(sliced)

        return QueryResult(
            particle_types=ptypes,
            ww_values=result_values,
            energy_intervals=result_energy,
            time_intervals=result_time,
            spatial_intervals=spatial_intervals,
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, filepath: str, overwrite: bool = False) -> None:
        """Write this WWINP object to a file in the WWINP text format.

        Parameters
        ----------
        filepath
            Output path.
        overwrite
            If False, an existing file is refused.

        Raises
        ------
        FileExistsError
            If *filepath* exists and *overwrite* is False.
        """
        from kika.wwinp.writers.wwinp_writer import write_wwinp

        write_wwinp(self, filepath, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        h = self.header
        fm = self.geometry.fine_mesh
        labels = self.geometry.axis_labels

        lines = [
            f"WWINP  mesh_type={h.mesh_type}  particles={h.ni}  "
            f"time_dep={'yes' if h.has_time_dependency else 'no'}",
        ]
        if self.filename:
            lines.append(f"  file: {self.filename}")

        lines.append("  Geometry:")
        for label in labels:
            grid = fm[label]
            lines.append(
                f"    {label}: [{grid[0]:.4g}, {grid[-1]:.4g}]  "
                f"bins={len(grid)-1}"
            )

        for p in sorted(self.values.ww_values.keys()):
            arr = self.values.ww_values[p]
            pos = arr[arr > 0]
            e_bins = self.energy_bins.get(p, np.array([]))
            lines.append(f"  Particle {p}:")
            lines.append(f"    shape={arr.shape}  energies={len(e_bins)}")
            if pos.size > 0:
                lines.append(
                    f"    min={pos.min():.3e}  max={pos.max():.3e}  "
                    f"nonzero={pos.size}/{arr.size} "
                    f"({100*pos.size/arr.size:.1f}%)"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WWINP(particles={self.header.ni}, "
            f"mesh={self.header.mesh_type}, "
            f"spatial=({self.header.nfx},{self.header.nfy},{self.header.nfz}))"
        )

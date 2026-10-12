"""Readers for MCNP FMESH results: the MESHTAL text file and the 6.3+ runtape.

:func:`read_meshtal` takes either file and returns a :class:`~kika.mcnp.meshtal.Meshtal`.

MESHTAL text (MCNP 5, 6.1, 6.2, 6.3)
    The column formats are read: ``out=col`` (the default), ``cf`` (adds the
    ``Volume`` and ``Rslt * Vol`` columns), and the MCNP 6.3 ``colsci`` /
    ``cfsci`` variants, which print every real in scientific notation. Rows
    run with the energy bin outermost, then time, then ``i j k`` with ``k``
    fastest; an energy or time ``Total`` row block follows the bins when there
    is more than one. Every row's coordinates are checked against the header
    bounds, so a layout the reader does not expect fails loudly instead of
    being misplaced.

    The matrix formats (``ij``/``ik``/``jk``) are not read: they repeat the same
    numbers with the relative error cut to five decimals. Cell-under-voxel and
    nuclide-binned meshes (D1SUNED) are not read either.

    The header prints spatial bounds with two decimals and energy/time bounds
    with three significant digits, so bounds read from text carry that
    rounding. The runtape holds them in full precision.

HDF5 runtape (MCNP 6.3+, ``out=xdmf``)
    ``/results/mesh_tally/mesh_tally_<n>`` (MCNP 6.3.1 manual, appendix D.4):
    ``mean`` and ``relative_standard_error`` with the spatial axes stored
    reversed (``k j i``), the ``grid_*`` edge datasets and the descriptive
    attributes. Without ``out=xdmf`` MCNP does not write this group and the
    reader says so.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from ._fortran import fields, numbers
from .mctal import MCNP6_PARTICLES
from .meshtal import SPATIAL_DIMS, Meshtal, MeshTally

_HEADER = re.compile(r"^\s*(\S+)\s+version\s+(\S+)\s+ld=\s*(\S+)\s+probid\s*=\s*(.*?)\s*$")
_TALLY = re.compile(r"^\s*Mesh Tally Number\s+(\d+)\s*$")
_NPS = re.compile(r"Number of histories used for normalizing tallies\s*=\s*(\S+)")
_PARTICLE = re.compile(r"^\s*(?:This is an?\s+)?(.*?)\s+mesh tally\.\s*$")  # MCNP5: "This is a neutron mesh tally."
_DIRECTION = re.compile(r"^\s*(X|Y|Z|R|Theta) direction(?: \(revolutions\))?:(.*)$")
_MATRIX = re.compile(r"^\s*(Energy Bin:|Time Bin:|Tally Results:|[XYZR] bin:|Theta bin:)")

_GEOMETRY = {("x", "y", "z"): "rectangular", ("r", "z", "theta"): "cylindrical"}

#: Long names of the MCNP6 particle numbers used by ``particle_number``.
_PARTICLE_NAMES = {1: "neutron", 2: "photon", 3: "electron", 9: "proton", 31: "deuteron",
                   32: "triton", 33: "helion", 34: "alpha"}

# Coordinates are printed as f10.3 from bounds printed with two decimals; this
# tolerance covers both roundings. Energy/time row labels are 4 significant
# digits against 3 in the header.
_SPATIAL_ATOL = 6e-3
_BIN_RTOL = 1e-2


def read_meshtal(filename: Union[str, Path]) -> Meshtal:
    """Read MCNP FMESH results from a MESHTAL text file or an MCNP 6.3+ runtape.

    :param filename: A MESHTAL file (``meshtal``, ``*.msht``), or the HDF5
        runtape of a run whose FMESH cards used ``out=xdmf`` (``runtpe.h5``,
        ``<name>r.h5``).
    :returns: The mesh tallies of the file.
    :raises ValueError: If the file is neither, or its contents are not
        consistent with the header bounds.
    :raises NotImplementedError: For the matrix (``ij``/``ik``/``jk``),
        cell-under-voxel and nuclide-binned layouts.
    """
    path = Path(filename)
    if _is_hdf5(path):
        return _read_hdf5(path)
    with open(path, "r", errors="replace") as handle:
        lines = handle.read().splitlines()
    return _read_text(lines, str(path))


def _is_hdf5(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(8) == b"\x89HDF\r\n\x1a\n"


# ---------------------------------------------------------------------------
# MESHTAL text
# ---------------------------------------------------------------------------

def _read_text(lines: List[str], name: str) -> Meshtal:
    meshtal = Meshtal(source="text")
    starts = [n for n, line in enumerate(lines) if _TALLY.match(line)]
    if not starts:
        raise ValueError(f"{name}: no 'Mesh Tally Number' line; not a MESHTAL file")

    preamble = [line for line in lines[: starts[0]] if line.strip()]
    if preamble and (m := _HEADER.match(preamble[0])):
        meshtal.code, meshtal.version, meshtal.build_date, meshtal.probid = m.groups()
        preamble = preamble[1:]
    for line in preamble:
        if m := _NPS.search(line):
            meshtal.nps = float(m.group(1))
        elif not meshtal.title:
            meshtal.title = line.strip()

    for n, start in enumerate(starts):
        stop = starts[n + 1] if n + 1 < len(starts) else len(lines)
        tally = _text_tally(lines, start, stop, name)
        meshtal.tallies[tally.id] = tally
    return meshtal


def _text_tally(lines: List[str], start: int, stop: int, name: str) -> MeshTally:
    tally_id = int(_TALLY.match(lines[start]).group(1))
    where = f"{name}, mesh tally {tally_id}"
    tally = MeshTally(id=tally_id)
    i = start + 1

    comment = []
    while i < stop and not _PARTICLE.match(lines[i]):
        if lines[i].strip():
            comment.append(lines[i].strip())
        i += 1
    if i == stop:
        raise ValueError(f"{where}: no '... mesh tally.' particle line")
    particle = _PARTICLE.match(lines[i]).group(1).strip()
    if "cell-under-voxel" in particle or "cell-under-voxel" in lines[i]:
        raise NotImplementedError(f"{where}: cell-under-voxel mesh tallies are not supported")
    tally.particle = particle
    tally.comment = "\n".join(comment)
    i += 1

    while i < stop and "Tally bin boundaries:" not in lines[i]:
        if lines[i].strip():
            tally.notes.append(lines[i].strip())
        i += 1
    i += 1

    bounds: Dict[str, np.ndarray] = {}
    header = None
    while i < stop:
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        if _MATRIX.match(line):
            raise NotImplementedError(
                f"{where}: matrix layout (out=ij/ik/jk) is not supported; use out=col, cf, "
                "colsci, cfsci or xdmf")
        if "origin at" in line:
            vals = numbers(line)
            tally.origin = np.array(vals[0:3])
            if "axis in" in line:
                tally.axis, tally.vec = np.array(vals[3:6]), np.array(vals[6:9])
        elif m := _DIRECTION.match(line):
            bounds[m.group(1).lower()] = np.array(numbers(m.group(2)))
        elif "Energy bin boundaries:" in line:
            tally.energy_bounds = np.array(numbers(line.split(":", 1)[1]))
        elif "Time bin boundaries:" in line:
            tally.time_bounds = np.array(numbers(line.split(":", 1)[1]))
        elif "Result" in line and "Rel Error" in line:
            header = line
            break
        else:
            raise NotImplementedError(f"{where}: unsupported header line {line.strip()!r}")
    if header is None:
        raise ValueError(f"{where}: no column header ('... Result  Rel Error')")

    geometry = _GEOMETRY.get(tuple(bounds))
    if geometry is None:
        raise ValueError(f"{where}: unexpected mesh directions {list(bounds)}")
    tally.geometry = geometry
    tally.bounds = bounds

    columns = header.split()
    has_energy, has_time = "Energy" in columns, "Time" in columns
    n_bin_cols = int(has_energy) + int(has_time)
    has_volume = "Volume" in columns
    n_cols = n_bin_cols + 3 + 2 + 2 * int(has_volume)

    rows = [fields(line) for line in lines[i:stop] if line.strip()]
    bad = [r for r in rows if len(r) != n_cols]
    if bad:
        raise ValueError(f"{where}: rows with {len(bad[0])} fields, the column header needs {n_cols}")
    data = np.array(rows, dtype=float).reshape(-1, n_cols)

    col = 0
    n_e, n_t = tally.n_energy, tally.n_time
    if has_energy:
        tally.has_energy_total = bool(np.isnan(data[:, col]).any())
        col += 1
    elif n_e > 1:
        raise ValueError(f"{where}: {n_e} energy bins but no Energy column")
    if has_time:
        tally.has_time_total = bool(np.isnan(data[:, col]).any())
    elif n_t > 1:
        raise ValueError(f"{where}: {n_t} time bins but no Time column")

    ne_rows = n_e + int(tally.has_energy_total)
    nt_rows = n_t + int(tally.has_time_total)
    grid = tally.shape
    expected = ne_rows * nt_rows * int(np.prod(grid))
    if data.shape[0] != expected:
        raise ValueError(f"{where}: {data.shape[0]} rows, the bounds need {expected}")
    cube = data.reshape(ne_rows, nt_rows, *grid, n_cols)

    col = 0
    if has_energy:
        _check_bin_column(cube[..., col], tally.energy_bounds, tally.has_energy_total, 0, where, "energy")
        col += 1
    if has_time:
        _check_bin_column(cube[..., col], tally.time_bounds, tally.has_time_total, 1, where, "time")
        col += 1
    for axis, dim in enumerate(tally.spatial_dims):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        centre = tally.centers(dim).reshape(shape)
        if not np.allclose(cube[..., col], centre, rtol=0, atol=_SPATIAL_ATOL):
            raise ValueError(f"{where}: {dim} column does not follow the {dim} bounds; "
                             "unexpected row order")
        col += 1
    tally.result = cube[..., col].copy()
    tally.error = cube[..., col + 1].copy()
    if has_volume:
        tally.attrs["volume"] = cube[0, 0, ..., col + 2].copy()
    return tally


def _check_bin_column(values, bounds, has_total, axis, where, label):
    """Energy/time column: upper bounds per bin, NaN (``Total``) on the last."""
    upper = np.asarray(bounds[1:], dtype=float)
    n = len(upper)
    moved = np.moveaxis(values, axis, 0)
    if has_total and not np.isnan(moved[n]).all():
        raise ValueError(f"{where}: {label} Total rows are not the last {label} block")
    real = moved[:n]
    target = upper.reshape((n,) + (1,) * (real.ndim - 1))
    if not np.allclose(real, np.broadcast_to(target, real.shape), rtol=_BIN_RTOL, atol=0):
        raise ValueError(f"{where}: {label} column does not follow the {label} bounds")


# ---------------------------------------------------------------------------
# HDF5 runtape (MCNP 6.3+)
# ---------------------------------------------------------------------------

def _read_hdf5(path: Path) -> Meshtal:
    import h5py

    with h5py.File(path, "r") as h5:
        meshtal = Meshtal(source="hdf5")
        if "config_control" in h5:
            cc = h5["config_control"].attrs
            meshtal.code = _text(cc.get("code_name", b""))
            if "version_code" in cc:
                meshtal.version = ".".join(str(int(v)) for v in cc["version_code"])
            meshtal.build_date = _text(cc.get("build_date_code", b""))
        if "results/mesh_tally" not in h5:
            raise ValueError(
                f"{path}: HDF5 file without /results/mesh_tally; MCNP writes it only "
                "for FMESH cards with out=xdmf")
        groups = h5["results/mesh_tally"]
        for key in sorted(groups, key=lambda k: int(k.rsplit("_", 1)[1])):
            tally = _hdf5_tally(groups[key], int(key.rsplit("_", 1)[1]), f"{path}, {key}")
            meshtal.tallies[tally.id] = tally
            nps = groups[key].attrs.get("number_of_normalizing_histories")
            if nps is not None:
                meshtal.nps = float(nps)
    return meshtal


def _hdf5_tally(group, tally_id: int, where: str) -> MeshTally:
    attrs = {k: _plain(v) for k, v in group.attrs.items()}
    system = str(attrs.get("coordinate_system", "")).lower()
    if system == "cartesian":
        geometry, grids = "rectangular", ("grid_x", "grid_y", "grid_z")
    elif system == "cylindrical":
        geometry, grids = "cylindrical", ("grid_r", "grid_z", "grid_t")
    else:
        raise ValueError(f"{where}: unknown coordinate_system {system!r}")

    tally = MeshTally(id=tally_id, geometry=geometry)
    number = int(attrs.get("particle_number", 0))
    tally.particle = _PARTICLE_NAMES.get(number) or (
        MCNP6_PARTICLES[number - 1] if 0 < number <= len(MCNP6_PARTICLES) else str(number))
    comments = sorted((k for k in group if k.startswith("comment_lines_")),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
    tally.comment = "\n".join(_text(group[k][()]).strip() for k in comments)

    for dim, grid in zip(SPATIAL_DIMS[geometry], grids):
        edges = np.asarray(group[grid][()], dtype=float)
        if dim == "theta" and edges.max() > 1.0 + 1e-9:
            edges = edges / (2.0 * np.pi)  # radians -> revolutions, the FMESH/MESHTAL unit
        tally.bounds[dim] = edges
    tally.energy_bounds = np.asarray(group["grid_energy"][()], dtype=float)
    tally.time_bounds = np.asarray(group["grid_time"][()], dtype=float)
    if geometry == "cylindrical":
        if "basis_axis" in group:
            tally.axis = np.asarray(group["basis_axis"][()], dtype=float)
        if "basis_vector" in group:
            tally.vec = np.asarray(group["basis_vector"][()], dtype=float)

    mean = np.asarray(group["mean"][()], dtype=float)
    rse = np.asarray(group["relative_standard_error"][()], dtype=float)
    tally.result, tally.error, tally.has_energy_total, tally.has_time_total = _hdf5_layout(
        mean, rse, tally, where)

    for key in group:
        if key in grids or key in ("mean", "relative_standard_error", "grid_energy", "grid_time",
                                   "coords", "basis_axis", "basis_vector", "basis_cross") \
                or key.startswith("comment_lines_"):
            continue
        attrs[key] = _plain(group[key][()])
    tally.attrs = attrs
    return tally


def _hdf5_layout(mean, rse, tally: MeshTally, where: str):
    """Put ``mean``/``rse`` in ``(energy, time, i, j, k)`` order.

    MCNP stores ``(time, energy, k, j, i)`` (LANL LA-UR-21-26363: transposing
    gives ``x y z e t``). The leading sizes are checked against the energy and
    time grids, allowing one extra total bin each.
    """
    if mean.ndim != 5 or mean.shape != rse.shape:
        raise ValueError(f"{where}: mean/relative_standard_error of shape {mean.shape}, expected 5-D")
    grid = tally.shape
    if mean.shape[2:] != grid[::-1]:
        raise ValueError(f"{where}: spatial shape {mean.shape[2:]} does not match the grids "
                         f"{grid} reversed")
    n_t, n_e = tally.n_time, tally.n_energy

    def fits(size, n):
        return size == n or (n > 1 and size == n + 1)

    if not (fits(mean.shape[0], n_t) and fits(mean.shape[1], n_e)):
        raise ValueError(f"{where}: leading shape {mean.shape[:2]} does not match "
                         f"(time, energy) = ({n_t}, {n_e}) bins")
    order = (1, 0, 4, 3, 2)
    return (mean.transpose(order).copy(), rse.transpose(order).copy(),
            mean.shape[1] == n_e + 1, mean.shape[0] == n_t + 1)


def _text(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii", "replace").rstrip("\x00 ")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return _text(value.item())
    return str(value)


def _plain(value):
    if isinstance(value, (bytes, np.bytes_)) or (isinstance(value, np.ndarray) and value.dtype.kind == "S"):
        return _text(value)
    if isinstance(value, np.generic):
        return value.item()
    return value

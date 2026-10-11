"""OpenMC's ``Function1D`` datasets ↔ the model's one-dimensional functions.

``function.py:79-97`` dispatches on ``@type``:

=============  ===========================================  =====================
``@type``      layout                                       model
=============  ===========================================  =====================
Tabulated1D    (2, N): x, y; ``@breakpoints`` (1-based,     :class:`XYs1d` when one
               cumulative) and ``@interpolation`` (ENDF     region, else
               codes), one per region                       :class:`Regions1d`
Polynomial     (n,): coefficients, ascending power of E/eV  :class:`Polynomial1d`
=============  ===========================================  =====================

``Sum`` (a group of ``func_1..n``) occurs only as the cross section of a
thermal ``mixed_elastic`` and is read there (:mod:`kika.hdf5.thermal_decode`).
A ``Polynomial`` has no domain in the file; the model's is ``[0, inf)`` and is
not written back.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import records as R
from .exceptions import HDF5FormatError, HDF5UnsupportedError

__all__ = ["decodeFunction", "encodeFunction", "tabulated"]


def _model():
    from kika.nuclear_data import model

    return model


def decodeFunction(node, where: str = "", label: Optional[str] = None):
    """A ``Function1D`` dataset → a model function."""
    m = _model()
    if not isinstance(node, R.Dataset):
        kind = R.text(R.attr(node, "type", b"group"))
        raise HDF5UnsupportedError(f"a {kind} group is not a one-dimensional function "
                                   f"kika reads here [{where}]")
    kind = R.text(R.attr(node, "type", b""))
    data = node.data
    if kind == "Tabulated1D":
        if data.ndim != 2 or data.shape[0] != 2:
            raise HDF5FormatError(f"Tabulated1D of shape {data.shape}", node=where)
        breakpoints = np.asarray(R.attr(node, "breakpoints"), dtype=int)
        codes = np.asarray(R.attr(node, "interpolation"), dtype=int)
        x, y = data[0], data[1]
        if len(breakpoints) == 1 and int(breakpoints[0]) == len(x):
            return m.XYs1d(xs=x, ys=y, interpolation=m.ENDF_INT_TO_INTERPOLATION[int(codes[0])],
                           label=label)
        return m.Regions1d.fromEndfRegions(x, y, list(zip(breakpoints, codes)), label=label)
    if kind == "Polynomial":
        return m.Polynomial1d(coefficients=data, domainMin_=0.0, domainMax_=float("inf"),
                              label=label)
    raise HDF5UnsupportedError(f"Function1D type {kind!r} is not read here [{where}]")


def tabulated(x, y, breakpoints, codes) -> R.Dataset:
    """A ``Tabulated1D`` dataset with OpenMC's dtypes."""
    dataset = R.Dataset.of(np.vstack([np.asarray(x, float), np.asarray(y, float)]))
    dataset.attrs["type"] = R.Attribute.of("Tabulated1D")
    dataset.attrs["breakpoints"] = R.Attribute.of(np.asarray(breakpoints, dtype=np.int64))
    dataset.attrs["interpolation"] = R.Attribute.of(np.asarray(codes, dtype=np.int64))
    return dataset


def encodeFunction(function) -> R.Dataset:
    """A model function → a ``Function1D`` dataset."""
    m = _model()
    if isinstance(function, m.XYs1d):
        return tabulated(function.xs, function.ys, [len(function)], [function.endfInterpolationCode])
    if isinstance(function, m.Regions1d):
        x, y, pairs = function.toEndfRegions()
        return tabulated(x, y, [p[0] for p in pairs], [p[1] for p in pairs])
    if isinstance(function, m.Polynomial1d):
        dataset = R.Dataset.of(np.asarray(function.coefficients, dtype=float))
        dataset.attrs["type"] = R.Attribute.of("Polynomial")
        return dataset
    if isinstance(function, m.Constant1d):
        dataset = R.Dataset.of(np.asarray([function.constant], dtype=float))
        dataset.attrs["type"] = R.Attribute.of("Polynomial")
        return dataset
    raise HDF5UnsupportedError(
        f"a {type(function).__name__} has no Function1D layout; OpenMC reads "
        f"Tabulated1D and Polynomial")

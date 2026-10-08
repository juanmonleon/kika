"""Owned, validated table snapshots for repeated evaluation.

No global cache and no reference to mutable source arrays. Values and limits
use the same evaluator as the ordinary algebra entry points.
"""
from dataclasses import dataclass
import numpy as np
from .laws import validate
from .evaluate import _read_checked, OUTSIDE


def _owned_readonly(array):
    # A bytes-backed view cannot be made writable with setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class _Evaluator:
    _x: object
    _y: object
    _laws: object

    def _read(self, q, side, outside):
        if outside not in OUTSIDE:
            raise ValueError(f'outside must be one of {OUTSIDE}, got {outside!r}')
        return _read_checked(self._x, self._y, self._laws, q, side, outside)

    def __call__(self, q, outside='zero'):
        return self._read(q, 'point', outside)

    def left_limit(self, q, outside='zero'):
        return self._read(q, 'left', outside)

    def right_limit(self, q, outside='zero'):
        return self._read(q, 'right', outside)


def prepare_evaluator(x, y, laws):
    """Validate and own a 1-d table once; return a callable snapshot.

    Later edits to the source do not affect the snapshot. Domain rules,
    repeated nodes, endpoint values, and one-sided limits match ``evaluate``,
    ``left_limit`` and ``right_limit``. All interpolation remains in algebra.
    """
    x, y, laws = validate(x, y, laws)
    if y.ndim != 1:
        raise ValueError('prepared evaluation requires one-dimensional ordinates')
    return _Evaluator(*(_owned_readonly(a) for a in (x, y, laws)))

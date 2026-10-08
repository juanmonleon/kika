"""One-dimensional interpolation on ENDF/GNDS ``(NBT, INT)`` regions.

An adapter, not an implementation: it turns ENDF's cumulative ``(NBT, INT)``
pairs into one law per interval and hands the table to
:func:`kika.algebra.evaluate`, which is the one evaluator of a tabulated
function in kika. Nothing is interpolated here.

``kika.endf.utils.interpolate_1d_endf`` stays as a live re-export of
:func:`interpolate_1d`: eight call sites inside ``kika/endf`` import that name.

What changed when the evaluation moved to :mod:`kika.algebra` (7-oct-2026):
a log law over a negative value, and any code outside 1-5 (INT=6, the
two-dimensional 11-25), now raise instead of being read lin-lin without a word;
and the last point of a histogram reads its own tabulated value. A log-y law
with an end at exactly 0 is read as the limit of the law -- 0 inside the panel,
the jump at the other end (:func:`kika.algebra.laws.vanishing_panels`, 8-oct-2026):
JEFF-4.0 writes it in the MT102 of 15 tapes, which the first rule refused.
"""
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from kika.algebra import evaluate, interval_laws

__all__ = ["interpolate_1d"]


def interpolate_1d(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    nbt_int_pairs: Sequence[Tuple[int, int]],
    xq: Union[float, ArrayLike],
    out_of_range: str = "zero",
) -> Union[float, np.ndarray]:
    """The table ``(x_grid, y_grid)`` on its ``(NBT, INT)`` regions, at *xq*.

    *out_of_range* is ``'zero'`` (the default), ``'hold'`` (the end value
    continues) or ``'raise'``. Inside, the value is right-continuous at a
    repeated abscissa and closed at both ends; see :func:`kika.algebra.evaluate`.
    """
    x = np.asarray(x_grid, dtype=float)
    return evaluate(x, y_grid, interval_laws(x.size, nbt_int_pairs), xq,
                    outside=out_of_range)

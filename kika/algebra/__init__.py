"""Algebra of tabulated one-dimensional functions.

A tabulated function here is three arrays: abscissae ``x`` (non-decreasing),
ordinates ``y``, and one interpolation law per *interval*, ``laws``, with the
ENDF/GNDS codes 1-5 (histogram, lin-lin, lin-log, log-lin, log-log). Nothing
else: no units, no reaction numbers, no file formats, no physics. Those live
above this package, which **imports nothing from kika** -- ``kika/tests/
test_layering.py`` enforces it -- so the model, the format adapters and the
physics can all build on it without a cycle.

The rules every function in the package keeps:

* **One implementation per operation.** Evaluating, integrating or summing a
  table is done here or nowhere.
* **Exact, or it raises.** Each law is evaluated and integrated in closed form.
  What has no closed form is first re-expressed as lin-lin to a tolerance the
  caller states (:func:`to_linlin`). A log law on a non-positive value is an
  error, never a silent lin-lin.
* **A discontinuity is a repeated abscissa.** ``(x, y_left), (x, y_right)`` is
  a step, kept as such by every operation; nothing turns it into a ramp.
* **Zero outside the domain**, unless the caller asks otherwise.
* **Speed from numpy, pass by pass.** Recursive refinement is a sequence of
  vectorised passes over every panel still being refined.

Submodules: :mod:`.laws` (law codes and their ENDF ``(NBT, INT)`` spelling),
:mod:`.evaluate` (values and one-sided limits), :mod:`.grid` (unions of grids,
discontinuities, regions), :mod:`.integrate` (closed-form integrals, group
integrals, 1/x weight), :mod:`.refine` (the adaptive refinement engine and
:func:`to_linlin`), :mod:`.arithmetic` (sums).
"""
from .laws import (HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, LAWS,
                   interval_laws, pairs_from_laws, validate)
from .evaluate import (evaluate, interpolate_between, left_limit, right_limit,
                       sample_on_union)
from .grid import discontinuities, join_pieces, split_at_discontinuities, union
from .integrate import (cumulative_integral, group_integrals, integral,
                        panel_integrals)
from .refine import RefinementError, RefineResult, refine, to_linlin
from .arithmetic import add

__all__ = [
    "HISTOGRAM", "LINLIN", "LINLOG", "LOGLIN", "LOGLOG", "LAWS",
    "interval_laws", "pairs_from_laws", "validate",
    "evaluate", "interpolate_between", "left_limit", "right_limit", "sample_on_union",
    "discontinuities", "join_pieces", "split_at_discontinuities", "union",
    "cumulative_integral", "group_integrals", "integral", "panel_integrals",
    "RefinementError", "RefineResult", "refine", "to_linlin",
    "add",
]

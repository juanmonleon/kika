"""Algebra of tabulated one-dimensional functions.

A tabulated function here is three arrays: abscissae ``x`` (non-decreasing),
ordinates ``y``, and one interpolation law per *interval*, ``laws``, with the
ENDF/GNDS codes 1-5 (histogram, lin-lin, lin-log, log-lin, log-log). Nothing
else: no units, no reaction numbers, no file formats, no physics. Those live
above this package, which **imports nothing from kika** -- ``kika/tests/
test_layering.py`` enforces it -- so the model, the format adapters and the
physics can all build on it without a cycle.

**This package is mathematics only, and that is its purpose.** A function here
takes and returns numbers, tables and laws: no energies or cross sections as
such, no flux, no reaction, no evaluation, no physical model or constant. The
law codes are ENDF's numbering of the five interpolation laws, which is a
convention for the mathematics, not physics. A docstring may say *where* a rule
was learned -- a tape that broke a sum, a processor that disagreed -- but
nothing here depends on it. Something that needs a physical model (the ENDF
fission-spectrum laws, a resonance formalism, a flux shape with physical
parameters) belongs in the calculation layer that builds on this one: the
spectra moved to :mod:`kika.nuclear_data.spectrum_laws` in October 2026 for
that reason. ``tests/test_algebra.py`` freezes the list of submodules so that a
new one is added on purpose.

The rules every function in the package keeps:

* **One implementation per operation.** Evaluating, integrating or summing a
  table is done here or nowhere.
* **Exact, or it raises.** Each law is evaluated and integrated in closed form.
  What has no closed form is first re-expressed as lin-lin to a tolerance the
  caller states (:func:`to_linlin`). A log law on a non-positive value is an
  error, never a silent lin-lin. A log-y panel with an end at exactly 0 (and
  none below) is read as the limit of its own law, not refused: 0 inside the
  panel, with the jump at the non-zero end (:func:`~kika.algebra.laws.vanishing_panels`).
  It is what NJOY's ``terp1`` gives, and JEFF-4.0 writes it (MT102 from 0 to 0
  under the resolved range, and from 3.7e-4 b to 0 between 30 and 200 MeV).
* **A discontinuity is a repeated abscissa.** ``(x, y_left), (x, y_right)`` is
  a step, kept as such by every operation; nothing turns it into a ramp.
* **Zero outside the domain**, unless the caller asks otherwise.
* **Speed from numpy, pass by pass.** Recursive refinement is a sequence of
  vectorised passes over every panel still being refined.

Submodules: :mod:`.laws` (law codes and their ENDF ``(NBT, INT)`` spelling),
:mod:`.evaluate` (values and one-sided limits), :mod:`.grid` (unions of grids,
discontinuities, regions), :mod:`.integrate` (closed-form integrals, group
integrals, 1/x weight), :mod:`.refine` (the adaptive refinement engine and
:func:`to_linlin`), :mod:`.arithmetic` (sums, and tables read on one common
grid to compare them), :mod:`.fold` (Gaussian folds).
"""
from .laws import (HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, LAWS,
                   interval_laws, laws_on_refinement, pairs_from_laws,
                   validate)
from .evaluate import (evaluate, interpolate_between, left_limit, right_limit,
                       sample_on_union)
from .grid import compress_flat, discontinuities, join_pieces, split_at_discontinuities, union
from .integrate import (cumulative_integral, group_averages, group_integrals,
                        integral, interval_averages, interval_integrals,
                        legendre_coefficients, legendre_moments, panel_integrals)
from .fold import box_gaussian_fold_nodes, fold_tabulated, gaussian_fold_nodes
from .refine import RefinementError, RefineResult, refine, to_linlin
from .arithmetic import add, on_common_grid
from .prepared import prepare_evaluator

__all__ = [
    "HISTOGRAM", "LINLIN", "LINLOG", "LOGLIN", "LOGLOG", "LAWS",
    "interval_laws", "laws_on_refinement", "pairs_from_laws", "validate",
    "evaluate", "interpolate_between", "left_limit", "right_limit", "sample_on_union",
    "compress_flat", "discontinuities", "join_pieces", "split_at_discontinuities", "union",
    "cumulative_integral", "group_averages", "group_integrals", "integral",
    "interval_averages", "interval_integrals",
    "legendre_coefficients", "legendre_moments", "panel_integrals", "box_gaussian_fold_nodes", "fold_tabulated", "gaussian_fold_nodes",
    "RefinementError", "RefineResult", "refine", "to_linlin",
    "add", "on_common_grid", "prepare_evaluator",
]

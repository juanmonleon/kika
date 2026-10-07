"""Exact integrals of a tabulated 1-d function, under the law it states.

**Why the model needs this and did not have it.** MF35 is the covariance of the
*group-integrated probabilities* of an MF5 spectrum — measured, not assumed; see
``docs/pfns/pfns_mf5_mf35_roadmap.md`` fact 4 — so perturbing a fission spectrum
from its own covariance means integrating the node over the covariance's groups,
scaling, and integrating again to check what was written. Nothing in
:mod:`kika.nuclear_data.model.functions` could integrate anything: a
:class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` knew how to *evaluate*
itself and nothing else, so every integral of a model node lived in whichever
format class happened to need one.

``MF5PartialTabulated`` had exactly the four operations the node lacked —
``group_integrals``, ``normalisation``, ``table``, ``replace_table`` — which is
why perturbing MF5 was format work by construction. These functions and the
2-d methods that use them are what moves that capability onto the model; the
arithmetic underneath is :mod:`kika.algebra`, the same functions the ENDF
class calls, so the two cannot drift.

**Exact under every law.** The whole normalisation argument of a PFNS draw
rests on the integral being the evaluator's own rather than a quadrature of it.
Each of the five laws integrates in closed form (:mod:`kika.algebra.integrate`);
until 7-oct-2026 only histogram and lin-lin did, and a log-interpolated table
was refused.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ....algebra import evaluate, group_integrals, integral, interval_laws

__all__ = ["tabulateFunction1d", "integrateFunction1d", "groupIntegralsOf",
           "evaluateExactly"]


def tabulateFunction1d(function1d, what: str = "") -> Tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """``(xs, ys, per-interval INT codes)`` of a tabulated 1-d function.

    Works off :meth:`toEndfRegions`, which both
    :class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` and
    :class:`~kika.nuclear_data.model.functions.regions1d.Regions1d` answer, so
    one region and many are the same call. A node that has no such method —
    a Legendre child, a polynomial, an isotropic distribution — is not a table
    and is refused by name rather than duck-typed into a wrong answer.
    """
    what = what or type(function1d).__name__
    toRegions = getattr(function1d, "toEndfRegions", None)
    if toRegions is None:
        raise TypeError(
            f"{what} is a {type(function1d).__name__}, which is not a tabulated "
            f"function: it has no grid to integrate over. Only XYs1d and "
            f"Regions1d carry one"
        )
    xs, ys, pairs = toRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return xs, ys, interval_laws(xs.size, pairs)


def integrateFunction1d(function1d, domainMin: Optional[float] = None,
                        domainMax: Optional[float] = None,
                        what: str = "") -> float:
    """``int_{domainMin}^{domainMax} f(x) dx``, exactly on the stated law.

    Either limit may fall inside a panel, and either may be outside the table
    altogether — the part outside contributes nothing, which is the convention
    a probability density stated on a finite grid needs: the function *is* zero
    there, it is not merely unknown.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    return integral(xs, ys, codes, domainMin, domainMax)


def groupIntegralsOf(function1d, boundaries: ArrayLike,
                     what: str = "") -> np.ndarray:
    """``P_j = int_{g_j}^{g_j+1} f`` for every group of *boundaries*.

    Each edge cuts its panel into two of the same law, valued by that law, so
    nothing is sampled (:func:`kika.algebra.group_integrals`). That is what
    makes ``P_j`` the same number the
    evaluator's own integral gives — which matters here because it is the
    quantity an MF35 matrix is the covariance *of*, not a discretisation of it.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    return group_integrals(xs, ys, codes, boundaries)


def evaluateExactly(function1d, points: ArrayLike, what: str = "") -> np.ndarray:
    """*function1d* at *points*, zero outside its own support.

    The same evaluator :meth:`XYs1d.evaluate` reaches, :func:`kika.algebra.evaluate`,
    and the same one the integrals cut their panels with -- so a node inserted
    into a table cannot disagree with the table's integral by a rounding of a
    different code path.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    return evaluate(xs, ys, codes, np.asarray(points, dtype=float))

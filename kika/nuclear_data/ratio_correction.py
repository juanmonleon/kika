r"""Correct an evaluation with a measurement through smooth ratios (the *ratio method*).

Roadmap G4NDL §0bis, Fase 13.  A measurement of elastic scattering sees the
evaluation through its energy resolution (:mod:`kika.nuclear_data.forward`), so
pasting the measured numbers into a library broadens the resonances twice --
once by the experiment, once more by the transport code's own Doppler -- and
erases exactly the structure that governs self-shielding.  Here the evaluation
at 0 K keeps its structure, and the measurement fixes only what it can see:

* **energy** -- :math:`\sigma_\text{new}(E) = \sigma(E)\,r(E)`, with
  :math:`r_i = \sigma^\text{meas}_i / \langle\sigma\rangle_i` smoothed over a
  width never narrower than the reading itself (bin through resolution);
* **angle** -- :math:`f_\text{new}(\mu, E) = f(\mu, E) + \sum_{l=1}^{L}
  \tfrac{2l+1}{2}\,\delta a_l(E)\,P_l(\mu)`, additive in the distribution's own
  Legendre coefficients, with :math:`\delta a_l` fitted in each measured energy
  to the residual DCS and smoothed in energy the same way.

**Why additive (D13-1, 9-oct, provisional).**  At fixed σ the folded DCS is
*linear* in f, so the residual is fitted as it is, without dividing by the
evaluation.  A multiplicative :math:`c(\mu)` divides by the DCS exactly in the
diffraction minima, where the data and the evaluations disagree most, and a
high-degree c fitted on the detectors' cosines extrapolates badly towards
:math:`\mu \to 1`, where f is largest and no detector sits (closure of 8-oct,
roadmap Fase 13).  With :math:`\delta a_l` constant over a reading the response
is exact: :math:`\langle\sigma\rangle_i\,J\,\tfrac{2l+1}{2}P_l(\mu_i)/2\pi`.

**Not inside the resolved resonance region.**  There the shape comes from the
resonance parameters, which a ratio smooth in energy cannot recreate (closure of
8-oct: χ²/N 233 in 1-1.5 MeV).  ``emin_ev`` defaults to the top of the suite's
resolved region, and measured energies whose correction would reach below it are
not used.

**Level and shape are separate.**  Each energy fits a level :math:`\lambda`
(DCS × (1 + λ)) beside the shape, so a level mismatch does not leak into the
:math:`\delta a_l`.  By default (``level="separate"``) λ is fitted and dropped:
the level enters through :math:`r(E)` and must not be applied twice.
``level="from_dcs"`` also multiplies σ by :math:`1 + \lambda(E)`: the level from
the differential data, which is what an angle-integrated σ built from the same
detectors amounts to.

**Smoothing.**  A local-linear kernel regression, weighted by
:math:`1/\delta r_i^2`, with a Gaussian kernel whose width at energy :math:`E`
is ``factor`` times the reading's own width
:math:`s(E) = \sqrt{\sigma_K(E)^2 + w(E)^2/12}` (:math:`\sigma_K` from
:func:`kika.nuclear_data.forward.kernel_sigma_ev`, :math:`w` the bin).  The
factor is chosen by leave-one-out cross-validation among values that keep the
kernel at least ``min_width`` FWHM of the reading wide, so the ratio cannot put
back structure the experiment does not resolve.

**Outside the measured range** the edge value is held over the reach of the
edge reading (``READING_REACH`` reading widths, which that reading folds), and
then goes back to one over a ramp of one reading width (FWHM).

**Iterated.**  The correction is first order: :math:`\langle\sigma r\rangle =
r\langle\sigma\rangle` only if r is flat inside the kernel.  Each iteration
re-reads the corrected suite and corrects the residual ratio, with the width
chosen in the first one, until the residual moves no bin by more than
``tolerance`` of its own statistical uncertainty.

**What the method trusts.**  The *positions* of the evaluation's resonances: a
displaced resonance makes r oscillate, and the smoothing flattens it rather
than moving it.  That is a limit to report, not to hide.

The corrected suite is a deep copy; the input is never modified.  The σ that is
corrected is the pointwise one the forward operator reads (``recon`` when the
suite has it), and it is written on a new grid: the evaluation's own points plus
whatever the product with r needs to stay lin-lin within ``grid_tolerance``.
Energies are in eV, cross sections in b, differential ones in b/sr.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.forward import (
    ElasticView, ForwardSetup, _detector_cosines, forward_dcs, forward_sigma, kernel_sigma_ev,
)

__all__ = [
    "SmoothRatio",
    "fit_smooth_ratio",
    "CrossSectionCorrection",
    "AngularCorrection",
    "correct_cross_section",
    "correct_angular",
]

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
#: Reading widths past the first and last measured energy over which their ratio
#: is held before the ramp to one starts: what the edge readings themselves fold.
READING_REACH = 3.0
ELASTIC_MT = 2
Level = Literal["separate", "from_dcs"]


# =============================================================================
# The smoother
# =============================================================================

def _local_linear(x_eval, h_eval, x, y, w, chunk: int = 4096):
    """Local-linear kernel regression of ``y(x)`` with weights ``w`` at ``x_eval``.

    ``y`` may be ``(n,)`` or ``(n, k)``: k series smoothed with the same kernel.
    """
    x_eval = np.atleast_1d(np.asarray(x_eval, float))
    h_eval = np.broadcast_to(np.asarray(h_eval, float), x_eval.shape)
    y2 = y[:, None] if y.ndim == 1 else y
    out = np.empty((x_eval.size, y2.shape[1]))
    for s in range(0, x_eval.size, chunk):
        d = x[None, :] - x_eval[s:s + chunk, None]
        k = w[None, :] * np.exp(-0.5 * (d / h_eval[s:s + chunk, None]) ** 2)
        s0, s1, s2 = k.sum(1), (k * d).sum(1), (k * d * d).sum(1)
        t0, t1 = k @ y2, (k * d) @ y2
        det = s0 * s2 - s1 * s1
        ok = det > 1e-300 * np.maximum(s0 * s2, 1e-300)
        num = s2[:, None] * t0 - s1[:, None] * t1
        out[s:s + chunk] = np.where(ok[:, None], num / np.where(ok, det, 1.0)[:, None],
                                    t0 / np.maximum(s0, 1e-300)[:, None])
    return out[:, 0] if y.ndim == 1 else out


def _loo_score(x, y, w, h) -> float:
    """Weighted leave-one-out residual of the local-linear fit at its own nodes."""
    d = x[None, :] - x[:, None]
    k = w[None, :] * np.exp(-0.5 * (d / h[:, None]) ** 2)
    s0, s1, s2 = k.sum(1), (k * d).sum(1), (k * d * d).sum(1)
    det = s0 * s2 - s1 * s1
    yhat = (s2 * (k @ y) - s1 * ((k * d) @ y)) / det
    lii = w * s2 / det
    resid = (y - yhat) / (1.0 - lii)
    return float(np.sum(w * resid ** 2) / np.sum(w))


@dataclass
class SmoothRatio:
    r"""A ratio (or several) smoothed in energy, and how it was smoothed.

    ``energies``/``values``/``errors`` are the raw ratios, ``factor`` the chosen
    kernel width in units of the reading width, ``factors``/``scores`` the
    cross-validation curve.  Calling it gives the smoothed ratio at any energy,
    ramped to ``identity`` outside the measured range.
    """

    energies: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    reading_width: Callable[[np.ndarray], np.ndarray]
    factor: float
    factors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    identity: np.ndarray = field(default_factory=lambda: np.ones(1))

    @property
    def support(self) -> Tuple[float, float]:
        """``(lo, hi)``: where the correction is not identity (measured range plus ramps)."""
        lo, hi = float(self.energies[0]), float(self.energies[-1])
        return (lo - self.hold(lo) - self.ramp(lo), hi + self.hold(hi) + self.ramp(hi))

    def hold(self, energy: float) -> float:
        """How far past an edge its value is kept: the reach of the edge reading."""
        return float(READING_REACH * self.reading_width(np.array([energy]))[0])

    def ramp(self, energy: float) -> float:
        return float(FWHM_PER_SIGMA * self.reading_width(np.array([energy]))[0])

    def kernel_width(self, energies) -> np.ndarray:
        return _kernel_width(self.factor, self.reading_width, self.energies, energies)

    def anchors(self) -> np.ndarray:
        """Energies where the smoothed ratio must be sampled to be followed lin-lin:
        the measured range at a quarter of the kernel width, and the two ramp ends."""
        lo, hi = float(self.energies[0]), float(self.energies[-1])
        if lo == hi:
            return np.array([self.support[0], lo - self.hold(lo), lo,
                             hi + self.hold(hi), self.support[1]])
        pts = [lo]
        while pts[-1] < hi:
            pts.append(pts[-1] + 0.25 * float(self.kernel_width([pts[-1]])[0]))
        pts[-1] = hi
        return np.concatenate(([self.support[0], lo - self.hold(lo)], pts,
                               [hi + self.hold(hi), self.support[1]]))

    def __call__(self, energies) -> np.ndarray:
        e = np.atleast_1d(np.asarray(energies, float))
        ident = np.asarray(self.identity, float)
        vals = self.values if self.values.ndim > 1 else self.values[:, None]
        out = np.broadcast_to(ident, (e.size, vals.shape[1])).astype(float).copy()
        lo, hi = float(self.energies[0]), float(self.energies[-1])
        inside = (e >= lo) & (e <= hi)
        if self.energies.size == 1:
            edge_lo = edge_hi = vals[0]
        else:
            w = 1.0 / self.errors.reshape(len(self.energies), -1).mean(1) ** 2
            if np.any(inside):
                out[inside] = _local_linear(e[inside], self.kernel_width(e[inside]),
                                            self.energies, vals, w)
            edge_lo, edge_hi = _local_linear(np.array([lo, hi]), self.kernel_width([lo, hi]),
                                             self.energies, vals, w)
        if self.energies.size == 1:
            out[e == lo] = vals[0]
        for edge, val, sign in ((lo, edge_lo, -1.0), (hi, edge_hi, 1.0)):
            hold, ramp = self.hold(edge), self.ramp(edge)
            past = sign * (e - edge)
            out[(past > 0) & (past <= hold)] = val
            sel = (past > hold) & (past < hold + ramp)
            t = ((past[sel] - hold) / ramp)[:, None]
            out[sel] = (1.0 - t) * val + t * ident
        return out[:, 0] if self.values.ndim == 1 else out


def _kernel_width(factor: float, reading_width: Callable, measured: np.ndarray, energies):
    """``factor`` reading widths, but never narrower than the spacing of the measured
    energies: a kernel that does not reach the next point steps between them."""
    x = np.atleast_1d(np.asarray(energies, float))
    h = factor * reading_width(x)
    if measured.size > 1:
        h = np.maximum(h, np.interp(x, measured, np.gradient(measured)))
    return h


def fit_smooth_ratio(energies, values, errors, reading_width: Callable, *,
                     min_width: float = 1.0, factors: Optional[Sequence[float]] = None,
                     identity=None) -> SmoothRatio:
    """Smooth ``values(energies)`` with the width chosen by leave-one-out CV.

    ``reading_width(E)`` is the one-sigma width of a reading at ``E`` (bin through
    resolution); the kernel is never narrower than ``min_width`` FWHM of it, nor
    than the spacing of the measured energies.
    ``values`` may be ``(n, k)``: k series share one kernel, chosen on their sum
    of scores.
    """
    e = np.asarray(energies, float)
    order = np.argsort(e)
    e, v, err = e[order], np.asarray(values, float)[order], np.asarray(errors, float)[order]
    ident = np.ones(1) if identity is None else np.asarray(identity, float)
    if e.size == 1:
        return SmoothRatio(e, v, err, reading_width, float(min_width * FWHM_PER_SIGMA),
                           identity=ident)
    floor = min_width * FWHM_PER_SIGMA
    cand = (np.geomspace(floor, 30.0 * floor, 25) if factors is None
            else np.array([f for f in factors if f >= floor], float))
    if cand.size == 0:
        raise ValueError(f"no candidate factor is at least min_width*FWHM = {floor:.3g}")
    v2 = v if v.ndim > 1 else v[:, None]
    e2 = err if err.ndim > 1 else err[:, None]
    scores = np.array([sum(_loo_score(e, v2[:, j], 1.0 / e2[:, j] ** 2,
                                      _kernel_width(f, reading_width, e, e))
                           for j in range(v2.shape[1])) for f in cand])
    best = float(cand[int(np.argmin(scores))])
    return SmoothRatio(e, v, err, reading_width, best, cand, scores, identity=ident)


def _reading_width(view: ElasticView, setup: ForwardSetup, energies, bins):
    """``E -> s(E)`` from the measurement's own energies and bins."""
    e = np.asarray(energies, float)
    if bins is None:
        bw = np.zeros_like(e)
    else:
        bw = np.array([0.0 if b is None else float(b[1]) - float(b[0]) for b in bins])
    order = np.argsort(e)
    e, bw = e[order], bw[order]

    def width(x):
        x = np.atleast_1d(np.asarray(x, float))
        sk = np.array([kernel_sigma_ev(setup, float(xi), view.awr) for xi in x])
        return np.sqrt(sk ** 2 + np.interp(x, e, bw) ** 2 / 12.0)
    return width


# =============================================================================
# Applying a ratio to the model
# =============================================================================

def _cross_section_label(suite, label: Optional[str]) -> str:
    labels = list(suite.reactions[ELASTIC_MT].crossSection.keys())
    return label or ("recon" if "recon" in labels else "eval")


def _multiply_cross_section(suite, label: str, ratio: Callable, anchors: np.ndarray,
                            grid_tolerance: float) -> Tuple[int, int]:
    """σ ← σ·ratio in place, on the union of σ's grid and ``anchors``, refined lin-lin."""
    from kika.algebra import refine
    from kika.nuclear_data.model.functions import XYs1d

    container = suite.reactions[ELASTIC_MT].crossSection
    form = container[label]
    x, y, pairs = form.toEndfRegions()
    if {int(c) for _, c in pairs} - {2}:
        raise NotImplementedError("only a lin-lin cross section is corrected")
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = anchors[(anchors > x[0]) & (anchors < x[-1])]
    a = a[~np.isin(a, x)]
    xs = np.sort(np.concatenate([x, a]), kind="stable")
    ys = np.interp(xs, x, y)
    # a step (repeated abscissa) in σ keeps its two values
    for i in np.flatnonzero(np.diff(x) == 0):
        j = np.searchsorted(xs, x[i])
        ys[j], ys[j + 1] = y[i], y[i + 1]

    def product(q, owner):
        return np.interp(q, x, y) * ratio(q)

    def exceeds(actual, chord):
        return np.abs(actual - chord) / (grid_tolerance * np.abs(actual) + 1e-300)

    lo, hi = anchors[0], anchors[-1]
    active = (xs[1:] > lo) & (xs[:-1] < hi) & (np.diff(xs) > 0)
    yr = ys * ratio(xs)
    res = refine(xs, yr, product, exceeds, active=active, max_points=5_000_000)
    axes = getattr(form, "axes", None)
    container[label] = XYs1d(res.x, res.y, axes=axes, label=getattr(form, "label", None))
    return int(x.size), int(res.x.size)


def _angular_form(suite):
    from kika.nuclear_data.model.distributions import AngularTwoBody

    dist = suite.reactions[ELASTIC_MT].outputChannel.products.byPid("n")[0].distribution["eval"]
    if not isinstance(dist, AngularTwoBody):
        raise NotImplementedError(f"elastic distribution of type {type(dist).__name__}")
    return dist


def _leaves(node):
    from kika.nuclear_data.model.functions import Regions2d, XYs2d

    if isinstance(node, XYs2d):
        return [node]
    if isinstance(node, Regions2d):
        return [leaf for child in node.function2ds for leaf in _leaves(child)]
    raise NotImplementedError(f"angular node of type {type(node).__name__}")


def _shape_series(delta: np.ndarray) -> np.ndarray:
    """``δa_1..δa_L`` as the Legendre series of δf: ``Σ (2l+1)/2 δa_l P_l`` (no l = 0)."""
    l = np.arange(1, delta.size + 1)
    return np.r_[0.0, delta * (2 * l + 1) / 2.0]


def _legendre_plus(a: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """``a`` (ENDF a_l, a_0 = 1) plus ``δa_1..δa_L``: exact, and still normalised."""
    n = max(a.size, delta.size + 1)
    return np.pad(a, (0, n - a.size)) + np.pad(np.r_[0.0, delta], (0, n - delta.size - 1))


def _table_plus(f1d, delta: np.ndarray, tol: float):
    """A tabulated f(μ) plus ``Σ (2l+1)/2 δa_l P_l``, refined lin-lin.

    The added series integrates to zero; the renormalisation only removes what
    the lin-lin table loses to its own chords.
    """
    from numpy.polynomial import legendre as L

    from kika.algebra import integral, refine
    from kika.nuclear_data.model.functions import XYs1d

    mu, p, pairs = f1d.toEndfRegions()
    if {int(c) for _, c in pairs} - {2}:
        raise NotImplementedError("only a lin-lin angular table is corrected")
    mu, p = np.asarray(mu, float), np.asarray(p, float)
    series = _shape_series(delta)

    def total(q, owner):
        return np.interp(q, mu, p) + L.legval(q, series)

    res = refine(mu, total(mu, None), total,
                 lambda act, ch: np.abs(act - ch) / (tol * np.abs(act) + 1e-12))
    norm = float(integral(res.x, res.y, np.full(res.x.size - 1, 2)))
    return XYs1d(res.x, res.y / norm, axes=f1d.axes, label=f1d.label,
                 outerDomainValue=f1d.outerDomainValue)


def _record_at(leaf, energy: float):
    """The leaf's record at ``energy`` (lin-lin between its neighbours)."""
    from kika.nuclear_data.model.functions import Legendre, XYs1d

    fs = list(leaf.function1ds)
    es = np.array([float(f.outerDomainValue) for f in fs])
    hit = np.flatnonzero(es == energy)
    if hit.size:
        return copy.deepcopy(fs[int(hit[-1])])
    k = int(np.searchsorted(es, energy))
    f0, f1 = fs[k - 1], fs[k]
    t = (energy - es[k - 1]) / (es[k] - es[k - 1])
    if isinstance(f0, Legendre):
        n = max(f0.coefficients.size, f1.coefficients.size)
        c0 = np.pad(f0.coefficients, (0, n - f0.coefficients.size))
        c1 = np.pad(f1.coefficients, (0, n - f1.coefficients.size))
        return Legendre((1 - t) * c0 + t * c1, axes=f0.axes, outerDomainValue=energy)
    mu = np.union1d(np.asarray(f0.xs, float), np.asarray(f1.xs, float))
    p = (1 - t) * np.interp(mu, f0.xs, f0.ys) + t * np.interp(mu, f1.xs, f1.ys)
    return XYs1d(mu, p, axes=f0.axes, outerDomainValue=energy)


def _add_angular(suite, delta: Callable, anchors: np.ndarray,
                 table_tolerance: float, positivity_points: int):
    """f ← f + Σ (2l+1)/2 δa_l(E) P_l in place, ``delta(E) -> (n, L)``.

    Returns ``(problems, inherited, touched)``: records the correction made
    negative as ``(E, f_min)``, records the evaluation already had negative and
    the correction did not deepen as ``(E, f_min before, f_min after)``, and the
    number of records changed.
    """
    from kika.nuclear_data.model.functions import Legendre

    dist = _angular_form(suite)
    lo, hi = float(anchors[0]), float(anchors[-1])
    problems, inherited, touched = [], [], 0
    mu_check = np.linspace(-1.0, 1.0, positivity_points)
    for leaf in _leaves(dist.angular):
        es = np.array([float(f.outerDomainValue) for f in leaf.function1ds])
        new_e = anchors[(anchors > es[0]) & (anchors < es[-1]) & ~np.isin(anchors, es)]
        energies = np.sort(np.concatenate([es, new_e]), kind="stable")
        records = []
        seen = {}
        for e in energies:
            if e in new_e:
                rec = _record_at(leaf, float(e))
            else:
                idx = seen.get(e, -1) + 1
                where = np.flatnonzero(es == e)
                rec = leaf.function1ds[int(where[min(idx, where.size - 1)])]
                seen[e] = idx
            if lo < e < hi or e in (lo, hi):
                d = np.asarray(delta(np.array([e]))[0], float)
                fmin_before = float(np.min(rec.evaluate(mu_check)))
                if isinstance(rec, Legendre):
                    rec = Legendre(_legendre_plus(np.asarray(rec.coefficients, float), d),
                                   axes=rec.axes, label=rec.label,
                                   outerDomainValue=rec.outerDomainValue)
                else:
                    rec = _table_plus(rec, d, table_tolerance)
                fmin = float(np.min(rec.evaluate(mu_check)))
                # A record the evaluation already had negative is reported, not
                # blamed on the correction, unless the correction deepens it.
                if fmin < 0.0 and fmin_before < 0.0 and fmin >= fmin_before:
                    inherited.append((float(e), fmin_before, fmin))
                elif fmin < 0.0:
                    problems.append((float(e), fmin))
                touched += 1
            records.append(rec)
        leaf.function1ds = records
    return problems, inherited, touched


# =============================================================================
# The two corrections
# =============================================================================

@dataclass
class CrossSectionCorrection:
    """What :func:`correct_cross_section` did.

    ``ratio`` is the cumulative smoothed r(E) (the product of every iteration's);
    ``first`` the first iteration's fit, with the CV curve; ``chi2`` χ²/N of the
    folded suite against the data before correcting and after each iteration;
    ``max_pull`` per iteration the largest |r_k - 1| over δr_k, the convergence
    statistic; ``grid`` the number of σ points before and after.
    """

    label: str
    first: SmoothRatio
    iterations: List[SmoothRatio]
    chi2: List[float]
    max_pull: List[float]
    grid: Tuple[int, int]
    converged: bool

    def ratio(self, energies) -> np.ndarray:
        out = np.ones(np.size(energies))
        for it in self.iterations:
            out = out * it(energies)
        return out


@dataclass
class AngularCorrection:
    """What :func:`correct_angular` did.

    ``iterations`` holds each iteration's fit smoothed in energy, as
    ``[λ, δa_1, …, δa_L]`` per energy (``values`` is ``(n_E, degree+1)``);
    ``chi2``/``max_pull`` as for σ, over every (energy, angle), the pull being
    the applied correction over the data's uncertainty; ``emin`` the energy
    below which nothing was corrected and ``excluded`` how many measured
    energies that left out; ``inherited_negative`` the records the evaluation
    already had negative somewhere on [-1, 1] and the correction did not
    deepen, as ``(E, f_min before, f_min after)``; one the correction makes
    negative, or more negative, raises.
    """

    level: str
    degree: int
    emin: float
    excluded: int
    iterations: List[SmoothRatio]
    chi2: List[float]
    max_pull: List[float]
    records_touched: int
    converged: bool
    cross_section: Optional[CrossSectionCorrection] = None
    inherited_negative: List[Tuple[float, float, float]] = field(default_factory=list)

    def delta(self, energies) -> np.ndarray:
        """Cumulative ``δa_1..δa_L`` at ``energies``: ``(n, degree)``."""
        out = np.zeros((np.size(energies), self.degree))
        for it in self.iterations:
            out = out + np.atleast_2d(it(energies))[:, 1:]
        return out

    def level_factor(self, energies) -> np.ndarray:
        """Cumulative ``1 + λ`` at ``energies`` (applied to σ only with ``from_dcs``)."""
        out = np.ones(np.size(energies))
        for it in self.iterations:
            out = out * (1.0 + np.atleast_2d(it(energies))[:, 0])
        return out


def _chi2(model, values, errors) -> float:
    m = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    return float(np.mean(((values[m] - model[m]) / errors[m]) ** 2))


def correct_cross_section(suite, energies_ev, values, errors, setup: ForwardSetup, *,
                          bins_ev: Optional[Sequence[Tuple[float, float]]] = None,
                          cross_section_label: Optional[str] = None,
                          min_width: float = 1.0, factors: Optional[Sequence[float]] = None,
                          max_iterations: int = 5, tolerance: float = 0.1,
                          grid_tolerance: float = 1e-4):
    r"""σ corrected by the smoothed ratio to an angle-integrated measurement.

    ``energies_ev``, ``values`` [b], ``errors`` [b] (statistical, one sigma) and
    ``bins_ev`` (``(lo, hi)`` per point, or ``None`` for points) describe the
    measurement; ``setup`` what the experiment does to the evaluation.

    Returns ``(corrected_suite, CrossSectionCorrection)``.
    """
    e = np.asarray(energies_ev, float)
    y = np.asarray(values, float)
    dy = np.asarray(errors, float)
    keep = np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    e, y, dy = e[keep], y[keep], dy[keep]
    bins = None if bins_ev is None else [b for b, k in zip(bins_ev, keep) if k]
    out = copy.deepcopy(suite)
    label = _cross_section_label(out, cross_section_label)
    view = ElasticView.from_suite(out, cross_section_label=label)
    width = _reading_width(view, setup, e, bins)

    model = forward_sigma(view, e, setup, bins)
    chi2 = [_chi2(model, y, dy)]
    iterations, pulls, grid, first = [], [], None, None
    converged = False
    for _ in range(max_iterations):
        r, dr = y / model, dy / model
        fit = (fit_smooth_ratio(e, r, dr, width, min_width=min_width, factors=factors)
               if first is None else
               SmoothRatio(*_sorted(e, r, dr), width, first.factor))
        first = first or fit
        pulls.append(float(np.max(np.abs(fit(e) - 1.0) / dr)))
        if pulls[-1] < tolerance:
            converged = True
            break
        n = _multiply_cross_section(out, label, fit, fit.anchors(), grid_tolerance)
        grid = (n[0], n[1]) if grid is None else (grid[0], n[1])
        iterations.append(fit)
        view = ElasticView.from_suite(out, cross_section_label=label)
        model = forward_sigma(view, e, setup, bins)
        chi2.append(_chi2(model, y, dy))
    return out, CrossSectionCorrection(label, first, iterations, chi2, pulls,
                                       grid or (0, 0), converged)


def _sorted(e, *arrays):
    order = np.argsort(e)
    return (e[order],) + tuple(a[order] for a in arrays)


def _resolved_top(suite) -> float:
    """Top of the suite's resolved resonance region [eV]; 0 when it has none."""
    res = getattr(suite, "resonances", None)
    if res is None or not res.resolved:
        return 0.0
    return max(float(r.domainMax) for r in res.resolved)


def _shape_basis(view: ElasticView, mu_lab, setup: ForwardSetup, sigma_avg, degree: int):
    r"""``(n_E, n_mu, degree)``: the folded DCS's response to ``δa_l = 1``, l = 1..degree.

    Exact while δa_l is constant over the reading: the elastic two-body cosine
    does not depend on the incident energy, so the fold leaves only
    :math:`\langle\sigma\rangle` in front of the detector's Legendre values.
    """
    from numpy.polynomial import legendre as L

    l = np.arange(1, degree + 1)
    p = np.zeros((np.size(mu_lab), degree))
    for j, mu in enumerate(np.atleast_1d(mu_lab)):
        for m, wm in zip(*_detector_cosines(setup, mu)):
            mu_n, jac = view.native_cosine(m)
            p[j] += wm * jac * (2 * l + 1) / 2.0 * L.legvander(np.array([mu_n]), degree)[0, 1:]
    return np.asarray(sigma_avg, float)[:, None, None] * p[None] / (2 * np.pi)


def _fit_delta(resid, dy, model, basis):
    """Weighted least squares of ``resid = λ·model + basis @ δa`` at one energy:
    ``([λ, δa_1…], their errors)``."""
    m = np.isfinite(resid) & np.isfinite(dy) & (dy > 0)
    k = basis.shape[1] + 1
    if m.sum() <= k:
        raise ValueError(f"{int(m.sum())} angles cannot fit a level and {k - 1} shape "
                         f"terms with fewer parameters than angles")
    a = np.column_stack([model[m], basis[m]]) / dy[m, None]
    p, *_ = np.linalg.lstsq(a, resid[m] / dy[m], rcond=None)
    cov = np.linalg.pinv(a.T @ a)
    return p, np.sqrt(np.diag(cov))


def correct_angular(suite, energies_ev, mu_lab, values, errors, setup: ForwardSetup, *,
                    bins_ev: Optional[Sequence[Tuple[float, float]]] = None,
                    degree: int = 4, level: Level = "separate",
                    emin_ev: Optional[float] = None,
                    cross_section_label: Optional[str] = None,
                    min_width: float = 1.0, factors: Optional[Sequence[float]] = None,
                    max_iterations: int = 5, tolerance: float = 0.1,
                    table_tolerance: float = 1e-4, grid_tolerance: float = 1e-4,
                    positivity_points: int = 2001):
    r"""The elastic angular shape corrected by an additive δa_l(E) from measured DCS.

    ``values``/``errors`` are ``(n_E, n_mu)`` laboratory dσ/dΩ [b/sr] at
    ``energies_ev`` (with ``bins_ev``) and ``mu_lab``; a NaN is a missing point.
    ``degree`` is the highest Legendre order corrected; with the level it must
    leave fewer parameters than angles.  ``level`` is ``"separate"`` (shape
    only) or ``"from_dcs"`` (σ also scaled by 1 + λ).  ``emin_ev`` is the energy
    below which nothing is corrected, by default the top of the suite's
    resolved resonance region (``0`` corrects everywhere); a measured energy
    whose reach (hold and ramp) would cross it is left out.

    A record the correction makes negative anywhere on ``[-1, 1]`` (or more
    negative than the evaluation had it) raises: it is reported, never clipped.
    Returns ``(corrected_suite, AngularCorrection)``.
    """
    if level not in ("separate", "from_dcs"):
        raise ValueError(f"level is 'separate' or 'from_dcs', not {level!r}")
    e = np.atleast_1d(np.asarray(energies_ev, float))
    mus = np.atleast_1d(np.asarray(mu_lab, float))
    y = np.asarray(values, float).reshape(e.size, mus.size)
    dy = np.asarray(errors, float).reshape(e.size, mus.size)
    out = copy.deepcopy(suite)
    label = _cross_section_label(out, cross_section_label)
    view = ElasticView.from_suite(out, cross_section_label=label)

    emin = _resolved_top(out) if emin_ev is None else float(emin_ev)
    reach = (READING_REACH + FWHM_PER_SIGMA) * _reading_width(view, setup, e, bins_ev)(e)
    keep = e - reach >= emin
    if not keep.any():
        raise ValueError(f"no measured energy is clear of emin = {emin:.6g} eV "
                         f"(the top of the resolved region unless given)")
    e, y, dy = e[keep], y[keep], dy[keep]
    bins = None if bins_ev is None else [b for b, k in zip(bins_ev, keep) if k]
    width = _reading_width(view, setup, e, bins)
    identity = np.zeros(degree + 1)

    fwd = forward_dcs(view, e, mus, setup, bins)
    model = fwd["dcs"]
    chi2 = [_chi2(model, y, dy)]
    iterations, pulls, touched, inherited = [], [], 0, []
    factor, converged, grid = None, False, None
    for _ in range(max_iterations):
        basis = _shape_basis(view, mus, setup, fwd["sigma"], degree)
        fits = [_fit_delta(y[i] - model[i], dy[i], model[i], basis[i]) for i in range(e.size)]
        p = np.array([f[0] for f in fits])
        dp = np.array([f[1] for f in fits])
        if factor is None:
            fit = fit_smooth_ratio(e, p, dp, width, min_width=min_width,
                                   factors=factors, identity=identity)
            factor = fit.factor
        else:
            fit = SmoothRatio(*_sorted(e, p, dp), width, factor, identity=identity)
        s = np.atleast_2d(fit(e))
        applied = np.einsum("ijl,il->ij", basis, s[:, 1:])
        if level == "from_dcs":
            applied = applied + s[:, :1] * model
        pulls.append(float(np.nanmax(np.abs(applied) / dy)))
        if pulls[-1] < tolerance:
            converged = True
            break
        anchors = fit.anchors()
        problems, inh, touched = _add_angular(
            out, lambda q: np.atleast_2d(fit(q))[:, 1:], anchors, table_tolerance,
            positivity_points)
        first_before = {en: b for en, b, _ in inherited}
        inherited = [(en, first_before.get(en, b), a) for en, b, a in inh]
        if problems:
            worst = min(problems, key=lambda q: q[1])
            raise ValueError(
                f"the corrected angular distribution is negative at {len(problems)} "
                f"records (worst f = {worst[1]:.3g} at E = {worst[0]:.6g} eV); lower "
                f"`degree` or widen the smoothing (`min_width`). Not clipped")
        if level == "from_dcs":
            n = _multiply_cross_section(out, label, lambda q: 1.0 + np.atleast_2d(fit(q))[:, 0],
                                        anchors, grid_tolerance)
            grid = (n[0], n[1]) if grid is None else (grid[0], n[1])
        iterations.append(fit)
        view = ElasticView.from_suite(out, cross_section_label=label)
        fwd = forward_dcs(view, e, mus, setup, bins)
        model = fwd["dcs"]
        chi2.append(_chi2(model, y, dy))

    xs_corr = None
    if level == "from_dcs" and iterations:
        xs_corr = CrossSectionCorrection(label, iterations[0], [], [], [], grid or (0, 0), True)
    return out, AngularCorrection(level, degree, emin, int((~keep).sum()), iterations, chi2,
                                  pulls, touched, converged, xs_corr, inherited)

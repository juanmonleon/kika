"""C10: what an MF35 band states for the spectrum's mean energy, reported with the run.

**Why this is here and not in a validation script.** Decision PD-9
(2026-10-08): the production path keeps the library's MF35 *as stated* -- no
constraint, no clipping of replicas to a band, no rejection of replicas -- and
**reports C10 next to it**. A library's stated uncertainty on ``<E'>`` can be
far wider than what is independently known (JEFF-4.0 U-235 thermal states
2.00 % where the IAEA standards evaluation of the same spectrum gives 0.45 %,
``pfns_mf5_mf35_roadmap.md`` Q3/C10), and the ensemble is a correct propagation
of the file either way. Which of the two numbers a reader is looking at is not
in any output tape, so it goes in the run's metadata and log, every run.

**What is computed, per band.** On one probe node -- the first evaluated
incident node the band contains, by the applier's own band rule -- the group
probabilities ``P_g``, the group mean energies ``e_g`` and, for each threshold
``t``, the fraction ``a_g(t)`` of group ``g``'s mass above ``t``. The applier
keeps the in-group shape, so any linear functional ``q = sum_g w_g P_g`` of
the written spectrum moves by ``w . dP`` exactly, and its stated variance is
``w^T C w``:

* ``<E'>``: ``w_g = e_g``, normalised by ``sum P``;
* ``F(>t)``: ``w_g = a_g(t)``.

``w`` is centred on ``q`` before the sandwich (``C 1 ~ 0`` for a probability
covariance, so centring only removes the drift of a matrix whose rows do not
sum to exactly zero). These are the numbers of
``kika_dev/sampling/checks/pfns_c10_stated_moments.py``, computed on the
model's node and exactly: ``P_g`` and ``a_g`` are closed-form group integrals
(:func:`kika.algebra.group_integrals`), and ``e_g`` is an 8-point
Gauss-Legendre rule on every panel of the table cut at the group edges, which
is exact for a lin-lin table (``x chi`` is quadratic on a panel) and good to
rounding on a smooth log-interpolated one.

When the run has drawn two or more samples, the same ``w`` applied to the
drawn deltas gives the **drawn** spread beside the stated one -- the C9
analytic check, before any positivity projection.

**What this is not.** It is not a gate on the sampler (the sampler only has to
reproduce MF35, which C4/C5 check), and it does not change a draw. The
alternative PD-9 names -- a GLS update of ``C`` with a measured ``<E'>`` and its
correlation to the standard -- would be a named variant delivered through an
external covariance override (P5). **P5 does not exist yet, so the constraint
is pending and not built**; every record says so (``"constraint"``).

**References.** Only what the roadmap has read and quotes: the IAEA standards
evaluation of the U-235 thermal PFNS, ``<E'>`` = 2.000 +/- 0.009 MeV. Chi-Nu/CEA
Pu-239 (Marini 2020, 0.2-0.5 % from 0.7 MeV) has not been read against the
libraries yet and is not used.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = ["C10_REFERENCES", "C10_THRESHOLDS", "C10_CONSTRAINT",
           "spectrumFunctionals", "statedSpectrumMoments", "statedMomentsNote"]

#: The thresholds of the tail fractions, in eV.
C10_THRESHOLDS = (1.0e6, 3.0e6)

#: What PD-9 decided the production path does with the library's MF35.
C10_CONSTRAINT = ("none: MF35 as stated (PD-9); a GLS update with a measured "
                  "<E'> would be a named variant through P5, which is pending "
                  "and not built")

#: ``ZA -> reference``. ``incident`` is the incident energy (eV) the reference
#: is about; it applies to the band that contains it.
C10_REFERENCES: Dict[int, Dict[str, Any]] = {
    92235: {
        "incident": 0.0253,
        "meanEnergy": 2.000e6,
        "sigmaMeanEnergy": 0.009e6,
        "source": ("IAEA neutron standards evaluation of the U-235 thermal "
                   "PFNS, as quoted in pfns_mf5_mf35_roadmap.md (Q3, C10)"),
    },
}

_GAUSS = np.polynomial.legendre.leggauss(8)


def _firstMoments(xs, ys, laws, edges) -> np.ndarray:
    """``int_{e_g}^{e_g+1} x chi(x) dx`` for every group, cut at every node."""
    from kika.algebra import evaluate

    lo, hi = float(edges[0]), float(edges[-1])
    cuts = np.unique(np.concatenate([xs[(xs > lo) & (xs < hi)], edges]))
    a, b = cuts[:-1], cuts[1:]
    wide = b > a
    a, b = a[wide], b[wide]
    nodes, weights = _GAUSS
    half, middle = 0.5 * (b - a), 0.5 * (b + a)
    points = middle[:, None] + half[:, None] * nodes[None, :]
    values = np.asarray(evaluate(xs, ys, laws, points.ravel()),
                        dtype=float).reshape(points.shape)
    pieces = half * np.sum(weights[None, :] * points * values, axis=1)
    group = np.clip(np.searchsorted(edges, middle, side="right") - 1,
                    0, edges.size - 2)
    return np.bincount(group, weights=pieces, minlength=edges.size - 1)


def spectrumFunctionals(xs, ys, laws, edges,
                        thresholds: Sequence[float] = C10_THRESHOLDS) -> dict:
    """``P_g`` and the weights of ``<E'>`` and of each ``F(>t)`` on *edges*.

    Returns ``{"P": P, "meanEnergy": (w, q), "F>1": (w, q), ...}`` where
    ``q = w . P / sum P`` is the functional's value and ``w`` its weight per
    group. A group with no mass gets the weight it would have if it had some:
    its midpoint for ``<E'>``, 0 or 1 for ``F(>t)`` -- it moves nothing either
    way, since its ``P`` is 0 and so is its variance.
    """
    from kika.algebra import group_integrals

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    edges = np.asarray(edges, dtype=float)
    P = group_integrals(xs, ys, laws, edges)
    M = _firstMoments(xs, ys, laws, edges)
    total = float(P.sum())
    live = P > 0.0
    safe = np.where(live, P, 1.0)
    ebar = np.where(live, M / safe, 0.5 * (edges[:-1] + edges[1:]))
    out = {"P": P, "meanEnergy": (ebar, float(M.sum() / total) if total else 0.0)}
    for t in thresholds:
        split = np.unique(np.concatenate([edges, [t]]))
        split = split[(split >= edges[0]) & (split <= edges[-1])]
        pieces = group_integrals(xs, ys, laws, split)
        owner = np.clip(np.searchsorted(edges, split[:-1], side="right") - 1,
                        0, edges.size - 2)
        above = np.bincount(owner, weights=np.where(split[:-1] >= t, pieces, 0.0),
                            minlength=edges.size - 1)
        w = np.where(live, above / safe, (edges[:-1] >= t).astype(float))
        out[f"F>{t / 1e6:g}MeV"] = (w, float(above.sum() / total) if total else 0.0)
    return out


def statedSpectrumMoments(energyForm, bands: Mapping[Any, Sequence[float]],
                          grids: Mapping[Any, Sequence[float]],
                          matrices: Mapping[Any, np.ndarray], *,
                          za: Optional[int] = None,
                          drawn: Optional[Mapping[Any, np.ndarray]] = None) -> dict:
    """C10 per band: ``{band: record}``, from the evaluated node and the stated C.

    *bands*, *grids*, *matrices* and *drawn* are keyed by the band the file
    states. *matrices* are the **stated** absolute covariances of the group
    probabilities (before any conditioning), and *drawn* the drawn deltas,
    ``(nSamples, nGroups)``, when there are any. Relative sigmas are fractions
    of the functional's value.
    """
    from kika.nuclear_data.model.perturbation import _asBandMap, _bandOf

    bandMap = _asBandMap(bands)
    nodes = np.asarray(energyForm.outerDomainValues, dtype=float)
    reference = C10_REFERENCES.get(int(za)) if za is not None else None
    out: Dict[Any, Dict[str, Any]] = {}
    for band, (lo, hi) in sorted(bandMap.items(), key=lambda item: item[1][0]):
        record: Dict[str, Any] = {"band": [float(lo), float(hi)],
                                  "constraint": C10_CONSTRAINT}
        inside = [k for k, value in enumerate(nodes)
                  if _bandOf(float(value), bandMap) == band]
        if not inside or band not in grids or band not in matrices:
            record["probe"] = None
            out[band] = record
            continue
        k = inside[0]
        xs, ys = energyForm.table(k)
        laws = energyForm.tableLaws(k)
        edges = np.asarray(grids[band], dtype=float)
        C = np.asarray(matrices[band], dtype=float)
        functionals = spectrumFunctionals(xs, ys, laws, edges)
        record["probe"] = float(nodes[k])
        record["sumP"] = float(functionals["P"].sum())
        rows = None
        if drawn is not None and band in drawn:
            rows = np.asarray(drawn[band], dtype=float)
            rows = rows if rows.ndim == 2 and rows.shape[0] >= 2 else None
        for name, value in functionals.items():
            if name == "P":
                continue
            w, q = value
            centred = (w - q) / (functionals["P"].sum() or 1.0)
            stated = float(np.sqrt(max(centred @ C @ centred, 0.0)))
            record[name] = q
            record[f"sigma({name})_stated_rel"] = stated / q if q else 0.0
            if rows is not None:
                spread = float(np.std(rows @ centred, ddof=1))
                record[f"sigma({name})_drawn_rel"] = spread / q if q else 0.0
        if reference is not None and lo <= reference["incident"] and (
                reference["incident"] < hi or (lo == hi == reference["incident"])):
            refRel = reference["sigmaMeanEnergy"] / reference["meanEnergy"]
            record["reference"] = {
                "source": reference["source"],
                "meanEnergy": reference["meanEnergy"],
                "sigma(meanEnergy)_rel": refRel,
                "stated_over_reference": (record["sigma(meanEnergy)_stated_rel"]
                                          / refRel),
            }
        out[band] = record
    return out


def statedMomentsNote(mt: int, moments: Mapping[Any, Mapping[str, Any]]) -> Optional[str]:
    """One line for the run's notes, only where a reference exists.

    Every band's numbers are in the metadata and the log; the note is kept to
    the bands where the library can be read against independent knowledge,
    because that is where the reader of an ensemble needs to be told which of
    the two uncertainties the ensemble carries.
    """
    parts = []
    for band, record in moments.items():
        reference = record.get("reference")
        if not reference:
            continue
        parts.append(
            f"band {band} states sigma(<E'>) = "
            f"{100 * record['sigma(meanEnergy)_stated_rel']:.2f} % at "
            f"E = {record['probe']:.4g} eV, against "
            f"{100 * reference['sigma(meanEnergy)_rel']:.2f} % in the "
            f"{reference['source'].split(',')[0]} "
            f"(x{reference['stated_over_reference']:.1f})")
    if not parts:
        return None
    return (f"C10, MF35/MT{mt} (reported, not a gate; the ensemble carries the "
            f"library's MF35 as stated, PD-9): " + "; ".join(parts))

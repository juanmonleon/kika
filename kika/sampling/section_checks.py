"""The file's own faults in the pre-flight: layer 1 next to layer 2.

The pre-flight (:mod:`kika.cov.conditioning`, *layer 2*) sees the assembled
joint matrix, and can say that |rho| > 1 somewhere in it or that it is not
positive semi-definite. It cannot say *why*: by the time a block is assembled,
the LB/LS records it came from are gone. Layer 1
(:func:`kika.endf.check_covariances`) saw those records, and
:func:`~kika.endf.model_adapter.covariances.decodeCovarianceSuite` leaves what it
found on each section's ``provenance.covarianceFindings``.

This module joins the two. :func:`sectionFindings` collects the layer-1
findings of the sections a run perturbs, and :func:`attribute` takes each
layer-2 finding that matters -- a correlation outside [-1, 1], a negative
eigenvalue, a negative variance -- down to the pair of components that carries
it and the layer-1 findings stated for that pair. The result reads like
"|rho| 4.63 in MF34/MT2 L=1 x MF34/MT2 L=5 <- MF34 MT2xMT2 L1xL5 NI[0] LB=5 LS=1:
ls1_in_cross_block".

Nothing here changes what blocks a draw. Layer-1 findings only inform, and the
attribution is an explanation attached to a decision layer 2 already made.
Plan: kika-workspace ``docs/library/cov_checks_roadmap.md``, phase C6.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["sectionFindings", "relevantFindings", "attribute", "Attribution"]

#: Layer-2 checks traced back to the file, and the severities that count.
ATTRIBUTED = {"correlation_bound": ("blocks",), "definiteness": ("distorts", "blocks"),
              "negative_variance": ("blocks",)}

#: |rho| above 1 by more than this is a correlation outside the bound (the
#: pre-flight's own line between the file's rounding and a real breach).
RHO_EXCESS = 1e-5

#: Pairs reported per layer-2 finding, largest first.
MAX_PAIRS = 8

#: Above this order the negative-mass split (one more eigendecomposition) is
#: not computed, as in the pre-flight itself.
MAX_ORDER = 1500


def sectionFindings(covariances) -> Tuple[Any, ...]:
    """Every layer-1 finding of a decoded covariance suite.

    The suite's whole report when the decoder left one (``covarianceChecks``),
    which also holds what belongs to no decoded section; otherwise whatever the
    sections' provenances carry. Empty for a suite that was never checked --
    one read from GNDS, or decoded with ``checks=False``.
    """
    report = getattr(covariances, "covarianceChecks", None)
    if report is not None:
        return tuple(report)
    seen, out = set(), []
    for section in getattr(covariances, "covarianceSections", ()) or ():
        provenance = getattr(section, "provenance", None)
        if provenance is None or id(provenance) in seen:
            continue
        seen.add(id(provenance))
        out.extend(getattr(provenance, "covarianceFindings", ()) or ())
    return tuple(out)


def relevantFindings(findings: Sequence[Any], index) -> Tuple[Any, ...]:
    """The findings about the MFs and MTs the assembled blocks are made of."""
    components = {(c.mf, c.mt) for meta in index.values() for c in meta["components"]}
    mfs = {mf for mf, _ in components}
    out = []
    for f in findings:
        loc = f.location
        if loc.mf not in mfs:
            continue
        if loc.mt is None or (loc.mf, loc.mt) in components or (
                loc.mt1 is not None and not loc.mat1 and (loc.mf, loc.mt1) in components):
            out.append(f)
    return tuple(out)


class Attribution(dict):
    """One layer-2 finding, one pair of components, and what layer 1 said there.

    A ``dict`` so it goes into the run log and ``run_metadata.json`` as it is:
    ``block``, ``check``, ``severity``, ``pair`` (two component descriptions),
    ``measure`` (the worst |rho|, the share of the negative mass, or the number
    of negative variances), ``layer1`` (the findings' text) and ``text``.
    """

    def __str__(self) -> str:
        return self["text"]


def _side(component) -> Tuple[int, int]:
    return (component.mt, component.index if component.mf == 34 else 0)


def _matches(finding, row, col) -> bool:
    """Is this layer-1 finding about the block between components *row* and *col*?"""
    loc = finding.location
    if loc.mf != row.mf or row.mf != col.mf or loc.mat1:
        return False
    if loc.mt1 is None:
        # A section-level finding (a missing self block, a parse error) is about
        # every block of its MT.
        return loc.mt in (row.mt, col.mt)
    if row.mf == 34 and (loc.l is None or loc.l1 is None):
        return {loc.mt, loc.mt1} == {row.mt, col.mt}
    a = (loc.mt, loc.l if row.mf == 34 else 0)
    b = (loc.mt1, loc.l1 if row.mf == 34 else 0)
    return {a, b} == {_side(row), _side(col)}


def _slices(meta) -> List[Tuple[Any, slice]]:
    stride = int(meta["stride"])
    return [(c, slice(i * stride, i * stride + int(meta["widths"][c])))
            for i, c in enumerate(meta["components"])]


def _rho_pairs(matrix: np.ndarray, meta) -> List[Tuple[Any, Any, float]]:
    d = np.diag(matrix)
    sigma = np.sqrt(np.where(np.isfinite(d) & (d > 0), d, np.nan))
    parts = _slices(meta)
    out = []
    for i, (row, rs) in enumerate(parts):
        for col, cs in parts[i:]:
            block = matrix[rs, cs]
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = np.abs(block / np.outer(sigma[rs], sigma[cs]))
            if row == col:
                np.fill_diagonal(rho, 0.0)
            rho = rho[np.isfinite(rho)]
            worst = float(rho.max()) if rho.size else 0.0
            if worst > 1.0 + RHO_EXCESS:
                out.append((row, col, worst))
    return sorted(out, key=lambda t: -t[2])


def _negative_pairs(matrix: np.ndarray, meta) -> List[Tuple[Any, Any, float]]:
    from kika.cov.conditioning import negative_mass_by_family

    if matrix.shape[0] > MAX_ORDER:
        return []
    names = {c.describe(): c for c in meta["components"]}
    labels = [c.describe() for c in meta["components"] for _ in range(int(meta["stride"]))]
    largest = float(np.max(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)))
    pairs = negative_mass_by_family(matrix, labels,
                                    tolerance=matrix.shape[0] * np.finfo(float).eps * largest)
    return [(names[p["pair"][0]], names[p["pair"][1]], float(p["share"]))
            for p in pairs if p["share"] >= 0.05]


def _negative_variance_pairs(matrix: np.ndarray, meta) -> List[Tuple[Any, Any, float]]:
    d = np.diag(matrix)
    return [(c, c, float(np.sum(d[s] < 0))) for c, s in _slices(meta) if np.any(d[s] < 0)]


def attribute(blocks, index, report, findings: Sequence[Any]) -> Tuple[Attribution, ...]:
    """Trace layer-2 findings to component pairs and the layer-1 findings there.

    *blocks* and *index* are what :func:`~kika.sampling.joint_blocks.assembleRequest`
    returned, *report* the pre-flight's
    :class:`~kika.cov.conditioning.ConditioningReport` on them, *findings* the
    layer-1 findings (:func:`sectionFindings`). Only layer-1 warnings and
    defects are used: a note is not a fault that could explain one.
    """
    from kika.cov.conditioning import block_key_text

    if report is None:
        return ()
    faults = [f for f in findings if f.level in ("warn", "defect")]
    matrices = {block_key_text(key): np.asarray(m, dtype=float) for key, m in blocks}
    metas = {block_key_text(key): meta for key, meta in index.items()}
    out: List[Attribution] = []
    for block in report.blocks:
        text = block_key_text(block.key)
        matrix, meta = matrices.get(text), metas.get(text)
        if matrix is None or meta is None:
            continue
        for finding in block.findings:
            if finding.severity not in ATTRIBUTED.get(finding.check, ()):
                continue
            if finding.check == "correlation_bound":
                pairs, unit = _rho_pairs(matrix, meta), "|rho| {:.4g}"
            elif finding.check == "definiteness":
                pairs, unit = _negative_pairs(matrix, meta), "{:.0%} of the negative mass"
            else:
                pairs, unit = _negative_variance_pairs(matrix, meta), "{:.0f} negative variance(s)"
            for row, col, measure in pairs[:MAX_PAIRS]:
                there = [f for f in faults if _matches(f, row, col)]
                where = (row.describe() if row == col
                         else f"{row.describe()} x {col.describe()}")
                cause = ("; ".join(f"{f.location}: {f.check}" for f in there)
                         if there else "no layer-1 finding on this block")
                out.append(Attribution(
                    block=text, check=finding.check, severity=finding.severity,
                    pair=[row.describe(), col.describe()], measure=measure,
                    layer1=[str(f) for f in there],
                    text=f"{finding.check}: {unit.format(measure)} in {where} <- {cause}"))
    return tuple(out)

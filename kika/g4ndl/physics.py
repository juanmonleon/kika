"""Phase 5: the physics of an elastic suite read from G4NDL, checked and evaluated.

Two things, kept apart on purpose:

:func:`checkElastic`
    Per incident-energy record, what the data *is*: is ``p(mu)`` normalised,
    finite and non-negative, and does it return its own Legendre moments? The
    integrals are exact on each segment's stated law (lin-lin, lin-log,
    log-lin, log-log), never trapezoids on a law that is not linear, and the
    minimum of a Legendre sum is found from the roots of its derivative, not
    on a grid — a negative lobe between grid points is the case that matters.
    Nothing is corrected: an anomaly of the library is reported as such
    (roadmap §5 Fase 5, "no forzar artificialmente todos los datos a cumplir
    una prueba").

:func:`angularPdf` and :func:`differentialCrossSection`
    ``p(mu|E)`` and ``dsigma/dOmega`` at any incident energy inside the data.
    The model leaves ``XYs2d.evaluate`` out deliberately, because the general
    question has two undecided conventions. For an angular distribution read
    from G4NDL both are settled by the domain and by the consumer, Geant4
    11.4.3, measured against the compiled code
    (``kika-workspace/myworkspace/G4NDL/harness``):

    * **No unit-base question.** Every record spans ``mu`` in [-1, 1], so
      interpolating at fixed ``mu`` and on the unit base are the same thing.
    * **Legendre records of different order** are interpolated as if the
      shorter one had zero coefficients above its order, which is
      interpolating ``p(mu)`` itself: Geant4 does exactly that, and for the
      only energy laws in either library (INT=2 and INT=3, both linear in y)
      it is the same as interpolating coefficients.
    * **Energy law:** INT=2 and INT=3; anything else raises, as INT=1 already
      does at read time.
    * **Tables:** each bracketing record is evaluated at ``mu`` on its own
      law, then the two values are interpolated in energy. Geant4 does the
      energy step first; the two orders differ only where ``mu`` is LOGLIN
      (C-12: up to 5e-5 relative).
    * **Boundaries and repeats.** No extrapolation: outside the first and
      last incident energy is an error. At a repeated energy, and at the
      ``repFlag=3`` transition, ``side`` picks the record: ``"right"`` (the
      default, what :class:`~kika.nuclear_data.model.XYs1d` does) or
      ``"left"`` (what Geant4 does: Legendre up to and including the
      transition energy).

This module works on the model objects, not on the parsed records, so what
it checks is what :func:`kika.g4ndl.decode.decodeElastic` produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as npleg

__all__ = [
    "Finding", "ElasticPhysicsCheck", "checkElastic", "angularPdf",
    "differentialCrossSection", "legendreDensity", "legendreMinimum",
    "tableIntegral", "NORMALISATION_TOLERANCE",
]

#: Roadmap §5 Fase 5: "para integral/PDF interpolada, empezar con 1e-6".
NORMALISATION_TOLERANCE = 1.0e-6

#: Order of the Gauss-Legendre rule used on each table segment for moments.
_SEGMENT_NODES = 16


# ---------------------------------------------------------------- Legendre

def legendreDensity(coefficients, mu) -> np.ndarray:
    """``p(mu) = sum_l (2l+1)/2 a_l P_l(mu)``, ``coefficients`` including ``a_0``."""
    a = np.asarray(coefficients, dtype=float)
    return npleg.legval(np.asarray(mu, dtype=float), a * (2 * np.arange(a.size) + 1) / 2)


def legendreMinimum(coefficients) -> Tuple[float, float]:
    """The exact minimum of a Legendre density on [-1, 1], as ``(p_min, mu)``.

    Candidates are both ends and every real root of ``dp/dmu`` inside the
    interval; for ``L <= 30`` (the largest order in either library) the
    companion-matrix roots are accurate to far below any value that matters.
    """
    a = np.asarray(coefficients, dtype=float)
    scaled = a * (2 * np.arange(a.size) + 1) / 2
    candidates = [-1.0, 1.0]
    if scaled.size > 2:
        roots = npleg.Legendre(scaled).deriv().roots()
        real = roots[np.abs(roots.imag) < 1e-9].real
        candidates.extend(real[(real > -1.0) & (real < 1.0)].tolist())
    mus = np.asarray(candidates)
    values = npleg.legval(mus, scaled)
    k = int(np.argmin(values))
    return float(values[k]), float(mus[k])


# ------------------------------------------------------------------ tables

def _intervalCodes(nbt: Sequence[int], codes: Sequence[int], n: int) -> np.ndarray:
    """ENDF semantics: the interval from point j to j+1 (1-based) takes the
    code of the first region with ``NBT >= j + 1``."""
    point = np.empty(n, dtype=int)
    lo = 0
    for b, c in zip(nbt, codes):
        point[lo:b] = c
        lo = b
    return point[1:]


def _segmentIntegral(code: int, x1, x2, y1, y2) -> float:
    """``int_{x1}^{x2} y dx`` exactly on ENDF law ``code``."""
    dx = x2 - x1
    if dx == 0.0:
        return 0.0
    if code == 1:
        return y1 * dx
    if code == 2:
        return 0.5 * (y1 + y2) * dx
    if code == 3:                              # y linear in ln x
        r = np.log(x2 / x1)
        return y1 * dx + (y2 - y1) / r * (x2 * r - dx)
    if code == 4:                              # ln y linear in x
        if y1 == y2:
            return y1 * dx
        return (y2 - y1) * dx / np.log(y2 / y1)
    if code == 5:                              # ln y linear in ln x
        b = np.log(y2 / y1) / np.log(x2 / x1)
        if abs(b + 1.0) < 1e-12:
            return y1 * x1 * np.log(x2 / x1)
        return y1 * x1 / (b + 1.0) * ((x2 / x1) ** (b + 1.0) - 1.0)
    raise ValueError(f"interpolation code {code} is not an ENDF law")


def _segmentValues(code: int, x1, x2, y1, y2, x) -> np.ndarray:
    """``y`` at ``x`` inside one segment, on its law. Independent of the model."""
    t_lin = (x - x1) / (x2 - x1)
    if code == 1:
        return np.full_like(x, y1)
    if code == 2:
        return y1 + (y2 - y1) * t_lin
    if code == 3:
        return y1 + (y2 - y1) * np.log(x / x1) / np.log(x2 / x1)
    if code == 4:
        return y1 * np.exp(np.log(y2 / y1) * t_lin) if y1 > 0 and y2 > 0 else np.zeros_like(x)
    if code == 5:
        return y1 * np.exp(np.log(y2 / y1) * np.log(x / x1) / np.log(x2 / x1))
    raise ValueError(f"interpolation code {code} is not an ENDF law")


def tableIntegral(mu, p, nbt, codes) -> float:
    """``int p dmu`` over the table, exactly on each segment's law."""
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    iv = _intervalCodes(nbt, codes, mu.size)
    return float(sum(_segmentIntegral(int(c), mu[j], mu[j + 1], p[j], p[j + 1])
                     for j, c in enumerate(iv)))


def _tableMoments(mu, p, nbt, codes, order: int) -> np.ndarray:
    """``int p P_l dmu`` for ``l = 0..order``: a 16-point Gauss rule per
    segment on the segment's own law. Not exact for the log laws, but
    independent of how the model evaluates."""
    nodes, weights = npleg.leggauss(_SEGMENT_NODES)
    iv = _intervalCodes(nbt, codes, len(mu))
    out = np.zeros(order + 1)
    for j, c in enumerate(iv):
        x1, x2 = mu[j], mu[j + 1]
        if x2 == x1:
            continue
        x = 0.5 * (x2 - x1) * nodes + 0.5 * (x1 + x2)
        y = _segmentValues(int(c), x1, x2, p[j], p[j + 1], x)
        out += 0.5 * (x2 - x1) * (npleg.legvander(x, order).T @ (weights * y))
    return out


def _tableArrays(function) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    mu, p, pairs = function.toEndfRegions()
    return (np.asarray(mu, dtype=float), np.asarray(p, dtype=float),
            [int(a) for a, _ in pairs], [int(b) for _, b in pairs])


# ------------------------------------------------- the model's angular tree

@dataclass
class _Block:
    """One representation block: all its records in order, and the code of
    each interval between consecutive records (ENDF TAB2 semantics)."""

    kind: str                      # "Legendre" or "table"
    functions: List
    intervalCodes: np.ndarray

    @property
    def energies(self) -> np.ndarray:
        return np.array([f.outerDomainValue for f in self.functions], dtype=float)


def _regions(node) -> List:
    """The XYs2d leaves of a (possibly nested) Regions2d, in order."""
    if hasattr(node, "function2ds"):
        out = []
        for child in node.function2ds:
            out.extend(_regions(child))
        return out
    return [node]


def _blocks(angular) -> List[_Block]:
    """Group the leaves into blocks by representation (repFlag=3 has two).

    Regions of one block share their boundary record for interpolation (the
    TAB2 rule ``fromEndfTab2`` documents); two blocks do not, the switch is a
    jump at the transition energy.
    """
    blocks: List[_Block] = []
    for leaf in _regions(angular):
        kind = "Legendre" if hasattr(leaf.function1ds[0], "coefficients") else "table"
        code = int(leaf.endfInterpolationCode)
        if blocks and blocks[-1].kind == kind:
            b = blocks[-1]
            b.functions.extend(leaf.function1ds)
            b.intervalCodes = np.concatenate([b.intervalCodes,
                                              np.full(len(leaf.function1ds), code)])
        else:
            blocks.append(_Block(kind, list(leaf.function1ds),
                                 np.full(len(leaf.function1ds) - 1, code)))
    return blocks


def _distribution(suiteOrDistribution):
    d = suiteOrDistribution
    if hasattr(d, "reactions"):
        d = d.reactions[2].outputChannel.products.byPid("n")[0].distribution["eval"]
    return d


# ------------------------------------------------------------- the checks

@dataclass
class Finding:
    """One thing the data does that a normalised, non-negative pdf would not."""

    kind: str          # "negative", "normalisation", "moments", "nonfinite"
    block: str         # "Legendre" or "table"
    index: int
    energy: float
    value: float
    detail: str = ""


@dataclass
class ElasticPhysicsCheck:
    """What :func:`checkElastic` found. Library anomalies, not adapter errors:
    the adapter's own faithfulness is what the tests pin."""

    target: str
    records: Dict[str, int] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    #: Largest |integral - 1| over all records, and its record.
    maxNormalisationError: float = 0.0
    #: Lowest density anywhere, exact for Legendre, at the nodes for tables.
    minDensity: float = float("inf")
    #: repFlag=3: (1/2) int |p_Legendre - p_table| dmu at the transition energy.
    transitionJump: Optional[float] = None
    repeatedEnergies: int = 0

    def byKind(self, kind: str) -> List[Finding]:
        return [f for f in self.findings if f.kind == kind]

    @property
    def isClean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        counts = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        parts = [f"{n} {k}" for k, n in sorted(counts.items())] or ["clean"]
        return (f"{self.target}: {', '.join(parts)}; max |int p - 1| = "
                f"{self.maxNormalisationError:.2e}; min p = {self.minDensity:.3g}")


def checkElastic(suite, tolerance: float = NORMALISATION_TOLERANCE) -> ElasticPhysicsCheck:
    """Check every incident-energy record of the elastic angular distribution.

    Legendre: ``a_0`` must be 1 (so the expansion is normalised by
    construction and the import added no factor), the moments recovered by
    a Gauss rule exact for the order must equal the coefficients, and the
    exact minimum must not be negative. Tables: the exact integral on each
    segment's law must be 1 within ``tolerance``, every value non-negative.
    """
    angular = _distribution(suite)
    out = ElasticPhysicsCheck(target=getattr(suite, "target", "?"))
    if not hasattr(angular, "angular"):                     # Isotropic2d
        out.minDensity = 0.5
        return out

    blocks = _blocks(angular.angular)
    for block in blocks:
        out.records[block.kind] = len(block.functions)
        e = block.energies
        out.repeatedEnergies += int(np.sum(np.diff(e) == 0))
        for i, f in enumerate(block.functions):
            energy = float(f.outerDomainValue)
            if block.kind == "Legendre":
                _checkLegendre(out, f, i, energy)
            else:
                _checkTable(out, f, i, energy, tolerance)

    if len(blocks) == 2:
        out.transitionJump = _transitionJump(blocks[0].functions[-1],
                                             blocks[1].functions[0])
    return out


def _checkLegendre(out: ElasticPhysicsCheck, f, i: int, energy: float) -> None:
    a = np.asarray(f.coefficients, dtype=float)
    if not np.all(np.isfinite(a)):
        out.findings.append(Finding("nonfinite", "Legendre", i, energy, float("nan")))
        return
    norm = abs(a[0] - 1.0)
    out.maxNormalisationError = max(out.maxNormalisationError, norm)
    if norm > 0:
        out.findings.append(Finding("normalisation", "Legendre", i, energy, a[0] - 1.0,
                                    "a_0 is not 1"))
    # Moments by a rule exact for degree 2L: recovers a_l iff no factor slipped in.
    order = a.size - 1
    nodes, weights = npleg.leggauss(order + 1)
    moments = npleg.legvander(nodes, order).T @ (weights * f.evaluate(nodes))
    err = float(np.max(np.abs(moments - a)))
    if err > 1e-10:
        out.findings.append(Finding("moments", "Legendre", i, energy, err,
                                    "int p P_l dmu differs from a_l"))
    pmin, mu = legendreMinimum(a)
    out.minDensity = min(out.minDensity, pmin)
    if pmin < 0:
        out.findings.append(Finding("negative", "Legendre", i, energy, pmin,
                                    f"at mu = {mu:.6f}"))


def _checkTable(out: ElasticPhysicsCheck, f, i: int, energy: float,
                tolerance: float) -> None:
    mu, p, nbt, codes = _tableArrays(f)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(p))):
        out.findings.append(Finding("nonfinite", "table", i, energy, float("nan")))
        return
    out.minDensity = min(out.minDensity, float(p.min()))
    if p.min() < 0:
        out.findings.append(Finding("negative", "table", i, energy, float(p.min())))
    integral = tableIntegral(mu, p, nbt, codes)
    norm = abs(integral - 1.0)
    out.maxNormalisationError = max(out.maxNormalisationError, norm)
    if norm > tolerance:
        out.findings.append(Finding("normalisation", "table", i, energy, integral - 1.0,
                                    f"int p dmu = {integral:.9g}"))


def _transitionJump(legendre, table) -> float:
    """``(1/2) int |p_L - p_T| dmu`` at the transition: 0 continuous, 1 disjoint.
    On the table's own nodes refined tenfold, lin-lin between them."""
    mu, _, _, _ = _tableArrays(table)
    fine = np.unique(np.concatenate([np.linspace(a, b, 11) for a, b in zip(mu[:-1], mu[1:])]))
    diff = np.abs(legendre.evaluate(fine) - table.evaluate(fine))
    return float(0.5 * np.trapezoid(diff, fine))


# ------------------------------------------------------------- evaluation

def angularPdf(suiteOrDistribution, energy: float, mu, side: str = "right") -> np.ndarray:
    """``p(mu|E)`` for the elastic neutron, in the product frame of the data.

    See the module docstring for the conventions. ``energy`` in eV; ``mu``
    scalar or array. Raises outside the incident-energy range.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    d = _distribution(suiteOrDistribution)
    mu = np.asarray(mu, dtype=float)
    if not hasattr(d, "angular"):                           # Isotropic2d
        return np.full_like(mu, 0.5)

    blocks = _blocks(d.angular)
    lo, hi = blocks[0].energies[0], blocks[-1].energies[-1]
    if not lo <= energy <= hi:
        raise ValueError(f"E = {energy!r} eV is outside the data, [{lo!r}, {hi!r}] eV; "
                         "kika does not extrapolate (Geant4 does, differently for "
                         "Legendre and tables)")
    block = _pickBlock(blocks, energy, side)
    e = block.energies
    # Records at exactly this energy: take the last ('right') or first ('left').
    hit = np.flatnonzero(e == energy)
    if hit.size:
        f = block.functions[hit[-1] if side == "right" else hit[0]]
        return np.asarray(f.evaluate(mu), dtype=float)
    upper = int(np.searchsorted(e, energy, side="right"))
    lower = upper - 1
    code = int(block.intervalCodes[lower])
    if code == 2:
        w = (energy - e[lower]) / (e[upper] - e[lower])
    elif code == 3:
        w = np.log(energy / e[lower]) / np.log(e[upper] / e[lower])
    else:
        raise NotImplementedError(
            f"energy interpolation INT={code} between {e[lower]!r} and {e[upper]!r} eV; "
            "only INT=2 and INT=3 (the laws in both libraries) are implemented")
    low = np.asarray(block.functions[lower].evaluate(mu), dtype=float)
    high = np.asarray(block.functions[upper].evaluate(mu), dtype=float)
    return low + w * (high - low)


def _pickBlock(blocks: List[_Block], energy: float, side: str) -> _Block:
    if len(blocks) == 1:
        return blocks[0]
    transition = blocks[0].energies[-1]
    if energy < transition or (energy == transition and side == "left"):
        return blocks[0]
    return blocks[1]


def differentialCrossSection(suite, energy: float, mu, side: str = "right") -> np.ndarray:
    """``dsigma/dOmega = sigma_el(E) p(mu|E) / (2 pi)`` in b/sr, frame of the data.

    ``sigma_el`` is the ``recon`` form, lin-lin, exactly what the file holds.
    """
    form = suite.reactions[2].crossSection["recon"]
    sigma = float(form.evaluate(energy))
    return sigma * angularPdf(suite, energy, mu, side) / (2.0 * np.pi)

"""Phases 2 and 3: the elastic cross-section and final-state grammars.

Each parser consumes one :class:`~kika.g4ndl.tokens.TokenStream` to its last
token and returns a record from :mod:`kika.g4ndl.records`. The grammar is the
consumer's, Geant4 v11.4.3:

* ``Elastic/CrossSection`` — ``G4ParticleHPIsoData::Init``
  (``src/G4ParticleHPIsoData.cc:91-95``) and the three-argument
  ``G4ParticleHPVector::Init``: two bookkeeping integers, ``N``, ``N`` pairs.
* ``Elastic/FS`` — ``G4ParticleHPElasticFS::Init``
  (``src/G4ParticleHPElasticFS.cc:104-215``): ``repFlag targetMass frameFlag``
  and then the block(s) ``repFlag`` selects.
* Interpolation records — ``G4InterpolationManager::Init``
  (``include/G4InterpolationManager.hh:99-123``).

The consumer reads all of this with ``operator>>`` and checks almost nothing.
These parsers check what a misread would silently corrupt, and every check is
one the 1 153 real elastic files of JEFF-4.0 and G4NDL 4.7.1 pass
(``test_full_libraries.py``): counts, integer fields, region tables, grid
order, the sign of σ and p(μ), μ inside [-1, 1], and the ``repFlag=3``
transition. **Physics is not checked here** — normalisation of p(μ),
positivity of a Legendre sum, comparison against the consumer's interpolator
are Phase 5.

Where kika stops and the consumer does not, the difference is listed in
``G4NDL_token_spec.md`` §6.
"""
from __future__ import annotations

from typing import List

import numpy as np

from kika.g4ndl.records import (
    REP_ISOTROPIC, REP_LEGENDRE, REP_MIXED, REP_TABULATED,
    AngularBlock, CrossSectionRecord, ElasticFSRecord, Interpolation,
    LegendreRecord, TabulatedRecord,
)
from kika.g4ndl.tokens import TokenStream

__all__ = ["parse_cross_section", "parse_elastic_fs", "parse_interpolation",
           "QUALIFIED_INTERPOLATION", "SUPPORTED_INTERPOLATION", "TRANSITION_RTOL"]

#: ENDF codes 1-5. ``G4InterpolationManager::MakeScheme`` also accepts the
#: ``C…`` (11-15) and ``U…`` (21-25) variants; no real elastic file uses them,
#: so kika refuses them until one does rather than guess their meaning.
SUPPORTED_INTERPOLATION = frozenset({1, 2, 3, 4, 5})
#: The corresponding-point (``C…``, 11-15) and unit-base (``U…``, 21-25)
#: variants, admitted only where ``qualified=True``.
QUALIFIED_INTERPOLATION = frozenset(range(11, 16)) | frozenset(range(21, 26))
_LOG_X = frozenset({3, 5})  # ln x: x must be > 0
_LOG_Y = frozenset({4, 5})  # ln y: y must be > 0

#: The consumer's tolerance on the repFlag=3 transition
#: (``G4ParticleHPElasticFS.cc:192``). Geant4 only prints a warning beyond it;
#: kika refuses, because which representation applies between the two
#: energies is then undefined.
TRANSITION_RTOL = 1.0e-15

_REP_FLAGS = (REP_ISOTROPIC, REP_LEGENDRE, REP_TABULATED, REP_MIXED)
_FRAMES = (1, 2)


# ------------------------------------------------------------ interpolation

def parse_interpolation(stream: TokenStream, n: int, what: str, *,
                        qualified: bool = False) -> Interpolation:
    """Read ``NR (NBT INT)×NR`` governing ``n`` points.

    ``qualified`` admits the corresponding-point (11-15) and unit-base
    (21-25) variants ``G4InterpolationManager::MakeScheme`` knows. No elastic
    file uses them; the incident-energy tables of inelastic spectra do
    (``G4NDL_token_spec.md`` §10), and there the consumer applies them.

    ``NBT`` is cumulative (the consumer's ``start[i] = range[i-1]``): it must
    increase strictly and end at ``n``, else regions overlap or leave points
    uncovered. The consumer accepts either silently.
    """
    start = stream.position
    nr = stream.int(f"{what}: NR")
    if nr < 1:
        stream._fail(f"NR={nr}; an interpolation record needs at least one region",
                     f"{what}: NR", start)
    nbt: List[int] = []
    codes: List[int] = []
    for k in range(nr):
        pos = stream.position
        b = stream.int(f"{what}: NBT[{k}]")
        if b < 1 or (nbt and b <= nbt[-1]):
            stream._fail(f"NBT={b} does not increase strictly from "
                         f"{nbt[-1] if nbt else 0}", f"{what}: NBT[{k}]", pos)
        pos = stream.position
        c = stream.int(f"{what}: INT[{k}]")
        if c not in SUPPORTED_INTERPOLATION and not (qualified and c in QUALIFIED_INTERPOLATION):
            kind = ("a C/U variant Geant4 knows but no real file uses"
                    if c in range(11, 16) or c in range(21, 26)
                    else "not a code Geant4 knows (it would throw)")
            stream._fail(f"interpolation code {c} is not supported: {kind}",
                         f"{what}: INT[{k}]", pos)
        nbt.append(b)
        codes.append(c)
    if nbt[-1] != n:
        stream._fail(f"the last NBT is {nbt[-1]} but the record has {n} points",
                     f"{what}: NBT[{nr - 1}]", stream.position - 2)
    return Interpolation(tuple(nbt), tuple(codes))


def _interval_codes(interp: Interpolation, n: int) -> np.ndarray:
    """The code of each of the ``n - 1`` intervals.

    ENDF semantics: the interval from point ``j`` to ``j + 1`` (1-based)
    belongs to the first region with ``NBT >= j + 1``. A region's first
    node is therefore also the last node of the region before it.
    """
    point = np.empty(n, dtype=int)
    lo = 0
    for b, c in zip(interp.nbt, interp.codes):
        point[lo:b] = c
        lo = b
    return point[1:]


def _check_log_domain(stream, interp, x, y, what, start):
    """A log axis needs strictly positive values on every node of every
    interval it governs."""
    n = len(x)
    if n < 2:
        return
    iv = _interval_codes(interp, n) % 10  # a C/U variant takes the logs of its base code
    for axis, values, logs in (("x", x, _LOG_X), ("y", y, _LOG_Y)):
        log_iv = np.isin(iv, list(logs))
        touched = np.zeros(n, dtype=bool)
        touched[:-1] |= log_iv
        touched[1:] |= log_iv
        bad = np.flatnonzero(touched & (values <= 0))
        if bad.size:
            k = int(bad[0])
            code = iv[k - 1] if k > 0 and iv[k - 1] in logs else iv[k]
            stream._fail(f"interpolation code {code} takes ln {axis} but "
                         f"{axis}={float(values[k])!r} at point {k}", what, start)


def _check_non_decreasing(stream, values, what, start, stride, name):
    step = np.diff(values)
    bad = np.flatnonzero(step < 0)
    if bad.size:
        k = int(bad[0]) + 1
        stream._fail(f"{name} decreases at point {k}: {float(values[k - 1])!r} -> "
                     f"{float(values[k])!r}", what, start + stride * k)


# ------------------------------------------------------------ cross section

def parse_cross_section(stream: TokenStream, *, toEnd: bool = True,
                        nonnegative: bool = True) -> CrossSectionRecord:
    """``Elastic/CrossSection``: ``<int> <int> N (E σ)×N``, nothing after.

    ``toEnd=False`` stops after the pairs: a fission-chance file
    (``Fission/FC`` …) goes on with a final state after its σ.
    ``nonnegative=False`` admits a negative σ, which Geant4 reads as written:
    JEFF-4.0's Ac-225 first chance has one of -1e-9 b (spec §12).

    The two bookkeeping integers are kept; both libraries write ``0 0``
    everywhere, so they identify nothing and MT2 comes from the directory.
    Repeated energies are kept in order (``%e`` print collisions, spec §7);
    an energy that *decreases* is an error, as is a negative σ.
    """
    bk = (stream.int("CrossSection: bookkeeping[0]"),
          stream.int("CrossSection: bookkeeping[1]"))
    pos = stream.position
    n = stream.int("CrossSection: N")
    if n < 1:
        stream._fail(f"N={n}; a cross section needs at least one point",
                     "CrossSection: N", pos)
    start = stream.position
    energy, sigma = stream.pairs(n, f"CrossSection: {n} (E, sigma) pairs")
    if toEnd:
        stream.expectEnd("CrossSection: end of file")
    if energy[0] <= 0:
        stream._fail(f"first energy {float(energy[0])!r} eV is not positive",
                     "CrossSection: E[0]", start)
    _check_non_decreasing(stream, energy, "CrossSection: E", start, 2, "energy")
    neg = np.flatnonzero(sigma < 0)
    if neg.size and nonnegative:
        k = int(neg[0])
        stream._fail(f"negative cross section {float(sigma[k])!r} b at E={float(energy[k])!r} eV "
                     f"({neg.size} negative values)", "CrossSection: sigma",
                     start + 2 * k + 1)
    return CrossSectionRecord(stream.path, stream.header, bk, energy, sigma)


# -------------------------------------------------------------- final state

def parse_elastic_fs(stream: TokenStream) -> ElasticFSRecord:
    """``Elastic/FS``: ``repFlag targetMass frameFlag`` and its block(s)."""
    pos = stream.position
    rep = stream.int("FS: repFlag")
    if rep not in _REP_FLAGS:
        stream._fail(f"repFlag={rep}; Geant4 knows 0, 1, 2 and 3", "FS: repFlag", pos)
    pos = stream.position
    mass = stream.float("FS: targetMass")
    if mass <= 0:
        stream._fail(f"targetMass={mass!r} is not positive", "FS: targetMass", pos)
    frame = _frame(stream, "FS: frameFlag")

    frame2 = legendre = tabulated = None
    if rep == REP_ISOTROPIC:
        # The consumer reads frameFlag a second time and keeps that one.
        frame2 = _frame(stream, "FS repFlag=0: second frameFlag")
    if rep in (REP_LEGENDRE, REP_MIXED):
        legendre = _block(stream, tabulated=False, tag=f"FS repFlag={rep} Legendre")
    if rep in (REP_TABULATED, REP_MIXED):
        tabulated = _block(stream, tabulated=True, tag=f"FS repFlag={rep} table")
    if rep == REP_MIXED:
        _check_transition(stream, legendre, tabulated)
    stream.expectEnd("FS: end of file")
    return ElasticFSRecord(stream.path, stream.header, rep, mass, frame, frame2,
                           legendre, tabulated)


def _frame(stream: TokenStream, what: str) -> int:
    pos = stream.position
    f = stream.int(what)
    if f not in _FRAMES:
        stream._fail(f"frameFlag={f}; 1 is laboratory, 2 centre of mass", what, pos)
    return f


def _block(stream: TokenStream, *, tabulated: bool, tag: str) -> AngularBlock:
    pos = stream.position
    ne = stream.int(f"{tag}: NE")
    if ne < 1:
        # Fatal in the consumer for repFlag=3 (:154); meaningless for 1 and 2.
        stream._fail(f"NE={ne}; a block needs at least one incident energy",
                     f"{tag}: NE", pos)
    interp = parse_interpolation(stream, ne, f"{tag}: energy interpolation")
    start = stream.position
    records = []
    for i in range(ne):
        what = f"{tag}, energy {i + 1} of {ne}"
        temp = stream.float(f"{what}: T")
        energy = stream.float(f"{what}: E")
        tempdep = stream.int(f"{what}: tempdep")
        pos = stream.position
        n = stream.int(f"{what}: {'NP' if tabulated else 'NL'}")
        if n < (1 if tabulated else 0):
            stream._fail(f"{'NP' if tabulated else 'NL'}={n} is not a valid count",
                         what, pos)
        if tabulated:
            records.append(_tabulated(stream, temp, energy, tempdep, n, what))
        else:
            coeffs = stream.floats(n, f"{what}: {n} Legendre coefficients")
            records.append(LegendreRecord(temp, energy, tempdep, coeffs))
    block = AngularBlock(interp, tuple(records))
    e = block.energies
    if e[0] <= 0:
        stream._fail(f"first incident energy {float(e[0])!r} eV is not positive",
                     f"{tag}: E", start)
    bad = np.flatnonzero(np.diff(e) < 0)
    if bad.size:
        k = int(bad[0]) + 1
        stream._fail(f"incident energy decreases at record {k + 1}: {float(e[k - 1])!r} -> "
                     f"{float(e[k])!r}", f"{tag}: E", start)
    return block


def _tabulated(stream, temp, energy, tempdep, n, what) -> TabulatedRecord:
    # The mu interpolation record sits after NP and before the pairs (:138).
    interp = parse_interpolation(stream, n, f"{what}: mu interpolation")
    start = stream.position
    mu, p = stream.pairs(n, f"{what}: {n} (mu, p) pairs")
    out = np.flatnonzero((mu < -1.0) | (mu > 1.0))
    if out.size:
        k = int(out[0])
        stream._fail(f"mu={float(mu[k])!r} outside [-1, 1]", f"{what}: mu", start + 2 * k)
    _check_non_decreasing(stream, mu, f"{what}: mu", start, 2, "mu")
    neg = np.flatnonzero(p < 0)
    if neg.size:
        k = int(neg[0])
        stream._fail(f"negative probability density {float(p[k])!r} at mu={float(mu[k])!r}",
                     f"{what}: p", start + 2 * k + 1)
    _check_log_domain(stream, interp, mu, p, f"{what}: (mu, p)", start)
    return TabulatedRecord(temp, energy, tempdep, interp, mu, p)


def _check_transition(stream, legendre: AngularBlock, tabulated: AngularBlock) -> None:
    last = legendre.records[-1].energy
    first = tabulated.records[0].energy
    if abs(first - last) / last > TRANSITION_RTOL:
        stream._fail(f"the table starts at {float(first)!r} eV but the Legendre block "
                     f"ends at {float(last)!r} eV; Geant4 warns and goes on, leaving "
                     "the representation between them undefined",
                     "FS repFlag=3: transition", None)

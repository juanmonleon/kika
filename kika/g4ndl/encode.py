""":class:`~kika.nuclear_data.model.suite.ReactionSuite` → G4NDL elastic files.

Phase 6 of the G4NDL roadmap, the mirror of :mod:`kika.g4ndl.decode`. The
model goes to the same intermediate records the parser produces
(:mod:`kika.g4ndl.records`), and the records to text. Going through the records
is what makes the writer checkable: every file is parsed back with the strict
reader before it is handed out, and the records that come back must equal the
ones that went in, array for array. A writer that cannot pass its own reader
has no business writing for Geant4's.

**What is derived from the model and what is carried.** The physics comes from
the model; the tokens the model has no slot for come from the
:class:`~kika.nuclear_data.model.provenance.G4NDLProvenance` when the suite was
read from G4NDL, and take the values every real file has when it was not:

* ``repFlag`` is the **shape** of the angular form, never the provenance:
  ``Isotropic2d`` → 0, Legendre → 1, tables → 2, Legendre then tables → 3. A
  suite whose distribution was replaced (the collaborator's tabulated p(μ) in
  place of a Legendre fit) must not be written with the flag it was read with.
* ``frameFlag`` is the product frame. For ``repFlag=0`` the consumer reads it
  twice and keeps the second, so the second is the model's frame and the first
  is carried only while the two agree.
* ``targetMass`` (AWR), the two bookkeeping integers (``0 0``), the optional
  ``G4NDL <source>`` header, and ``T``/``tempdep`` per incident energy (``0``)
  are carried. ``T`` and ``tempdep`` are carried only while the block still has
  as many records as when it was read; otherwise they are reset, with a warning.

**Numbers.** Every float is written with ``repr``, the shortest string that
reads back to the same double, so at most 17 significant digits and never a
rounding. The distributed libraries print 7, which is where their repeated
energies come from (U-238: 25 % of the points, ``G4NDL_token_spec.md`` §7).
Every integer is written as an integer: Geant4 reads ``repFlag``, ``NE``,
``NR``, ``NBT``, ``INT``, ``NL``/``NP`` and ``tempdep`` with ``>> G4int``, and a
``2.0`` there leaves ``.0`` in the stream and misaligns the rest of the file
without an error.

**What is refused** (:class:`~kika.g4ndl.exceptions.G4NDLUnsupportedError`),
because writing it would make Geant4 read something else than the model says:

* a histogram in the model (interpolation code 1), in μ or in incident
  energy. Geant4 evaluates code 1 lin-lin, so writing it would say something
  else; Juan decided (2026-10-06) the writer refuses it rather than rewrite the
  step. A code 1 *read* from a file is held lin-lin and written back as it
  was declared (``tabulatedCode1``, roadmap Fase 10, D10-3);
* any code outside 2-5, including unit-base and corresponding-point qualifiers;
* a cross section that is not one pointwise lin-lin table. By default the
  ``recon`` form is written: G4NDL means σ at 0 K, pointwise, and an ENDF MF3
  ``eval`` is that only when the evaluation has no resonance region. Passing
  ``crossSectionLabel='eval'`` is the caller saying it is, and a suite whose
  ``resonances`` has a resolved or unresolved region is refused even then. For
  an ENDF tape, :func:`kika.endf.model_adapter.pendf.readReconstructed` adds
  the ``recon`` form from NJOY RECONR (roadmap Phase 9);
* a Legendre expansion whose ``a_0`` is not 1 (the file stores ``a_1 … a_NL``
  and the consumer takes ``a_0 = 1``), and units other than eV and barn;
* a suite with no ``targetMass`` to write: the AWR is physics (it sets the
  centre-of-mass kinematics), so it is never defaulted.
"""
from __future__ import annotations

import os
import re
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.g4ndl.exceptions import G4NDLError, G4NDLFormatError, G4NDLUnsupportedError
from kika.g4ndl.names import IsotopeKey, file_name
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.records import (
    FRAME_CM, FRAME_LAB, REP_ISOTROPIC, REP_LEGENDRE, REP_MIXED, REP_TABULATED,
    AngularBlock, CrossSectionRecord, ElasticFSRecord, Interpolation,
    LegendreRecord, TabulatedRecord,
)
from kika.g4ndl.tokens import HEADER_TAG, TokenStream
from kika.g4ndl.decode import ELASTIC_MT, EVALUATED_LABEL, RECONSTRUCTED_LABEL
from kika.nuclear_data.model import (
    AngularTwoBody, ConversionReport, Frame, G4NDLProvenance, Isotropic2d,
    Legendre, Regions1d, Regions2d, XYs1d, XYs2d,
)

__all__ = ["encodeElastic", "formatCrossSection", "formatElasticFS",
           "recordDifferences", "suiteProcesses", "targetKey", "writeElastic",
           "writeSuite", "KEEP"]

#: ``header=KEEP`` writes back the ``G4NDL <source>`` header the provenance
#: recorded (none, for a suite that did not come from G4NDL).
KEEP = "keep"

FRAME_TO_FLAG = {Frame.lab: FRAME_LAB, Frame.centerOfMass: FRAME_CM}

#: Interpolation codes the writer emits. 1 is refused (see the module
#: docstring); 11-15 and 21-25 are refused by the reader as well.
WRITABLE_INTERPOLATION = frozenset({2, 3, 4, 5})

#: How far ``a_0`` may sit from 1 and still be the implicit ``a_0 = 1``.
A0_TOLERANCE = 1.0e-12

_PAIRS_PER_LINE = 3
_VALUES_PER_LINE = 6
_METASTABLE_RE = re.compile(r"_m(\d+)$")


# ------------------------------------------------------------------ identity

def targetKey(suite) -> IsotopeKey:
    """The isotope the suite is about, as G4NDL names it.

    From the target's nuclide in ``PoPs`` (``Z``, ``A``; ``A`` of 0 or ``None``
    is the natural element) and the metastable suffix of its id
    (``Co58_m1``). An excited-level id (``_e2``) is refused: G4NDL names an
    isomer by ``M`` and kika will not guess which ``M`` a level is.
    """
    target = suite.target
    try:
        nuclide = suite.PoPs[target]
    except KeyError:
        nuclide = None
    z = getattr(nuclide, "Z", None)
    a = getattr(nuclide, "A", None)
    if z is None:
        raise G4NDLUnsupportedError(
            f"target {target!r} has no nuclide with Z in the suite's PoPs, so "
            f"the G4NDL file name <Z>_<A>_<Element> cannot be built")
    if "_e" in target:
        raise G4NDLUnsupportedError(
            f"target {target!r} names a nuclear level; G4NDL names isomers by "
            f"their metastable index M ('_m1'), and kika does not map one to the other")
    m = _METASTABLE_RE.search(target)
    return IsotopeKey(int(z), int(a) if a else None, int(m.group(1)) if m else 0)


# ------------------------------------------------------------------ model → records

def encodeElastic(suite, *, crossSectionLabel: Optional[str] = None,
                  angularLabel: str = EVALUATED_LABEL,
                  targetMass: Optional[float] = None, header=KEEP,
                  report: Optional[ConversionReport] = None,
                  ) -> Tuple[CrossSectionRecord, ElasticFSRecord, ConversionReport]:
    """The elastic channel (MT2) of ``suite`` → ``(crossSection, finalState, report)``.

    Parameters
    ----------
    crossSectionLabel
        The style of the cross section to write. Default ``'recon'``; see the
        module docstring for why ``'eval'`` must be asked for.
    angularLabel
        The style of the neutron's angular distribution. Default ``'eval'``.
    targetMass
        The AWR to write. Default: the one the provenance recorded
        (G4NDL ``targetMass``, or ENDF/ACE ``awr``). Refused when there is none.
    header
        ``KEEP`` (default) writes back the provenance's ``G4NDL <source>``
        header, ``None`` writes none, and a string writes ``G4NDL <string>``
        in both files. The label must be one token.

    The records come back with ``path=None``; :func:`writeElastic` puts them on
    disk.
    """
    report = report if report is not None else ConversionReport()
    reaction = _elasticReaction(suite)
    provenance = _g4ndlProvenance(suite, reaction)

    energy, sigma = _crossSectionArrays(suite, reaction, crossSectionLabel)
    bookkeeping = provenance.bookkeeping if provenance and provenance.bookkeeping else (0, 0)
    csHeader, fsHeader = _headers(header, provenance)
    cs = CrossSectionRecord(None, csHeader, tuple(int(b) for b in bookkeeping),
                            energy, sigma)

    form = _angularForm(reaction, angularLabel)
    mass = _targetMass(targetMass, suite, reaction, provenance)
    fs = _finalState(form, mass, fsHeader, provenance, report)
    return cs, fs, report


def _elasticReaction(suite):
    try:
        return suite.reactions[ELASTIC_MT]
    except (KeyError, IndexError):
        raise G4NDLUnsupportedError(
            f"the suite has no MT{ELASTIC_MT} reaction; the G4NDL writer writes "
            f"the elastic channel only") from None


def _g4ndlProvenance(suite, reaction) -> Optional[G4NDLProvenance]:
    for p in (getattr(reaction, "provenance", None), getattr(suite, "provenance", None)):
        if isinstance(p, G4NDLProvenance):
            return p
    return None


def _headers(header, provenance) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    if header == KEEP:
        if provenance is None:
            return None, None
        return provenance.crossSectionHeader, provenance.finalStateHeader
    if header is None:
        return None, None
    label = str(header)
    if not label or len(label.split()) != 1:
        raise ValueError(f"header label {label!r} must be one non-empty token: the "
                         f"consumer consumes exactly two tokens, '{HEADER_TAG}' and the label")
    return (HEADER_TAG, label), (HEADER_TAG, label)


# ------------------------------------------------------------ cross section

def _crossSectionArrays(suite, reaction, label: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    container = reaction.crossSection
    labels = list(container.keys()) if container is not None else []
    wanted = label if label is not None else RECONSTRUCTED_LABEL
    if wanted not in labels:
        hint = ("" if label is not None else
                f". G4NDL is pointwise sigma at 0 K, lin-lin, which is a "
                f"'{RECONSTRUCTED_LABEL}' form. For an ENDF tape, add it with "
                f"kika.endf.model_adapter.pendf.readReconstructed (NJOY RECONR, 0 K); "
                f"an ENDF 'eval' MF3 is pointwise sigma only without a resonance "
                f"region, and then crossSectionLabel='eval' writes it")
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT} has no cross section labelled {wanted!r} "
            f"(it has {labels}){hint}")
    form = container[wanted]
    resonances = getattr(suite, "resonances", None)
    if (wanted != RECONSTRUCTED_LABEL and resonances is not None
            and (resonances.resolved or resonances.unresolved is not None)):
        lo, hi = resonances.domain or (None, None)
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT} cross section {wanted!r}: the suite has a resonance region "
            f"({lo!r}-{hi!r} eV), so its MF3 there is only the background. Reconstruct "
            f"it first (kika.endf.model_adapter.pendf.readReconstructed, NJOY RECONR at "
            f"0 K) and write the 'recon' form")
    if isinstance(form, XYs1d):
        x, y, pairs = form.toEndfRegions()
        axes = form.axes
    elif isinstance(form, Regions1d):
        x, y, pairs = form.toEndfRegions()
        axes = form.axes or (form.function1ds[0].axes if form.function1ds else None)
    else:
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT} cross section {wanted!r} is a {type(form).__name__}; "
            f"G4NDL holds one pointwise table, so only XYs1d or Regions1d can be written")
    codes = {int(c) for _, c in pairs}
    if codes != {2}:
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT} cross section {wanted!r} has interpolation codes "
            f"{sorted(codes)}; the file has no interpolation record and Geant4 "
            f"reads it lin-lin, so anything but code 2 would change sigma between "
            f"nodes. Linearise it first")
    _checkUnits(axes, {"energy_in": "eV", "crossSection": "b"}, "cross section")
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def _checkUnits(axes, expected, what: str) -> None:
    for axis in getattr(axes, "axes", None) or []:
        want = expected.get(axis.label)
        if want is not None and axis.unit and axis.unit != want:
            raise G4NDLUnsupportedError(
                f"{what}: axis {axis.label!r} is in {axis.unit!r}; G4NDL is in "
                f"{want!r} and kika does not convert on the way out")


# ------------------------------------------------------------ final state

def _angularForm(reaction, label: str):
    products = reaction.outputChannel.products.byPid("n")
    if not products or products[0].distribution is None:
        raise G4NDLUnsupportedError(f"MT{ELASTIC_MT} has no outgoing neutron with a distribution")
    distribution = products[0].distribution
    if label not in distribution.keys():
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT}: the neutron has no distribution labelled {label!r} "
            f"(it has {list(distribution.keys())})")
    return distribution[label]


def _targetMass(explicit, suite, reaction, provenance) -> float:
    if explicit is not None:
        mass = float(explicit)
    elif provenance is not None and provenance.targetMass is not None:
        mass = float(provenance.targetMass)
    else:
        mass = None
        for p in (getattr(reaction, "provenance", None), getattr(suite, "provenance", None)):
            awr = getattr(p, "awr", None)
            if awr is not None:
                mass = float(awr)
                break
    if mass is None:
        raise G4NDLUnsupportedError(
            "no targetMass to write: the suite carries no G4NDL targetMass and no "
            "ENDF/ACE AWR. It sets the centre-of-mass kinematics, so pass "
            "targetMass= rather than let kika guess it")
    if not np.isfinite(mass) or mass <= 0:
        raise G4NDLUnsupportedError(f"targetMass={mass!r} is not a positive number")
    return mass


def _finalState(form, mass: float, header, provenance, report) -> ElasticFSRecord:
    if isinstance(form, AngularTwoBody) and isinstance(form.angular, Isotropic2d):
        form = Isotropic2d(productFrame=form.productFrame)
    if isinstance(form, Isotropic2d):
        frame2 = FRAME_TO_FLAG[form.productFrame]
        frame = frame2
        if (provenance is not None and provenance.repFlag == REP_ISOTROPIC
                and provenance.frameFlag2 == frame2 and provenance.frameFlag is not None):
            frame = int(provenance.frameFlag)
        return ElasticFSRecord(None, header, REP_ISOTROPIC, mass, frame, frame2)

    if not isinstance(form, AngularTwoBody) or form.angular is None:
        raise G4NDLUnsupportedError(
            f"MT{ELASTIC_MT}: the neutron's distribution is a {type(form).__name__}; "
            f"the elastic FS holds an isotropic, Legendre or tabulated p(mu|E) only")
    if form.productFrame not in FRAME_TO_FLAG:
        raise G4NDLUnsupportedError(f"product frame {form.productFrame!r} has no frameFlag")
    _checkUnits(form.angular.axes, {"energy_in": "eV"}, "angular distribution")
    frame = FRAME_TO_FLAG[form.productFrame]

    blocks = _splitBlocks(form.angular)
    legendre = tabulated = None
    for kind, node in blocks:
        functions, pairs = _tab2(node, kind)
        if kind == "Legendre":
            legendre = _legendreBlock(functions, pairs, provenance, report)
        else:
            tabulated = _tableBlock(functions, pairs, provenance, report)
    rep = {(True, False): REP_LEGENDRE, (False, True): REP_TABULATED,
           (True, True): REP_MIXED}[(legendre is not None, tabulated is not None)]
    return ElasticFSRecord(None, header, rep, mass, frame, None, legendre, tabulated)


def _leaves(node) -> List:
    if isinstance(node, Regions2d):
        out = []
        for child in node.function2ds:
            out.extend(_leaves(child))
        return out
    if isinstance(node, XYs2d):
        return [node]
    raise G4NDLUnsupportedError(f"angular form {type(node).__name__} is not XYs2d or Regions2d")


def _kind(node) -> str:
    kinds = set()
    for leaf in _leaves(node):
        for f in leaf.function1ds:
            kinds.add("Legendre" if isinstance(f, Legendre) else
                      "table" if isinstance(f, (XYs1d, Regions1d)) else type(f).__name__)
    if len(kinds) != 1 or not kinds <= {"Legendre", "table"}:
        return "mixed" if kinds <= {"Legendre", "table"} else "/".join(sorted(kinds))
    return kinds.pop()


def _splitBlocks(angular) -> List[Tuple[str, object]]:
    """The angular form → its one or two representation blocks.

    One kind throughout is one block (an ``XYs2d``, or a ``Regions2d`` of them
    for several energy regions). Two kinds is ``repFlag=3`` and must be the
    shape both decoders build: a ``Regions2d`` of exactly two children, the
    Legendre block then the table block, each an ``XYs2d`` or a ``Regions2d``
    of ``XYs2d``.
    """
    kind = _kind(angular)
    if kind in ("Legendre", "table"):
        return [(kind, angular)]
    if kind == "mixed" and isinstance(angular, Regions2d) and len(angular.function2ds) == 2:
        kinds = [_kind(c) for c in angular.function2ds]
        if kinds == ["Legendre", "table"]:
            return list(zip(kinds, angular.function2ds))
    if kind == "mixed":
        raise G4NDLUnsupportedError(
            "the angular form mixes Legendre and tabulated records in a shape "
            "other than repFlag=3's: a Regions2d of two children, the Legendre "
            "block first, each of one kind")
    raise G4NDLUnsupportedError(f"angular records of type {kind} cannot be written to G4NDL")


def _tab2(node, kind: str):
    """``toEndfTab2`` for a block, allowing one level of ``Regions2d``."""
    if isinstance(node, XYs2d):
        functions, pairs = list(node.function1ds), [(len(node), node.endfInterpolationCode)]
    else:
        functions, pairs = [], []
        for region in node.function2ds:
            if not isinstance(region, XYs2d):
                raise G4NDLUnsupportedError(
                    f"{kind} block: a Regions2d nested twice is not one interpolation record")
            functions.extend(region.function1ds)
            pairs.append((len(functions), region.endfInterpolationCode))
    _checkCodes([c for _, c in pairs], f"{kind} block, incident energy")
    return functions, Interpolation(tuple(int(b) for b, _ in pairs),
                                    tuple(int(c) for _, c in pairs))


def _checkCodes(codes: Sequence[int], what: str) -> None:
    for c in codes:
        if int(c) == 1:
            raise G4NDLUnsupportedError(
                f"{what}: interpolation code 1 (histogram). Geant4 evaluates it as "
                f"lin-lin (G4ParticleHPInterpolator.hh, the Histogram call is "
                f"commented out), so the file would mean something else to its "
                f"consumer than to the model; kika refuses to write it")
        if int(c) not in WRITABLE_INTERPOLATION:
            raise G4NDLUnsupportedError(
                f"{what}: interpolation code {c} is not one G4NDL can carry here "
                f"(2-5 only; unit-base and corresponding-point qualifiers are not read)")


def _carried(provenance, temps: str, deps: str, n: int, tag: str, report):
    """``T`` and ``tempdep`` from the provenance while the count still matches."""
    t = list(getattr(provenance, temps, None) or []) if provenance is not None else []
    d = list(getattr(provenance, deps, None) or []) if provenance is not None else []
    if len(t) == n and len(d) == n:
        return [float(x) for x in t], [int(x) for x in d]
    if t or d:
        report.warn(
            f"{tag} block: the provenance has T/tempdep for {len(t)} incident "
            f"energies and the block now has {n}; all are written as 0 (Geant4 "
            f"reads neither, G4ParticleHPElasticFS.cc:112-138)")
    return [0.0] * n, [0] * n


def _legendreBlock(functions, interp, provenance, report) -> AngularBlock:
    temps, deps = _carried(provenance, "legendreTemperatures", "legendreTempdeps",
                           len(functions), "Legendre", report)
    records = []
    for i, f in enumerate(functions):
        c = np.asarray(f.coefficients, dtype=np.float64)
        if c.size == 0 or abs(c[0] - 1.0) > A0_TOLERANCE:
            raise G4NDLUnsupportedError(
                f"Legendre record {i + 1} at E={f.outerDomainValue!r} eV has "
                f"a_0={c[0] if c.size else None!r}; the file stores a_1..a_NL and "
                f"the consumer takes a_0 = 1, so a distribution not normalised "
                f"that way would be read as a different one")
        records.append(LegendreRecord(temps[i], float(f.outerDomainValue), deps[i],
                                      c[1:].copy()))
    return AngularBlock(interp, tuple(records))


def _tableBlock(functions, interp, provenance, report) -> AngularBlock:
    temps, deps = _carried(provenance, "tabulatedTemperatures", "tabulatedTempdeps",
                           len(functions), "table", report)
    records = []
    code1 = dict(getattr(provenance, "tabulatedCode1", None) or {})
    for i, f in enumerate(functions):
        mu, p, pairs = f.toEndfRegions()
        what = f"table record {i + 1} at E={f.outerDomainValue!r} eV, mu"
        _checkCodes([c for _, c in pairs], what)
        declared = [(int(b), int(c)) for b, c in code1.get(str(i), ())]
        if declared and [(int(b), int(c)) for b, c in pairs] == [
                (b, 2 if c == 1 else c) for b, c in declared]:
            pairs = declared
        records.append(TabulatedRecord(
            temps[i], float(f.outerDomainValue), deps[i],
            Interpolation(tuple(int(b) for b, _ in pairs), tuple(int(c) for _, c in pairs)),
            np.array(mu, dtype=np.float64), np.array(p, dtype=np.float64)))
    return AngularBlock(interp, tuple(records))


# ------------------------------------------------------------------ records → text

def _f(x) -> str:
    """Shortest round-trip repr of a double: never rounds, at most 17 digits."""
    return repr(float(x))


def _lines(values: Sequence[str], per: int) -> List[str]:
    return [" ".join(values[k:k + per]) for k in range(0, len(values), per)]


def _interleaved(x: np.ndarray, y: np.ndarray) -> List[str]:
    flat = np.empty(2 * len(x), dtype=np.float64)
    flat[0::2] = x
    flat[1::2] = y
    return [repr(v) for v in flat.tolist()]


def _headerLine(header) -> List[str]:
    return [f"{header[0]} {header[1]}"] if header is not None else []


def _interpolationLines(interp: Interpolation) -> List[str]:
    return [str(interp.nRegions),
            " ".join(f"{int(b)} {int(c)}" for b, c in zip(interp.nbt, interp.codes))]


def formatCrossSection(record: CrossSectionRecord) -> str:
    """``Elastic/CrossSection`` text: ``b0 b1 N`` and ``N`` pairs, three per line."""
    out = _headerLine(record.header)
    out.append(f"{int(record.bookkeeping[0])} {int(record.bookkeeping[1])} {len(record.energy)}")
    out += _lines(_interleaved(record.energy, record.sigma), 2 * _PAIRS_PER_LINE)
    return "\n".join(out) + "\n"


def formatElasticFS(record: ElasticFSRecord) -> str:
    """``Elastic/FS`` text, in the order ``G4ParticleHPElasticFS::Init`` reads it."""
    out = _headerLine(record.header)
    out.append(f"{int(record.repFlag)} {_f(record.targetMass)} {int(record.frameFlag)}")
    if record.repFlag == REP_ISOTROPIC:
        out.append(str(int(record.frameFlag2)))
    for block, tabulated in ((record.legendre, False), (record.tabulated, True)):
        if block is None:
            continue
        out.append(str(len(block.records)))
        out += _interpolationLines(block.interpolation)
        for r in block.records:
            if tabulated:
                out.append(f"{_f(r.temperature)} {_f(r.energy)} {int(r.tempdep)} {len(r.mu)}")
                out += _interpolationLines(r.interpolation)
                out += _lines(_interleaved(r.mu, r.probability), 2 * _PAIRS_PER_LINE)
            else:
                out.append(f"{_f(r.temperature)} {_f(r.energy)} {int(r.tempdep)} "
                           f"{len(r.coefficients)}")
                out += _lines([repr(v) for v in r.coefficients.tolist()], _VALUES_PER_LINE)
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ verification

def _eqArray(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def recordDifferences(a, b) -> List[str]:
    """Every field where two records of the same kind differ; ``path`` is ignored.

    Exact comparison: arrays element for element (so repeated energies and
    their order count), interpolation tables tuple for tuple, floats by ``==``.
    """
    diffs: List[str] = []
    if isinstance(a, CrossSectionRecord):
        for name in ("header", "bookkeeping"):
            if tuple(getattr(a, name) or ()) != tuple(getattr(b, name) or ()):
                diffs.append(f"{name}: {getattr(a, name)!r} != {getattr(b, name)!r}")
        for name in ("energy", "sigma"):
            if not _eqArray(getattr(a, name), getattr(b, name)):
                diffs.append(f"{name} differs")
        return diffs
    for name in ("header", "repFlag", "targetMass", "frameFlag", "frameFlag2"):
        va, vb = getattr(a, name), getattr(b, name)
        if (tuple(va) if isinstance(va, tuple) else va) != (tuple(vb) if isinstance(vb, tuple) else vb):
            diffs.append(f"{name}: {va!r} != {vb!r}")
    for name in ("legendre", "tabulated"):
        ba, bb = getattr(a, name), getattr(b, name)
        if (ba is None) != (bb is None):
            diffs.append(f"{name}: present in one record only")
            continue
        if ba is None:
            continue
        if ba.interpolation != bb.interpolation:
            diffs.append(f"{name}: energy interpolation {ba.interpolation} != {bb.interpolation}")
        if len(ba.records) != len(bb.records):
            diffs.append(f"{name}: {len(ba.records)} != {len(bb.records)} records")
            continue
        for i, (ra, rb) in enumerate(zip(ba.records, bb.records)):
            for field in ("temperature", "energy", "tempdep"):
                if getattr(ra, field) != getattr(rb, field):
                    diffs.append(f"{name}[{i}].{field}: {getattr(ra, field)!r} != {getattr(rb, field)!r}")
            if name == "legendre":
                if not _eqArray(ra.coefficients, rb.coefficients):
                    diffs.append(f"{name}[{i}].coefficients differ")
            else:
                if ra.interpolation != rb.interpolation:
                    diffs.append(f"{name}[{i}].interpolation differs")
                if not (_eqArray(ra.mu, rb.mu) and _eqArray(ra.probability, rb.probability)):
                    diffs.append(f"{name}[{i}] (mu, p) differ")
    return diffs


def _verify(text: str, record, parser, what: str):
    """Parse ``text`` back with the strict reader; it must give ``record`` again."""
    try:
        back = parser(TokenStream(text))
    except G4NDLFormatError as exc:
        raise G4NDLUnsupportedError(
            f"{what}: the model holds something the G4NDL grammar refuses, so the "
            f"file was not written. The reader says: {exc}") from None
    diffs = recordDifferences(record, back)
    if diffs:  # a writer bug, not a property of the data
        raise G4NDLError(f"{what}: the written text does not read back to what was "
                         f"encoded: {'; '.join(diffs[:5])}")
    return back


# ------------------------------------------------------------------ on disk

def _replaceFile(path: Path, payload: bytes) -> None:
    """Write via a sibling temp file and ``os.replace``.

    The directory entry is swapped, never written through: if ``path`` is a
    hard link into another library, that library is left untouched.
    """
    tmp = path.with_name(f".{path.name}.kika-tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def writeElastic(suite, root, *, compressed: bool = False,
                 crossSectionLabel: Optional[str] = None,
                 angularLabel: str = EVALUATED_LABEL,
                 targetMass: Optional[float] = None, header=KEEP,
                 elementName: Optional[str] = None) -> ConversionReport:
    """Write the suite's elastic channel into the library directory ``root``.

    Writes ``root/Elastic/CrossSection/<name>`` and ``root/Elastic/FS/<name>``
    (with ``.z`` when ``compressed``: a raw zlib stream, what
    ``G4ParticleHPManager::GetDataStream`` uncompresses). If the other
    variant of either file is already there it is **removed**, and the report
    says so: Geant4 prefers the ``.z``, so a stale twin would silently shadow
    or be shadowed by what was just written. Nothing else in ``root`` is
    touched. Both texts are parsed back before anything is written.

    ``elementName`` overrides the file name's element (default: Geant4's own
    table, :data:`kika.g4ndl.names.GEANT4_ELEMENT_NAMES`).

    This is ``kika.write(suite, root, format="g4ndl")``. To replace an isotope
    in a copy of a whole library, use :func:`kika.g4ndl.patch_elastic`.
    """
    cs, fs, report = encodeElastic(suite, crossSectionLabel=crossSectionLabel,
                                   angularLabel=angularLabel, targetMass=targetMass,
                                   header=header)
    key = targetKey(suite)
    stem = file_name(key)
    if elementName is not None:
        stem = stem.rsplit("_", 1)[0] + "_" + elementName
    texts = {
        "Elastic/CrossSection": (formatCrossSection(cs), cs, parse_cross_section),
        "Elastic/FS": (formatElasticFS(fs), fs, parse_elastic_fs),
    }
    payloads = {}
    for sub, (text, record, parser) in texts.items():
        _verify(text, record, parser, f"{sub}/{stem}")
        data = text.encode("ascii")
        payloads[sub] = zlib.compress(data, 9) if compressed else data

    root = Path(root)
    for sub, payload in payloads.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        target = d / (stem + (".z" if compressed else ""))
        twin = d / (stem if compressed else stem + ".z")
        _replaceFile(target, payload)
        if twin.exists():
            twin.unlink()
            report.warn(f"removed {twin}: Geant4 reads the .z when both exist, so the "
                        f"{'plain' if compressed else 'compressed'} twin would have "
                        f"{'been shadowed by' if compressed else 'shadowed'} the file just written")
    return report


# ------------------------------------------------------------------ the whole suite

def suiteProcesses(suite) -> List[str]:
    """The G4NDL processes ``suite`` holds data for: ``elastic`` (MT2), ``inelastic``."""
    from kika.g4ndl.inelastic_decode import INELASTIC_SUM_LABEL
    from kika.g4ndl.inelastic_encode import channelOf

    out = []
    try:
        suite.reactions[ELASTIC_MT]
        out.append("elastic")
    except (KeyError, IndexError):
        pass
    if any(channelOf(r) is not None for r in list(suite.reactions) + list(suite.sums)) or any(
            r.id.label == INELASTIC_SUM_LABEL and r.id.ENDF_MT is None for r in suite.sums):
        out.append("inelastic")
    return out


def writeSuite(suite, root, *, processes: Optional[Sequence[str]] = None,
               compressed: bool = False, crossSectionLabel: Optional[str] = None,
               angularLabel: str = EVALUATED_LABEL, targetMass: Optional[float] = None,
               header=KEEP, elementName: Optional[str] = None) -> ConversionReport:
    """Write every process of ``suite`` into the library directory ``root``.

    ``processes`` defaults to :func:`suiteProcesses`: the elastic channel
    through :func:`writeElastic` (``crossSectionLabel``, ``angularLabel``) and
    the inelastic channels through
    :func:`kika.g4ndl.inelastic_encode.writeInelastic` (its distributions are
    always the ``eval`` style, and its sums follow their parts). This is
    ``kika.write(suite, root, format="g4ndl")``. Every file is encoded and
    read back before any is written, process by process.
    """
    from kika.g4ndl.inelastic_encode import writeInelastic

    wanted = suiteProcesses(suite) if processes is None else list(processes)
    unknown = set(wanted) - {"elastic", "inelastic"}
    if unknown or not wanted:
        raise ValueError(f"processes must be a non-empty subset of ('elastic', 'inelastic'), "
                         f"got {wanted!r}")
    report = ConversionReport()
    if "elastic" in wanted:
        report.extend(writeElastic(suite, root, compressed=compressed,
                                   crossSectionLabel=crossSectionLabel,
                                   angularLabel=angularLabel, targetMass=targetMass,
                                   header=header, elementName=elementName))
    if "inelastic" in wanted:
        report.extend(writeInelastic(suite, root, compressed=compressed, header=header,
                                     targetMass=targetMass, elementName=elementName))
    return report

"""Every ENDF/B-VIII.1 tape: kika's GNDS against NNDC's, and the 0 K cross section.

The library-wide counterpart of the three Fe-56 oracles beside this file. For
each tape of the library it

1. writes kika's GNDS of the tape and reads it back, reads NNDC's GNDS of the
   same evaluation (written by FUDGE), and compares the two models: the cross
   sections point by point, the resolved resonances as a set, the unresolved
   parameters, the covariances (gridded matrices on their union grid, the other
   components and the resonance-parameter covariances term by term) and every
   other field reaction by reaction;
2. reconstructs the 0 K cross section from the tape and from kika's GNDS and
   asks for the same points and the same values.

A difference is either zero or one of :data:`EXPECTED`, each with its reason and,
where it has a size, its bound. Anything else fails the test and is printed in
full. There is no tolerance outside that list.

Data: the tapes from ``KIKA_LIB_TAPES`` (``endfb81/``), NNDC's GNDS from
``KIKA_GNDS_LIB`` or ``gnds/endfb81`` beside ``KIKA_LIB_TAPES``. Without them
the tests skip. ``KIKA_LIBRARY_TAPES`` restricts the run to a comma-separated
list of tapes, by file stem (``n-026_Fe_056``) or by nuclide (``Fe-56``,
``Am-242m1``)::

    KIKA_LIBRARY_TAPES=H-1,Fe-56 pytest -m slow kika/gnds/tests/test_library_oracle.py

Marked ``slow`` and ``tape``: the whole library is hours of work.
"""
from __future__ import annotations

import collections
import dataclasses
import importlib.util
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from kika.nuclear_data.model import Reference, Regions1d, ResonancesWithBackground, XYs1d
from kika.nuclear_data.model.compare import modelDifferences
from kika.nuclear_data.model.covariances import (CovarianceMatrix, Mixed,
                                                 ShortRangeSelfScalingVariance, Sum)

# the repository's conftest, by path: a bare `from conftest import` takes whichever
# conftest a full collection put first on sys.path
_spec = importlib.util.spec_from_file_location(
    "_kika_root_conftest", Path(__file__).resolve().parents[3] / "conftest.py")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)

pytestmark = [pytest.mark.slow, pytest.mark.tape, pytest.mark.gnds]

#: Bookkeeping the field comparison skips everywhere: none of it is physics, and
#: ``crossSection`` is compared number by number in its own test.
IGNORED_FIELDS = ("report", "provenance", "applicationData", "documentation", "crossSection")


@dataclasses.dataclass(frozen=True)
class Expected:
    """A difference that is known and explained; *bound* caps its relative size."""

    check: str
    pattern: str
    reason: str
    bound: float = 0.0
    tapes: str = ".*"


@dataclasses.dataclass(frozen=True)
class Difference:
    check: str
    key: str
    size: Optional[float] = None

    def __str__(self):
        return self.key if self.size is None else f"{self.key} (rel {self.size:.2e})"


EXPECTED: List[Expected] = [
    Expected("xs", r"\w+ MT\d+: forms \['eval'\] != \['eval', 'recon'\]",
             "NNDC also carries FUDGE's reconstructed resonance cross section as a 'recon' form"),
    Expected("xs", r"Sums MT8(5[1-9]|6\d|7[01]): values", bound=4e-9,
             reason="MF33 lumped sums: each code adds the summands on its own and rounds the result"),
    Expected("field", r".*Q\.domainMin: # != #",
             "a constant Q: kika states it from the lowest incident energy, FUDGE from the threshold"),
    Expected("field", r".*multiplicity\.forms\['eval'\]: a constant XYs1d against Constant1d, same value",
             "a constant multiplicity: kika keeps ENDF's table, FUDGE writes constant1d"),
    Expected("field", r".*outputChannel\.process: None != '[^']*'",
             "outputChannel/@process is FUDGE's descriptive tag; kika does not write it"),
    Expected("field", r".*products\[[^\]]+\]: only in NNDC, distribution Unspecified",
             "FUDGE adds the residual ENDF does not tabulate, with an unspecified distribution"),
    Expected("field", r"Sums MT\d+ \| summands: MT \[.*\] != \[.*\]",
             "kika sums over partial sums (MT3, MT4…), FUDGE over the leaves; the xs test compares the values"),
    Expected("res", r"unresolved\.tabulatedWidths\.resonanceReactions\[i\]\.(label|href): .*",
             "reaction labels: kika's are MT numbers, FUDGE's the outgoing particles"),
    Expected("res", r"unresolved\.tabulatedWidths\.spinGroups\[i\]\.levelSpacingFunction\.label: 'eval' != None",
             "a form label kika writes and FUDGE omits"),
    Expected("res", r"unresolved\.tabulatedWidths\.spinGroups\[i\]\.channels: length # != #",
             "kika keeps the all-zero fission and competitive width columns of LRU=2; FUDGE drops them"),
    Expected("res", r"unresolved\.tabulatedWidths\.PoPs: PoPs != NoneType",
             "kika writes the URR's own PoPs (target spin and mass); FUDGE reads them from the suite's"),
    Expected("res", r"unresolved\.tabulatedWidths\.spinGroups\[i\]\.levelSpacingFunction\.interpolation: "
                    r"<Interpolation\.linlin: 'lin-lin'> != <Interpolation\.loglog: 'log-log'>",
             "the URR's INT is a declared write loss of kika's GNDS writer"),
    Expected("field", r".*products\[[^\]]+\]\.label: '[^']*' != '[^']*'",
             "a product label is a name; the photons of a channel are matched by line, not by label"),
    Expected("recon", r"MT8(5[1-9]|6\d|7[01]): values", bound=1e-15,
             reason="an MF33 lumped sum rebuilt from its summands; the order of the additions differs"),
    Expected("recon", r"MT\d+: npts \d+ != \d+", bound=1e-15,
             reason="a point collinear with its neighbours dropped where two regions merge, as FUDGE does"),
    Expected("recon", r"MT(1|2|102): npts \d+ != \d+", bound=2.3e-3, tapes="n-038_Sr_090",
             reason="the URR's INT=5 is a declared write loss, so the GNDS side interpolates lin-lin"),
    Expected("recon", r"MT(1|2|3|102|103): values", bound=2.4e-11, tapes="n-014_Si_028|n-026_Fe_054",
             reason="LRF=7 channel masses are a declared write loss; the GNDS side takes them from PoPs"),
]


def _explained(tape, difference):
    for entry in EXPECTED:
        if (entry.check == difference.check and re.fullmatch(entry.tapes, tape)
                and re.fullmatch(entry.pattern, difference.key)
                and (difference.size is None or difference.size <= entry.bound)):
            return entry
    return None


def _assertExplained(tape, differences):
    unexplained = [d for d in differences if _explained(tape, d) is None]
    assert not unexplained, (
        f"{tape}: {len(unexplained)} unexplained difference(s):\n  "
        + "\n  ".join(str(d) for d in unexplained))


def _nndcDirectory() -> Optional[Path]:
    stated = os.environ.get("KIKA_GNDS_LIB")
    if stated:
        return Path(stated)
    tapes = os.environ.get("KIKA_LIB_TAPES")
    return Path(tapes).parent / "gnds" / "endfb81" if tapes else None


def _stemOf(token):
    token = token.strip()
    if token.startswith("n-"):
        return token
    match = re.fullmatch(r"([A-Za-z]+)-?(\d+)(m\d*)?", token)
    if match is None:
        raise ValueError(f"KIKA_LIBRARY_TAPES: {token!r} is neither a tape stem nor a nuclide")
    symbol, mass, meta = match.groups()
    return f"_{symbol.capitalize()}_{int(mass):03d}{meta or ''}"


def _tapes():
    library = _root.resolve_library("endfb81")
    if library is None:
        return None, "the ENDF/B-VIII.1 tapes (set KIKA_LIB_TAPES)"
    stems = sorted(p.stem for p in library.glob("n-*.endf"))
    wanted = os.environ.get("KIKA_LIBRARY_TAPES")
    if wanted:
        keys = [_stemOf(t) for t in wanted.split(",") if t.strip()]
        stems = [s for s in stems if any(s == k or s.endswith(k) for k in keys)]
    return library, stems


def pytest_generate_tests(metafunc):
    if "tape" not in metafunc.fixturenames:
        return
    library, stems = _tapes()
    if library is None:
        metafunc.parametrize("tape", [pytest.param(None, marks=pytest.mark.skip(
            reason=f"{stems} not reachable"))], scope="module")
        return
    metafunc.parametrize("tape", stems, ids=stems, scope="module")


@pytest.fixture(scope="module")
def paths(tape):
    library, _ = _tapes()
    nndc = _nndcDirectory()
    endf = library / f"{tape}.endf"
    if nndc is None or not (nndc / f"{tape}.endf.gnds.xml").is_file():
        pytest.skip(f"NNDC's GNDS of {tape} not reachable (set KIKA_GNDS_LIB or KIKA_LIB_TAPES)")
    return endf, nndc


def _writeKikaGnds(endf, directory, covariances):
    import kika

    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{endf.stem}.endf.gnds.xml"
    kika.write(kika.read(str(endf), covariances=covariances), out, format="gnds")
    return out


def _readGnds(path, covariances=False):
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document

    document = Document.parse(path)
    suite, _ = readReactionSuite(document)
    if not covariances:
        return suite, None
    covarianceFile = path.parent / "Covariances" / path.name.replace(".gnds.xml", ".gnds-covar.xml")
    if not covarianceFile.is_file():
        return suite, None
    return suite, readCovarianceSuite(Document.parse(covarianceFile), {"reactions": document})[0]


@pytest.fixture(scope="module")
def kikaGnds(tape, paths, tmp_path_factory):
    return _writeKikaGnds(paths[0], tmp_path_factory.mktemp(tape), covariances=True)


@pytest.fixture(scope="module")
def suites(tape, paths, kikaGnds):
    return _readGnds(kikaGnds)[0], _readGnds(paths[1] / f"{tape}.endf.gnds.xml")[0]


def _byMT(suite):
    out = collections.defaultdict(list)
    for container in (suite.reactions, suite.sums, suite.productions, suite.fissionComponents,
                      suite.incompleteReactions, getattr(suite, "orphanProducts", [])):
        for reaction in container:
            if reaction.ENDF_MT is not None:
                out[(type(container).__name__, reaction.ENDF_MT)].append(reaction)
    return out


def _pointwise(form):
    if isinstance(form, ResonancesWithBackground):
        pieces = [_pointwise(region) for region in form.background.regions.values()
                  if region is not None]
        if not pieces or any(piece is None for piece in pieces):
            return None
        return (np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces]))
    if isinstance(form, XYs1d):
        return np.asarray(form.xs, float), np.asarray(form.ys, float)
    if isinstance(form, Regions1d):
        xs, ys, _ = form.toEndfRegions()
        return np.asarray(xs, float), np.asarray(ys, float)
    return None


def _collapseRepeatedPoints(xs, ys):
    """Drop each point that repeats the previous one in both coordinates."""
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = ~((xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1]))
    return xs[keep], ys[keep]


def _relative(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if b.size else 0.0


def _comparePointwise(check, name, mine, theirs):
    """Differences between two tabulated functions, after collapsing redundant repeats."""
    (mx, my), (tx, ty) = (_collapseRepeatedPoints(*mine), _collapseRepeatedPoints(*theirs))
    if mx.size != tx.size:
        grid = np.union1d(mx, tx)
        size = _relative(np.interp(grid, mx, my), np.interp(grid, tx, ty))
        return [Difference(check, f"{name}: npts {mx.size} != {tx.size}", size)]
    if np.array_equal(mx, tx) and np.array_equal(my, ty):
        return []
    return [Difference(check, f"{name}: values", max(_relative(mx, tx), _relative(my, ty)))]


def crossSectionDifferences(mine, theirs):
    out = []
    M, T = _byMT(mine), _byMT(theirs)
    for key in sorted(set(M) ^ set(T)):
        out.append(Difference("xs", f"{key[0]} MT{key[1]}: only in {'kika' if key in M else 'NNDC'}"))
    for key in sorted(set(M) & set(T)):
        name = f"{key[0]} MT{key[1]}"
        if len(M[key]) != len(T[key]):
            out.append(Difference("xs", f"{name}: {len(M[key])} != {len(T[key])} reactions"))
            continue
        for mr, tr in zip(M[key], T[key]):
            mf = mr.crossSection.forms if mr.crossSection else {}
            tf = tr.crossSection.forms if tr.crossSection else {}
            if set(mf) != set(tf):
                out.append(Difference("xs", f"{name}: forms {sorted(mf)} != {sorted(tf)}"))
            a, b = mf.get("eval"), tf.get("eval")
            if a is None and b is None:
                continue
            if isinstance(a, Reference) or isinstance(b, Reference):
                if not (isinstance(a, Reference) and isinstance(b, Reference)):
                    out.append(Difference("xs", f"{name}: {type(a).__name__} != {type(b).__name__}"))
                continue
            pa, pb = _pointwise(a), _pointwise(b)
            if pa is None or pb is None:
                out.append(Difference("xs", f"{name}: not compared, {type(a).__name__} "
                                            f"against {type(b).__name__}"))
                continue
            out += _comparePointwise("xs", name, pa, pb)
    return out


def test_cross_sections_are_nndcs(tape, suites):
    _assertExplained(tape, crossSectionDifferences(*suites))


def _resonanceRows(form):
    rows = []
    for group in getattr(form, "spinGroups", None) or []:
        spins = group.spins if group.spins else [group.spin] * len(group.energies)
        for energy, widths, spin in zip(group.energies, group.widths, spins):
            rows.append((float(energy), abs(float(spin or 0)))
                        + tuple(sorted(abs(float(w)) for w in widths if w)))
    parameters = getattr(form, "resonanceParameters", None)
    for group in (parameters.spinGroups if parameters else []):
        for r in group.resonances:
            rows.append((float(r.energy), abs(float(r.spin)), float(group.L), r.totalWidth,
                         r.neutronWidth, r.captureWidth, r.fissionWidth))
    return sorted(rows)


def resonanceDifferences(mine, theirs):
    a, b = mine.resonances, theirs.resonances
    if a is None and b is None:
        return []
    if (a is None) != (b is None):
        return [Difference("res", f"resonances only in {'kika' if a is not None else 'NNDC'}")]
    out = []
    if len(a.resolved) != len(b.resolved):
        out.append(Difference("res", f"resolved regions {len(a.resolved)} != {len(b.resolved)}"))
    for k, (ra, rb) in enumerate(zip(a.resolved, b.resolved)):
        name = f"resolved[{k}]"
        if type(ra.formalism) is not type(rb.formalism):
            out.append(Difference("res", f"{name}: {type(ra.formalism).__name__} != "
                                         f"{type(rb.formalism).__name__}"))
            continue
        if (ra.domainMin, ra.domainMax) != (rb.domainMin, rb.domainMax):
            out.append(Difference("res", f"{name}: domain [{ra.domainMin}, {ra.domainMax}] != "
                                         f"[{rb.domainMin}, {rb.domainMax}]"))
        A, B = _resonanceRows(ra.formalism), _resonanceRows(rb.formalism)
        if not A and not B:
            out.append(Difference("res", f"{name}: no resonance rows on either side"))
        elif A != B:
            out.append(Difference("res", f"{name}: resonances differ ({len(A)} != {len(B)} rows, "
                                         f"{sum(x != y for x, y in zip(A, B))} rows unequal)"))
    ua, ub = a.unresolved, b.unresolved
    if (ua is None) != (ub is None):
        out.append(Difference("res", f"unresolved only in {'kika' if ua is not None else 'NNDC'}"))
    elif ua is not None:
        for d in modelDifferences(ua, ub, path="unresolved", ignore=IGNORED_FIELDS):
            out.append(Difference("res", _normalised(d)))
    return out


def test_resonances_are_nndcs(tape, suites):
    _assertExplained(tape, resonanceDifferences(*suites))


def _normalisedPath(path):
    return re.sub(r"\[\d+\]", "[i]", re.sub(r"^@\.?", "", path))


def _normalised(message):
    path, _, rest = message.partition(": ")
    rest = re.sub(r"(?<![A-Za-z_\d.])[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", "#", rest)
    return f"{_normalisedPath(path)}: {rest}"[:240]


def _resolve(root, path):
    node = root
    for attribute, index, key in re.findall(r"\.(\w+)|\[(\d+)\]|\['([^']*)'\]", path):
        node = (getattr(node, attribute) if attribute else
                node[int(index)] if index else node[key])
    return node


def _tabulated(node):
    if isinstance(node, XYs1d):
        return np.asarray(node.xs, float), np.asarray(node.ys, float)
    return None


def _sized(check, where, message, a, b):
    """One ``modelDifferences`` message as a Difference, sized where it has a size.

    Arrays of the same shape get their largest relative difference; the abscissa
    or ordinate of two tables of different length gets the largest relative
    difference of the two tables interpolated on their union grid; a table whose
    values are all one number against a ``Constant1d`` of that number is named as
    such, since it states the same function.
    """
    path, _, rest = message.partition(": ")
    relative = path[1:]
    if rest == "arrays differ":
        x, y = (np.asarray(_resolve(o, relative), float) for o in (a, b))
        return Difference(check, f"{where}{_normalised(message)}", _relative(x, y))
    if rest.startswith("shape") and relative.endswith((".xs", ".ys")):
        pa, pb = (_tabulated(_resolve(o, relative[:-3])) for o in (a, b))
        if pa is not None and pb is not None:
            grid = np.union1d(pa[0], pb[0])
            size = _relative(np.interp(grid, *pa), np.interp(grid, *pb))
            return Difference(check, f"{where}{_normalised(message)}", size)
    if rest in ("XYs1d != Constant1d", "Constant1d != XYs1d"):
        x, y = _resolve(a, relative), _resolve(b, relative)
        table, constant = (x, y) if isinstance(x, XYs1d) else (y, x)
        if np.all(np.asarray(table.ys, float) == float(constant.constant)):
            return Difference(check, f"{where}{_normalisedPath(path)}: a constant XYs1d against "
                                     f"Constant1d, same value")
    return Difference(check, f"{where}{_normalised(message)}")


def _productOrder(product):
    """Photons of one channel in a code-independent order: by kind of line, then energy."""
    form = product.distribution.forms.get("eval") if product.distribution else None
    energy = getattr(form, "energy", None)
    return type(energy).__name__, float(getattr(energy, "value", 0.0) or 0.0)


def _channelDifferences(a, b, where):
    """*a* and *b*'s output channels, with the products matched by particle."""
    out = [_sized("field", where, d, a, b)
           for d in modelDifferences(a, b, path="@", ignore=IGNORED_FIELDS + ("products",))]
    pa = collections.defaultdict(list)
    pb = collections.defaultdict(list)
    for p in (a.products.products if a.products else []):
        pa[p.pid].append(p)
    for p in (b.products.products if b.products else []):
        pb[p.pid].append(p)
    for group in (*pa.values(), *pb.values()):
        group.sort(key=_productOrder)
    for pid in sorted(set(pa) | set(pb)):
        for k in range(max(len(pa[pid]), len(pb[pid]))):
            name = f"{where}products[{pid}#{k}]"
            if k >= len(pa[pid]) or k >= len(pb[pid]):
                extra = (pb if k >= len(pa[pid]) else pa)[pid][k]
                form = extra.distribution.forms.get("eval") if extra.distribution else None
                out.append(Difference("field", f"{name}: only in "
                                               f"{'NNDC' if k >= len(pa[pid]) else 'kika'}, "
                                               f"distribution {type(form).__name__}"))
                continue
            p, q = pa[pid][k], pb[pid][k]
            out += [_sized("field", f"{name}.", d, p, q)
                    for d in modelDifferences(p, q, path="@",
                                              ignore=IGNORED_FIELDS + ("outputChannel",))]
            if (p.outputChannel is None) != (q.outputChannel is None):
                out.append(Difference("field", f"{name}.outputChannel: only in "
                                               f"{'kika' if q.outputChannel is None else 'NNDC'}"))
            elif p.outputChannel is not None:
                out += _channelDifferences(p.outputChannel, q.outputChannel,
                                           f"{name}.outputChannel.")
    return out


def _summandMTs(reaction, labels):
    summands = getattr(reaction, "summands", None)
    if summands is None:
        return None
    return sorted(labels.get(re.search(r"\[@label='([^']*)'\]", s.href).group(1), s.href)
                  for s in summands.summands)


def _labelsToMT(suite):
    return {r.label: r.ENDF_MT for rs in _byMT(suite).values() for r in rs}


def fieldDifferences(mine, theirs):
    """Every field of every reaction, ``id`` aside: it restates the label and products.

    Summands are compared by the MT they point at, since the two codes label reactions
    differently.
    """
    out = []
    M, T = _byMT(mine), _byMT(theirs)
    labelsM, labelsT = _labelsToMT(mine), _labelsToMT(theirs)
    for key in sorted(set(M) & set(T)):
        where = f"{key[0]} MT{key[1]} | "
        for mr, tr in zip(M[key], T[key]):
            out += [_sized("field", where, d, mr, tr)
                    for d in modelDifferences(mr, tr, path="@", ignore=IGNORED_FIELDS
                                              + ("outputChannel", "id", "summands"))]
            sm, st = _summandMTs(mr, labelsM), _summandMTs(tr, labelsT)
            if sm != st:
                out.append(Difference("field", f"{where}summands: MT {sm} != {st}"))
            if (mr.outputChannel is None) != (tr.outputChannel is None):
                out.append(Difference("field", f"{where}outputChannel: only in "
                                               f"{'kika' if tr.outputChannel is None else 'NNDC'}"))
            elif mr.outputChannel is not None:
                out += _channelDifferences(mr.outputChannel, tr.outputChannel,
                                           f"{where}outputChannel.")
    return out


def test_other_fields_are_nndcs(tape, suites):
    _assertExplained(tape, fieldDifferences(*suites))


def _components(form):
    if isinstance(form, Mixed):
        return [c for component in form.components for c in _components(component)]
    return [form]


def _dense(matrices, grid):
    mids = 0.5 * (grid[:-1] + grid[1:])
    total = np.zeros((mids.size, mids.size))
    for c in matrices:
        rows = c.rowGrid
        columns = c.columnGrid if c.columnGrid is not None else rows
        i = np.searchsorted(rows, mids, side="right") - 1
        j = np.searchsorted(columns, mids, side="right") - 1
        oki = (i >= 0) & (i < c.matrix.shape[0])
        okj = (j >= 0) & (j < c.matrix.shape[1])
        block = np.zeros_like(total)
        block[np.ix_(oki, okj)] = c.matrix[np.ix_(i[oki], j[okj])]
        total += block
    return total


def _sectionKey(section):
    def end(link):
        sl = getattr(link, "slices", None)
        L = tuple(x.domainValue for x in sl.slices) if sl is not None and sl.slices else ()
        return (link.ENDF_MF, link.ENDF_MT, L)

    row = end(section.rowData)
    return row, end(section.columnData) if section.columnData is not None else row


def _formDifferences(name, a, b, targetsA, targetsB):
    out = []
    ca, cb = _components(a), _components(b)
    for relative in (True, False):
        ga = [c for c in ca if isinstance(c, CovarianceMatrix) and c.rowGrid is not None
              and c.isRelative == relative]
        gb = [c for c in cb if isinstance(c, CovarianceMatrix) and c.rowGrid is not None
              and c.isRelative == relative]
        if not ga and not gb:
            continue
        grids = [g for c in ga + gb for g in (c.rowGrid, c.columnGrid) if g is not None]
        grid = np.unique(np.concatenate([np.asarray(g, float) for g in grids]))
        da, db = _dense(ga, grid), _dense(gb, grid)
        if not np.array_equal(da, db):
            scale = np.abs(db).max() or 1.0
            out.append(Difference("cov", f"{name}: {'relative' if relative else 'absolute'} "
                                         f"matrix", float(np.abs(da - db).max() / scale)))
    gridded = lambda c: isinstance(c, CovarianceMatrix) and c.rowGrid is not None  # noqa: E731
    for kind in (ShortRangeSelfScalingVariance, Sum):
        sa = [c for c in ca if isinstance(c, kind)]
        sb = [c for c in cb if isinstance(c, kind)]
        if len(sa) != len(sb):
            out.append(Difference("cov", f"{name}: {kind.__name__} {len(sa)} != {len(sb)}"))
            continue
        for x, y in zip(sa, sb):
            if kind is Sum:
                out += _sumDifferences(name, x, y, targetsA, targetsB)
                continue
            for d in modelDifferences(x, y, path=kind.__name__, ignore=("label",)):
                out.append(Difference("cov", f"{name}: {_normalised(d)}"))
    rest = [type(c).__name__ for c in ca + cb
            if not gridded(c) and not isinstance(c, (ShortRangeSelfScalingVariance, Sum))]
    if rest:
        out.append(Difference("cov", f"{name}: not compared, {sorted(set(rest))}"))
    return out


def _sumDifferences(name, a, b, targetsA, targetsB):
    """A ``sum`` by what it adds: the summed sections' (row, column) keys and coefficients."""
    def terms(form, targets):
        return sorted((str(targets.get(re.search(r"covarianceSection\[@label='([^']*)'\]",
                                                 s.href).group(1), s.href)), s.coefficient)
                      for s in form.summands)

    out = []
    if (a.domainMin, a.domainMax) != (b.domainMin, b.domainMax):
        out.append(Difference("cov", f"{name}: sum domain [{a.domainMin}, {a.domainMax}] != "
                                     f"[{b.domainMin}, {b.domainMax}]"))
    if terms(a, targetsA) != terms(b, targetsB):
        out.append(Difference("cov", f"{name}: sum terms {terms(a, targetsA)} != "
                                     f"{terms(b, targetsB)}"))
    return out


def _parameterDifferences(mine, theirs):
    out = []
    if len(mine) != len(theirs):
        return [Difference("cov", f"parameter covariances {len(mine)} != {len(theirs)}")]
    for k, (a, b) in enumerate(zip(mine, theirs)):
        fa, fb = a.form, b.form
        name = f"parameterCovariance[{k}]"
        ma, mb = getattr(fa, "matrix", None), getattr(fb, "matrix", None)
        if ma is None or mb is None:
            out.append(Difference("cov", f"{name}: not compared, {type(fa).__name__} "
                                         f"against {type(fb).__name__}"))
            continue
        ma, mb = np.asarray(ma, float), np.asarray(mb, float)
        if ma.shape != mb.shape:
            out.append(Difference("cov", f"{name}: shape {ma.shape} != {mb.shape}"))
        elif not np.array_equal(ma, mb):
            scale = np.abs(mb).max() or 1.0
            out.append(Difference("cov", f"{name}: matrix", float(np.abs(ma - mb).max() / scale)))
        if getattr(fa, "isRelative", None) != getattr(fb, "isRelative", None):
            out.append(Difference("cov", f"{name}: isRelative {fa.isRelative} != {fb.isRelative}"))
        na = sum(p.nParameters for p in getattr(fa, "parameters", []))
        nb = sum(p.nParameters for p in getattr(fb, "parameters", []))
        if na != nb:
            out.append(Difference("cov", f"{name}: {na} != {nb} parameters"))
    return out


def covarianceDifferences(mine, theirs):
    if mine is None and theirs is None:
        return []
    if (mine is None) != (theirs is None):
        return [Difference("cov", f"covariance file only in {'kika' if mine is not None else 'NNDC'}")]
    out = []
    M, T = collections.defaultdict(list), collections.defaultdict(list)
    for section in mine.covarianceSections:
        M[_sectionKey(section)].append(section)
    for section in theirs.covarianceSections:
        T[_sectionKey(section)].append(section)
    targetsA = {s.label: _sectionKey(s) for s in mine.covarianceSections}
    targetsB = {s.label: _sectionKey(s) for s in theirs.covarianceSections}
    for key in sorted(set(M) ^ set(T), key=str):
        out.append(Difference("cov", f"section {key}: only in {'kika' if key in M else 'NNDC'}"))
    for key in sorted(set(M) & set(T), key=str):
        if len(M[key]) != len(T[key]):
            out.append(Difference("cov", f"section {key}: {len(M[key])} != {len(T[key])} sections"))
            continue
        for a, b in zip(M[key], T[key]):
            out += _formDifferences(f"section {key}", a.form, b.form, targetsA, targetsB)
    out += _parameterDifferences(list(mine.parameterCovariances), list(theirs.parameterCovariances))
    return out


def test_covariances_are_nndcs(tape, paths, kikaGnds):
    theirs = paths[1] / f"{tape}.endf.gnds.xml"
    _assertExplained(tape, covarianceDifferences(_readGnds(kikaGnds, True)[1],
                                                 _readGnds(theirs, True)[1]))


def _reconstructed(path, covariances):
    """*covariances* for the tape: its MF33 lumped sums are reactions of the GNDS side too."""
    import kika
    from kika.processing.resonances.suite import reconstruct_suite

    result = reconstruct_suite(kika.read(str(path), covariances=covariances), workers=2)
    forms = result.forms_by_mt() if callable(result.forms_by_mt) else result.forms_by_mt
    return dict(forms)


def _flattened(form):
    if hasattr(form, "xs"):
        return np.asarray(form.xs, float), np.asarray(form.ys, float)
    return (np.concatenate([np.asarray(f.xs, float) for f in form.function1ds]),
            np.concatenate([np.asarray(f.ys, float) for f in form.function1ds]))


def reconstructionDifferences(fromTape, fromGnds):
    out = [Difference("recon", f"MT{mt}: only from the {'tape' if mt in fromTape else 'GNDS'}")
           for mt in sorted(set(fromTape) ^ set(fromGnds))]
    for mt in sorted(set(fromTape) & set(fromGnds)):
        out += _comparePointwise("recon", f"MT{mt}", _flattened(fromGnds[mt]), _flattened(fromTape[mt]))
    return out


def test_reconstruction_from_kikas_gnds_is_the_tapes(tape, paths, kikaGnds):
    from kika.processing.resonances.grid import ReconstructionConvergenceError
    from kika.processing.resonances.prepare import UnsupportedResonanceError

    try:
        fromTape = _reconstructed(paths[0], True)
    except (UnsupportedResonanceError, ReconstructionConvergenceError) as refusal:
        pytest.skip(f"the reconstruction refuses the tape itself: {refusal}")
    _assertExplained(tape, reconstructionDifferences(fromTape, _reconstructed(kikaGnds, False)))

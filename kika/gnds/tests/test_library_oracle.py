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
the tests skip. ``KIKA_LIBRARY=jeff40`` runs the library ``jeff40/`` instead;
with no NNDC GNDS for it, only the reconstruction test runs. ``KIKA_LIBRARY_TAPES``
restricts the run to a comma-separated list of tapes, by file stem
(``n-026_Fe_056``, ``Fe56_jeff4.0``) or by nuclide (``Fe-56``, ``Am-242m1``)::

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

from kika.nuclear_data.model import (Interpolation, Reference, Regions1d,
                                     ResonancesWithBackground, XYs1d)
from kika.nuclear_data.model.compare import modelDifferences
from kika.nuclear_data.model.covariances import (CovarianceMatrix, Mixed,
                                                 ShortRangeSelfScalingVariance, Sum)

# by path: a bare `from conftest import` takes whichever conftest is first on sys.path
_spec = importlib.util.spec_from_file_location(
    "_kika_root_conftest", Path(__file__).resolve().parents[3] / "conftest.py")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)

pytestmark = [pytest.mark.slow, pytest.mark.tape, pytest.mark.gnds]

#: Bookkeeping the field comparison skips everywhere: none of it is physics, and
#: ``crossSection`` is compared number by number in its own test.
IGNORED_FIELDS = ("report", "provenance", "applicationData", "documentation", "crossSection")

LIBRARY = os.environ.get("KIKA_LIBRARY", "endfb81")
TAPE_GLOBS = {"endfb81": "n-*.endf", "jeff40": "*_jeff4.0.endf"}


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
    Expected("xs", r"Sums MT8(5[1-9]|6\d|7[01]): values( \(NNDC's nine digits\))?", bound=4e-9,
             reason="MF33 lumped sums: each code adds the summands on its own and rounds the result"),
    Expected("xs", r"Sums MT103: values \(NNDC's nine digits\)", tapes="n-026_Fe_056",
             reason="a summed MT only MF33 states: kika keeps the sum of the partials as doubles, "
                    "FUDGE writes it to nine significant digits"),
    Expected("field", r".*Q\.domainMin: kika's 1e-05 below NNDC's",
             "a constant Q: kika states it from the lowest incident energy, FUDGE from the threshold"),
    Expected("field", r".*multiplicity\.forms\['eval'\]: a constant XYs1d against Constant1d, same value",
             "a constant multiplicity: kika keeps ENDF's table, FUDGE writes constant1d"),
    Expected("field", r".*outputChannel\.process: None != '[^']*'",
             "outputChannel/@process is FUDGE's descriptive tag; kika does not write it"),
    Expected("field", r".*products\[[^\]]+\]: only in NNDC, distribution Unspecified",
             "FUDGE adds the residual ENDF does not tabulate, with an unspecified distribution"),
    Expected("field", r"Sums MT\d+ \| summands: MT \[.*\] != \[.*\]",
             "kika sums over partial sums (MT3, MT4…), FUDGE over the leaves; the xs test compares the values"),
    Expected("field", r"OrphanProducts MT\d+ \| outputChannel\.products\[photon#\d+\]\.multiplicity"
                      r"\.forms\['eval'\](\.function1ds\[i\])?\.(xs|ys): shape \(#,\) != \(#,\)",
             bound=7.9, tapes="n-005_B_011|n-011_Na_023",
             reason="MF13 sigma_g/sigma: kika divides on the union grid (m*sigma is the tape's sigma_g), "
                    "FUDGE on MF13's grid only"),
    Expected("field", r"OrphanProducts MT\d+ \| outputChannel\.products\[photon#\d+\]\.distribution"
                      r"\.forms\['eval'\]\.energy\.domainMin: # != #", bound=3.4, tapes="n-005_B_011",
             reason="a line's domain: kika starts it where its MF13 table does, FUDGE at the threshold"),
    Expected("field", r"OrphanProducts MT3 \| outputChannel\.products\[photon#0\]\.multiplicity\.forms"
                      r"\['eval'\]\.function1ds\[i\]\.interpolation: <Interpolation\.linlin: 'lin-lin'> != "
                      r"<Interpolation\.flat: 'flat'>", tapes="n-011_Na_023",
             reason="the tape's histogram region has sigma_g = 0; a zero multiplicity is the same in any law"),
    Expected("field", r".*: only the last value of a histogram, which no interval uses",
             "kika keeps the tape's trailing value of a flat table, FUDGE writes 0"),
    Expected("field", r".*: NNDC moved a step or end point by at most 0\.001 eV",
             "FUDGE moves a repeated or final outgoing energy by 1e-3 eV; kika keeps the tape's table"),
    Expected("field", r".*: two regions in kika, one table with a step moved by at most 0\.001 eV in NNDC",
             "a step in outgoing energy: kika keeps it as two regions, FUDGE moves the point by 1e-6 eV"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.(products\[[^\]]+\]\.outputChannel\.)?"
                      r"products\[photon#\d+\]\.(multiplicity\.forms\['eval'\]\.ys|distribution\.forms"
                      r"\['eval'\]\.energy\.function1ds\[i\]\.ys): arrays differ", bound=5e-15,
             reason="photon yields and normalised spectra are computed doubles; FUDGE writes them at 15 "
                    "significant digits, at most 5e-15 off (Pb-207 MT16: kika's equal the tape's decode)"),
    Expected("field", r".*delayedNeutrons\[i\]\.label: '#' != '#'",
             "delayed-neutron families: FUDGE labels them from 0, kika from 1"),
    Expected("field", r".*delayedNeutrons\[i\]\.product\.distribution\.forms\['eval'\]\.energy\.g\.axes"
                      r"\.axes\[i\]\.label: 'energy_out / theta\(energy_in\)' != 'energy_out / theta\( energy_in \)'",
             "axis label text; FUDGE pads the parentheses"),
    Expected("field", r".*delayedNeutrons\[i\]\.product\.multiplicity\.forms\['eval'\]\.ys: arrays differ",
             bound=2e-16, reason="p_k(E) * nu_d(E) evaluated by each code"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.products\[[^\]]+\]\.outputChannel\.Q\.value: # != #",
             bound=2e-16, reason="a level energy QM - QI computed in binary; FUDGE keeps the decimal"),
    Expected("field", r"Reactions MT802 \| outputChannel\.products\[Cr53_e2#0\]\.outputChannel\.Q\.value: # != #",
             bound=1e-7, tapes="n-026_Fe_056",
             reason="NNDC wrong: the tape's QM - QI is 1006270.1 eV, FUDGE writes its PoPs level, 1006270.0"),
    Expected("field", r"Reactions MT6(0[2-9]|1[0-6]) \| outputChannel\.products\[Tl207_e\d+#0\]\.outputChannel"
                      r"\.Q\.value: # != #", bound=2.3e-7, tapes="n-082_Pb_207",
             reason="the tape disagrees with itself: MF3's QM - QI puts each level 0.3 eV below MF12's ES "
                    "(MT602: 1348179.7 against 1348180 eV); kika takes MF3's, as its PoPs does, FUDGE MF12's"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.products\[[^\]]+\]\.distribution\.forms\['eval'\]"
                      r"\.energy\.mass\.value: # != #", bound=2.1e-10,
             reason="phase-space mass APSX * m_n: kika's m_n is CODATA's 1.00866491595 u, FUDGE's differs"),
    Expected("field", r"Productions MT\d+ \| outputChannel\.products\[[^\]]+\]\.multiplicity\.forms\['eval'\]"
                      r"\.domain(Min|Max)_: # != #", bound=3e12, tapes="n-025_Mn_055|n-040_Zr_090|n-092_U_235",
             reason="kika states the MF10 table's domain (Mn-55 MT30: 29.4-60 MeV); FUDGE the reaction's, or 1e-5-20 MeV"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.products\[n#0\]\.distribution\.forms\['eval'\]"
                      r"\.energy\.weighted\[i\]\.weight\.label: None != 'weight'",
             "a form label FUDGE writes and kika omits"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.products\[photon#\d+\]\.distribution\.forms\['eval'\]: "
                      r"Uncorrelated != AngularTwoBody",
             "an MF6 LAW=2 photon with AWP: kika keeps the stated energy as primaryGamma, which "
             "angularTwoBody cannot hold; the Legendre table is the same"),
    Expected("field", r"Reactions MT\d+ \| outputChannel\.products\[[^\]]+\]\.distribution\.forms\['eval'\]"
                      r"\.recoilHref: \"\.\./\.\./\.\./\.\./product\[@label='photon'\]/distribution/uncorrelated"
                      r"\[@label='eval'\]\" != \"\.\./\.\./\.\./\.\./product\[@label='photon'\]/distribution/"
                      r"angularTwoBody\[@label='eval'\]\"",
             "the recoil points at the photon's form, uncorrelated in kika for the reason above"),
    Expected("field", r"Reactions MT18 \| outputChannel\.Q\.value: # != #", bound=5e-15,
             reason="MF1/458's EFR+ENP+EGP at 1e-5 eV: kika keeps the double, FUDGE writes 15 digits"),
    Expected("field", r".*\.energy\.function1ds\[i\]: all zero in kika, a unit box on \[0, 2e-05\] eV in NNDC",
             "NNDC wrong: the tape's continuum is zero there (Cl-35 MT32 at 16.01184 MeV, on [0, 3e5] eV); "
             "FUDGE replaces it by a box"),
    Expected("xs", r"Sums MT852: npts 830 != 7910", bound=8.93e-4, tapes="n-092_U_235",
             reason="NNDC wrong: FUDGE's lumped sum is 8.9e-4 off the tape's log-log summands, kika's 7.9e-5"),
    Expected("res", r"unresolved\.tabulatedWidths\.scatteringRadius: # != #", bound=2e-16,
             reason="AP * 10 to fm in binary; FUDGE writes the decimal"),
    Expected("res", r"unresolved\.tabulatedWidths\.PoPs\.particles\['[^']+'\]\.mass\.value: # != #", bound=2.1e-10,
             reason="the target mass is AWRI * m_n in kika, FUDGE's PoPs value"),
    Expected("res", r"unresolved\.tabulatedWidths\.PoPs\.particles\['[^']+'\]\.(charge|energy): None != .*",
             "kika's URR PoPs carries the target's mass and spin only"),
    Expected("res", r"unresolved\.spinGroups\[i\]\.levelSpacingFunction\.label: 'eval' != None",
             "a form label kika writes and FUDGE omits"),
    Expected("res", r"unresolved\.tabulatedWidths\.PoPs: PoPs != NoneType",
             "kika writes the URR's own PoPs (target spin and mass); FUDGE reads them from the suite's"),
    Expected("field", r".*products\[[^\]]+\]\.label: '[^']*' != '[^']*'",
             "a product label is a name; the photons of a channel are matched by line, not by label"),
    Expected("cov", r"parameterCovariance\[\d+\]: matrix", bound=1e-15,
             reason="FUDGE writes the parameter covariance with 15 significant figures, kika every digit"),
    Expected("recon", r"MT8(5[1-9]|6\d|7[01]): values", bound=1e-15,
             reason="an MF33 lumped sum rebuilt from its summands; the order of the additions differs"),
    Expected("recon", r"MT\d+: npts \d+ != \d+", bound=1e-15,
             reason="a point collinear with its neighbours dropped where two regions merge, as FUDGE does"),
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
    return Path(tapes).parent / "gnds" / LIBRARY if tapes else None


def _stemOf(token):
    token = token.strip()
    if token.startswith("n-") or "_jeff" in token:
        return token
    match = re.fullmatch(r"([A-Za-z]+)-?(\d+)(m\d*)?", token)
    if match is None:
        raise ValueError(f"KIKA_LIBRARY_TAPES: {token!r} is neither a tape stem nor a nuclide")
    symbol, mass, meta = match.groups()
    if LIBRARY == "jeff40":
        return f"{symbol.capitalize()}{int(mass)}{'m' if meta else ''}_jeff4.0"
    return f"_{symbol.capitalize()}_{int(mass):03d}{meta or ''}"


def _tapes():
    if LIBRARY not in TAPE_GLOBS:
        raise ValueError(f"KIKA_LIBRARY: {LIBRARY!r} is not one of {sorted(TAPE_GLOBS)}")
    library = _root.resolve_library(LIBRARY)
    if library is None:
        return None, f"the {LIBRARY} tapes (set KIKA_LIB_TAPES)"
    stems = sorted(p.stem for p in library.glob(TAPE_GLOBS[LIBRARY]))
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
def endf(tape):
    return _tapes()[0] / f"{tape}.endf"


@pytest.fixture(scope="module")
def paths(tape, endf):
    nndc = _nndcDirectory()
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
def kikaGnds(tape, endf, tmp_path_factory):
    return _writeKikaGnds(endf, tmp_path_factory.mktemp(tape), covariances=True)


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
    if check == "xs" and np.array_equal(mx, tx) and np.array_equal(_nineDigits(my), ty):
        return [Difference(check, f"{name}: values (NNDC's nine digits)")]
    return [Difference(check, f"{name}: values", max(_relative(mx, tx), _relative(my, ty)))]


def _nineDigits(values):
    """*values* as FUDGE writes a computed table: nine significant digits."""
    return np.array([float(f"{v:.8e}") for v in values])


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
                        + tuple(sorted(float(w) for w in widths)))
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
        out += _unresolvedDifferences(ua, ub, _labelsToMT(mine), _labelsToMT(theirs))
    return out


def _channelRoles(resonanceReactions, channels, labels):
    """``{reaction MT: channel}`` for the channels whose widths are not all zero."""
    reactionOf = {r.label: labels.get(re.search(r"\[@label='([^']*)'\]", r.href).group(1), r.href)
                  for r in resonanceReactions}
    return {reactionOf.get(c.label, c.label): c for c in channels
            if c.constantWidth or (c.widths is not None and np.any(np.asarray(c.widths) != 0))}


def _unresolvedDifferences(ua, ub, labelsA, labelsB):
    # resonanceReactions only name the channels' reactions; the channels are paired through them
    ignore = IGNORED_FIELDS + ("spinGroups", "resonanceReactions")
    out = [_sized("res", "unresolved.", d, ua, ub)
           for d in modelDifferences(ua, ub, path="@", ignore=ignore)]
    ta, tb = ua.tabulatedWidths, ub.tabulatedWidths
    if ta is None or tb is None or len(ta.spinGroups) != len(tb.spinGroups):
        return out + [Difference("res", "unresolved: spin groups not paired")]
    for k, (ga, gb) in enumerate(zip(ta.spinGroups, tb.spinGroups)):
        name = f"unresolved.spinGroups[{k}]"
        out += [_sized("res", "unresolved.spinGroups[i].", d, ga, gb) for d in
                modelDifferences(ga, gb, path="@", ignore=IGNORED_FIELDS + ("channels",))]
        ca = _channelRoles(ta.resonanceReactions, ga.channels, labelsA)
        cb = _channelRoles(tb.resonanceReactions, gb.channels, labelsB)
        if set(ca) != set(cb):
            out.append(Difference("res", f"{name}: channels with widths for MT {sorted(ca, key=str)} "
                                         f"!= {sorted(cb, key=str)}"))
        for role in sorted(set(ca) & set(cb), key=str):
            out += [_sized("res", f"unresolved.spinGroups[i].channel[MT{role}].", d, ca[role], cb[role])
                    for d in modelDifferences(ca[role], cb[role], path="@",
                                              ignore=IGNORED_FIELDS + ("label",))]
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


#: How far FUDGE moves a repeated or final abscissa when it rewrites a table.
NUDGE = 1e-3


def _snapped(mine, theirs):
    """*theirs* with each value within NUDGE of one of *mine* replaced by it."""
    mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
    nearest = mine[np.abs(theirs[:, None] - mine[None, :]).argmin(axis=1)]
    return np.where(np.abs(nearest - theirs) <= NUDGE + 1e-12 * np.abs(theirs), nearest, theirs)


def _nudged(mine, theirs):
    """*theirs* is *mine* with steps moved by at most NUDGE eV, and maybe closed by a final zero."""
    (xa, ya), (xb, yb) = mine, theirs
    if np.array_equal(xa, xb):
        return False
    b = list(zip(_snapped(xa, xb), yb))
    a = list(zip(xa, ya))
    return b == a or (b[:-1] == a and b[-1] == (a[-1][0], 0.0))


#: The width of the box FUDGE puts in place of an all-zero outgoing spectrum.
NUDGE_BOX = 2e-5


def _zeroAgainstBox(mine, theirs):
    if not (isinstance(mine, XYs1d) and isinstance(theirs, XYs1d)):
        return False
    return (not np.any(np.asarray(mine.ys, float)) and list(theirs.xs) == [0.0, NUDGE_BOX]
            and np.allclose(theirs.ys, [1.0 / NUDGE_BOX, 0.0], rtol=1e-12, atol=0))


def _regionsAreNudgedTable(regions, table):
    mine = [f for region in regions.function2ds for f in region.function1ds]
    theirs = list(table.function1ds)
    outerA = [f.outerDomainValue for f in mine]
    outerB = list(_snapped(outerA, [f.outerDomainValue for f in theirs]))
    if outerA != outerB:
        return False
    return all(not modelDifferences(f, g, ignore=IGNORED_FIELDS + ("outerDomainValue", "index"))
               for f, g in zip(mine, theirs))


def _sized(check, where, message, a, b):
    """One ``modelDifferences`` message as a Difference, with its relative size where it has one."""
    path, _, rest = message.partition(": ")
    relative = path[1:]
    if rest == "arrays differ":
        x, y = (np.asarray(_resolve(o, relative), float) for o in (a, b))
        if relative.endswith((".xs", ".ys")) and _zeroAgainstBox(_resolve(a, relative[:-3]),
                                                                _resolve(b, relative[:-3])):
            return Difference(check, f"{where}{_normalisedPath(path[:-3])}: all zero in kika, "
                                     f"a unit box on [0, {NUDGE_BOX:g}] eV in NNDC")
        table = _resolve(a, relative[:-3]) if relative.endswith(".ys") else None
        if (isinstance(table, XYs1d) and table.interpolation == Interpolation.flat
                and np.array_equal(x[:-1], y[:-1])):
            return Difference(check, f"{where}{_normalisedPath(path)}: only the last value of a "
                                     f"histogram, which no interval uses")
        return Difference(check, f"{where}{_normalised(message)}",
                          float(np.max(np.abs(x - y)) / (np.max(np.abs(y)) or 1.0)))
    if rest.startswith("shape") and relative.endswith((".xs", ".ys")):
        pa, pb = (_tabulated(_resolve(o, relative[:-3])) for o in (a, b))
        if pa is not None and pb is not None:
            if _nudged(pa, pb):
                return Difference(check, f"{where}{_normalisedPath(path[:-3])}: NNDC moved a step "
                                         f"or end point by at most {NUDGE} eV")
            grid = np.union1d(pa[0], pb[0])
            size = _relative(np.interp(grid, *pa), np.interp(grid, *pb))
            return Difference(check, f"{where}{_normalised(message)}", size)
    if relative.endswith("Q.domainMin"):
        x, y = _resolve(a, relative), _resolve(b, relative)
        if x == 1e-5 and y > x:
            return Difference(check, f"{where}{_normalisedPath(path)}: kika's 1e-05 below NNDC's")
    if rest == "Regions2d != XYs2d" and _regionsAreNudgedTable(_resolve(a, relative),
                                                               _resolve(b, relative)):
        return Difference(check, f"{where}{_normalisedPath(path)}: two regions in kika, one table "
                                 f"with a step moved by at most {NUDGE} eV in NNDC")
    if rest in ("XYs1d != Constant1d", "Constant1d != XYs1d"):
        x, y = _resolve(a, relative), _resolve(b, relative)
        table, constant = (x, y) if isinstance(x, XYs1d) else (y, x)
        if np.all(np.asarray(table.ys, float) == float(constant.constant)):
            return Difference(check, f"{where}{_normalisedPath(path)}: a constant XYs1d against "
                                     f"Constant1d, same value")
    numbers = re.fullmatch(r"(\S+) != (\S+)", rest)
    if numbers and all(re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", v) for v in numbers.groups()):
        x, y = (float(v) for v in numbers.groups())
        return Difference(check, f"{where}{_normalised(message)}", abs(x - y) / (abs(y) or 1.0))
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
    """Every field of every reaction; ``id`` restates the label, summands are compared by MT."""
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
    """(MAT1, MF, MT, slices) of row and column: kika labels reactions MT<n>, FUDGE by process."""
    from kika.endf.model_adapter.covariances import _otherMaterial

    def end(link):
        sl = getattr(link, "slices", None)
        L = tuple(x.domainValue for x in sl.slices) if sl is not None and sl.slices else ()
        return (_otherMaterial(link), link.ENDF_MF, link.ENDF_MT, L)

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


def test_reconstruction_from_kikas_gnds_is_the_tapes(tape, endf, kikaGnds):
    from kika.processing.resonances.grid import ReconstructionConvergenceError
    from kika.processing.resonances.prepare import UnsupportedResonanceError

    try:
        fromTape = _reconstructed(endf, True)
    except (UnsupportedResonanceError, ReconstructionConvergenceError) as refusal:
        pytest.skip(f"the reconstruction refuses the tape itself: {refusal}")
    _assertExplained(tape, reconstructionDifferences(fromTape, _reconstructed(kikaGnds, False)))

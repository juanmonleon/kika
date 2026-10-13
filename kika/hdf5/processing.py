"""An evaluation → the processed suite the OpenMC encoder writes (roadmap H4).

:func:`processSuite` takes a suite decoded from ENDF or GNDS (``eval`` forms)
and builds a **new** suite in the shape :mod:`kika.hdf5.neutron_encode`
serialises, without editing the one it was given:

1. **0 K** (``recon``): the suite's ``recon`` forms, or kika's resonance
   reconstruction (:func:`kika.processing.resonances.reconstruct_suite`).
2. **Each temperature** (``heated_<T>``): kika's free-gas broadening
   (:func:`kika.processing.doppler.broaden_suite`) up to ``thnmax`` -- by
   default where NJOY BROADR stops when OpenMC's ``make_ace`` leaves it at its
   default (:func:`njoyThnmax`: the top of the resolved region) -- and the 0 K
   cross sections above it, as BROADR leaves them. Every reaction
   is then on one grid per temperature, the union of the broadened grid
   below ``thnmax`` and the 0 K one above, each from its threshold index.
3. **Reactions** (roadmap D5, the official exporter's set): every exclusive
   partial; the redundant sums 4, 16, 103-107 -- rebuilt from their levels when
   the evaluation does not give them, as ``from_ace`` does for depletion -- and
   18 when the fission chances are given; gas production 203-207 as
   Σ multiplicity × σ over the reactions that emit p, d, t, ³He, α (what
   GASPR computes); KERMA 301/901 and damage 444 by kika's HEATR
   (:mod:`kika.processing.heating`, H8) and the URR probability tables by
   kika's PURR (:mod:`kika.processing.resonances.probability_tables`, H8). No
   NJOY: every piece of a neutron file is kika's.
4. **Distributions and yields** tabulated with their CDF
   (:mod:`kika.processing.tabulation`); neutrons and photons only, the
   products OpenMC's files carry.
5. **Fission**: the prompt neutron, the delayed families (yield ν_d·p_g),
   the total ν̄, and ``fission_energy_release`` after OpenMC's Sher-Beck step.

The temperatures are named and valued as ACE headers carry them (kT in MeV to
five significant figures), which is what the official libraries' ``kTs`` hold.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .exceptions import HDF5FormatError
from .names import (EVAL_LABEL, FISSION_MTS, K_BOLTZMANN, MONTE_CARLO_LABEL, RECON_LABEL,
                    heatedLabel, temperatureName, urrLabel)

__all__ = ["processSuite", "aceKT", "GAS_MTS", "gasMultiplicities", "DEFAULT_THNMAX", "njoyThnmax"]

#: ``thnmax="njoy"``: BROADR's default upper limit (:func:`njoyThnmax`).
DEFAULT_THNMAX = "njoy"

#: BROADR never broadens above this, eV (``broadr.f90``, ``e6pt5``).
NJOY_THNMAX_CAP = 6.5e6

#: Gas production MT → the light particle it counts (GASPR).
GAS_MTS = {203: "H1", 204: "H2", 205: "H3", 206: "He3", 207: "He4"}

#: Sum MTs rebuilt from their levels when absent (``neutron.py:618-632``).
LEVEL_SUMS = {16: range(875, 892), 103: range(600, 650), 104: range(650, 700),
              105: range(700, 750), 106: range(750, 800), 107: range(800, 850),
              4: range(51, 92)}

#: Never written: the C++ rebuilds them (or reads reaction_001 as elastic).
NEVER = (1, 3, 27, 101)


def _m():
    from kika.nuclear_data import model

    return model


def njoyThnmax(suite, firstThreshold: Optional[float] = None) -> float:
    """The upper energy NJOY BROADR broadens to when ``thnmax`` is left at 0.

    That is what OpenMC's ``make_ace`` leaves it at, so it is what the official
    libraries were broadened to. Since NJOY2012.75 (``broadr.f90:107-128``): the
    top of the resolved resonance region; with only an unresolved region, its
    start; with no resonance parameters, the lesser of 6.5 MeV and the first
    threshold. Never above 6.5 MeV. Above it the 0 K cross sections are used.
    """
    resonances = suite.resonances
    resolved = list(getattr(resonances, "resolved", None) or []) if resonances is not None else []
    unresolved = getattr(resonances, "unresolved", None) if resonances is not None else None
    if resolved:
        limit = max(float(r.domainMax) for r in resolved)
    elif unresolved is not None:
        limit = float(unresolved.domainMin)
    else:
        limit = min(NJOY_THNMAX_CAP, firstThreshold if firstThreshold else NJOY_THNMAX_CAP)
    return min(limit, NJOY_THNMAX_CAP)


def aceKT(temperature: float) -> float:
    """kT in eV as an ACE header states it: MeV, five significant figures."""
    return float(f"{temperature * K_BOLTZMANN * 1e-6:.5E}") * 1.0e6


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _flatten(form) -> Tuple[np.ndarray, np.ndarray]:
    m = _m()
    if isinstance(form, m.XYs1d):
        return form.xs, form.ys
    if isinstance(form, m.Regions1d):
        x, y, _ = form.toEndfRegions()
        return np.asarray(x, float), np.asarray(y, float)
    raise HDF5FormatError(f"a pointwise cross section is needed, got {type(form).__name__}")


def _strictly(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A repeated abscissa (a step) nudged up by one part in 1e12, as ACER moves it.

    OpenMC's grid must be strictly increasing; the value on the right of the
    step is kept at the nudged point.
    """
    x = np.array(x, dtype=float)
    for i in np.flatnonzero(np.diff(x) <= 0):
        x[i + 1] = max(x[i] * (1.0 + 1.0e-12), np.nextafter(x[i], np.inf))
    order = np.argsort(x, kind="stable")
    return x[order], np.asarray(y, float)[order]


def _onGrid(form, grid) -> np.ndarray:
    x, y = _flatten(form)
    x, y = _strictly(x, y)
    return np.interp(grid, x, y, left=0.0, right=0.0)


def _threshold(values: np.ndarray, firstEnergy: float, grid: np.ndarray) -> int:
    """The threshold index: the grid point where the reaction's own data start."""
    index = int(np.searchsorted(grid, firstEnergy, side="left"))
    return max(0, min(index, len(grid) - 1))


@dataclass
class _Leaf:
    reaction: object
    container: str
    mt: int


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------

def _pointwise(form) -> bool:
    m = _m()
    return isinstance(form, (m.XYs1d, m.Regions1d))


def _fromLabel(suite, label):
    """A copy whose evaluated forms are the ``label`` ones where they exist.

    What a perturbation run's realisation looks like to this module: the
    sample's forms sit beside the evaluation under a ``realization`` label
    (:mod:`kika.sampling.model_perturbation`), and processing the sample is
    processing those forms -- cross sections, multiplicities, distributions --
    and the evaluation everywhere the sample left alone.
    """
    work = copy.deepcopy(suite)
    for container in ("reactions", "sums", "fissionComponents", "productions", "orphanProducts",
                      "incompleteReactions"):
        for reaction in getattr(work, container):
            _promote(reaction.crossSection, label)
            channel = reaction.outputChannel
            products = list(channel.products)
            if channel.fissionFragmentData is not None:
                products += [d.product for d in channel.fissionFragmentData.delayedNeutrons
                             if d.product is not None]
            for product in products:
                if product.multiplicity is not None:
                    _promote(product.multiplicity, label)
                if product.distribution is not None:
                    _promote(product.distribution, label)
    for total in work.sums.multiplicitySums:
        if total.multiplicity is not None:
            _promote(total.multiplicity, label)
    return work


def _promote(component, label):
    if label in component.forms:
        form = component.forms.pop(label)
        component.forms[EVAL_LABEL] = form


def _mf3HoldsTheResonances(suite) -> bool:
    """The evaluated cross sections already include the resonance contribution.

    True for ENDF LRP=2 (parameters given for information, MF3 complete) and
    LRP<=0 (no parameters to add), or when the suite has no resonance regions.
    LRP=1 -- the common case -- puts only the background in MF3, pointwise as
    every MF3 is, so the shape of the forms says nothing.
    """
    resonances = suite.resonances
    if resonances is None or (not resonances.resolved and resonances.unresolved is None):
        return True
    fields = getattr(suite.provenance, "headerFields", None) or {}
    lrp = fields.get("lrp")
    return lrp is not None and int(lrp) != 1


def _reconstructed(suite, context, report):
    """The suite with a ``recon`` style, reconstructed if it has none.

    A suite whose cross sections are already pointwise everywhere -- read from
    a tape whose MF3 holds the 0 K reconstruction (LRP=2), as a perturbation
    run's base tape does -- takes them as its ``recon`` forms unchanged.
    """
    if RECON_LABEL in suite.styles.labels:
        return suite
    m = _m()
    from kika.processing.resonances.suite import _entries

    entries = _entries(suite)
    if _mf3HoldsTheResonances(suite) and all(_pointwise(r.crossSection.forms.get(EVAL_LABEL))
                                             for r in entries.values()):
        working = copy.deepcopy(suite)
        for reaction in _entries(working).values():
            reaction.crossSection[RECON_LABEL] = reaction.crossSection[EVAL_LABEL]
        from datetime import date
        working.styles.add(m.CrossSectionReconstructed(label=RECON_LABEL, derivedFrom=EVAL_LABEL,
                                                       date=date.today().isoformat()))
        report.warn("the evaluated cross sections are pointwise (the 0 K reconstruction is in the "
                    "file) and were taken as the 0 K forms")
        return working
    from kika.endf.reconstruction import DEFAULT_MAX_POINTS
    from kika.processing.resonances import (ReconstructionOptions, attach_reconstruction,
                                            reconstruct_suite)

    working = copy.deepcopy(suite)
    # Orphan products (MF13 photons on a sum, MT4/103-107) carry production
    # cross sections with no resonance contribution, under the sum's MT; the
    # reconstructor takes the shared MT as ambiguous. They sit out.
    orphans = working.orphanProducts
    working.orphanProducts = type(orphans)()
    # The point budget kika.endf.reconstruct_endf reconstructs with (U-238
    # needs more than the engine's default).
    result = reconstruct_suite(working, context,
                               options=ReconstructionOptions(max_points=DEFAULT_MAX_POINTS))
    attach_reconstruction(working, result)
    working.orphanProducts = orphans
    report.warn("the 0 K cross sections were reconstructed by kika from the resonance "
                "parameters (kika.processing.resonances)")
    return working


#: The tolerance a 0 K form under another law is re-expressed lin-lin to:
#: ten times inside the thinning tolerance, so the two errors do not add up
#: to more than the official files' 0.1 %.
LINEARISATION_TOLERANCE = 1.0e-4


def _linearised(suite, report):
    """Every ``recon`` form re-expressed lin-lin (OpenMC interpolates lin-lin only).

    Outside the resonance region a reconstruction keeps MF3 as the evaluation
    states it, and MF3 may use any of ENDF's laws; RECONR linearises them to its
    tolerance, and so does this, with :func:`kika.algebra.to_linlin`.
    """
    m = _m()
    from kika.algebra import interval_laws, to_linlin
    from kika.processing.resonances.suite import _entries

    changed = []
    for reaction in _entries(suite).values():
        form = reaction.crossSection.forms.get(RECON_LABEL)
        if form is None:
            continue
        regions = form.function1ds if isinstance(form, m.Regions1d) else [form]
        if all(r.interpolation == m.Interpolation.linlin for r in regions):
            continue
        whole = form if isinstance(form, m.Regions1d) else m.Regions1d(function1ds=[form])
        x, y, pairs = whole.toEndfRegions()
        lx, ly = to_linlin(x, y, interval_laws(len(x), pairs), tol=LINEARISATION_TOLERANCE)
        reaction.crossSection[RECON_LABEL] = _pieces(lx, ly)
        changed.append(reaction.ENDF_MT)
    if changed:
        report.approximated(f"0 K cross sections of MT{changed} re-expressed lin-lin to "
                            f"{LINEARISATION_TOLERANCE:g} (RECONR does the same to its tolerance)")


def _pieces(x, y):
    """A lin-lin table as one XYs1d, or a regions1d cut at each step (a repeated abscissa)."""
    m = _m()
    from kika.algebra import LINLIN, split_at_discontinuities

    pieces = split_at_discontinuities(np.asarray(x, float), np.asarray(y, float),
                                      np.full(max(len(x) - 1, 0), LINLIN))
    forms = [m.XYs1d(xs=px, ys=py) for px, py, _ in pieces]
    if len(forms) == 1:
        forms[0].label, forms[0].axes = RECON_LABEL, m.crossSectionAxes()
        return forms[0]
    return m.Regions1d(function1ds=forms, axes=m.crossSectionAxes(), label=RECON_LABEL)


def _dopplerSource(form, lo, top):
    """``form`` on exactly ``[lo, top]``, zero-padded below its start, cut at steps.

    ``broaden_suite`` asks every leaf to cover one common domain; only the
    region below the broadening ceiling matters, so each leaf is cut at the
    shortest leaf's end (above the ceiling) and, if it starts late, padded with
    zeros from ``lo`` -- what it is there.
    """
    x, y = _flatten(form)
    keep = x <= top
    x, y = x[keep], y[keep]
    if x[-1] < top:
        x, y = np.append(x, top), np.append(y, np.interp(top, *_flatten(form)))
    if x[0] > lo:
        if y[0] != 0.0:
            x, y = np.concatenate([[lo, x[0]], x]), np.concatenate([[0.0, 0.0], y])
        else:
            x, y = np.concatenate([[lo], x]), np.concatenate([[0.0], y])
    return _pieces(x, y)


CHUNK_KNOTS = 4000
"""Source knots per energy chunk of the resonant group (see ``_broadenPlan``)."""


def _broadenPlan(suite, thnmax):
    """What ``_broadened`` broadens, independent of the temperature.

    ``(entries, leaves, lo, hi, tasks)``; each task is ``(keys, domain)``, a
    group of leaves broadened together on the output domain ``domain``.
    """
    from kika.processing.resonances.suite import _entries, _graph

    # Orphan photon products (MF13 on a sum) carry no 0 K form: nothing to broaden.
    entries = {k: r for k, r in _entries(suite).items() if RECON_LABEL in r.crossSection.forms}
    graph, _ = _graph(entries, RECON_LABEL)
    leaves = [k for k in entries if k not in graph]
    lows = [_flatten(entries[k].crossSection[RECON_LABEL])[0][0] for k in leaves]
    lo = float(min(lows))
    domainHi = float(max(_flatten(entries[k].crossSection[RECON_LABEL])[0][-1] for k in leaves))
    hi = min(float(thnmax), domainHi)
    # Only the leaves that are not zero below the ceiling are broadened: a
    # threshold reaction above it is zero over the whole broadened domain, and
    # above the ceiling every reaction keeps its 0 K values, as BROADR does.
    # The redundant sums are left out too -- the writer rebuilds the ones it
    # writes from the partials -- which spares broaden_suite's check of each
    # source sum, a point-by-point Python loop that dominated the cost.
    active, counts, knots = [], {}, {}
    for key in leaves:
        x, y = _flatten(entries[key].crossSection[RECON_LABEL])
        if np.any(y[x < hi] != 0):
            active.append(key)
            counts[key] = int(np.count_nonzero(x < hi))
            knots[key] = x[x < hi]
    # Broadened in two groups, each on its own adaptive grid. broaden_suite
    # puts all the curves it is given on one output grid; a smooth MF3
    # reaction given with the resonant ones is evaluated, probes and all, at
    # every point the resonances need (Fe-56: 73 smooth reactions on 126 781
    # points). Apart, it converges on the few points it needs, and the writer
    # puts every reaction on the union grid afterwards.
    tasks = []
    if active:
        dense = max(counts.values())
        resonant = [k for k in active if counts[k] >= 0.2 * dense]
        smooth = [k for k in active if counts[k] < 0.2 * dense]
        # The resonant group is cut into energy chunks of CHUNK_KNOTS source
        # knots, broadened independently (in parallel by _broadenAll). Each
        # chunk sees the whole 0 K source, so its thermal neighbourhood is
        # complete; only the output domain is the chunk's. The refinement
        # accepts every interval on its own verification, so a chunked table
        # meets the same tolerance, and the shared boundary is one exact
        # evaluation of the same point in both chunks. The cuts depend on the
        # data alone, never on the number of workers, so the result does not.
        grid = np.unique(np.concatenate([knots[k] for k in resonant]))
        grid = grid[(grid > lo) & (grid < hi)]
        pieces = max(1, len(grid) // CHUNK_KNOTS)
        cuts = [lo]
        for i in range(1, pieces):
            j = i * len(grid) // pieces
            # Between two distinct knots, never on a discontinuity.
            cuts.append(float(np.sqrt(grid[j - 1] * grid[j])))
        cuts.append(hi)
        tasks += [(resonant, (a, b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        if smooth:
            tasks.append((smooth, (lo, hi)))
    return entries, leaves, lo, hi, tasks


def _joined(parts):
    """One heated form from chunk forms in increasing energy that share their ends."""
    xs, ys = [], []
    for form in parts:
        x, y = _flatten(form)
        if xs and x[0] == xs[-1][-1]:
            x, y = x[1:], y[1:]
        xs.append(x)
        ys.append(y)
    return _pieces(np.concatenate(xs), np.concatenate(ys))


def _assembled(leaves, hi, tasks, results):
    """``({label: heated form}, hi)`` from the results of ``_broadenPlan``'s tasks."""
    byLabel = {}
    for (_, domain), result in zip(tasks, results):
        for label, form in result.items():
            byLabel.setdefault(label, []).append((domain[0], form))
    out = {label: parts[0][1] if len(parts) == 1
           else _joined([form for _, form in sorted(parts, key=lambda p: p[0])])
           for label, parts in byLabel.items()}
    zero = None
    for key in leaves:
        if key.label not in out:
            if zero is None:
                grid = _flatten(next(iter(out.values())))[0]
                zero = _m().XYs1d(xs=grid, ys=np.zeros_like(grid))
            out[key.label] = zero
    return out, hi


def _broadened(suite, temperature, thnmax, options):
    """``{reaction label: heated form below thnmax}`` from kika's free-gas broadening, in-process."""
    return _broadenAll(suite, [temperature], thnmax, options, 1)[temperature]


THERMAL_REACH = 27.0
"""Doppler widths of 0 K source a chunk keeps on each side of its output domain."""


def _window(form, a, b):
    """``form`` restricted to ``a``..``b`` (end values read lin-lin)."""
    x, y = _flatten(form)
    keep = (x > a) & (x < b)
    return _pieces(np.concatenate([[a], x[keep], [b]]),
                   np.concatenate([[np.interp(a, x, y)], y[keep], [np.interp(b, x, y)]]))


def _reducedSuite(suite, entries, keys, lo, window=None):
    """The suite broaden_suite is given for the leaves ``keys``: those leaves alone,
    their 0 K forms cut and padded by ``_dopplerSource``, no sums.

    ``window`` (eV pair) restricts the 0 K forms further, for one energy chunk:
    the kernel never integrates beyond 26 Doppler widths (``max_radius``) and
    the Gaussian is exactly 0 in double beyond 27 (e^-729), so a source cut
    THERMAL_REACH widths around the chunk gives the chunk's points the same
    integrals, and spares broaden_suite the preparation and fingerprint of
    the whole tape for every chunk (U-235: 127 chunks of a 500 000-point source).
    The tail envelope is then the window's max |sigma|, a bound of everything
    the kernel weighs.
    """
    top = min(float(_flatten(entries[k].crossSection[RECON_LABEL])[0][-1]) for k in keys)
    def sourced(reaction):
        clone = copy.copy(reaction)
        clone.crossSection = copy.copy(reaction.crossSection)
        form = _dopplerSource(reaction.crossSection[RECON_LABEL], lo, top)
        if window is not None:
            form = _window(form, *window)
        clone.crossSection.forms = {RECON_LABEL: form}
        return clone
    reduced = copy.copy(suite)
    reduced.reactions = type(suite.reactions)(
        [sourced(entries[k]) for k in keys if k.container == "reactions"])
    reduced.fissionComponents = type(suite.fissionComponents)(
        [sourced(entries[k]) for k in keys if k.container == "fissionComponents"])
    # The evaluated domain declared to the broadening is the one it is given.
    reduced.styles = copy.deepcopy(suite.styles)
    evaluated = reduced.styles.evaluatedFor(RECON_LABEL)
    if evaluated.projectileEnergyDomain is not None:
        evaluated.projectileEnergyDomain = _m().RangeQuantity(*(window or (lo, top)), "eV")
    if suite.resonances is not None:
        # broaden_suite reads the resonances only to ask whether there is an
        # unresolved range (whose 0 K forms must be declared dilute means);
        # the resolved parameters are not sent with every chunk.
        reduced.resonances = copy.copy(suite.resonances)
        reduced.resonances.resolved = []
    for name in ("sums", "productions", "orphanProducts", "incompleteReactions"):
        setattr(reduced, name, type(getattr(suite, name))())
    reduced.sums.multiplicitySums = suite.sums.multiplicitySums
    return reduced


def _broadenReduced(reduced, lo, hi, temperature, options, domain=None):
    """``{label: heated form}`` of every leaf of ``reduced`` (from ``_reducedSuite``).

    The 0 K source spans ``lo``..``hi``; the heated table covers ``domain``
    (default all of it).
    """
    from kika.processing.doppler import DopplerOptions, SupportPolicy, broaden_suite
    from kika.processing.resonances.suite import _entries

    support = {k: SupportPolicy("one_over_v", "constant") for k in _entries(reduced)}
    opts = options or DopplerOptions(max_points=400_000_000, max_work_bytes=4_000_000_000)
    extra = {}
    if reduced.resonances is not None and reduced.resonances.unresolved is not None:
        # The 0 K forms in the unresolved range are the infinitely dilute
        # average reconstruction (kika's or RECONR's), which is the statement
        # broaden_suite asks to be made explicitly before it accepts a URR.
        from kika.processing.doppler import DopplerInputAccuracy

        extra = dict(urr_policy="dilute_mean",
                     input_accuracy=DopplerInputAccuracy(infinite_dilution=True))
    result = broaden_suite(reduced, None, source_style=RECON_LABEL, label=heatedLabel("tmp"),
                           temperature=float(temperature), support=support,
                           source_temperature=0.0, options=opts, output_domain=domain or (lo, hi),
                           negative_policy="preserve_signed", **extra)
    return {key.label: form for key, form in result.forms.items()}


def _broadenTask(args):
    """One (temperature, group, chunk) of ``_broadenAll``, in a worker process."""
    reduced, lo, hi, temperature, options, domain = args
    return _broadenReduced(reduced, lo, hi, temperature, options, domain)


def _broadenAll(suite, temperatures, thnmax, options, workers, progress=None):
    """``{T: (heated forms, ceiling)}``; every temperature and energy chunk in parallel processes.

    ``progress(index, fraction)`` is called as the chunks of ``temperatures[index]``
    finish (``kika.endf.process_endf`` reports it); the result does not depend on it.
    """
    from kika.processing.doppler.suite import _mass

    entries, leaves, lo, hi, tasks = _broadenPlan(suite, thnmax)
    awr = _mass(suite, None).atomic_weight_ratio
    top = {id(keys): min(float(_flatten(entries[k].crossSection[RECON_LABEL])[0][-1]) for k in keys)
           for keys, _ in tasks}
    jobs = []
    for t in temperatures:
        h = math.sqrt(K_BOLTZMANN * t / awr)
        for keys, domain in tasks:
            window = None
            if domain != (lo, hi):
                a, b = domain
                window = (max(lo, (max(0.0, math.sqrt(a) - THERMAL_REACH * h)) ** 2),
                          min(top[id(keys)], (math.sqrt(b) + THERMAL_REACH * h) ** 2))
                if window == (lo, top[id(keys)]):
                    window = None
            jobs.append((_reducedSuite(suite, entries, keys, lo, window), lo, hi, t, options, domain))
    workers = min(len(jobs), os.cpu_count() or 1, 8) if workers is None else max(1, int(workers))
    n = len(tasks)
    done = [0] * len(temperatures)

    def finished(i):
        if progress is not None:
            done[i // n] += 1
            progress(i // n, done[i // n] / n)

    results = [None] * len(jobs)
    if workers == 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            results[i] = _broadenTask(job)
            finished(i)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_broadenTask, job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                finished(futures[future])
    return {t: _assembled(leaves, hi, tasks, results[i * n:(i + 1) * n])
            for i, t in enumerate(temperatures)}


def processSuite(suite, temperatures=None, *, report=None, thnmax=DEFAULT_THNMAX,
                 endf=None, omit: Sequence[str] = (), awr: Optional[float] = None,
                 context=None, doppler=None, tabulation=None, ed: Optional[float] = None,
                 thinning: Optional[float] = 1.0e-3,
                 workers: Optional[int] = None, label: Optional[str] = None,
                 urrTables: Optional[Dict[float, object]] = None,
                 broadened: Optional[Dict[float, tuple]] = None):
    """The processed suite of ``suite`` at ``temperatures`` (kelvin; default 293.6 K).

    ``endf`` is the ENDF tape the suite was read from, for the target's mass
    ratio and spin as MF1/MF2 state them (the reconstruction's context).
    KERMA, damage and the URR probability tables are kika's (H8); ``omit``
    leaves any of ``"urr"``, ``"kerma"``, ``"damage"`` out. ``ed`` is the
    displacement threshold of the damage (default: HEATR's table by Z).
    ``awr`` overrides the
    atomic weight ratio (D11; default MF1/451's, as ACER takes it).
    ``thinning`` is the relative tolerance to which each temperature's grid is
    thinned (roadmap H5): points are dropped while every reaction, read
    lin-lin, reproduces every dropped point to it. 1e-3 is what the official
    libraries are processed to (BROADR's ``errthn``); ``None`` keeps the full
    union grid. The 0 K elastic is thinned to the same tolerance (RECONR's
    ``err`` for the official files). ``workers`` broadens that
    many temperatures at once, one process each (default: one per temperature,
    at most four -- the broadening is single-threaded), and computes that many
    URR energies at once (default: up to eight). Processes are spawned on
    Windows, so a script that calls this needs the usual
    ``if __name__ == "__main__":`` guard; ``workers=1`` stays in-process. ``label`` processes
    the forms of that style -- a perturbation run's ``realization`` -- in
    place of the evaluated ones wherever it has them (roadmap H7, D12: the
    sample is multiplied at 0 K and broadened, never the reverse).
    ``urrTables`` passes URR probability tables already computed
    (:func:`urrTablesOf`, ``{kelvin: ProbabilityTables}``), as a perturbation
    run shares the unperturbed evaluation's between its samples (D12).
    ``broadened`` passes the broadening already done on this suite's 0 K forms
    (``_broadenAll``'s ``{kelvin: (heated forms, ceiling)}``, as
    ``kika.endf.process_endf`` keeps it); each temperature's ceiling is its own.
    """
    m = _m()
    from kika.processing.tabulation import TabulationContext, TabulationOptions, tabulateDistribution

    report = report if report is not None else m.ConversionReport()
    temperatures = [293.6] if temperatures is None else [float(t) for t in temperatures]
    omit = tuple(omit)
    unknown = set(omit) - {"urr", "kerma", "damage"}
    if unknown:
        raise ValueError(f"omit names {sorted(unknown)}; the pieces that can be left out are "
                         f"urr, kerma and damage")
    tabOptions = tabulation or TabulationOptions()

    target = suite.PoPs[suite.target] if suite.target in suite.PoPs else None
    if target is None or getattr(target, "Z", None) is None:
        raise HDF5FormatError(f"the target {suite.target!r} has no Z/A in PoPs")
    Z, A = int(target.Z), int(target.A)
    metastable = int(getattr(suite.provenance, "headerFields", {}).get("lis0", 0) or 0) \
        if suite.provenance is not None else 0
    if awr is None:
        awr = getattr(suite.provenance, "awr", None)
    if awr is None:
        from kika._constants import NEUTRON_MASS_AMU
        awr = float(target.mass.value) / NEUTRON_MASS_AMU
    awr = float(awr)

    if context is None and endf is not None and RECON_LABEL not in suite.styles.labels:
        # The target's mass ratio and spin as the evaluation declares them,
        # which is what kika.endf.reconstruct_endf reconstructs with.
        from kika.endf.read_endf import read_endf
        from kika.endf.reconstruction import context_from_endf

        context, _ = context_from_endf(read_endf(str(endf), mf_numbers=[1, 2]), suite)
    if label is not None and label != EVAL_LABEL:
        suite = _fromLabel(suite, label)
        report.warn(f"processed the {label!r} forms where they exist, the evaluation elsewhere")
    work = _reconstructed(suite, context, report)
    _linearised(work, report)

    # --- which reactions --------------------------------------------------
    leaves: List[_Leaf] = []
    for container in ("reactions", "fissionComponents"):
        for reaction in getattr(work, container):
            mt = reaction.ENDF_MT
            if mt is None or mt in NEVER:
                continue
            leaves.append(_Leaf(reaction, container, mt))
    present = {leaf.mt for leaf in leaves}
    chances = present & {19, 20, 21, 38}

    # --- per temperature cross sections ------------------------------------
    styleKT = {}
    heated: Dict[str, Dict[int, Tuple[np.ndarray, int]]] = {}
    grids: Dict[str, np.ndarray] = {}
    if thnmax == "njoy":
        # The thresholds MF3 states (the 0 K forms may start lower, with zeros).
        thresholds = [_reactionRange(leaf.reaction)[0] for leaf in leaves
                      if (leaf.reaction.outputChannel.Q.value or 0.0) < 0]
        thnmax = njoyThnmax(work, min(thresholds) if thresholds else None)
    if broadened is None:
        broadened = _broadenAll(work, temperatures, float(thnmax), doppler, workers)
    else:
        missing = [t for t in temperatures if t not in broadened]
        if missing:
            raise ValueError(f"broadened has no forms at {missing} K")
    for temperature in temperatures:
        kT = aceKT(temperature)
        T = temperatureName(kT)
        styleKT[T] = kT
        below, hi = broadened[temperature]
        reconForms = {leaf.reaction.label: leaf.reaction.crossSection[RECON_LABEL] for leaf in leaves}
        # Each form's abscissae with a step's repeated energy nudged up first, as
        # ACER does (D36): a bare np.unique collapses the two faces of a step
        # into one knot, and the jump becomes a ramp to the next point -- Au-197
        # B-VIII.1's elastic at the top of the URR, 100 keV, ramped over 250 eV.
        hotGrid = np.unique(np.concatenate([_strictly(*_flatten(below[label]))[0]
                                            for label in reconForms]))
        coldGrid = np.unique(np.concatenate([_strictly(*_flatten(f))[0]
                                             for f in reconForms.values()]))
        grid = np.unique(np.concatenate([hotGrid, coldGrid[coldGrid > hi]]))
        grid, _ = _strictly(grid, grid)
        grids[T] = grid
        values = {}
        for leaf in leaves:
            label = leaf.reaction.label
            hot = _onGrid(below[label], grid)
            cold = _onGrid(reconForms[label], grid)
            y = np.where(grid <= hi, hot, cold)
            if np.any(y < 0):
                # OpenMC clips a negative cross section to zero when it reads
                # one (reaction.py:1102-1106); the file carries it clipped.
                report.approximated(f"{T} MT{leaf.mt}: negative cross section down to "
                                    f"{float(y.min()):.3e} b set to 0, as OpenMC does on read")
                y = np.maximum(y, 0.0)
            first = _flatten(reconForms[label])[0]
            nonzero = np.flatnonzero(np.asarray(_flatten(reconForms[label])[1]) != 0)
            start = first[max(int(nonzero[0]) - 1, 0)] if nonzero.size else first[0]
            values[leaf.mt] = (y, _threshold(y, start, grid))
        if thinning:
            grid, values = _thin(grid, values, float(thinning))
            grids[T] = grid
        heated[T] = values
        report.warn(f"{T}: broadened by kika below {hi:g} eV, 0 K above (NJOY BROADR's thnmax)")

    # --- the processed suite -----------------------------------------------
    out = m.ReactionSuite(evaluation=suite.evaluation or "", projectile="n", target=suite.target)
    out.PoPs = copy.deepcopy(work.PoPs)
    out.provenance = m.Hdf5Provenance(attributes={"Z": Z, "A": A, "metastable": metastable,
                                                  "atomic_weight_ratio": awr})
    out.styles.add(m.Evaluated(label=EVAL_LABEL))
    for T, kT in styleKT.items():
        out.styles.add(m.Heated(label=heatedLabel(T), derivedFrom=EVAL_LABEL,
                                temperature=m.PhysicalQuantity(value=kT, unit="eV/k")))
    out.styles.add(m.CrossSectionReconstructed(label=RECON_LABEL, derivedFrom=EVAL_LABEL))
    out.styles.add(m.MonteCarlo_cdf(label=MONTE_CARLO_LABEL, derivedFrom=EVAL_LABEL))

    def heatedForms(mt):
        cs = m.CrossSection()
        for T, values in heated.items():
            y, start = values[mt]
            cs[heatedLabel(T)] = m.XYs1d(xs=grids[T][start:], ys=y[start:],
                                         axes=m.crossSectionAxes(), label=heatedLabel(T))
        return cs

    def summedForms(components):
        cs = m.CrossSection()
        for T, values in heated.items():
            parts = [values[c] for c in components if c in values]
            y = np.sum([p[0] for p in parts], axis=0)
            start = min(p[1] for p in parts)
            cs[heatedLabel(T)] = m.XYs1d(xs=grids[T][start:], ys=y[start:],
                                         axes=m.crossSectionAxes(), label=heatedLabel(T))
        return cs

    def channelOf(reaction):
        channel = m.OutputChannel(Q=m.Q(value=reaction.outputChannel.Q.value, unit="eV"),
                                  genre=reaction.outputChannel.genre)
        return channel

    zaTarget = 1000 * Z + A
    # Fission given as chances under a summed MT18: when MF3 gives the chances
    # (19, 20, 21, 38) and MT18 only as their sum, the fission products (nu,
    # prompt and delayed neutrons, MF13 photons) are stated on MT18. The
    # official exporter writes MT18 as the fission reaction, not redundant,
    # with σ the sum of the chances and those products, and leaves the chances
    # out (Pu-240); written as chances, the file had no fission neutrons.
    summedFission = next((r for r in work.sums if r.ENDF_MT == 18), None) \
        if chances and 18 not in present else None
    for leaf in leaves:
        reaction = leaf.reaction
        mt = leaf.mt
        if summedFission is not None and mt in chances:
            continue
        cs = heatedForms(mt)
        if mt == 2:
            x, y = _strictly(*_flatten(reaction.crossSection[RECON_LABEL]))
            if thinning:
                # The 0 K elastic (resonance scattering's source) thinned to the
                # tolerance RECONR is run at for the official files (err 0.001):
                # the same accuracy class, at RECONR's density rather than at
                # kika's finer reconstruction tolerance.
                from kika.algebra import thin_common_grid

                kept = thin_common_grid(x, y[None, :], rtol=float(thinning),
                                        atol=1.0e-10 * float(np.max(np.abs(y))))
                x, y = x[kept], y[kept]
            cs[RECON_LABEL] = m.XYs1d(xs=x, ys=y, axes=m.crossSectionAxes(), label=RECON_LABEL)
        channel = channelOf(reaction)
        _products(reaction, channel, mt, awr, zaTarget, tabOptions, report, work.PoPs)
        redundant = 1 if (mt == 18 and chances) else 0
        rid = m.ReactionId(label=reaction.label, ENDF_MT=mt, fissionGenre=reaction.id.fissionGenre)
        new = (m.CrossSectionSum if redundant else m.Reaction)(
            id=rid, crossSection=cs, outputChannel=channel,
            provenance=m.Hdf5Provenance(attributes={"redundant": redundant}))
        (out.sums if redundant else (out.fissionComponents if mt in chances else out.reactions)).append(new)
    if summedFission is not None:
        channel = channelOf(summedFission)
        _products(summedFission, channel, 18, awr, zaTarget, tabOptions, report, work.PoPs)
        out.reactions.append(m.Reaction(
            id=m.ReactionId(label=summedFission.label, ENDF_MT=18),
            crossSection=summedForms(sorted(chances)), outputChannel=channel,
            provenance=m.Hdf5Provenance(attributes={"redundant": 0})))
        report.warn(f"fission written as MT18, the sum of the chances {sorted(chances)} that "
                    f"carries the fission products, as the official exporter writes it; the "
                    f"chances themselves are not written")

    # --- level sums 4, 16, 103-107 ----------------------------------------
    for mt, levels in LEVEL_SUMS.items():
        if mt == 4:
            # Written only when the URR tables name it (below); the official
            # exporter keeps MT4 for that and for photons it carries.
            continue
        components = [c for c in levels if c in present]
        if not components or mt in present:
            if mt in present and components:
                # The evaluation gives both the sum and its levels: the levels
                # are the partials, the sum is written redundant.
                _demote(out, mt)
            continue
        reaction = m.CrossSectionSum(id=m.ReactionId(label=f"MT{mt}", ENDF_MT=mt),
                                     crossSection=summedForms(components),
                                     outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                                     provenance=m.Hdf5Provenance(attributes={"redundant": 1}))
        out.sums.append(reaction)

    # --- photons stated on a sum (MF13 on MT4, 103-107: orphan products) -----
    # The official exporter writes them as the products of that sum, written
    # redundant (O-16: reaction_004 with 15 photons, reaction_103 with 4).
    for orphan in work.orphanProducts:
        mt = orphan.ENDF_MT
        target = out.findReactionByENDF_MT(mt)
        if target is None:
            # MT3, the nonelastic, is every partial but the elastic (Pu-240 and
            # Pu-241 state photons on it; the official file writes it redundant).
            levels = ([c for c in sorted(present) if c != 2] if mt == 3 else
                      [c for c in LEVEL_SUMS.get(mt, ()) if c in present])
            if not levels:
                report.lost(f"MT{mt}: {len(orphan.outputChannel.products)} photons stated on a "
                            f"sum the evaluation gives no partials of; not written")
                continue
            target = m.CrossSectionSum(id=m.ReactionId(label=f"MT{mt}", ENDF_MT=mt),
                                       crossSection=summedForms(levels),
                                       outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                                       provenance=m.Hdf5Provenance(attributes=_sumAttributes(mt, _hasURR(work))))
            out.sums.append(target)
        # The photons' range is the sum's, as the evaluation states it.
        source = work.findReactionByENDF_MT(mt)
        proxy = copy.copy(orphan)
        if source is not None:
            proxy.crossSection = source.crossSection
        _products(proxy, target.outputChannel, mt, awr, zaTarget, tabOptions, report)

    # --- gas production -----------------------------------------------------
    for gasMT, cs in _gas(work, leaves, heated, grids).items():
        out.productions.append(m.Reaction(id=m.ReactionId(label=f"MT{gasMT}", ENDF_MT=gasMT),
                                          crossSection=cs,
                                          outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                                          provenance=m.Hdf5Provenance(attributes={
                                              "redundant": 1, "center_of_mass": 0})))

    # --- fission ------------------------------------------------------------
    _fission(work, out, report)

    # --- unresolved-resonance probability tables (kika's PURR, H8) --------
    if "urr" in omit or not _hasURR(work):
        urrTables = {}
    elif urrTables is not None:
        missing = [t for t in temperatures if float(t) not in urrTables]
        if missing:
            raise ValueError(f"urrTables has no tables at {missing} K")
        urrTables = {T: urrTables[float(t)] for t, T in zip(temperatures, styleKT)}
        report.warn("URR probability tables passed in, computed on another suite (D12: a "
                    "perturbation run's unperturbed evaluation); they are not perturbed")
    else:
        from kika.processing.resonances.probability_tables import purrGrid, purrInputs, competitionFlags
        energyURR = purrGrid(work.resonances.unresolved)
        _, competingURR = purrInputs(work, energyURR, form=_evaluated)
        inelasticURR, _ = competitionFlags(competingURR)
        transportXS = {float(kelvin): _urrTransportXS(out, T, energyURR, inelasticURR)
                       for kelvin, T in zip(temperatures, styleKT)}
        byKelvin = urrTablesOf(work, temperatures, context=context, workers=workers,
                              report=report, transportXS=transportXS)
        urrTables = {T: byKelvin[float(t)] for t, T in zip(temperatures, styleKT)}
    if urrTables:
        _attachURR(out, urrTables)

    # --- KERMA and damage (kika's HEATR, H8) ---------------------------------
    wanted = [p for p in ("kerma", "damage") if p not in omit]
    if wanted:
        _attachHeating(out, work, temperatures, styleKT, grids, wanted, awr, Z, A, ed, report)
    # The URR tables name the reaction whose smooth cross section stands for
    # the inelastic competition, and the C++ requires it to exist
    # (nuclide.cpp:308-320): MT4 is written for that, as the official exporter
    # writes it, redundant and summed from the levels.
    inelastic = int(next(iter(urrTables.values())).inelastic) if urrTables else 0
    if inelastic > 0 and out.findReactionByENDF_MT(inelastic) is None:
        levels = [c for c in LEVEL_SUMS.get(inelastic, ()) if c in present]
        if not levels:
            raise HDF5FormatError(f"the URR tables name MT{inelastic} and the "
                                  f"evaluation has none of its levels")
        out.sums.append(m.CrossSectionSum(
            id=m.ReactionId(label=f"MT{inelastic}", ENDF_MT=inelastic),
            crossSection=summedForms(levels),
            outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
            provenance=m.Hdf5Provenance(attributes=_sumAttributes(inelastic, True))))
        report.warn(f"MT{inelastic} written as the redundant sum of {len(levels)} levels, "
                    f"because the URR tables name it")
    for piece in omit:
        report.lost(f"{piece} was not written: omitted on request")
    # Photons the ENDF reader kept verbatim (no cross section to hang from:
    # Pu-241's MF13 on MT3, which MF3 does not give) are not in the model, so
    # not in the file either; the official file has them on a redundant MT3.
    from kika.endf.model_adapter.photons import PHOTONS_VERBATIM_KEY

    kept = (getattr(getattr(work, "provenance", None), "headerFields", None) or {}).get(PHOTONS_VERBATIM_KEY) or {}
    for mt in sorted({int(key.split("/")[1]) for key in kept}):
        files = "/".join(sorted(key.split("/")[0] for key in kept if int(key.split("/")[1]) == mt))
        report.lost(f"MT{mt}: the photons of MF{files} were kept verbatim by the ENDF reader "
                    f"(no MF3 cross section to hang from) and are not written")
    out.report = report
    return out


def _sumAttributes(mt, urr=False):
    """A redundant sum's attributes as the official exporter writes them.

    A sum OpenMC builds itself takes ``Reaction``'s default frame, the centre of
    mass: MT3 (Pu-240) and MT4 carrying photons (O-16). An MT4 that comes from
    the ACE table -- ACER writes it for a nuclide with URR tables, which name it
    as the inelastic competition -- has TYR = 0 there, so the lab (U-235,
    U-238, Pu-240). The frame of a redundant reaction is never sampled.
    """
    return {"redundant": 1, "center_of_mass": 1 if mt == 3 or (mt == 4 and not urr) else 0}


def _thin(grid, values, rtol):
    """The grid and every leaf on it, thinned jointly (``kika.algebra.thin_common_grid``)."""
    from kika.algebra import thin_common_grid

    mts = list(values)
    rows = np.vstack([values[mt][0] for mt in mts])
    # A floor per reaction far below anything transport sees, so a row that is
    # zero over a stretch does not force every point of it to stay.
    floors = 1.0e-10 * np.max(np.abs(rows), axis=1)
    keep = sorted({values[mt][1] for mt in mts})
    kept = thin_common_grid(grid, rows, rtol=rtol, atol=floors, keep=keep)
    thinned = {mt: (values[mt][0][kept], int(np.searchsorted(kept, values[mt][1]))) for mt in mts}
    return grid[kept], thinned


def _demote(out, mt):
    """Move a reaction the levels already partition into ``sums``, as redundant."""
    m = _m()
    for container in (out.reactions,):
        for reaction in list(container):
            if reaction.ENDF_MT == mt:
                container.reactions.remove(reaction)
                summed = m.CrossSectionSum(id=reaction.id, crossSection=reaction.crossSection,
                                           outputChannel=reaction.outputChannel,
                                           provenance=m.Hdf5Provenance(attributes={"redundant": 1}))
                out.sums.append(summed)


def _urrContext(suite, context):
    """The NeutronContext of the URR: the given one, or the model's mass and spin."""
    if context is not None:
        return context
    from kika.processing.resonances.context import NeutronContext

    target, neutron = suite.PoPs[suite.target], suite.PoPs[suite.projectile]
    return NeutronContext(target.mass.convertedTo("amu").value / neutron.mass.convertedTo("amu").value,
                          target.spin.convertedTo("hbar").value)


def urrTablesOf(suite, temperatures, *, context=None, workers=None, report=None, options=None, transportXS=None):
    """``{kelvin: ProbabilityTables}`` from the URR parameters by kika's PURR (roadmap H8).

    The tables :func:`processSuite` writes. ``context`` is the target's
    NeutronContext (default: the model's mass and spin); ``options`` a
    ``ProbabilityTableOptions`` (default: PURR's 20 bands, 64 ladders).
    """
    from kika.processing.resonances.probability_tables import (ProbabilityTableOptions,
                                                               probabilityTables, purrGrid,
                                                               purrInputs)

    temperatures = [float(t) for t in temperatures]
    region = suite.resonances.unresolved
    energies = purrGrid(region)
    background, competing = purrInputs(suite, energies, form=_evaluated)
    options = options or ProbabilityTableOptions()
    tables = probabilityTables(region, _urrContext(suite, context), energies, temperatures,
                               background=background, competing=competing, options=options,
                               workers=workers or min(len(energies), os.cpu_count() or 1, 8), transportXS=transportXS)
    if report is not None:
        report.warn(f"URR probability tables computed by kika (PURR's method: {options.ladders} "
                    f"ladders of {options.samples} samples, {options.bins} bands, on "
                    f"{len(energies)} energies); statistically, not bitwise, equal to NJOY PURR's")
        for T, table in tables.items():
            report.warn(_urrErrorLine(T, table.statistics))
    return tables


def _urrErrorLine(T, statistics) -> str:
    """One line on a temperature's statistical error: per reaction with an
    average, the worst relative error of f(σ0) at the smallest σ0 and the
    worst band share ``p · factor``, over the energies."""
    worst = statistics.worst()
    s0 = float(statistics.sigma0[0])
    parts = [f"{row} f({s0:g} b) {100 * w['shielding'][s0]:.2f} %, band {100 * w['contribution']:.2f} %"
             for row, w in worst.items() if w["contribution"] > 0]
    ladders = statistics.ladders
    drawn = (f"{int(ladders.min())}" if ladders.min() == ladders.max()
             else f"{int(ladders.min())}-{int(ladders.max())}")
    target = ""
    if statistics.errorTarget is not None:
        missed = int((~statistics.converged).sum())
        target = (f"; target {100 * statistics.errorTarget:g} % "
                  + ("reached at every energy" if not missed else f"NOT reached at {missed} energies"))
    return (f"URR {T:g} K statistical error (1 sigma, jackknife over {drawn} ladders, {statistics.shieldingBasis}, worst energy): "
            + "; ".join(parts) + target)


def _attachURR(out, tables):
    m = _m()
    for T, table in tables.items():
        ulabel = urrLabel(T)
        if ulabel not in out.styles.labels:
            out.styles.add(m.URR_probabilityTables(label=ulabel, derivedFrom=heatedLabel(T)))
        for mt in (2, 18, 102):
            reaction = out.findReactionByENDF_MT(mt)
            if reaction is not None:
                reaction.crossSection[ulabel] = m.URR_probabilityTables1d(label=ulabel, table=table)


def _hasURR(suite) -> bool:
    return suite.resonances is not None and suite.resonances.unresolved is not None


# ----------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------

def _mf6Record(reaction, product):
    provenance = getattr(reaction, "provenance", None)
    fields = getattr(provenance, "headerFields", None) or {}
    block = fields.get("mf6")
    records = block.get("products", []) if isinstance(block, dict) else (block or [])
    for record in records if isinstance(records, list) else []:
        if isinstance(record, dict) and record.get("label") == product.label:
            return record
    return None


def _zaOf(pid: str) -> int:
    from kika.nuclear_data.model import zaFromPid

    return zaFromPid(pid)


def _evaluated(component):
    """The evaluated function of a component. A GNDS ``resonancesWithBackground``
    is an instruction plus a background; its function is the background, its
    three regions joined in one ``regions1d`` (what an ENDF tape's MF3 states)."""
    if component is None or not component.forms:
        return None
    form = component.forms.get(EVAL_LABEL) or next(iter(component.forms.values()))
    m = _m()
    if isinstance(form, m.ResonancesWithBackground) and form.background is not None:
        background = form.background
        pieces = [v for v in (background.resolvedRegion, background.unresolvedRegion,
                              background.fastRegion) if v is not None]
        children = [c for piece in pieces
                    for c in (piece.function1ds if isinstance(piece, m.Regions1d) else [piece])]
        if children:
            children.sort(key=lambda c: c.domainMin)
            return m.Regions1d(function1ds=children, axes=children[0].axes)
    return form


def _products(reaction, channel, mt, awr, zaTarget, options, report, pops=None):
    """Neutron and photon products with Monte Carlo distributions, onto ``channel``."""
    m = _m()
    from kika.processing.tabulation import TabulationContext, tabulateDistribution

    explicitPhotons = False
    for product in reaction.outputChannel.products:
        if product.pid not in ("n", "photon"):
            continue
        new = _tabulatedProduct(product, reaction, mt, awr, zaTarget, options, report)
        if new is not None:
            channel.products.products.append(new)
            explicitPhotons |= product.pid == "photon"
    if not explicitPhotons:
        for photon in _cascadePhotons(pops, reaction, awr):
            channel.products.products.append(photon)
    data = reaction.outputChannel.fissionFragmentData
    if data is not None and len(data.delayedNeutrons):
        channel.fissionFragmentData = m.FissionFragmentData()
        for family in data.delayedNeutrons:
            product = family.product
            new = _tabulatedProduct(product, reaction, mt, awr, zaTarget, options, report,
                                    delayed=True) if product is not None else None
            if new is None:
                report.lost(f"MT{mt}: delayed family {family.label} has no spectrum or "
                            f"multiplicity and was not written")
                continue
            channel.fissionFragmentData.delayedNeutrons.append(
                m.DelayedNeutron(label=family.label, rate=family.rate, product=new))
        channel.fissionFragmentData.fissionEnergyReleases = list(data.fissionEnergyReleases)


def _isotropicPhotonAngle(lo, hi):
    """The explicit isotropic table OpenMC's files carry for a photon: histogram, two energies."""
    m = _m()
    table = lambda e: m.Xs_pdf_cdf1d(xs=[-1.0, 1.0], pdf=[0.5, 0.5], cdf=[0.0, 1.0],  # noqa: E731
                                     interpolation=m.Interpolation.flat, outerDomainValue=e)
    return m.XYs2d(function1ds=[table(float(lo)), table(float(hi))], label=MONTE_CARLO_LABEL)


def _reactionRange(reaction):
    """``(threshold, top)`` as the evaluation states them: MF3's first and last
    energy, which ACER takes for the reaction's range. The 0 K forms may run
    lower, with zeros, when the reconstruction put them on its union grid."""
    m = _m()
    form = reaction.crossSection.forms.get(EVAL_LABEL)
    if not isinstance(form, (m.XYs1d, m.Regions1d)):
        form = reaction.crossSection[RECON_LABEL]
    x = _flatten(form)[0]
    return float(x[0]), float(x[-1])


def _explicitPhotonAngle(form, reaction):
    """A photon's isotropic angle written as OpenMC's files write it, not left out."""
    m = _m()
    targets = form.weighted if isinstance(form, m.DistributionMixture) else [m.Weighted(functional=form)]
    for term in targets:
        f = term.functional
        if isinstance(f, m.Uncorrelated) and isinstance(f.angular, m.Isotropic2d):
            lo, hi = _reactionRange(reaction)
            f.angular = _isotropicPhotonAngle(lo, hi)


def _cascadePhotons(pops, reaction, awr):
    """The gamma lines of the residual level a reaction leaves, as photon products.

    An evaluation such as ENDF/B-VIII.1 Fe-56 states a level's de-excitation
    once, on the level (MF12 LO=2, in PoPs ``decayData``), not on every
    reaction that populates it; ACER turns the cascade into one discrete line
    per transition with a constant yield over the reaction's range, which is
    what the official files carry (Fe-56 MT52: 1 238 327 eV x 0.999862 and
    846 778 eV x 0.9997041).
    """
    m = _m()
    if pops is None:
        return []
    residuals = [p.pid for p in reaction.outputChannel.products if p.pid not in ("n", "photon")]
    # A discrete-level reaction whose residual came as the ground state (U-234
    # MT52: the recoil the two-body MF4 adapter made, to which the level's
    # decay was then attached) de-excites the level its MT names.
    from kika.endf.model_adapter.residuals import levelSeries

    series = levelSeries(int(reaction.ENDF_MT)) if reaction.ENDF_MT is not None else None
    if series is not None and series[2] > 0:
        for i, pid in enumerate(residuals):
            try:
                za = m.zaFromPid(pid)
            except ValueError:
                continue
            level = m.pidFromZA(za, series[2])
            if level != pid and level in pops and m.pidFromZA(za, 0) == pid:
                residuals[i] = level
    out = []
    lo, hi = _reactionRange(reaction)
    for pid in residuals:
        if pid not in pops:
            continue
        for line in m.cascadeLines(pops, pid):
            if not line.photons > 0:
                # A transition with no photons (all conversion electrons, or a
                # zero branching) is not written, as ACER does not write it.
                continue
            product = m.Product(pid="photon", label="photon")
            product.multiplicity = m.Multiplicity(form=m.XYs1d(xs=[lo, hi], ys=[line.photons] * 2))
            product.distribution = m.Distribution()
            product.distribution[MONTE_CARLO_LABEL] = m.Uncorrelated(
                angular=_isotropicPhotonAngle(lo, hi),
                energy=m.DiscretePhotonEnergy(primaryFlag=0, energy=float(line.energy),
                                              atomicWeightRatio=awr),
                label=MONTE_CARLO_LABEL, productFrame=m.Frame.lab)
            product.provenance = m.Hdf5Provenance(attributes={"emission_mode": "prompt"})
            out.append(product)
    return out


def _tabulatedProduct(product, reaction, mt, awr, zaTarget, options, report, delayed=False):
    m = _m()
    from kika.processing.tabulation import TabulationContext, tabulateDistribution

    form = _evaluated(product.distribution)
    record = _mf6Record(reaction, product)
    discrete = None
    if record is not None:
        discrete = (record.get("law_fields") or {}).get("nd")
    context = TabulationContext(awr=awr, zaTarget=zaTarget, mt=mt,
                                Q=reaction.outputChannel.Q.value, pid=product.pid,
                                zaEmitted=_zaOf(product.pid) if product.pid != "photon" else 0,
                                discrete=discrete, options=options, report=report)
    if form is None or isinstance(form, m.Isotropic2d):
        if mt == 2 or (product.pid == "n" and 51 <= mt <= 90):
            # Isotropic in the centre of mass where the evaluation gives no MF4.
            form = m.AngularTwoBody(angular=m.Isotropic2d(), productFrame=m.Frame.centerOfMass)
        else:
            report.lost(f"MT{mt} {product.label}: no distribution; product not written")
            return None
    try:
        tabulated = tabulateDistribution(form, context)
    except (TypeError, ValueError) as exc:
        report.lost(f"MT{mt} {product.label}: {type(form).__name__} not tabulated ({exc}); "
                    f"product not written")
        return None
    if tabulated is None:
        report.lost(f"MT{mt} {product.label}: a {type(form).__name__} has no Monte Carlo form; "
                    f"product not written")
        return None
    if product.pid == "photon":
        _explicitPhotonAngle(tabulated, reaction)
    new = m.Product(pid=product.pid, label=product.label)
    new.distribution = m.Distribution()
    new.distribution[MONTE_CARLO_LABEL] = tabulated
    multiplicity = product.multiplicity.form if product.multiplicity is not None else None
    if multiplicity is None or not isinstance(multiplicity, m.Function1d):
        if product.pid == "n" and not delayed:
            multiplicity = m.Polynomial1d(coefficients=np.array([1.0]), domainMin_=0.0,
                                          domainMax_=float("inf"))
        else:
            report.lost(f"MT{mt} {product.label}: multiplicity "
                        f"{type(multiplicity).__name__} is not a number; product not written")
            return None
    new.multiplicity = m.Multiplicity(form=multiplicity)
    new.provenance = m.Hdf5Provenance(attributes={"emission_mode": "delayed" if delayed else "prompt"})
    return new


# ----------------------------------------------------------------------
# Gas, fission, NJOY
# ----------------------------------------------------------------------

def gasMultiplicities(reaction) -> Dict[int, object]:
    """``{gas MT: multiplicity function}`` for the light particles a reaction emits."""
    out = {}
    for product in reaction.outputChannel.products:
        for gasMT, pid in GAS_MTS.items():
            if product.pid == pid:
                form = product.multiplicity.form if product.multiplicity is not None else None
                if form is None:
                    # A two-body product (MF4 alone, MT600-849) states no
                    # multiplicity because it is one; GASPR counts it so.
                    form = _m().Constant1d(constant=1.0, domainMin_=0.0, domainMax_=float("inf"))
                elif not isinstance(form, _m().Function1d):
                    continue
                if gasMT in out:
                    out[gasMT] = (out[gasMT], form)
                else:
                    out[gasMT] = form
    return out


def _gas(work, leaves, heated, grids):
    m = _m()
    from kika.hdf5.fission_energy import evaluate

    totals: Dict[int, Dict[str, np.ndarray]] = {}
    starts: Dict[int, Dict[str, int]] = {}
    for leaf in leaves:
        for gasMT, form in gasMultiplicities(leaf.reaction).items():
            forms = form if isinstance(form, tuple) else (form,)
            for T, values in heated.items():
                y, start = values[leaf.mt]
                grid = grids[T]
                k = sum(np.asarray(evaluate(f, grid) if not isinstance(f, m.Constant1d)
                                   else np.full(grid.shape, f.constant)) for f in forms)
                totals.setdefault(gasMT, {}).setdefault(T, np.zeros_like(grid))
                totals[gasMT][T] = totals[gasMT][T] + k * y
                starts.setdefault(gasMT, {})
                starts[gasMT][T] = min(starts[gasMT].get(T, start), start)
    out = {}
    for gasMT in sorted(totals):
        cs = m.CrossSection()
        for T, y in totals[gasMT].items():
            start = starts[gasMT][T]
            cs[heatedLabel(T)] = m.XYs1d(xs=grids[T][start:], ys=y[start:],
                                         axes=m.crossSectionAxes(), label=heatedLabel(T))
        out[gasMT] = cs
    return out


def _fission(work, out, report):
    m = _m()
    from .fission_energy import openmcTerms
    from .neutron_decode import TOTAL_NUBAR_LABEL

    fission = next((r for r in out.reactions if r.ENDF_MT in FISSION_MTS), None) \
        or next((r for r in out.sums if r.ENDF_MT == 18), None)
    if fission is None:
        return
    total = work.sums.multiplicitySums.byENDF_MT(452)
    if total is not None and total.multiplicity is not None and \
            any(p.pid == "n" for p in fission.outputChannel.products):
        summed = m.MultiplicitySum(label=TOTAL_NUBAR_LABEL, multiplicity=total.multiplicity,
                                   ENDF_MT=452)
        prompt = next(p for p in fission.outputChannel.products if p.pid == "n")
        if work.sums.multiplicitySums.byENDF_MT(455) is not None or \
                (fission.outputChannel.fissionFragmentData is not None
                 and len(fission.outputChannel.fissionFragmentData.delayedNeutrons)):
            out.sums.multiplicitySums.append(summed)
        else:
            prompt.provenance.attributes["emission_mode"] = "total"
    data = fission.outputChannel.fissionFragmentData
    if data is not None and data.fissionEnergyReleases:
        neutron = next((p for p in fission.outputChannel.products if p.pid == "n"), None)
        nu = neutron.multiplicity.form if neutron is not None else None
        release = openmcTerms(data.fissionEnergyReleases[0], nu, report)
        data.fissionEnergyReleases = [release]
    elif data is None:
        evaluated = next((r for r in work.reactions if r.ENDF_MT == fission.ENDF_MT), None)
        source = evaluated.outputChannel.fissionFragmentData if evaluated is not None else None
        if source is not None and source.fissionEnergyReleases:
            neutron = next((p for p in fission.outputChannel.products if p.pid == "n"), None)
            nu = neutron.multiplicity.form if neutron is not None else None
            fission.outputChannel.fissionFragmentData = m.FissionFragmentData(
                fissionEnergyReleases=[openmcTerms(source.fissionEnergyReleases[0], nu, report)])



def _urrTransportXS(out, T, grid, inelastic):
    """Smooth XS used by OpenMC's URR reader, on the probability-table grid.

    Capture in the reader denotes all non-fission disappearance, not MT102
    alone. Redundant sums are excluded from absorption and fission.
    """
    values = {}
    result = np.zeros((len(grid), 4))
    for container in (out.reactions, out.fissionComponents, out.sums):
        for reaction in container:
            mt = reaction.ENDF_MT
            form = reaction.crossSection.forms.get(heatedLabel(T))
            if mt is None or form is None:
                continue
            x = np.interp(grid, form.xs, form.ys, left=0., right=float(form.ys[-1]))
            values[mt] = x
            attributes = getattr(reaction.provenance, "attributes", {}) or {}
            if attributes.get("redundant", 0):
                continue
            if mt == 2:
                result[:, 0] += x
            elif mt in FISSION_MTS:
                result[:, 1] += x
            elif (101 <= mt <= 117 or 600 <= mt <= 849
                  or mt in (118, 119, 120, 121, 122, 124)):
                result[:, 2] += x
    if inelastic > 0:
        if inelastic in values:
            result[:, 3] = values[inelastic]
        elif inelastic == 4:
            result[:, 3] = sum((x for mt, x in values.items() if 51 <= mt <= 91),
                               np.zeros(len(grid)))
        else:
            raise ValueError(f"URR competing MT{inelastic} has no processed smooth cross section")
    return result


def _totalOn(out, T, grid):
    """σ_t and σ_f of the processed suite at one temperature, on ``grid``."""
    total = np.zeros_like(grid)
    fission = np.zeros_like(grid)
    for container in (out.reactions, out.fissionComponents):
        for reaction in container:
            form = reaction.crossSection[heatedLabel(T)]
            values = np.zeros_like(grid)
            values[len(grid) - len(form):] = form.ys
            total += values
            if reaction.ENDF_MT in FISSION_MTS:
                fission += values
    return total, fission


def _attachHeating(out, work, temperatures, styleKT, grids, wanted, awr, Z, A, ed, report):
    """MT301, MT901 and MT444 by :func:`kika.processing.heating.heatingOf`, plus fission.

    The fission part comes from MF1/458 as OpenMC's ``from_njoy`` adds it to
    HEATR's ``301 − 318``: ``(EFR + EB)·σ_f`` for 301, ``(EFR + EGP + EGD + EB)·σ_f``
    for 901.
    """
    m = _m()
    from kika.processing.heating import heatingOf
    from .fission_energy import evaluate

    qm = {}
    for container in (work.reactions, work.sums, work.fissionComponents):
        for reaction in container:
            value = getattr(reaction.provenance, "qm", None)
            if reaction.ENDF_MT is not None and value is not None:
                qm[reaction.ENDF_MT] = float(value)

    release = None
    fission = next((r for r in out.reactions if r.ENDF_MT in FISSION_MTS), None)
    if fission is not None and fission.outputChannel.fissionFragmentData is not None:
        releases = fission.outputChannel.fissionFragmentData.fissionEnergyReleases
        release = releases[0] if releases else None
    responses = {301: m.CrossSection(), 901: m.CrossSection(), 444: m.CrossSection()}
    notes = None
    for temperature, (T, kT) in zip(temperatures, styleKT.items()):
        grid = grids[T]
        total, sigmaF = _totalOn(out, T, grid)
        label = heatedLabel(T)
        sigma = {}
        for reaction in out.reactions:
            form = reaction.crossSection.forms.get(label)
            if form is not None and reaction.ENDF_MT is not None:
                values = np.zeros_like(grid)
                values[len(grid) - len(form):] = form.ys
                sigma[reaction.ENDF_MT] = values
        for reaction in out.sums:
            form = reaction.crossSection.forms.get(label)
            if form is not None and reaction.ENDF_MT is not None and reaction.ENDF_MT not in sigma:
                values = np.zeros_like(grid)
                values[len(grid) - len(form):] = form.ys
                sigma[reaction.ENDF_MT] = values
        result = heatingOf(out, grid, sigma, awr=awr, Z=Z, qm=qm, ed=ed, evaluation=work,
                           zaTarget=1000 * Z + A)
        notes = result.notes
        if "kerma" in wanted:
            nonlocal_ = result.kerma
            local = result.kermaLocal
            if release is not None:
                efr = evaluate(release.promptProductKE, grid)
                eb = evaluate(release.delayedBetaEnergy, grid)
                egp = evaluate(release.promptGammaEnergy, grid)
                egd = evaluate(release.delayedGammaEnergy, grid)
                nonlocal_ = nonlocal_ + (efr + eb) * sigmaF
                local = local + (efr + egp + egd + eb) * sigmaF
            responses[301][label] = m.XYs1d(xs=grid, ys=nonlocal_, label=label)
            responses[901][label] = m.XYs1d(xs=grid, ys=local, label=label)
        if "damage" in wanted:
            responses[444][label] = m.XYs1d(xs=grid, ys=result.damage, label=label)
    names = {301: "heating", 901: "heating-local", 444: "damage-energy"}
    for mt, cs in responses.items():
        if len(cs.forms):
            out.productions.append(m.Reaction(
                id=m.ReactionId(label=names[mt], ENDF_MT=mt), crossSection=cs,
                outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                provenance=m.Hdf5Provenance(attributes={
                    "redundant": 1, "center_of_mass": 0 if mt == 444 else 1})))
    for note in notes or ():
        report.warn(note)

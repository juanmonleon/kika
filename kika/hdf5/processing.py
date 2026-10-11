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
   GASPR computes); KERMA 301/901 and damage 444 from NJOY (D7).
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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .exceptions import HDF5FormatError, MissingNJOYPieceError
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
    if all(_pointwise(r.crossSection.forms.get(EVAL_LABEL)) for r in entries.values()):
        working = copy.deepcopy(suite)
        for reaction in _entries(working).values():
            reaction.crossSection[RECON_LABEL] = reaction.crossSection[EVAL_LABEL]
        working.styles.add(m.CrossSectionReconstructed(label=RECON_LABEL, derivedFrom=EVAL_LABEL))
        report.warn("the evaluated cross sections are pointwise (the 0 K reconstruction is in the "
                    "file) and were taken as the 0 K forms")
        return working
    from kika.processing.resonances import attach_reconstruction, reconstruct_suite

    working = copy.deepcopy(suite)
    result = reconstruct_suite(working, context)
    attach_reconstruction(working, result)
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


def _broadened(suite, temperature, thnmax, options):
    """``{reaction label: heated form below thnmax}`` from kika's free-gas broadening."""
    from kika.processing.doppler import DopplerOptions, SupportPolicy, broaden_suite
    from kika.processing.resonances.suite import _entries, _graph

    entries = _entries(suite)
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
    active = []
    for key in leaves:
        x, y = _flatten(entries[key].crossSection[RECON_LABEL])
        if np.any(y[x < hi] != 0):
            active.append(key)
    top = min(float(_flatten(entries[k].crossSection[RECON_LABEL])[0][-1]) for k in active)
    def sourced(reaction):
        clone = copy.copy(reaction)
        clone.crossSection = copy.copy(reaction.crossSection)
        clone.crossSection.forms = dict(reaction.crossSection.forms)
        clone.crossSection[RECON_LABEL] = _dopplerSource(reaction.crossSection[RECON_LABEL], lo, top)
        return clone
    reduced = copy.copy(suite)
    reduced.reactions = type(suite.reactions)(
        [sourced(entries[k]) for k in active if k.container == "reactions"])
    reduced.fissionComponents = type(suite.fissionComponents)(
        [sourced(entries[k]) for k in active if k.container == "fissionComponents"])
    # The evaluated domain declared to the broadening is the one it is given.
    reduced.styles = copy.deepcopy(suite.styles)
    evaluated = reduced.styles.evaluatedFor(RECON_LABEL)
    if evaluated.projectileEnergyDomain is not None:
        evaluated.projectileEnergyDomain = _m().RangeQuantity(lo, top, "eV")
    for name in ("sums", "productions", "orphanProducts", "incompleteReactions"):
        setattr(reduced, name, type(getattr(suite, name))())
    reduced.sums.multiplicitySums = suite.sums.multiplicitySums
    support = {k: SupportPolicy("one_over_v", "constant") for k in active}
    opts = options or DopplerOptions(max_points=400_000_000, max_work_bytes=4_000_000_000)
    extra = {}
    if suite.resonances is not None and suite.resonances.unresolved is not None:
        # The 0 K forms in the unresolved range are the infinitely dilute
        # average reconstruction (kika's or RECONR's), which is the statement
        # broaden_suite asks to be made explicitly before it accepts a URR.
        from kika.processing.doppler import DopplerInputAccuracy

        extra = dict(urr_policy="dilute_mean",
                     input_accuracy=DopplerInputAccuracy(infinite_dilution=True))
    result = broaden_suite(reduced, None, source_style=RECON_LABEL, label=heatedLabel("tmp"),
                           temperature=float(temperature), support=support,
                           source_temperature=0.0, options=opts, output_domain=(lo, hi), **extra)
    out = {key.label: form for key, form in result.forms.items()}
    zero = None
    for key in leaves:
        if key.label not in out:
            if zero is None:
                grid = _flatten(next(iter(out.values())))[0]
                zero = _m().XYs1d(xs=grid, ys=np.zeros_like(grid))
            out[key.label] = zero
    return out, hi


def _broadenOne(args):
    suite, temperature, thnmax, options = args
    return temperature, _broadened(suite, temperature, thnmax, options)


def _broadenAll(suite, temperatures, thnmax, options, workers):
    """``{T: (heated forms, ceiling)}``, the temperatures in parallel processes."""
    workers = min(len(temperatures), 4) if workers is None else max(1, int(workers))
    if workers == 1 or len(temperatures) == 1:
        return {t: _broadened(suite, t, thnmax, options) for t in temperatures}
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_broadenOne, [(suite, t, thnmax, options) for t in temperatures]))


def processSuite(suite, temperatures=None, *, report=None, thnmax=DEFAULT_THNMAX,
                 njoy=None, endf=None, omit: Sequence[str] = (), awr: Optional[float] = None,
                 context=None, doppler=None, tabulation=None, ed: Optional[float] = None,
                 njoyPieces=None, thinning: Optional[float] = 1.0e-3,
                 workers: Optional[int] = None, label: Optional[str] = None):
    """The processed suite of ``suite`` at ``temperatures`` (kelvin; default 293.6 K).

    ``endf`` is the ENDF tape NJOY processes for the D7 pieces (``urr``,
    ``kerma``, ``damage``); without it, or without NJOY, they must be named in
    ``omit``. ``njoyPieces`` passes pieces already computed
    (:func:`kika.hdf5.njoy_pieces.runNjoyPieces`). ``awr`` overrides the
    atomic weight ratio (D11; default MF1/451's, as ACER takes it).
    ``thinning`` is the relative tolerance to which each temperature's grid is
    thinned (roadmap H5): points are dropped while every reaction, read
    lin-lin, reproduces every dropped point to it. 1e-3 is what the official
    libraries are processed to (BROADR's ``errthn``); ``None`` keeps the full
    union grid. The 0 K elastic is thinned to the same tolerance (RECONR's
    ``err`` for the official files). ``workers`` broadens that
    many temperatures at once, one process each (default: one per temperature,
    at most four -- the broadening is single-threaded). ``label`` processes
    the forms of that style -- a perturbation run's ``realization`` -- in
    place of the evaluated ones wherever it has them (roadmap H7, D12: the
    sample is multiplied at 0 K and broadened, never the reverse).
    """
    m = _m()
    from kika.processing.tabulation import TabulationContext, TabulationOptions, tabulateDistribution

    report = report if report is not None else m.ConversionReport()
    temperatures = [293.6] if temperatures is None else [float(t) for t in temperatures]
    omit = tuple(omit)
    unknown = set(omit) - {"urr", "kerma", "damage"}
    if unknown:
        raise ValueError(f"omit names pieces {sorted(unknown)} that are not taken from NJOY")
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
        thresholds = [_flatten(leaf.reaction.crossSection[RECON_LABEL])[0][0] for leaf in leaves
                      if (leaf.reaction.outputChannel.Q.value or 0.0) < 0]
        thnmax = njoyThnmax(work, min(thresholds) if thresholds else None)
    broadened = _broadenAll(work, temperatures, float(thnmax), doppler, workers)
    for temperature in temperatures:
        kT = aceKT(temperature)
        T = temperatureName(kT)
        styleKT[T] = kT
        below, hi = broadened[temperature]
        reconForms = {leaf.reaction.label: leaf.reaction.crossSection[RECON_LABEL] for leaf in leaves}
        hotGrid = np.unique(np.concatenate([_flatten(below[label])[0] for label in reconForms]))
        coldGrid = np.unique(np.concatenate([_flatten(f)[0] for f in reconForms.values()]))
        grid = np.unique(np.concatenate([hotGrid, coldGrid[coldGrid > hi]]))
        grid, _ = _strictly(grid, grid)
        grids[T] = grid
        values = {}
        for leaf in leaves:
            label = leaf.reaction.label
            hot = _onGrid(below[label], grid)
            cold = _onGrid(reconForms[label], grid)
            y = np.where(grid <= hi, hot, cold)
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
    for leaf in leaves:
        reaction = leaf.reaction
        mt = leaf.mt
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

    # --- gas production -----------------------------------------------------
    for gasMT, cs in _gas(work, leaves, heated, grids).items():
        out.productions.append(m.Reaction(id=m.ReactionId(label=f"MT{gasMT}", ENDF_MT=gasMT),
                                          crossSection=cs,
                                          outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                                          provenance=m.Hdf5Provenance(attributes={
                                              "redundant": 1, "center_of_mass": 0})))

    # --- fission ------------------------------------------------------------
    _fission(work, out, report)

    # --- NJOY pieces --------------------------------------------------------
    wanted = [p for p in ("urr", "kerma", "damage") if p not in omit]
    if wanted:
        if njoyPieces is None:
            if endf is None:
                raise MissingNJOYPieceError(
                    f"the pieces {wanted} are taken from NJOY (roadmap D7) and need the ENDF tape: "
                    f"pass endf=..., or omit={tuple(wanted)} to write without them", wanted)
            from .njoy_pieces import runNjoyPieces
            mat = getattr(suite.provenance, "mat", None)
            njoyPieces = runNjoyPieces(endf, temperatures, mat=int(mat), njoy=njoy,
                                       urr="urr" in wanted and _hasURR(suite),
                                       ed=_displacementEnergy(Z) if ed is None else ed)
        _attachNjoy(out, njoyPieces, temperatures, styleKT, grids, heated, wanted, report)
        # The URR tables name the reaction whose smooth cross section stands
        # for the inelastic competition, and the C++ requires it to exist
        # (nuclide.cpp:308-320): MT4 is written for that, as the official
        # exporter writes it, redundant and summed from the levels.
        for piece in njoyPieces.temperatures.values():
            inelastic = int(piece.urr.inelastic) if piece.urr is not None else 0
            if inelastic > 0 and out.findReactionByENDF_MT(inelastic) is None:
                levels = [c for c in LEVEL_SUMS.get(inelastic, ()) if c in present]
                if not levels:
                    raise HDF5FormatError(f"the URR tables name MT{inelastic} and the "
                                          f"evaluation has none of its levels")
                out.sums.append(m.CrossSectionSum(
                    id=m.ReactionId(label=f"MT{inelastic}", ENDF_MT=inelastic),
                    crossSection=summedForms(levels),
                    outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                    provenance=m.Hdf5Provenance(attributes={"redundant": 1})))
                report.warn(f"MT{inelastic} written as the redundant sum of {len(levels)} levels, "
                            f"because the URR tables name it")
    for piece in omit:
        report.lost(f"{piece} was not written: omitted on request (it is taken from NJOY, D7)")
    out.report = report
    return out


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


def _hasURR(suite) -> bool:
    return suite.resonances is not None and suite.resonances.unresolved is not None


def _displacementEnergy(Z: int) -> float:
    """HEATR's default ``ed`` is 25 eV; OpenMC passes 0 and lets HEATR choose (``make_ace``)."""
    return 0.0


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
    if component is None or not component.forms:
        return None
    return component.forms.get(EVAL_LABEL) or next(iter(component.forms.values()))


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
    x = _flatten(reaction.crossSection[RECON_LABEL])[0]
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
    out = []
    lo, hi = _reactionRange(reaction)
    for pid in residuals:
        if pid not in pops:
            continue
        for line in m.cascadeLines(pops, pid):
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
    if form is None:
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


def _attachNjoy(out, pieces, temperatures, styleKT, grids, heated, wanted, report):
    m = _m()
    from .fission_energy import evaluate

    release = None
    fission = next((r for r in out.reactions if r.ENDF_MT in FISSION_MTS), None)
    if fission is not None and fission.outputChannel.fissionFragmentData is not None:
        releases = fission.outputChannel.fissionFragmentData.fissionEnergyReleases
        release = releases[0] if releases else None
    responses = {301: m.CrossSection(), 901: m.CrossSection(), 444: m.CrossSection()}
    for temperature, (T, kT) in zip(temperatures, styleKT.items()):
        piece = pieces.temperatures.get(float(temperature))
        if piece is None:
            raise MissingNJOYPieceError(f"no NJOY pieces at {temperature} K", wanted)
        grid = grids[T]
        total, sigmaF = _totalOn(out, T, grid)
        label = heatedLabel(T)
        if "kerma" in wanted:
            nonlocal_ = np.interp(grid, *piece.kerma) * total
            local = np.interp(grid, *piece.kermaLocal) * total
            if release is not None:
                efr = evaluate(release.promptProductKE, grid)
                eb = evaluate(release.delayedBetaEnergy, grid)
                egp = evaluate(release.promptGammaEnergy, grid)
                egd = evaluate(release.delayedGammaEnergy, grid)
                nonlocal_ = nonlocal_ + (efr + eb) * sigmaF
                local = local + (efr + egp + egd + eb) * sigmaF
            responses[301][label] = m.XYs1d(xs=grid, ys=nonlocal_, label=label)
            responses[901][label] = m.XYs1d(xs=grid, ys=local, label=label)
        if "damage" in wanted and piece.damage is not None:
            responses[444][label] = m.XYs1d(xs=grid, ys=np.interp(grid, *piece.damage) * total,
                                            label=label)
        if "urr" in wanted and piece.urr is not None:
            ulabel = urrLabel(T)
            if ulabel not in out.styles.labels:
                out.styles.add(m.URR_probabilityTables(label=ulabel, derivedFrom=label))
            for mt in (2, 18, 102):
                reaction = out.findReactionByENDF_MT(mt)
                if reaction is not None:
                    reaction.crossSection[ulabel] = m.URR_probabilityTables1d(label=ulabel,
                                                                             table=piece.urr)
    names = {301: "heating", 901: "heating-local", 444: "damage-energy"}
    for mt, cs in responses.items():
        if len(cs.forms):
            out.productions.append(m.Reaction(
                id=m.ReactionId(label=names[mt], ENDF_MT=mt), crossSection=cs,
                outputChannel=m.OutputChannel(Q=m.Q(value=0.0)),
                provenance=m.Hdf5Provenance(attributes={
                    "redundant": 1, "center_of_mass": 0 if mt == 444 else 1})))
    report.warn(f"from NJOY ({pieces.njoy}), per roadmap D7: {', '.join(wanted)}")

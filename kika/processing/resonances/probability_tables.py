"""Unresolved-resonance probability tables from the model's URR parameters.

kika's own PURR. At each energy of the unresolved grid the average parameters
are frozen and resonance ladders are drawn from them -- spacings from Wigner's
distribution, widths from chi-square distributions of the stated degrees of
freedom -- the cross sections are evaluated at random energies inside each
ladder with single-level Breit-Wigner and free-gas Doppler broadening (the
psi/chi line shapes, through the Faddeeva function), and the samples are
binned by total cross section into probability bands with their conditional
partial cross sections. The procedure is NJOY PURR's (``purr.f90``,
``unrest``/``ladr2``), with its choices kept where they are choices:

- a ladder is abstract: its span is ``0.9 nermax`` times the smallest mean
  spacing (``nermax = 1000``), the 10 000 sample energies avoid 300 mean
  spacings at each end, and every spin sequence is drawn over the whole span;
- 64 ladders (OpenMC's ``make_ace`` asks PURR for ``20 64``), shared by every
  temperature so that the tables of different temperatures are correlated as
  PURR's are;
- a resonance contributes within ``31.83 Γ + max(31.83 Γ, 20 Δ)`` of its
  energy, Δ the Doppler width;
- ladders are not rescaled one by one (PURR's default ``nmode = 0``; the
  per-ladder rescaling of ``nmode = 1`` divides by a ladder's mean resonance
  elastic, which interference can put near zero): the table as a whole is
  normalised to the dilute averages, which the factors below do;
- band boundaries come from the first ladder's sorted totals with PURR's
  spacing, finer in both tails (``purr.f90:2274-2318``), per temperature;
- the tables are factors on the smooth cross sections (``multiply_smooth``):
  each band's conditional mean over the mean of all samples;
- the competition flags follow PURR's rule (``purr.f90:1159-1190``).

The random numbers are kika's (NumPy, seeded), so the tables agree with PURR's
statistically, band by band, not bit for bit.

Every table carries its statistical error (:class:`ProbabilityTableStatistics`):
the ladder is the statistical unit -- the samples of one ladder share its
resonances and are not independent -- so the errors are delete-one-ladder
jackknife errors over the ladders, with the bands held at the first ladder's
bounds (as FRENDY's ``check_statistical_error``, which uses the CLT on each
band's ``p · σ``). ``errorTarget`` stops drawing ladders once they are small
enough, with ``ladders`` as the cap.

``background`` adds the smooth cross section PURR takes from MF3 (:func:`purrInputs`):
for LSSF=0 the file's MF3 values; for LSSF=1, where MF3 already holds the dilute
average, no elastic, fission or capture background, and only the competition kept in
the total.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ProbabilityTableOptions", "probabilityTables", "competitionFlags", "bandBounds"]

#: PURR's constants (``purr.f90``): resonances per ladder, resonances kept
#: clear of each end, and the reach of a resonance in widths and Doppler widths.
NERMAX = 1000
NAVOID = 300
REACH_WIDTHS = 31.83
REACH_DOPPLER = 20.0

#: PURR's chi-square table (``ladr2``): 20 equiprobable quantiles for 1-4
#: degrees of freedom. A width is ``mean/ν × CHISQ[n, ν]`` with ``n`` drawn
#: uniformly -- a discrete distribution with no extreme tail, which is what
#: shapes PURR's outer bands.
CHISQ = np.array([
    [1.31003e-3, 9.19501e-3, .0250905, .049254, .0820892, .124169, .176268, .239417,
     .314977, .404749, .511145, .637461, .788315, .970419, 1.194, 1.47573, 1.84547,
     2.36522, 3.20371, 5.58201],
    [.0508548, .156167, .267335, .38505, .510131, .643564, .786543, .940541, 1.1074,
     1.28947, 1.48981, 1.71249, 1.96314, 2.24984, 2.58473, 2.98744, 3.49278, 4.17238,
     5.21888, 7.99146],
    [.206832, .470719, .691933, .901674, 1.10868, 1.31765, 1.53193, 1.75444, 1.98812,
     2.23621, 2.50257, 2.79213, 3.11143, 3.46967, 3.88053, 4.36586, 4.96417, 5.75423,
     6.94646, 10.0048],
    [.459462, .893735, 1.21753, 1.50872, 1.78605, 2.05854, 2.33194, 2.61069, 2.89878,
     3.20032, 3.51995, 3.86331, 4.23776, 4.65345, 5.12533, 5.67712, 6.35044, 7.22996,
     8.541, 11.8359]])


@dataclass(frozen=True)
class ProbabilityTableOptions:
    bins: int = 20
    #: Ladders per energy; with :attr:`errorTarget` the most that are drawn.
    ladders: int = 64
    samples: int = 10000
    seed: int = 1
    #: The C kernel of ``_rm_native`` (Humlicek's W4 line shapes) when built;
    #: ``False`` evaluates the Faddeeva function exactly in NumPy, the reference.
    native: bool = True
    #: Stop drawing ladders at an energy once every statistical error checked
    #: is at most this (``None``: always draw :attr:`ladders`, PURR's way). The
    #: errors checked are each band's ``p_b · factor_b`` (its share of the
    #: dilute average) and the relative error of ``f(σ0)`` at :attr:`sigma0`,
    #: for every reaction with a nonzero average at every temperature.
    errorTarget: Optional[float] = None
    #: How often, in ladders, the target is checked (FRENDY: 10).
    checkEvery: int = 10
    #: Background cross sections (b) of the Bondarenko factors in the statistics.
    sigma0: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __post_init__(self):
        if self.errorTarget is not None and not self.errorTarget > 0:
            raise ValueError(f"errorTarget must be positive; got {self.errorTarget}")
        if self.checkEvery < 2:
            raise ValueError(f"checkEvery must be at least 2 ladders; got {self.checkEvery}")


def _nativeKernel():
    try:
        from . import _rm_native
    except ImportError:
        return None
    return _rm_native if hasattr(_rm_native, "urr_ladder") else None


def bandBounds(totals: np.ndarray, bins: int) -> np.ndarray:
    """PURR's band boundaries (``tval``) from one ladder's totals; the last is +inf."""
    es = np.sort(np.asarray(totals, dtype=float))
    nsamp = len(es)
    nebin = int(nsamp / (bins - 10 + 1.76))
    ibin = max(nebin // 200, 1)
    tval = np.empty(bins)
    for i in range(1, bins):
        ibin = min(ibin, nsamp)
        tval[i - 1] = es[ibin - 1]
        if i > 1 and tval[i - 1] <= tval[i - 2]:
            tval[i - 1] = tval[i - 2] + tval[i - 2] / 20
        if i == 1:
            ibin += nebin // 40
        elif i == 2:
            ibin += nebin // 10
        elif i == 3:
            ibin += nebin // 4
        elif i == 4:
            ibin += nebin // 2
        elif 4 < i < bins - 5:
            ibin += nebin
        elif i == bins - 5:
            ibin += nebin // 2
        elif i == bins - 4:
            ibin += nebin // 4
        elif i == bins - 3:
            ibin += nebin // 10
        elif i == bins - 2:
            ibin += nebin // 40
        elif i == bins - 1:
            ibin += nebin // 200
    tval[-1] = np.inf
    return tval


def competitionFlags(competingMTs: Sequence[int]):
    """``(inelastic, absorption)`` as PURR sets them from the competing reactions."""
    inelastic, absorption = -1, -1
    for mt in competingMTs:
        if mt <= 4 or mt in (18, 19, 102):
            continue
        if 51 <= mt <= 91:
            inelastic = mt if inelastic < 0 else 4
        else:
            absorption = mt if absorption < 0 else 0
    return inelastic, absorption


def _prepared(region, context):
    """The URR spin groups of a model region, prepared even when LSSF=1."""
    from .unresolved import prepare_unresolved

    source = region.tabulatedWidths
    if getattr(source, "selfShieldingOnly", False):
        region = replace(region, tabulatedWidths=replace(source, selfShieldingOnly=False))
    return prepare_unresolved(region, context, [])


def _psiChi(x, xi):
    """SLBW Doppler line shapes (ENDF-102 D.1.4): ψ, χ from the Faddeeva function."""
    from scipy.special import wofz

    w = wofz(0.5 * xi * (x + 1j))
    factor = 0.5 * np.sqrt(np.pi) * xi
    return factor * w.real, factor * w.imag


def _sequence(group, energy):
    """A spin sequence's frozen parameters at ``energy``."""
    from .channel_functions import neutral_channel_functions

    ctx = group.context
    k2 = ctx.k_squared_per_ev * energy
    rho = np.sqrt(k2) * float(group.channel_radius.evaluate([energy])[0])
    p, _, _ = neutral_channel_functions(group.l, np.array([rho]))
    _, _, phi = neutral_channel_functions(
        group.l, np.array([np.sqrt(k2) * float(group.phase_radius.evaluate([energy])[0])]))
    means = np.array([float(v.evaluate([energy])[0]) for v in group.averages])
    if group.reduced_neutron and means[0] > 0:
        means[0] = means[0] * np.sqrt(energy) * group.degrees[0] * float(p[0]) / rho
    return dict(group=group, D=float(group.spacing.evaluate([energy])[0]), means=means,
                degrees=np.asarray(group.degrees, dtype=float), phi=float(phi[0]), k2=k2,
                g=(2 * group.spin + 1) / (2 * (2 * ctx.target_spin + 1)))


def _ladder(rng, seq, elow, ehigh):
    """PURR's ``ladr2``: resonance energies from Wigner, widths from chi-square."""
    dcon = seq["D"] * np.sqrt(4.0 / np.pi)
    expected = int((ehigh - elow) / seq["D"] * 1.5) + 20
    steps = dcon * np.sqrt(-np.log(rng.uniform(size=expected)))
    er = elow + dcon * rng.uniform() + np.concatenate([[0.0], np.cumsum(steps[:-1])])
    er = er[er <= ehigh + dcon]
    n = len(er)
    widths = []
    for mean, nu in zip(seq["means"], seq["degrees"]):
        if mean <= 0:
            widths.append(np.zeros(n))
        elif nu <= 0:
            widths.append(np.full(n, mean))
        elif float(nu).is_integer() and 1 <= nu <= 4:
            pick = rng.integers(0, 20, n)
            widths.append(mean / nu * CHISQ[int(nu) - 1, pick])
        else:
            widths.append(mean * rng.chisquare(nu, n) / nu)
    while len(widths) < 4:
        widths.append(np.zeros(n))
    return er, widths


def _estimates(C, S, T, sigma0, transportXS=None):
    """A table's estimates from its ladder sums, over any leading axes.

    ``C`` (..., M) samples per band, ``S`` (..., 4, M) their sums of total,
    elastic, fission and capture, ``T`` (..., 4) the sums over all samples.
    Returns the band probabilities, the factors, each band's share
    ``p · factor`` of the dilute average, and the Bondarenko factors ``f(σ0)``
    of the four rows from the bands' cross sections, ``(..., 4, K)``.
    """
    n = C.sum(-1)
    p = C / n[..., None]
    means = np.where(C[..., None, :] > 0, S / np.maximum(C, 1)[..., None, :], 0.0)
    overall = T / n[..., None]
    live = overall != 0
    factor = np.where(live[..., None], means / np.where(live, overall, 1.0)[..., None], 1.0)
    contribution = p[..., None, :] * factor
    if transportXS is not None:
        # OpenMC multiplies the partial factors by smooth elastic, fission
        # and ALL non-fission absorption, then reconstructs the total. The
        # published total row is not used by the transport reader.
        elastic, fission, capture, inelastic = transportXS
        means = factor * np.asarray([0., elastic, fission, capture])[..., None]
        means[..., 0, :] = means[..., 1:4, :].sum(-2) + inelastic
    dilute = (p[..., None, :] * means).sum(-1)
    shielding = np.empty(means.shape[:-1] + (len(sigma0),))
    for k, s0 in enumerate(sigma0):
        w = p / (means[..., 0, :] + s0)
        shielding[..., k] = (w[..., None, :] * means).sum(-1) / w.sum(-1)[..., None]
    shielding = np.where(dilute[..., None] != 0,
                         shielding / np.where(dilute != 0, dilute, 1.0)[..., None], 1.0)
    return dict(probability=p, factor=factor, contribution=contribution, shielding=shielding)


def _ladderStatistics(lc, ls, lt, sigma0, transportXS=None):
    """Estimates and their delete-one-ladder jackknife errors from per-ladder sums.

    ``lc`` (L, M), ``ls`` (L, 4, M), ``lt`` (L, 4): one row per ladder, the
    bands fixed. The ladders are independent and identically distributed
    given the bands, so the jackknife over them is the error of the table.
    """
    lc, ls, lt = (np.asarray(a, dtype=float) for a in (lc, ls, lt))
    L = lc.shape[0]
    C, S, T = lc.sum(0), ls.sum(0), lt.sum(0)
    full = _estimates(C, S, T, sigma0, transportXS)
    out = dict(full, ladders=L)
    if L < 2:
        for key, value in full.items():
            out[key + "Sigma"] = np.full_like(value, np.inf)
        return out
    loo = _estimates(C - lc, S - ls, T - lt, sigma0, transportXS)
    for key, value in loo.items():
        out[key + "Sigma"] = np.sqrt((L - 1) / L * ((value - value.mean(0)) ** 2).sum(0))
    return out


def _converged(stats, target):
    """Every band share and every relative ``f(σ0)`` error of a live row within ``target``."""
    for st in stats:
        # A row without an average (fission of a non-fissile) is all ones.
        for r in (r for r in range(4) if np.any(st["factor"][r] != 1.0)):
            if not np.all(st["contributionSigma"][r] <= target):
                return False
            if not np.all(st["shieldingSigma"][r] <= target * np.abs(st["shielding"][r])):
                return False
    return True


def _tablesAt(job, keepLadders=False):
    """The tables of one energy at every temperature: ``(nT, 6, bins)``, with
    their statistics (one dict per temperature) and, with ``keepLadders``,
    the per-ladder sums.

    Its own random stream, seeded by ``(seed, index)``, so the tables do not
    depend on how the energies are split between processes.
    """
    from kika._constants import BOLTZMANN_CONSTANT

    prepared, A, E, ie, temperatures, bkg, competitionBkg, options = job[:8]
    transportXS = job[8] if len(job) > 8 else [None] * len(temperatures)
    native = _nativeKernel() if options.native else None
    rng = np.random.default_rng([options.seed, ie])
    nT, bins = len(temperatures), options.bins
    tables = np.zeros((nT, 6, bins))
    sequences = [_sequence(g, E) for g in prepared.groups]
    potential, seen = 0.0, set()
    for s in sequences:
        if s["group"].l not in seen:
            potential += 4 * np.pi * 0.01 / s["k2"] * (2 * s["group"].l + 1) * np.sin(s["phi"]) ** 2
            seen.add(s["group"].l)
    density = sum(1.0 / s["D"] for s in sequences)
    dmin = min(s["D"] for s in sequences)
    elow = 10.0
    erange = 0.9 * NERMAX * dmin
    ehigh = elow + erange
    emin, emax = elow + NAVOID / density, ehigh - NAVOID / density
    if emax <= emin:
        raise ValueError("the ladder is too short for PURR's resonance-avoidance margins")
    dopplers = [np.sqrt(4.0 * t * BOLTZMANN_CONSTANT * 1.0e6 * E / A) for t in temperatures]
    bounds = [None] * nT
    counts = np.zeros((nT, bins))
    sums = np.zeros((nT, 4, bins))
    everything = np.zeros((nT, 4))
    # The same sums ladder by ladder, for the errors; the running sums above
    # are the table's, untouched by them.
    lc = np.zeros((nT, options.ladders, bins))
    ls = np.zeros((nT, options.ladders, 4, bins))
    lt = np.zeros((nT, options.ladders, 4))
    used = options.ladders
    for ladder in range(options.ladders):
        es = np.sort(rng.uniform(emin, emax, options.samples))
        el = np.zeros((nT, len(es)))
        ca = np.zeros((nT, len(es)))
        fi = np.zeros((nT, len(es)))
        for s in sequences:
            er, (gn, gg, gf, gx) = _ladder(rng, s, elow, ehigh)
            gt = gn + gg + gf + gx
            keep = gt > 0
            er, gn, gg, gf, gt = (np.ascontiguousarray(v[keep], dtype=float)
                                  for v in (er, gn, gg, gf, gt))
            sigma0 = 4 * np.pi * 0.01 / s["k2"] * s["g"] * gn / gt
            c2, s2 = np.cos(2 * s["phi"]), np.sin(2 * s["phi"])
            for it, delta in enumerate(dopplers):
                if native is not None:
                    native.urr_ladder(len(es), len(er), c2, s2, float(delta), REACH_WIDTHS,
                                      REACH_DOPPLER, es, er, gt, gn, gg, gf, sigma0,
                                      el[it], ca[it], fi[it])
                    continue
                reach = REACH_WIDTHS * gt + np.maximum(REACH_WIDTHS * gt, REACH_DOPPLER * delta)
                lo = np.searchsorted(es, er - reach, side="left")
                hi = np.searchsorted(es, er + reach, side="right")
                counts_ = hi - lo
                if counts_.sum() == 0:
                    continue
                owner = np.repeat(np.arange(len(er)), counts_)
                start = np.repeat(lo - np.concatenate([[0], np.cumsum(counts_)[:-1]]), counts_)
                index = np.arange(counts_.sum()) + start
                x = 2.0 * (es[index] - er[owner]) / gt[owner]
                if delta > 0:
                    psi, chi = _psiChi(x, gt[owner] / delta)
                else:
                    psi, chi = 1.0 / (1.0 + x * x), x / (1.0 + x * x)
                base = sigma0[owner]
                ratio = gn[owner] / gt[owner]
                size = len(es)
                el[it] += np.bincount(index, base * ((c2 - (1.0 - ratio)) * psi + s2 * chi), size)
                ca[it] += np.bincount(index, base * gg[owner] / gt[owner] * psi, size)
                fi[it] += np.bincount(index, base * gf[owner] / gt[owner] * psi, size)
        for it in range(nT):
            elastic = np.maximum(potential + el[it] + bkg[2], 1.0e-6)
            capture = ca[it] + bkg[102]
            fission = fi[it] + bkg[18]
            totals = elastic + capture + fission + competitionBkg
            if bounds[it] is None:
                bounds[it] = bandBounds(totals, bins)
            which = np.minimum(np.searchsorted(bounds[it], totals, side="left"), bins - 1)
            np.add.at(counts[it], which, 1)
            for row, values in enumerate((totals, elastic, fission, capture)):
                np.add.at(sums[it, row], which, values)
            everything[it] += [totals.sum(), elastic.sum(), fission.sum(), capture.sum()]
            lc[it, ladder] = np.bincount(which, minlength=bins)
            for row, values in enumerate((totals, elastic, fission, capture)):
                ls[it, ladder, row] = np.bincount(which, values, bins)
            lt[it, ladder] = [totals.sum(), elastic.sum(), fission.sum(), capture.sum()]
        if (options.errorTarget is not None and (ladder + 1) % options.checkEvery == 0
                and ladder + 1 < options.ladders):
            n = ladder + 1
            if _converged([_ladderStatistics(lc[it, :n], ls[it, :n], lt[it, :n], options.sigma0, transportXS[it])
                           for it in range(nT)], options.errorTarget):
                used = n
                break
    lc, ls, lt = lc[:, :used], ls[:, :used], lt[:, :used]
    stats = [_ladderStatistics(lc[it], ls[it], lt[it], options.sigma0, transportXS[it]) for it in range(nT)]
    converged = None if options.errorTarget is None else _converged(stats, options.errorTarget)
    for st in stats:
        st["converged"] = converged
    for it in range(nT):
        n = counts[it].sum()
        means = np.where(counts[it] > 0, sums[it] / np.maximum(counts[it], 1), 0.0)
        overall = everything[it] / n
        safe = np.where(overall != 0, overall, 1.0)
        factors = np.where(overall[:, None] != 0, means / safe[:, None], 1.0)
        cdf = np.array([float(f"{c:.6e}") for c in np.cumsum(counts[it] / n)])
        cdf[-1] = 1.0
        tables[it, 0] = cdf
        tables[it, 1:5] = factors
        # Not used in transport (nuclide.cpp:963-987): a unit factor in the
        # scale OpenMC stores it (it multiplies the row by 1e6 on read).
        tables[it, 5] = 1.0e6
    if keepLadders:
        return tables, stats, dict(counts=lc, sums=ls, totals=lt)
    return tables, stats


def probabilityTables(region, context, energies, temperatures, *,
                      background: Optional[Dict[int, np.ndarray]] = None,
                      competing: Sequence[int] = (),
                      options: ProbabilityTableOptions = ProbabilityTableOptions(),
                      workers: int = 1, transportXS=None):
    """``{T: ProbabilityTables}`` at the temperatures (K) on ``energies`` (eV).

    ``region`` is the model's :class:`UnresolvedRegion`, ``context`` the
    :class:`NeutronContext`. ``background`` gives a smooth cross section per
    MT (1, 2, 18, 102) at each energy (:func:`purrInputs`); ``competing`` the
    MTs whose smooth cross sections compete in the URR, for the flags.
    ``workers`` computes that many energies at once, one process each.

    Each table carries its statistical error as ``statistics``
    (:class:`ProbabilityTableStatistics`); ``options.errorTarget`` draws
    ladders at each energy only until that error is reached. ``transportXS``
    optionally maps each kelvin temperature to an (nE, 4) array of smooth
    elastic, fission, non-fission absorption and competing inelastic XS (b).
    With it, shielding and its stopping error follow OpenMC's actual reader;
    without it they describe the unscaled ladder cross sections. Fixed-count
    table values are unaffected by this diagnostic choice.
    """
    from kika.nuclear_data.model import ProbabilityTables, ProbabilityTableStatistics

    temperatures = [float(t) for t in np.atleast_1d(temperatures)]
    energies = np.asarray(energies, dtype=float)
    results = _run(_tablesAt, region, context, energies, temperatures, background, options, workers, transportXS)
    tables = np.stack([r[0] for r in results], axis=1)              # (nT, nE, 6, bins)
    inelastic, absorption = competitionFlags(competing)
    out = {}
    for it, t in enumerate(temperatures):
        st = [r[1][it] for r in results]

        def stack(key):
            return np.stack([s[key] for s in st])

        statistics = ProbabilityTableStatistics(
            ladders=np.array([s["ladders"] for s in st]), sigma0=np.asarray(options.sigma0, dtype=float),
            probabilitySigma=stack("probabilitySigma"), factorSigma=stack("factorSigma"),
            contributionSigma=stack("contributionSigma"), shielding=stack("shielding"),
            shieldingSigma=stack("shieldingSigma"), errorTarget=options.errorTarget,
            converged=None if options.errorTarget is None else np.array([s["converged"] for s in st]),
            shieldingBasis="openmc_transport" if transportXS is not None else "ladder_cross_sections")
        out[t] = ProbabilityTables(energies=energies, table=tables[it], interpolation=2,
                                   inelastic=inelastic, absorption=absorption, multiplySmooth=True,
                                   statistics=statistics)
    return out


def _ladderAccumulators(region, context, energies, temperatures, *, background=None,
                        competing=(), options=ProbabilityTableOptions(), workers=1):
    """Per energy, the per-ladder sums behind :func:`probabilityTables` (diagnostics):
    ``counts`` (nT, L, M), ``sums`` (nT, L, 4, M), ``totals`` (nT, L, 4)."""
    temperatures = [float(t) for t in np.atleast_1d(temperatures)]
    results = _run(partial(_tablesAt, keepLadders=True), region, context,
                   np.asarray(energies, dtype=float), temperatures, background, options, workers)
    return [r[2] for r in results]


def _run(task, region, context, energies, temperatures, background, options, workers, transportXS=None):
    """``task`` on every energy's job, in ``workers`` processes."""
    prepared = _prepared(region, context)
    background = background or {}
    if transportXS is not None:
        transportXS = {float(t): np.asarray(transportXS[float(t)], dtype=float)
                       for t in temperatures}
        for values in transportXS.values():
            if values.shape != (len(energies), 4) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("transportXS must provide finite nonnegative (nE, 4) arrays "
                                 "of elastic, fission, non-fission absorption and inelastic XS")

    def bkgAt(mt, ie):
        return float(np.asarray(background[mt])[ie]) if mt in background else 0.0

    jobs = []
    for ie, E in enumerate(energies):
        bkg = {mt: bkgAt(mt, ie) for mt in (2, 18, 102)}
        competitionBkg = (bkgAt(1, ie) - bkg[2] - bkg[18] - bkg[102]) if 1 in background else 0.0
        jobs.append((prepared, float(context.atomic_weight_ratio), float(E), ie, temperatures,
                     bkg, competitionBkg, options,
                     [None if transportXS is None else transportXS[t][ie] for t in temperatures]))
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            return list(pool.map(task, jobs))
    return [task(job) for job in jobs]


# ----------------------------------------------------------------------
# PURR's inputs: the energy grid, the backgrounds and the competition
# ----------------------------------------------------------------------

#: RECONR/PURR's standard grid (``egridu``), used to fill gaps in the URR grid.
EGRIDU = np.array([
    1.0e1, 1.25e1, 1.5e1, 1.7e1, 2.0e1, 2.5e1, 3.0e1, 3.5e1, 4.0e1, 5.0e1, 6.0e1, 7.2e1,
    8.5e1, 1.0e2, 1.25e2, 1.5e2, 1.7e2, 2.0e2, 2.5e2, 3.0e2, 3.5e2, 4.0e2, 5.0e2, 6.0e2,
    7.2e2, 8.5e2, 1.0e3, 1.25e3, 1.5e3, 1.7e3, 2.0e3, 2.5e3, 3.0e3, 3.5e3, 4.0e3, 5.0e3,
    6.0e3, 7.2e3, 8.5e3, 1.0e4, 1.25e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 3.5e4, 4.0e4,
    5.0e4, 6.0e4, 7.2e4, 8.5e4, 1.0e5, 1.25e5, 1.5e5, 1.7e5, 2.0e5, 2.5e5, 3.0e5, 3.5e5,
    4.0e5, 5.0e5, 6.0e5, 7.2e5, 8.5e5, 1.0e6, 1.25e6, 1.5e6, 1.7e6, 2.0e6, 2.5e6, 3.0e6,
    3.5e6, 4.0e6, 5.0e6, 6.0e6, 7.2e6, 8.5e6])
#: A gap wider than this ratio is filled from :data:`EGRIDU` (``wide``).
WIDE = 1.26


def sigfig(x: float, digits: int = 7, shade: int = 0) -> float:
    """NJOY's ``sigfig`` (``util.f90``): ``x`` to ``digits`` figures, shaded by ``shade`` in the last."""
    if x == 0:
        return 0.0
    a = np.log10(abs(x))
    power = int(a)
    if a < 0:
        power -= 1
    power = digits - 1 - power
    ii = int(np.floor(x * 10.0 ** power + 10.0 ** (digits - 11) + 0.5))
    if ii >= 10 ** digits:
        ii //= 10
        power -= 1
    return (ii + shade) * 10.0 ** (-power) * 1.0000000000001


def purrGrid(region) -> np.ndarray:
    """The energies PURR tabulates at (``rdf2un``): the range's ends, shaded
    inward, the parameters' interior nodes, and the standard grid in the gaps."""
    source = region.tabulatedWidths
    lo, hi = float(region.domainMin), float(region.domainMax)
    first = source.spinGroups[0] if source.spinGroups else None
    declared = (first.levelSpacingEnergies if first is not None and first.levelSpacingEnergies is not None
                else source.energyGrid)
    independent = declared is None
    nodes = {sigfig(lo, 7, +1), sigfig(hi, 7, -1)}
    if declared is not None:
        nodes.update(float(e) for e in np.asarray(declared)[1:-1] if lo <= e <= hi)
    nodes = sorted(nodes)
    out = [nodes[0]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        if independent or b >= WIDE * a:
            et = a
            while True:
                above = EGRIDU[EGRIDU > et + et / 100]
                et = float(above[0]) if above.size else b
                if et >= b:
                    break
                out.append(et)
        out.append(b)
    return np.array(out)


#: The MF3 sections PURR reads (``rdf3un``): their thresholds decide the competition.
def _purrReads(mt: int) -> bool:
    return (mt < 6 or 10 < mt < 12 or 15 < mt < 43 or 43 < mt < 46 or 49 < mt < 92
            or 100 < mt < 110 or 110 < mt < 118 or 150 < mt <= 200 or mt >= 600) and mt <= 891


def purrInputs(suite, energies, form=None):
    """``(background, competing)`` for :func:`probabilityTables`, by PURR's rules.

    ``form(component)`` returns the evaluated cross-section function of a
    reaction (default: the ``eval`` form, else the first). The competition
    exists only where the total exceeds elastic + fission + capture; then
    every reaction PURR reads whose threshold is below the top of the grid
    competes. For LSSF=1 the elastic, fission and capture backgrounds are
    zero -- the file's MF3 *is* the average -- and only the competition is
    kept in the total (``purr.f90:1195-1232``); for LSSF=0 the MF3 values are
    the backgrounds.
    """
    energies = np.asarray(energies, dtype=float)
    if form is None:
        def form(component):
            forms = component.forms if component is not None else None
            if not forms:
                return None
            return forms.get("eval") or next(iter(forms.values()))
    sections = {}
    for container in (suite.reactions, suite.sums, suite.fissionComponents):
        for reaction in container:
            mt = reaction.ENDF_MT
            if mt is None or mt in sections or not _purrReads(mt):
                continue
            f = form(reaction.crossSection)
            if f is not None:
                sections[mt] = f
    at = energies.copy()
    at[0] *= 1.0 + 1e-6
    at[-1] *= 1.0 - 1e-6

    def value(mt):
        return np.asarray(sections[mt].evaluate(at), dtype=float) if mt in sections else np.zeros(len(at))

    parts = value(2) + value(18) + value(102)
    if 1 in sections:
        total = value(1)
    else:
        # No MT1: the sum of the exclusive reactions, which is what it states.
        total = np.zeros(len(at))
        for reaction in suite.reactions:
            f = form(reaction.crossSection)
            if f is not None:
                total = total + np.asarray(f.evaluate(at), dtype=float)
    excess = total - parts
    tol = 1.0e-6
    competing = []
    start = np.flatnonzero(excess > 1.0e-12)
    if start.size:
        for mt, f in sorted(sections.items()):
            if mt > 4 and mt not in (18, 19, 102) and (1.0 + 1e-6) * float(f.domainMin) < energies[-1]:
                competing.append(mt)
    lssf = bool(getattr(suite.resonances.unresolved.tabulatedWidths, "selfShieldingOnly", False))
    if lssf:
        competition = np.where(excess > tol * total, excess, 0.0)
        if not competing:
            competition[:] = 0.0
        else:
            competition[energies < energies[start[0]]] = 0.0
        background = {1: competition, 2: np.zeros(len(at)), 18: np.zeros(len(at)), 102: np.zeros(len(at))}
    else:
        background = {1: total, 2: value(2), 18: value(18), 102: value(102)}
    return background, competing

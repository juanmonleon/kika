"""ACE photon production → the photon products of a reaction.

A continuous-energy neutron ACE table states its photons in five blocks, one
entry per *photon production reaction* (MTRP, usually ``MT*1000 + i``): SIGP
says how many (a yield on the MT's σ, or a production cross section), LANDP/
ANDP the angle, LDLWP/DLWP the energy, and GPD the sum of them all. The ENDF
adapter puts MF12-15 on the model as **products** — one ``photon`` per line,
its multiplicity on the product and an ``uncorrelated`` distribution — and this
module produces the same shape from ACE, so a reader that walks
``reaction.outputChannel.products`` for photons does not need to know which
format the suite came from:

=======================================  ========================================
ACE                                      model
=======================================  ========================================
SIGP MFTYPE 12 / 16 (yield on MTMULT)    multiplicity ``XYs1d`` on reaction MTMULT
SIGP MFTYPE 13 (σ_γ)                     multiplicity σ_γ / σ_MT, *approximated*
SIGP MFTYPE 13 with no MT to divide by   an orphan product per MT carrying the sum
                                         of its photons' σ_γ, each photon y =
                                         σ_γ / Σ (y = 1 when it is the only one)
DLWP law 2, LP = 0 or 1                  ``Uncorrelated(energy=DiscreteGamma)``
DLWP law 2, LP = 2                       ``Uncorrelated(energy=PrimaryGamma)``
DLWP law 4                               ``Uncorrelated(energy=XYs2d)`` continuum
LANDP = 0 (isotropic)                    ``Uncorrelated(angular=Isotropic2d)``
ANDP 32 equiprobable bins / tabulated    ``XYs2d`` of flat / tabulated ``XYs1d``
GPD                                      ``suite.photonProductionTotal`` (not GNDS)
=======================================  ========================================

**The label is the table's ``heated_<T>``, not ``eval``.** The decoder that
calls this declares that nothing in an ACE-derived suite carries an
eval-labelled form, because ACE refers to an evaluation it does not contain;
the photon data is no exception (NJOY's ACER has linearised it, thinned it and
expanded every level cascade into lines). So it hangs off the heated style,
the table's statement at its temperature, as the cross sections do (ACE
:func:`~kika.ace.model_adapter.decode.aceLabels`). A reader
written for ENDF suites asks for ``distribution["eval"]`` and must ask for
that label here; ``multiplicity.form`` works for both, because a multiplicity
with a single form answers ``.form`` whatever its label.

**What ACE does not have, and the model therefore does not get.** ACE has no
level scheme: ACER turned every MF12 LO=2 cascade into one line per transition
with its own yield on the level's MT. So a discrete-level reaction's photons
sit on the reaction's own output channel, as plain discrete gammas, and PoPs
gets no ``decayData``. The ENDF adapter puts the same photons in the decay of
the excited residual; the *numbers* agree (the tests compare them line by
line), the *structure* cannot.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    AceProvenance,
    ConversionReport,
    Constant1d,
    DiscreteGamma,
    Distribution,
    Frame,
    Interpolation,
    Isotropic2d,
    Multiplicity,
    OutputChannel,
    PrimaryGamma,
    Product,
    Reaction,
    ReactionId,
    Regions1d,
    Uncorrelated,
    XYs1d,
    angularAxes,
    crossSectionAxes,
    energyAxes,
    fromEndfTab2,
    multiplicityAxes,
)

from .decode import aceLabels
from .secondaries import MONTE_CARLO_LABEL

__all__ = ["attachAcePhotons", "photonProductionTotal"]

#: ACE stores energies in MeV and densities per MeV; the model works in eV.
MEV_TO_EV = 1.0e6


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def _uniqueLabel(pid: str, taken: set) -> str:
    """FUDGE's ``uniqueLabel``: ``photon``, ``photon__a`` … ``photon__z``, ``photon__aa``.

    The same sequence the ENDF adapter writes, so the two decoders name the
    photons of a reaction alike. Re-implemented here because the ENDF adapter
    is not importable from outside its package.
    """
    if pid not in taken:
        return pid
    n = 0
    while True:
        letters, k = "", n + 1
        while k:
            k, r = divmod(k - 1, 26)
            letters = "abcdefghijklmnopqrstuvwxyz"[r] + letters
        label = f"{pid}__{letters}"
        if label not in taken:
            return label
        n += 1


def _tab1(x, y, nbt, codes, axes, label=None):
    """An ACE TAB1 → an :class:`XYs1d`, or a :class:`Regions1d` when NR > 1.

    NR = 0 is the ACE way of saying lin-lin, and is read as one lin-lin region.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pairs = [(int(b), int(c)) for b, c in zip(list(nbt), list(codes))]
    if len(pairs) <= 1:
        code = pairs[0][1] if pairs else 2
        if code == 2:
            return XYs1d(xs=x, ys=y, interpolation=Interpolation.linlin,
                         axes=axes, label=label)
        pairs = [(int(x.size), code)]
    function = Regions1d.fromEndfRegions(x, y, pairs, axes=axes, label=label)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
        function.axes, function.label = axes, label
    return function


def _domainOf(function) -> Tuple[float, float]:
    return float(function.domainMin), float(function.domainMax)


def photonProductionTotal(ace) -> Optional[XYs1d]:
    """GPD as an :class:`XYs1d` in eV and barns, labelled ``heated_<T>``, or ``None``."""
    total = ace.photon_production_total() if hasattr(ace, "photon_production_total") else None
    if total is None:
        return None
    energies, sigma = total
    label = aceLabels(ace)[0]
    return XYs1d(xs=energies * MEV_TO_EV, ys=sigma, interpolation=Interpolation.linlin,
                 axes=crossSectionAxes(), label=label)


# ---------------------------------------------------------------------------
# Angle and energy
# ---------------------------------------------------------------------------

def _angular(distribution, mtrp: int, report: ConversionReport):
    """One ANDP entry → ``Isotropic2d`` or an ``XYs2d`` of P(mu|E), or ``None``.

    A 32-equiprobable-bin table is a histogram: each bin holds 1/32 over its
    width, written as a flat ``XYs1d`` on the 33 edges (the last value repeats
    the last bin, as a flat function's final point carries no interval).
    """
    from kika.ace.classes.angular_distribution.distributions import (
        EquiprobableAngularDistribution, IsotropicAngularDistribution,
        TabulatedAngularDistribution)

    if distribution is None or isinstance(distribution, IsotropicAngularDistribution):
        return Isotropic2d(productFrame=Frame.lab)

    energies = np.asarray(distribution._energies, dtype=float) * MEV_TO_EV
    functions = []
    if isinstance(distribution, EquiprobableAngularDistribution):
        for k, edges in enumerate(distribution._cosine_bins):
            edges = np.asarray(edges, dtype=float)
            widths = np.diff(edges)
            density = np.where(widths > 0, (1.0 / widths.size) / np.where(widths > 0, widths, 1.0), 0.0)
            ys = np.append(density, density[-1])
            function = XYs1d(xs=edges, ys=ys, interpolation=Interpolation.flat)
            function.outerDomainValue, function.index = float(energies[k]), k
            functions.append(function)
    elif isinstance(distribution, TabulatedAngularDistribution):
        for k, (mu, pdf) in enumerate(zip(distribution._cosine_grid, distribution._pdf)):
            jj = int(distribution.interpolation[k]) if k < len(distribution.interpolation) else 2
            function = XYs1d(xs=np.asarray(mu, dtype=float), ys=np.asarray(pdf, dtype=float),
                             interpolation=Interpolation.linlin if jj == 2 else Interpolation.flat)
            function.outerDomainValue, function.index = float(energies[k]), k
            functions.append(function)
    else:
        report.unsupportedNode(
            f"MTRP {mtrp}: the photon angular distribution is a "
            f"{type(distribution).__name__}, which has no photon meaning in ACE; "
            f"the product's angular half is left empty")
        return None
    return fromEndfTab2(functions, [(len(functions), 2)], axes=angularAxes())


def _continuum(law4, mtrp: int, report: ConversionReport):
    """DLWP law 4 → an ``XYs2d`` of P(E'|E), eV and 1/eV, or ``None``.

    A table that mixes discrete lines into its continuum (``N_D > 0``) is
    refused: the lines would need a form of their own beside the continuum and
    dropping them would leave the rest unnormalised.
    """
    if any(int(t.get("n_discrete", 0) or 0) > 0 for t in law4.distributions):
        report.unsupportedNode(
            f"MTRP {mtrp}: the law-4 photon spectrum mixes discrete lines into "
            f"its continuum (N_D > 0), which the model has no single form for; "
            f"the product's energy half is left empty")
        return None
    functions = []
    for k, (ein, table) in enumerate(zip(law4.incident_energies, law4.distributions)):
        intt = int(table.get("intt", 2)) % 10
        function = XYs1d(
            xs=np.asarray(table["e_out"], dtype=float) * MEV_TO_EV,
            ys=np.asarray(table["pdf"], dtype=float) / MEV_TO_EV,
            interpolation=Interpolation.flat if intt == 1 else Interpolation.linlin)
        function.outerDomainValue, function.index = float(ein) * MEV_TO_EV, k
        functions.append(function)
    pairs = list(zip(list(law4.nbt), list(law4.interp))) or [(len(functions), 2)]
    return fromEndfTab2(functions, pairs, axes=energyAxes())


def _energy(laws, multiplicity, mtrp: int, report: ConversionReport):
    """The DLWP entry of one photon → its energy form, or ``None`` (reported)."""
    from kika.ace.classes.energy_distribution.distributions.discrete import \
        DiscreteEnergyDistribution
    from kika.ace.classes.energy_distribution.distributions.tabular import \
        ContinuousTabularDistribution

    laws = list(laws or [])
    if not laws:
        report.lost(f"MTRP {mtrp}: no DLWP entry, so this photon has no energy distribution")
        return None
    if len(laws) > 1:
        report.unsupportedNode(
            f"MTRP {mtrp}: {len(laws)} energy laws with incident-energy "
            f"probabilities; the model's uncorrelated form takes one, so the "
            f"product's energy half is left empty")
        return None
    law = laws[0]
    if isinstance(law, DiscreteEnergyDistribution):
        lo, hi = _domainOf(multiplicity)
        kind = PrimaryGamma if int(law.lp) == 2 else DiscreteGamma
        return kind(value=float(law.eg) * MEV_TO_EV, domainMin=lo, domainMax=hi,
                    axes=energyAxes())
    if isinstance(law, ContinuousTabularDistribution):
        return _continuum(law, mtrp, report)
    report.unsupportedNode(
        f"MTRP {mtrp}: the photon energy distribution is ACE law {law.law} "
        f"({law.law_name}), which this decoder does not convert; the product's "
        f"energy half is left empty")
    return None


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def _yieldMultiplicity(sigp, label: str) -> Multiplicity:
    """MFTYPE 12/16: the yield *is* the multiplicity, on its own grid."""
    function = _tab1(np.asarray(sigp.energies, dtype=float) * MEV_TO_EV, sigp.yields,
                     sigp.interpolation_bounds, sigp.interpolation_schemes,
                     axes=multiplicityAxes(), label=label)
    return Multiplicity(form=function)


def _directSigma(ace, sigp) -> Tuple[np.ndarray, np.ndarray]:
    """MFTYPE 13: σ_γ in eV and barns, on the union-grid slice it starts at (IE)."""
    grid = np.asarray(ace.esz_block.energies, dtype=float)
    start = max(int(sigp.energy_grid_index) - 1, 0)
    values = np.asarray(sigp.cross_sections, dtype=float)
    return grid[start:start + values.size] * MEV_TO_EV, values


def _divide(x, sigmaGamma, sigmaForm) -> Tuple[np.ndarray, int]:
    """σ_γ / σ at σ_γ's points; a point where σ = 0 gets 0 and is counted."""
    sigma = np.interp(x, np.asarray(sigmaForm.xs, dtype=float),
                      np.asarray(sigmaForm.ys, dtype=float))
    zero = sigma == 0.0
    return np.where(zero, 0.0, sigmaGamma / np.where(zero, 1.0, sigma)), int(zero.sum())


# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------

def _isDiscreteLevel(mt: int) -> bool:
    """MT51-90 and the (n,p) … (n,α) level series, without their continua (x49)."""
    if 51 <= mt <= 90:
        return True
    return any(start <= mt <= start + 48 for start in (600, 650, 700, 750, 800))


def _orphanSigmas(ace, suite, entries, label: str):
    """``{MT: (x, Σσ_γ, count)}`` for the MFTYPE 13 photons whose MT the table has no σ for.

    Several photons can name the same missing MT (six on MT 28 in ENDF/B-VIII.1
    N-14). Their orphan carries the sum of their σ_γ on the union of their
    grid slices, and each photon its share of it, as the ENDF adapter states
    MF13's partials against its total; one orphan holding the first photon's
    σ_γ would give the others that photon's numbers.
    """
    from kika.ace.classes.photon_production_xs import DirectCrossSection

    grid = np.asarray(ace.esz_block.energies, dtype=float) * MEV_TO_EV
    groups = {}
    for mtrp, entry in entries:
        if not isinstance(entry, DirectCrossSection):
            continue
        mt = int(mtrp) // 1000
        host = suite.findReactionByENDF_MT(mt) if mt else None
        if host is not None and host.crossSection.get(label) is not None:
            continue
        start = max(int(entry.energy_grid_index) - 1, 0)
        values = np.asarray(entry.cross_sections, dtype=float)
        lo, hi, total, count = groups.get(mt, (start, start + values.size, np.zeros(grid.size), 0))
        total[start:start + values.size] += values
        groups[mt] = (min(lo, start), max(hi, start + values.size), total, count + 1)
    return {mt: (grid[lo:hi], total[lo:hi], count) for mt, (lo, hi, total, count) in groups.items()}


def _orphan(suite, mt: int, x, sigmaGamma, provenance, label: str):
    """The orphan product an MFTYPE 13 with no reaction to divide by hangs on."""
    for orphan in suite.orphanProducts:
        if orphan.ENDF_MT == mt:
            return orphan
    orphan = Reaction(id=ReactionId(label=f"Orphan product {len(suite.orphanProducts) + 1}",
                                    ENDF_MT=int(mt)),
                      outputChannel=OutputChannel(genre="NBody"), provenance=provenance)
    # σ_γ is on the union grid, so it is the heated style's, like every other
    # cross section in an ACE-derived suite.
    orphan.crossSection[label] = XYs1d(
        xs=x, ys=sigmaGamma, interpolation=Interpolation.linlin,
        axes=crossSectionAxes(), label=label)
    suite.orphanProducts.append(orphan)
    return orphan


def _monteCarlo(ace, laws, angular, mtrp: int, report: ConversionReport):
    """The ``MonteCarlo`` form of one photon: its DLWP chain with its ANDP angles."""
    from .secondaries import angularForm, lawForm

    if not laws:
        return None
    mu = angularForm(angular) if angular is not None else Isotropic2d()
    awr = float(getattr(getattr(ace, "header", None), "atomic_weight_ratio", 0.0) or 0.0)
    if len(laws) > 1:
        report.unsupportedNode(f"MTRP {mtrp}: a chain of {len(laws)} photon laws; only "
                               f"chains of one have a Monte Carlo form here")
        return None
    try:
        return lawForm(laws[0], mu, Frame.lab, awr, report, f"DLWP MTRP {mtrp}")
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        report.unsupportedNode(f"MTRP {mtrp}: no Monte Carlo form for its DLWP law ({e})")
        return None


def attachAcePhotons(ace, suite, report: Optional[ConversionReport] = None,
                     provenance=None) -> ConversionReport:
    """Put every MTRP entry of *ace* on the photon products of its reaction.

    Runs after the reactions are built, because MFTYPE 12/16 hang off reaction
    MTMULT and MFTYPE 13 divides by its heated σ. Returns the report; never
    raises on an unsupported law — that photon is reported and either kept
    with the half that could be read or, when it has no host, left out.
    """
    report = report if report is not None else ConversionReport()
    sigp = getattr(ace, "photon_production_xs", None)
    if sigp is None or not getattr(sigp, "cross_sections", None):
        return report

    from kika.ace.classes.photon_production_xs import (DirectCrossSection,
                                                        YieldBasedCrossSection)

    label = aceLabels(ace)[0]
    angular = getattr(getattr(ace, "angular_distributions", None), "photon_production", None) or {}
    energy = getattr(getattr(ace, "energy_distributions", None), "photon_production", None) or {}

    decoded, divided, zeros, orphaned, levels = 0, [], 0, [], 0
    orphanSigmas = _orphanSigmas(ace, suite, list(sigp.cross_sections.items()), label)
    for mtrp, entry in sigp.cross_sections.items():
        mtrp = int(mtrp)
        unrepresented = None
        if isinstance(entry, YieldBasedCrossSection):
            mt = int(entry.mtmult)
            host = suite.findReactionByENDF_MT(mt)
            if host is None or not any(host is r for r in suite.reactions):
                report.unsupportedNode(
                    f"MTRP {mtrp}: a yield on MT{mt}, which this suite has no "
                    f"reaction for, so there is no σ for the photon to multiply "
                    f"and it is left out")
                continue
            multiplicity = _yieldMultiplicity(entry, label)
        elif isinstance(entry, DirectCrossSection):
            mt = mtrp // 1000
            x, sigmaGamma = _directSigma(ace, entry)
            host = suite.findReactionByENDF_MT(mt) if mt else None
            sigmaForm = host.crossSection.get(label) if host is not None else None
            if sigmaForm is not None:
                y, n = _divide(x, sigmaGamma, sigmaForm)
                zeros += n
                # σ_γ where σ_MT = 0 has no multiplicity to be; kept for the ACE writer.
                hostSigma = np.interp(x, np.asarray(sigmaForm.xs, dtype=float),
                                      np.asarray(sigmaForm.ys, dtype=float))
                lostAt = np.flatnonzero((hostSigma == 0.0) & (sigmaGamma != 0.0))
                if lostAt.size:
                    unrepresented = {int(i): float(sigmaGamma[i]) for i in lostAt}
                divided.append(mtrp)
                form = XYs1d(xs=x, ys=y, interpolation=Interpolation.linlin,
                             axes=multiplicityAxes(), label=label)
            else:
                xOrphan, sigmaOrphan, count = orphanSigmas[mt]
                host = _orphan(suite, mt, xOrphan, sigmaOrphan, provenance, label)
                orphaned.append(mtrp)
                if count == 1:
                    form = Constant1d(constant=1.0, domainMin_=float(x[0]), domainMax_=float(x[-1]),
                                      axes=multiplicityAxes(), label=label)
                else:
                    y, _ = _divide(x, sigmaGamma, host.crossSection[label])
                    form = XYs1d(xs=x, ys=y, interpolation=Interpolation.linlin,
                                 axes=multiplicityAxes(), label=label)
            multiplicity = Multiplicity(form=form)
        else:
            report.unsupportedNode(
                f"MTRP {mtrp}: SIGP entry of type {type(entry).__name__} "
                f"(MFTYPE {getattr(entry, 'mftype', '?')}) is not a photon "
                f"production form this decoder knows; left out")
            continue

        energyForm = _energy(energy.get(mtrp), multiplicity.form, mtrp, report)
        angularForm = _angular(angular.get(mtrp), mtrp, report)
        distribution = Distribution()
        distribution[label] = Uncorrelated(angular=angularForm, energy=energyForm,
                                           label=label, productFrame=Frame.lab)
        # The Monte Carlo form beside it: the DLWP laws and the
        # ANDP tables as they are sampled, which the HDF5 and ACE writers take.
        mc = _monteCarlo(ace, energy.get(mtrp), angular.get(mtrp), mtrp, report)
        if mc is not None:
            distribution[MONTE_CARLO_LABEL] = mc

        channel = host.outputChannel
        taken = {p.label for p in channel.products}
        product = Product(pid="photon", label=_uniqueLabel("photon", taken),
                          multiplicity=multiplicity, distribution=distribution)
        # What the ACE writer needs back: the entry's number and form, and the
        # law numbers of its DLWP chain, which the ACE writer gives back.
        laws = list(energy.get(mtrp) or [])
        from .secondaries import chainAttributes

        attributes = {"MTRP": mtrp, "MFTYPE": int(entry.mftype), **chainAttributes(laws)}
        if unrepresented:
            attributes["sigmaGammaWhereSigmaIsZero"] = unrepresented
        if len(laws) == 1 and hasattr(laws[0], "applicability_energies"):
            # The Monte Carlo form of a one-law chain has no weight, so its P(E) goes here.
            attributes["applicability"] = {
                "energies": [float(e) for e in laws[0].applicability_energies],
                "probabilities": [float(v) for v in laws[0].applicability_probabilities],
                "nbt": [int(v) for v in laws[0].applicability_nbt],
                "interp": [int(v) for v in laws[0].applicability_interp]}
        product.provenance = AceProvenance(attributes=attributes)
        channel.products.products.append(product)
        decoded += 1
        if _isDiscreteLevel(mt):
            levels += 1

    total = photonProductionTotal(ace)
    if total is not None:
        # Not a GNDS node: a reactionSuite has no slot for a total photon
        # production σ, and a `production` reaction would put a second copy of
        # every photon in front of readers that sum products. An attribute,
        # like `suite.report`, so GPD stays reachable.
        suite.photonProductionTotal = total
    gpd = getattr(ace, "photon_production_data", None)
    if gpd is not None and getattr(gpd, "has_outgoing_energies", False):
        report.unsupportedNode(
            "the GPD block's obsolete 30x20 matrix of equiprobable photon "
            "energies is not decoded; the per-MTRP DLWP spectra are")

    if divided:
        report.approximated(
            f"{len(divided)} photon(s) are given by ACE as production cross "
            f"sections (MFTYPE 13) and the model as a multiplicity, so each is "
            f"σ_γ/σ_MT on σ_γ's union-grid points, the ENDF adapter's MF13 "
            f"convention" + (f"; σ_MT = 0 at {zeros} point(s), written as 0"
                             if zeros else ""))
    if orphaned:
        report.approximated(
            f"{len(orphaned)} MFTYPE 13 photon(s) name an MT this table has no "
            f"cross section for (MTRP {orphaned[:5]}...); they sit on an orphan "
            f"product per MT whose cross section is the sum of their σ_γ, each with "
            f"its share of it as multiplicity (1 when it is alone)")
    if levels:
        report.lost(
            f"{levels} photon line(s) come from discrete-level reactions: ACE "
            f"has no level scheme (ACER expanded every MF12 LO=2 cascade into "
            f"lines), so they sit on the reaction's output channel as discrete "
            f"gammas and PoPs carries no decayData for the levels")
    return report

"""ACE → :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

**ACE is processed data, and the model has to say so.** An ENDF evaluation and
an ACE file are not two formats holding the same thing: ACE has already been
reconstructed from resonance parameters, Doppler-broadened to a temperature and
put on a unionised grid. GNDS §9 exists to record exactly that, through a chain
of styles linked by ``derivedFrom``. So this decoder does not produce an
``eval``-labelled cross section the way the ENDF decoder does — it produces a
``heated_<T>`` one, hanging off the ``evaluated`` style the file *refers to and
does not contain*.

**The labels are the processed suite's.** ``heated_<T>``,
``URR_<T>`` and ``MonteCarlo``, with ``<T>`` named from kT as OpenMC names its
temperature groups (``heated_294K``): the labels the HDF5 decoder and kika's
own processing give, and the ones the HDF5 and ACE encoders require. A suite
read from ACE is therefore a processed suite like any other, and goes to HDF5
or back to ACE without relabelling.

**ACE carries Q values.** ACE's LQR block carries one Q per reaction, in MeV, aligned with MTR, and it is correct: Fe-56
capture reads +7.646 MeV, (n,2n) −11.197 MeV, (n,p) −2.913 MeV. What ACE lacks
is **QM** — the mass-difference Q — and **LR**. So the Q = 0 defect on the ACE
side was never "the format has no Q"; it was ``CrossSection.from_ace`` not
reading a block that is there. This decoder reads it.

**Not gated in CI.** No ACE table is committed (a real one is 16 MB), so
every test here is ``@pytest.mark.tape`` and runs only where an ACE file is
reachable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from kika._constants import mf3SumComponents

from kika.nuclear_data.model import (
    AceProvenance,
    ConversionReport,
    CrossSection,
    Evaluated,
    Frame,
    Heated,
    Interpolation,
    MonteCarlo_cdf,
    Nuclide,
    OutputChannel,
    PhysicalQuantity,
    PoPs,
    Q,
    Reaction,
    ReactionId,
    ReactionSuite,
    URR_probabilityTables,
    URR_probabilityTables1d,
    XYs1d,
    crossSectionAxes,
    pidFromZA,
)

from .secondaries import MONTE_CARLO_LABEL

__all__ = ["decodeAce", "aceStyles", "aceLabels", "qValuesByMT",
           "EVALUATED_LABEL", "MONTE_CARLO_LABEL"]

#: Style labels this decoder assigns: the processed suite's, the same
#: as :mod:`kika.hdf5.names`. GNDS §9 fixes the node *names* and leaves the
#: labels to the writer; the per-temperature ones come from :func:`aceLabels`.
EVALUATED_LABEL = "eval"

#: ACE stores energies and Q values in MeV; the model works in eV, as ENDF does.
MEV_TO_EV = 1.0e6


def _kTeV(ace) -> Optional[float]:
    """ACE's header temperature is **kT in MeV**, not a temperature in kelvin.

    ``Header.temperature``'s inline comment says "Temperature in K" and its
    class docstring says MeV. The docstring is right — Fe-56 at ``2.53e-08``
    is 293.6 K. Returned in eV; ``None`` when the header has none. A kT of 0
    is a 0 K table, not a missing temperature.
    """
    kT = getattr(getattr(ace, "header", None), "temperature", None)
    return float(kT) * MEV_TO_EV if kT is not None else None


def aceLabels(ace):
    """``(heated, urr)``: the labels of this table's temperature, ``heated_294K``/``URR_294K``.

    Named from kT exactly as OpenMC names a temperature group
    (:func:`kika.hdf5.names.temperatureName`), because the HDF5 encoder checks
    that the name and the stated kT agree. A table with no header temperature
    is named ``0K``.
    """
    from kika.hdf5.names import heatedLabel, temperatureName, urrLabel

    kT = _kTeV(ace)
    name = temperatureName(kT if kT is not None else 0.0)
    return heatedLabel(name), urrLabel(name)


def qValuesByMT(ace, report: Optional[ConversionReport] = None) -> Dict[int, float]:
    """``{MT: Q in eV}`` from the LQR block, aligned with MTR.

    LQR is parallel to MTR — one Q per *non-elastic* reaction, in the same order
    — so the mapping is positional and there is no MT written next to a Q. MT 2
    is absent from MTR because elastic scattering has Q = 0 by definition, which
    is knowledge and not a default, so it is filled in here.
    """
    report = report if report is not None else ConversionReport()

    qBlock = getattr(ace, "q_values", None)
    reactions = getattr(ace, "reaction_mt_data", None)
    if qBlock is None or not getattr(qBlock, "has_q_values", False) or reactions is None:
        report.lost("this ACE file carries no LQR block, so no reaction has a Q value")
        return {}

    mtNumbers = list(reactions.get_mt_values("neutron"))
    qValues = qBlock.get_all_q_values()
    if len(mtNumbers) != len(qValues):
        report.warn(
            f"MTR has {len(mtNumbers)} entries and LQR has {len(qValues)}; they are "
            f"positionally aligned by definition, so the shorter one is used and "
            f"the rest have no Q"
        )

    byMT = {int(mt): float(q) * MEV_TO_EV for mt, q in zip(mtNumbers, qValues)}
    byMT.setdefault(2, 0.0)
    return byMT


def aceStyles(ace, report: Optional[ConversionReport] = None):
    """The ``derivedFrom`` chain an ACE file implies, and nothing more.

    ``evaluated`` → ``heated_<T>`` [→ ``URR_<T>``], and ``evaluated`` →
    ``MonteCarlo``: the chain the HDF5 decoder builds for one temperature.

    **``crossSectionReconstructed`` is deliberately not in it.** A continuous-energy
    ACE has almost certainly been through RECONR, but the file does not say so,
    and a style chain is a claim about provenance. Inventing a step because it is
    probable is how a processed file starts asserting things nobody checked.

    The ``evaluated`` style *is* declared, empty: §9 makes it the root every
    ``derivedFrom`` chain resolves to, and ACE refers to an evaluation it does
    not contain. That absence is reported rather than hidden.
    """
    report = report if report is not None else ConversionReport()
    styles = []

    styles.append(Evaluated(label=EVALUATED_LABEL,
                            library=(getattr(ace.header, "comment", None) or "").strip() or None))
    report.lost(
        "ACE refers to an evaluation it does not contain, so the 'eval' style is "
        "declared with no data behind it; nothing in this reactionSuite carries an "
        "eval-labelled form"
    )

    kT = _kTeV(ace)
    if kT is None:
        report.warn("this ACE file has no header temperature; its heated style is named 0K "
                    "and states none")
    heated, urrLabel = aceLabels(ace)
    styles.append(Heated(
        label=heated, derivedFrom=EVALUATED_LABEL,
        temperature=PhysicalQuantity(value=kT, unit="eV/k") if kT is not None else None,
    ))

    urr = getattr(ace, "unresolved_resonance", None)
    if urr is not None and getattr(urr, "has_data", False):
        styles.append(URR_probabilityTables(label=urrLabel, derivedFrom=heated))
    styles.append(MonteCarlo_cdf(label=MONTE_CARLO_LABEL, derivedFrom=EVALUATED_LABEL))

    return styles, report


def decodeAce(ace, report: Optional[ConversionReport] = None):
    """A parsed :class:`~kika.ace.classes.ace.Ace` → a :class:`ReactionSuite`.

    The cross sections land under the ``heated_<T>`` style, because that is
    what they are: values on ACE's unionised grid at the file's temperature.
    Nothing is labelled ``eval``.
    """
    report = report if report is not None else ConversionReport()

    styles, report = aceStyles(ace, report)
    provenance = AceProvenance(
        zaid=getattr(ace.header, "zaid", None),
        extension=getattr(ace.header, "extension", None),
        awr=getattr(ace.header, "atomic_weight_ratio", None),
        comment=getattr(ace.header, "comment", None),
        date=getattr(ace.header, "date", None),
        matid=getattr(ace.header, "matid", None),
        formatVersion=getattr(ace.header, "format_version", None),
        temperatureMeV=getattr(ace.header, "temperature", None),
    )
    # The order NJOY gave MTR and YP, which is not ascending (MT 4 goes last
    # when a photon yield needs it): the ACE writer gives it back.
    provenance.attributes["MTR"] = [int(v) for v in ace.reaction_mt_data.incident_neutron]
    provenance.attributes["MTRP"] = [int(v) for v in ace.reaction_mt_data.photon_production]
    # NXS(9..16): zero in Lib81, Z and A in NXS(10)/(11) from NJOY 2016 and JEFF-4.0.
    nxs = list(getattr(ace.header, "nxs_array", None) or [])
    if len(nxs) >= 17:
        provenance.attributes["NXS"] = [int(v) for v in nxs[9:17]]
        # NXS(2) is the ground state's ZA even when the ZAID names an isomer (27058 for 27458)
        provenance.attributes["ZA"] = int(nxs[2])
    yp = getattr(ace, "photon_yield_multipliers", None)
    if yp is not None:
        provenance.attributes["YP"] = [int(v) for v in yp.multiplier_mts]

    za = int(provenance.zaid) if provenance.zaid is not None else 0
    pops = PoPs()
    if za:
        pops.add(Nuclide(id=pidFromZA(za), Z=za // 1000, A=za % 1000))

    suite = ReactionSuite(
        evaluation=(provenance.comment or "").strip() or "",
        projectile="n",
        target=next(iter(pops.particles), "unknown"),
        projectileFrame=Frame.lab,
    )
    suite.PoPs = pops
    suite.provenance = provenance
    for style in styles:
        suite.styles.add(style)

    heated, urrLabel = aceLabels(ace)
    qByMT = qValuesByMT(ace, report)
    for mt in _mtNumbers(ace):
        reaction, report = decodeAceReaction(ace, mt, qByMT, provenance, report, label=heated)
        if reaction is not None:
            suite.reactions.append(reaction)

    _attachURR(ace, suite, report, urrLabel)
    # AND, DLW, TYR, NU/DNU, BDD and DNED become the products.
    from .secondaries import attachAceParticles, attachAceSecondaries

    attachAceSecondaries(ace, suite, report)
    # PTYPE/NTRO/IXS: the light charged particles.
    attachAceParticles(ace, suite, report)

    # What ACE genuinely does not have. Stated per item, because "ACE carries no
    # Q values" was the previous formulation and it is false.
    report.unsupportedNode(
        "ACE has no QM (the mass-difference Q) and no LR (the complex-breakup "
        "flag): both are ENDF MF3 fields with no ACE counterpart. QI *is* present, "
        "in the LQR block, and is decoded."
    )
    report.unsupportedNode(
        "ACE carries no covariances at all, so there is no covarianceSuite to "
        "build from one"
    )
    # The photon production blocks (MTRP/SIGP/LANDP/ANDP/DLWP)
    # become photon products, the shape the ENDF adapter gives MF12-15.
    from .photons import attachAcePhotons

    attachAcePhotons(ace, suite, report, provenance=provenance)
    _place(ace, suite)
    _attachHeating(ace, suite, provenance, heated)
    # Both ways to the same object: the tuple is unchanged, and the
    # attribute is the one that survives `suite, _ = ...`. §11.4.
    suite.report = report
    return suite, report


def _mtNumbers(ace) -> List[int]:
    """Every MT the file exposes a cross section for, composites included.

    Composites (MT 4, 18, 101, ...) are summed on demand by
    ``CrossSectionData._get_or_compute_reaction``, so they are real cross
    sections with no Q of their own — which is correct: a sum of reactions with
    different Q values has none.
    """
    from kika._constants import MT_COMPOSITE_ORDER

    direct = list(getattr(ace.cross_section, "mt_numbers", []) or [])
    composites = [mt for mt in MT_COMPOSITE_ORDER if mt not in direct]
    return sorted(set(direct)) + [mt for mt in composites]


def decodeAceReaction(ace, mt: int, qByMT: Dict[int, float],
                      provenance: AceProvenance, report: ConversionReport, label: Optional[str] = None):
    """One MT → a :class:`Reaction` with a ``heated_<T>``-labelled cross section.

    ``label`` is the table's heated label (:func:`aceLabels`), computed from the
    header when not given.

    An ``XYs1d`` rather than the ``regions1d`` the ENDF decoder builds: ACE has
    no ``(NBT, INT)`` regions, it is lin-lin on one union grid, and a
    single-region ``regions1d`` would assert a structure the file does not have.
    """
    reactionData = ace.cross_section._get_or_compute_reaction(mt)
    if reactionData is None:
        return None, report

    energies = np.asarray(reactionData.energies, dtype=float) * MEV_TO_EV
    values = np.asarray(reactionData.xs_values, dtype=float)
    if mt == 444:
        # damage-energy production: ACE states MeV·b, the model eV·b (as OpenMC)
        values = values * MEV_TO_EV

    label = label if label is not None else aceLabels(ace)[0]
    crossSection = CrossSection()
    crossSection[label] = XYs1d(
        xs=energies, ys=values, interpolation=Interpolation.linlin,
        axes=crossSectionAxes(), label=label,
    )

    channel = OutputChannel(Q=Q(value=qByMT.get(mt), unit="eV"))
    if mt not in qByMT:
        report.lost(
            f"MT{mt}: no entry in ACE's LQR block, so this reaction's Q is unknown "
            f"(it is a composite or a summed reaction if it is one of "
            f"4, 18, 101, 27, 3)"
        )

    return Reaction(
        id=ReactionId(label=f"MT{mt}", ENDF_MT=mt),
        crossSection=crossSection,
        outputChannel=channel,
        provenance=provenance,
    ), report


#: Redundant MTs that are response functions of the processed library, not reactions
#: (gas production, KERMA, damage energy): ``suite.productions``, as the HDF5 decoder puts them.
RESPONSE_MTS = (203, 204, 205, 206, 207, 301, 444, 901)


def _place(ace, suite: ReactionSuite) -> None:
    """Each reaction into the container its role puts it in, as the HDF5 decoder does.

    A reaction is redundant when the MF3 sum rules make it the sum of others
    the file carries (OpenMC's ``get_reaction_components``): MT 103 beside
    MT 600-649, MT 18 beside its chances, and the sums kika itself adds
    (MT 3, 4, 27, 101...). Redundant sums go to ``suite.sums`` as
    :class:`CrossSectionSum` with their summands, the response functions to
    ``suite.productions``, and the fission chances, when MT 18 is there, to
    ``suite.fissionComponents``. Left in ``reactions``, a sum would be counted
    twice by anything that adds the reactions up (OpenMC's C++ rebuilds the
    total that way).
    """
    from kika.nuclear_data.model import Add, CrossSectionSum

    direct = {int(m) for m in getattr(ace.reaction_mt_data, "incident_neutron", [])} | {2}
    present = {r.ENDF_MT for r in suite.reactions}
    keep = []
    for reaction in list(suite.reactions):
        mt = reaction.ENDF_MT
        if mt in RESPONSE_MTS:
            suite.productions.append(reaction)
            continue
        components = mf3SumComponents(mt, direct)
        if (components and components != [mt]) or mt not in direct:
            total = CrossSectionSum(id=reaction.id, crossSection=reaction.crossSection,
                                    outputChannel=reaction.outputChannel, provenance=reaction.provenance)
            for c in components:
                if c in present:
                    total.summands.append(Add(
                        href=f"/reactionSuite/reactions/reaction[@label='MT{c}']/crossSection"))
            suite.sums.append(total)
            continue
        if mt in (19, 20, 21, 38) and 18 in present:
            suite.fissionComponents.append(reaction)
            continue
        keep.append(reaction)
    suite.reactions = type(suite.reactions)(keep)


def _attachHeating(ace, suite: ReactionSuite, provenance, label: str) -> None:
    """ESZ's heating numbers as the KERMA MT301, in eV·b: number × σ_t, as OpenMC builds it.

    ACER stores the heating as an average energy deposited per collision
    (KERMA/σ_t, MeV); the model and OpenMC store the KERMA itself. Skipped
    when the file has MT301 in MTR already.
    """
    esz = getattr(ace, "esz_block", None)
    if esz is None or len(esz.heating_numbers) == 0 or suite.findReactionByENDF_MT(301) is not None:
        return
    energies = np.asarray(esz.energies, dtype=float) * MEV_TO_EV
    kerma = np.asarray(esz.heating_numbers, dtype=float) * MEV_TO_EV * np.asarray(esz.total_xs, dtype=float)
    crossSection = CrossSection()
    crossSection[label] = XYs1d(xs=energies, ys=kerma, interpolation=Interpolation.linlin,
                                axes=crossSectionAxes(), label=label)
    suite.productions.append(Reaction(id=ReactionId(label="MT301", ENDF_MT=301), crossSection=crossSection,
                                      outputChannel=OutputChannel(), provenance=provenance))


def _attachURR(ace, suite: ReactionSuite, report: ConversionReport, label: str) -> None:
    """The URR probability tables, on the reactions they apply to.

    One :class:`ProbabilityTables` (the HDF5 decoder's node: energies in eV, the
    (N, 6, M) table with the heating row in eV, the INT/ILF/IOA/IFF flags),
    held by a ``URR_probabilityTables1d`` form of elastic, fission and capture,
    as OpenMC's ``urr`` group is.
    """
    from kika.nuclear_data.model import ProbabilityTables

    urr = getattr(ace, "unresolved_resonance", None)
    if urr is None or not getattr(urr, "has_data", False):
        return
    m = urr.table_length
    table = np.empty((len(urr.energies), 6, m))
    for i, t in enumerate(urr.tables):
        table[i] = [t.cumulative_probabilities, t.total_xs, t.elastic_xs, t.fission_xs, t.capture_xs,
                    np.asarray(t.heating_numbers, dtype=float) * MEV_TO_EV]
    tables = ProbabilityTables(energies=np.asarray(urr.energies, dtype=float) * MEV_TO_EV, table=table,
                               interpolation=int(urr.interpolation), inelastic=int(urr.inelastic_flag),
                               absorption=int(urr.other_absorption_flag),
                               multiplySmooth=int(urr.factors_flag) == 1)
    for mt in (2, 18, 102):
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is not None:
            reaction.crossSection[label] = URR_probabilityTables1d(label=label, table=tables)

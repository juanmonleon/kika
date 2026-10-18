"""MF8/MF9/MF10 of a reaction ↔ ``reactionSuite/productions`` (roadmap E6).

An incident-particle evaluation says which radioactive nuclides a reaction makes
in three files: **MF8** lists the final states and, through ``LMF``, says where
their yield is; **MF9** gives a multiplicity per state, to be multiplied by the
reaction's MF3 cross section; **MF10** gives the production cross section per
state directly. GNDS has one node for all three: a ``production`` (§14.1.1)
holding a cross section and an output channel with the one product made.

**The mapping.** One ``production`` per final state, labelled ``MT<mt> <pid>``:

* MF9 (and MF8 ``LMF=3``, which is MF9 with a multiplicity of one): the cross
  section is a ``reference`` to the reaction's own, so a perturbed or
  reconstructed σ reaches the production without being copied; the product's
  multiplicity is the MF9 table (or the constant 1).
* MF10: the cross section is the MF10 table; the multiplicity is the constant 1.
* The channel's Q is the state's ``QI``. ``QM`` has no GNDS node — the
  excitation ``QM - QI`` goes to the product's PoPs level instead — and is kept
  in the production's provenance so the record comes back as it was read.

**Where this departs from FUDGE** (``ENDF_ITYPE_0.py``, ``readMF8``), on purpose:

* MF9 and MF10 are read **without MF8**. Each TAB1 names its product
  (``IZAP``, ``LFS``) and its level (``QM - QI``), so MF8 is an index to them,
  not their key. FUDGE only reaches MF9/MF10 through MF8, and drops them when MF8
  is missing or disagrees.
* The label does not encode the *form* of the data. FUDGE writes
  ``m(E)*Am242_e2 [Am241(n,g)]`` for a tabulated multiplicity and
  ``Am242_e2 [...]`` for a constant one, so editing the multiplicity would
  rename the node every covariance points at. ``MT102 Am242_e2`` is kika's
  ``MT<n>`` reaction label plus the product, and stays put.
* The Q and the constant multiplicity span the **table's** domain, not the
  whole projectile domain: that is what the file states.
* The ``metaStable`` aliases are numbered by level within each nuclide, not in
  the order MT sections happen to be read, so ``Am242_m1`` does not depend on
  which reaction reached the isomer first.
* MF8 itself is **kept**, and re-emitted verbatim while it still agrees with the
  productions: its decay-chain LISTs (``NO=0``) and ``LMF=6`` states have no
  node, and rebuilding a section the model did not change would lose them. When
  the productions change, or the suite never saw ENDF, MF8 is rebuilt from them
  with ``NO=1`` (decay data in the decay sublibrary, ENDF-6 §8.2).

What stays unmodelled, and is said per occurrence: ``LMF=6`` states (the
multiplicity lives in MF6) and MF9/MF10 states whose ``IZAP`` is not a nuclide
(JENDL-5's sub-actinide fission, ``ZAP=-1``); both travel as the file's text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model import (
    EVAL_LABEL,
    ConversionReport,
    EndfProvenance,
    Multiplicity,
    Nuclide,
    OutputChannel,
    PhysicalQuantity,
    Product,
    Q,
    Reaction,
    ReactionId,
    crossSectionAxes,
    multiplicityAxes,
    pidFromZA,
)
# crossSectionFormTag lives in the model, where kika.processing reaches it too.
from kika.nuclear_data.model.cross_section_forms import Reference, crossSectionFormTag
from kika.nuclear_data.model.functions.simple import Constant1d
from kika.nuclear_data.model.pops import MetaStable, zaFromPid

from .encode import usableInterpolationRegions
from .photons import _function1d, _tab1Of
from .residuals import _unspecified

__all__ = ["attachProductions", "encodeProductionSections", "ACTIVATION_MF",
           "ACTIVATION_MF8_KEY", "ACTIVATION_MF8_OPAQUE_KEY", "ACTIVATION_VERBATIM_KEY", "productionLabel",
           "crossSectionFormTag"]

#: The files this module reads and writes, in tape order.
ACTIVATION_MF = (8, 9, 10)

#: On the suite's provenance: ``{MT: [lines]}``, each MF8 section of a reaction
#: as read (SEND excluded). Re-emitted verbatim while it agrees with the model.
ACTIVATION_MF8_KEY = "mf8Activation"

#: On the suite's provenance: ``{MT: [(ZAP, LFS), ...]}``, the MF8 states the
#: decode did not turn into a production (LMF=6, a state whose MF9/MF10 is kept
#: as text or missing). They ride along with MF8 untouched and never make the
#: kept text disagree with the model.
ACTIVATION_MF8_OPAQUE_KEY = "mf8Opaque"

#: On the suite's provenance: ``{"MF/MT": [lines]}``, MF9/MF10 sections the
#: model does not carry (a state whose IZAP is not a nuclide).
ACTIVATION_VERBATIM_KEY = "activationVerbatim"

#: ``LMF`` values of MF8: where the state's yield is stated.
LMF_MF3, LMF_MF6, LMF_MF9, LMF_MF10 = 3, 6, 9, 10


def productionLabel(mt: int, pid: str) -> str:
    """``MT102 Am242_e2``: the reaction's kika label, then the nuclide made."""
    return f"MT{int(mt)} {pid}"



# ---------------------------------------------------------------------------
# MF8 of a reaction: just enough of it to tell where each state's yield is
# ---------------------------------------------------------------------------

@dataclass
class _Mf8State:
    zap: int
    elfs: float
    lmf: int
    lfs: int
    #: The CONT and, when NO=0, its decay LIST, as read.
    lines: List[str] = field(default_factory=list)


@dataclass
class _Mf8Section:
    za: float
    awr: float
    lis: int
    liso: int
    no: int
    head: str
    states: List[_Mf8State] = field(default_factory=list)


def _readMf8(lines: List[str]) -> _Mf8Section:
    """HEAD ``ZA, AWR, LIS, LISO, NS, NO``, then per state ``ZAP, ELFS, LMF, LFS, ND6``.

    With ``NO=0`` a state's CONT is followed by ``ND6`` values of decay data
    (ENDF-6 §8.2), which are carried as lines and not interpreted.
    """
    from kika.endf.utils import parse_line

    head = parse_line(lines[0])
    section = _Mf8Section(za=float(head.get("C1") or 0.0), awr=float(head.get("C2") or 0.0),
                          lis=int(head.get("C3") or 0), liso=int(head.get("C4") or 0),
                          no=int(head.get("C6") or 0), head=lines[0])
    idx = 1
    for _ in range(int(head.get("C5") or 0)):
        record = parse_line(lines[idx])
        nd6 = int(record.get("C5") or 0)
        count = 1 + ((nd6 + 5) // 6 if section.no == 0 else 0)
        section.states.append(_Mf8State(
            zap=int(round(float(record.get("C1") or 0.0))), elfs=float(record.get("C2") or 0.0),
            lmf=int(record.get("C3") or 0), lfs=int(record.get("C4") or 0),
            lines=list(lines[idx:idx + count])))
        idx += count
    if idx != len(lines):
        raise ValueError(f"MF8: read {idx} of {len(lines)} records")
    return section


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _host(suite, mt: int) -> Optional[Tuple[Reaction, str]]:
    """The reaction (or sum) whose σ an MF9/LMF=3 state multiplies, with its container path."""
    for container, path in ((suite.reactions, "reactions/reaction"),
                            (suite.sums, "sums/crossSectionSums/crossSectionSum"),
                            (suite.fissionComponents, "fissionComponents/fissionComponent")):
        for reaction in container:
            if reaction.ENDF_MT == mt:
                return reaction, path
    return None


def _reference(host: Tuple[Reaction, str]) -> Optional[Reference]:
    reaction, path = host
    form = reaction.crossSection.get(EVAL_LABEL) if hasattr(reaction.crossSection, "get") else None
    if form is None:
        return None
    return Reference(href=(f"/reactionSuite/{path}[@label='{reaction.label}']/crossSection/"
                           f"{crossSectionFormTag(form)}[@label='{EVAL_LABEL}']"),
                     label=EVAL_LABEL)


def _domainOf(function) -> Tuple[float, float]:
    return float(function.domainMin), float(function.domainMax)


def _one(domain: Tuple[float, float]) -> Multiplicity:
    return Multiplicity(form=Constant1d(constant=1.0, domainMin_=domain[0], domainMax_=domain[1],
                                        axes=multiplicityAxes(), label=EVAL_LABEL))


def _addNuclide(suite, pid: str, za: int, lfs: int, energy: float) -> None:
    pops = suite.PoPs
    if pid in pops.particles:
        particle = pops.particles[pid]
        if lfs and getattr(particle, "energy", None) is None:
            particle.energy = PhysicalQuantity(value=float(energy), unit="eV")
        return
    pops.add(Nuclide(id=pid, Z=za // 1000, A=za % 1000, nuclearLevel=int(lfs),
                     energy=PhysicalQuantity(value=float(energy), unit="eV") if lfs else None))


def _addMetaStables(suite, levels: Dict[int, Dict[int, str]]) -> None:
    """``Am242_m1`` → ``Am242_e2``: excited states made, numbered by level within each nuclide."""
    aliased = {alias.pid for alias in suite.PoPs.aliases.values()}
    for za, byLevel in sorted(levels.items()):
        isotope = pidFromZA(za)
        for index, lfs in enumerate(sorted(l for l in byLevel if l > 0), start=1):
            pid = byLevel[lfs]
            alias = f"{isotope}_m{index}"
            if pid in aliased or alias in suite.PoPs.aliases:
                continue
            suite.PoPs.aliases[alias] = MetaStable(id=alias, pid=pid, metaStableIndex=index)


def _production(suite, mt, pid, crossSectionForm, multiplicity, qi, domain, provenance):
    production = Reaction(
        id=ReactionId(label=productionLabel(mt, pid), residual=pid, ENDF_MT=int(mt)),
        outputChannel=OutputChannel(genre="production",
                                    Q=Q(value=float(qi), unit="eV",
                                        domainMin=domain[0], domainMax=domain[1])),
        provenance=provenance)
    production.crossSection[EVAL_LABEL] = crossSectionForm
    production.outputChannel.products.products.append(
        Product(pid=pid, label=pid, multiplicity=multiplicity, distribution=_unspecified()))
    return production


def _keep(suite, key: str, entry, lines) -> None:
    header = suite.provenance.headerFields
    header.setdefault(key, {})[entry] = [line.rstrip("\r\n") for line in lines]


def _sectionLines(section) -> List[str]:
    return str(section).splitlines()[:-1]


def attachProductions(suite, endf, report: Optional[ConversionReport] = None) -> ConversionReport:
    """Put every MF9/MF10 final state (and MF8 ``LMF=3``) into ``suite.productions``.

    Runs after the reactions exist: an MF9 state's cross section is a reference
    to its reaction's. MF8 sections are kept on the suite's provenance whatever
    is modelled from them, so the tape comes back with its index.
    """
    from kika.endf.classes.mf8.raw import MF8Raw

    report = report if report is not None else ConversionReport()
    files = {mf: getattr(endf.mf.get(mf), "mt", {}) for mf in ACTIVATION_MF
             if hasattr(endf, "mf") and mf in endf.mf}
    for mf in files:
        for mt, error in sorted((getattr(endf.mf[mf], "parse_errors", None) or {}).items()):
            report.lost(f"MF{mf}/MT{mt} is on the tape and could not be read ({error}); "
                        f"it is absent from this reactionSuite and from a tape written from it")
    for mt, section in sorted(files.get(8, {}).items()):
        if not isinstance(section, MF8Raw):
            # MT454/457/459 are a decay or yield sublibrary's (E7), not a
            # reaction's products; in an incident-particle tape they have no home.
            report.unsupportedNode(
                f"MF8/MT{mt} is decay or fission-yield data in an incident-particle "
                f"evaluation; it is read but not decoded into this reactionSuite "
                f"(read it as its own sublibrary)", unaffectedScopes=("cross-sections",))
    mf8 = {mt: s for mt, s in files.get(8, {}).items() if isinstance(s, MF8Raw)}
    mts = sorted(set(mf8) | set(files.get(9, {})) | set(files.get(10, {})))
    if not mts:
        return report
    if suite.provenance is None:
        report.lost("MF8-10 are present and the suite has no ENDF provenance to "
                    "keep their bookkeeping on, so no production is decoded")
        return report

    levels: Dict[int, Dict[int, str]] = {}
    #: pid -> [(level energy, "MF<n>/MT<m>")]: what each section says it is.
    stated: Dict[str, List[Tuple[float, str]]] = {}
    for mt in mts:
        index = None
        if mt in mf8:
            _keep(suite, ACTIVATION_MF8_KEY, mt, mf8[mt].raw_lines)
            try:
                index = _readMf8(mf8[mt].raw_lines)
            except (ValueError, IndexError, TypeError) as exc:
                report.lost(f"MF8/MT{mt}: its records do not walk ({exc}); it is kept "
                            f"as read and no LMF=3 state is decoded from it")
        modelled = set()
        for mf in (9, 10):
            section = files.get(mf, {}).get(mt)
            if section is None:
                continue
            why = _whyNotModelled(suite, section, mf, mt)
            if why is not None:
                _keep(suite, ACTIVATION_VERBATIM_KEY, f"{mf}/{mt}", _sectionLines(section))
                report.unsupportedNode(f"MF{mf}/MT{mt}: {why}; the section is kept as "
                                       f"read and written back unchanged",
                                       unaffectedScopes=("cross-sections",))
                continue
            for order, state in enumerate(section.states):
                pid = pidFromZA(int(state.izap), int(state.lfs))
                _addNuclide(suite, pid, int(state.izap), int(state.lfs), state.level_energy)
                levels.setdefault(int(state.izap), {})[int(state.lfs)] = pid
                stated.setdefault(pid, []).append((float(state.level_energy), f"MF{mf}/MT{mt}"))
                table = _function1d(state.interp, state.x, state.y,
                                    axes=multiplicityAxes() if mf == 9 else crossSectionAxes())
                table.label = EVAL_LABEL
                # Lifted out of a one-region Regions1d, a curve keeps its region
                # index 0, which means nothing on a curve standing alone.
                if hasattr(table, "index"):
                    table.index = None
                domain = _domainOf(table)
                if mf == 9:
                    form, multiplicity = _reference(_host(suite, mt)), Multiplicity(form=table)
                else:
                    form, multiplicity = table, _one(domain)
                provenance = EndfProvenance(
                    mat=section._mat, awr=section._awr, za=section.zaid, qm=float(state.qm),
                    interpolationRegions=list(state.interp),
                    headerFields={f"mf{mf}": {"order": order, "lis": section._lis,
                                              "l2": section._l2, "n2": section._n2,
                                              "pad": _pad(section)}})
                suite.productions.append(_production(suite, mt, pid, form, multiplicity,
                                                     state.qi, domain, provenance))
                modelled.add((int(state.izap), int(state.lfs)))
        if index is not None:
            _fromMf8(suite, index, mt, modelled, levels, report)
            opaque = [[s.zap, s.lfs] for s in index.states if (s.zap, s.lfs) not in modelled]
            if opaque:
                suite.provenance.headerFields.setdefault(ACTIVATION_MF8_OPAQUE_KEY, {})[mt] = opaque
            for s in index.states:
                if (s.zap, s.lfs) in modelled:
                    stated.setdefault(pidFromZA(s.zap, s.lfs), []).append((float(s.elfs), f"MF8/MT{mt}"))
    _addMetaStables(suite, levels)
    _reportLevelDisagreements(suite, stated, report)
    return report


#: Two statements of one level energy disagree beyond this relative difference:
#: FUDGE's tolerance between MF8 and MF9/MF10 (``readMF8``), below which the
#: differences are the rounding of a 7-digit field (B-VIII.1 Fe-55: 3 633 000
#: and 3 633 001 eV).
LEVEL_ENERGY_RTOL = 2.0e-4


def _reportLevelDisagreements(suite, stated, report) -> None:
    """One message per product whose level energy the sections state differently.

    PoPs holds one energy per nuclide -- the first statement read -- while each
    MF9/MF10 state keeps its own ``QM`` in its provenance, so a tape written from
    this suite comes back as read. A GNDS file has only the PoPs energy, and a
    tape written through it rebuilds every ``QM`` from it.
    """
    for pid, values in sorted(stated.items()):
        energies = [e for e, _ in values]
        spread = max(energies) - min(energies)
        if spread <= LEVEL_ENERGY_RTOL * max(abs(e) for e in energies):
            continue
        kept = getattr(getattr(suite.PoPs.particles.get(pid), "energy", None), "value", None)
        listed = ", ".join(f"{e:g} eV ({where})" for e, where in values)
        keeps = f"{kept:g} eV" if kept is not None else "no energy (a ground state)"
        report.approximated(
            f"{pid}: the sections state its level energy differently -- {listed}; PoPs "
            f"keeps {keeps}. An ENDF tape written from this suite keeps each state's "
            f"QM; through GNDS every QM is rebuilt from the PoPs energy",
            unaffectedScopes=("cross-sections",))


def _whyNotModelled(suite, section, mf, mt) -> Optional[str]:
    bad = [int(s.izap) for s in section.states if int(s.izap) < 1000]
    if bad:
        return f"IZAP={bad[0]} is not a nuclide, so the state has no product to name"
    if mf == 9:
        host = _host(suite, mt)
        if host is None or _reference(host) is None:
            return ("a multiplicity with no MF3 cross section to multiply, so the "
                    "production has no cross section to refer to")
    return None


def _fromMf8(suite, index: _Mf8Section, mt: int, modelled, levels, report) -> None:
    """MF8 states MF9/MF10 did not already give: ``LMF=3`` decoded, the rest declared."""
    for order, state in enumerate(index.states):
        key = (state.zap, state.lfs)
        if key in modelled:
            continue
        if state.lmf in (LMF_MF9, LMF_MF10):
            report.lost(f"MF8/MT{mt}: state ZAP={state.zap} LFS={state.lfs} points at "
                        f"MF{state.lmf}, which does not state it; MF8 is kept as read")
            continue
        if state.lmf != LMF_MF3 or state.zap < 1000:
            report.unsupportedNode(
                f"MF8/MT{mt}: state ZAP={state.zap} LFS={state.lfs} has LMF={state.lmf}, "
                f"which has no production node here (LMF=6: the multiplicity is in MF6); "
                f"MF8 is kept as read and written back", unaffectedScopes=("cross-sections",))
            continue
        host = _host(suite, mt)
        reference = _reference(host) if host is not None else None
        if reference is None:
            # No MF3 to refer to, so no cross section it could change (JENDL-5 Li-7
            # MT22 and S-34 MT104-111 were refused by reconstruction for it).
            report.lost(f"MF8/MT{mt}: LMF=3 state ZAP={state.zap} LFS={state.lfs} with no "
                        f"MF3 cross section to refer to; MF8 is kept as read",
                        unaffectedScopes=("cross-sections",))
            continue
        form = host[0].crossSection[EVAL_LABEL]
        domain = (float(form.domainMin), float(form.domainMax))
        qi = getattr(host[0].outputChannel.Q, "value", None)
        pid = pidFromZA(state.zap, state.lfs)
        _addNuclide(suite, pid, state.zap, state.lfs, state.elfs)
        levels.setdefault(state.zap, {})[state.lfs] = pid
        provenance = EndfProvenance(mat=getattr(suite.provenance, "mat", None),
                                    awr=index.awr, za=int(round(index.za)),
                                    headerFields={"mf8": {}})
        suite.productions.append(_production(suite, mt, pid, reference, _one(domain),
                                             qi if qi is not None else 0.0, domain, provenance))
        modelled.add(key)


def _pad(section) -> dict:
    return {"pairs": section.pad.pairs, "values": section.pad.values,
            "interp": section.pad.interp}


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

@dataclass
class _State:
    """One production as the ENDF files need it."""

    production: Reaction
    mt: int
    lmf: int
    izap: int
    lfs: int
    qi: float
    qm: float
    order: Tuple[int, int, int]
    table: Optional[Tuple[List[Tuple[int, int]], List[float], List[float]]] = None
    fields: dict = field(default_factory=dict)


def _levelOf(suite, pid: str) -> Tuple[int, float]:
    """``(LFS, excitation energy)`` of a product, from PoPs or from its ``_eN`` id."""
    particle = suite.PoPs.particles.get(pid) if suite.PoPs is not None else None
    level = getattr(particle, "nuclearLevel", None)
    if level is None:
        match = re.search(r"_e(\d+)$", pid)
        level = int(match.group(1)) if match else 0
    energy = getattr(particle, "energy", None)
    value = getattr(energy, "value", energy)
    return int(level), float(value or 0.0)


def _mergedRegions(pairs) -> List[Tuple[int, int]]:
    """Adjacent regions with the same law, as one.

    A GNDS writer splits a table at a step (a repeated abscissa) into a
    ``regions1d`` of two, and reading it back gives NR=2 where the tape had
    NR=1. ENDF-6 lets a region hold a repeated abscissa, so the two are the same
    table; this gives back the smallest NR that states it.
    """
    merged: List[Tuple[int, int]] = []
    for nbt, code in pairs:
        if merged and merged[-1][1] == code:
            merged[-1] = (nbt, code)
        else:
            merged.append((nbt, code))
    return merged


def _tableWithKeptRegions(function, kept) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
    """The TAB1 of *function*: with the file's own regions while they still fit it."""
    interp, x, y = _tab1Of(function)
    pairs = usableInterpolationRegions(kept, len(x)) if kept else None
    return (pairs if pairs is not None else _mergedRegions(interp)), x, y


def _classify(suite, production, report) -> Optional[_State]:
    mt = production.ENDF_MT
    products = [p for p in production.outputChannel.products if p.pid != "photon"]
    if mt is None or len(products) != 1:
        report.lost(f"{production.label}: a production with "
                    f"{'no MT' if mt is None else f'{len(products)} products'} has no "
                    f"MF8-10 equivalent and is not written")
        return None
    product = products[0]
    try:
        izap = zaFromPid(product.pid)
    except ValueError:
        izap = 0
    if izap < 1000:
        report.lost(f"{production.label}: product {product.pid!r} is not a nuclide; "
                    f"MF8-10 name nuclides only, so it is not written")
        return None
    lfs, energy = _levelOf(suite, product.pid)
    provenance = getattr(production, "provenance", None)
    header = getattr(provenance, "headerFields", None) or {}
    # Only regions this module kept: a provenance derived for a suite read from
    # GNDS carries the regions the model already has, steps split.
    kept = (getattr(provenance, "interpolationRegions", None)
            if ("mf9" in header or "mf10" in header) else None)
    qi = getattr(production.outputChannel.Q, "value", None)
    qi = float(qi) if qi is not None else 0.0
    qm = getattr(provenance, "qm", None)
    qm = float(qm) if qm is not None and ("mf9" in header or "mf10" in header) else qi + energy
    sigma = production.crossSection.get(EVAL_LABEL) if hasattr(production.crossSection, "get") else None
    multiplicity = getattr(product.multiplicity, "form", None) if product.multiplicity else None
    if sigma is None or multiplicity is None:
        report.lost(f"{production.label}: no evaluated cross section or multiplicity, "
                    f"so the state is not written")
        return None

    constantOne = isinstance(multiplicity, Constant1d) and float(multiplicity.constant) == 1.0
    if isinstance(sigma, Reference):
        lmf = LMF_MF3 if (constantOne and "mf9" not in header) else LMF_MF9
        table = None if lmf == LMF_MF3 else _tableWithKeptRegions(multiplicity, kept)
    else:
        lmf = LMF_MF10
        interp, x, y = _tableWithKeptRegions(sigma, kept)
        if not constantOne:
            if not isinstance(multiplicity, Constant1d):
                report.lost(f"{production.label}: a tabulated cross section times a "
                            f"tabulated multiplicity has no MF10 form; not written")
                return None
            y = [v * float(multiplicity.constant) for v in y]
            report.approximated(f"{production.label}: MF10 written as σ × "
                                f"{float(multiplicity.constant):g}, the constant multiplicity")
        table = (interp, x, y)
    kind = {LMF_MF3: "mf8", LMF_MF9: "mf9", LMF_MF10: "mf10"}[lmf]
    fields = header.get(kind) or {}
    order = fields.get("order")
    return _State(production=production, mt=int(mt), lmf=lmf, izap=izap, lfs=lfs, qi=qi,
                  qm=qm, order=(0 if order is not None else 1, order or 0, izap * 1000 + lfs),
                  table=table, fields=fields)


def _sectionFor(states: List[_State], mf: int, mt: int, mat, suite):
    from kika.endf.classes.mf9.base import MF9MT, MF10MT, ProductionState
    from kika.endf.utils import PadStyle

    first = states[0]
    provenance = getattr(first.production, "provenance", None)
    suiteProvenance = getattr(suite, "provenance", None)
    za = getattr(provenance, "za", None) or getattr(suiteProvenance, "za", None)
    awr = getattr(provenance, "awr", None) or getattr(suiteProvenance, "awr", None)
    lis = first.fields.get("lis")
    if lis is None:
        lis = int((getattr(suiteProvenance, "headerFields", None) or {}).get("lis") or 0)
    cls = MF9MT if mf == 9 else MF10MT
    section = cls(number=mt, _za=float(za) if za is not None else 0.0,
                  _awr=float(awr) if awr is not None else 0.0, _lis=int(lis),
                  _l2=int(first.fields.get("l2") or 0), _n2=int(first.fields.get("n2") or 0),
                  _mat=mat)
    pad = first.fields.get("pad")
    section.pad = PadStyle(**pad) if pad else PadStyle()
    for state in states:
        interp, x, y = state.table
        section.states.append(ProductionState(qm=state.qm, qi=state.qi, izap=state.izap,
                                              lfs=state.lfs, interp=list(interp),
                                              x=list(x), y=list(y)))
    return section


def _stamp(line: str, mat: int, mt: int, seq: int) -> str:
    return f"{line[:66]:<66}{int(mat):>4}{8:>2}{int(mt):>3}{seq:>5}"


def _mf8For(suite, mt: int, states: List[_State], kept: Optional[List[str]], opaque, mat, report):
    """MF8 of *mt*: the kept text if it still indexes these states, else rebuilt."""
    from kika.endf.classes.mf8.raw import MF8Raw
    from kika.endf.classes.mf12.base import emit_cont

    wanted = [(s.izap, s.lfs, s.lmf) for s in states]
    index = None
    if kept:
        try:
            index = _readMf8(kept)
        except (ValueError, IndexError, TypeError):
            index = None
    if index is not None:
        stated = [(s.zap, s.lfs, s.lmf) for s in index.states if (s.zap, s.lfs) not in opaque]
        if sorted(stated) == sorted(wanted):
            lines = [_stamp(line, mat, mt, k) for k, line in enumerate(kept, start=1)]
            return MF8Raw(number=mt, raw_lines=lines, _mat=mat)
        report.approximated(f"MF8/MT{mt}: the productions changed, so MF8 is rebuilt from "
                            f"them; states the model does not carry keep their records")

    suiteProvenance = getattr(suite, "provenance", None)
    header = getattr(suiteProvenance, "headerFields", None) or {}
    if index is not None:
        za, awr, lis, liso, no = index.za, index.awr, index.lis, index.liso, index.no
    else:
        first = getattr(states[0].production, "provenance", None)
        za = getattr(first, "za", None) or getattr(suiteProvenance, "za", None) or 0
        awr = getattr(first, "awr", None) or getattr(suiteProvenance, "awr", None) or 0.0
        lis, liso, no = int(header.get("lis") or 0), int(header.get("liso") or 0), 1
        report.approximated(
            f"MF8/MT{mt}: derived from the productions with NO=1, so the decay data "
            f"of each state is the decay sublibrary's (ENDF-6 §8.2); a source tape's "
            f"NO=0 decay LISTs have no node and are not reconstructed")
    blocks: List[List[str]] = []
    keptByKey = {(s.zap, s.lfs): s for s in index.states} if index is not None else {}
    for state in states:
        old = keptByKey.get((state.izap, state.lfs))
        if old is not None and old.lmf == state.lmf:
            blocks.append(old.lines)
            continue
        _, energy = _levelOf(suite, pidFromZA(state.izap, state.lfs))
        line, _ = emit_cont(float(state.izap), energy, state.lmf, state.lfs, 0, 0, mat, 8, mt, 0)
        blocks.append([line])
    if index is not None:
        blocks.extend(s.lines for s in index.states if (s.zap, s.lfs) in opaque)
    head, _ = emit_cont(float(za), float(awr), int(lis), int(liso), len(blocks), int(no),
                        mat, 8, mt, 0)
    lines = [head] + [line for block in blocks for line in block]
    lines = [_stamp(line, mat, mt, k) for k, line in enumerate(lines, start=1)]
    return MF8Raw(number=mt, raw_lines=lines, _mat=mat)


def encodeProductionSections(suite, mat: Optional[int] = None,
                             report: Optional[ConversionReport] = None):
    """The MF8, MF9 and MF10 sections of *suite*: ``([(MF, MT, section)], report)``.

    From ``suite.productions`` where the model carries them, from the kept text
    where it does not. Works the same for a suite read from GNDS, whose
    productions carry no ENDF provenance: every field is then derived from the
    production itself and PoPs.
    """
    from kika.endf.parsers.parse_mf9_10 import parse_mf9_mt, parse_mf10_mt

    report = report if report is not None else ConversionReport()
    header = getattr(getattr(suite, "provenance", None), "headerFields", None) or {}
    mat = int(mat if mat is not None else getattr(suite.provenance, "mat", 0) or 0)

    byMt: Dict[int, List[_State]] = {}
    for production in suite.productions:
        state = _classify(suite, production, report)
        if state is not None:
            byMt.setdefault(state.mt, []).append(state)

    keptMf8 = {int(mt): lines for mt, lines in (header.get(ACTIVATION_MF8_KEY) or {}).items()}
    opaque = {int(mt): {tuple(k) for k in keys}
              for mt, keys in (header.get(ACTIVATION_MF8_OPAQUE_KEY) or {}).items()}
    verbatim = header.get(ACTIVATION_VERBATIM_KEY) or {}
    sections = []
    for mt in sorted(set(byMt) | set(keptMf8)):
        states = byMt.get(mt, [])
        for mf, lmf in ((9, LMF_MF9), (10, LMF_MF10)):
            chosen = sorted((s for s in states if s.lmf == lmf), key=lambda s: s.order)
            if chosen:
                sections.append((mf, mt, _sectionFor(chosen, mf, mt, mat, suite)))
        sections.append((8, mt, _mf8For(suite, mt, _mf8Order(states, keptMf8.get(mt)),
                                        keptMf8.get(mt), opaque.get(mt, set()), mat, report)))

    parsers = {9: parse_mf9_mt, 10: parse_mf10_mt}
    for key, lines in verbatim.items():
        mf, mt = (int(v) for v in key.split("/"))
        section = parsers[mf](lines, mt)
        section._mat = mat
        sections.append((mf, mt, section))
    return sections, report


def _mf8Order(states: List[_State], kept: Optional[List[str]]) -> List[_State]:
    """The states in the kept MF8's order where it names them, then the rest."""
    if not kept:
        return sorted(states, key=lambda s: (s.lmf != LMF_MF3, s.order))
    try:
        index = _readMf8(kept)
    except (ValueError, IndexError, TypeError):
        return states
    rank = {(s.zap, s.lfs): k for k, s in enumerate(index.states)}
    return sorted(states, key=lambda s: (rank.get((s.izap, s.lfs), len(rank)), s.order))

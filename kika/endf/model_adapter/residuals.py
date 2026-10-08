"""The excited residual of a discrete-level reaction, as GNDS states it.

ENDF says "the residual is left in level *n*" with two numbers in the MF3 header:
``QI``, the Q of reaching that level, and ``QM``, the Q to the ground state —
plus ``LR`` when the level breaks up into light particles. GNDS has no QM and no
LR; it states the physics they encode. The residual is a product in its own
right (``Fe56_e1``), its own output channel is the decay, and that decay's
``Q`` is ``QM − QI``: the excitation energy for a gamma cascade, or the energy
freed by the breakup. This is how FUDGE converts an evaluation
(``ENDF_ITYPE_0_Misc.py:4219-4420``) and how the 558 distributed GNDS files
read, and :mod:`kika.gnds.decode` already reads it that way.

The ENDF decoder used to keep QM and LR in provenance only, so a GNDS file kika
wrote from a tape lost both — the excited residual was not there to carry them —
and a suite read back from that file could not be written as ENDF. With the
residual in the model, :mod:`.derive.reactions` gets QM and LR back from it.
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["levelSeries", "attachResiduals"]

#: First MT of each discrete-level series → the ejectile's ZA. The last MT of a
#: series (91, 649, 699, 749, 799, 849, 891) is its continuum and has no level.
_SERIES = {50: 1, 600: 1001, 650: 1002, 700: 1003, 750: 2003, 800: 2004, 875: 2}
_CONTINUUM = {91, 649, 699, 749, 799, 849, 891}


def levelSeries(mt: int) -> Optional[Tuple[int, int, int]]:
    """``(series start, ejectile ZA, level index)`` for a discrete-level MT, else ``None``."""
    if mt in _CONTINUUM:
        return None
    for start, ejectile in _SERIES.items():
        end = 91 if start == 50 else 891 if start == 875 else start + 49
        if start <= mt < end:
            return start, ejectile, mt - start
    return None


def _uniqueLabel(pid: str, taken: set) -> str:
    """FUDGE's ``uniqueLabel``: ``He4``, then ``He4__a``, ``He4__b``…"""
    if pid not in taken:
        return pid
    for suffix in "abcdefghijklmnopqrstuvwxyz":
        label = f"{pid}__{suffix}"
        if label not in taken:
            return label
    raise ValueError(f"more than 27 {pid} products in one channel")


def _unspecified():
    """§18's ``unspecified``: ENDF gives no distribution for the residual or its
    decay products outside MF6, and GNDS requires the node, so it says so."""
    from kika.nuclear_data.model import EVAL_LABEL, Distribution, Unspecified

    distribution = Distribution()
    distribution[EVAL_LABEL] = Unspecified(label=EVAL_LABEL)
    return distribution


#: FUDGE's ``lightIsotopeNames``: the order a breakup's light products are listed in.
_LIGHT_ORDER = ("n", "H1", "H2", "H3", "He3", "He4")


def _constantMultiplicity(count: int, domain):
    from kika.nuclear_data.model import Multiplicity, multiplicityAxes
    from kika.nuclear_data.model.functions.simple import Constant1d

    low, high = domain
    return Multiplicity(form=Constant1d(constant=float(count), domainMin_=low,
                                        domainMax_=high, axes=multiplicityAxes()))


def _domain(reaction):
    """The reaction's cross-section domain, which a breakup multiplicity spans, or ``None``."""
    from kika.nuclear_data.model import EVAL_LABEL

    try:
        form = reaction.crossSection[EVAL_LABEL]
        return float(form.domainMin), float(form.domainMax)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


def _decayChannel(residualZA: int, lr: int, q: float, report, mt: int, domain):
    """The residual's decay: to its ground state by gammas, or by breakup.

    A breakup is laid out as FUDGE lays it out
    (``fillRemainingProductsResidualForBreakup``): **one** product per light
    particle, carrying how many of it come out as its multiplicity, in
    ``_LIGHT_ORDER``, then whatever nucleus is left, last. The way back
    (``derive.reactions._lr``) reads that layout, as FUDGE's ``toENDF6`` does.
    """
    from kika.nuclear_data.model import OutputChannel, Product, Q, pidFromZA

    from .derive.reactions import LR_PRODUCTS

    channel = OutputChannel(genre="NBody", Q=Q(value=float(q), unit="eV"))
    taken: set = set()

    def add(pid, multiplicity=None):
        label = _uniqueLabel(pid, taken)
        taken.add(label)
        channel.products.products.append(Product(pid=pid, label=label,
                                                 multiplicity=multiplicity,
                                                 distribution=_unspecified()))

    if lr == 0:
        add(pidFromZA(residualZA))
        add("photon")
        return channel

    light = LR_PRODUCTS.get(lr)
    if light is None:
        report.lost(
            f"MT{mt}: LR={lr} is not a breakup ENDF-102's table describes, so the "
            f"residual's decay is not modelled and LR stays in the provenance only"
        )
        return None
    if domain is None:
        report.lost(
            f"MT{mt}: LR={lr} breakup with no evaluated cross section to give its "
            f"multiplicities a domain, so the residual's decay is not modelled"
        )
        return None
    from kika.nuclear_data.model.pops import zaFromPid

    remaining = residualZA
    for pid in _LIGHT_ORDER:
        count = light.get(pid, 0)
        if not count:
            continue
        add(pid, _constantMultiplicity(count, domain))
        # An elemental residual (A = 0) loses charge only, as in FUDGE.
        za = zaFromPid(pid)
        remaining -= count * (za if residualZA % 1000 else 1000 * (za // 1000))
    if remaining:
        add(pidFromZA(remaining))
    return channel


def attachResiduals(suite, report):
    """Give every discrete-level reaction whose QM ≠ QI, or LR ≠ 0, its decaying residual.

    The residual becomes a product of the reaction — or, when MF6 already gave
    it one, that product gets the decay as its own output channel — and a PoPs
    entry for the level. Reactions to a level with QM = QI and LR = 0 (a series'
    ground state) have nothing to say and are left alone.
    """
    from kika.nuclear_data.model import Nuclide, Product, pidFromZA
    from kika.nuclear_data.model.pops import zaFromPid

    targetZA = None
    try:
        targetZA = zaFromPid(suite.target)
    except ValueError:
        return report
    if suite.projectile != "n" or not targetZA:
        return report

    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        series = levelSeries(int(mt)) if mt is not None else None
        provenance = getattr(reaction, "provenance", None)
        if series is None or provenance is None:
            continue
        qi = getattr(reaction.outputChannel.Q, "value", None)
        qm, lr = getattr(provenance, "qm", None), int(getattr(provenance, "lr", 0) or 0)
        if qi is None or qm is None or (qm == qi and lr == 0):
            continue

        _, ejectile, level = series
        residualZA = targetZA + 1 - ejectile
        decay = _decayChannel(residualZA, lr, qm - qi, report, mt, _domain(reaction))
        if decay is None:
            continue
        pid = pidFromZA(residualZA, level)

        existing = next((p for p in reaction.outputChannel.products
                         if p.pid != "photon" and _za(p.pid) == residualZA), None)
        if existing is not None:
            if existing.outputChannel is None:
                existing.outputChannel = decay
        else:
            reaction.outputChannel.products.products.append(
                Product(pid=pid, label=pid, distribution=_unspecified(),
                        outputChannel=decay))
        if reaction.outputChannel.genre is None:
            reaction.outputChannel.genre = "twoBody"
        if pid not in suite.PoPs.particles:
            suite.PoPs.add(Nuclide(id=pid, Z=residualZA // 1000, A=residualZA % 1000,
                                   nuclearLevel=level))
    return report


def _za(pid: str) -> Optional[int]:
    from kika.nuclear_data.model.pops import zaFromPid

    try:
        return zaFromPid(pid)
    except ValueError:
        return None

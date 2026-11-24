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

__all__ = ["levelSeries", "attachResiduals", "residualOf"]

from kika.nuclear_data.model.endf_conversion import levelSeries

def _suffix(n: int) -> str:
    """0 → ``a``, 25 → ``z``, 26 → ``aa``: FUDGE's ``incrementSuffix`` sequence."""
    letters = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        letters = "abcdefghijklmnopqrstuvwxyz"[r] + letters
    return letters


def _uniqueLabel(pid: str, taken: set) -> str:
    """FUDGE's ``uniqueLabel``: ``He4``, then ``He4__a`` … ``He4__z``, ``He4__aa``…

    The two-letter tail is not hypothetical: ENDF/B-VIII.1 N-14's capture
    states 59 photon lines, and NNDC's GNDS of it runs to ``photon__bf``
    (``fudge/suites.py`` ``uniqueLabel``).
    """
    if pid not in taken:
        return pid
    n = 0
    while True:
        label = f"{pid}__{_suffix(n)}"
        if label not in taken:
            return label
        n += 1


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
    from kika.nuclear_data.model import EVAL_LABEL, Multiplicity, multiplicityAxes
    from kika.nuclear_data.model.functions.simple import Constant1d

    low, high = domain
    # Labelled, as every evaluated form is: FUDGE refuses an unlabelled one.
    return Multiplicity(form=Constant1d(constant=float(count), domainMin_=low,
                                        domainMax_=high, axes=multiplicityAxes(),
                                        label=EVAL_LABEL))


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

    # The nucleus that is left carries multiplicity 1, as in FUDGE and NNDC's
    # files: FUDGE's toENDF6 reads a multiplicity off every product it walks
    # and stops on one without (found by the FUDGE oracle).
    one = _constantMultiplicity(1, domain) if domain is not None else None
    if lr == 0:
        # The photons are the file's to state (MF6 here, MF12 in `photons.py`); a
        # level with none has only its ground state, as in NNDC's Li-6 MT52.
        add(pidFromZA(residualZA), one)
        return channel

    light = LR_PRODUCTS.get(lr)
    if light is None:
        report.lost(
            f"MT{mt}: LR={lr} is not a breakup ENDF-102's table describes, so the "
            f"residual's decay is not modelled and LR stays in the provenance only",
            # LR names the residual's decay, never the reaction's cross section
            # (B-10 MT604, LR=40, was refused by reconstruction for it).
            unaffectedScopes=("cross-sections",),
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
        add(pidFromZA(remaining), one)
    return channel


def _zas(suite) -> Optional[Tuple[int, int]]:
    """``(target ZA, projectile ZA)``, or ``None`` when the suite names no nuclide."""
    from kika.nuclear_data.model.pops import zaFromPid

    try:
        targetZA = zaFromPid(suite.target)
        projectileZA = zaFromPid(suite.projectile or "n")
    except ValueError:
        return None
    return (targetZA, projectileZA) if targetZA else None


def residualOf(suite, reaction, report, *, force: bool = False):
    """The decaying residual product of a discrete-level *reaction*, built if needed.

    ``None`` for a reaction that is not to a discrete level, or whose level has
    nothing to say (QM = QI and LR = 0) unless *force* -- which is what an MF12
    LO=2 cascade is: a level whose decay the file does state.
    """
    from kika.nuclear_data.model import Nuclide, Product, pidFromZA

    zas = _zas(suite)
    mt = reaction.ENDF_MT
    series = levelSeries(int(mt)) if mt is not None else None
    provenance = getattr(reaction, "provenance", None)
    if zas is None or series is None or provenance is None:
        return None
    targetZA, projectileZA = zas
    qi = getattr(reaction.outputChannel.Q, "value", None)
    qm, lr = getattr(provenance, "qm", None), int(getattr(provenance, "lr", 0) or 0)
    if qi is None or qm is None or (qm == qi and lr == 0 and not force):
        return None

    _, ejectile, level = series
    # The projectile's ZA, not a neutron's: a charged-particle tape names its
    # projectile through NSUB (decode._projectile).
    residualZA = targetZA + projectileZA - ejectile
    pid = pidFromZA(residualZA, level)
    existing = next((p for p in reaction.outputChannel.products
                     if p.pid != "photon" and _za(p.pid) == residualZA), None)
    if existing is not None and existing.outputChannel is not None:
        return existing
    decay = _decayChannel(residualZA, lr, qm - qi, report, mt, _domain(reaction))
    if decay is None:
        return None
    if existing is not None:
        existing.outputChannel = decay
        _nameAsLevel(suite, existing, pid, reaction)
        residual = existing
    else:
        domain = _domain(reaction)
        residual = Product(pid=pid, label=pid, distribution=_unspecified(),
                           outputChannel=decay,
                           multiplicity=(_constantMultiplicity(1, domain)
                                         if domain is not None else None))
        reaction.outputChannel.products.products.append(residual)
    if lr == 0:
        _photonsIntoDecay(suite, reaction, residual)
    if reaction.outputChannel.genre is None:
        reaction.outputChannel.genre = "twoBody"
    if pid not in suite.PoPs.particles:
        suite.PoPs.add(Nuclide(id=pid, Z=residualZA // 1000, A=residualZA % 1000,
                               nuclearLevel=level))
    return residual


def _nameAsLevel(suite, product, pid: str, reaction) -> None:
    """Give an MF6 residual the level's pid, as FUDGE does, and tell MF6's record.

    MF6 states the residual with LIP=0 (Li-6 MT52 gives ZAP=3006), so it was read
    as the ground state. The record keeps the label the ENDF writer looks the
    product up by, and the ``MF6`` conversion flag the href the way back reads.
    """
    if product.pid == pid:
        return
    old = product.label or product.pid
    taken = {p.label for p in reaction.outputChannel.products if p is not product}
    product.pid, product.label = pid, _uniqueLabel(pid, taken)
    fields = (getattr(reaction.provenance, "headerFields", None) or {}).get("mf6") or {}
    for record in fields.get("products", ()):
        if record.get("label") == old:
            record["label"], record["pid"] = product.label, pid
    _repointFlag(suite, reaction, old, _productHref(reaction, product.label))


def _photonsIntoDecay(suite, reaction, residual) -> None:
    """MF6's photons of a level reaction are its residual's de-excitation, so they go
    into the residual's decay, after its ground state, as FUDGE writes them."""
    photons = [p for p in reaction.outputChannel.products if p.pid == "photon"]
    if not photons:
        return
    reaction.outputChannel.products.products[:] = [
        p for p in reaction.outputChannel.products if p.pid != "photon"]
    residual.outputChannel.products.products.extend(photons)
    for photon in photons:
        _repointFlag(suite, reaction, photon.label,
                     _productHref(reaction, residual.label)
                     + f"/outputChannel/products/product[@label='{photon.label}']")
    # `decode._channelGenre`'s rule, now that the photons' LAW=1 is not the reaction's.
    fields = (getattr(reaction.provenance, "headerFields", None) or {}).get("mf6") or {}
    laws = {record["law"] for record in fields.get("products", ()) if record["zap"] != 0}
    if reaction.outputChannel.genre == "NBody" and laws and laws <= {2, 3, 4}:
        reaction.outputChannel.genre = "twoBody"


def _productHref(reaction, label: str) -> str:
    from .derive.reactions import reactionHref

    return f"{reactionHref(reaction)}/outputChannel/products/product[@label='{label}']"


def _repointFlag(suite, reaction, oldLabel: str, href: str) -> None:
    """Move the conversion flag ``decode._flagMF6`` put on a product that moved."""
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    flags = EndfConversionFlags.of(suite)
    if flags is None:
        return
    old = _productHref(reaction, oldLabel)
    flags.conversions[:] = [(href if target == old else target, text)
                            for target, text in flags.conversions]


def attachResiduals(suite, report):
    """Give every discrete-level reaction whose QM ≠ QI, or LR ≠ 0, its decaying residual.

    The residual becomes a product of the reaction — or, when MF6 already gave
    it one, that product gets the decay as its own output channel — and a PoPs
    entry for the level. Reactions to a level with QM = QI and LR = 0 (a series'
    ground state) have nothing to say and are left alone, unless MF12 states
    their cascade (:func:`residualOf` with ``force``, from ``photons.py``).
    """
    if _zas(suite) is None:
        return report
    for reaction in suite.reactions:
        residualOf(suite, reaction, report)
    return report


def _za(pid: str) -> Optional[int]:
    from kika.nuclear_data.model.pops import zaFromPid

    try:
        return zaFromPid(pid)
    except ValueError:
        return None

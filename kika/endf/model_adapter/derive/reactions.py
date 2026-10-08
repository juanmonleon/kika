"""G2: MF3's bookkeeping — QM, QI, LR and the interpolation regions — from the model.

FUDGE's rules (``toENDF6/reactions/base.py:105-158``,
``toENDF6/reactionData/crossSection.py:30-61``), stated in the model's terms:

- **QI** is the output channel's ``Q``, unless the file's
  ``ENDFconversionFlags`` give a ``QI=`` for it. FUDGE writes that flag on a
  continuum reaction (MT91, MT649…), whose GNDS ``Q`` is the *ground-state* Q.
- **QM** is, in that flagged case, the ``Q`` itself. Otherwise, when the
  residual is a level that decays, ``QI`` plus the decay's ``Q`` — the
  representation GNDS gives an excited residual. MT4 with a ground-state
  target is 0 (``crossSection.py:44``). Anything else has QM = QI.
- **LR** is 0 unless the residual breaks up into light particles, in which case
  the products of that decay are looked up in ENDF-102's LR table.

The interpolation regions are the ``regions1d``'s own, which is what the MF3
encoder rebuilds anyway when it has none kept; deriving them here makes them
inspectable and comparable by the oracle.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from . import DERIVED, DerivationContext, register

__all__ = ["LR_PRODUCTS", "reactionHref", "deriveReaction"]

#: ENDF-102's LR flags, by the light particles the residual breaks up into
#: (``endf_endl.endfLRtoC_ProductLists``). 39 and 40 are not allowed in ENDF-6.
LR_PRODUCTS = {
    22: {"He4": 1}, 23: {"He4": 3}, 24: {"n": 1, "He4": 1},
    25: {"n": 2, "He4": 1}, 28: {"H1": 1}, 29: {"He4": 2},
    30: {"n": 1, "He4": 2}, 32: {"H2": 1}, 33: {"H3": 1}, 34: {"He3": 1},
    35: {"H2": 1, "He4": 2}, 36: {"H3": 1, "He4": 2},
}
_LIGHT = {"n", "H1", "H2", "H3", "He3", "He4"}


def reactionHref(reaction) -> str:
    """The xPath FUDGE's ``ENDFconversionFlags`` use to name *reaction*."""
    from kika.nuclear_data.model import CrossSectionSum

    if isinstance(reaction, CrossSectionSum):
        container = "sums/crossSectionSums/crossSectionSum"
    else:
        container = "reactions/reaction"
    return f"/reactionSuite/{container}[@label='{reaction.label}']"


def _count(product) -> int:
    form = getattr(getattr(product, "multiplicity", None), "form", None)
    value = getattr(form, "constant", getattr(form, "value", None))
    try:
        return max(1, int(round(float(value))))
    except (TypeError, ValueError):
        return 1


def _decayingResidual(reaction):
    """The product whose own output channel is a decay, or ``None``."""
    for product in reaction.outputChannel.products:
        if product.pid != "photon" and getattr(product, "outputChannel", None) is not None:
            return product
    return None


def _lr(residual, label: str, report) -> int:
    """LR from what *residual* decays into: 0 for a gamma cascade.

    FUDGE's rule (``toENDF6/reactions/base.py:120-150``), on the layout both it
    and ``residuals.py`` write: one product per light particle, then the nucleus
    left over, last. A breakup whose products all carry distributions is LR=1
    (MF6 has them). Otherwise the light particles are matched against ENDF-102's
    table **by multiplicity**, the leftover nucleus excluded.

    Two departures from FUDGE, both where it is wrong rather than different:
    it counts *entries*, so a light product with multiplicity 2 counts once and
    LR=29/30/36 with a leftover come back as 22/24/33; and it takes the last
    entry as the leftover always, which is what its Be-8 and B-10 kludges patch.
    Here the last entry is the leftover when it is a nucleus, or a light particle
    already listed (Be-8 → α + α under LR=22: ``He4``, ``He4__a``). A breakup
    with no LR in the table is refused (D3); FUDGE writes 0.
    """
    channel = residual.outputChannel
    base = residual.pid.split("_")[0]
    entries = [p for p in channel.products if p.pid not in (base, "photon")]
    if not entries:
        return 0
    from kika.nuclear_data.model import Unspecified

    def specified(product):
        forms = list(product.distribution.values()) if product.distribution is not None else []
        return any(not isinstance(form, Unspecified) for form in forms)

    withDistributions = sum(specified(p) for p in entries)
    if withDistributions:
        if withDistributions != len(entries):
            raise ValueError(
                f"{label}: the residual {residual.pid} breaks up into products only "
                f"some of which carry a distribution, so MF3's LR is neither 1 nor "
                f"a breakup of ENDF-102's table"
            )
        return 1

    last = entries[-1]
    leftover = last.pid not in _LIGHT or any(p.pid == last.pid for p in entries[:-1])
    light = Counter()
    heavy = []
    for product in entries[:-1] if leftover else entries:
        if product.pid in _LIGHT:
            light[product.pid] += _count(product)
        else:
            heavy.append(product.pid)
    if not heavy:
        for lr, products in LR_PRODUCTS.items():
            if dict(light) == products:
                return lr
    raise ValueError(
        f"{label}: the residual {residual.pid} breaks up into {dict(light)}"
        f"{' + ' + str(heavy) if heavy else ''}, which is no LR of ENDF-102's "
        f"table, so MF3's LR cannot be written"
    )


#: A series' continuum MT → the range of its discrete levels.
_CONTINUUM_SERIES = {91: range(50, 91), 649: range(600, 649), 699: range(650, 699),
                     749: range(700, 749), 799: range(750, 799), 849: range(800, 849),
                     891: range(875, 891)}


def _continuum(reaction, mt: int, qi, context: DerivationContext, report):
    """QM and LR of a continuum reaction, from the discrete levels of its series.

    QM is the Q to the residual's ground state, a property of the channel and
    not of the level, so every level of the series states the same one: it is
    read off the lowest level whose decaying residual carries it. LR is the
    breakup the series' levels share, when they all share one. Evaluations do
    not always agree with themselves here — JEFF-4.0 Fe-56 gives MT649 a QM
    400 eV from MT601's — so the report says where the number came from.
    """
    levels = []
    for sibling in context.suite.reactions:
        smt = sibling.ENDF_MT
        if smt is None or int(smt) not in _CONTINUUM_SERIES[mt]:
            continue
        residual = _decayingResidual(sibling)
        sqi = _q(getattr(sibling.outputChannel.Q, "value", None))
        if residual is None or sqi is None:
            if int(smt) == min(_CONTINUUM_SERIES[mt]) and sqi is not None:
                levels.append((int(smt), sqi, 0))
            continue
        decayQ = _q(getattr(residual.outputChannel.Q, "value", None))
        if decayQ is None:
            continue
        try:
            lr = _lr(residual, sibling.label, report)
        except ValueError:
            lr = None
        levels.append((int(smt), sqi + decayQ, lr))
    if not levels:
        report.approximated(
            f"MT{mt}: a continuum reaction with no discrete level of its series "
            f"in the suite and no QI flag, so QM is written equal to QI"
        )
        return qi, 0
    levels.sort()
    source, qm, _ = levels[0]
    breakups = {lr for _, _, lr in levels if lr}
    lr = breakups.pop() if len(breakups) == 1 else 0
    report.approximated(
        f"MT{mt}: QM of the continuum taken from MT{source}, the lowest level of "
        f"its series (ENDF's QM is the channel's ground-state Q)"
        + (f"; LR={lr}, the breakup its levels share" if lr else "")
    )
    return qm, lr


def _q(value) -> Optional[float]:
    return None if value is None else float(value)


def deriveReaction(reaction, path, context: DerivationContext, report):
    """MF3's provenance for one reaction or cross-section sum."""
    from kika.nuclear_data.model import CrossSectionSum, EVAL_LABEL, EndfProvenance
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    if reaction.ENDF_MT is None:
        return None
    mt = int(reaction.ENDF_MT)
    q = _q(getattr(reaction.outputChannel.Q, "value", None))

    flags = EndfConversionFlags.of(context.suite)
    stated = flags.flagsFor(reactionHref(reaction) + "/outputChannel/Q") if flags else {}

    lr = 0
    if "QI" in stated and stated["QI"] is not None:
        qi, qm = float(stated["QI"]), q
    else:
        qi = q
        residual = None if isinstance(reaction, CrossSectionSum) else _decayingResidual(reaction)
        if residual is not None:
            decayQ = _q(getattr(residual.outputChannel.Q, "value", None))
            qm = None if qi is None or decayQ is None else qi + decayQ
            lr = _lr(residual, reaction.label, report)
        elif mt == 4 and context.suite.projectile == "n" and "_e" not in context.suite.target:
            qm = 0.0
        elif mt in _CONTINUUM_SERIES and not isinstance(reaction, CrossSectionSum):
            qm, lr = _continuum(reaction, mt, qi, context, report)
        else:
            qm = qi

    regions = []
    if EVAL_LABEL in reaction.crossSection and hasattr(reaction.crossSection[EVAL_LABEL], "toEndfRegions"):
        _, _, regions = reaction.crossSection[EVAL_LABEL].toEndfRegions()

    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, qm=qm, lr=lr,
                          interpolationRegions=[tuple(pair) for pair in regions],
                          headerFields={"qi": qi} if qi != q else {})


def _isReaction(node) -> bool:
    from kika.nuclear_data.model import Reaction

    return isinstance(node, Reaction)


register("reactions", _isReaction, deriveReaction)

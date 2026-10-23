"""A processed suite → the reactions of a dosimetry ACE table (``.NNy``).

What goes in, in this order (the order ACER writes MF 3 then MF 10):

1. **Every cross section with an MT** at the table's temperature, sums
   included (MT 1, 3, 4, 103-107 …), as the processing tabulated them: lin-lin,
   one region, reconstructed and broadened. A level sum the tape does not state
   (MT 103 from MT 600-649, 107 from 800-849, 16 from 875-891 …,
   :data:`kika.hdf5.processing.LEVEL_SUMS`) is added up on the union grid, as
   RECONR adds it to the PENDF ACER copies: Fe-56 (n,p) is MT 103 of a tape that
   states only its levels. ACER skips MT 1; kika keeps it, since a
   total-collision tally is as legitimate as any other and costs one table.
2. **MT 301 and MT 444** (KERMA and damage energy, eV·b) when the heating stage
   ran, and **MT 203-207** (gas production) computed as kika's GASPR computes
   them for the ``.c`` table: each reaction's σ times the light particles it
   emits, summed on the temperature's union grid. These are what dosimetry
   tallies ask for (dpa from MT 444, helium from MT 207); ACER has them only
   when the PENDF went through HEATR and GASPR first.
3. **Production cross sections**, numbered ``MT + 1000·(10 + LFS)``, or
   ``(50 + LFS)·10⁶ + ZA`` for the products of MT 5
   (:func:`kika.ace.classes.dosimetry.production_mt`, IRDFF-II's numbering):

   - MF 10 as the evaluation tabulates it, with its own interpolation laws
     (exact, and what ACER copies);
   - MF 9, which ACER ignores: the multiplicity times the reaction's processed
     σ, on the union of both grids, refined where the product read lin-lin
     misses the exact one by more than the processing tolerance.

A number the table cannot tell apart (two products of one MT and LFS other
than MT 5, which no evaluation known states) is not written: whoever reads
the table looks a number up and would find one of them. Each such product is
named in the omitted list instead. A cross section zero everywhere is not
written either.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.ace.classes.dosimetry import DosimetryReaction, production_mt

#: The processing tolerance the MF 9 products are refined to (lin-lin against the exact product).
PRODUCT_RTOL = 1.0e-3


def _nonzero(y) -> bool:
    return bool(np.any(np.asarray(y) != 0.0))


def _gas(suite, crossSections) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """MT 203-207 on the union grid, as :func:`kika.hdf5.processing._gas` sums them."""
    from kika.hdf5.fission_energy import evaluate
    from kika.hdf5.processing import NEVER, gasMultiplicities
    from kika.nuclear_data.model.functions.simple import Constant1d

    if 1 in crossSections:
        grid = np.asarray(crossSections[1][0], dtype=float)
    else:
        grid = np.unique(np.concatenate([np.asarray(e, dtype=float) for e, _ in crossSections.values()]))
    totals: Dict[int, np.ndarray] = {}
    target = suite.PoPs[suite.target] if suite.PoPs is not None and suite.target in suite.PoPs else None
    zaTarget = None if getattr(target, "Z", None) is None else 1000 * int(target.Z) + int(target.A)
    for container in ("reactions", "fissionComponents"):
        for reaction in getattr(suite, container):
            mt = reaction.ENDF_MT
            if mt is None or mt in NEVER or mt not in crossSections:
                continue
            e, sigma = (np.asarray(v, dtype=float) for v in crossSections[mt])
            on = np.interp(grid, e, sigma, left=0.0, right=0.0)
            for gasMT, form in gasMultiplicities(reaction, zaTarget).items():
                forms = form if isinstance(form, tuple) else (form,)
                k = sum(np.full(grid.shape, f.constant) if isinstance(f, Constant1d)
                        else np.asarray(evaluate(f, grid)) for f in forms)
                totals[gasMT] = totals.get(gasMT, np.zeros_like(grid)) + k * on
    out = {}
    for gasMT, y in sorted(totals.items()):
        if not _nonzero(y):
            continue
        start = max(0, int(np.argmax(y != 0.0)) - 1)
        out[gasMT] = (grid[start:], y[start:])
    return out


def _levelSums(crossSections) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """The level sums the tape leaves out, each on the union of its levels' grids."""
    from kika.hdf5.processing import LEVEL_SUMS

    out = {}
    for mt, levels in LEVEL_SUMS.items():
        parts = [crossSections[c] for c in levels if c in crossSections]
        if mt in crossSections or not parts:
            continue
        grid = np.unique(np.concatenate([np.asarray(e, dtype=float) for e, _ in parts]))
        total = sum(np.interp(grid, np.asarray(e, dtype=float), np.asarray(y, dtype=float), left=0.0, right=0.0)
                    for e, y in parts)
        if _nonzero(total):
            start = max(0, int(np.argmax(total != 0.0)) - 1)
            out[mt] = (grid[start:], total[start:])
    return out


def _refinedProduct(multiplicity, e, sigma, rtol=PRODUCT_RTOL, rounds=40, sigmaAt=None):
    """``m(E)·σ(E)`` tabulated lin-lin to ``rtol``: m follows its own law, σ is lin-lin on ``e``
    (or ``sigmaAt(x)`` when given, a table under its own laws).

    A step of either (a node written twice, or the edge of the multiplicity's
    domain) stays a step: the product is tabulated piece by piece between the
    steps, each piece read from inside, and the pieces meet at a repeated node.
    """
    e = np.asarray(e, dtype=float)
    mx = np.asarray(multiplicity.toEndfRegions()[0], dtype=float) if hasattr(multiplicity, "toEndfRegions")         else np.empty(0)
    lo, hi = float(e[0]), float(e[-1])
    cuts = set(e[1:][np.diff(e) == 0.0]) | set(mx[1:][np.diff(mx) == 0.0])
    cuts |= {float(multiplicity.domainMin), float(multiplicity.domainMax)}
    bounds = [lo] + sorted(c for c in cuts if lo < c < hi) + [hi]

    def exact(x, a, b):
        inside = np.clip(x, np.nextafter(a, b), np.nextafter(b, a))      # read each piece from inside
        on = sigmaAt(inside) if sigmaAt is not None else np.interp(inside, e, sigma)
        return np.asarray(multiplicity.evaluate(inside, outOfRange="zero"), dtype=float) * on

    xs, ys = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        nodes = np.concatenate([[a, b], e[(e > a) & (e < b)], mx[(mx > a) & (mx < b)]])
        grid = np.unique(nodes)
        y = exact(grid, a, b)
        for _ in range(rounds):
            mid = 0.5 * (grid[:-1] + grid[1:])
            want = exact(mid, a, b)
            bad = np.abs(0.5 * (y[:-1] + y[1:]) - want) > rtol * np.abs(want) + 1e-30
            bad &= mid > grid[:-1]
            if not bad.any():
                break
            grid = np.insert(grid, np.flatnonzero(bad) + 1, mid[bad])
            y = exact(grid, a, b)
        if xs and xs[-1][-1] == grid[0] and ys[-1][-1] == y[0]:
            grid, y = grid[1:], y[1:]                                  # no step here after all
        xs.append(grid)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def _ownTable(number, form) -> DosimetryReaction:
    """A table as the evaluation states it: its nodes and its interpolation laws."""
    x, y, regions = form.toEndfRegions()
    nbt = [int(n) for n, _ in regions]
    law = [int(i) for _, i in regions]
    if len(law) == 1 and law[0] == 2:
        nbt, law = [], []
    return DosimetryReaction(number, np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                             breakpoints=nbt, interpolation=law)


def _productions(suite, crossSections, omitted, sigmaForms=None) -> List[DosimetryReaction]:
    from kika.endf.model_adapter.productions import _levelOf
    from kika.nuclear_data.model.cross_section_forms import Reference
    from kika.nuclear_data.model.pops import zaFromPid

    found: List[Tuple[int, str, DosimetryReaction]] = []
    for production in suite.productions:
        mt = production.ENDF_MT
        products = [p for p in production.outputChannel.products if p.pid != "photon"]
        if mt is None or len(products) != 1:
            omitted.append(f"{production.label}: no single product, no dosimetry number")
            continue
        product = products[0]
        lfs, _ = _levelOf(suite, product.pid)
        particle = suite.PoPs.particles.get(product.pid) if suite.PoPs is not None else None
        za = getattr(particle, "ZA", None) or zaFromPid(product.pid)
        number = production_mt(mt, lfs, za)
        forms = production.crossSection.forms
        form = next(iter(forms.values()), None) if forms else None
        if form is None:
            omitted.append(f"{production.label}: no cross section")
            continue
        if isinstance(form, Reference):
            if mt not in crossSections:
                omitted.append(f"{production.label}: MF 9 of MT {mt}, which has no processed cross section")
                continue
            mult = product.multiplicity.form if product.multiplicity is not None else None
            if mult is None:
                omitted.append(f"{production.label}: MF 9 without a multiplicity")
                continue
            e, sigma = (np.asarray(v, dtype=float) for v in crossSections[mt])
            host = (sigmaForms or {}).get(mt)
            sigmaAt = (lambda x, f=host: np.asarray(f.evaluate(x, outOfRange="zero"), dtype=float)) if host else None
            x, y = _refinedProduct(mult, e, sigma, sigmaAt=sigmaAt)
            reaction = DosimetryReaction(number, x, y)
        else:
            reaction = _ownTable(number, form)
        if not _nonzero(reaction.cross_section):
            continue
        found.append((number, production.label, reaction))
    counts: Dict[int, int] = {}
    for number, _, _ in found:
        counts[number] = counts.get(number, 0) + 1
    out = []
    for number, label, reaction in found:
        if counts[number] > 1:
            omitted.append(f"{label}: one of {counts[number]} products that share number {number} "
                           f"(same MT and LFS); a dosimetry table cannot tell them apart")
            continue
        out.append(reaction)
    return out


def dosimetryReactions(suite, crossSections, heating: Optional[dict] = None
                       ) -> Tuple[List[DosimetryReaction], List[str]]:
    """``(reactions, omitted)`` of the dosimetry table of ``suite`` at one temperature.

    ``crossSections`` is ``{MT: (E, σ)}`` at that temperature (what
    :meth:`kika.endf.EndfProcessing.cross_sections` returns), ``heating``
    ``{301: (E, h), 444: (E, d)}`` or ``None``. ``suite`` is the whole decoded
    evaluation (its reactions' products give the gas production, its
    ``productions`` the MF 9/10 cross sections).
    """
    omitted: List[str] = []
    rows: Dict[int, DosimetryReaction] = {}
    for mt, (e, sigma) in crossSections.items():
        e, sigma = np.asarray(e, dtype=float), np.asarray(sigma, dtype=float)
        if e.size and _nonzero(sigma):
            rows[int(mt)] = DosimetryReaction(int(mt), e, sigma)
    for mt, (e, values) in _levelSums(crossSections).items():
        rows[mt] = DosimetryReaction(mt, e, values)
    for mt, (e, values) in (heating or {}).items():
        if _nonzero(values):
            rows[int(mt)] = DosimetryReaction(int(mt), np.asarray(e, dtype=float), np.asarray(values, dtype=float))
    for mt, (e, values) in _gas(suite, crossSections).items():
        if mt in rows:
            continue                      # the evaluation states it: MF 3 wins, as in the .c table
        rows[mt] = DosimetryReaction(mt, e, values)
    reactions = [rows[mt] for mt in sorted(rows)]
    reactions += _productions(suite, crossSections, omitted)
    return reactions, omitted


def pointwiseDosimetryReactions(suite) -> Tuple[List[DosimetryReaction], List[str]]:
    """``(reactions, omitted)`` of an evaluation stated pointwise (no resonance parameters).

    Every MF 3 cross section (sums included) and every MF 9/10 production as
    the evaluation tabulates it, each with its own interpolation laws: nothing
    reconstructed, linearised or broadened, which is what a dosimetry
    evaluation such as IRDFF-II is made to be used as, and what ACER copies.
    An MF 9 production is the multiplicity times the stated σ, tabulated lin-lin
    to :data:`PRODUCT_RTOL`.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    omitted: List[str] = []
    rows: Dict[int, DosimetryReaction] = {}
    forms = {}
    for container in ("reactions", "sums"):
        for reaction in getattr(suite, container):
            mt = reaction.ENDF_MT
            if mt is None:
                continue
            form = reaction.crossSection.forms.get(EVAL_LABEL) or next(iter(reaction.crossSection.forms.values()), None)
            if form is None or not hasattr(form, "toEndfRegions"):
                omitted.append(f"{reaction.label}: no tabulated cross section")
                continue
            forms[mt] = form
            row = _ownTable(int(mt), form)
            if _nonzero(row.cross_section):
                rows[int(mt)] = row
    crossSections = {mt: (r.energies, r.cross_section) for mt, r in rows.items()}
    reactions = [rows[mt] for mt in sorted(rows)]
    reactions += _productions(suite, crossSections, omitted, sigmaForms=forms)
    return reactions, omitted

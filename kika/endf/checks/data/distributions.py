"""MF4, MF5 and MF6: are the distributions probability densities, and possible?

**Densities.** A tabulated angular distribution covers mu in [-1, 1], is
nowhere negative and integrates to 1; a Legendre one has |a_l| <= 1 and no
negative region (:mod:`.legendre`, graded by the negative area). The partial
probabilities p_k(E) of MF5 add up to 1 at every energy, each tabulated
spectrum integrates to 1 and is not negative; the analytic ones have positive
parameters. An MF6 LAW=1 node's discrete lines and continuum add up to 1,
Kalbach's r is a fraction, LAW=2 and LAW=7 are densities in mu.

**Kinematics.** No product of a reaction carries more kinetic energy than the
reaction makes available: ``E + Q`` in the laboratory, ``E A/(A + a) + Q`` in
the centre of mass, with Q the larger of QM and QI of MF3. A spectrum that
reaches beyond is a warning above ``KINEMATIC_WARN`` of the limit, a defect
above ``KINEMATIC_DEFECT`` (FIZCON's EPMAX, ACER's consis).

**Products.** An MF6 section lists every product with its multiplicity. The
neutrons of an MT whose products ENDF-102 fixes (two for MT16, three for MT17,
one for MT22 ...) come with that multiplicity, and charge and mass number are
conserved: the products' ``sum y Z`` and ``sum y A`` are the target's and the
projectile's, at every incident energy (FUDGE's check). A sum short of it with
no residual listed is a note: the residual is left for the code to work out.

**Completeness.** Every reaction that emits neutrons has an angular
distribution (MF4 or MF6) and, unless it is two-body, an energy distribution
(MF5 or MF6); no distribution is given for a reaction MF3 does not have, and
no reaction has both MF6 and MF4/MF5.

Findings on the nodes of one distribution -- the incident energies of one MT
-- are reported once, with how many nodes share them.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from ..findings import DEFECT, NOTE, WARN, Finding
from .context import Out, Table, Tape, fmt_e
from .legendre import negative_area
from .reactions import emitted, is_level_mt
from .tables import check_table

#: Normalisation of a density, |integral - 1|: a note above the first, a
#: warning above the second (FUDGE's 1e-5, FIZCON's 1e-3).
NORM_NOTE = 1e-4
NORM_WARN = 1e-2
#: Negative area of a Legendre series: a note above the first, a warning above
#: the second, a defect above the third.
NEG_AREA_NOTE = 1e-6
NEG_AREA_WARN = 1e-3
NEG_AREA_DEFECT = 1e-2
#: Probabilities p_k(E) summing to 1, absolute.
PSUM_NOTE = 1e-5
PSUM_WARN = 1e-3
#: Outgoing energy beyond the kinematic limit, relative to the limit (with
#: floors of 1 keV and 10 keV). At 100-200 MeV the gap between relativistic and
#: non-relativistic kinematics, and between lab and CM for an incident photon,
#: is about 1 %: below that a spectrum cannot be faulted.
KINEMATIC_WARN = 1e-2
KINEMATIC_DEFECT = 5e-2
#: A kinematic violation that carries less than this fraction of the spectrum's
#: probability beyond the limit changes nothing measurable: a note.
KINEMATIC_MASS = 1e-4
#: A kinematic violation in a reaction whose cross section is below this
#: fraction of the total at that energy changes nothing measurable: a note.
KINEMATIC_SHARE = 1e-4
#: |a_l| or a cosine beyond 1 by more than this is not rounding.
UNIT_SLACK = 1e-6
#: Kalbach's r outside [0, 1] by more than this; a defect beyond the second.
KALBACH_SLACK = 1e-6
KALBACH_DEFECT = 1e-2
#: ZA balance of an MF6 section: tolerance on sum y Z and sum y A.
ZA_SLACK = 1e-3

FISSION_MT = frozenset({18, 19, 20, 21, 38})


# --------------------------------------------------------------------------- node findings

class _Nodes:
    """Collects findings over the nodes of one distribution; :meth:`flush` reports each once."""

    def __init__(self, out: Out, mf: int, mt: int, what: str) -> None:
        self.out, self.mf, self.mt, self.what = out, mf, mt, what
        self.scratch = Out(out.mat)
        self.total = 0

    def node(self) -> Out:
        self.total += 1
        return self.scratch

    def flush(self) -> None:
        groups: "OrderedDict[Tuple, List[Finding]]" = OrderedDict()
        for f in self.scratch.findings:
            groups.setdefault((f.check, f.level, f.location), []).append(f)
        for (check, level, loc), fs in groups.items():
            first = fs[0]
            summary = first.summary
            if len(fs) > 1:
                summary += (f" (and at {len(fs) - 1} other incident energ"
                            f"{'ies' if len(fs) > 2 else 'y'} of {self.total})")
            self.out.findings.append(Finding(check, level, summary, loc,
                                             dict(first.evidence, nodes=len(fs), of=self.total)))


def _at(e: float) -> str:
    return f"at E = {fmt_e(e)}"


def _check_grid(energies, out: Out, mf: int, mt: int, what: str, **where) -> bool:
    e = np.asarray(energies, dtype=float)
    if e.size and np.any(np.diff(e) < 0):
        i = int(np.flatnonzero(np.diff(e) < 0)[0])
        out.add("incident_grid_order", DEFECT,
                f"{what}: incident energies decrease from {fmt_e(e[i])} to {fmt_e(e[i + 1])}",
                mf=mf, mt=mt, index=i + 1, **where)
        return False
    return True


def _q_limit(tape: Tape, mt: int) -> Optional[float]:
    qm, qi = tape.q(mt)
    if qm is None or mt in FISSION_MT or mt in (5, 1, 3, 4, 27, 101):
        return None
    return max(qm, qi)


def _share(tape: Tape, mt: int, e: float) -> Optional[float]:
    """The reaction's cross section as a fraction of the total at *e*; ``None`` if MF3 cannot say."""
    t, tot = tape.xs(mt), tape.xs(1)
    if t is None or tot is None or not t.x.size or not tot.x.size:
        return None
    s, st = float(t(e)), float(tot(e))
    return s / st if st > 0 else None


def _kinematic(nodes: Out, mf: int, mt: int, what: str, e: float, e_out_max,
               q: float, lab: bool, awr: float, awi: float, tape: Optional[Tape] = None,
               spectrum=None, **where) -> None:
    """*e_out_max* is ``(top, start)``: where the spectrum's probability ends, and
    where its last interval with probability begins (a discrete line: both).
    *spectrum*, ``(line energies, line probabilities, E', f, histogram)``, lets
    the finding say how much probability lies beyond the limit."""
    e_out_max, start = e_out_max
    avail = e + q if lab else e * awr / (awr + awi) + q
    if avail <= 0:
        return
    excess = e_out_max - avail
    if excess <= max(KINEMATIC_WARN * avail, 1e3):
        return
    level = DEFECT if excess > max(KINEMATIC_DEFECT * avail, 1e4) else WARN
    share = _share(tape, mt, e) if tape is not None else None
    mass = _mass_beyond(spectrum, avail) if spectrum is not None else None
    tail = f"; {mass:.2g} of its probability lies beyond" if mass is not None else ""
    if start <= avail + max(KINEMATIC_WARN * avail, 1e3):
        # Only the last interval, which starts within the limit, crosses it: the
        # resolution of the outgoing grid (coarse bins near a threshold).
        level = NOTE
        tail = f"; only its last interval, from {fmt_e(start)}, crosses the limit"
    elif mass is not None and mass < KINEMATIC_MASS:
        level = NOTE
    elif share is not None and share < KINEMATIC_SHARE:
        level = NOTE
        tail = ("; the reaction has no cross section there" if share == 0 else
                f"; the reaction is {share:.1g} of the total cross section there")
    nodes.add("kinematic_limit", level,
              f"{what}: outgoing energies reach {fmt_e(e_out_max)} {_at(e)}, beyond the "
              f"{fmt_e(avail)} the reaction makes available ({'lab' if lab else 'CM'}, Q = {fmt_e(q)}){tail}",
              mf=mf, mt=mt, at=e, e_out=e_out_max, available=avail, q=q, share=share,
              probability_beyond=mass, **where)


def _mass_beyond(spectrum, avail: float) -> Optional[float]:
    """The fraction of a spectrum's probability above *avail*."""
    lx, lf, cx, cf, histogram = (np.asarray(v, dtype=float) if i < 4 else v for i, v in enumerate(spectrum))
    total = float(lf.sum())
    beyond = float(lf[lx > avail].sum())
    if cx.size > 1:
        dx = np.diff(cx)
        if histogram:
            total += float(np.sum(cf[:-1] * dx))
            overlap = np.clip(cx[1:] - np.maximum(cx[:-1], avail), 0.0, None)
            beyond += float(np.sum(cf[:-1] * overlap))
        else:
            total += float(np.sum((cf[:-1] + cf[1:]) / 2 * dx))
            for i in np.flatnonzero(cx[1:] > avail):
                a = max(cx[i], avail)
                fa = cf[i] + (cf[i + 1] - cf[i]) * (a - cx[i]) / dx[i] if dx[i] > 0 else cf[i]
                beyond += float((fa + cf[i + 1]) / 2 * (cx[i + 1] - a))
    return beyond / total if total > 0 else None


def _support_top(x: np.ndarray, f: np.ndarray):
    """``(top, start)``: where the probability ends -- the point after the last
    positive value -- and that last positive point."""
    pos = np.flatnonzero(f > 0)
    if not pos.size:
        return 0.0, 0.0
    i = int(pos[-1])
    return float(x[min(i + 1, x.size - 1)]), float(x[i])


def _integral(t: Table) -> float:
    from kika.algebra import integral

    laws = t.laws()
    laws = np.where(np.isin(laws, (1, 2, 3, 4, 5)), laws, 2)
    try:
        return float(integral(t.x, t.y, laws))
    except ValueError:
        return float(np.trapz(t.y, t.x))


def _norm(nodes: Out, mf: int, mt: int, what: str, e: float, s: float, **where) -> None:
    d = abs(s - 1.0)
    if d > NORM_NOTE:
        nodes.add("distribution_normalisation", WARN if d > NORM_WARN else NOTE,
                  f"{what} integrates to {s:.6g} {_at(e)}, not 1", mf=mf, mt=mt, at=e, integral=s, **where)


def _legendre(nodes: Out, coeffs, energies, mf: int, mt: int, what: str, **where) -> None:
    """|a_l| <= 1 and positivity of every node's series."""
    if not len(coeffs):
        return
    order = max((len(c) for c in coeffs), default=0)
    if order == 0:
        return
    a = np.zeros((len(coeffs), order))
    for i, c in enumerate(coeffs):
        a[i, :len(c)] = c
    e = np.asarray(energies, dtype=float)[:len(coeffs)]
    big = np.abs(a).max(axis=1) > 1 + UNIT_SLACK
    for i in np.flatnonzero(big):
        l = int(np.argmax(np.abs(a[i]))) + 1
        nodes.add("legendre_coefficient", DEFECT,
                  f"{what}: |a_{l}| = {abs(a[i, l - 1]):.6g} > 1 {_at(e[i])}, which no density has",
                  mf=mf, mt=mt, at=float(e[i]), l_order=l, value=float(a[i, l - 1]), **where)
    area, fmin = negative_area(a)
    for i in np.flatnonzero(area > NEG_AREA_NOTE):
        lv = DEFECT if area[i] > NEG_AREA_DEFECT else WARN if area[i] > NEG_AREA_WARN else NOTE
        nodes.add("legendre_negative", lv,
                  f"{what}: the Legendre series is negative {_at(e[i])}: negative area {area[i]:.2g}, "
                  f"minimum {fmin[i]:.3g}", mf=mf, mt=mt, at=float(e[i]), area=float(area[i]),
                  f_min=float(fmin[i]), **where)


def _mu_table(nodes: Out, mu, p, interp, e: float, mf: int, mt: int, what: str, **where) -> None:
    t = Table.of(mu, p, interp)
    if not check_table(t, nodes, mf=mf, mt=mt, what=what, x_name="mu", **where):
        return
    if t.x.size and (t.x[0] < -1 - UNIT_SLACK or t.x[-1] > 1 + UNIT_SLACK):
        nodes.add("cosine_range", DEFECT, f"{what}: cosines from {t.x[0]:.6g} to {t.x[-1]:.6g}, "
                  f"outside [-1, 1] {_at(e)}", mf=mf, mt=mt, at=e, mu_min=float(t.x[0]),
                  mu_max=float(t.x[-1]), **where)
    elif t.x.size and (t.x[0] > -1 + UNIT_SLACK or t.x[-1] < 1 - UNIT_SLACK):
        nodes.add("cosine_range", WARN, f"{what}: cosines only from {t.x[0]:.6g} to {t.x[-1]:.6g} "
                  f"{_at(e)}: the rest of [-1, 1] has no probability", mf=mf, mt=mt, at=e,
                  mu_min=float(t.x[0]), mu_max=float(t.x[-1]), **where)
    if t.y.size and t.y.min() < 0:
        nodes.add("distribution_negative", DEFECT, f"{what}: negative probability density "
                  f"{t.y.min():.3g} {_at(e)}", mf=mf, mt=mt, at=e, value=float(t.y.min()), **where)
    _norm(nodes, mf, mt, what, e, _integral(t), **where)


# --------------------------------------------------------------------------- MF4

def _mf4(tape: Tape, out: Out) -> None:
    for mt, sec in sorted(tape.sections(4).items()):
        ltt = getattr(sec, "_ltt", None)
        lct = getattr(sec, "_lct", None)
        what = f"MT{mt}"
        if ltt in (1, 2, 3) and lct not in (1, 2):
            out.add("frame_invalid", DEFECT, f"MT{mt}: LCT = {lct}, neither lab (1) nor CM (2)",
                    mf=4, mt=mt, lct=lct)
        if ltt in (1, 3):
            e = getattr(sec, "_energies", [])
            if _check_grid(e, out, 4, mt, what):
                nodes = _Nodes(out, 4, mt, what)
                nodes.total = len(e)
                _legendre(nodes.scratch, getattr(sec, "_legendre_coeffs", []), e, 4, mt, what)
                nodes.flush()
        if ltt in (2, 3):
            e = getattr(sec, "_energies" if ltt == 2 else "_tabulated_energies", [])
            mus = getattr(sec, "_cosines" if ltt == 2 else "_tabulated_cosines", [])
            ps = getattr(sec, "_probabilities" if ltt == 2 else "_tabulated_probabilities", [])
            ints = getattr(sec, "_angular_interpolation", [])
            if _check_grid(e, out, 4, mt, what):
                nodes = _Nodes(out, 4, mt, what)
                for k, (en, mu, p) in enumerate(zip(e, mus, ps)):
                    _mu_table(nodes.node(), mu, p, ints[k] if k < len(ints) else [], float(en), 4, mt, what)
                nodes.flush()
        _coverage(tape, out, 4, mt, _top_energy(sec))


def _top_energy(sec) -> Optional[float]:
    tops = [float(getattr(sec, n)[-1]) for n in ("_tabulated_energies", "_energies") if getattr(sec, n, None)]
    return max(tops) if tops else None


def _coverage(tape: Tape, out: Out, mf: int, mt: int, top: Optional[float]) -> None:
    """A distribution that stops below where its reaction still has a cross section."""
    t = tape.xs(mt)
    if top is None or t is None or not t.x.size:
        return
    nz = np.flatnonzero(t.y > 0)
    if not nz.size:
        return
    xs_top = float(t.x[min(int(nz[-1]) + 1, t.x.size - 1)])
    if top < xs_top * (1 - 1e-6):
        out.add("distribution_coverage", WARN,
                f"MT{mt}: MF{mf} stops at {fmt_e(top)} but MF3 has a cross section up to {fmt_e(xs_top)}: "
                "above it a code holds the last distribution", mf=mf, mt=mt, top=top, xs_top=xs_top)


# --------------------------------------------------------------------------- MF5

def _mf5(tape: Tape, out: Out) -> None:
    awr = tape.awr or 1.0
    for mt, sec in sorted(tape.sections(5).items()):
        parts = list(getattr(sec, "partials", []) or [])
        what = f"MT{mt}"
        p_tables = []
        q = _q_limit(tape, mt)
        for k, part in enumerate(parts):
            pt = Table.of(getattr(part, "p_energies", []), getattr(part, "p_values", []),
                          getattr(part, "p_interp", []))
            if check_table(pt, out, mf=5, mt=mt, what=f"{what} p[{k}]", k=k):
                p_tables.append(pt)
                if pt.y.size and (pt.y.min() < 0 or pt.y.max() > 1 + UNIT_SLACK):
                    bad = pt.y.min() if pt.y.min() < 0 else pt.y.max()
                    out.add("probability_range", DEFECT,
                            f"{what}: the probability of partial distribution {k} is {bad:.6g}, outside [0, 1]",
                            mf=5, mt=mt, k=k, value=float(bad))
            lf = getattr(part, "lf", None)
            if lf == 1:
                _mf5_tabulated(tape, out, part, mt, k, q, awr)
            elif lf in (5, 7, 9, 11, 12):
                _mf5_analytic(out, part, mt, k, q)
            else:
                out.add("spectrum_law", DEFECT, f"{what}: partial {k} has LF = {lf}, not a law ENDF-6 defines",
                        mf=5, mt=mt, k=k, lf=lf)
        if len(p_tables) == len(parts) and p_tables:
            _probability_sum(out, 5, mt, what, p_tables)
        tops = [float(t.x[-1]) for t in p_tables if t.x.size]
        if mt != 455:
            _coverage(tape, out, 5, mt, min(tops) if tops else None)


def _probability_sum(out: Out, mf: int, mt: int, what: str, tables: List[Table]) -> None:
    from kika.algebra import union

    grid = union([t.x for t in tables])
    lo = max(t.x[0] for t in tables)
    hi = min(t.x[-1] for t in tables)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if not grid.size:
        return
    total = sum(t(grid) for t in tables)
    d = np.abs(total - 1.0)
    i = int(np.argmax(d))
    if d[i] > PSUM_NOTE:
        out.add("probability_sum", WARN if d[i] > PSUM_WARN else NOTE,
                f"{what}: the probabilities of its {len(tables)} partial distribution"
                f"{'s' if len(tables) != 1 else ''} add up to {total[i]:.6g} at {fmt_e(grid[i])}, not 1",
                mf=mf, mt=mt, at=float(grid[i]), total=float(total[i]))


def _mf5_tabulated(tape, out, part, mt, k, q, awr) -> None:
    what = f"MT{mt} partial {k}"
    e_in = getattr(part, "incident_energies", [])
    if not _check_grid(e_in, out, 5, mt, what, k=k):
        return
    nodes = _Nodes(out, 5, mt, what)
    for j, e in enumerate(e_in):
        scratch = nodes.node()
        t = Table.of(part.outgoing_grids[j], part.chi[j],
                     part.outgoing_interp[j] if j < len(part.outgoing_interp) else [])
        if not check_table(t, scratch, mf=5, mt=mt, what=what, x_name="E'", k=k):
            continue
        if t.y.size and t.y.min() < 0:
            scratch.add("distribution_negative", DEFECT, f"{what}: negative spectrum {t.y.min():.3g} {_at(float(e))}",
                        mf=5, mt=mt, at=float(e), value=float(t.y.min()), k=k)
        if t.x.size > 1:
            _norm(scratch, 5, mt, what, float(e), _integral(t), k=k)
        if q is not None:
            hist = bool(t.pairs) and all(code == 1 for _, code in t.pairs)
            _kinematic(scratch, 5, mt, what, float(e), _support_top(t.x, t.y), q, True, awr, tape.awi,
                       tape=tape, spectrum=((), (), t.x, t.y, hist), k=k)
    nodes.flush()


def _mf5_analytic(out, part, mt, k, q) -> None:
    what = f"MT{mt} partial {k}"
    lf = part.lf
    names = {5: [("theta", "theta")], 7: [("theta", "theta")], 9: [("theta", "theta")],
             11: [("a", "a"), ("b", "b")], 12: [("tm", "T_M")]}[lf]
    for attr, label in names:
        v = np.asarray(getattr(part, f"{attr}_values", []) or [], dtype=float)
        if v.size and v.min() <= 0:
            out.add("spectrum_parameter", DEFECT,
                    f"{what} (LF = {lf}): {label} = {v.min():.6g}, not positive", mf=5, mt=mt, k=k,
                    parameter=label, value=float(v.min()))
    if lf == 12:
        for attr in ("efl", "efh"):
            v = float(getattr(part, attr, 0.0) or 0.0)
            if v <= 0:
                out.add("spectrum_parameter", DEFECT, f"{what} (LF = 12): {attr.upper()} = {v:g}, not positive",
                        mf=5, mt=mt, k=k, parameter=attr.upper(), value=v)
    u = float(getattr(part, "u", 0.0) or 0.0)
    if lf in (7, 9) and q is not None and q < 0 and mt not in FISSION_MT and mt != 455:
        # E' runs up to E - U; above E + Q there is no energy for it.
        if -u > q + max(KINEMATIC_WARN * abs(q), 1e3):
            out.add("kinematic_limit", WARN,
                    f"{what} (LF = {lf}): U = {fmt_e(u)} lets E' reach E - U, beyond the E + Q "
                    f"(Q = {fmt_e(q)}) the reaction makes available", mf=5, mt=mt, k=k, u=u, q=q)


# --------------------------------------------------------------------------- MF6

def _mf6(tape: Tape, out: Out) -> None:
    awr = tape.awr or 1.0
    awi = tape.awi
    for mt, sec in sorted(tape.sections(6).items()):
        lct = getattr(sec, "_lct", None)
        if lct not in (1, 2, 3, 4):
            out.add("frame_invalid", DEFECT, f"MT{mt}: LCT = {lct}, not a frame MF6 defines",
                    mf=6, mt=mt, lct=lct)
        q = _q_limit(tape, mt)
        yields = []
        for k, prod in enumerate(list(getattr(sec, "products", []) or [])):
            zap = int(round(float(getattr(prod, "zap", 0.0))))
            what = f"MT{mt} product {k} (ZAP = {zap})"
            yt = Table.of(prod.y_energies, prod.y_values, prod.y_interp)
            ok = check_table(yt, out, mf=6, mt=mt, what=f"{what} multiplicity", k=k, zap=zap)
            if ok and yt.y.size and yt.y.min() < 0:
                # A value a millionth of the largest below zero is the rounding of a zero.
                tiny = -yt.y.min() <= 1e-6 * max(float(np.abs(yt.y).max()), 1e-300)
                out.add("multiplicity_negative", NOTE if tiny else DEFECT,
                        f"{what}: negative multiplicity {yt.y.min():.6g}",
                        mf=6, mt=mt, k=k, zap=zap, value=float(yt.y.min()))
            yields.append((zap, yt if ok else None))
            # LCT = 3: light particles in the CM, heavy recoils in the lab; photons are lab always.
            lab = lct == 1 or (lct == 3 and zap > 2004) or zap == 0
            _law(tape, out, mt, k, zap, prod.law_data, q, lab, awr, awi)
        _products(tape, out, mt, yields)
        tops = [float(yt.x[-1]) for _, yt in yields if yt is not None and yt.x.size]
        _coverage(tape, out, 6, mt, min(tops) if tops else None)


def _law(tape, out, mt, k, zap, body, q, lab, awr, awi) -> None:
    law = getattr(body, "law", None)
    what = f"MT{mt} product {k} (ZAP = {zap})"
    if law == 1:
        e_in = body.incident_energies
        if not _check_grid(e_in, out, 6, mt, what, k=k, zap=zap):
            return
        nodes = _Nodes(out, 6, mt, what)
        for j, e in enumerate(e_in):
            scratch = nodes.node()
            e = float(e)
            try:
                block = body.block(j)
            except ValueError:
                scratch.add("law_record", DEFECT, f"{what}: the LAW=1 record {_at(e)} does not hold "
                            "whole groups of NA + 2 values", mf=6, mt=mt, k=k, zap=zap, at=e)
                continue
            nd = int(body.nd[j])
            x, f0 = block[:, 0], block[:, 1]
            disc_x, disc_f = x[:nd], f0[:nd]
            cx, cf = x[nd:], f0[nd:]
            if cx.size > 1 and np.any(np.diff(cx) < 0):
                scratch.add("outgoing_grid_order", DEFECT, f"{what}: outgoing energies decrease {_at(e)}",
                            mf=6, mt=mt, k=k, zap=zap, at=e)
                continue
            if f0.size and f0.min() < 0:
                scratch.add("distribution_negative", DEFECT, f"{what}: negative spectrum {f0.min():.3g} {_at(e)}",
                            mf=6, mt=mt, k=k, zap=zap, at=e, value=float(f0.min()))
            s_cont = 0.0
            if cx.size > 1:
                s_cont = float(np.sum(cf[:-1] * np.diff(cx))) if body.lep == 1 else float(np.trapz(cf, cx))
            if x.size:
                _norm(scratch, 6, mt, what, e, float(disc_f.sum()) + s_cont, k=k, zap=zap)
            if body.lang == 2 and block.shape[1] >= 3:
                r = block[:, 2]
                bad = (r < -KALBACH_SLACK) | (r > 1 + KALBACH_SLACK)
                if bad.any():
                    i = int(np.flatnonzero(bad)[0])
                    off = max(-r.min(), r.max() - 1.0)
                    scratch.add("kalbach_fraction", DEFECT if off > KALBACH_DEFECT else WARN,
                                f"{what}: Kalbach's pre-compound fraction r = {r[i]:.6g} {_at(e)}, outside [0, 1]",
                                mf=6, mt=mt, k=k, zap=zap, at=e, r=float(r[i]))
            if body.lang == 1 and block.shape[1] > 2:
                pos = f0 > 0
                if pos.any():
                    a = block[pos, 2:] / f0[pos, None]
                    _legendre(scratch, a, np.full(int(pos.sum()), e), 6, mt, what, k=k, zap=zap)
            if q is not None and x.size:
                # A negative discrete photon energy is a primary photon: |E'| + E A/(A+1).
                gx = np.abs(disc_x) + (np.where(disc_x < 0, e * awr / (awr + 1), 0.0) if zap == 0 else 0.0)
                live = gx[disc_f > 0]          # lines with no probability are not emitted
                tops = [(float(live.max()), float(live.max()))] if live.size else []
                if cx.size and np.any(cf > 0):
                    tops.append(_support_top(cx, cf))
                if tops:
                    _kinematic(scratch, 6, mt, what, e, max(tops), q, lab, awr, awi, tape=tape,
                               spectrum=(gx, disc_f, cx, cf, body.lep == 1), k=k, zap=zap)
        nodes.flush()
    elif law == 2:
        e_in = body.incident_energies
        if not _check_grid(e_in, out, 6, mt, what, k=k, zap=zap):
            return
        nodes = _Nodes(out, 6, mt, what)
        leg_e, leg_c = [], []
        for j, e in enumerate(e_in):
            scratch = nodes.node()
            lang = body.lang[j]
            vals = body.values[j]
            if lang == 0:
                leg_e.append(float(e))
                leg_c.append(list(vals))
            elif lang in (12, 14, 15):
                arr = np.asarray(vals, dtype=float)
                if arr.size % 2:
                    scratch.add("law_record", DEFECT, f"{what}: LAW=2 {_at(float(e))} holds an odd number "
                                "of (mu, f) values", mf=6, mt=mt, k=k, zap=zap, at=float(e))
                    continue
                mu, p = arr[0::2], arr[1::2]
                code = {12: 2, 14: 4, 15: 5}[lang]
                _mu_table(scratch, mu, p, [(mu.size, code)], float(e), 6, mt, what, k=k, zap=zap)
            else:
                scratch.add("law_record", DEFECT, f"{what}: LAW=2 LANG = {lang} {_at(float(e))} is not 0, 12, "
                            "14 or 15", mf=6, mt=mt, k=k, zap=zap, at=float(e), lang=lang)
        if leg_c:
            _legendre(nodes.scratch, leg_c, leg_e, 6, mt, what, k=k, zap=zap)
        nodes.flush()
    elif law == 7:
        nodes = _Nodes(out, 6, mt, what)
        for blk in body.blocks:
            scratch = nodes.node()
            e = float(blk.energy)
            mus = np.array([a.mu for a in blk.angles], dtype=float)
            if mus.size and (mus.min() < -1 - UNIT_SLACK or mus.max() > 1 + UNIT_SLACK or np.any(np.diff(mus) < 0)):
                scratch.add("cosine_range", DEFECT, f"{what}: LAW=7 cosines {_at(e)} are outside [-1, 1] or "
                            "not increasing", mf=6, mt=mt, k=k, zap=zap, at=e)
                continue
            dens = []
            top = (0.0, 0.0)
            for a in blk.angles:
                t = Table.of(a.e_out, a.f, a.interp)
                if not check_table(t, scratch, mf=6, mt=mt, what=what, x_name="E'", k=k, zap=zap):
                    dens = None
                    break
                if t.y.size and t.y.min() < 0:
                    scratch.add("distribution_negative", DEFECT, f"{what}: negative density {t.y.min():.3g} {_at(e)}",
                                mf=6, mt=mt, k=k, zap=zap, at=e, value=float(t.y.min()))
                dens.append(_integral(t) if t.x.size > 1 else 0.0)
                top = max(top, _support_top(t.x, t.y))
            if dens and mus.size > 1:
                _norm(scratch, 6, mt, what, e, float(np.trapz(dens, mus)), k=k, zap=zap)
            if q is not None and top[0]:
                _kinematic(scratch, 6, mt, what, e, top, q, True, awr, awi, tape=tape, k=k, zap=zap)
        nodes.flush()


def _products(tape: Tape, out: Out, mt: int, yields) -> None:
    """Neutron multiplicity by the MT, and charge and mass number conserved."""
    za_t = tape.za
    if za_t is None or mt in FISSION_MT or mt == 5 or mt >= 201:
        return
    good = [(zap, yt) for zap, yt in yields if yt is not None]
    expected = emitted(mt)
    # A residual that breaks up (LR > 0) adds its own particles to the MT's.
    if expected is not None and tape.neutron_projectile and not tape.lr(mt):
        n_exp = expected.get(1, 0) + (1 if mt == 2 else 0)
        n_tabs = [yt for zap, yt in good if zap == 1 and yt.y.size]
        if n_tabs and all(np.all(yt.y == yt.y[0]) for yt in n_tabs):
            n = sum(float(yt.y[0]) for yt in n_tabs)
            if abs(n - n_exp) > 1e-6:
                out.add("multiplicity_reaction", DEFECT,
                        f"MT{mt}: {n:g} neutron{'s' if n != 1 else ''} per reaction where the MT means "
                        f"{n_exp}", mf=6, mt=mt, neutrons=n, expected=n_exp)
    if za_t % 1000 == 0:
        return
    massive = [(zap, yt) for zap, yt in good if zap > 0 and zap != 11 and yt.x.size]
    if not massive:
        return
    from kika.algebra import union

    proj = tape.projectile[1]
    z_tot = za_t // 1000 + max(proj, 0) // 1000
    a_tot = za_t % 1000 + max(proj, 0) % 1000
    grid = union([yt.x for _, yt in massive])
    lo = max(yt.x[0] for _, yt in massive)
    hi = min(yt.x[-1] for _, yt in massive)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if not grid.size:
        return
    zs = np.zeros(grid.size)
    as_ = np.zeros(grid.size)
    for zap, yt in massive:
        y = yt(grid)
        zs += y * (zap // 1000)
        as_ += y * (zap % 1000)
    dz, da = zs - z_tot, as_ - a_tot
    i = int(np.argmax(np.abs(da) + np.abs(dz)))
    if abs(dz[i]) <= ZA_SLACK and abs(da[i]) <= ZA_SLACK:
        return
    # Short of the total with no residual listed is how most MF6 sections are
    # written (only the particles a code transports): not a finding.
    has_residual = any((zap % 1000) > 4 for zap, _ in massive)
    over = dz[i] > ZA_SLACK or da[i] > ZA_SLACK
    if not over and not has_residual:
        return
    bad = int(np.sum((np.abs(dz) > ZA_SLACK) | (np.abs(da) > ZA_SLACK)))
    out.add("za_balance", DEFECT,
            f"MT{mt}: the products carry Z = {zs[i]:.6g}, A = {as_[i]:.6g} at {fmt_e(grid[i])}, where target "
            f"and projectile bring {z_tot}, {a_tot}: charge or mass number is not conserved "
            f"({bad} of {grid.size} energies)",
            mf=6, mt=mt, at=float(grid[i]), z=float(zs[i]), a=float(as_[i]), z_total=z_tot, a_total=a_tot,
            energies=bad)


# --------------------------------------------------------------------------- completeness

def _completeness(tape: Tape, out: Out) -> None:
    mf3 = tape.sections(3)
    mf4, mf5, mf6 = (set(tape.sections(m)) for m in (4, 5, 6))
    if not mf3 or not tape.neutron_projectile:
        return
    present = set(mf3)
    from kika._constants import mf3SumComponents

    # Fission's spectra are written under MT18 and stand for its chances
    # (MT19-21, 38), which carry cross sections only: that is the convention
    # processors read. MT18 itself must then have them.
    fission_dists = 18 in mf4 | mf5 | mf6
    for mt in sorted(present):
        if mt in (1, 3, 27, 101) or (mt in (4, 16, 103, 104, 105, 106, 107)
                                     and [p for p in mf3SumComponents(mt, present) if p != mt]):
            continue
        if mt in (19, 20, 21, 38) and fission_dists:
            continue
        prods = emitted(mt)
        neutrons = mt in FISSION_MT or mt == 2 or (prods is not None and 1 in prods)
        if not neutrons:
            continue
        if mt not in mf4 and mt not in mf6:
            out.add("distribution_missing", WARN,
                    f"MT{mt} emits neutrons but has no angular distribution (MF4 or MF6): codes assume "
                    "isotropy", mf=3, mt=mt, missing="angle")
        two_body = mt == 2 or is_level_mt(mt)
        if not two_body and mt not in mf5 and mt not in mf6:
            out.add("distribution_missing", DEFECT,
                    f"MT{mt} emits neutrons but has no energy distribution (MF5 or MF6): there is no way to "
                    "give them an energy", mf=3, mt=mt, missing="energy")
    for mf, mts in ((4, mf4), (5, mf5), (6, mf6)):
        orphans = sorted(m for m in mts if m not in present and m != 455 and m < 201)
        if orphans:
            out.add("distribution_orphan", WARN,
                    f"MF{mf} gives {len(orphans)} distribution{'s' if len(orphans) != 1 else ''} for reactions "
                    f"MF3 does not have: {', '.join(f'MT{m}' for m in orphans[:8])}"
                    f"{'...' if len(orphans) > 8 else ''}", mf=mf, mt=orphans[0], mts=orphans)
    # MF6 for the photons (prompt fission photons in MT18) beside MF4/MF5 for the
    # neutrons is how evaluations are written; a duplicate is MF6 giving the
    # neutrons too.
    def gives_neutrons(mt):
        return any(int(round(float(getattr(p, "zap", 0)))) == 1 and getattr(p.law_data, "law", 0) > 0
                   for p in getattr(tape.section(6, mt), "products", []) or [])

    both = sorted(m for m in mf6 if (m in mf4 or m in mf5) and gives_neutrons(m))
    if both:
        out.add("distribution_duplicate", WARN,
                f"{len(both)} reaction{'s' if len(both) != 1 else ''} with neutrons in both MF6 and MF4/MF5: "
                f"{', '.join(f'MT{m}' for m in both[:8])}; ENDF-6 gives one or the other, and codes differ in "
                "which they take", mf=6, mt=both[0], mts=both)


def check_distributions(tape: Tape, out: Out) -> None:
    _mf4(tape, out)
    _mf5(tape, out)
    _mf6(tape, out)
    _completeness(tape, out)


# --------------------------------------------------------------------------- legend and method

def descriptions(D, _g, _pct):
    ALL = (4, 5, 6)
    return {
        "incident_grid_order": D(
            "Incident energies decrease", ALL + (14, 15), {DEFECT: "always"},
            "The incident energies of a distribution (a TAB2) never decrease."),
        "outgoing_grid_order": D(
            "Outgoing energies decrease", (6,), {DEFECT: "always"},
            "The outgoing energies of a LAW=1 continuum never decrease."),
        "frame_invalid": D(
            "Undefined frame of reference", ALL, {DEFECT: "always"},
            "LCT is 1 (lab) or 2 (CM) in MF4, 1-4 in MF6; any other value leaves the frame unknown."),
        "legendre_coefficient": D(
            "Legendre coefficient beyond 1", (4, 6, 14), {DEFECT: "always"},
            "|a_l| <= 1 for every density in mu, because |P_l| <= 1 (FIZCON)."),
        "legendre_negative": D(
            "Negative Legendre distribution", (4, 6, 14),
            {DEFECT: f"negative area above {_g(NEG_AREA_DEFECT)}",
             WARN: f"above {_g(NEG_AREA_WARN)}", NOTE: f"above {_g(NEG_AREA_NOTE)}"},
            "The series sum_l (2l+1)/2 a_l P_l(mu) is negative somewhere in [-1, 1]. Graded by the "
            "negative area, the probability a sampler has to clip (PSYCHE's Sturm test and PREPRO's "
            "LEGEND look for the same). Truncated forward-peaked series dip slightly negative."),
        "cosine_range": D(
            "Cosines outside or short of [-1, 1]", ALL + (14,),
            {DEFECT: "a cosine outside [-1, 1]", WARN: "the table does not reach -1 or 1"},
            "A tabulated angular distribution spans mu from -1 to 1 (FUDGE: severe if it does not)."),
        "distribution_negative": D(
            "Negative probability density", ALL + (14, 15), {DEFECT: "always"},
            "A tabulated angular, energy or energy-angle density is negative somewhere."),
        "distribution_normalisation": D(
            "Distribution not normalised", ALL + (14, 15),
            {WARN: f"off 1 by more than {_g(NORM_WARN)}", NOTE: f"off 1 by more than {_g(NORM_NOTE)}"},
            "A density integrates to 1 at every incident energy (FUDGE 1e-5, FIZCON 1e-3; for MF6 "
            "LAW=1 the discrete lines count with the continuum). Processors renormalise, so a small "
            "gap is rounding and a large one is a distribution that does not say what it means."),
        "distribution_coverage": D(
            "Distribution stops before its cross section", ALL, {WARN: "always"},
            "MF4/MF5/MF6 of a reaction stop below the last energy where MF3 has a cross section."),
        "probability_range": D(
            "Probability outside [0, 1]", (5,), {DEFECT: "always"},
            "The weight p_k(E) of a partial energy distribution is a probability."),
        "probability_sum": D(
            "Partial probabilities do not add up to 1", (5,),
            {WARN: f"off by more than {_g(PSUM_WARN)}", NOTE: f"off by more than {_g(PSUM_NOTE)}"},
            "The partial distributions of MF5 share the reaction: their weights add up to 1 at every "
            "energy (FUDGE's test only fires when the sum is never 1; this one at every energy)."),
        "spectrum_law": D(
            "Undefined energy law", (5,), {DEFECT: "always"},
            "LF must be 1, 5, 7, 9, 11 or 12."),
        "spectrum_parameter": D(
            "Non-positive spectrum parameter", (5,), {DEFECT: "always"},
            "Temperatures (theta, T_M), Watt's a and b and Madland-Nix's fragment energies are positive "
            "(FUDGE: severe)."),
        "kinematic_limit": D(
            "Outgoing energy beyond what the reaction makes available", (5, 6),
            {DEFECT: f"by more than {_pct(KINEMATIC_DEFECT)} of the limit",
             WARN: f"by more than {_pct(KINEMATIC_WARN)} of it",
             NOTE: f"either, with less than {_g(KINEMATIC_MASS)} of the probability beyond the limit, in "
                   f"a reaction below {_g(KINEMATIC_SHARE)} of the total cross section there, or only in a "
                   "last interval that starts within the limit"},
            "No product carries more kinetic energy than E + Q in the laboratory, or E A/(A+a) + Q in "
            "the centre of mass, with Q the larger of QM and QI (FIZCON's EPMAX, ACER's consis). For "
            "MF5 LF = 7 and 9 the restriction energy U must keep E' = E - U within it. Where the "
            "reaction is a negligible share of the total (near its threshold, typically) the "
            "consequence is too, and the finding is a note. Found in ENDF/B releases: Ar-40, "
            "Sn-120, Hf."),
        "law_record": D(
            "Malformed MF6 law record", (6,), {DEFECT: "always"},
            "An MF6 law body whose values cannot be read as the law says (LAW=1 groups of NA + 2, "
            "LAW=2 (mu, f) pairs, a LANG the law does not define)."),
        "kalbach_fraction": D(
            "Kalbach pre-compound fraction outside [0, 1]", (6,),
            {DEFECT: f"by more than {_g(KALBACH_DEFECT)}", WARN: "by less"},
            "Kalbach-Mann's r is the pre-compound fraction of the emission (FUDGE: severe; ACER). Found in "
            "the ENDF/B-VIII.1 photonuclear sublibrary (r = 2 in Pb-206 and Pb-207), as NNDC reported."),
        "multiplicity_negative": D(
            "Negative multiplicity", (6,),
            {DEFECT: "always", NOTE: "within a millionth of the largest value: the rounding of a zero"},
            "A product's yield is not negative."),
        "multiplicity_reaction": D(
            "Neutron multiplicity contradicts the MT", (6,), {DEFECT: "always"},
            "MT16 emits two neutrons, MT17 three, MT22 one ...: a constant neutron multiplicity "
            "other than the MT's is a mislabelled reaction or a wrong yield. Reactions whose residual "
            "breaks up (LR > 0) are left out: the breakup adds particles."),
        "za_balance": D(
            "Charge or mass number not conserved", (6,),
            {DEFECT: "the products carry more than target and projectile, or less with the residual listed"},
            "The products of an MF6 section, weighted by their multiplicities, carry the charge and "
            "mass number of target plus projectile at every energy (FUDGE: fatal). A section that lists "
            "only the particles a code transports, without the residual, is not flagged. Found in the "
            "ENDF/B-VIII.1 photonuclear sublibrary (114 of 426 points)."),
        "distribution_missing": D(
            "Reaction without a distribution", (3, 4, 5, 6),
            {DEFECT: "neutrons with no energy distribution",
             WARN: "neutrons with no angular distribution"},
            "Every reaction that emits neutrons needs an angular distribution, and, unless it is "
            "two-body, an energy distribution (FUDGE: severe). Without the first codes assume isotropy; "
            "without the second there is no energy to give the neutrons."),
        "distribution_orphan": D(
            "Distribution for a reaction MF3 lacks", ALL, {WARN: "always"},
            "A distribution weighs nothing without its cross section."),
        "distribution_duplicate": D(
            "Neutrons in both MF6 and MF4/MF5", ALL, {WARN: "always"},
            "A reaction's neutrons have their distributions in MF6 or in MF4/MF5, not both; codes "
            "differ in which they take. MF6 for its photons beside MF4/MF5 is normal."),
    }


def method(_g, _pct):
    return [
        ("Distributions",
         "MF4, MF5 and MF6 as probability densities, against kinematics and against the products the "
         "MT names. Findings on the incident energies of one distribution are reported once.",
         [(DEFECT, f"a negative density or multiplicity; |a_l| > 1; a Legendre negative area above "
                   f"{_g(NEG_AREA_DEFECT)}; Kalbach's r outside [0, 1]; outgoing energy more than "
                   f"{_pct(KINEMATIC_DEFECT)} beyond E + Q; charge or mass not conserved; a neutron "
                   "multiplicity the MT contradicts; neutrons with no energy distribution."),
          (WARN, f"normalisation off by more than {_g(NORM_WARN)}; partial probabilities off 1 by more "
                 f"than {_g(PSUM_WARN)}; a Legendre negative area above {_g(NEG_AREA_WARN)}; "
                 "cosines short of [-1, 1]; neutrons with no angular distribution; a distribution that "
                 "stops before its cross section, has no cross section, or is given twice."),
          (NOTE, f"normalisation off by more than {_g(NORM_NOTE)}; small negative Legendre areas; a "
                 f"kinematic violation in a reaction below {_g(KINEMATIC_SHARE)} of the total.")]),
    ]

"""MF3: the cross sections, their thresholds, Q values, levels and sums.

**Values.** Every table gets the structural checks of :mod:`.tables`. A
negative cross section is a defect, except inside the resolved resonance
range of a tape whose MF2 adds the resonances (LRP = 1): there MF3 is a
background, which may be negative where the resonances make up for it (a
note; whether they do is the reconstruction's check).

**Threshold.** A reaction with QI < 0 cannot happen below
``E_th = -QI (A + a) / A`` (relativistically a little more, see
:func:`.reactions.threshold`). A cross section that is non-zero below it is a
defect if it is by more than ``THRESHOLD_DEFECT`` of E_th, a warning above
``THRESHOLD_WARN``; closer than that is how evaluators round the threshold.
A table that starts at its threshold with a non-zero value starts with a step
(a note: NJOY moves such points, FUDGE calls it severe).

**Q against masses.** QM is the ground-state Q of the products the MT names
(ENDF-102 Appendix B), which AME2020 gives. Levels and continua of one family
share the family's QM. Reactions whose residual breaks up (LR > 0), lumped
and fission MTs, elemental targets and nuclei off the mass table are left
out.

**Levels.** Within a family (MT51-91, 600-649, ...) the excitation of each
level, QM - QI, is non-negative and grows with the MT, and the continuum
starts at or above the last discrete level.

**Sums.** A redundant reaction stated in MF3 (MT1, 3, 4, 16, 18, 101,
103-107, 27) against the sum of its partials, on the union of their grids
(the rules of :data:`kika._constants.MF3_SUM_RULES`).

**Coverage.** A reaction that stops below EMAX at a non-zero value drops to
zero there.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Table, Tape, fmt_b, fmt_e
from .reactions import (emitted, is_level_mt, level_family, q_from_masses, threshold)
from .tables import check_table

#: Non-zero cross section below the kinematic threshold, relative to E_th.
THRESHOLD_WARN = 1e-3
THRESHOLD_DEFECT = 1e-2
#: QM against AME2020, eV: a note above max(Q_NOTE_EV, Q_NOTE_REL |Q|), a
#: warning above max(Q_WARN_EV, Q_WARN_REL |Q|).
Q_NOTE_EV, Q_NOTE_REL = 1e3, 1e-3
Q_WARN_EV, Q_WARN_REL = 1e5, 1e-2
#: Level energies agree within this (eV) -- one keV, FUDGE's dQ.
LEVEL_TOLERANCE = 1e3
#: A stated sum against its partials: a note above SUM_NOTE relative (with
#: a floor of SUM_FLOOR of the sum's peak), a warning where the gap is both
#: above SUM_MATERIAL_REL of the stated value and above SUM_MATERIAL_B --
#: the reconstruction report's "material" disagreement.
SUM_NOTE = 1e-3
SUM_FLOOR = 1e-6
SUM_MATERIAL_REL = 1e-2
SUM_MATERIAL_B = 1e-3
#: A material gap at no more than this many nodes, and no larger than
#: SUM_FEW_REL, is a local slip (a node at a partial's threshold): a note.
SUM_FEW_POINTS = 3
SUM_FEW_REL = 5e-2
FISSION_MT = frozenset({18, 19, 20, 21, 38})


def check_xs(tape: Tape, out: Out) -> None:
    secs = tape.sections(3)
    if not secs:
        return
    awr = tape.awr
    rr_top = tape.resolved_upper() if tape.lrp == 1 else None
    background = tape.resonance_mts() if rr_top else frozenset()
    tables: Dict[int, Table] = {}
    for mt in sorted(secs):
        t = tape.xs(mt)
        if t is None:
            continue
        if not check_table(t, out, mf=3, mt=mt, what=f"MT{mt}"):
            continue
        tables[mt] = t
        _negative(t, mt, rr_top, background, out)
        if awr:
            _threshold(tape, t, mt, awr, out)
        _coverage(tape, t, mt, out)
    _q_values(tape, out)
    _levels(tape, out)
    _sums(tape, tables, set(secs), rr_top, out)


def _negative(t: Table, mt: int, rr_top: Optional[float], background, out: Out) -> None:
    neg = np.flatnonzero(t.y < 0)
    if not neg.size:
        return
    i = int(neg[np.argmin(t.y[neg])])
    inside = rr_top is not None and rr_top > 0 and t.x[neg].max() <= rr_top * (1 + 1e-9)
    if inside and mt in background:
        out.add("xs_negative", NOTE,
                f"MT{mt}: negative background at {neg.size} point{'s' if neg.size != 1 else ''} inside "
                f"the resolved range (down to {fmt_b(t.y[i])} at {fmt_e(t.x[i])}); the resonances "
                "of MF2 are added to it", mf=3, mt=mt, count=int(neg.size), value=float(t.y[i]),
                at=float(t.x[i]), background=True)
        return
    out.add("xs_negative", DEFECT,
            f"MT{mt}: negative cross section at {neg.size} point{'s' if neg.size != 1 else ''}, down to "
            f"{fmt_b(t.y[i])} at {fmt_e(t.x[i])}", mf=3, mt=mt, count=int(neg.size),
            value=float(t.y[i]), at=float(t.x[i]), background=False)


def _threshold(tape: Tape, t: Table, mt: int, awr: float, out: Out) -> None:
    qm, qi = tape.q(mt)
    if (qi is None or qi >= 0 or mt in FISSION_MT or mt in (1, 3, 4, 5, 10, 27, 101)
            or 201 <= mt <= 207 or not t.x.size):     # 201-207: particle production, no one Q
        return
    e_rel = threshold(qi, awr, tape.awi)
    e_non = -qi * (awr + tape.awi) / awr
    e_th = min(e_rel, e_non)
    nz = np.flatnonzero(t.y != 0)
    if not nz.size:
        return
    first = int(nz[0])
    e1 = float(t.x[first])
    if first > 0 and t.x[first - 1] == e1:
        first -= 1      # a step at e1: the value on its right is what counts
    short = (e_th - e1) / e_th
    if short > THRESHOLD_WARN:
        level = DEFECT if short > THRESHOLD_DEFECT else WARN
        out.add("xs_below_threshold", level,
                f"MT{mt}: {fmt_b(t.y[nz[0]])} at {fmt_e(e1)}, below the threshold {fmt_e(e_th)} that "
                f"QI = {fmt_e(qi)} allows ({short:.2g} short)", mf=3, mt=mt, at=e1,
                threshold=e_th, qi=qi, short=short, value=float(t.y[nz[0]]))
        return
    if nz[0] == 0 and abs(e1 - e_th) <= THRESHOLD_DEFECT * e_th:
        out.add("xs_threshold_step", NOTE,
                f"MT{mt} starts at its threshold {fmt_e(e1)} with {fmt_b(t.y[0])}, not zero: a step",
                mf=3, mt=mt, at=e1, value=float(t.y[0]), threshold=e_th)


def _coverage(tape: Tape, t: Table, mt: int, out: Out) -> None:
    emax = tape.emax
    if emax is None or not t.x.size:
        return
    last = float(t.x[-1])
    if last < emax * (1 - 1e-6) and t.y[-1] != 0:
        out.add("xs_coverage", WARN,
                f"MT{mt} stops at {fmt_e(last)} with {fmt_b(t.y[-1])}, below EMAX = {fmt_e(emax)}: it "
                "drops to zero there", mf=3, mt=mt, last=last, value=float(t.y[-1]), emax=emax)


def _q_values(tape: Tape, out: Out) -> None:
    za = tape.za
    if za is None or za % 1000 == 0:
        return
    proj = tape.projectile[1]
    # The levels and continuum of one family share their QM: one finding for
    # the family, at its first MT, not one per level.
    seen_family = set()
    from kika._constants import mf3SumComponents

    present = set(tape.sections(3))
    for mt, sec in sorted(tape.sections(3).items()):
        if mt in FISSION_MT or mt == 2 or mt in (1, 3, 5, 10, 27, 101, 4) or tape.lr(mt):
            continue
        # A redundant total (MT103 over MT600-649 ...) has no Q of its own:
        # evaluators write 0 or the first partial's, both conventions.
        if [p for p in mf3SumComponents(mt, present) if p != mt]:
            continue
        if emitted(mt) is None:
            continue
        expected = q_from_masses(za, proj, mt, elis=tape.elis)
        if expected is None:
            continue
        qm, qi = tape.q(mt)
        diff = abs(qm - expected)
        if tape.elis:
            # An isomeric target: whether QM counts the target's excitation is
            # written both ways; take the reading closer to the file.
            alt = expected - tape.elis
            if abs(qm - alt) < diff:
                expected, diff = alt, abs(qm - alt)
        note_at = max(Q_NOTE_EV, Q_NOTE_REL * abs(expected))
        warn_at = max(Q_WARN_EV, Q_WARN_REL * abs(expected))
        if diff <= note_at:
            continue
        fam = level_family(mt)
        if fam is not None:
            key = (fam, round(qm))
            if key in seen_family:
                continue
            seen_family.add(key)
            same = [m for m in tape.sections(3)
                    if level_family(m) == fam and round(tape.q(m)[0]) == round(qm)]
            what = f"MT{min(same)}-{max(same)} ({len(same)} sections)" if len(same) > 1 else f"MT{mt}"
        else:
            what = f"MT{mt}"
        level = WARN if diff > warn_at else NOTE
        if qm == 0 and abs(expected) > warn_at:
            level = DEFECT
        out.add("q_mass", level,
                f"{what}: QM = {fmt_e(qm)}, AME2020 gives {fmt_e(expected)} (off by {fmt_e(diff)})",
                mf=3, mt=mt, qm=qm, ame=expected, difference=qm - expected)


def _levels(tape: Tape, out: Out) -> None:
    secs = tape.sections(3)
    fams: Dict[Tuple[int, int, int], List[int]] = {}
    for mt in secs:
        fam = level_family(mt)
        if fam is not None:
            fams.setdefault(fam, []).append(mt)
    for fam, mts in fams.items():
        mts.sort()
        # A level whose residual breaks up (LR > 0) states the Q of the breakup
        # as its QM: the family's ground-state Q comes from the others, and so
        # does every level's excitation.
        plain = [(mt, *tape.q(mt)) for mt in mts if mt <= fam[1] and not tape.lr(mt)]
        if not plain:
            continue
        q_ref = float(np.median([qm for _, qm, _ in plain]))
        levels = [(mt, q_ref, tape.q(mt)[1]) for mt in mts if mt <= fam[1]]
        qms = {round(qm, 0) for _, qm, _ in plain}
        if len(qms) > 1:
            spread = max(qm for _, qm, _ in plain) - min(qm for _, qm, _ in plain)
            if spread > LEVEL_TOLERANCE:
                out.add("level_qm", WARN,
                        f"the levels MT{fam[0]}-{fam[1]} state {len(qms)} different QM, spread "
                        f"{fmt_e(spread)}: one reaction has one ground-state Q", mf=3, mt=mts[0],
                        spread=spread)
        prev = None
        for mt, qm, qi in levels:
            ex = qm - qi
            if ex < -LEVEL_TOLERANCE:
                out.add("level_energy", DEFECT,
                        f"MT{mt}: QI = {fmt_e(qi)} above QM = {fmt_e(qm)}, a level below the ground state",
                        mf=3, mt=mt, qm=qm, qi=qi)
            if prev is not None and ex < prev[1] - LEVEL_TOLERANCE:
                out.add("level_order", WARN,
                        f"MT{mt} is at {fmt_e(ex)}, below MT{prev[0]} at {fmt_e(prev[1])}: levels go up "
                        "with the MT", mf=3, mt=mt, excitation=ex, previous=prev[1])
            prev = (mt, ex)
        discrete = [mt for mt, _, _ in levels]
        first = fam[0] + (1 if fam[0] == 50 and tape.neutron_projectile else 0)
        if discrete:
            gaps = [m for m in range(first, discrete[-1]) if m not in secs]
            if gaps:
                out.add("level_gap", NOTE,
                        f"levels MT{first}-{discrete[-1]} skip {len(gaps)} MT "
                        f"({', '.join(f'MT{m}' for m in gaps[:6])}{'...' if len(gaps) > 6 else ''}); "
                        "each level takes the next MT", mf=3, mt=gaps[0], missing=gaps)
        if fam[2] in secs and prev is not None:
            qm, qi = tape.q(fam[2])
            start = (q_ref - qi) if not tape.lr(fam[2]) else (qm - qi)
            # QI = QM on a continuum is the convention of not stating where it starts.
            if start != 0 and start < prev[1] - LEVEL_TOLERANCE:
                out.add("level_continuum_overlap", NOTE,
                        f"the continuum MT{fam[2]} starts at {fmt_e(start)}, below the last level "
                        f"MT{prev[0]} at {fmt_e(prev[1])}: the two overlap in excitation",
                        mf=3, mt=fam[2], start=start, last_level=prev[1])


def _sums(tape: Tape, tables: Dict[int, Table], present, rr_top: Optional[float], out: Out) -> None:
    """Stated totals against their partials; above the resonance ranges when MF2 adds to MF3.

    Compared at the energies the total is stated at, where the evaluator wrote a
    number: between them the total is interpolated over its own nodes, which do
    not include the thresholds of its partials, and the gap there is a matter of
    interpolation (one point at each threshold), not of what the file says. At a
    node where the partials step, the side closer to the stated value is taken.

    Inside the ranges of a tape with LRP = 1 MF3 holds backgrounds, and a total
    written there with its resonances (JENDL-5 Cl-35 MT103) is not the sum of its
    partials' backgrounds: whether the reconstructed sums agree is the
    reconstruction's check. A total one of whose partials could not be read is
    not compared.
    """
    from kika._constants import MF3_SUM_ORDER, mf3SumComponents
    from kika.algebra import left_limit, right_limit

    for mt in MF3_SUM_ORDER:
        if mt not in tables:
            continue
        parts = [p for p in mf3SumComponents(mt, present) if p != mt]
        if not parts or any(p not in tables for p in parts):
            continue
        stated = tables[mt]
        grid = stated.x
        try:
            lo = np.zeros(grid.size)
            hi = np.zeros(grid.size)
            for p in parts:
                tp = tables[p]
                lo += left_limit(tp.x, tp.y, _laws(tp), grid)
                hi += right_limit(tp.x, tp.y, _laws(tp), grid)
        except ValueError:
            continue
        s_all = stated.y
        total = np.where(np.abs(lo - s_all) <= np.abs(hi - s_all), lo, hi)
        inside = np.ones(grid.size, dtype=bool)
        if rr_top:
            inside &= grid > rr_top
        if not inside.any():
            continue
        g, s, total = grid[inside], s_all[inside], total[inside]
        peak = float(np.max(np.abs(s))) or 1.0
        gap = np.abs(total - s)
        rel = gap / np.maximum(np.abs(s), SUM_FLOOR * peak)
        i = int(np.argmax(rel))
        if rel[i] <= SUM_NOTE:
            continue
        material = (gap > SUM_MATERIAL_REL * np.maximum(np.abs(s), np.abs(total))) & (gap > SUM_MATERIAL_B)
        if material.any():
            j = int(np.flatnonzero(material)[np.argmax(gap[material])])
            share = f"{gap[j] / abs(s[j]):.2g} of it" if s[j] != 0 else "the total is stated as 0"
            local = (int(material.sum()) <= SUM_FEW_POINTS and s[j] != 0
                     and gap[j] <= SUM_FEW_REL * abs(s[j]))
            out.add("xs_sum", NOTE if local else WARN,
                    f"MT{mt} differs from the sum of its {len(parts)} partials by up to {fmt_b(gap[j])} "
                    f"({share}) at {fmt_e(g[j])}: {fmt_b(s[j])} stated, "
                    f"{fmt_b(total[j])} summed; {int(material.sum())} points beyond 1 % and 1 mb",
                    mf=3, mt=mt, at=float(g[j]), stated=float(s[j]), summed=float(total[j]),
                    points=int(material.sum()), partials=parts)
        else:
            out.add("xs_sum", NOTE,
                    f"MT{mt} differs from the sum of its {len(parts)} partials by up to {rel[i]:.2g} of it "
                    f"at {fmt_e(g[i])} ({fmt_b(s[i])} stated, {fmt_b(total[i])} summed), nowhere by both "
                    "1 % and 1 mb", mf=3, mt=mt, at=float(g[i]), stated=float(s[i]),
                    summed=float(total[i]), relative=float(rel[i]), partials=parts)


def _laws(t: Table) -> np.ndarray:
    laws = t.laws()
    return np.where(np.isin(laws, (1, 2, 3, 4, 5)), laws, 2)

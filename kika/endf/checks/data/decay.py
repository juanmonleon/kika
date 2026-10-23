"""MF8/MT457 (decay data) and MF8/MT454, MT459 (fission product yields).

**Decay.** The half-life is not negative, and is zero exactly when the
nuclide is stable (NST). The branching ratios lie in [0, 1] and add up to 1
(FIZCON: within 1e-4, or within their stated uncertainties). Each mode's Q
against AME2020 -- which no other checker does -- for the modes whose
daughter the RTYP fixes (beta-, EC/beta+, alpha, n, p; IT against the
parent's excitation), with the parent's excitation ELIS added and modes to an
excited daughter (RFS > 0) left out. The average energies of the radiations
do not add up to more than the branching-weighted Q (energy would be
created; the neutrinos carry the rest). A discrete line is below the largest
Q, and its intensity is not negative; a continuous spectrum is a TAB1 with
non-negative values below Q.

**Fission yields.** Yields are not negative. At each incident energy the
independent yields (MT454) add up to 2, carry the charge of the fissioning
nucleus (sum Z y = Z_c) and imply a prompt nu-bar ``A_c - sum A y`` between 0
and 10; no independent yield exceeds the cumulative one of the same product
beyond their uncertainties. Only the first is checked by FIZCON; the others by
no code surveyed.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import AMU_EV, Out, Table, Tape, fmt_e
from .reactions import mass_u
from .tables import check_table

#: Branching ratios summing to 1, beyond their stated uncertainty.
BR_NOTE = 1e-4
BR_WARN = 1e-2
#: Decay Q against AME2020: a note above max(Q_NOTE_EV, Q_NOTE_REL Q), a
#: warning above max(Q_WARN_EV, Q_WARN_REL Q).
Q_NOTE_EV, Q_NOTE_REL = 1e3, 1e-3
Q_WARN_EV, Q_WARN_REL = 1e5, 1e-2
#: A decay Q off AME2020 is a warning only for a nuclide that lives this long
#: (s): the exotic ones, short-lived, have masses AME extrapolates, and their Q
#: weighs nothing in decay heat or inventories. A Q off by this factor is a
#: unit error, a defect whatever the half-life.
Q_WARN_HALFLIFE = 1.0
Q_UNIT_FACTOR = 100.0
#: Sum of average decay energies above sum br Q.
ENERGY_SLACK_REL, ENERGY_SLACK_EV = 1e-2, 1e3
#: A discrete line above the largest Q.
LINE_SLACK_REL, LINE_SLACK_EV = 1e-3, 1e3
#: Fission yields: sum of independent yields against 2, charge against Z_c.
YSUM_NOTE, YSUM_WARN = 2e-3, 2e-2
CHARGE_NOTE, CHARGE_WARN = 1e-3, 1e-2
#: Independent above cumulative by more than this many standard deviations.
IFY_CFY_SIGMAS = 3.0

# RTYP's first digit -> (dZ, dA) of the daughter, emitted light particle ZA.
_MODES = {1: (+1, 0, None), 2: (-1, 0, None), 4: (-2, -4, 2004), 5: (0, -1, 1), 7: (-1, -1, 1001)}
_MODE_NAME = {1: "beta-", 2: "EC/beta+", 3: "IT", 4: "alpha", 5: "n", 6: "SF", 7: "p", 0: "gamma"}


def _q_ame(za: int, digit: int, elis: float) -> Optional[float]:
    if digit not in _MODES:
        return None
    dz, da, light = _MODES[digit]
    z, a = divmod(za, 1000)
    daughter = 1000 * (z + dz) + (a + da)
    m_p, m_d = mass_u(za), mass_u(daughter)
    m_l = mass_u(light) if light else 0.0
    if m_p is None or m_d is None or m_l is None:
        return None
    return (m_p - m_d - m_l) * AMU_EV + elis


def _decay(tape: Tape, out: Out) -> None:
    sec = tape.section(8, 457)
    if sec is None:
        return
    za = int(round(float(sec._za))) if sec._za is not None else tape.za
    stable = sec.is_stable
    t = float(sec.halflife)
    if t < 0:
        out.add("decay_halflife", DEFECT, f"negative half-life {t:g} s", mf=8, mt=457, halflife=t)
    elif stable and t != 0:
        out.add("decay_halflife", WARN, f"NST = 1 (stable) with a half-life of {t:g} s", mf=8, mt=457, halflife=t)
    elif not stable and t == 0:
        out.add("decay_halflife", WARN, "an unstable nuclide (NST = 0) with a half-life of 0", mf=8, mt=457)
    modes = sec.decay_modes
    if stable or not modes:
        return
    br = np.array([m.br for m in modes], dtype=float)
    dbr = np.array([m.dbr for m in modes], dtype=float)
    if br.size and (br.min() < 0 or br.max() > 1 + 1e-9):
        out.add("decay_branching", DEFECT, f"a branching ratio outside [0, 1] ({br.min():.6g} to {br.max():.6g})",
                mf=8, mt=457, br_min=float(br.min()), br_max=float(br.max()))
    d = abs(float(br.sum()) - 1.0)
    if d > max(BR_NOTE, float(np.abs(dbr).sum())):
        out.add("decay_branching", WARN if d > BR_WARN else NOTE,
                f"the {br.size} branching ratios add up to {br.sum():.7g}, not 1", mf=8, mt=457,
                total=float(br.sum()), stated_uncertainty=float(np.abs(dbr).sum()))
    if br.size and np.any(br == 0):
        out.add("decay_branching", NOTE, f"{int((br == 0).sum())} decay mode{'s' if (br == 0).sum() != 1 else ''} "
                "with a branching ratio of 0", mf=8, mt=457)
    elis = tape.elis
    isomer_unknown = (tape.liso > 0 or int(tape._h("_lis", 0) or 0) > 0) and elis == 0
    qmax = 0.0
    for m in modes:
        q = float(m.q)
        qmax = max(qmax, q)
        digit = int(m.rtyp)
        if m.rfs != 0 or za is None or za % 1000 == 0 or isomer_unknown:
            continue
        if digit == 3:
            expected = elis if elis > 0 else None
        elif float(m.rtyp) != digit:
            continue        # a chain (beta- then n, ...): one Q for two steps
        else:
            expected = _q_ame(za, digit, elis)
        if expected is None:
            continue
        diff = abs(q - expected)
        if diff > max(Q_NOTE_EV, Q_NOTE_REL * abs(expected)):
            level = WARN if diff > max(Q_WARN_EV, Q_WARN_REL * abs(expected)) and t >= Q_WARN_HALFLIFE else NOTE
            if q > 0 and expected > 0 and max(q / expected, expected / q) >= Q_UNIT_FACTOR:
                level = DEFECT
            out.add("decay_q", level,
                    f"{_MODE_NAME.get(digit, digit)} decay: Q = {fmt_e(q)}, AME2020 gives {fmt_e(expected)} "
                    f"(off by {fmt_e(diff)})", mf=8, mt=457, mode=float(m.rtyp), q=q, ame=expected)
    q_eff = float(sum(m.br * m.q for m in modes))
    energies = sec.average_energy_pairs
    total_e = float(sum(e for e, _ in energies[:3])) if len(energies) >= 3 else None
    if total_e is not None and q_eff > 0:
        if total_e > q_eff * (1 + ENERGY_SLACK_REL) + ENERGY_SLACK_EV:
            out.add("decay_energy", WARN,
                    f"the radiations carry {fmt_e(total_e)} on average, more than the {fmt_e(q_eff)} the decay "
                    "releases (sum of branching ratio times Q)", mf=8, mt=457, average=total_e, q_effective=q_eff)
    if qmax <= 0:
        return
    high, neg = [], []
    for k, spec in enumerate(sec.spectra):
        for line in spec.lines:
            if line.er > qmax * (1 + LINE_SLACK_REL) + LINE_SLACK_EV and float(spec.styp) not in (6.0,):
                high.append((k, float(line.er), float(spec.styp)))
            ri = line.intensity[0] if hasattr(line, "intensity") else None
            if ri is not None and ri < 0:
                neg.append((k, float(line.er)))
        if spec.continuum is not None:
            t = Table.of(spec.continuum.table.x, spec.continuum.table.y, spec.continuum.table.interp)
            if check_table(t, out, mf=8, mt=457, what=f"spectrum {k} (STYP = {spec.styp:g}) continuum", x_name="E'", k=k):
                if t.y.size and t.y.min() < 0:
                    out.add("decay_spectrum", DEFECT, f"spectrum {k} (STYP = {spec.styp:g}): negative continuum "
                            f"{t.y.min():.3g}", mf=8, mt=457, k=k, value=float(t.y.min()))
                nz = np.flatnonzero(t.y > 0)
                if nz.size and t.x[min(nz[-1] + 1, t.x.size - 1)] > qmax * (1 + LINE_SLACK_REL) + LINE_SLACK_EV \
                        and float(spec.styp) not in (5.0, 6.0):
                    out.add("decay_spectrum", WARN,
                            f"spectrum {k} (STYP = {spec.styp:g}): the continuum reaches "
                            f"{fmt_e(float(t.x[min(nz[-1] + 1, t.x.size - 1)]))}, above the largest Q = {fmt_e(qmax)}",
                            mf=8, mt=457, k=k)
    if high:
        k, e, styp = high[0]
        out.add("decay_spectrum", WARN,
                f"{len(high)} discrete line{'s' if len(high) != 1 else ''} above the largest Q = {fmt_e(qmax)}; first "
                f"{fmt_e(e)} in spectrum {k} (STYP = {styp:g})", mf=8, mt=457, k=k, count=len(high), energy=e)
    if neg:
        k, e = neg[0]
        out.add("decay_spectrum", DEFECT, f"{len(neg)} line{'s' if len(neg) != 1 else ''} with a negative intensity; "
                f"first at {fmt_e(e)} in spectrum {k}", mf=8, mt=457, k=k, count=len(neg))


def _yields(tape: Tape, out: Out) -> None:
    ind, cum = tape.section(8, 454), tape.section(8, 459)
    if ind is None and cum is None:
        return
    za = tape.za
    if za is None:
        return
    z_c, a_t = divmod(za, 1000)
    a_c = a_t + (1 if tape.nsub == 11 else 0)
    for mt, sec in ((454, ind), (459, cum)):
        if sec is None:
            continue
        for e_index, fy in enumerate(sec.energies):
            v = np.asarray(fy.values, dtype=float).reshape(-1, 4) if len(fy.values) % 4 == 0 else None
            if v is None:
                out.add("yield_record", DEFECT, f"MT{mt} at E = {fmt_e(fy.energy)}: values not in groups of four",
                        mf=8, mt=mt)
                continue
            if v[:, 2].min(initial=0) < 0:
                out.add("yield_negative", DEFECT,
                        f"MT{mt} at E = {fmt_e(fy.energy)}: a negative yield ({v[:, 2].min():.3g})", mf=8, mt=mt,
                        at=float(fy.energy), value=float(v[:, 2].min()))
            if mt != 454:
                continue
            s = float(v[:, 2].sum())
            d = abs(s - 2.0) / 2.0
            if d > YSUM_NOTE:
                out.add("yield_sum", WARN if d > YSUM_WARN else NOTE,
                        f"independent yields at E = {fmt_e(fy.energy)} add up to {s:.6g}, not 2", mf=8, mt=454,
                        at=float(fy.energy), total=s)
            zap = np.rint(v[:, 0]).astype(int)
            zsum = float(np.sum((zap // 1000) * v[:, 2]))
            asum = float(np.sum((zap % 1000) * v[:, 2]))
            dz = abs(zsum - z_c) / z_c
            if dz > CHARGE_NOTE:
                out.add("yield_charge", WARN if dz > CHARGE_WARN else NOTE,
                        f"independent yields at E = {fmt_e(fy.energy)} carry charge {zsum:.5g}, the fissioning "
                        f"nucleus {z_c}", mf=8, mt=454, at=float(fy.energy), charge=zsum, z=z_c)
            nu = a_c - asum
            if not 0 <= nu <= 10:
                out.add("yield_mass", WARN,
                        f"independent yields at E = {fmt_e(fy.energy)} carry mass {asum:.5g}, which leaves "
                        f"{nu:.3g} prompt neutrons from A = {a_c}", mf=8, mt=454, at=float(fy.energy), mass=asum)
    if ind is None or cum is None:
        return
    for fi in ind.energies:
        match = [fc for fc in cum.energies if abs(fc.energy - fi.energy) <= 1e-6 * max(abs(fi.energy), 1.0)]
        if not match:
            continue
        vc = np.asarray(match[0].values, dtype=float).reshape(-1, 4)
        vi = np.asarray(fi.values, dtype=float).reshape(-1, 4)
        cum_of: Dict[Tuple[int, int], Tuple[float, float]] = {
            (int(round(r[0])), int(round(r[1]))): (r[2], r[3]) for r in vc}
        bad = []
        for r in vi:
            key = (int(round(r[0])), int(round(r[1])))
            if key in cum_of:
                c, dc = cum_of[key]
                if r[2] > c + IFY_CFY_SIGMAS * np.hypot(r[3], dc) + 1e-12:
                    bad.append((key, r[2], c))
        if bad:
            key, y, c = max(bad, key=lambda b: b[1] - b[2])
            out.add("yield_ify_cfy", WARN,
                    f"at E = {fmt_e(fi.energy)}, {len(bad)} product{'s' if len(bad) != 1 else ''} with an independent "
                    f"yield above the cumulative one beyond {IFY_CFY_SIGMAS:g} sigma; worst ZA {key[0]} "
                    f"(state {key[1]}): {y:.4g} > {c:.4g}", mf=8, mt=454, at=float(fi.energy), count=len(bad))


def check_decay(tape: Tape, out: Out) -> None:
    _decay(tape, out)
    _yields(tape, out)


def descriptions(D, _g, _pct):
    return {
        "decay_halflife": D(
            "Half-life", (8,), {DEFECT: "negative", WARN: "zero for an unstable nuclide, or non-zero for a stable one"},
            "A half-life is not negative, zero exactly for a stable nuclide (OpenMC refuses an unstable one with 0)."),
        "decay_branching": D(
            "Branching ratios", (8,),
            {DEFECT: "one outside [0, 1]", WARN: f"their sum off 1 by more than {_g(BR_WARN)}",
             NOTE: f"by more than {_g(BR_NOTE)} and their stated uncertainty, or a ratio of 0"},
            "The branching ratios of a nuclide's decay modes add up to 1 (FIZCON: 1e-4, or the stated "
            "uncertainties)."),
        "decay_q": D(
            "Decay Q against AME2020", (8,),
            {DEFECT: f"off by a factor of {_g(Q_UNIT_FACTOR)} or more (a unit error)",
             WARN: f"off by more than {_g(Q_WARN_EV / 1e3)} keV and {_pct(Q_WARN_REL)}, for a half-life of "
                   f"{_g(Q_WARN_HALFLIFE)} s or more",
             NOTE: f"off by more than {_g(Q_NOTE_EV / 1e3)} keV and {_pct(Q_NOTE_REL)}"},
            "A decay mode's Q is the mass difference of parent (plus its excitation) and daughter, which "
            "AME2020 gives for beta-, EC/beta+, alpha, neutron and proton emission; an isomeric transition "
            "releases the parent's excitation. Modes to an excited daughter, chains and isomers whose MF1 "
            "gives no excitation are left out. Short-lived exotic nuclides have masses AME extrapolates and "
            "Q values from older tables; their differences are notes. Found in ENDF/B-VIII.1: Cu-81, Rh-123, "
            "Pd-125, Pd-126 and I-145 state Q in keV where MeV is meant. No other checker compares decay Q "
            "with a mass table."),
        "decay_energy": D(
            "Average decay energies exceed Q", (8,), {WARN: "always"},
            f"The average energies of the light, electromagnetic and heavy radiations add up to at most the "
            f"branching-weighted Q (neutrinos take the rest); beyond {_pct(ENERGY_SLACK_REL)} energy is created "
            "(FIZCON's SUMCK)."),
        "decay_spectrum": D(
            "Decay spectrum", (8,),
            {DEFECT: "a negative intensity or continuum", WARN: "a line or continuum above the largest Q"},
            "Lines and continua are below the decay's Q and have non-negative intensities (FIZCON's CHKDEC)."),
        "yield_record": D(
            "Malformed yield record", (8,), {DEFECT: "always"}, "Yields come in groups of ZAFP, FPS, Y, DY."),
        "yield_negative": D(
            "Negative fission yield", (8,), {DEFECT: "always"}, "A fission product yield is not negative."),
        "yield_sum": D(
            "Independent yields do not add up to 2", (8,),
            {WARN: f"off by more than {_pct(YSUM_WARN)}", NOTE: f"by more than {_pct(YSUM_NOTE)}"},
            "Two fragments per fission: the independent yields add up to 2 (FIZCON: 1e-3)."),
        "yield_charge": D(
            "Independent yields do not conserve charge", (8,),
            {WARN: f"off by more than {_pct(CHARGE_WARN)}", NOTE: f"by more than {_pct(CHARGE_NOTE)}"},
            "The fragments carry the charge of the fissioning nucleus: sum Z y = Z_c. No code surveyed checks it."),
        "yield_mass": D(
            "Independent yields imply an impossible nu-bar", (8,), {WARN: "outside 0 to 10"},
            "A_c - sum A y is the prompt neutrons per fission (Jaffke et al., NSE 190 (2018) 258)."),
        "yield_ify_cfy": D(
            "Independent yield above the cumulative one", (8,), {WARN: f"beyond {_g(IFY_CFY_SIGMAS)} sigma"},
            "A cumulative yield includes the independent one: it cannot be smaller (found by NNDC in Pu-239; no "
            "code surveyed checks it)."),
    }


def method(_g, _pct):
    return [
        ("Decay and fission yields",
         "MF8: decay data against AME2020 and energy conservation; fission yields against the sum rules of "
         "fission.",
         [(DEFECT, "a negative half-life, intensity or yield; a branching ratio outside [0, 1]."),
          (WARN, f"decay Q off AME2020 by more than {_g(Q_WARN_EV / 1e3)} keV and {_pct(Q_WARN_REL)}; branching "
                 f"ratios off 1 by more than {_g(BR_WARN)}; radiations carrying more than Q; lines above Q; "
                 f"independent yields off 2 by more than {_pct(YSUM_WARN)} or off the charge by more than "
                 f"{_pct(CHARGE_WARN)}; an independent yield above the cumulative."),
          (NOTE, "smaller disagreements.")]),
    ]

"""MF12-MF15: photon production, against the levels it comes from.

**Level photons.** A photon from a level carries at most the level's
excitation: a discrete photon of a level reaction (MT51-90, 600-648, ...) is
not above the excitation QM - QI of its MT, nor above the ES it says it comes
from; a capture photon not above the capture Q (plus E A/(A+1) for a primary
one). With transition probabilities (LO = 2) the level's ES is its MT's
excitation, every lower level is below it and is a level of the scheme (or the
ground state), and the probabilities of one level add up to 1 (FIZCON 1e-4).

**Bookkeeping.** A continuum photon (LF = 1) has its spectrum in MF15, and an
MF15 section has a continuum photon to belong to; MF14 gives as many photons
as MF12/MF13 and exists for each; an MF12 multiplicity has its MF3 cross
section to multiply (MF13 states its own). MF12's total yield is the sum of
its photons'.

**Densities.** MF14 angular distributions and MF15 spectra are densities
(:mod:`.distributions`), the MF15 weights add up to 1.

**Capture energy.** At the lowest incident energy, the energy MF12/MF15 give
the capture photons, ``sum y E_gamma``, is the capture Q plus E A/(A+1): the
whole excitation of the compound nucleus goes into photons. Off by more than
``CAPTURE_WARN`` is a warning, ``CAPTURE_DEFECT`` a defect.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import trapezoid

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Table, Tape, fmt_e
from .distributions import (_Nodes, _integral, _legendre, _mu_table, _norm, _probability_sum)
from .reactions import is_level_mt, level_family
from .tables import check_table

#: A photon energy above its level by more than this (eV, or relative) is not rounding.
LEVEL_SLACK_EV = 1e3
LEVEL_SLACK_REL = 1e-3
#: Transition probabilities of one level summing to 1 (FIZCON's 1e-4).
TP_NOTE = 1e-4
TP_WARN = 1e-2
#: MF12's total yield against the sum of its photons.
YIELD_SUM_NOTE = 1e-4
YIELD_SUM_WARN = 1e-2
#: Capture photon energy against Q + E A/(A+1).
CAPTURE_WARN = 0.05
CAPTURE_DEFECT = 0.2


def _slack(e: float) -> float:
    return max(LEVEL_SLACK_EV, LEVEL_SLACK_REL * abs(e))


def _excitation(tape: Tape, mt: int) -> Optional[float]:
    if not is_level_mt(mt) or tape.section(3, mt) is None:
        return None
    qm, qi = tape.q(mt)
    return qm - qi


def _levels(tape: Tape, mt: int) -> List[float]:
    """The excitations of every level of *mt*'s family on this tape, the ground state first."""
    fam = level_family(mt)
    out = [0.0]
    if fam is None:
        return out
    for m in tape.sections(3):
        if fam[0] <= m <= fam[1]:
            qm, qi = tape.q(m)
            out.append(qm - qi)
    if fam[0] == 50 and tape.elis > 0:
        # (z,n') of an isomeric target: the residual is the target, and its own
        # isomeric state is a level a cascade can stop at (QM includes ELIS).
        out.append(tape.elis)
    return sorted(out)


def _lo1(tape: Tape, out: Out, mf: int, mt: int, sec, mf15: set) -> None:
    what = f"MT{mt}"
    tables = []
    if sec.total is not None:
        t = Table.of(sec.total.x, sec.total.y, sec.total.interp)
        if check_table(t, out, mf=mf, mt=mt, what=f"{what} total"):
            tables.append(("total", None, t))
    exc = _excitation(tape, mt)
    q_capture = tape.q(102)[0] if mt == 102 else None
    continuum = False
    for k, ph in enumerate(sec.photons):
        t = Table.of(ph.x, ph.y, ph.interp)
        if not check_table(t, out, mf=mf, mt=mt, what=f"{what} photon {k}", k=k):
            continue
        tables.append(("photon", k, t))
        if t.y.size and t.y.min() < 0:
            tiny = -t.y.min() <= 1e-6 * max(float(np.abs(t.y).max()), 1e-300)
            out.add("photon_negative", NOTE if tiny else DEFECT,
                    f"{what} photon {k}: negative {'multiplicity' if mf == 12 else 'cross section'} "
                    f"{t.y.min():.6g}", mf=mf, mt=mt, k=k, value=float(t.y.min()))
        eg, es, lp, lf = float(ph.c1), float(ph.c2), int(ph.l1), int(ph.l2)
        if lf == 1:
            continuum = True
            continue
        if lf != 2:
            out.add("photon_record", DEFECT, f"{what} photon {k}: LF = {lf}, neither 1 (continuum) nor 2 "
                    "(discrete)", mf=mf, mt=mt, k=k, lf=lf)
            continue
        if eg <= 0:
            out.add("photon_energy", DEFECT, f"{what} photon {k}: a discrete photon with EG = {fmt_e(eg)}",
                    mf=mf, mt=mt, k=k, eg=eg)
            continue
        if exc is not None and eg > exc + _slack(exc):
            out.add("photon_energy", DEFECT,
                    f"{what} photon {k}: EG = {fmt_e(eg)} from a level at {fmt_e(exc)}: more energy than the "
                    "level has", mf=mf, mt=mt, k=k, eg=eg, level_energy=exc)
        elif es > 0 and eg > es + _slack(es):
            out.add("photon_energy", DEFECT,
                    f"{what} photon {k}: EG = {fmt_e(eg)} from ES = {fmt_e(es)}: more energy than the level "
                    "it comes from", mf=mf, mt=mt, k=k, eg=eg, es=es)
        elif q_capture is not None and q_capture > 0 and eg > q_capture + _slack(q_capture):
            out.add("photon_energy", DEFECT,
                    f"{what} photon {k}: EG = {fmt_e(eg)}, above the capture Q = {fmt_e(q_capture)}",
                    mf=mf, mt=mt, k=k, eg=eg, q=q_capture)
    if continuum and mt not in mf15:
        out.add("photon_spectrum_missing", DEFECT,
                f"{what}: a continuum photon (LF = 1) in MF{mf} without its spectrum in MF15", mf=mf, mt=mt)
    parts = [t for kind, _, t in tables if kind == "photon"]
    tot = next((t for kind, _, t in tables if kind == "total"), None)
    if tot is not None and parts and len(parts) == len(sec.photons) and tot.x.size:
        from kika.algebra import left_limit, right_limit

        g = tot.x
        try:
            lo = sum(left_limit(p.x, p.y, _laws(p), g) for p in parts)
            hi = sum(right_limit(p.x, p.y, _laws(p), g) for p in parts)
        except ValueError:
            return
        # at a node where photons step on or off, the side closer to the total
        s = np.where(np.abs(lo - tot.y) <= np.abs(hi - tot.y), lo, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(tot.y != 0, np.abs(s - tot.y) / np.abs(tot.y), 0.0)
        i = int(np.argmax(rel))
        if rel[i] > YIELD_SUM_NOTE:
            out.add("photon_sum", WARN if rel[i] > YIELD_SUM_WARN else NOTE,
                    f"{what}: the total {'yield' if mf == 12 else 'cross section'} is {tot.y[i]:.6g} at "
                    f"{fmt_e(g[i])}, its {len(parts)} photons add up to {s[i]:.6g}",
                    mf=mf, mt=mt, at=float(g[i]), total=float(tot.y[i]), summed=float(s[i]))


def _laws(t: Table) -> np.ndarray:
    laws = t.laws()
    return np.where(np.isin(laws, (1, 2, 3, 4, 5)), laws, 2)


def _lo2(tape: Tape, out: Out, mf: int, mt: int, sec) -> None:
    what = f"MT{mt}"
    es_ns = float(getattr(sec, "es_ns", 0.0) or 0.0)
    exc = _excitation(tape, mt)
    if exc is not None and abs(es_ns - exc) > _slack(exc):
        out.add("level_scheme", WARN,
                f"{what}: transitions from ES = {fmt_e(es_ns)}, but MF3 puts the level at {fmt_e(exc)}",
                mf=mf, mt=mt, es=es_ns, level_energy=exc)
    trans = sec.transitions
    if not trans:
        out.add("photon_record", DEFECT, f"{what}: LO = 2 with no transitions", mf=mf, mt=mt)
        return
    levels = _levels(tape, mt)
    tp = np.array([t[1] for t in trans], dtype=float)
    for t in trans:
        es_i = float(t[0])
        if es_i >= es_ns - 1e-9 * max(es_ns, 1.0):
            out.add("photon_energy", DEFECT,
                    f"{what}: a transition from {fmt_e(es_ns)} to {fmt_e(es_i)}, not below it",
                    mf=mf, mt=mt, es=es_ns, to=es_i)
        elif levels and min(abs(es_i - lv) for lv in levels) > _slack(es_i) and len(levels) > 1:
            near = min(levels, key=lambda lv: abs(lv - es_i))
            out.add("level_scheme", WARN,
                    f"{what}: a transition to {fmt_e(es_i)}, which is no level of MF3 (nearest {fmt_e(near)})",
                    mf=mf, mt=mt, to=es_i, nearest=near)
    if np.any((tp < 0) | (tp > 1 + 1e-6)):
        out.add("transition_probability", DEFECT,
                f"{what}: a transition probability outside [0, 1] ({tp.min():.6g} to {tp.max():.6g})",
                mf=mf, mt=mt, tp_min=float(tp.min()), tp_max=float(tp.max()))
    elif np.any(tp == 0):
        out.add("transition_probability", NOTE,
                f"{what}: {int((tp == 0).sum())} transition{'s' if (tp == 0).sum() != 1 else ''} listed with "
                "probability 0", mf=mf, mt=mt, zeros=int((tp == 0).sum()))
    d = abs(float(tp.sum()) - 1.0)
    if d > TP_NOTE:
        out.add("transition_probability", WARN if d > TP_WARN else NOTE,
                f"{what}: the transition probabilities from {fmt_e(es_ns)} add up to {tp.sum():.6g}, not 1",
                mf=mf, mt=mt, total=float(tp.sum()))


def _mf14(tape: Tape, out: Out, counts: Dict[int, int]) -> None:
    for mt, sec in sorted(tape.sections(14).items()):
        what = f"MT{mt}"
        nk = int(getattr(sec, "_nk", 0) or 0)
        all_isotropic = int(getattr(sec, "_li", 0) or 0) == 1
        if counts.get(mt) and nk != counts[mt] and not all_isotropic:
            out.add("photon_count", DEFECT,
                    f"{what}: MF14 gives {nk} photon{'s' if nk != 1 else ''}, MF12/MF13 {counts[mt]}",
                    mf=14, mt=mt, nk=nk, expected=counts[mt])
        for k, ph in enumerate(getattr(sec, "anisotropic", []) or []):
            nodes = _Nodes(out, 14, mt, f"{what} photon {k}")
            leg_c, leg_e = [], []
            for node in ph.nodes:
                scratch = nodes.node()
                if node.coefficients:
                    leg_c.append(list(node.coefficients))
                    leg_e.append(float(node.energy))
                elif node.mu:
                    _mu_table(scratch, node.mu, node.p, node.interp, float(node.energy), 14, mt,
                              f"{what} photon {k}", k=k)
            if leg_c:
                _legendre(nodes.scratch, leg_c, leg_e, 14, mt, f"{what} photon {k}", k=k)
            nodes.flush()


def _mf15(tape: Tape, out: Out, continuum_mts: set) -> None:
    for mt, sec in sorted(tape.sections(15).items()):
        what = f"MT{mt}"
        if mt not in continuum_mts:
            out.add("photon_spectrum_orphan", WARN,
                    f"{what}: an MF15 spectrum with no continuum photon (LF = 1) in MF12/MF13 to belong to",
                    mf=15, mt=mt)
        weights = []
        for k, spec in enumerate(sec.spectra):
            w = Table.of(spec.weight.x, spec.weight.y, spec.weight.interp)
            if check_table(w, out, mf=15, mt=mt, what=f"{what} weight {k}", k=k):
                weights.append(w)
            nodes = _Nodes(out, 15, mt, f"{what} spectrum {k}")
            for dist in spec.distributions:
                scratch = nodes.node()
                t = Table.of(dist.x, dist.y, dist.interp)
                e = float(dist.c2)
                if not check_table(t, scratch, mf=15, mt=mt, what=f"{what} spectrum {k}", x_name="E'", k=k):
                    continue
                if t.y.size and t.y.min() < 0:
                    scratch.add("distribution_negative", DEFECT,
                                f"{what} spectrum {k}: negative spectrum {t.y.min():.3g} at E = {fmt_e(e)}",
                                mf=15, mt=mt, k=k, at=e, value=float(t.y.min()))
                if t.x.size > 1:
                    _norm(scratch, 15, mt, f"{what} spectrum {k}", e, _integral(t), k=k)
            nodes.flush()
        if len(weights) == len(sec.spectra) and weights:
            _probability_sum(out, 15, mt, what, weights)


def _capture_energy(tape: Tape, out: Out) -> None:
    sec = tape.section(12, 102)
    if sec is None or getattr(sec, "_lo", 1) != 1 or not tape.neutron_projectile or not tape.awr:
        return
    qm, _ = tape.q(102)
    if not qm or qm <= 0:
        return
    starts = [float(ph.x[0]) for ph in sec.photons if ph.x]
    if not starts:
        return
    e0 = min(starts)
    awr = tape.awr
    total = 0.0
    for ph in sec.photons:
        t = Table.of(ph.x, ph.y, ph.interp)
        if not t.x.size:
            continue
        y = float(t(np.array([e0]))[0]) if t.x[0] <= e0 else 0.0
        if int(ph.l2) == 2:
            eg = float(ph.c1) + (e0 * awr / (awr + 1) if int(ph.l1) == 2 else 0.0)
            total += y * eg
        else:
            mean = _mf15_mean(tape, e0)
            if mean is None:
                return
            total += y * mean
    avail = qm + e0 * awr / (awr + 1)
    d = (total - avail) / avail
    if abs(d) > CAPTURE_WARN:
        out.add("capture_energy", DEFECT if abs(d) > CAPTURE_DEFECT else WARN,
                f"MT102: the capture photons carry {fmt_e(total)} at E = {fmt_e(e0)}, where the compound nucleus "
                f"has {fmt_e(avail)} to give them ({d:+.1%})", mf=12, mt=102, at=e0, photons=total,
                available=avail, relative=d)


def _mf15_mean(tape: Tape, e: float) -> Optional[float]:
    """The mean photon energy of MF15/MT102 at the incident energy nearest *e*."""
    sec = tape.section(15, 102)
    if sec is None:
        return None
    means, weights = [], []
    for spec in sec.spectra:
        if not spec.distributions:
            continue
        dist = min(spec.distributions, key=lambda d: abs(float(d.c2) - e))
        x, y = np.asarray(dist.x, float), np.asarray(dist.y, float)
        norm = trapezoid(y, x)
        if norm <= 0:
            continue
        w = Table.of(spec.weight.x, spec.weight.y, spec.weight.interp)
        means.append(float(trapezoid(x * y, x) / norm))
        weights.append(float(w(np.array([e]))[0]) if w.x.size else 1.0)
    if not means:
        return None
    wsum = sum(weights) or 1.0
    return sum(m * w for m, w in zip(means, weights)) / wsum


def check_photons(tape: Tape, out: Out) -> None:
    mf15 = set(tape.sections(15))
    counts: Dict[int, int] = {}
    continuum_mts = set()
    mf3 = set(tape.sections(3))
    for mf in (12, 13):
        for mt, sec in sorted(tape.sections(mf).items()):
            lo = getattr(sec, "_lo", 1) if mf == 12 else 1
            if lo == 1:
                _lo1(tape, out, mf, mt, sec, mf15)
                counts[mt] = len(sec.photons)
                if any(int(ph.l2) == 1 for ph in sec.photons):
                    continuum_mts.add(mt)
            elif lo == 2:
                _lo2(tape, out, mf, mt, sec)
                counts[mt] = None     # MF14 counts the whole cascade: not comparable
            else:
                out.add("photon_record", DEFECT, f"MT{mt}: LO = {lo}, neither 1 nor 2", mf=mf, mt=mt, lo=lo)
            if mf == 12 and mf3 and mt not in mf3 and mt != 460:
                out.add("photon_orphan", DEFECT,
                        f"MT{mt}: MF12 gives photon multiplicities for a reaction MF3 does not have, so no "
                        "code can produce them", mf=12, mt=mt)
    missing14 = sorted(m for m in counts if m not in tape.sections(14))
    if missing14 and tape.has(12) | tape.has(13):
        out.add("photon_angles_missing", WARN,
                f"{len(missing14)} photon-producing reaction{'s' if len(missing14) != 1 else ''} without MF14: "
                f"{', '.join(f'MT{m}' for m in missing14[:8])}{'...' if len(missing14) > 8 else ''}; codes "
                "assume isotropy", mf=14, mt=missing14[0], mts=missing14)
    _mf14(tape, out, counts)
    _mf15(tape, out, continuum_mts)
    _capture_energy(tape, out)


def descriptions(D, _g, _pct):
    P = (12, 13)
    return {
        "photon_negative": D(
            "Negative photon multiplicity or cross section", P,
            {DEFECT: "always", NOTE: "within a millionth of the largest value: the rounding of a zero"},
            "A photon yield (MF12) or photon production cross section (MF13) is not negative."),
        "photon_record": D(
            "Malformed photon record", P, {DEFECT: "always"},
            "LO is 1 or 2, a photon's LF is 1 or 2, an LO = 2 section lists its transitions."),
        "photon_energy": D(
            "Photon with more energy than its level", P,
            {DEFECT: "always"},
            "A photon from a level carries at most the level's excitation: the excitation QM - QI of "
            "its MT, the ES it states, or the capture Q; a transition goes to a lower level. (The "
            "2.289 MeV photon of a 1.069 MeV level of U-234 in the official ACE files is ACER's, not "
            "the evaluation's: its MF12 is consistent.)"),
        "level_scheme": D(
            "Level scheme disagrees with MF3", P, {WARN: "always"},
            f"With transition probabilities (LO = 2) the level is at its MT's excitation and every "
            f"lower level is a level of MF3 or the ground state, within {_g(LEVEL_SLACK_EV / 1e3)} keV "
            "(FIZCON's TESTEG, FUDGE's translator)."),
        "transition_probability": D(
            "Transition probabilities", P,
            {DEFECT: "one outside [0, 1]", WARN: f"their sum off 1 by more than {_g(TP_WARN)}",
             NOTE: f"by more than {_g(TP_NOTE)}, or a transition listed with probability 0"},
            "The probabilities of the transitions out of one level add up to 1 (FIZCON: 1e-4)."),
        "photon_sum": D(
            "Total photon yield is not the sum of its photons", P,
            {WARN: f"off by more than {_g(YIELD_SUM_WARN)}", NOTE: f"by more than {_g(YIELD_SUM_NOTE)}"},
            "MF12/MF13's total, given when there are several photons, is their sum (FIZCON's SUMPAR)."),
        "photon_spectrum_missing": D(
            "Continuum photon without MF15", P + (15,), {DEFECT: "always"},
            "A continuum photon (LF = 1) has its spectrum in MF15; without it a code has no energy to give it."),
        "photon_spectrum_orphan": D(
            "MF15 without a continuum photon", (15,), {WARN: "always"},
            "An MF15 spectrum belongs to a continuum photon of MF12/MF13 (FUDGE's translator warns both ways)."),
        "photon_count": D(
            "MF14 and MF12/MF13 give different numbers of photons", (14,), {DEFECT: "always"},
            "MF14 has one angular distribution per photon of MF12/MF13 (LO = 1), in the same order. Not "
            "compared when MF14 says every photon is isotropic (LI = 1), nor for transition probabilities "
            "(LO = 2), where MF14 counts the whole cascade."),
        "photon_angles_missing": D(
            "Photons without MF14", (12, 13, 14), {WARN: "always"},
            "Every photon-producing section has its angular distributions in MF14; without them codes "
            "assume isotropy."),
        "photon_orphan": D(
            "Photon multiplicities for a reaction MF3 lacks", (12,), {DEFECT: "always"},
            "MF12 gives multiplicities, which a code multiplies by the reaction's MF3 cross section; "
            "without that section the photons are never produced."),
        "capture_energy": D(
            "Capture photons do not carry the capture energy", (12, 15),
            {DEFECT: f"off by more than {_pct(CAPTURE_DEFECT)}", WARN: f"off by more than {_pct(CAPTURE_WARN)}"},
            "At the lowest incident energy the photons of radiative capture carry the excitation of "
            "the compound nucleus, Q + E A/(A+1): sum of multiplicity times photon energy, the "
            "continuum's mean from MF15. A deficit is energy that heating will miss; an excess is "
            "energy created."),
    }


def method(_g, _pct):
    return [
        ("Photons",
         "MF12-MF15 against the levels the photons come from and against each other.",
         [(DEFECT, "a photon with more energy than its level or than the capture Q; a negative yield; "
                   f"capture photons off the capture energy by more than {_pct(CAPTURE_DEFECT)}; a continuum "
                   "photon without MF15; MF14 and MF12 counts that differ; multiplicities for a reaction "
                   "MF3 lacks."),
          (WARN, f"a level scheme off MF3 by more than {_g(LEVEL_SLACK_EV / 1e3)} keV; transition "
                 f"probabilities off 1 by more than {_g(TP_WARN)}; capture energy off by more than "
                 f"{_pct(CAPTURE_WARN)}; photons without MF14; MF15 without a continuum photon."),
          (NOTE, "smaller disagreements, down to rounding.")]),
    ]

"""MF2/MT151: the resonance parameters, as written.

**Ranges.** Each isotope's energy ranges have EL < EH, follow each other with
no gap and no overlap, and the unresolved range lies above the resolved one;
the abundances of a natural element add up to 1 (FIZCON's CKF2).

**Spins.** A neutron (spin 1/2) on a target of spin I forms the channel spins
``s = |I - 1/2|, I + 1/2``; with orbital momentum l a resonance's J must be
reachable, ``|s - l| <= J <= s + l`` in integer steps, for some s (FIZCON,
FUDGE: severe). Pu-243 and U-241 of ENDF/B-VIII.1 give J = I at l = 0, which
no channel reaches. An R-matrix spin group's channels couple their own
particle spins, l and channel spin to the group's J. Every resolved range
states the same target spin (FRENDY's check_spin_data).

**Widths.** SLBW/MLBW: widths non-negative and GT = GN + GG + GF, a
competitive width making up any difference only when LRX says there is one
(rounded to the eleven characters of a field). Reich-Moore: GN and GG
non-negative (the fission widths are signed amplitudes). Unresolved: level
spacing positive, average widths non-negative, the degrees of freedom of the
neutron width 1 or 2 (FIZCON's TESTDF), and the energy-dependent parameters
covering the range they are given for (FUDGE: fatal otherwise).

**Scattering radius.** AP against the systematics ``0.123 A^(1/3) + 0.08``
(10^-12 cm): outside a factor ``AP_FACTOR`` a note. FUDGE's own check of this
is dead code; PSYCHE's band (0.07-0.17 A^(1/3)) is narrower.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Tape, fmt_e

#: Relative rounding of an eleven-column ENDF field (six significant figures).
ROUNDING = 5e-6
#: Range boundaries meet within this, relative.
RANGE_TOLERANCE = 1e-5
#: Abundances summing to 1 (FIZCON's 1e-3).
ABUNDANCE_WARN = 1e-3
#: AP outside [systematics / AP_FACTOR, systematics * AP_FACTOR].
AP_FACTOR = 2.0
#: A width sum GT - (GN + GG + GF) beyond rounding, relative to GT.
WIDTH_SUM_NOTE = 1e-4
WIDTH_SUM_WARN = 1e-2


def _reachable(j: float, I: float, l: int, s_values=None) -> bool:
    j = abs(j)
    ss = s_values if s_values is not None else sorted({abs(I - 0.5), I + 0.5})
    for s in ss:
        lo, hi = abs(s - l), s + l
        if lo - 1e-6 <= j <= hi + 1e-6 and abs((j - lo) - round(j - lo)) < 1e-6:
            return True
    return False


def _allowed(I: float, l: int) -> List[float]:
    out = set()
    for s in {abs(I - 0.5), I + 0.5}:
        j = abs(s - l)
        while j <= s + l + 1e-9:
            out.add(round(j * 2) / 2)
            j += 1
    return sorted(out)


def check_resonances(tape: Tape, out: Out) -> None:
    sec = tape.section(2, 151)
    if sec is None:
        return
    isotopes = list(getattr(sec, "isotopes", None) or getattr(sec, "_isotopes", []) or [])
    if len(isotopes) > 1:
        s = sum(float(getattr(i, "abn", 0.0)) for i in isotopes)
        if abs(s - 1) > ABUNDANCE_WARN:
            out.add("resonance_abundance", WARN, f"the abundances of the {len(isotopes)} isotopes add up to {s:.6g}",
                    mf=2, mt=151, total=s)
    spins = set()
    for iso_index, iso in enumerate(isotopes):
        ranges = list(getattr(iso, "energy_ranges", []) or [])
        _ranges(out, ranges, iso_index)
        for r_index, rng in enumerate(ranges):
            p = getattr(rng, "parameters", None)
            if p is None:
                continue
            spi = getattr(p, "spi", None)
            if spi is not None and rng.lru in (1, 2):
                spins.add(round(float(spi), 3))
            ap = float(getattr(p, "ap", 0.0) or 0.0)
            awri = _awri(p) or tape.awr
            if rng.lru == 1 and rng.lrf in (1, 2, 3):
                _resolved(out, p, rng, r_index)
            elif rng.lru == 1 and rng.lrf == 7:
                _rml(out, p, r_index)
            elif rng.lru == 2:
                _unresolved(out, p, rng, r_index)
            if ap > 0 and awri and rng.lru in (0, 1, 2) and int(getattr(rng, "nro", 0) or 0) == 0:
                a = awri * 1.00866491595
                sys_ap = 0.123 * a ** (1 / 3) + 0.08
                if not sys_ap / AP_FACTOR <= ap <= sys_ap * AP_FACTOR:
                    out.add("scattering_radius", NOTE,
                            f"range {r_index}: AP = {ap:.5g} x 1e-12 cm, the systematics give {sys_ap:.3g}: "
                            f"outside a factor {AP_FACTOR:g}", mf=2, mt=151, range_index=r_index, ap=ap,
                            systematics=sys_ap)
            elif ap <= 0 and rng.lru in (0, 1, 2) and rng.lrf != 7 and int(getattr(rng, "nro", 0) or 0) == 0:
                out.add("scattering_radius", DEFECT,
                        f"range {r_index}: AP = {ap:g}, a scattering radius that is not positive",
                        mf=2, mt=151, range_index=r_index, ap=ap)
    if len(spins) > 1:
        out.add("resonance_spin", WARN,
                f"the resonance ranges state different target spins ({', '.join(f'{s:g}' for s in sorted(spins))})",
                mf=2, mt=151, spins=sorted(spins))


def _awri(p) -> Optional[float]:
    for lv in getattr(p, "l_values", []) or []:
        a = getattr(lv, "awri", None)
        if a:
            return float(a)
    return None


def _ranges(out: Out, ranges, iso_index: int) -> None:
    prev = None
    seen_urr = False
    for i, r in enumerate(ranges):
        el, eh = float(r.el), float(r.eh)
        if not el < eh:
            out.add("resonance_ranges", DEFECT, f"range {i}: EL = {fmt_e(el)} is not below EH = {fmt_e(eh)}",
                    mf=2, mt=151, range_index=i, el=el, eh=eh)
        if r.lru == 2:
            seen_urr = True
        elif r.lru == 1 and seen_urr:
            out.add("resonance_ranges", DEFECT, f"range {i}: a resolved range after an unresolved one",
                    mf=2, mt=151, range_index=i)
        if prev is not None and r.lru != 0 and prev.lru != 0:
            p_eh = float(prev.eh)
            if el > p_eh * (1 + RANGE_TOLERANCE):
                out.add("resonance_ranges", WARN,
                        f"ranges {i - 1} and {i} leave a gap from {fmt_e(p_eh)} to {fmt_e(el)}: no resonances "
                        "and only MF3 there", mf=2, mt=151, range_index=i, gap_from=p_eh, gap_to=el)
            elif el < p_eh * (1 - RANGE_TOLERANCE):
                out.add("resonance_ranges", DEFECT,
                        f"ranges {i - 1} and {i} overlap from {fmt_e(el)} to {fmt_e(p_eh)}: two sets of "
                        "resonances there", mf=2, mt=151, range_index=i, overlap_from=el, overlap_to=p_eh)
        prev = r


def _resolved(out: Out, p, rng, r_index: int) -> None:
    I = float(p.spi)
    lrf = rng.lrf
    bad_j, neg, sums = [], [], []
    present = {}
    for lv in p.l_values:
        l = int(lv.l)
        for res in lv.resonances:
            present.setdefault(l, set()).add(round(abs(float(res.spin)) * 2) / 2)
            if not _reachable(float(res.spin), I, l):
                bad_j.append((l, float(res.spin), float(res.energy)))
            if lrf in (1, 2):
                gt, gn, gg, gf = res.c3, res.c4, res.c5, res.c6
                if min(gn, gg, gf, gt) < 0:
                    neg.append((float(res.energy), l))
                elif gt > 0:
                    d = gt - (gn + gg + gf)
                    rel = abs(d) / gt
                    if rel > max(WIDTH_SUM_NOTE, ROUNDING) and (int(getattr(lv, "lrx", 0) or 0) == 0 or d < 0):
                        sums.append((rel, float(res.energy), l, d))
            else:
                if res.c3 < 0 or res.c4 < 0:
                    neg.append((float(res.energy), l))
    if bad_j:
        l, j, e = bad_j[0]
        out.add("resonance_spin", DEFECT,
                f"range {r_index}: {len(bad_j)} resonance{'s' if len(bad_j) != 1 else ''} with a J that a neutron "
                f"on a target of spin {I:g} cannot reach; first J = {j:g} at l = {l}, E = {fmt_e(e)} "
                f"(allowed: {', '.join(f'{x:g}' for x in _allowed(I, l))})", mf=2, mt=151, range_index=r_index,
                count=len(bad_j), j=j, l_value=l, at=e, spi=I)
    for l, js in present.items():
        missing = [j for j in _allowed(I, l) if j not in js]
        if missing and len(js) > 0:
            out.add("resonance_channels", NOTE,
                    f"range {r_index}: l = {l} has resonances only for J = {', '.join(f'{x:g}' for x in sorted(js))}; "
                    f"none for {', '.join(f'{x:g}' for x in missing)}", mf=2, mt=151, range_index=r_index,
                    l_value=l, missing=missing)
    if neg:
        e, l = neg[0]
        out.add("resonance_width", DEFECT,
                f"range {r_index}: {len(neg)} resonance{'s' if len(neg) != 1 else ''} with a negative width; first "
                f"at E = {fmt_e(e)}, l = {l}", mf=2, mt=151, range_index=r_index, count=len(neg), at=e)
    if sums:
        rel, e, l, d = max(sums)
        out.add("resonance_width", WARN if rel > WIDTH_SUM_WARN else NOTE,
                f"range {r_index}: {len(sums)} resonance{'s' if len(sums) != 1 else ''} whose GT is not GN + GG + GF "
                f"(and no competitive width is declared, or it would be negative); worst {rel:.2g} of GT at "
                f"E = {fmt_e(e)}", mf=2, mt=151, range_index=r_index, count=len(sums), relative=rel, at=e)


def _rml(out: Out, p, r_index: int) -> None:
    pairs = list(getattr(p, "particle_pairs", []) or [])
    bad = []
    for g_index, sg in enumerate(getattr(p, "spin_groups", []) or []):
        J = abs(float(sg.aj))
        for ch in sg.channels:
            ipp = int(ch.ipp)
            if not 1 <= ipp <= len(pairs):
                bad.append((g_index, f"channel names particle pair {ipp} of {len(pairs)}"))
                continue
            pp = pairs[ipp - 1]
            s, l = abs(float(ch.sch)), int(ch.l)
            ia, ib = abs(float(pp.ia)), abs(float(pp.ib))
            if pp.mt == 102:
                continue        # the capture channel's spins are a convention
            if not (abs(ia - ib) - 1e-6 <= s <= ia + ib + 1e-6):
                bad.append((g_index, f"channel spin {s:g} from particle spins {ia:g} and {ib:g}"))
            elif not _reachable(J, 0.0, l, s_values=[s]):
                bad.append((g_index, f"J = {J:g} from l = {l} and channel spin {s:g} (MT{pp.mt})"))
    if bad:
        g, why = bad[0]
        out.add("resonance_spin", DEFECT,
                f"range {r_index}: {len(bad)} R-matrix channel{'s' if len(bad) != 1 else ''} whose spins do not "
                f"couple; first in spin group {g}: {why}", mf=2, mt=151, range_index=r_index, count=len(bad))


def _unresolved(out: Out, p, rng, r_index: int) -> None:
    I = float(p.spi)
    el, eh = float(rng.el), float(rng.eh)
    bad_j, bad_d, bad_w, dof, cover = [], [], [], [], []
    for lv in getattr(p, "l_values", []) or []:
        l = int(lv.l)
        for js in lv.j_states:
            aj = float(js.aj)
            if not _reachable(aj, I, l):
                bad_j.append((l, aj))
            amun = float(getattr(js, "amun", 1) or 0)
            if not (1 - 1e-6 <= amun <= 2 + 1e-6) or abs(amun - round(amun)) > 1e-6:
                dof.append((l, aj, amun))
            pts = getattr(js, "energy_points", None)
            if pts:
                es = [float(q.es) for q in pts]
                if es[0] > el * (1 + RANGE_TOLERANCE) or es[-1] < eh * (1 - RANGE_TOLERANCE):
                    cover.append((l, aj, es[0], es[-1]))
                for q in pts:
                    if q.d <= 0:
                        bad_d.append((l, aj, float(q.es)))
                    if min(q.gn0, q.gg, q.gf, q.gx) < 0:
                        bad_w.append((l, aj, float(q.es)))
            else:
                if getattr(js, "d", 1.0) <= 0:
                    bad_d.append((l, aj, el))
                widths = [getattr(js, "gn0", 0.0), getattr(js, "gg", 0.0)] + list(getattr(js, "gf", []) or [])
                if min(widths) < 0:
                    bad_w.append((l, aj, el))
    if bad_j:
        l, j = bad_j[0]
        out.add("resonance_spin", DEFECT,
                f"unresolved range {r_index}: {len(bad_j)} (l, J) state{'s' if len(bad_j) != 1 else ''} a neutron on "
                f"spin {I:g} cannot form; first J = {j:g} at l = {l} (allowed: "
                f"{', '.join(f'{x:g}' for x in _allowed(I, l))})", mf=2, mt=151, range_index=r_index,
                count=len(bad_j), j=j, l_value=l, spi=I)
    if bad_d:
        l, j, e = bad_d[0]
        out.add("urr_parameters", DEFECT, f"unresolved range {r_index}: level spacing D not positive (l = {l}, "
                f"J = {j:g}, E = {fmt_e(e)})", mf=2, mt=151, range_index=r_index, count=len(bad_d))
    if bad_w:
        l, j, e = bad_w[0]
        out.add("urr_parameters", DEFECT, f"unresolved range {r_index}: a negative average width (l = {l}, "
                f"J = {j:g}, E = {fmt_e(e)})", mf=2, mt=151, range_index=r_index, count=len(bad_w))
    if dof:
        l, j, mu = dof[0]
        out.add("urr_degrees_of_freedom", NOTE,
                f"unresolved range {r_index}: {len(dof)} state{'s' if len(dof) != 1 else ''} with a neutron-width "
                f"degree of freedom other than 1 or 2; first {mu:g} (l = {l}, J = {j:g})",
                mf=2, mt=151, range_index=r_index, count=len(dof), amun=mu)
    if cover:
        l, j, lo, hi = cover[0]
        out.add("urr_coverage", WARN,
                f"unresolved range {r_index} ({fmt_e(el)} to {fmt_e(eh)}): the parameters of {len(cover)} state"
                f"{'s' if len(cover) != 1 else ''} only span {fmt_e(lo)} to {fmt_e(hi)} (first l = {l}, J = {j:g}); "
                "beyond them a code extrapolates", mf=2, mt=151, range_index=r_index, count=len(cover))


def descriptions(D, _g, _pct):
    return {
        "resonance_ranges": D(
            "Resonance ranges malformed", (2,),
            {DEFECT: "EL >= EH, ranges that overlap, a resolved range above an unresolved one",
             WARN: "a gap between ranges"},
            "Each isotope's resonance ranges follow each other in energy, without gaps or overlaps, "
            "the unresolved above the resolved (FIZCON's CKF2; FUDGE's translator)."),
        "resonance_abundance": D(
            "Isotope abundances do not add up to 1", (2,), {WARN: f"off by more than {_g(ABUNDANCE_WARN)}"},
            "The isotopes of a natural element in MF2 share it (FIZCON)."),
        "resonance_spin": D(
            "Resonance spin that cannot be reached", (2,),
            {DEFECT: "a J no channel reaches, or R-matrix channel spins that do not couple",
             WARN: "ranges with different target spins"},
            "A neutron on a target of spin I forms channel spins |I - 1/2| and I + 1/2, and with orbital "
            "momentum l a resonance's J lies between |s - l| and s + l in integer steps (FIZCON; FUDGE: "
            "severe). Found in ENDF/B-VIII.1 Pu-243 and U-241 (J = I at l = 0)."),
        "resonance_channels": D(
            "Spin channel without resonances", (2,), {NOTE: "always"},
            "An allowed (l, J) has no resonance (FUDGE's missing-channel check). Codes still give it potential "
            "scattering; whether it is real or the evaluation left it out is for the evaluator."),
        "resonance_width": D(
            "Resonance widths", (2,),
            {DEFECT: "a negative width (SLBW/MLBW; GN or GG for Reich-Moore)",
             WARN: f"GT != GN + GG + GF by more than {_pct(WIDTH_SUM_WARN)} without a competitive width",
             NOTE: f"by more than {_g(WIDTH_SUM_NOTE)}"},
            "Widths are rates: not negative. In SLBW/MLBW the total width is the sum of the partial ones, a "
            "competitive width (LRX = 1) making up the difference (FIZCON's TESTW)."),
        "scattering_radius": D(
            "Scattering radius", (2,),
            {DEFECT: "not positive", NOTE: f"outside a factor {_g(AP_FACTOR)} of 0.123 A^(1/3) + 0.08"},
            "AP against the systematics 1.23 A^(1/3) + 0.8 fm. FUDGE's version of this check is dead code; "
            "a radius far from it is possible (light nuclei, fitted radii) and so only a note."),
        "urr_parameters": D(
            "Unresolved parameters out of range", (2,), {DEFECT: "always"},
            "In the unresolved range the mean level spacing is positive and the average widths are not "
            "negative (FUDGE: severe)."),
        "urr_degrees_of_freedom": D(
            "Neutron-width degrees of freedom", (2,), {NOTE: "always"},
            "The neutron width of an s-wave or p-wave channel has 1 or 2 degrees of freedom (FIZCON's "
            "TESTDF). TALYS writes non-integer values; codes read them as written."),
        "urr_coverage": D(
            "Unresolved parameters do not span their range", (2,), {WARN: "always"},
            "Energy-dependent unresolved parameters cover the range they are given for (FUDGE: fatal). "
            "Found in JENDL-5 Pr-141."),
    }


def method(_g, _pct):
    return [
        ("Resonances",
         "MF2/MT151 as written: ranges, the spin algebra of every resonance and R-matrix channel, widths, "
         "radii and the unresolved parameters. Nothing is reconstructed.",
         [(DEFECT, "a J no neutron channel reaches; overlapping or reversed ranges; a negative width or level "
                   "spacing; a scattering radius that is not positive."),
          (WARN, f"gaps between ranges; GT != GN + GG + GF by more than {_pct(WIDTH_SUM_WARN)}; unresolved "
                 "parameters that do not span their range; different target spins."),
          (NOTE, f"a radius outside a factor {_g(AP_FACTOR)} of the systematics; channels without resonances; "
                 "non-integer degrees of freedom; smaller width sums.")]),
    ]

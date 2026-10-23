"""MF1: the header, the directory, nu-bar, delayed neutrons and MT458.

**Header (MF1/451).** Every section repeats the material's ZA and AWR in its
HEAD record; they must be the header's. The flags must say what the tape
holds: LRP whether MF2 is there, LFI whether there is fission, LIS and ELIS
together. EMAX is the top of the evaluation and the cross sections should
reach it. AWR is compared with AME2020.

**Directory.** MF1/451 lists every section with its number of lines. A
section missing from it, or listed and absent, misleads any reader that
trusts the directory (CHECKR's TESTD); a wrong line count is a note, since
no processor uses it to read.

**nu-bar.** MT452 = MT455 + MT456 wherever all three are given (FIZCON's
SUM452, 1e-3); every nu-bar positive and in range (0-20, delayed 0-1); the
tables reaching the top of fission's MF3. A delayed nu-bar needs its
spectra (MF5/455) and as many precursor families there as MF1/455 has
decay constants, each positive.

**MT458.** The components non-negative; ET the sum of the first seven, ER
= ET - ENU (FIZCON's CKS458, 1e-3); the LFC=1 tables, which no other code
checks, non-negative.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import NEUTRON_MASS_U, Out, Table, Tape, fmt_e
from .reactions import mass_u
from .tables import check_table

#: AWR against AME2020, relative: a note above the first, a warning above the
#: second (a 0.1 % error in AWR is a 0.1 % error in every threshold and recoil).
AWR_NOTE = 1e-4
AWR_WARN = 1e-3
#: AWR of a section's HEAD against the header's, relative.
HEAD_AWR_NOTE = 1e-5
HEAD_AWR_WARN = 1e-3
#: nu-bar sum rule, relative to MT452 (FIZCON's 1e-3).
NUBAR_SUM_NOTE = 1e-5
NUBAR_SUM_WARN = 1e-3
#: Physically possible nu-bar (FIZCON's CHKNUB bounds).
NUBAR_MAX = 20.0
NUBAR_DELAYED_MAX = 1.0
#: MT458 sum rules, relative to ET (FIZCON's 1e-3).
FER_SUM_NOTE = 1e-5
FER_SUM_WARN = 1e-3
#: Delayed-group abundances summing to 1 (ACER's consis: 1e-3).
ABUNDANCE_NOTE = 1e-5
ABUNDANCE_WARN = 1e-3
#: EMAX against where MF3 stops, relative.
EMAX_TOLERANCE = 1e-6

FISSION_MT = (18, 19, 20, 21, 38)


def check_general(tape: Tape, out: Out) -> None:
    h = tape.header
    if h is None:
        out.add("header_missing", DEFECT,
                "no MF1/MT451: the tape does not say what material, projectile or "
                "sublibrary it is", mf=1, mt=451)
        return
    _heads(tape, out)
    _flags(tape, out)
    _directory(tape, out)
    _awr(tape, out)
    _emax(tape, out)
    _nubar(tape, out)
    _delayed(tape, out)
    _fission_energy(tape, out)


# --------------------------------------------------------------------------- header

def _heads(tape: Tape, out: Out) -> None:
    za, awr = tape.za, tape.awr
    if za is None or awr is None:
        return
    bad_za: List[Tuple[int, int, int]] = []
    awr_off: List[Tuple[int, int, float]] = []
    for mf in tape.present:
        if mf == 1 or mf not in tape.wanted:
            continue
        for mt, sec in tape.sections(mf).items():
            sza = getattr(sec, "_za", None)
            sawr = getattr(sec, "_awr", None)
            if sza is not None and int(round(float(sza))) != za and sza != 0:
                bad_za.append((mf, mt, int(round(float(sza)))))
            if sawr is not None and float(sawr) > 0 and awr > 0:
                rel = abs(float(sawr) - awr) / awr
                if rel > HEAD_AWR_NOTE:
                    awr_off.append((mf, mt, float(sawr)))
    for mf, mt, sza in bad_za[:20]:
        out.add("head_za_mismatch", DEFECT,
                f"the HEAD of MF{mf}/MT{mt} says ZA = {sza}, the header ZA = {za}",
                mf=mf, mt=mt, za=sza, header_za=za)
    if len(bad_za) > 20:
        out.add("head_za_mismatch", DEFECT, f"... and {len(bad_za) - 20} more sections",
                mf=bad_za[20][0], mt=bad_za[20][1])
    by_mf: Dict[int, List[Tuple[int, float]]] = {}
    for mf, mt, sawr in awr_off:
        by_mf.setdefault(mf, []).append((mt, sawr))
    for mf, items in by_mf.items():
        worst_mt, worst = max(items, key=lambda it: abs(it[1] - awr))
        rel = abs(worst - awr) / awr
        out.add("head_awr_mismatch", WARN if rel > HEAD_AWR_WARN else NOTE,
                f"{len(items)} section{'s' if len(items) != 1 else ''} of MF{mf} state an AWR other "
                f"than the header's {awr:.7g}; the furthest, MT{worst_mt}, {worst:.7g} "
                f"({rel:.2g} relative)", mf=mf, mt=worst_mt, sections=len(items), awr=worst,
                header_awr=awr, relative=rel)


def _flags(tape: Tape, out: Out) -> None:
    nsub = tape.nsub
    has2 = tape.has(2)
    lrp = tape.lrp
    neutron = nsub == 10
    if neutron and lrp is not None:
        if lrp == -1 and has2:
            out.add("header_flags", WARN, "LRP = -1 says there is no MF2, but the tape has one",
                    mf=1, mt=451, flag="LRP", value=lrp)
        if lrp in (0, 1, 2) and not has2:
            out.add("header_flags", DEFECT,
                    f"LRP = {lrp} says MF2 is given (with at least the scattering radius), but the "
                    "tape has none", mf=1, mt=451, flag="LRP", value=lrp)
    lfi = tape.lfi
    fission = [mt for mt in FISSION_MT if tape.section(3, mt) is not None]
    if neutron and lfi is not None:
        if lfi == 0 and fission:
            out.add("header_flags", WARN,
                    f"LFI = 0 says the material does not fission, but MF3 has MT{fission[0]}",
                    mf=1, mt=451, flag="LFI", value=lfi)
        if lfi == 1 and not fission and tape.has(3):
            out.add("header_flags", NOTE,
                    "LFI = 1 says the material fissions, but MF3 has no fission reaction",
                    mf=1, mt=451, flag="LFI", value=lfi)
    lis = int(tape._h("_lis", 0) or 0)
    elis = tape.elis
    if (lis == 0) != (elis == 0):
        # The decay sublibrary often leaves an isomer's ELIS at 0: true, and
        # worth knowing, but no processing reads it there.
        out.add("header_flags", NOTE if tape.nsub == 4 else WARN,
                f"LIS = {lis} and ELIS = {fmt_e(elis)}: a ground state has no excitation energy "
                "and an excited state has one", mf=1, mt=451, flag="LIS/ELIS", lis=lis, elis=elis)
    awi = tape._h("_awi")
    if awi is not None and nsub in (10, 12) and abs(float(awi) - 1.0) > 1e-6:
        out.add("header_flags", WARN,
                f"AWI = {float(awi):g}: an incident-neutron sublibrary (NSUB = {nsub}) has the "
                "neutron as projectile, AWI = 1", mf=1, mt=451, flag="AWI", value=float(awi))


def _directory(tape: Tape, out: Out) -> None:
    h = tape.header
    entries = list(getattr(h, "_directory", []) or [])
    nxc = getattr(h, "_nxc", None)
    if nxc is not None and nxc != len(entries):
        out.add("directory_mismatch", DEFECT,
                f"NXC = {nxc} directory entries declared, {len(entries)} read",
                mf=1, mt=451, nxc=nxc, read=len(entries))
    raw = tape.raw()
    if raw is None or not entries:
        return
    listed = {(int(e[0]), int(e[1])): int(e[2]) for e in entries}
    present = {k: v for k, v in raw.lines.items()}
    missing = [k for k in raw.order if k not in listed]
    absent = [k for k in listed if k not in present]
    if missing:
        out.add("directory_mismatch", WARN,
                f"{len(missing)} section{'s' if len(missing) != 1 else ''} on the tape not in the "
                f"MF1/451 directory: {_keys(missing)}", mf=1, mt=451, sections=_keys(missing, 0))
    if absent:
        out.add("directory_mismatch", WARN,
                f"{len(absent)} section{'s' if len(absent) != 1 else ''} in the directory not on the "
                f"tape: {_keys(absent)}", mf=1, mt=451, sections=_keys(absent, 0))
    wrong = [(k, listed[k], present[k]) for k in listed if k in present and listed[k] != present[k]]
    if wrong:
        k, nc, n = wrong[0]
        out.add("directory_line_count", NOTE,
                f"{len(wrong)} directory entr{'ies' if len(wrong) != 1 else 'y'} give a line count "
                f"other than the section's; first MF{k[0]}/MT{k[1]}: NC = {nc}, {n} lines",
                mf=1, mt=451, entries=len(wrong), first=[k[0], k[1], nc, n])
    if raw.split:
        out.add("section_split", DEFECT,
                f"{len(raw.split)} section{'s' if len(raw.split) != 1 else ''} split in two places on "
                f"the tape: {_keys(raw.split)}; a reader takes the first part only",
                mf=raw.split[0][0], mt=raw.split[0][1], sections=_keys(raw.split, 0))
    mats = sorted(raw.mats)
    if len(mats) > 1:
        out.add("tape_materials", NOTE,
                f"the tape holds {len(mats)} materials (MAT {', '.join(map(str, mats[:6]))}); only "
                f"MAT {tape.mat} was checked", mf=1, mt=451, mats=mats)


def _keys(keys, limit: int = 8):
    text = [f"MF{a}/MT{b}" for a, b in keys]
    if limit == 0:
        return text
    return ", ".join(text[:limit]) + (f" (+{len(text) - limit})" if len(text) > limit else "")


def _awr(tape: Tape, out: Out) -> None:
    za, awr = tape.za, tape.awr
    if za is None or awr is None or tape.nsub in (3, 6, 113, 12) or za % 1000 == 0:
        return
    m = mass_u(za)
    if m is None:
        return
    expected = m / NEUTRON_MASS_U
    rel = abs(awr - expected) / expected
    if rel > AWR_NOTE:
        out.add("awr_mass", WARN if rel > AWR_WARN else NOTE,
                f"AWR = {awr:.7g}, AME2020 gives {expected:.7g} for ZA {za} ({rel:.2g} relative)",
                mf=1, mt=451, awr=awr, ame=expected, relative=rel)


def _emax(tape: Tape, out: Out) -> None:
    emax = tape.emax
    if emax is None or not tape.has(3) or tape.nsub in (3, 6, 113, 12):
        return
    top = 0.0
    for mt in tape.sections(3):
        t = tape.xs(mt)
        if t is not None and t.x.size:
            top = max(top, float(np.max(t.x)))
    if top <= 0:
        return
    if top < emax * (1 - EMAX_TOLERANCE):
        out.add("emax_coverage", WARN,
                f"EMAX = {fmt_e(emax)}, but no cross section goes beyond {fmt_e(top)}: above it "
                "every reaction reads as zero", mf=1, mt=451, emax=emax, top=top)
    elif top > emax * (1 + EMAX_TOLERANCE):
        out.add("emax_coverage", NOTE,
                f"cross sections reach {fmt_e(top)}, above EMAX = {fmt_e(emax)}",
                mf=1, mt=451, emax=emax, top=top)


# --------------------------------------------------------------------------- nu-bar

def _nu_table(sec) -> Optional[Table]:
    if sec is None or getattr(sec, "_lnu", None) != 2:
        return None
    return Table.of(getattr(sec, "_energies", []), getattr(sec, "_nubar", []),
                    getattr(sec, "_interpolation", []))


def _nu_values(sec, grid: np.ndarray) -> Optional[np.ndarray]:
    lnu = getattr(sec, "_lnu", None)
    if lnu == 1:
        c = [float(v) for v in getattr(sec, "_coefficients", [])]
        return np.polynomial.polynomial.polyval(grid, c) if c else None
    t = _nu_table(sec)
    if t is None or t.x.size == 0:
        return None
    return t(np.clip(grid, t.x[0], t.x[-1]))


def _fission_top(tape: Tape) -> Optional[float]:
    tops = [float(tape.xs(mt).x[-1]) for mt in FISSION_MT
            if tape.xs(mt) is not None and tape.xs(mt).x.size]
    return max(tops) if tops else None


def _nubar(tape: Tape, out: Out) -> None:
    secs = {mt: tape.section(1, mt) for mt in (452, 455, 456)}
    if not any(secs.values()):
        if tape.neutron_projectile and tape.section(3, 18) is not None:
            out.add("nubar_missing", DEFECT,
                    "MF3 has fission (MT18) but MF1 gives no nu-bar (MT452)", mf=1, mt=452)
        return
    if secs[452] is None and (secs[455] is not None or secs[456] is not None):
        out.add("nubar_missing", DEFECT,
                "MF1 gives prompt or delayed nu-bar but not the total (MT452), which ENDF-6 "
                "requires whenever either is given", mf=1, mt=452)
    top = _fission_top(tape) or tape.emax
    grids = []
    names = {452: "total nu-bar", 455: "delayed nu-bar", 456: "prompt nu-bar"}
    for mt, sec in secs.items():
        if sec is None:
            continue
        lnu = getattr(sec, "_lnu", None)
        if lnu not in (1, 2):
            out.add("nubar_representation", DEFECT, f"{names[mt]}: LNU = {lnu} is not 1 or 2",
                    mf=1, mt=mt, lnu=lnu)
            continue
        t = _nu_table(sec)
        if t is not None:
            if not check_table(t, out, mf=1, mt=mt, what=names[mt]):
                continue
            grids.append(t.x)
            if top is not None and t.x.size and t.x[-1] < top * (1 - EMAX_TOLERANCE):
                out.add("nubar_coverage", WARN,
                        f"{names[mt]} stops at {fmt_e(t.x[-1])}, fission goes on to {fmt_e(top)}: "
                        "above it a code holds the last value or reads zero",
                        mf=1, mt=mt, last=float(t.x[-1]), fission_top=top)
        hi = top or 2e7
        grid = t.x if t is not None else np.geomspace(1e-5, hi, 200)
        nu = _nu_values(sec, grid)
        if nu is None:
            continue
        cap = NUBAR_DELAYED_MAX if mt == 455 else NUBAR_MAX
        neg = np.flatnonzero(nu < 0)
        if neg.size:
            i = int(neg[0])
            out.add("nubar_range", DEFECT,
                    f"{names[mt]} is negative at {neg.size} point{'s' if neg.size != 1 else ''}, "
                    f"{nu[i]:.6g} at {fmt_e(grid[i])}", mf=1, mt=mt, count=int(neg.size),
                    value=float(nu[i]), at=float(grid[i]))
        high = np.flatnonzero(nu > cap)
        if high.size:
            i = int(high[np.argmax(nu[high])])
            out.add("nubar_range", WARN,
                    f"{names[mt]} reaches {nu[i]:.6g} at {fmt_e(grid[i])}, above the {cap:g} no "
                    "fission releases", mf=1, mt=mt, value=float(nu[i]), at=float(grid[i]))
    if all(secs[mt] is not None for mt in (452, 455, 456)):
        grid = np.unique(np.concatenate(grids)) if grids else np.geomspace(1e-5, top or 2e7, 200)
        if top is not None:
            grid = grid[grid <= top]
        tot = _nu_values(secs[452], grid)
        dl = _nu_values(secs[455], grid)
        pr = _nu_values(secs[456], grid)
        if tot is None or dl is None or pr is None or not grid.size:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(tot - dl - pr) / np.abs(tot)
        rel = np.where(np.isfinite(rel), rel, 0.0)
        i = int(np.argmax(rel))
        if rel[i] > NUBAR_SUM_NOTE:
            out.add("nubar_sum", WARN if rel[i] > NUBAR_SUM_WARN else NOTE,
                    f"MT452 differs from MT455 + MT456 by {rel[i]:.2g} of it at {fmt_e(grid[i])} "
                    f"({tot[i]:.6g} against {dl[i]:.6g} + {pr[i]:.6g})",
                    mf=1, mt=452, relative=float(rel[i]), at=float(grid[i]),
                    total=float(tot[i]), delayed=float(dl[i]), prompt=float(pr[i]))


def _delayed(tape: Tape, out: Out) -> None:
    sec = tape.section(1, 455)
    if sec is None:
        return
    ldg = getattr(sec, "_ldg", 0)
    lam: List[float] = []
    if ldg == 0:
        lam = [float(v) for v in getattr(sec, "_decay_constants", [])]
    else:
        for e, values in zip(getattr(sec, "_decay_energies", []), getattr(sec, "_decay_data", [])):
            pairs = np.asarray(values, dtype=float).reshape(-1, 2) if len(values) % 2 == 0 else None
            if pairs is None:
                continue
            lam += pairs[:, 0].tolist()
            s = float(pairs[:, 1].sum())
            if abs(s - 1) > ABUNDANCE_NOTE:
                out.add("delayed_abundances", WARN if abs(s - 1) > ABUNDANCE_WARN else NOTE,
                        f"the precursor abundances at {fmt_e(float(e))} sum to {s:.7g}, not 1",
                        mf=1, mt=455, at=float(e), total=s)
    bad = [v for v in lam if not v > 0]
    if bad:
        out.add("delayed_constants", DEFECT,
                f"{len(bad)} precursor decay constant{'s' if len(bad) != 1 else ''} not positive "
                f"({bad[0]:g} s^-1)", mf=1, mt=455, values=bad[:10])
    nnf = int(getattr(sec, "_nnf", 0) or 0)
    nu = _nu_values(sec, np.array([0.0253, 1e6]))
    if nu is None or not np.any(nu > 0):
        return
    spec5 = tape.section(5, 455)
    spec6 = tape.section(6, 455)
    if spec5 is None and spec6 is None:
        out.add("delayed_spectra_missing", DEFECT,
                f"MF1/455 gives a delayed nu-bar ({float(np.max(nu)):.4g}) but there is no delayed "
                "neutron spectrum (MF5/455): a code has no energy to give those neutrons",
                mf=1, mt=455, nubar=float(np.max(nu)))
        return
    if spec5 is not None:
        nk = len(getattr(spec5, "partials", []) or [])
        if nnf and nk and nk != nnf:
            out.add("delayed_groups_mismatch", DEFECT,
                    f"MF1/455 has {nnf} precursor families, MF5/455 {nk} spectra: one spectrum per "
                    "family is what pairs them", mf=1, mt=455, nnf=nnf, nk=nk)


# --------------------------------------------------------------------------- MT458

_NAMES = ["EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET"]


def _fission_energy(tape: Tape, out: Out) -> None:
    sec = tape.section(1, 458)
    if sec is None:
        if tape.neutron_projectile and tape.section(3, 18) is not None:
            out.add("fission_energy_missing", NOTE,
                    "MF3 has fission but MF1 gives no fission energy release (MT458): heating has "
                    "to assume how the energy is shared", mf=1, mt=458)
        return
    lfc = getattr(sec, "_lfc", 0)
    vals = (getattr(sec, "_thermal_values", None) if lfc == 1 else getattr(sec, "_coefficients", None)) or []
    if len(vals) < 18:
        out.add("fission_energy_record", DEFECT,
                f"MT458 holds {len(vals)} values, fewer than the 18 of its thermal components",
                mf=1, mt=458, values=len(vals))
        return
    v = np.asarray(vals[:18], dtype=float).reshape(9, 2)
    comp, unc = v[:, 0], v[:, 1]
    neg = [(_NAMES[i], comp[i]) for i in range(9) if comp[i] < 0]
    if neg:
        out.add("fission_energy_negative", DEFECT,
                "negative fission energy component" + ("s" if len(neg) > 1 else "") + ": "
                + ", ".join(f"{n} = {fmt_e(x)}" for n, x in neg), mf=1, mt=458,
                components={n: float(x) for n, x in neg})
    neg_u = [_NAMES[i] for i in range(9) if unc[i] < 0]
    if neg_u:
        out.add("fission_energy_negative", WARN,
                f"negative uncertainty of {', '.join(neg_u)}", mf=1, mt=458, uncertainties=neg_u)
    et = comp[8]
    if et > 0:
        s7 = float(comp[:7].sum())
        r1 = abs(s7 - et) / et
        r2 = abs((et - comp[6]) - comp[7]) / et
        for rel, text in ((r1, f"ET = {fmt_e(et)} but its seven components add up to {fmt_e(s7)}"),
                          (r2, f"ER = {fmt_e(comp[7])} but ET - ENU = {fmt_e(et - comp[6])}")):
            if rel > FER_SUM_NOTE:
                out.add("fission_energy_sum", WARN if rel > FER_SUM_WARN else NOTE,
                        f"{text} ({rel:.2g} of ET)", mf=1, mt=458, relative=float(rel))
    if lfc == 1:
        for k, tab in enumerate(getattr(sec, "_tab_components", []) or []):
            ifc = int(tab.get("ifc", 0))
            name = _NAMES[ifc - 1] if 1 <= ifc <= 9 else f"IFC={ifc}"
            t = Table.of(tab.get("energies", []), tab.get("values", []), tab.get("interpolation", []))
            if not check_table(t, out, mf=1, mt=458, what=f"the {name} table", k=k):
                continue
            if t.y.size and np.min(t.y) < 0:
                i = int(np.argmin(t.y))
                out.add("fission_energy_negative", DEFECT,
                        f"the tabulated {name} is negative at {fmt_e(t.x[i])} ({fmt_e(t.y[i])})",
                        mf=1, mt=458, k=k, component=name, at=float(t.x[i]), value=float(t.y[i]))

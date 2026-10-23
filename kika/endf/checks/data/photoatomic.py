"""MF23, MF27, MF28: photo-atomic cross sections, form factors and relaxation.

**Cross sections (MF23).** Every table is a TAB1 with non-negative values; the
total (MT501) is the sum of coherent (502), incoherent (504), pair production
(516) and photoelectric (522) scattering, pair production the sum of its
nuclear (515) and electron (517) fields, photoelectric the sum of its
subshells (534-599) -- compared at the energies the total is stated at
(FIZCON's optional TSTTOT, 1e-3).

**Form factors (MF27).** The coherent form factor tends to Z at zero momentum
transfer, the incoherent scattering function to Z at large momentum transfer
and to 0 at zero -- limits FUDGE leaves unchecked (its photo-atomic checks
return nothing).

**Relaxation (MF28).** The subshells of the neutral atom hold Z electrons,
their binding energies are positive, the probabilities of the transitions
that fill a vacancy add up to 1, and no transition releases more than the
vacancy's binding energy.
"""
from __future__ import annotations

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Table, Tape, fmt_e
from .tables import check_table

#: A stated photo-atomic total against its parts, relative.
SUM_NOTE = 1e-3
SUM_WARN = 1e-2
#: Form-factor limits against Z, relative.
FF_NOTE = 1e-3
FF_WARN = 1e-2
#: Relaxation probabilities of one subshell summing to 1.
RELAX_NOTE = 1e-3
RELAX_WARN = 1e-2
#: A transition energy above the vacancy's binding energy, relative: EADL takes
#: the two from different sources, and agrees to about 1e-3.
RELAX_ENERGY_NOTE = 1e-3
RELAX_ENERGY_WARN = 1e-2

_SUMS = {501: (502, 504, 516, 522), 516: (515, 517), 522: tuple(range(534, 600))}


def _stated_vs_parts(out: Out, tables, mt: int, parts) -> None:
    from kika.algebra import left_limit, right_limit

    stated = tables[mt]
    present = [p for p in parts if p in tables]
    if not present:
        return
    if mt == 501 and 516 not in tables and (515 in tables or 517 in tables):
        present += [p for p in (515, 517) if p in tables]
    g = stated.x
    lo = np.zeros(g.size)
    hi = np.zeros(g.size)
    for p in present:
        t = tables[p]
        laws = np.where(np.isin(t.laws(), (1, 2, 3, 4, 5)), t.laws(), 2)
        try:
            lo += left_limit(t.x, t.y, laws, g)
            hi += right_limit(t.x, t.y, laws, g)
        except ValueError:
            return
    total = np.where(np.abs(lo - stated.y) <= np.abs(hi - stated.y), lo, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(stated.y > 0, np.abs(total - stated.y) / stated.y, 0.0)
    i = int(np.argmax(rel))
    if rel[i] > SUM_NOTE:
        out.add("photoatomic_sum", WARN if rel[i] > SUM_WARN else NOTE,
                f"MT{mt} is {stated.y[i]:.6g} b at {fmt_e(g[i])}, its {len(present)} parts add up to {total[i]:.6g} b "
                f"({rel[i]:.2g} relative)", mf=23, mt=mt, at=float(g[i]), stated=float(stated.y[i]),
                summed=float(total[i]), relative=float(rel[i]))


def check_photoatomic(tape: Tape, out: Out) -> None:
    za = tape.za
    z = za // 1000 if za else None
    tables = {}
    for mf in (23, 27):
        for mt, sec in sorted(tape.sections(mf).items()):
            tab = getattr(sec, "table", None)
            if tab is None:
                continue
            t = Table.of(tab.x, tab.y, tab.interp)
            if not check_table(t, out, mf=mf, mt=mt, what=f"MT{mt}", x_name="E" if mf == 23 else "x"):
                continue
            if mf == 23:
                tables[mt] = t
                if t.y.size and t.y.min() < 0:
                    out.add("photoatomic_negative", DEFECT,
                            f"MT{mt}: negative cross section {t.y.min():.3g} b", mf=23, mt=mt, value=float(t.y.min()))
            elif z and mt in (502, 504) and t.y.size:
                if mt == 502:
                    v, what = float(t.y[0]), f"the coherent form factor at x = {t.x[0]:g}"
                else:
                    v, what = float(t.y[-1]), f"the incoherent scattering function at x = {t.x[-1]:g}"
                d = abs(v - z) / z
                if d > FF_NOTE:
                    out.add("form_factor_limit", WARN if d > FF_WARN else NOTE,
                            f"MT{mt}: {what} is {v:.6g}, the limit is Z = {z}", mf=27, mt=mt, value=v, z=z)
                if mt == 504 and t.y[0] != 0 and t.x[0] == 0:
                    out.add("form_factor_limit", WARN, f"MT504: S(0) = {t.y[0]:.6g}, not 0", mf=27, mt=mt,
                            value=float(t.y[0]))
    for mt, parts in _SUMS.items():
        if mt in tables:
            _stated_vs_parts(out, tables, mt, parts)
    _relaxation(tape, out, z)


def _relaxation(tape: Tape, out: Out, z) -> None:
    sec = tape.section(28, 533)
    if sec is None:
        return
    shells = list(getattr(sec, "subshells", []) or [])
    if not shells:
        return
    electrons = float(sum(s.electrons for s in shells))
    if z and abs(electrons - z) > 1e-3:
        out.add("relaxation_electrons", WARN, f"the subshells hold {electrons:g} electrons, the atom has Z = {z}",
                mf=28, mt=533, electrons=electrons, z=z)
    for s in shells:
        if not s.binding_energy > 0:
            out.add("relaxation_binding", DEFECT, f"subshell {s.name or s.subi}: binding energy {s.binding_energy:g}",
                    mf=28, mt=533, subshell=s.subi)
            continue
        tr = s.transitions
        if not tr:
            continue
        p = float(sum(t.probability for t in tr))
        if abs(p - 1) > RELAX_NOTE:
            out.add("relaxation_probability", WARN if abs(p - 1) > RELAX_WARN else NOTE,
                    f"subshell {s.name or s.subi}: its {len(tr)} transition probabilities add up to {p:.6g}",
                    mf=28, mt=533, subshell=s.subi, total=p)
        if any(t.probability < 0 for t in tr):
            out.add("relaxation_probability", DEFECT, f"subshell {s.name or s.subi}: a negative transition "
                    "probability", mf=28, mt=533, subshell=s.subi)
        hot = [t for t in tr if t.energy > s.binding_energy * (1 + RELAX_ENERGY_NOTE) + 1.0]
        if hot:
            t = max(hot, key=lambda t: t.energy)
            excess = t.energy / s.binding_energy - 1
            out.add("relaxation_energy", WARN if excess > RELAX_ENERGY_WARN else NOTE,
                    f"subshell {s.name or s.subi}: a transition releases {fmt_e(t.energy)}, more than its binding "
                    f"energy {fmt_e(s.binding_energy)}", mf=28, mt=533, subshell=s.subi, energy=t.energy,
                    binding=s.binding_energy)


def descriptions(D, _g, _pct):
    return {
        "photoatomic_negative": D(
            "Negative photo-atomic cross section", (23,), {DEFECT: "always"},
            "A photo-atomic cross section is not negative."),
        "photoatomic_sum": D(
            "Photo-atomic total differs from its parts", (23,),
            {WARN: f"by more than {_pct(SUM_WARN)}", NOTE: f"by more than {_g(SUM_NOTE)}"},
            "MT501 = 502 + 504 + 516 + 522, MT516 = 515 + 517, MT522 = the subshells 534-599, at the energies "
            "the total is stated at (FIZCON's TSTTOT)."),
        "form_factor_limit": D(
            "Form factor off its limit", (27,),
            {WARN: f"off Z by more than {_pct(FF_WARN)}, or S(0) != 0", NOTE: f"by more than {_g(FF_NOTE)}"},
            "The coherent form factor tends to Z at zero momentum transfer, the incoherent scattering function "
            "to Z at large momentum transfer and to 0 at zero. No other checker looks."),
        "relaxation_electrons": D(
            "Subshell occupancies do not add up to Z", (28,), {WARN: "always"},
            "The subshells of MF28 describe the neutral atom: their electrons add up to Z."),
        "relaxation_binding": D(
            "Non-positive binding energy", (28,), {DEFECT: "always"},
            "A subshell's binding energy is positive."),
        "relaxation_probability": D(
            "Relaxation probabilities", (28,),
            {DEFECT: "a negative one", WARN: f"their sum off 1 by more than {_g(RELAX_WARN)}",
             NOTE: f"by more than {_g(RELAX_NOTE)}"},
            "The radiative and non-radiative transitions that fill a vacancy add up to probability 1."),
        "relaxation_energy": D(
            "Relaxation transition above the binding energy", (28,),
            {WARN: f"by more than {_pct(RELAX_ENERGY_WARN)}", NOTE: f"by more than {_g(RELAX_ENERGY_NOTE)}"},
            "Filling a vacancy releases at most the vacancy's binding energy. EADL computes transition "
            "energies and binding energies from different sources, so they disagree by about 1e-3."),
    }


def method(_g, _pct):
    return [
        ("Photo-atomic data",
         "MF23 cross sections and their sums, MF27 form-factor limits, MF28 relaxation.",
         [(DEFECT, "a negative cross section, binding energy or transition probability."),
          (WARN, f"a transition releasing more than {_pct(RELAX_ENERGY_WARN)} beyond its binding energy; "
                 f"a total off its parts by more than {_pct(SUM_WARN)}; a form factor off Z by more than "
                 f"{_pct(FF_WARN)}; occupancies that do not add up to Z; relaxation probabilities off 1 by "
                 f"more than {_g(RELAX_WARN)}."),
          (NOTE, "smaller disagreements.")]),
    ]

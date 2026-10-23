"""MF7: thermal scattering laws.

**Shapes.** Every temperature of a table holds as many values as its grid
(JEFF-4.0's Zr in Zircaloy-4 has 1738 values on a 1740-point grid at 351 K),
temperatures increase, the beta grid increases, and each alpha table is a
TAB1 like any other -- so a logarithmic law over the stored ``ln S`` of an
LLN = 1 table that has non-positive values is caught there (JEFF-4.0 para-D).

**Values.** S(alpha, beta) is not negative where it is stored as itself
(LLN = 0). Coherent elastic: the Bragg edges increase and the cumulative
structure factor S(E) never decreases. Incoherent elastic: the bound cross
section and the Debye-Waller integral are positive, the latter growing with
temperature. The effective temperature of the short-collision time
approximation is not below the physical one; free cross sections and mass
ratios of the scatterers are positive.

Not checked: detailed balance and the sum rules of S(alpha, beta). Both need
the whole beta range, which tables truncate, and the sign conventions of
LASYM; no threshold could be set without false positives.
"""
from __future__ import annotations

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Table, Tape, fmt_e
from .tables import check_table

#: T_eff below T by more than this fraction.
TEFF_SLACK = 1e-3


def _temperature_table(out: Out, table, mt: int, what: str, *, positive: bool, cumulative: bool) -> None:
    x = np.asarray(table.x, dtype=float)
    temps = list(table.temperatures) or [table.t0]
    if len(temps) > 1 and np.any(np.diff(temps) <= 0):
        out.add("tsl_temperatures", DEFECT, f"{what}: temperatures do not increase ({temps})",
                mf=7, mt=mt, temperatures=temps)
    for t_index, row in enumerate(table.values):
        if len(row) != x.size:
            out.add("tsl_shape", DEFECT,
                    f"{what}: {len(row)} values at T = {temps[t_index] if t_index < len(temps) else '?'} K for a "
                    f"grid of {x.size}", mf=7, mt=mt, t=t_index, values=len(row), grid=int(x.size))
            return
    if not table.values:
        return
    t0 = Table.of(x, table.values[0], table.interp)
    if not check_table(t0, out, mf=7, mt=mt, what=what, x_name="x"):
        return
    rows = np.asarray(table.values, dtype=float)
    if positive and rows.size and rows.min() < 0:
        t_index = int(np.argwhere(rows < 0)[0][0])
        out.add("tsl_negative", DEFECT, f"{what}: a negative value ({rows.min():.3g}) at T = {temps[t_index]} K",
                mf=7, mt=mt, t=t_index, value=float(rows.min()))
    if cumulative and rows.shape[1] > 1:
        dec = np.diff(rows, axis=1) < -1e-12 * np.abs(rows[:, 1:]).max(initial=1.0)
        if dec.any():
            t_index = int(np.argwhere(dec)[0][0])
            out.add("tsl_bragg", DEFECT,
                    f"{what}: the cumulative structure factor decreases at T = {temps[t_index]} K", mf=7, mt=mt,
                    t=t_index)


def check_thermal(tape: Tape, out: Out) -> None:
    secs = tape.sections(7)
    if not secs:
        return
    el = secs.get(2)
    if el is not None:
        coh, inc = getattr(el, "coherent", None), getattr(el, "incoherent", None)
        if coh is not None:
            _temperature_table(out, coh.table, 2, "coherent elastic", positive=True, cumulative=True)
        if inc is not None:
            if not inc.sb > 0:
                out.add("tsl_parameter", DEFECT, f"incoherent elastic: bound cross section SB = {inc.sb:g}",
                        mf=7, mt=2, parameter="SB", value=float(inc.sb))
            w = np.asarray(inc.w, dtype=float)
            t = np.asarray(inc.temperatures, dtype=float)
            if w.size and w.min() <= 0:
                out.add("tsl_parameter", DEFECT, f"incoherent elastic: a Debye-Waller integral W = {w.min():g}",
                        mf=7, mt=2, parameter="W", value=float(w.min()))
            elif w.size > 1 and np.any(np.diff(w) < 0) and np.all(np.diff(t) > 0):
                out.add("tsl_parameter", WARN,
                        "incoherent elastic: the Debye-Waller integral decreases with temperature", mf=7, mt=2,
                        parameter="W")
    inel = secs.get(4)
    if inel is None:
        return
    lln = getattr(inel, "_lln", 0) or 0
    betas = np.array([b.beta for b in inel.blocks], dtype=float)
    if betas.size > 1 and np.any(np.diff(betas) <= 0):
        out.add("tsl_shape", DEFECT, "S(alpha, beta): the beta grid does not increase", mf=7, mt=4)
    bad_shape = None
    for i, blk in enumerate(inel.blocks):
        rows = blk.table.values
        n = len(blk.table.x)
        if any(len(r) != n for r in rows):
            bad_shape = (i, blk.beta)
            break
    if bad_shape is not None:
        out.add("tsl_shape", DEFECT, f"S(alpha, beta) at beta = {bad_shape[1]:g}: a temperature with a number of "
                "values other than the alpha grid's", mf=7, mt=4, beta=bad_shape[1])
    else:
        scratch = Out(out.mat)
        for i, blk in enumerate(inel.blocks):
            t0 = Table.of(blk.table.x, blk.table.values[0] if blk.table.values else [], blk.table.interp)
            check_table(t0, scratch, mf=7, mt=4, what=f"S(alpha, beta) at beta = {blk.beta:g}", x_name="x")
            if not lln and blk.table.values:
                v = np.asarray(blk.table.values, dtype=float)
                if v.min() < 0:
                    scratch.add("tsl_negative", DEFECT,
                                f"S(alpha, beta) at beta = {blk.beta:g}: a negative value ({v.min():.3g})",
                                mf=7, mt=4, value=float(v.min()))
        groups = {}
        for f in scratch.findings:
            groups.setdefault((f.check, f.level), []).append(f)
        for (check, level), fs in groups.items():
            first = fs[0]
            extra = f" (and at {len(fs) - 1} other beta)" if len(fs) > 1 else ""
            out.add(check, level, first.summary + extra, mf=7, mt=4, betas=len(fs))
    for k, teff in enumerate(getattr(inel, "teff", []) or []):
        te = np.asarray(teff.teff, dtype=float)
        tt = np.asarray(teff.temperatures, dtype=float)
        if te.size and tt.size == te.size and np.any(te < tt * (1 - TEFF_SLACK)):
            i = int(np.flatnonzero(te < tt * (1 - TEFF_SLACK))[0])
            out.add("tsl_effective_temperature", WARN,
                    f"scatterer {k}: effective temperature {te[i]:g} K below the physical {tt[i]:g} K",
                    mf=7, mt=4, k=k, teff=float(te[i]), temperature=float(tt[i]))


def descriptions(D, _g, _pct):
    return {
        "tsl_shape": D(
            "Thermal table of the wrong shape", (7,), {DEFECT: "always"},
            "Every temperature of a File 7 table has as many values as its grid, and the beta grid "
            "increases. Found in JEFF-4.0 (Zr in Zircaloy-4, 1738 values for 1740 points at 351 K)."),
        "tsl_temperatures": D(
            "Temperatures do not increase", (7,), {DEFECT: "always"},
            "The temperatures of a File 7 table increase."),
        "tsl_negative": D(
            "Negative scattering law", (7,), {DEFECT: "always"},
            "S(alpha, beta) stored as itself (LLN = 0), and the coherent structure factor, are not negative."),
        "tsl_bragg": D(
            "Coherent structure factor decreases", (7,), {DEFECT: "always"},
            "S(E) of coherent elastic scattering is cumulative over the Bragg edges: it never decreases."),
        "tsl_parameter": D(
            "Thermal parameter out of range", (7,),
            {DEFECT: "a bound cross section or Debye-Waller integral that is not positive",
             WARN: "a Debye-Waller integral that decreases with temperature"},
            "Incoherent elastic scattering's bound cross section and Debye-Waller integral are positive, and "
            "the integral grows with temperature."),
        "tsl_effective_temperature": D(
            "Effective temperature below the physical one", (7,), {WARN: "always"},
            "The effective temperature of the short-collision time approximation includes the zero-point "
            "motion: it is not below the temperature of the material."),
    }


def method(_g, _pct):
    return [
        ("Thermal scattering",
         "MF7 tables: shapes, temperatures, S(alpha, beta) and the elastic parameters. Detailed balance and "
         "the sum rules are not checked: tables truncate beta and no threshold could be set.",
         [(DEFECT, "values that do not fit their grid; a negative S(alpha, beta) or structure factor; a "
                   "logarithmic law over the stored ln S where it is not positive; a non-positive bound cross "
                   "section or Debye-Waller integral."),
          (WARN, "an effective temperature below the physical one; a Debye-Waller integral that falls with "
                 "temperature.")]),
    ]

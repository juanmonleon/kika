"""The checks that need the cross sections reconstructed (``deep=True``).

The static checks read MF3 as written; inside the resonance ranges it is a
background, and whether the cross sections a code will use are positive, add
up and behave as 1/v at thermal energies is only known once MF2's resonances
are added. These checks run kika's 0 K reconstruction (the one the processing
runs, :func:`kika.endf.reconstruct_endf`) and read its tables and report:

* a negative reconstructed cross section (MLBW interference, a background
  more negative than the resonances make up for) is a defect;
* a stated total against the sum of its reconstructed parts, graded as the
  reconstruction report grades it (a warning where the gap is above 1 % and
  1 mb, a note below);
* capture as 1/v from 1e-5 to 0.1 eV: ``sigma sqrt(E)`` constant. No checker
  surveyed tests it. A resonance near thermal bends it legitimately, so a
  departure is a warning only when the nearest resonance is above
  ``ONE_OVER_V_CLEAR`` eV, a note otherwise.

They cost a reconstruction: seconds for most tapes, a minute for the heaviest.
"""
from __future__ import annotations

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Tape, fmt_b, fmt_e

#: sigma sqrt(E) from 1e-5 to 0.1 eV within this of its value at 1e-5 eV.
ONE_OVER_V_NOTE = 0.02
ONE_OVER_V_WARN = 0.10
#: A departure from 1/v is a warning only when no resonance lies below this (eV).
ONE_OVER_V_CLEAR = 10.0
#: The reconstruction report's "material" sum gap.
SUM_MATERIAL_REL, SUM_MATERIAL_B = 1e-2, 1e-3


def _nearest_resonance(tape: Tape) -> float:
    sec = tape.section(2, 151)
    best = np.inf
    for iso in getattr(sec, "isotopes", None) or []:
        for rng in iso.energy_ranges:
            p = rng.parameters
            for lv in getattr(p, "l_values", []) or []:
                for res in getattr(lv, "resonances", []) or []:
                    best = min(best, abs(float(res.energy)))
            for sg in getattr(p, "spin_groups", []) or []:
                for res in sg.resonances:
                    best = min(best, abs(float(res.er)))
    return float(best)


def check_deep(tape: Tape, out: Out) -> None:
    if tape.nsub not in (None, 10) or not tape.source or not tape.has(3):
        return
    from kika.endf.reconstruction import reconstruct_endf

    try:
        rec = reconstruct_endf(tape.source, workers=1)
    except Exception as error:  # noqa: BLE001 - the static checks stand on their own
        out.add("reconstruction_failed", NOTE,
                f"the cross sections could not be reconstructed ({type(error).__name__}: {error}); the checks "
                "that need them were not run", mf=2, mt=151, error=str(error))
        return
    d = rec.report_document().to_dict()
    for r in d.get("reactions", []):
        neg = r.get("negatives")
        if neg:
            out.add("xs_reconstructed_negative", DEFECT,
                    f"MT{r.get('mt')}: the reconstructed cross section is negative at {neg['count']} point"
                    f"{'s' if neg['count'] != 1 else ''}, down to {fmt_b(neg['min_b'])} at {fmt_e(neg['at_eV'])}",
                    mf=3, mt=r.get("mt"), count=neg["count"], value=neg["min_b"], at=neg["at_eV"])
    for x in d.get("sums", {}).get("source_discrepancies", []):
        if x.get("max_ratio", 0) <= 1:
            continue
        stated, parts = x.get("stated_b") or 0.0, x.get("components_b") or 0.0
        gap = abs(stated - parts)
        material = gap > SUM_MATERIAL_REL * max(abs(stated), abs(parts)) and gap > SUM_MATERIAL_B
        out.add("xs_reconstructed_sum", WARN if material else NOTE,
                f"{x.get('name')}: stated {fmt_b(stated)}, its reconstructed parts add up to {fmt_b(parts)} at "
                f"{fmt_e(x.get('at_eV') or 0.0)}", mf=3, mt=x.get("mt"), stated=stated, summed=parts,
                at=x.get("at_eV"))
    tables = rec.tables_by_mt()
    cap = tables.get(102)
    if cap is None or not cap.energies.size or cap.energies[0] > 1e-5 * (1 + 1e-9):
        return
    grid = np.geomspace(1e-5, 0.1, 61)
    sigma = np.interp(grid, cap.energies, cap.values)
    f = sigma * np.sqrt(grid)
    if f[0] <= 0:
        return
    dev = np.abs(f / f[0] - 1.0)
    i = int(np.argmax(dev))
    if dev[i] <= ONE_OVER_V_NOTE:
        return
    nearest = _nearest_resonance(tape)
    level = WARN if dev[i] > ONE_OVER_V_WARN and nearest > ONE_OVER_V_CLEAR else NOTE
    out.add("capture_one_over_v", level,
            f"MT102: sigma sqrt(E) departs from 1/v by {dev[i]:.2g} at {fmt_e(grid[i])} (nearest resonance at "
            f"{fmt_e(nearest)})", mf=3, mt=102, deviation=float(dev[i]), at=float(grid[i]),
            nearest_resonance=nearest)


def descriptions(D, _g, _pct):
    return {
        "reconstruction_failed": D(
            "Cross sections could not be reconstructed", (2, 3), {NOTE: "always"},
            "deep=True asked for the reconstructed cross sections and kika could not build them; the "
            "reconstruction's reason is given. The static checks are unaffected."),
        "xs_reconstructed_negative": D(
            "Negative reconstructed cross section", (2, 3), {DEFECT: "always"},
            "The cross section MF2 and MF3 give together is negative somewhere (FUDGE: severe; NJOY floors "
            "it). Codes use it as it comes out."),
        "xs_reconstructed_sum": D(
            "A total differs from its reconstructed parts", (2, 3),
            {WARN: f"by more than {_pct(SUM_MATERIAL_REL)} and {_g(SUM_MATERIAL_B * 1e3)} mb",
             NOTE: "by more than the reconstruction tolerance"},
            "The reconstruction report's source sums: a stated total against the sum of its parts once the "
            "resonances are in."),
        "capture_one_over_v": D(
            "Capture is not 1/v at thermal energies", (2, 3),
            {WARN: f"sigma sqrt(E) off constant by more than {_pct(ONE_OVER_V_WARN)}, no resonance below "
                   f"{_g(ONE_OVER_V_CLEAR)} eV",
             NOTE: f"by more than {_pct(ONE_OVER_V_NOTE)}"},
            "Far from any resonance, capture goes as 1/v between 1e-5 and 0.1 eV. No checker surveyed tests "
            "it; a resonance near thermal (Pu-239, Cd-113, Gd-157 ...) bends it legitimately."),
    }


def method(_g, _pct):
    return [
        ("Reconstructed cross sections (deep=True)",
         "Run only when asked: kika's 0 K reconstruction of MF2 + MF3, and three checks on what codes will use.",
         [(DEFECT, "a negative reconstructed cross section."),
          (WARN, f"a total off its reconstructed parts by {_pct(SUM_MATERIAL_REL)} and "
                 f"{_g(SUM_MATERIAL_B * 1e3)} mb; capture off 1/v by more than {_pct(ONE_OVER_V_WARN)} with no "
                 f"resonance below {_g(ONE_OVER_V_CLEAR)} eV."),
          (NOTE, "smaller departures.")]),
    ]

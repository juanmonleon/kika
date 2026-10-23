"""The reconstruction report: what was run, what was checked, what to know.

``EndfReconstruction.report_document()`` builds a :class:`ReconstructionReport`
from what the run already knows; nothing is recomputed except cheap scans of
the output tables. It renders three ways from one dictionary:

* ``to_dict()``  JSON-safe, with stable keys (``schema`` names the version);
* ``to_text()``  fixed-width ASCII for a terminal or a ``.txt`` file;
* ``to_html()``  a self-contained page (inline CSS, no script, light and dark).

The page reads like the covariance check: a verdict, the list of checks with
the result of each, the findings (only when there are any), what was done, one
row per reaction, the run's metadata, and at the bottom what each check
compares and what its levels mean. Engine internals (segments, iterations,
the engine's own notes) are folded away under the run details.

Errors are quoted against the tolerance: at each energy the table may differ
from the reference by ``rtol*|sigma| + atol``, and 100 % uses all of it.
Tabulation aims at half, leaving the other half for the rounding of the ENDF
file (``ReconstructionOptions``).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import html as _html
import json
import re
from pathlib import Path

import numpy as np

__all__ = ["ReconstructionReport", "SCHEMA", "CHECKS"]

#: Bump when a key is renamed or removed; adding keys keeps the version.
SCHEMA = "kika-reconstruction-report/1"

NLIB_NAMES = {0: "ENDF/B", 1: "ENDF/A", 2: "JEFF", 3: "EFF", 4: "ENDF/B (HE)", 5: "CENDL",
              6: "JENDL", 21: "SG-23", 31: "INDL/V", 32: "INDL/A", 33: "FENDL", 34: "IRDF",
              35: "BROND (IAEA)", 36: "INGDB-90", 37: "FENDL/A", 41: "BROND"}
LRF_NAMES = {1: "SLBW", 2: "MLBW", 3: "Reich-Moore", 4: "Adler-Adler", 7: "R-Matrix Limited"}
KRM_NAMES = {1: "SLBW", 2: "MLBW", 3: "Reich-Moore", 4: "full R-matrix"}


# --------------------------------------------------------------------------- build

def file_sha256(path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def evaluation_summary(tape, path) -> dict:
    """MF1/451 identification and one dict per MF2 energy range, read from the
    tape before it is decoded (the model does not keep the ENDF flags)."""
    out = dict(file=Path(path).name, path=str(Path(path).resolve()), sha256=file_sha256(path))
    head = tape.mf[1].mt[451] if 1 in tape.mf and 451 in tape.mf[1].mt else None
    if head is not None:
        g = lambda name: getattr(head, "_" + name, None)
        nlib = g("nlib")
        out.update(za=_int(g("za")), mat=_int(g("mat")), zsymam=(g("zsymam") or "").strip() or None,
                   awr=_num(g("awr")), nlib=_int(nlib), library=NLIB_NAMES.get(nlib),
                   nver=_int(g("nver")), lrel=_int(g("lrel")), nmod=_int(g("nmod")),
                   lrp=_int(g("lrp")), lfi=_int(g("lfi")), temperature_K=_num(g("temp")),
                   emax_eV=_num(g("emax")), evaluation_date=(g("edate") or "").strip() or None,
                   laboratory=(g("alab") or "").strip() or None,
                   authors=(g("auth") or "").strip() or None)
    ranges = []
    if 2 in tape.mf and 151 in tape.mf[2].mt:
        for isotope in tape.mf[2].mt[151].isotopes:
            for rng in isotope.energy_ranges:
                ranges.append(_range_row(isotope, rng))
    out["resonance_ranges"] = ranges
    return out


def _range_row(isotope, rng) -> dict:
    p = rng.parameters
    row = dict(el_eV=_num(rng.el), eh_eV=_num(rng.eh), lru=_int(rng.lru), lrf=_int(rng.lrf),
               nro=_int(rng.nro), naps=_int(rng.naps), ap_fm=_fm(getattr(p, "ap", None)),
               ap_energy_points=len(rng.ap_e.energies) if getattr(rng, "ap_e", None) else 0,
               lfw=_int(getattr(isotope, "lfw", None)), za=_int(getattr(isotope, "za", None)),
               abundance=_num(getattr(isotope, "abn", None)))
    # The radius a reader should see is the one used: an energy-dependent AP(E)
    # (NRO = 1) makes the range's AP a placeholder, and a Reich-Moore block's
    # APL replaces AP for its l.
    ap_e = getattr(rng, "ap_e", None)
    values = list(getattr(ap_e, "ap_values", None) or []) if ap_e is not None else []
    if values:
        row["ap_e_fm"] = [10 * float(min(values)), 10 * float(max(values))]
    if rng.lru == 1 and rng.lrf == 3:
        apl = sorted({10 * float(b.apl_or_qx) for b in getattr(p, "l_values", None) or []
                      if getattr(b, "apl_or_qx", None) and float(b.apl_or_qx) > 0})
        if apl:
            row["apl_fm"] = apl
    kind = type(p).__name__
    if rng.lru == 0:
        row.update(formalism="scattering radius only")
    elif rng.lru == 1 and rng.lrf == 7:
        groups = list(getattr(p, "spin_groups", []) or [])
        pairs = list(getattr(p, "particle_pairs", []) or [])
        krm = _int(getattr(p, "krm", None))
        row.update(formalism=f"R-Matrix Limited ({KRM_NAMES.get(krm, f'KRM={krm}')})", krm=krm,
                   spin_groups=len(groups),
                   channels=sum(len(g.channels) for g in groups),
                   resonances=sum(len(g.resonances) for g in groups),
                   particle_pairs=[_int(pp.mt) for pp in pairs],
                   l_values=len({c.l for g in groups for c in g.channels}))
    elif rng.lru == 1:
        blocks = list(getattr(p, "l_values", []) or [])
        row.update(formalism=LRF_NAMES.get(rng.lrf, f"LRF={rng.lrf}"), l_values=len(blocks),
                   spin_groups=len({(b.l, r.spin) for b in blocks for r in b.resonances}),
                   resonances=sum(len(b.resonances) for b in blocks))
    elif rng.lru == 2:
        case = {"UnresolvedCaseA": "A", "UnresolvedCaseB": "B", "UnresolvedCaseC": "C"}.get(kind)
        blocks = list(getattr(p, "l_values", []) or [])
        row.update(formalism="unresolved" + (f" (case {case})" if case else ""), urr_case=case,
                   lssf=_int(getattr(p, "lssf", None)), l_values=len(blocks),
                   spin_groups=sum(len(b.j_states) for b in blocks),
                   energy_points=_int(getattr(p, "ne", None)) if case == "B" else
                   (max((len(j.energy_points) for b in blocks for j in b.j_states), default=0) if case == "C" else None))
    return row


def build(rec) -> dict:
    """The report dictionary of an :class:`~kika.endf.reconstruction.EndfReconstruction`."""
    from kika import __version__
    from kika._constants import MT_TO_REACTION
    from kika.processing.resonances import native_available

    result, report = rec.result, rec.report
    options = result.options
    evaluation = dict(getattr(rec, "evaluation", None) or {})
    ranges = evaluation.pop("resonance_ranges", [])
    key_mt = {key: mt for mt, key in result._mt_keys.items()}
    name = lambda key: (f"MT{key_mt[key]}" if key in key_mt else str(getattr(key, "label", key)))
    reaction = lambda mt: reaction_name(mt, MT_TO_REACTION)

    context = rec.context
    material = dict(
        za=evaluation.get("za"), mat=evaluation.get("mat"), zsymam=evaluation.get("zsymam"),
        target=report.get("target"), projectile=report.get("projectile"),
        awr_mf1=evaluation.get("awr"),
        awr_used=_num(getattr(context, "atomic_weight_ratio", None)),
        target_spin=_num(getattr(context, "target_spin", None)),
        context_source=rec.context_source)
    identification = {k: evaluation.get(k) for k in
                      ("file", "path", "sha256", "library", "nlib", "nver", "lrel", "nmod", "lrp", "lfi",
                       "temperature_K", "emax_eV", "evaluation_date", "laboratory", "authors")}

    # Resonance ranges: the tape's flags joined with how the engine prepared them.
    prepared = list(getattr(result._prepared, "regions", ()) or ())
    for row in ranges:
        match = next((r for r in prepared if _close(r.low, row["el_eV"]) and _close(r.high, row["eh_eV"])), None)
        row["engine_approximation"] = getattr(match, "approximation", None)
        if row["lru"] == 1:
            row["treatment"] = "computed point by point at 0 K from the resonance parameters"
        elif row["lru"] == 2:
            data = getattr(match, "unresolved", None)
            if row.get("lssf") == 1 or getattr(data, "self_shielding_only", False):
                row["treatment"] = ("the evaluation's own smooth cross sections (MF3) are used as they are; "
                                    "its average parameters only describe self-shielding, which a 0 K "
                                    "reconstruction does not compute")
            else:
                law = getattr(data, "law", None)
                row["treatment"] = ("average (infinitely dilute) cross sections computed from the average "
                                    "parameters, with width-fluctuation corrections, "
                                    + ("on the evaluation's energy grid and interpolated between its points"
                                       if law else "at every energy of the output grid"))
                row["urr_interpolation"] = law
        else:
            row["treatment"] = "potential scattering from the scattering radius, added to MF3"

    segments = []
    for check in report.get("regions", ()):
        low, high = (float(v) for v in check["domain"])
        owner = next((r for r in prepared if r.low <= low and r.high >= high), None)
        segments.append(dict(
            low_eV=low, high_eV=high,
            kind=("resolved" if owner is not None and owner.unresolved is None else
                  "unresolved" if owner is not None else "background only"),
            points=int(check["points"]), iterations=int(check.get("iterations", 0)),
            evaluations=int(check.get("evaluations", 0)),
            refinement_max=_maxval(check.get("refinement_maxima")),
            verification_max=_maxval(check.get("verification_maxima"))))

    # Per reaction: the tables the user gets, plus the engine's own diagnostics.
    tables = rec.tables_by_mt()
    per_key_verification = {}
    per_key_worst = {}
    integrals = {}
    for check in report.get("regions", ()):
        for key, value in (check.get("verification_maxima") or {}).items():
            if float(value) >= per_key_verification.get(key, 0.):
                worst = (check.get("verification_worst") or {}).get(key)
                if worst is not None and float(value) > 0:
                    per_key_worst[key] = worst
            per_key_verification[key] = max(per_key_verification.get(key, 0.), float(value))
        for key, weights in (check.get("integrals") or {}).items():
            acc = integrals.setdefault(key, [0., 0.])
            dE = (weights or {}).get("dE") or {}
            acc[0] += float(dE.get("reference_estimate", 0.))
            acc[1] += float(dE.get("absolute_difference_estimate", 0.))

    pub = getattr(rec, "publication", None) or None
    pub_maxima, pub_worst_at = ((pub.get("maxima") or {}), (pub.get("worst") or {})) if pub else ({}, {})

    reactions = []
    for mt in sorted(tables):
        t = tables[mt]
        x, y = np.asarray(t.energies), np.asarray(t.values)
        dx = np.diff(x)
        positive = np.flatnonzero(dx > 0)
        relative = dx[positive] / x[positive]
        negative = np.flatnonzero(y < 0)
        nonzero = np.flatnonzero(y != 0)
        repeated = np.flatnonzero(dx == 0)
        key = result._mt_keys[mt]
        ref, diff = integrals.get(key, (0., 0.))
        reactions.append(dict(
            mt=mt, reaction=reaction(mt), points=int(len(x)),
            e_min_eV=float(x[0]), e_max_eV=float(x[-1]),
            threshold_eV=float(x[nonzero[0]]) if len(nonzero) else None,
            jumps=int(len(repeated)), jumps_at_eV=[float(x[i]) for i in repeated[:20]],
            jumps_b=[float(abs(y[i + 1] - y[i])) for i in repeated[:20]],
            decreasing=int(np.count_nonzero(dx < 0)),
            min_relative_spacing=float(relative.min()) if len(positive) else None,
            min_relative_spacing_at_eV=float(x[positive[np.argmin(relative)]]) if len(positive) else None,
            negatives=(dict(count=int(len(negative)), min_b=float(y[negative].min()),
                            at_eV=float(x[negative[np.argmin(y[negative])]])) if len(negative) else None),
            derived_sum=key in result._graph,
            tabulation_verification_max=per_key_verification.get(key),
            publication_verification_max=_num(pub_maxima.get(key)) if pub else None,
            integral_relative_difference=(diff / ref if ref > 0 else None),
            qm_eV=_num(t.qm), qi_eV=_num(t.qi), lr=t.lr))

    graph = []
    for key in result._order:
        parts = sorted((key_mt.get(p, 10**9), name(p)) for p in result._graph[key])
        mts = [mt for mt, _ in parts if mt < 10**9]
        graph.append(dict(mt=key_mt.get(key), name=name(key), parts=[n for _, n in parts],
                          parts_text=_ranges_text(mts) if len(mts) == len(parts) else
                          " + ".join(n for _, n in parts)))
    discrepancies = []
    worst_source = report.get("source_sum_discrepancies") or {}
    for key, ratio in (report.get("source_sum_error_ratios") or {}).items():
        w = worst_source.get(key) or {}
        discrepancies.append(dict(mt=key_mt.get(key), name=name(key), max_ratio=float(ratio),
                                  at_eV=_num(w.get("energy_eV")), side=w.get("side"),
                                  stated_b=_num(w.get("stated_b")), components_b=_num(w.get("components_b"))))

    publication = None
    if pub:
        publication = dict(
            path=str(pub.get("path")), file=Path(str(pub.get("path"))).name, mat=pub.get("mat"),
            reloaded_and_verified=True,
            rows=[dict(mt=key_mt.get(key), name=name(key), max_ratio=float(value),
                       at_eV=_num(pub_worst_at.get(key, (None,))[0]),
                       reference_b=_num(pub_worst_at.get(key, (None, None))[1]),
                       serialized_b=_num(pub_worst_at.get(key, (None, None, None))[2]))
                  for key, value in sorted(pub_maxima.items(), key=lambda kv: key_mt.get(kv[0], 10**9))])

    conversion = report.get("source_conversion_report")
    notes = dict(
        warnings=list(getattr(conversion, "warnings", []) or []),
        losses=list(getattr(conversion, "losses", []) or []),
        approximations=list(getattr(conversion, "approximations", []) or []),
        unsupported=list(getattr(conversion, "unsupported", []) or []),
        scope_exclusions=[dict(kind=e.get("kind"), message=e.get("message"), scopes=list(e.get("scopes", ())))
                          for e in (getattr(conversion, "scopeExclusions", []) or [])],
        clean_for_cross_sections=bool(conversion.isCleanFor("cross-sections")) if conversion is not None else None,
        preparation=list(report.get("preparation_notes") or ()))

    order = lambda key: key_mt.get(key, 10**9)
    tabulation_rows = [dict(mt=key_mt.get(k), name=name(k), reaction=reaction(key_mt.get(k)), max_ratio=float(v),
                            at_eV=_num(per_key_worst.get(k, (None,))[0]),
                            reference_b=_num(per_key_worst.get(k, (None, None))[1]),
                            tabulated_b=_num(per_key_worst.get(k, (None, None, None))[2]))
                       for k, v in sorted(per_key_verification.items(), key=lambda kv: order(kv[0])) if v > 0]

    def worst_of(rows):
        return max(rows, key=lambda r: r["max_ratio"]) if rows else None
    tab = worst_of(tabulation_rows)
    pub_worst = worst_of(publication["rows"]) if publication else None
    sums = worst_of(discrepancies)
    negatives = sum(r["negatives"]["count"] for r in reactions if r["negatives"])
    issues = len(notes["losses"]) + len(notes["approximations"]) + len(notes["unsupported"])
    passed = (tab is None or tab["max_ratio"] <= 1) and (pub_worst is None or pub_worst["max_ratio"] <= 1)
    summary = dict(
        verified=passed,
        worst_tabulation=tab, worst_publication=pub_worst, worst_source_sum=sums,
        source_sums_consistent=sums is None or sums["max_ratio"] <= 1,
        negative_values=negatives,
        reactions_with_negatives=[r["mt"] for r in reactions if r["negatives"]],
        conversion_issues=issues, clean_for_cross_sections=notes["clean_for_cross_sections"],
        reactions=len(reactions), stored_points=report.get("stored_points"),
        total_time_s=float(sum((rec.timings or {}).values())),
        pendf=publication["file"] if publication else None)

    low, high = report.get("domain", (None, None))
    data = _jsonable(dict(
        schema=SCHEMA,
        generated_utc=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        material=material, evaluation=identification,
        software=dict(kika_version=__version__, engine_version=rec.engine_version or report.get("engine"),
                      native_kernel=bool(getattr(rec, "native_kernel", native_available()))),
        options=dict(rtol=options.rtol, atol_b=options.atol, max_points=options.max_points,
                     max_iterations=options.max_iterations, tabulation_budget_fraction=0.5,
                     temperature_K=0.0, budget="atol + rtol*|sigma|"),
        domain_eV=[_num(low), _num(high)],
        points=dict(tabulated=report.get("points"), stored=report.get("stored_points")),
        resonance_ranges=ranges, segments=segments, reactions=reactions,
        sums=dict(policy=report.get("source_sum_policy"), graph=graph, source_discrepancies=discrepancies),
        verification=dict(empirical=bool(report.get("empirical_verification")),
                          global_error_bound=bool(report.get("global_error_bound")),
                          tabulation={name(k): float(v) for k, v in per_key_verification.items()},
                          tabulation_rows=tabulation_rows,
                          publication=publication),
        summary=summary,
        notes=notes,
        timings_s={k: float(v) for k, v in (rec.timings or {}).items()},
        fingerprints=dict(source_model_sha256=report.get("source_sha256"),
                          normalized_sha256=report.get("normalized_sha256"))))
    data["checks"] = evaluate_checks(data)
    data["summary"]["status"] = _overall(data["checks"])
    return data


# --------------------------------------------------------------------------- checks

PASS, NOTE, WARN, FAIL, SKIP = "passed", "note", "warning", "failed", "not run"
_RANK = {SKIP: -1, PASS: 0, NOTE: 1, WARN: 2, FAIL: 3}
_LABEL = {PASS: "Passed", NOTE: "Note", WARN: "Warning", FAIL: "Failed", SKIP: "Not run"}
_MARK = {PASS: "✓", NOTE: "i", WARN: "!", FAIL: "✕", SKIP: "–"}
_TEXT_MARK = {PASS: "[ OK ]", NOTE: "[NOTE]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[ -- ]"}
_CLASS = {PASS: "pass", NOTE: "note", WARN: "warn", FAIL: "fail", SKIP: "skip"}


class Check:
    """One entry of the legend: what a check compares and what each outcome means."""

    def __init__(self, id, title, what, levels):
        self.id, self.title, self.what, self.levels = id, title, what, levels


#: Every check the report can run, in the order the page lists them.
CHECKS = (
    Check("tabulation", "Tables reproduce the reference cross sections",
          "After the grid is refined, every reaction's table is interpolated linearly at four points "
          "inside each interval that were not used to build it, and compared with the reference: the "
          "cross section computed from the resonance parameters plus the evaluation's tabulated "
          "background (MF3). Outside the resonance ranges the reference is the evaluation's own table, "
          "copied unchanged. This is a test at sample points, not a mathematical bound between them.",
          {PASS: "Every difference is within the tolerance. The grid aims at half of it, so the other "
                 "half is left for rounding the values when a file is written.",
           FAIL: "Somewhere the table differs from the reference by more than the tolerance. The tables "
                 "should not be used; kika normally stops with an error rather than reach this."}),
    Check("pendf", "The written PENDF reproduces the reference",
          "The PENDF file is read back with kika's ENDF reader and every reaction is compared with the "
          "reference at every grid energy and at four points inside each interval. This covers what "
          "writing adds: the rounding of every number to the 11-character fields of ENDF-6.",
          {PASS: "The file, as written, is within the tolerance everywhere checked.",
           FAIL: "The file is not within the tolerance; it is not kept (the destination is only "
                 "replaced when this check passes).",
           SKIP: "No PENDF was written in this run; only the tables were produced."}),
    Check("source_sums", "The evaluation's totals equal the sum of their parts",
          "Where the evaluation stores a total (such as MT1 total or MT4 inelastic) together with the "
          "reactions it is made of, the stored total is compared with the sum of those reactions. kika "
          "always rebuilds totals from their parts, so this checks the evaluation, not the reconstruction.",
          {PASS: "Each stored total agrees with the sum of its parts within the tolerance.",
           NOTE: "A stored total disagrees with the sum of its parts, but by less than 1 % of the "
                 "stated total or less than 1 mb (typically a single point at a threshold). The output "
                 "uses the sum of the parts and is consistent; its total differs slightly from the one "
                 "printed in the evaluation there.",
           WARN: "A stored total disagrees with the sum of its parts by more than 1 % of it and more "
                 "than 1 mb. The evaluation is not self-consistent there: either the stored total or a "
                 "part is wrong, or the total counts a reaction the file does not state on its own. The "
                 "output uses the sum of the parts.",
           SKIP: "The evaluation stores no total alongside its parts."}),
    Check("negatives", "No negative cross sections",
          "Every value of every table is checked for sign.",
          {PASS: "All values are zero or positive.",
           WARN: "Some values are negative. This comes from the evaluation (some resonance formalisms, "
                 "such as SLBW or MLBW, or a negative background in MF3 can produce it). The values are "
                 "kept as computed, not clipped, so the table stays faithful to the evaluation; "
                 "transport and processing codes may set them to zero."}),
    Check("grid", "Tables are well formed",
          "Every table must have ascending energies. An energy may appear twice only where the cross "
          "section genuinely jumps, typically at the boundary between two resonance ranges; both values "
          "are then kept.",
          {PASS: "Energies ascend; any repeated energy is a real jump at a resonance-range boundary.",
           NOTE: "A jump larger than the absolute tolerance lies away from the resonance-range "
                 "boundaries. It comes from the evaluation's own tables (MF3) and is kept as stated.",
           FAIL: "Energies decrease somewhere: the table is malformed."}),
    Check("integrals", "Integrals over energy are preserved",
          "The integral of each table over energy is compared with an accurate estimate of the "
          "integral of the reference cross section.",
          {PASS: "Every relative difference is within the relative tolerance.",
           NOTE: "A larger relative difference, in a reaction small enough for the absolute part of "
                 "the tolerance to dominate. Each point is still within the tolerance."}),
    Check("parameters", "Resonance parameters used as stated",
          "Preparing the resonance parameters may need small, documented adjustments, such as a width "
          "that comes out slightly negative only because of rounding in the file.",
          {PASS: "No adjustment was needed.",
           NOTE: "Some parameters were adjusted within rounding, or a block uses its own declared mass "
                 "ratio; each case is listed. None changes the cross section beyond the file's precision."}),
    Check("conversion", "The evaluation was read completely",
          "Reading the ENDF file into kika's data model declares anything it cannot represent exactly. "
          "Only the resonance parameters (MF2) and the tabulated cross sections (MF3) matter here.",
          {PASS: "Nothing was lost or approximated in the data the reconstruction uses. Messages about "
                 "other parts of the file are listed under the run details.",
           NOTE: "A warning or an approximation in the data the reconstruction uses; each is listed.",
           WARN: "Something in the resonance parameters or the cross sections was lost or could not be "
                 "represented; the result may not reflect the evaluation there."}),
)
_CHECK = {c.id: c for c in CHECKS}

#: Engine notes that report an adjustment to the data, in words a reader can follow.
_ADJUSTMENTS = (
    (r"E=(\S+), L=(\d+): rounded negative GX=(\S+) to zero",
     lambda m: f"Resonance at {_energy(float(m[1]))}, l = {m[2]}: the competitive width "
               f"Γ − Γn − Γγ − Γf came out negative ({m[3]} eV) through rounding of the stored "
               "widths and was set to zero."),
    (r"E=(\S+), L=(\d+): GT=(\S+); using GN\+GG\+GF=(\S+) within rounding tolerance",
     lambda m: f"Resonance at {_energy(float(m[1]))}, l = {m[2]}: the stated total width "
               f"({_g(float(m[3]))} eV) differs from Γn + Γγ + Γf ({_g(float(m[4]))} eV) by rounding; "
               "the sum is used."),
    (r"(?:RM )?L=(\d+): (?:using )?declared AWRI=(\S+)",
     lambda m: f"l = {m[1]}: this block declares its own target mass ratio (AWRI = {m[2]}), which is "
               "used for it."),
    (r"RM (.+): using declared neutron-pair target mass ratio=(\S+)",
     lambda m: f"Spin group {m[1]}: uses the target mass ratio of its neutron channel, {m[2]}."),
)


def _adjustment(note):
    for pattern, say in _ADJUSTMENTS:
        m = re.fullmatch(pattern, note)
        if m:
            return say(m)
    return None


def _conversion_scope(d):
    """``(relevant, other)``: ``[(kind, message)]`` about MF2/MF3, and about the rest of the file."""
    n = d["notes"]
    excluded = {(e["kind"], e["message"]) for e in n["scope_exclusions"] if "cross-sections" in e["scopes"]}
    relevant, other = [], []
    for kind in ("losses", "unsupported", "approximations", "warnings"):
        for msg in n[kind]:
            mf = re.match(r"\s*MF(\d+)", msg)
            outside = (kind, msg) in excluded or (mf is not None and int(mf[1]) not in (2, 3))
            (other if outside else relevant).append((kind, msg))
    return relevant, other


def evaluate_checks(d) -> list:
    """``[{id, title, status, result, items}]`` from the report data, in ``CHECKS`` order.

    ``items`` are the findings behind a warning or a note, one sentence each."""
    tol = _tolerance(d)
    s, ranges = d["summary"], d["resonance_ranges"]
    out = []

    def add(cid, status, result, items=()):
        out.append(dict(id=cid, title=_CHECK[cid].title, status=status, result=result, items=list(items)))

    tab = s["worst_tabulation"]
    if tab is None:
        add("tabulation", PASS, "Every table is the evaluation's own, copied unchanged.")
    else:
        add("tabulation", PASS if tab["max_ratio"] <= 1 else FAIL,
            f"{len(d['reactions'])} reactions; worst difference {_pct(tab['max_ratio'])} of the tolerance "
            f"({_who(tab)} at {_energy(tab['at_eV'])}).")

    pub = s["worst_publication"]
    if d["verification"]["publication"] is None:
        add("pendf", SKIP, "No PENDF written in this run (tables only).")
    elif pub is None:
        add("pendf", PASS, f"{s['pendf']} read back; identical to the reference.")
    else:
        add("pendf", PASS if pub["max_ratio"] <= 1 else FAIL,
            f"{s['pendf']} read back; worst difference {_pct(pub['max_ratio'])} of the tolerance "
            f"({_who(pub)} at {_energy(pub['at_eV'])}).")

    disc = d["sums"]["source_discrepancies"]
    if not disc:
        add("source_sums", SKIP, "The evaluation stores no total alongside its parts.")
    else:
        bad = [x for x in disc if x["max_ratio"] > 1]
        names = ", ".join(x["name"] for x in disc)
        if not bad:
            worst = max(disc, key=lambda x: x["max_ratio"])
            add("source_sums", PASS, f"{names} checked; worst {_pct(worst['max_ratio'])} of the tolerance "
                                     f"({worst['name']}).")
        else:
            one = len(bad) == 1
            add("source_sums", WARN if any(_material(x) for x in bad) else NOTE,
                f"{', '.join(x['name'] for x in bad)} differ{'s' if one else ''} from the "
                f"sum of {'its' if one else 'their'} parts, by up to "
                f"{max(x['max_ratio'] for x in bad):.0f}× the tolerance.",
                [f"{x['name']} at {_energy(x['at_eV'])}: the evaluation states {_b(x['stated_b'])}, its parts "
                 f"add up to {_b(x['components_b'])} ({x['max_ratio']:.1f}× the tolerance). The output uses "
                 "the sum of the parts." for x in bad])

    neg = [r for r in d["reactions"] if r["negatives"]]
    if not neg:
        add("negatives", PASS, "No negative value in any table.")
    else:
        add("negatives", WARN, f"{s['negative_values']} negative values in {len(neg)} reaction(s), kept as computed.",
            [f"{_mtname(r)}: {r['negatives']['count']} negative values, the lowest "
             f"{_b(r['negatives']['min_b'])} at {_energy(r['negatives']['at_eV'])}." for r in neg])

    bounds = [b for r in ranges for b in (r["el_eV"], r["eh_eV"])]
    broken = [r for r in d["reactions"] if r.get("decreasing")]
    # A jump is worth a note when it is larger than the absolute tolerance: a step
    # of 1e-12 b where a channel opens (Au-197 (n,p) at 6 MeV) changes nothing.
    def sized(r):
        at, sizes = list(r.get("jumps_at_eV", ())), list(r.get("jumps_b") or ())
        return zip(at, sizes[:len(at)] + [None] * (len(at) - len(sizes)))
    stray = [(r, e, size) for r in d["reactions"] for e, size in sized(r)
             if not any(_close(e, b) for b in bounds) and (size is None or size > tol["atol"])]
    jumps = sum(r["jumps"] for r in d["reactions"])
    if broken:
        add("grid", FAIL, f"Energies decrease in {len(broken)} table(s).",
            [f"{_mtname(r)}: {r['decreasing']} decreasing step(s)." for r in broken])
    elif stray:
        add("grid", NOTE, f"{len(stray)} jump(s) away from the resonance-range boundaries.",
            [f"{_mtname(r)}: jump at {_energy(e)}" + (f" of {_b(size)}" if size is not None else "")
             + ", from the evaluation's table." for r, e, size in stray])
    else:
        add("grid", PASS, "Energies ascend in every table" + (
            f"; {jumps} jump(s), each at a resonance-range boundary or smaller than the tolerance."
            if jumps else "; no jumps."))

    diffs = [r for r in d["reactions"] if r.get("integral_relative_difference") is not None]
    if diffs:
        over = [r for r in diffs if r["integral_relative_difference"] > tol["rtol"]]
        worst = max(diffs, key=lambda r: r["integral_relative_difference"])
        if not over:
            add("integrals", PASS, f"{len(diffs)} reactions; worst relative difference "
                                   f"{_rel(worst['integral_relative_difference'])} ({_mtname(worst)}).")
        else:
            add("integrals", NOTE, f"{len(over)} reaction(s) above the relative tolerance of {_rel(tol['rtol'])}.",
                [f"{_mtname(r)}: {_rel(r['integral_relative_difference'])}." for r in over])

    adjusted = [a for a in (_adjustment(n) for n in d["notes"]["preparation"]) if a]
    add("parameters", NOTE if adjusted else PASS,
        f"{len(adjusted)} adjustment(s) within rounding." if adjusted else "Used exactly as stated in the file.",
        adjusted)

    relevant, other = _conversion_scope(d)
    serious = [x for x in relevant if x[0] in ("losses", "unsupported")]
    tail = (f" {len(other)} message(s) about other parts of the file, not used here, are listed under the "
            "run details." if other else "")
    if serious:
        add("conversion", WARN, f"{len(serious)} loss(es) in the data used." + tail,
            [msg for _, msg in relevant])
    elif relevant:
        add("conversion", NOTE, f"{len(relevant)} message(s) about the data used." + tail,
            [msg for _, msg in relevant])
    else:
        add("conversion", PASS, "Resonance parameters and cross sections read without loss." + tail)
    return out


#: A stored total's disagreement with its parts that is worth a warning: more
#: than this fraction of the stated total and more than this many barns.
MATERIAL_SUM_FRACTION, MATERIAL_SUM_BARNS = 1.0e-2, 1.0e-3


def _material(discrepancy) -> bool:
    """Whether a total's disagreement with its parts can change a result
    (the parts may miss a reaction), rather than being a point of rounding."""
    stated, parts = abs(discrepancy["stated_b"] or 0.0), abs(discrepancy["components_b"] or 0.0)
    gap = abs((discrepancy["stated_b"] or 0.0) - (discrepancy["components_b"] or 0.0))
    return gap > MATERIAL_SUM_FRACTION * max(stated, parts) and gap > MATERIAL_SUM_BARNS


def _overall(checks):
    worst = max((c["status"] for c in checks), key=_RANK.get, default=PASS)
    return worst if worst != SKIP else PASS


def _tolerance(d):
    o = d["options"]
    return dict(rtol=float(o["rtol"]), atol=float(o["atol_b"]))


def _tolerance_text(d):
    t = _tolerance(d)
    return f"{_rel(t['rtol'])} of σ + {_g(t['atol'])} b"


# --------------------------------------------------------------------------- object

class ReconstructionReport:
    """A reconstruction report. Build it with ``EndfReconstruction.report_document()``."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))

    def to_json(self, **kwargs) -> str:
        return json.dumps(self._data, **({"indent": 1} | kwargs))

    def to_text(self, *, verbose=False) -> str:
        """``verbose`` lists every background-only interval in the engine details."""
        return _render_text(self._data, verbose=verbose)

    def to_html(self, *, verbose=False) -> str:
        return _render_html(self._data, verbose=verbose)

    def save(self, path):
        """Write by extension: ``.html``/``.htm``, ``.json``, anything else as text."""
        path = Path(path)
        suffix = path.suffix.lower()
        body = (self.to_html() if suffix in (".html", ".htm") else
                self.to_json() if suffix == ".json" else self.to_text())
        path.write_text(body, encoding="utf-8")
        return path

    def __repr__(self):
        d = self._data
        return (f"<ReconstructionReport {_nuclide(d)} MAT={d['material'].get('mat')}: "
                f"{_VERDICT[d['summary']['status']][0].lower()}, {len(d['reactions'])} reactions, "
                f"{d['points']['stored']} stored points, "
                f"{'PENDF verified' if d['verification']['publication'] else 'no PENDF'}>")

    def _repr_html_(self):
        return _render_html(self._data, fragment=True)


# --------------------------------------------------------------------------- page content

_VERDICT = {
    PASS: ("Reconstructed and verified", "pass"),
    NOTE: ("Reconstructed and verified, with notes", "pass"),
    WARN: ("Reconstructed and verified, with warnings to review", "warn"),
    FAIL: ("Verification failed", "fail"),
}


def _verdict(d):
    s = d["summary"]
    status = s["status"]
    low, high = d["domain_eV"]
    counts = {lv: sum(1 for c in d["checks"] if c["status"] == lv) for lv in (FAIL, WARN, NOTE, PASS, SKIP)}
    parts = [f"{s['reactions']} reactions from {_energy(low)} to {_energy(high)} at 0 K, "
             f"{s['stored_points']:,} points."]
    if status != FAIL:
        parts.append(f"Every table is within the tolerance ({_tolerance_text(d)}) everywhere checked.")
    parts.append(f"PENDF {s['pendf']} written, read back and verified." if s["pendf"] else
                 "No PENDF written (tables only).")
    tally = ", ".join(f"{n} {_LABEL[lv].lower()}{'' if n == 1 or lv in (PASS, FAIL, SKIP) else 's'}"
                      for lv, n in counts.items() if n and lv != SKIP)
    parts.append(f"Checks: {tally}" + (f", {counts[SKIP]} not run." if counts[SKIP] else "."))
    return status, _VERDICT[status][0], " ".join(parts)


def _steps(d):
    """What the run did, in order, as sentences."""
    m, ev, s = d["material"], d["evaluation"], d["summary"]
    steps = [f"Read {ev.get('file') or 'the evaluation'}: {_library(ev)} evaluation of {_nuclide(d)} "
             f"(MAT {_s(m.get('mat'))}), its resonance parameters (MF2) and cross sections (MF3)."]
    if d["resonance_ranges"]:
        steps.append(f"Computed the cross sections in {len(d['resonance_ranges'])} resonance range(s) "
                     "(below) from the parameters, at 0 K, without Doppler broadening.")
    else:
        steps.append("No resonance parameters: the evaluation's tables are used as they are.")
    steps.append(f"Built one energy grid of {_count(d['points']['tabulated'])} points shared by every reaction, "
                 "refined until linear interpolation reproduces every cross section within half the tolerance.")
    if d["sums"]["graph"]:
        steps.append("Rebuilt every total from its parts (" + "; ".join(
            f"{g['name']} = {g['parts_text']}" for g in d["sums"]["graph"][:2])
            + ("; …" if len(d["sums"]["graph"]) > 2 else "") + "); the evaluation's own totals are only compared.")
    steps.append(f"Checked the tables against the reference ({len(d['checks'])} checks, listed above)." )
    if s["pendf"]:
        steps.append(f"Wrote {s['pendf']}, read it back and checked it again.")
    return steps


def _range_rows(d):
    rows = []
    for r in d["resonance_ranges"]:
        kind = {1: "Resolved", 2: "Unresolved", 0: "Scattering radius only"}.get(r["lru"], f"LRU={r['lru']}")
        model = r.get("formalism", "-") if r["lru"] == 1 else _URR_MODEL.get(r.get("urr_case"), "-")
        rows.append((_span(r["el_eV"], r["eh_eV"]), kind, model, _content(r), r.get("treatment") or "-",
                     f"LRU={r['lru']}, LRF={r['lrf']}"))
    return rows


#: The three ways ENDF-6 stores unresolved parameters (cases A, B and C of the manual).
_URR_MODEL = {"A": "constant average parameters", "B": "energy-dependent fission widths",
              "C": "energy-dependent average parameters"}


def _content(r):
    parts = []
    if r.get("resonances") is not None:
        parts.append(f"{r['resonances']:,} resonances")
    if r.get("spin_groups") is not None:
        parts.append(f"{r['spin_groups']} spin groups")
    if r.get("l_values"):
        parts.append(f"l = 0–{r['l_values'] - 1}" if r["l_values"] > 1 else "l = 0")
    if r.get("channels") is not None:
        parts.append(f"{r['channels']} channels")
    if r.get("energy_points"):
        parts.append(f"{r['energy_points']} energies")
    text = ", ".join(parts) or "-"
    ap, apl, ap_e = r.get("ap_fm"), r.get("apl_fm"), r.get("ap_e_fm")
    if ap_e:
        lo, hi = ap_e
        text += (f"; scattering radius {lo:.4g} fm (energy dependent)" if lo == hi else
                 f"; scattering radius {lo:.4g}–{hi:.4g} fm (energy dependent)")
    elif apl and not ap:
        text += "; scattering radius " + ", ".join(f"{v:.4g}" for v in apl) + " fm (stated per l)"
    elif ap is not None:
        text += f"; scattering radius {ap:.4g} fm" + (
            "; per l: " + ", ".join(f"{v:.4g}" for v in apl) + " fm" if apl else "")
    return text


def _reaction_rows(d):
    pub = d["verification"]["publication"] is not None
    rows = []
    for r in d["reactions"]:
        remark = []
        if r["derived_sum"]:
            remark.append("sum of its parts")
        if r["negatives"]:
            remark.append(f"{r['negatives']['count']} negative")
        row = [str(r["mt"]), _s(r["reaction"]), f"{r['points']:,}", _energy(r["threshold_eV"]),
               _pct(r["tabulation_verification_max"])]
        if pub:
            row.append(_pct(r["publication_verification_max"]))
        row += [_rel(r["integral_relative_difference"]), "; ".join(remark) or ""]
        rows.append(row)
    head = ["MT", "Reaction", "Points", "Starts at", "Table error"] + (["PENDF error"] if pub else []) \
        + ["Integral error", "Remark"]
    return head, rows


_SOURCE = (("MF2 AWRI", "the resonance parameters (MF2)"), ("MF1 AWR", "the file header (MF1)"),
           ("RML elastic pair", "the R-matrix elastic channel (MF2)"),
           ("no resonance physics", "not needed (no resonance parameters)"))
_STAGES = {"read": "reading", "decode": "decoding", "reconstruct": "reconstruction",
           "write_verify": "PENDF writing and check"}


def _run_groups(d):
    """``[(heading, [(label, value)])]``: the run's metadata."""
    m, ev, sw, op = d["material"], d["evaluation"], d["software"], d["options"]
    source = next((say for key, say in _SOURCE if (m.get("context_source") or "").startswith(key)),
                  m.get("context_source"))
    t = d["timings_s"]
    timing = ", ".join(f"{_STAGES.get(k, k)} {v:.1f} s" for k, v in t.items())
    return [
        ("Material", [
            ("Nuclide", f"{_nuclide(d)} (ZA {_s(m.get('za'))}, MAT {_s(m.get('mat'))})"),
            ("Target mass", f"{_g(m.get('awr_used'))} neutron masses, from {_s(source)}"),
            ("Target spin", _g(m.get("target_spin")))]),
        ("Evaluation", [
            ("Library", _library(ev)),
            ("Laboratory", _s(ev.get("laboratory"))),
            ("Date", _s(ev.get("evaluation_date"))),
            ("Authors", _s(ev.get("authors"))),
            ("File", _s(ev.get("path") or ev.get("file"))),
            ("SHA-256", _s(ev.get("sha256")))]),
        ("Run", [
            ("Generated", d["generated_utc"].replace("T", " ").replace("+00:00", " UTC")),
            ("kika", f"{sw['kika_version']} (engine {sw['engine_version']})"),
            ("Compiled kernel", "used" if sw["native_kernel"] else "not available: pure-Python fallback, "
                                                                   "same results, slower"),
            ("Tolerance", _tolerance_text(d)),
            ("Temperature", "0 K"),
            ("Energy range", _span(*d["domain_eV"])),
            ("Point limit", f"{op['max_points']:,}"),
            ("Time", f"{sum(t.values()):.1f} s ({timing})" if t else "-")]),
    ]


def _segment_rows(d, verbose):
    rows, background = [], []
    row = lambda s, kind: (_span(s["low_eV"], s["high_eV"]), kind, f"{s['points']:,}", str(s["iterations"]),
                           f"{s['evaluations']:,}", _pct(s["verification_max"]))
    for s in d["segments"]:
        if s["kind"] == "background only" and not verbose:
            background.append(s)
        else:
            rows.append(row(s, s["kind"]))
    if background:
        merged = dict(low_eV=min(s["low_eV"] for s in background), high_eV=max(s["high_eV"] for s in background),
                      points=sum(s["points"] for s in background),
                      iterations=max(s["iterations"] for s in background),
                      evaluations=sum(s["evaluations"] for s in background),
                      verification_max=max((s["verification_max"] or 0.) for s in background))
        rows.append(row(merged, f"no resonances ({len(background)} intervals)"))
    return rows


_SEGMENT_HEAD = ("Energy range", "Kind", "Points", "Refinement passes", "Evaluations", "Worst error")
_TOLERANCE_NOTE = ("Errors are given as a percentage of the tolerance: at each energy the table may differ "
                   "from the reference by {tol}, and 100 % uses all of it. The grid is refined until the "
                   "error is below 50 %, leaving the other half for the rounding of the values when a file "
                   "is written.")


# --------------------------------------------------------------------------- text

def _render_text(d, width=100, verbose=False):
    out = []
    bar = "=" * width
    status, headline, detail = _verdict(d)
    out += [bar, f" kika cross-section reconstruction  -  {_nuclide(d)}  MAT {_s(d['material'].get('mat'))}",
            bar, "", f" {_TEXT_MARK[status]} {headline.upper()}"]
    out += _wrap(detail, width, "   ")

    def section(title):
        out.extend(["", f" {title.upper()}", " " + "-" * len(title)])

    def table(head, rows):
        if not rows:
            out.append("   (none)")
            return
        cells = [list(head)] + [[_ascii(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(head))]
        fmt = lambda r: "   " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
        out.extend([fmt(cells[0]), "   " + "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]])

    section("Checks")
    for c in d["checks"]:
        out.append(f"   {_TEXT_MARK[c['status']]} {c['title']}")
        out += _wrap(c["result"], width, "          ")

    flagged = [c for c in d["checks"] if c["items"]]
    if flagged:
        section("Findings")
        for c in flagged:
            out.append(f"   {_TEXT_MARK[c['status']]} {c['title']}")
            for item in c["items"]:
                out += _wrap(item, width, "          ", first="        - ")

    section("What was done")
    for i, step in enumerate(_steps(d), 1):
        out += _wrap(step, width, "      ", first=f"   {i}. ")
    if d["resonance_ranges"]:
        out.append("")
        table(("Energy range", "Kind", "Model", "Content", "ENDF"),
              [(a, b, c, e, f) for a, b, c, e, _, f in _range_rows(d)])
        for r in _range_rows(d):
            out += _wrap(f"{r[0]}: {r[4]}.", width, "     ", first="   * ")

    section("Reactions")
    head, rows = _reaction_rows(d)
    table(head, rows)
    out += _wrap(_TOLERANCE_NOTE.format(tol=_tolerance_text(d)), width, "     ", first="   * ")

    section("Run details")
    for heading, items in _run_groups(d):
        out.append(f"   {heading}")
        k = max(len(a) for a, _ in items)
        out += [f"     {a.ljust(k)} : {_ascii(b)}" for a, b in items]

    section("Engine details")
    table(_SEGMENT_HEAD, _segment_rows(d, verbose))
    technical = _technical_notes(d)
    if technical:
        out.append("")
        out.append("   Notes from the resonance engine:")
        for msg in technical:
            out += _wrap(msg, width, "       ", first="     - ")
    _, other = _conversion_scope(d)
    if other:
        out.append("")
        out.append("   Messages about parts of the file the reconstruction does not use:")
        for kind, msg in other:
            out += _wrap(f"({kind}) {msg}", width, "       ", first="     - ")

    section("About the checks")
    out += _wrap(_TOLERANCE_NOTE.format(tol=_tolerance_text(d)), width, "   ")
    for c in CHECKS:
        out += ["", f"   {c.title}"]
        out += _wrap(c.what, width, "     ")
        for lv in (PASS, NOTE, WARN, FAIL, SKIP):
            if lv in c.levels:
                out += _wrap(c.levels[lv], width, "            ", first=f"     {_TEXT_MARK[lv]} ")
    out += ["", bar]
    return "\n".join(_ascii(line) for line in out) + "\n"


def _wrap(text, width, indent, first=None):
    import textwrap
    first = indent if first is None else first
    return textwrap.wrap(_ascii(text), width=width, initial_indent=first, subsequent_indent=indent) or [first]


def _technical_notes(d):
    return [n for n in d["notes"]["preparation"] if _adjustment(n) is None]


# --------------------------------------------------------------------------- html

_CSS = """
.kika-report{--fg:#1b1e24;--muted:#5f6672;--faint:#8a919c;--bg:#fff;--panel:#f6f7f9;--line:#e1e4e8;
--accent:#2f5d8a;--fail:#b42318;--fail-bg:#fdecea;--fail-line:#f4b9b2;--warn:#9a5b00;--warn-bg:#fff4e0;
--warn-line:#f2cd8d;--note:#3d5a80;--note-bg:#eaf1f8;--note-line:#bcd0e5;--pass:#1f7a4d;--pass-bg:#e8f5ee;
--pass-line:#b5dcc6;--skip:#6b7280;--skip-bg:#f1f2f4;--skip-line:#d6d9de;--nav-bg:#1f3a5a;--nav-fg:#f2f6fb}
@media (prefers-color-scheme:dark){.kika-report{--fg:#e7e9ed;--muted:#a3aab5;--faint:#7c8490;--bg:#14161a;
--panel:#1c1f25;--line:#2f343c;--accent:#8fb6dd;--fail:#ff8a7d;--fail-bg:#3a1d1b;--fail-line:#6e2e28;
--warn:#f2b54f;--warn-bg:#352a14;--warn-line:#6b5222;--note:#9fbbe0;--note-bg:#1d2836;--note-line:#34495f;
--pass:#6fd19f;--pass-bg:#16301f;--pass-line:#2b5a3f;--skip:#9aa1ab;--skip-bg:#22252b;--skip-line:#3a3f47;
--nav-bg:#2a4466;--nav-fg:#f2f6fb}}
.kika-report{color:var(--fg);background:var(--bg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
max-width:1180px;margin:0 auto;padding:28px 20px 56px;box-sizing:border-box}
.kika-report *{box-sizing:border-box}
.kika-report [id]{scroll-margin-top:64px}
.kika-report a{color:var(--accent);text-decoration:none}.kika-report a:hover{text-decoration:underline}
.kika-report code{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:12.5px;overflow-wrap:anywhere}
.kika-report .eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
.kika-report h1{font-size:28px;line-height:1.2;margin:4px 0 2px}
.kika-report .sub{color:var(--muted);margin:0 0 14px;font-size:13px;overflow-wrap:anywhere}
.kika-report h2{font-size:20px;margin:38px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.kika-report h3{font-size:15px;margin:18px 0 6px}
.kika-report p.lead{color:var(--muted);margin:-2px 0 12px}
.kika-report .verdict{display:flex;gap:14px;align-items:flex-start;border:1px solid var(--line);
border-left-width:6px;border-radius:10px;padding:14px 18px;margin:0 0 18px;background:var(--panel)}
.kika-report .verdict .big{font-size:18px;font-weight:700;margin:0}.kika-report .verdict p{margin:2px 0 0}
.kika-report .verdict.pass{border-color:var(--pass-line);border-left-color:var(--pass);background:var(--pass-bg)}
.kika-report .verdict.warn{border-color:var(--warn-line);border-left-color:var(--warn);background:var(--warn-bg)}
.kika-report .verdict.fail{border-color:var(--fail-line);border-left-color:var(--fail);background:var(--fail-bg)}
.kika-report .verdict.pass .big{color:var(--pass)}.kika-report .verdict.warn .big{color:var(--warn)}
.kika-report .verdict.fail .big{color:var(--fail)}
.kika-report .mark{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;
border-radius:50%;font-size:11px;font-weight:800;line-height:1;color:#fff;flex:none}
.kika-report .pass .mark{background:var(--pass)}.kika-report .note .mark{background:var(--note)}
.kika-report .warn .mark{background:var(--warn)}.kika-report .fail .mark{background:var(--fail)}
.kika-report .skip .mark{background:var(--skip)}
@media (prefers-color-scheme:dark){.kika-report .mark{color:#14161a}}
.kika-report .badge{display:inline-flex;align-items:center;gap:6px;padding:2px 10px 2px 3px;border-radius:999px;
font-size:13px;font-weight:700;white-space:nowrap;border:1px solid}
.kika-report .badge.pass{color:var(--pass);background:var(--pass-bg);border-color:var(--pass-line)}
.kika-report .badge.note{color:var(--note);background:var(--note-bg);border-color:var(--note-line)}
.kika-report .badge.warn{color:var(--warn);background:var(--warn-bg);border-color:var(--warn-line)}
.kika-report .badge.fail{color:var(--fail);background:var(--fail-bg);border-color:var(--fail-line)}
.kika-report .badge.skip{color:var(--skip);background:var(--skip-bg);border-color:var(--skip-line)}
.kika-report nav.toc{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:4px;margin:22px 0 0;
padding:6px;background:var(--nav-bg);border-radius:0 0 12px 12px}
.kika-report nav.toc a{font-weight:600;font-size:14px;color:var(--nav-fg);padding:5px 12px;border-radius:8px}
.kika-report nav.toc a:hover{background:rgba(255,255,255,.14);text-decoration:none}
.kika-report .wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:6px 0 10px}
.kika-report .wrap.tall{max-height:70vh}
.kika-report table{border-collapse:collapse;width:100%;font-size:14px}
.kika-report th,.kika-report td{padding:6px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
.kika-report tbody tr:last-child td{border-bottom:none}
.kika-report thead th{position:sticky;top:0;z-index:1;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
color:var(--muted);font-weight:600;background:var(--panel);white-space:nowrap}
.kika-report th.n,.kika-report td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.kika-report td.nw{white-space:nowrap}
.kika-report td.hot{color:var(--fail);font-weight:700}
.kika-report td.flag{color:var(--warn)}
.kika-report .muted{color:var(--muted)}.kika-report .faint{color:var(--faint);font-size:12.5px}
.kika-report table.checks td:first-child{width:1%}
.kika-report table.checks .ttl{font-weight:600}
.kika-report ol.steps{margin:0 0 10px;padding-left:22px}.kika-report ol.steps li{margin:3px 0}
.kika-report .group{margin:12px 0}.kika-report .ghead{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}
.kika-report .ghead .ttl{font-weight:700}
.kika-report ul.items{margin:6px 0 0;padding-left:22px}.kika-report ul.items li{margin:2px 0}
.kika-report ul.notes{margin:4px 0 0;padding-left:20px;color:var(--muted);font-size:14px}
.kika-report .meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.kika-report .meta section{border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.kika-report .meta h3{margin:0 0 6px}
.kika-report .meta dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 14px;margin:0}
.kika-report .meta dt{color:var(--muted)}.kika-report .meta dd{margin:0;overflow-wrap:anywhere}
.kika-report details{border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:0 14px}
.kika-report details>summary{cursor:pointer;padding:10px 0;font-weight:600}
.kika-report details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:8px}
.kika-report .entry{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:6px 28px;
border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:0 0 10px}
.kika-report .entry:target{outline:2px solid var(--accent)}
.kika-report .entry h3{margin:0 0 4px}.kika-report .entry p{margin:0;color:var(--muted);font-size:14px}
.kika-report .entry ul{list-style:none;padding:0;margin:0}
.kika-report .entry li{display:grid;grid-template-columns:104px 1fr;gap:10px;align-items:baseline;padding:4px 0}
.kika-report .entry li+li{border-top:1px solid var(--line)}.kika-report .entry li .badge{justify-self:start}
.kika-report .tol{border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:0 0 12px;background:var(--panel)}
@media (max-width:760px){.kika-report .entry{grid-template-columns:1fr}
.kika-report .entry li{grid-template-columns:1fr;gap:2px}}
@media print{.kika-report nav.toc{display:none}.kika-report .wrap.tall{max-height:none}
.kika-report .entry,.kika-report .group{break-inside:avoid}}
@page{size:A4;margin:14mm 12mm}
@media print{.kika-report{--fg:#1b1e24;--muted:#5f6672;--faint:#8a919c;--bg:#fff;--panel:#f6f7f9;--line:#e1e4e8;
--accent:#2f5d8a;--fail:#b42318;--fail-bg:#fdecea;--fail-line:#f4b9b2;--warn:#9a5b00;--warn-bg:#fff4e0;
--warn-line:#f2cd8d;--note:#3d5a80;--note-bg:#eaf1f8;--note-line:#bcd0e5;--pass:#1f7a4d;--pass-bg:#e8f5ee;
--pass-line:#b5dcc6;--skip:#6b7280;--skip-bg:#f1f2f4;--skip-line:#d6d9de}.kika-report .mark{color:#fff}
html,body{background:#fff}.kika-report *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
.kika-report .wrap{overflow:visible}.kika-report thead th{position:static}
.kika-report thead{display:table-header-group}.kika-report tr{break-inside:avoid}
.kika-report details:not([open])::details-content{content-visibility:visible;display:block}
.kika-report table{font-size:11.5px}.kika-report th,.kika-report td{padding:4px 6px}
.kika-report thead th{white-space:normal;font-size:10px}}
"""


def _e(value):
    return _html.escape("" if value is None else str(value))


def _badge(status):
    return (f'<span class="badge {_CLASS[status]}"><span class="mark">{_MARK[status]}</span>'
            f"{_LABEL[status]}</span>")


def _html_table(head, rows, *, numeric=(), nowrap=(), css="", tall=False, classes=None):
    th = "".join(f'<th{" class=n" if i in numeric else ""}>{_e(h)}</th>' for i, h in enumerate(head))
    body = []
    for k, row in enumerate(rows):
        tds = []
        for i, v in enumerate(row):
            cls = (["n"] if i in numeric else []) + (["nw"] if i in nowrap else []) \
                + ([classes(k, i)] if classes and classes(k, i) else [])
            cell = v.html if isinstance(v, _Raw) else _e(v)
            tds.append(f"<td{' class=' + chr(34) + ' '.join(cls) + chr(34) if cls else ''}>{cell}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="wrap{" tall" if tall else ""}"><table{" class=" + chr(34) + css + chr(34) if css else ""}>'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


class _Raw:
    """A cell already rendered as HTML."""

    def __init__(self, html):
        self.html = html


def _render_html(d, fragment=False, verbose=False):
    status, headline, detail = _verdict(d)
    nuclide = _nuclide(d)
    mat = _s(d["material"].get("mat"))
    ev = d["evaluation"]
    out = ['<div class="kika-report">',
           '<p class="eyebrow">kika · cross-section reconstruction</p>',
           f"<h1>{_e(nuclide)} <span class=\"muted\">· MAT {_e(mat)}</span></h1>",
           f'<p class="sub">{_e(_library(ev))} · {_e(ev.get("file"))} · kika {_e(d["software"]["kika_version"])} · '
           f'{_e(d["generated_utc"].replace("T", " ").replace("+00:00", " UTC"))}</p>',
           f'<div class="verdict {_VERDICT[status][1]}"><span class="mark">{_MARK[status]}</span>'
           f'<div><p class="big">{_e(headline)}</p><p>{_e(detail)}</p></div></div>']

    flagged = [c for c in d["checks"] if c["items"]]
    toc = [("checks", "Checks")] + ([("findings", "Findings")] if flagged else []) + [
        ("done", "What was done"), ("reactions", "Reactions"), ("run", "Run details"),
        ("about", "About the checks")]
    out.append('<nav class="toc">' + "".join(f'<a href="#kr-{a}">{_e(t)}</a>' for a, t in toc) + "</nav>")

    out.append('<h2 id="kr-checks">Checks</h2>')
    out.append('<p class="lead">What was verified in this run. Each check is explained under '
               '<a href="#kr-about">About the checks</a>.</p>')
    out.append(_html_table(("Status", "Check", "Result"), [
        (_Raw(_badge(c["status"])), _Raw(f'<a class="ttl" href="#kr-check-{c["id"]}">{_e(c["title"])}</a>'),
         c["result"]) for c in d["checks"]], css="checks"))

    if flagged:
        out.append('<h2 id="kr-findings">Findings</h2>')
        out.append('<p class="lead">The cases behind each warning or note, and what they mean.</p>')
        for c in flagged:
            out.append(f'<div class="group"><div class="ghead">{_badge(c["status"])}'
                       f'<span class="ttl">{_e(c["title"])}</span>'
                       f'<span class="muted">{len(c["items"])} case(s)</span></div>')
            meaning = _CHECK[c["id"]].levels.get(c["status"])
            if meaning:
                out.append(f'<p class="muted">{_e(meaning)}</p>')
            shown = c["items"][:8]
            out.append('<ul class="items">' + "".join(f"<li>{_e(x)}</li>" for x in shown) + "</ul>")
            if len(c["items"]) > len(shown):
                out.append(f'<details><summary>Show the other {len(c["items"]) - len(shown)}</summary>'
                           '<ul class="items">' + "".join(f"<li>{_e(x)}</li>" for x in c["items"][len(shown):])
                           + "</ul></details>")
            out.append("</div>")

    out.append('<h2 id="kr-done">What was done</h2>')
    out.append('<ol class="steps">' + "".join(f"<li>{_e(s)}</li>" for s in _steps(d)) + "</ol>")
    if d["resonance_ranges"]:
        out.append("<h3>Resonance ranges</h3>")
        out.append(_html_table(("Energy range", "Kind", "Model", "Content", "How it was used", "ENDF"),
                               [r[:5] + (_Raw(f'<span class="faint">{_e(r[5])}</span>'),) for r in _range_rows(d)],
                               nowrap=(0, 1, 5)))
    if d["sums"]["graph"]:
        out.append("<h3>Totals rebuilt from their parts</h3>")
        out.append(_html_table(("Total", "Sum of"), [(g["name"], g["parts_text"]) for g in d["sums"]["graph"]],
                               nowrap=(0,)))

    out.append('<h2 id="kr-reactions">Reactions</h2>')
    out.append(f'<p class="lead">{_e(_TOLERANCE_NOTE.format(tol=_tolerance_text(d)))} '
               "“Starts at” is the first energy with a non-zero cross section.</p>")
    head, rows = _reaction_rows(d)
    error_key = {"Table error": "tabulation_verification_max", "PENDF error": "publication_verification_max"}

    def hot(k, i):
        r = d["reactions"][k]
        if head[i] in error_key:
            return "hot" if (r[error_key[head[i]]] or 0) > 1 else None
        return "flag" if head[i] == "Remark" and r["negatives"] else None
    numeric = {i for i, h in enumerate(head) if h in ("MT", "Points", "Starts at", "Table error",
                                                      "PENDF error", "Integral error")}
    out.append(_html_table(head, rows, numeric=numeric, tall=True, classes=hot))

    out.append('<h2 id="kr-run">Run details</h2>')
    out.append('<div class="meta">' + "".join(
        f"<section><h3>{_e(h)}</h3><dl>" + "".join(
            f"<dt>{_e(a)}</dt><dd>{f'<code>{_e(b)}</code>' if a in ('SHA-256', 'File') else _e(b)}</dd>"
            for a, b in items) + "</dl></section>" for h, items in _run_groups(d)) + "</div>")
    out.append("<details><summary>Engine details: how the energy range was divided and refined</summary>")
    out.append(_html_table(_SEGMENT_HEAD, _segment_rows(d, verbose), numeric=(2, 3, 4, 5), nowrap=(0,)))
    out.append('<ul class="notes"><li>Each interval is refined independently; “Refinement passes” is how '
               "many times its grid was subdivided, “Evaluations” how many times the physics was computed. "
               "Intervals with no resonances keep the evaluation's own points.</li></ul>")
    technical = _technical_notes(d)
    if technical:
        out.append('<h3>Notes from the resonance engine</h3><ul class="notes">'
                   + "".join(f"<li><code>{_e(n)}</code></li>" for n in technical) + "</ul>")
    out.append("</details>")
    _, other = _conversion_scope(d)
    if other:
        out.append(f"<details><summary>{len(other)} message(s) about parts of the file the reconstruction "
                   "does not use</summary>")
        out.append('<p class="muted">Reading the whole evaluation also reports on data outside resonance '
                   "parameters and cross sections. None of it affects these results.</p>")
        out.append('<ul class="notes">' + "".join(f"<li>{_e(m)}</li>" for _, m in other) + "</ul></details>")

    out.append('<h2 id="kr-about">About the checks</h2>')
    out.append(f'<div class="tol"><strong>Tolerance.</strong> '
               f"{_e(_TOLERANCE_NOTE.format(tol=_tolerance_text(d)))}</div>")
    for c in CHECKS:
        out.append(f'<div class="entry" id="kr-check-{c.id}"><div><h3>{_e(c.title)}</h3><p>{_e(c.what)}</p></div><ul>'
                   + "".join(f"<li>{_badge(lv)}<span>{_e(c.levels[lv])}</span></li>"
                             for lv in (PASS, NOTE, WARN, FAIL, SKIP) if lv in c.levels) + "</ul></div>")
    out.append("</div>")
    content = "<style>" + _CSS + "</style>" + "\n".join(out)
    if fragment:
        return content
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            f"<title>Reconstruction of {_e(nuclide)}</title></head>"
            '<body style="margin:0;background:Canvas">' + content + "</body></html>")


# --------------------------------------------------------------------------- helpers

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _fm(ap):
    """ENDF states radii in 10^-12 cm; one of those is 10 fm."""
    ap = _num(ap)
    return None if ap is None else 10 * ap


def _int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9 * max(abs(float(a)), abs(float(b)), 1e-30)


def _maxval(mapping):
    values = [float(v) for v in (mapping or {}).values()]
    return max(values) if values else None


def _ranges_text(mts):
    """``[51, 52, ..., 91, 102]`` -> ``MT51-91 + MT102``; the readable sum."""
    mts = sorted(mts)
    if not mts:
        return "-"
    runs, start = [], mts[0]
    for a, b in zip(mts, mts[1:] + [None]):
        if b != a + 1:
            runs.append(f"MT{start}" if start == a else f"MT{start}-{a}")
            start = b
    return " + ".join(runs)


_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}


def _library(ev):
    """``ENDF/B-VIII.1``, ``JEFF-4.0``: the library as people write it."""
    name, nver, lrel = ev.get("library"), ev.get("nver"), ev.get("lrel")
    if not name:
        return "-" if ev.get("nlib") is None else f"library NLIB={ev['nlib']}"
    if nver is None:
        return name
    version = _ROMAN.get(nver, str(nver)) if name.startswith("ENDF/B") else str(nver)
    return f"{name}-{version}" + (f".{lrel}" if lrel is not None else "")


def _nuclide(d):
    """``Fe-56``, ``Am-242m1``."""
    m = d["material"]
    target = m.get("target") or ""
    match = re.fullmatch(r"([A-Za-z]+)(\d+.*)", target)
    if match:
        return f"{match[1]}-{match[2]}"
    return target or (m.get("zsymam") or "-").replace(" ", "")


def _who(row):
    mt = row.get("mt")
    if mt is None:
        return row.get("name") or "-"
    from kika._constants import MT_TO_REACTION
    label = reaction_name(mt, MT_TO_REACTION)
    return f"MT{mt} {label}" if label else f"MT{mt}"


def _mtname(r):
    return f"MT{r['mt']} {r['reaction']}" if r.get("reaction") else f"MT{r['mt']}"


def _s(v):
    return "-" if v is None or v == "" else str(v)


def _g(v):
    """``1e-8``, not ``1e-08``."""
    if v is None:
        return "-"
    return re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m[1] == "-" else "") + m[2], f"{float(v):.6g}")


def _energy(v):
    """An energy in the unit a reader expects: ``671.2 keV``, ``2.53e-2 eV``."""
    if v is None:
        return "-"
    v = float(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.4g} MeV"
    if a >= 1e3:
        return f"{v / 1e3:.4g} keV"
    if a >= 0.1 or a == 0:
        return f"{v:.4g} eV"
    return f"{_g(float(f'{v:.3g}'))} eV"


def _span(a, b):
    return f"{_energy(a)} – {_energy(b)}"


def _pct(ratio):
    """An error as a share of the tolerance."""
    if ratio is None:
        return "-"
    ratio = float(ratio)
    if ratio == 0:
        return "0"
    if ratio < 1e-3:
        return "< 0.1 %"
    return f"{100 * ratio:.1f} %"


def _rel(v):
    """A relative difference: ``0.1 %``, ``3e-5 %``."""
    if v is None:
        return "-"
    v = 100 * float(v)
    if v == 0:
        return "0"
    if v < 1e-4:
        return "< 0.0001 %"
    return f"{v:.3g} %" if v >= 1e-3 else f"{_g(float(f'{v:.1g}'))} %"


def _count(v):
    return "-" if v is None else f"{int(v):,}"


def _b(v):
    """A cross section precise enough for a difference to show."""
    return "-" if v is None else f"{float(v):.7g} b"


def reaction_name(mt, known=None):
    """Reaction label for an MT, including the per-level charged-particle MTs."""
    if mt is None:
        return None
    if known and mt in known:
        return known[mt]
    for base, particle in ((600, "p"), (650, "d"), (700, "t"), (750, "³He"), (800, "α")):
        if base <= mt < base + 50:
            level = mt - base
            return f"(n,{particle}{'c' if level == 49 else level})"
    return None


def _ascii(text):
    text = str(text)
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉³", "01234567893"))
    for a, b in (("α", "a"), ("γ", "g"), ("Γ", "G"), ("σ", "sigma"), ("–", "-"), ("—", "-"), ("−", "-"),
                 ("′", "'"), ("·", "*"), ("×", "x"), ("…", "..."), ("“", '"'), ("”", '"'), ("✓", "ok")):
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode("ascii")

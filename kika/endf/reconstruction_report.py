"""The reconstruction report: what was reconstructed, how, and how well.

``EndfReconstruction.report_document()`` builds a :class:`ReconstructionReport`
from what the run already knows; nothing is recomputed except cheap scans of
the output tables. It renders three ways from one dictionary:

* ``to_dict()``  JSON-safe, with stable keys (``schema`` names the version);
* ``to_text()``  fixed-width ASCII for a terminal or a ``.txt`` file;
* ``to_html()``  a self-contained page (inline CSS, no script, light and dark).

Sections: material and evaluation, software, options, resonance ranges, the
segments the domain was cut into, one row per tabulated MT, the sum graph and
the evaluation's own sums against the reconstructed ones, verification, the
conversion notes, and timings.

Error ratios are in units of the budget ``atol + rtol*|sigma|``: 1 means the
difference uses the whole tolerance. Tabulation aims at half of it, leaving the
other half for serialization (``ReconstructionOptions``).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import html as _html
import json
from pathlib import Path

import numpy as np

__all__ = ["ReconstructionReport", "SCHEMA"]

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
    out = dict(file=Path(path).name, sha256=file_sha256(path))
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


def _ap_fm(ap):
    """ENDF writes AP in 10^-12 cm; one of those is 10 fm."""
    value = _num(ap)
    return None if value is None else 10.0 * value


def _range_row(isotope, rng) -> dict:
    p = rng.parameters
    row = dict(el_eV=_num(rng.el), eh_eV=_num(rng.eh), lru=_int(rng.lru), lrf=_int(rng.lrf),
               nro=_int(rng.nro), naps=_int(rng.naps), ap_fm=_ap_fm(getattr(p, "ap", None)),
               ap_energy_points=len(rng.ap_e.energies) if getattr(rng, "ap_e", None) else 0,
               lfw=_int(getattr(isotope, "lfw", None)), za=_int(getattr(isotope, "za", None)),
               abundance=_num(getattr(isotope, "abn", None)))
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
                      ("file", "sha256", "library", "nlib", "nver", "lrel", "nmod", "lrp", "lfi",
                       "temperature_K", "emax_eV", "evaluation_date", "laboratory", "authors")}

    # Resonance ranges: the tape's flags joined with how the engine prepared them.
    prepared = list(getattr(result._prepared, "regions", ()) or ())
    for row in ranges:
        match = next((r for r in prepared if _close(r.low, row["el_eV"]) and _close(r.high, row["eh_eV"])), None)
        row["engine_approximation"] = getattr(match, "approximation", None)
        if row["lru"] == 1:
            row["treatment"] = "pointwise at 0 K from the resonance parameters"
        elif row["lru"] == 2:
            data = getattr(match, "unresolved", None)
            if row.get("lssf") == 1 or getattr(data, "self_shielding_only", False):
                row["treatment"] = "LSSF=1: MF3 already holds the dilute cross section; MF2 adds nothing at 0 K"
            else:
                law = getattr(data, "law", None)
                row["treatment"] = ("infinite-dilution average with Hauser-Feshbach fluctuation integrals, "
                                    + (f"computed on the evaluation's grid and interpolated with INT={law}"
                                       if law else "evaluated at every grid energy from the parameter functions"))
                row["urr_interpolation"] = law
        else:
            row["treatment"] = "potential scattering only"

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
    reactions = []
    for mt in sorted(tables):
        t = tables[mt]
        x, y = np.asarray(t.energies), np.asarray(t.values)
        dx = np.diff(x)
        positive = np.flatnonzero(dx > 0)
        relative = dx[positive] / x[positive]
        negative = np.flatnonzero(y < 0)
        key = result._mt_keys[mt]
        ref, diff = integrals.get(key, (0., 0.))
        reactions.append(dict(
            mt=mt, reaction=reaction(mt), points=int(len(x)),
            e_min_eV=float(x[0]), e_max_eV=float(x[-1]),
            jumps=int(np.count_nonzero(dx == 0)),
            min_relative_spacing=float(relative.min()) if len(positive) else None,
            min_relative_spacing_at_eV=float(x[positive[np.argmin(relative)]]) if len(positive) else None,
            negatives=(dict(count=int(len(negative)), min_b=float(y[negative].min()),
                            at_eV=float(x[negative[np.argmin(y[negative])]])) if len(negative) else None),
            derived_sum=key in result._graph,
            tabulation_verification_max=per_key_verification.get(key),
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
    pub = getattr(rec, "publication", None)
    if pub:
        maxima, worst = pub.get("maxima") or {}, pub.get("worst") or {}
        publication = dict(
            path=str(pub.get("path")), file=Path(str(pub.get("path"))).name, mat=pub.get("mat"),
            reloaded_and_verified=True,
            rows=[dict(mt=key_mt.get(key), name=name(key), max_ratio=float(value),
                       at_eV=_num(worst.get(key, (None,))[0]),
                       reference_b=_num(worst.get(key, (None, None))[1]),
                       serialized_b=_num(worst.get(key, (None, None, None))[2]))
                  for key, value in sorted(maxima.items(), key=lambda kv: key_mt.get(kv[0], 10**9))])

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
    return _jsonable(dict(
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
        """``verbose`` lists every background-only segment instead of one summary row."""
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
        m = d["material"]
        return (f"<ReconstructionReport {m.get('zsymam') or m.get('target')} MAT={m.get('mat')}: "
                f"{len(d['reactions'])} reactions, {d['points']['stored']} stored points, "
                f"{'PENDF verified' if d['verification']['publication'] else 'no PENDF'}>")

    def _repr_html_(self):
        return _render_html(self._data, fragment=True)


# --------------------------------------------------------------------------- shared rows

def _summary_rows(d):
    s = d["summary"]
    tab, pub, sums = s["worst_tabulation"], s["worst_publication"], s["worst_source_sum"]
    at = lambda r: f"{r['name']} at {_E(r['at_eV'])} eV" if r.get("at_eV") is not None else r["name"]
    verdict = "VERIFIED" if s["verified"] else "FAILED"
    if tab is not None:
        verdict += f": worst tabulation error {_r(tab['max_ratio'])} of the budget ({at(tab)})"
    rows = [("Verdict", verdict)]
    rows.append(("PENDF", "not written (tables only); write_pendf() adds a reload check" if pub is None else
                 f"{s['pendf']} written, reloaded and verified: worst {_r(pub['max_ratio'])} of the budget ({at(pub)})"))
    if sums is None:
        rows.append(("Evaluation's sums", "none stored"))
    elif s["source_sums_consistent"]:
        rows.append(("Evaluation's sums", f"consistent with their parts (worst {_r(sums['max_ratio'])}, {sums['name']})"))
    else:
        rows.append(("Evaluation's sums", f"{sums['name']} disagrees with the sum of its parts: {sums['max_ratio']:.1f}x the "
                     f"budget at {_E(sums['at_eV'])} eV; the output uses the sum of the parts"))
    rows.append(("Negative values", "none" if not s["negative_values"] else
                 f"{s['negative_values']} in " + ", ".join(f"MT{mt}" for mt in s["reactions_with_negatives"])
                 + " (kept as computed)"))
    rows.append(("Conversion", "clean, nothing lost or approximated" if not s["conversion_issues"] else
                 f"{s['conversion_issues']} declared issue(s); clean for cross sections: "
                 f"{'yes' if s['clean_for_cross_sections'] else 'NO'} (see notes)"))
    rows.append(("Output", f"{s['reactions']} reactions, {s['stored_points']:,} points"))
    rows.append(("Time", f"{s['total_time_s']:.1f} s"))
    return rows


def _content(r):
    parts = [f"{r[k]} {label}" for k, label in
             (("l_values", "l-values"), ("spin_groups", "J groups"), ("resonances", "resonances"),
              ("channels", "channels"), ("energy_points", "energies")) if r.get(k) is not None]
    text = ", ".join(parts) or "-"
    if r.get("particle_pairs"):
        text += "; pairs MT " + ", ".join(str(x) for x in r["particle_pairs"])
    return text


def _segment_rows(d, verbose):
    rows, background = [], []
    row = lambda s, kind: (_rng(s["low_eV"], s["high_eV"]), kind, f"{s['points']:,}", str(s["iterations"]),
                           f"{s['evaluations']:,}", _r(s["refinement_max"]), _r(s["verification_max"]))
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
                      refinement_max=max((s["refinement_max"] or 0.) for s in background),
                      verification_max=max((s["verification_max"] or 0.) for s in background))
        rows.append(row(merged, f"background only ({len(background)} segment{'s' if len(background) > 1 else ''})"))
    return rows


def _sections(d, verbose=False):
    """(id, title, headers, rows, notes) per section; headers None means key/value."""
    yield ("summary", "Summary", None, _summary_rows(d), [])
    m, ev, sw, op = d["material"], d["evaluation"], d["software"], d["options"]
    lib = " ".join(x for x in (ev.get("library"), _vers(ev)) if x) or "-"
    yield ("material", "Material and evaluation", None, [
        ("Material", f"{m.get('zsymam') or m.get('target')}"),
        ("ZA / MAT", f"{_s(m.get('za'))} / {_s(m.get('mat'))}"),
        ("Target / projectile", f"{_s(m.get('target'))} / {_s(m.get('projectile'))}"),
        ("AWR used", f"{_g(m.get('awr_used'))} (MF1: {_g(m.get('awr_mf1'))})"),
        ("Target spin", _g(m.get("target_spin"))),
        ("Mass and spin from", _s(m.get("context_source"))),
        ("Library", f"{lib} (NLIB={_s(ev.get('nlib'))}, NVER={_s(ev.get('nver'))}, "
                    f"LREL={_s(ev.get('lrel'))}, NMOD={_s(ev.get('nmod'))})"),
        ("Laboratory / date", f"{_s(ev.get('laboratory'))} / {_s(ev.get('evaluation_date'))}"),
        ("Authors", _s(ev.get("authors"))),
        ("LRP / LFI", f"{_s(ev.get('lrp'))} / {_s(ev.get('lfi'))}"),
        ("Upper energy", f"{_E(ev.get('emax_eV'))} eV"),
        ("File", _s(ev.get("file"))),
        ("SHA-256", _s(ev.get("sha256")))], [])
    yield ("processing", "Processing", None, [
        ("Generated (UTC)", d["generated_utc"]),
        ("kika / engine", f"{sw['kika_version']} / {sw['engine_version']}"),
        ("C kernel", "loaded" if sw["native_kernel"] else "NOT loaded (NumPy fallback, slower)"),
        ("Tolerance", f"rtol {_g(op['rtol'])}, atol {_g(op['atol_b'])} b; budget = atol + rtol·|σ|"),
        ("Budget split", "half for tabulation, half kept for serialization"),
        ("Limits", f"max_points {op['max_points']:,}, max_iterations {op['max_iterations']}"),
        ("Temperature", "0 K (no Doppler broadening)"),
        ("Domain", f"{_rng(*d['domain_eV'])} eV"),
        ("Points", f"{_s(d['points']['tabulated'])} on the shared grid, {_s(d['points']['stored'])} stored")], [])

    rows = [(_rng(r["el_eV"], r["eh_eV"]), f"{r['lru']}/{r['lrf']}", r.get("formalism", "-"),
             f"{r['nro']}/{r['naps']}",
             _g(r.get("ap_fm")) + (f" + AP(E), {r['ap_energy_points']} pts" if r.get("ap_energy_points") else ""),
             _content(r)) for r in d["resonance_ranges"]]
    notes = [f"{_rng(r['el_eV'], r['eh_eV'])} eV: {r['treatment']}" for r in d["resonance_ranges"] if r.get("treatment")]
    yield ("mf2", "Resonance ranges (MF2)",
           ("E range (eV)", "LRU/LRF", "Formalism", "NRO/NAPS", "AP (fm)", "Content"), rows, notes)

    yield ("segments", "Segments",
           ("E range (eV)", "Kind", "Points", "Iter", "Evaluations", "Refine max", "Verify max"),
           _segment_rows(d, verbose),
           ["Refine max: largest panel error over the budget while refining (target <= 0.5).",
            "Verify max: largest error at independent probes of the final table (must be <= 1)."]
           + ([] if verbose else ["Background-only segments copy the evaluated tables; verbose=True lists each."]))

    rows, notes = [], []
    for r in d["reactions"]:
        neg = r["negatives"]
        rows.append((str(r["mt"]) + (" (sum)" if r["derived_sum"] else ""), _s(r["reaction"]), f"{r['points']:,}",
                     _rng(r["e_min_eV"], r["e_max_eV"]), str(r["jumps"]),
                     _e(r["min_relative_spacing"]) + (f" at {_E(r['min_relative_spacing_at_eV'])}"
                                                       if r["min_relative_spacing_at_eV"] else ""),
                     "-" if neg is None else f"{neg['count']} (min {_E(neg['min_b'])} b at {_E(neg['at_eV'])})",
                     _r(r["tabulation_verification_max"]), _e(r["integral_relative_difference"])))
        if neg is not None:
            notes.append(f"MT{r['mt']}: {neg['count']} negative values kept as computed (reported, not clipped).")
    notes[:0] = ["Jumps: repeated energies, each a real discontinuity (range boundary); keep both points.",
                 "Min dE/E: the finest relative grid spacing and the energy (eV) where it occurs.",
                 "Integral diff: |linear - exact| / exact for the integral of σ dE, from the engine's quadrature."]
    yield ("reactions", "Tabulated reactions", ("MT", "Reaction", "Points", "E range (eV)", "Jumps", "Min dE/E",
                                               "Negative", "Verify max", "Integral diff"), rows, notes)

    s = d["sums"]
    yield ("sums", "Sums", ("Aggregate", "Derived from"), [(g["name"], g["parts_text"]) for g in s["graph"]],
           [f"Policy: {s['policy']}. Every aggregate is rebuilt from its exclusive leaves; "
            "the evaluation's own aggregate is only compared, never kept."])
    yield ("source-sums", "Evaluation's sums against their parts",
           ("Aggregate", "Max / budget", "At E (eV)", "Stated (b)", "Sum of parts (b)"),
           [(x["name"], _r(x["max_ratio"]), _E(x["at_eV"]), _V(x["stated_b"]), _V(x["components_b"]))
            for x in s["source_discrepancies"]],
           ["Above 1: the stored aggregate differs from the sum of its parts by more than the tolerance; "
            "the output uses the sum of the parts."] if s["source_discrepancies"] else [])

    v = d["verification"]
    total = len(v["tabulation"])
    shown = v["tabulation_rows"]
    rest = total - len(shown)
    yield ("verification", "Verification of the tabulation",
           ("Reaction", "Max / budget", "At E (eV)", "Exact (b)", "Tabulated (b)"),
           [(f"{x['name']} {x['reaction'] or ''}".strip(), _r(x["max_ratio"]), _E(x["at_eV"]),
             _V(x["reference_b"]), _V(x["tabulated_b"])) for x in shown],
           ([f"{rest} other reactions lie outside the resonance ranges: their evaluated lin-lin tables are "
             "passed through exactly (0.000)."] if rest else [])
           + ["Checked at four independent probes per panel against the physics; empirical, not a global bound."])
    pub = v["publication"]
    if pub is not None:
        rows = [(x["name"], _r(x["max_ratio"]), _E(x["at_eV"]), _V(x["reference_b"]), _V(x["serialized_b"]))
                for x in pub["rows"] if x["max_ratio"] > 0]
        rest = len(pub["rows"]) - len(rows)
        yield ("publication", "Verification of the PENDF (reloaded)",
               ("Reaction", "Max / budget", "At E (eV)", "Reference (b)", "Serialized (b)"), rows,
               [f"Written to {pub['file']} (MAT {_s(pub['mat'])}), read back and checked against the physics "
                "at every node and four interior probes per panel."]
               + ([f"{rest} other reactions: 0.000."] if rest else []))

    n = d["notes"]
    excluded = {e["message"]: e["scopes"] for e in n["scope_exclusions"]}
    rows = [(kind + (f" (not relevant to {', '.join(excluded[msg])})" if msg in excluded else ""), msg)
            for kind in ("losses", "approximations", "unsupported", "warnings") for msg in n[kind]]
    rows += [("engine", msg) for msg in n["preparation"]]
    yield ("notes", "Conversion and preparation notes", ("Kind", "Message"), rows,
           ["Nothing to declare: the evaluation converted cleanly."] if not rows else
           [f"Clean for cross sections: {'yes' if n['clean_for_cross_sections'] else 'NO'}."])

    t = d["timings_s"]
    yield ("timings", "Timings", ("Stage", "Seconds"),
           [(k, f"{x:.2f}") for k, x in t.items()] + [("total", f"{sum(t.values()):.2f}")], [])


# --------------------------------------------------------------------------- text

def _render_text(d, width=100, verbose=False):
    out = []
    bar = "=" * width
    m = d["material"]
    out += [bar, f" KIKA RESONANCE RECONSTRUCTION  -  {m.get('zsymam') or m.get('target')}  MAT {_s(m.get('mat'))}".rstrip(),
            bar]
    for _, title, headers, rows, notes in _sections(d, verbose):
        out += ["", f" {title.upper()}", " " + "-" * (len(title))]
        if headers is None:
            k = max(len(a) for a, _ in rows)
            out += [f"   {a.ljust(k)} : {_ascii(b)}" for a, b in rows]
        elif rows:
            cells = [list(headers)] + [[_ascii(c) for c in r] for r in rows]
            widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
            fmt = lambda r: "   " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
            out += [fmt(cells[0]), "   " + "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]]
        else:
            out.append("   (none)")
        out += [f"   * {_ascii(note)}" for note in notes]
    out += ["", bar]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- html

_CSS = """
.kika-report{--bg:#ffffff;--fg:#1f2328;--muted:#656d76;--line:#d8dee4;--head:#f6f8fa;--accent:#2a78d6;
 --warn:#b35900;--ok:#1a7f37;--box:#f6f8fa}
@media (prefers-color-scheme:dark){.kika-report{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;
 --head:#161b22;--accent:#58a6ff;--warn:#e3a14b;--ok:#3fb950;--box:#161b22}}
.kika-report{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
 max-width:1100px;margin:0 auto;padding:16px;font-variant-numeric:tabular-nums}
.kika-report h1{font-size:20px;margin:0 0 4px}
.kika-report .sub{color:var(--muted);margin:0 0 12px}
.kika-report nav{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;margin:0 0 8px;padding:6px 0;
 border-bottom:1px solid var(--line)}
.kika-report nav a{color:var(--accent);text-decoration:none}
.kika-report nav a:hover{text-decoration:underline}
.kika-report h2{font-size:15px;margin:22px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--line);color:var(--accent)}
.kika-report .wrap{overflow:auto;max-height:70vh;border:1px solid var(--line);border-radius:6px}
.kika-report table{border-collapse:collapse;width:100%}
.kika-report th,.kika-report td{text-align:left;padding:4px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.kika-report thead th{background:var(--head);font-weight:600;white-space:nowrap;position:sticky;top:0;z-index:1}
.kika-report td.num{text-align:right;white-space:nowrap}
.kika-report td.e{white-space:nowrap}
.kika-report .bad{color:var(--warn);font-weight:600}
.kika-report .ok{color:var(--ok);font-weight:600}
.kika-report dl.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:2px 24px;margin:0}
.kika-report dl.grid div{display:grid;grid-template-columns:150px 1fr;gap:8px;padding:3px 0;border-bottom:1px solid var(--line)}
.kika-report dl.summary{display:grid;grid-template-columns:170px 1fr;gap:4px 14px;margin:0;padding:10px 14px;
 background:var(--box);border:1px solid var(--line);border-radius:6px}
.kika-report dt{color:var(--muted)}
.kika-report dd{margin:0;overflow-wrap:anywhere}
.kika-report ul{margin:6px 0 0;padding-left:20px;color:var(--muted)}
.kika-report code{font-family:ui-monospace,Consolas,monospace;font-size:12px}
"""


def _render_html(d, fragment=False, verbose=False):
    e = _html.escape
    m = d["material"]
    title = f"{m.get('zsymam') or m.get('target')}, MAT {_s(m.get('mat'))}"
    sections = list(_sections(d, verbose))
    body = [f'<div class="kika-report"><h1>Resonance reconstruction: {e(title)}</h1>',
            f'<p class="sub">kika {e(str(d["software"]["kika_version"]))}, engine '
            f'{e(str(d["software"]["engine_version"]))}, {e(d["generated_utc"])}</p>',
            "<nav>" + "".join(f'<a href="#kr-{sid}">{e(heading)}</a>' for sid, heading, *_ in sections) + "</nav>"]
    for sid, heading, headers, rows, notes in sections:
        body.append(f'<h2 id="kr-{sid}">{e(heading)}</h2>')
        if headers is None and sid == "summary":
            items = []
            for a, b in rows:
                cls = ""
                if a == "Verdict":
                    cls = ' class="ok"' if b.startswith("VERIFIED") else ' class="bad"'
                elif a == "Evaluation's sums" and "disagrees" in b or a == "Negative values" and b != "none":
                    cls = ' class="bad"'
                items.append(f"<dt>{e(a)}</dt><dd{cls}>{e(b)}</dd>")
            body.append('<dl class="summary">' + "".join(items) + "</dl>")
        elif headers is None:
            body.append('<dl class="grid">' + "".join(
                f"<div><dt>{e(a)}</dt><dd>{_cell(b, a == 'SHA-256')}</dd></div>" for a, b in rows) + "</dl>")
        elif rows:
            head = "".join(f"<th>{e(h)}</th>" for h in headers)
            trs = "".join("<tr>" + "".join(_td(c, h) for c, h in zip(r, headers)) + "</tr>" for r in rows)
            body.append(f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table></div>')
        else:
            body.append('<p class="sub">(none)</p>')
        if notes:
            body.append("<ul>" + "".join(f"<li>{e(n)}</li>" for n in notes) + "</ul>")
    body.append("</div>")
    content = "<style>" + _CSS + "</style>" + "".join(body)
    if fragment:
        return content
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            f"<title>Reconstruction {e(title)}</title></head>"
            f'<body style="margin:0;background:Canvas">{content}</body></html>')


_NUMERIC = {"Points", "Iter", "Evaluations", "Refine max", "Verify max", "Jumps", "Integral diff",
            "Max / budget", "Seconds", "Stated (b)", "Sum of parts (b)", "Reference (b)", "Serialized (b)",
            "Exact (b)", "Tabulated (b)", "At E (eV)"}
_RATIO = {"Refine max", "Verify max", "Max / budget"}
_NOWRAP = {"E range (eV)", "Min dE/E", "Negative"}


def _td(value, header):
    cls = []
    if header in _NUMERIC:
        cls.append("num")
    if header in _NOWRAP:
        cls.append("e")
    if header in _RATIO:
        try:
            if float(value) > 1:
                cls.append("bad")
        except ValueError:
            pass
    if header == "Negative" and value != "-":
        cls.append("bad")
    attr = f' class="{" ".join(cls)}"' if cls else ""
    return f"<td{attr}>{_html.escape(str(value))}</td>"


def _cell(value, code=False):
    text = _html.escape(str(value))
    return f"<code>{text}</code>" if code else text


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


def _vers(ev):
    n = ev.get("nver")
    return None if n is None else f"v{n}"


def _s(v):
    return "-" if v is None or v == "" else str(v)


def _g(v):
    return "-" if v is None else f"{float(v):.6g}"


def _E(v):
    """Energies and cross sections: four significant figures, always scientific."""
    return "-" if v is None else f"{float(v):.3e}"


def _V(v):
    """Compared cross sections: seven significant figures, so the difference shows."""
    return "-" if v is None else f"{float(v):.6e}"


def _rng(a, b):
    return f"{_E(a)} – {_E(b)}"


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


def _r(v):
    return "-" if v is None else f"{float(v):.3f}"


def _e(v):
    return "-" if v is None else f"{float(v):.1e}"


def _ascii(text):
    text = str(text)
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉³", "01234567893"))
    for a, b in (("α", "a"), ("γ", "g"), ("σ", "sigma"), ("–", "-"), ("—", "-"), ("′", "'"), ("·", "*")):
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode("ascii")

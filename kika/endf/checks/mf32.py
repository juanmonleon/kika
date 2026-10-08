"""Layer 1 for MF32: the covariance of the resonance parameters of File 2.

The rows of these matrices are model parameters (ER, GN, GG ... of each
resonance), not bins of an energy grid, so the MF33 checks carry over with two
changes that matter:

* **PSD on the correlation matrix.** The parameters mix scales (an ER uncertainty
  in eV next to a width uncertainty in meV), and lambda_min / lambda_max of the
  covariance then measures the scales rather than the matrix: Rh-103 and Th-232
  of ENDF/B-VIII.1 give 1e-7 there, and -0.63 / -0.66 on their correlation
  matrices, which cannot be sampled. The same three levels are applied to R.
  LCOMP=2 stores R as NDIGIT-digit integers (INTG), and the rounding allowance is
  the one the MF33 checks use for quantised correlations.
* **No physical bound on a width.** A width is unbounded above and, in LRF=3 and
  LRF=7, signed (the sign of the reduced-width amplitude), so negative values are
  legitimate and sigma_rel > 1 is only a note.

What the file says about itself is checked first: INTG records whose row or
column lies outside the NNN x NNN matrix (JEFF-4.0 K-41 writes row 201 with
NNN = 93), counts the decoder cannot reconcile (``decodeMF32MT`` reports and
skips them), and agreement with MF2: the same energy ranges, the same formalism,
and resonance energies that MF2 actually has.

The decoding is :func:`kika.endf.model_adapter.parameter_covariances.decodeMF32MT`,
the one the model uses, so there is a single reader of the §32 layouts.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .findings import DEFECT, NOTE, WARN, CovarianceFinding, CovarianceLocation

#: Largest matrix whose eigenvalues are computed (Pt-192 of JEFF-4.0 is 5475).
EIG_MAX = 6000
#: Relative tolerance when matching MF32 energies to MF2's: ENDF-6 §32.3
#: (Correspondence, 2 and 3) identifies a resonance to 1e-5.
_E_RTOL = 1e-5
_ITEMS = 10


def check_mf32(ctx, mf_obj, out: List[CovarianceFinding]) -> None:
    from ..model_adapter.parameter_covariances import decodeMF32MT
    from ...nuclear_data.model import ConversionReport

    for mt, sec in sorted(mf_obj.mt.items()):
        mat = getattr(sec, "_mat", None) or getattr(sec, "mat", None)
        head = CovarianceLocation(mat=mat, mf=32, mt=mt)
        mf2_ranges = _mf2_ranges(ctx)

        readable = {}
        for i_iso, iso in enumerate(getattr(sec, "isotopes", []) or []):
            for k, rng in enumerate(iso.energy_ranges):
                loc = CovarianceLocation(mat=mat, mf=32, mt=mt, range_index=k)
                readable[(i_iso, k)] = _intg_indices(rng, loc, out)
                _spin_uncertainty(rng, loc, out)
                if mf2_ranges is not None:
                    _against_mf2(rng, i_iso, k, mf2_ranges, loc, out)

        # Decode range by range, so a range the decoder cannot read is located.
        for i_iso, iso in enumerate(getattr(sec, "isotopes", []) or []):
            for k, rng in enumerate(iso.energy_ranges):
                if rng.body is None or not readable[(i_iso, k)]:
                    continue
                loc = CovarianceLocation(mat=mat, mf=32, mt=mt, range_index=k)
                report = ConversionReport()
                try:
                    covs, report = decodeMF32MT(_one_range(sec, iso, rng), report)
                except (ValueError, IndexError) as exc:
                    out.append(CovarianceFinding(
                        "decode_error", DEFECT, f"kika cannot decode the range: {exc}", loc,
                        {"error": str(exc)}))
                    continue
                for msg in report.losses:
                    if msg.startswith("MF32/MT151: no parameter covariance decoded"):
                        continue
                    out.append(CovarianceFinding(
                        "decode_error", DEFECT, f"the decoder skipped this range: {msg}", loc,
                        {"error": msg}))
                for msg in report.unsupported:
                    out.append(CovarianceFinding(
                        "not_decoded", NOTE, f"kept verbatim but not checked: {msg}", loc,
                        {"reason": msg}))
                ndigit = _ndigit(rng)
                for j, cov in enumerate(covs):
                    bloc = CovarianceLocation(mat=mat, mf=32, mt=mt, range_index=k,
                                              block=j if len(covs) > 1 else None)
                    _check_form(cov.form, ndigit, rng, bloc, out)
                    if mf2_ranges is not None:
                        _energies_in_mf2(cov.form, rng, mf2_ranges, bloc, out)
                        _widths_in_mf2(cov.form, rng, mf2_ranges, bloc, out)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def _one_range(sec, iso, rng):
    """A shallow copy of the section holding one isotope with one range, for the decoder."""
    import copy

    one_iso = copy.copy(iso)
    one_iso.energy_ranges = [rng]
    one = copy.copy(sec)
    one.isotopes = [one_iso]
    return one


def _ndigit(rng) -> Optional[int]:
    corr = getattr(rng.body, "correlations", None)
    return int(corr.ndigit) if corr is not None else None


def _intg_indices(rng, loc, out) -> bool:
    """INTG records inside the NNN x NNN matrix. False (and a finding) if not."""
    corr = getattr(rng.body, "correlations", None)
    if corr is None:
        return True
    nnn = int(corr.nnn)
    bad = [(ii, jj) for ii, jj, _ in corr.entries
           if not 1 <= ii <= nnn or not 1 <= jj <= nnn]
    if bad:
        out.append(CovarianceFinding(
            "correlation_index_out_of_range", DEFECT,
            f"{len(bad)} INTG record(s) name a row or column outside the {nnn} x {nnn} matrix "
            f"(first II={bad[0][0]}, JJ={bad[0][1]}): the correlations they carry belong to no "
            "parameter", loc, {"nnn": nnn, "ii_jj": [list(b) for b in bad[:_ITEMS]],
                               "n": len(bad)}))
    return not bad


def _mf2_ranges(ctx):
    """[(isotope index, [EnergyRange, ...])] of MF2/MT151, or None if MF2 was not read."""
    mf2 = ctx.endf.files.get(2)
    if mf2 is None:
        return None
    sec = mf2.mt.get(151)
    if sec is None:
        return []
    return [list(getattr(iso, "energy_ranges", []) or []) for iso in getattr(sec, "isotopes", [])]


def _same_limits(a, b) -> bool:
    return (np.isclose(float(a.el), float(b.el), rtol=_E_RTOL, atol=0.0)
            and np.isclose(float(a.eh), float(b.eh), rtol=_E_RTOL, atol=0.0))


def _match(rng, ranges):
    """MF2's range for this one: the same limits, else one of the same LRU that overlaps."""
    for r in ranges:
        if _same_limits(r, rng):
            return r
    for r in ranges:
        if (int(r.lru) == int(rng.lru)
                and float(r.el) < float(rng.eh) and float(rng.el) < float(r.eh)):
            return r
    return None


def _against_mf2(rng, i_iso, k, mf2_ranges, loc, out) -> None:
    ranges = mf2_ranges[i_iso] if i_iso < len(mf2_ranges) else []
    twin = _match(rng, ranges)
    if twin is None:
        out.append(CovarianceFinding(
            "range_not_in_mf2", WARN,
            f"MF2 has no range of LRU={rng.lru} over {float(rng.el):.6g}-{float(rng.eh):.6g} eV: "
            "the covariance is of parameters File 2 does not give", loc,
            {"el": float(rng.el), "eh": float(rng.eh),
             "mf2_ranges": [[float(r.el), float(r.eh)] for r in ranges]}))
        return
    if not _same_limits(twin, rng):
        out.append(CovarianceFinding(
            "range_limits_differ_from_mf2", NOTE,
            f"the range is {float(rng.el):.6g}-{float(rng.eh):.6g} eV here and "
            f"{float(twin.el):.6g}-{float(twin.eh):.6g} eV in MF2 (ENDF-6 §32 asks for the "
            "same limits); its parameters are compared with that range", loc,
            {"mf32": [float(rng.el), float(rng.eh)], "mf2": [float(twin.el), float(twin.eh)]}))
    # §32.2.4 gives the unresolved range one covariance format whatever MF2's LRF
    # (evaluations write LRF=1 against an LRF=2 File 2), so only LRU is compared there.
    same = (int(twin.lru) == int(rng.lru)
            and (int(rng.lru) == 2 or int(twin.lrf) == int(rng.lrf)))
    if not same:
        out.append(CovarianceFinding(
            "formalism_mismatch", DEFECT,
            f"LRU={rng.lru} LRF={rng.lrf} here, LRU={twin.lru} LRF={twin.lrf} in MF2: the "
            "parameters the covariance is about are not the ones File 2 uses", loc,
            {"mf32": [int(rng.lru), int(rng.lrf)], "mf2": [int(twin.lru), int(twin.lrf)]}))


def _mf2_energies(twin) -> Optional[np.ndarray]:
    p = twin.parameters
    if p is None:
        return None
    if hasattr(p, "spin_groups"):
        return np.array([r.er for g in p.spin_groups for r in g.resonances], dtype=float)
    if hasattr(p, "l_values") and int(twin.lru) == 1:
        return np.array([r.energy for b in p.l_values for r in b.resonances], dtype=float)
    return None


def _energies_in_mf2(form, rng, mf2_ranges, loc, out) -> None:
    if int(rng.lru) != 1:
        return
    names = _names(form)
    values = np.asarray(form.parameterValues, dtype=float)
    er = values[[i for i, n in enumerate(names) if n == "ER"]]
    if er.size == 0:
        return
    twin = None
    for ranges in mf2_ranges:
        twin = _match(rng, ranges) or twin
    if twin is None or int(twin.lru) != int(rng.lru) or int(twin.lrf) != int(rng.lrf):
        return
    ref = _mf2_energies(twin)
    if ref is None or ref.size == 0:
        return
    ref = np.sort(ref)
    pos = np.clip(np.searchsorted(ref, er), 1, ref.size - 1) if ref.size > 1 else np.zeros(er.size, int)
    near = np.minimum(np.abs(ref[pos] - er), np.abs(ref[pos - 1] - er)) if ref.size > 1 \
        else np.abs(ref[0] - er)
    scale = np.maximum(np.abs(er), 1.0)
    missing = near > _E_RTOL * scale
    if missing.any():
        rel = near[missing] / scale[missing]
        out.append(CovarianceFinding(
            "parameters_not_in_mf2", WARN,
            f"{int(missing.sum())} of {er.size} resonance energies of the covariance are not in "
            f"MF2 (first {er[missing][0]:.6g} eV, nearest in MF2 off by {rel[0]:.2g}; up to "
            f"{rel.max():.2g}): the matrix is about parameter values File 2 does not use", loc,
            {"n": int(missing.sum()), "of": int(er.size),
             "energies": [float(x) for x in er[missing][:_ITEMS]],
             "max_relative_offset": float(rel.max())}))


#: Relative tolerance when comparing a width of MF32 with MF2's: both are
#: 11-column ENDF floats, so a copy agrees to the sixth figure.
_W_RTOL = 1e-5
#: A width that differs from MF2's by at least this fraction of its own sigma is
#: a warning; less, a note. JEFF-4.0 differs in 66 tapes, with a median of
#: 4e-4 of the width (census 8-oct-2026): copies with fewer digits, which move
#: nothing next to an uncertainty of percents.
_W_SIGMA_FRACTION = 0.1


def _mf2_rows(twin) -> Optional[List[Tuple[float, Dict[str, float]]]]:
    """[(ER, {slot name: value})] of an LRF=1/2/3 range of MF2, or None."""
    from ..model_adapter.parameter_covariances import _SLOTS

    slots = _SLOTS.get(int(twin.lrf))
    p = twin.parameters
    if slots is None or p is None or not hasattr(p, "l_values"):
        return None
    rows = []
    for b in p.l_values:
        for r in b.resonances:
            vals = (r.energy, r.spin, r.c3, r.c4, r.c5, r.c6)
            rows.append((float(r.energy), dict(zip(slots, map(float, vals)))))
    return rows


def _widths_in_mf2(form, rng, mf2_ranges, loc, out) -> None:
    """The widths MF32 states for a resonance against the ones MF2 has for it.

    LCOMP=0/1/2 repeat the File 2 parameters next to their covariances; FUDGE's
    converter compares (ER, GN, GG) and refuses a file where they differ. A
    resonance whose energy matches but whose widths do not is a covariance of a
    parameter set that is not the one the cross sections are built from. Several
    MF2 resonances may share an energy (different J), so the closest of them is
    the one compared. LRF=7 keeps its widths per channel and is not compared.
    """
    if int(rng.lru) != 1 or int(rng.lrf) not in (1, 2, 3):
        return
    if int(rng.lrf) == 3 and type(rng.body).__name__ == "LCOMP0Body":
        return  # LCOMP=0 reads GN/GG/GF from the LRF=1/2 slots (§32.2.1)
    twin = None
    for ranges in mf2_ranges:
        twin = _match(rng, ranges) or twin
    if twin is None or int(twin.lru) != 1 or int(twin.lrf) != int(rng.lrf):
        return
    rows = _mf2_rows(twin)
    if not rows:
        return
    names, labels = _names(form), _labels(form)
    values = np.asarray(form.parameterValues, dtype=float)
    by_energy: Dict[float, List[Dict[str, float]]] = {}
    for er, row in rows:
        by_energy.setdefault(er, []).append(row)
    ref = np.array(sorted(by_energy))
    # One resonance of the covariance = consecutive rows with the same label.
    groups: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels[:values.size]):
        groups.setdefault(lab, []).append(i)
    n_res = n_bad = n_matter = 0
    worst = (0.0, "", 0.0)
    worst_sigmas = 0.0
    examples = []
    diag = np.diag(np.asarray(form.matrix, dtype=float))
    for idx in groups.values():
        mine = {names[i]: float(values[i]) for i in idx}
        sigma = {names[i]: float(np.sqrt(max(diag[i], 0.0))) for i in idx if i < diag.size}
        er = mine.get("ER")
        widths = [n for n in mine if n not in ("ER", "AJ", "?")]
        if er is None or not widths:
            continue
        j = int(np.argmin(np.abs(ref - er)))
        if abs(ref[j] - er) > _E_RTOL * max(abs(er), 1.0):
            continue  # parameters_not_in_mf2 reports it
        n_res += 1
        best = None
        for cand in by_energy[float(ref[j])]:
            dev = max(abs(mine[n] - cand[n]) / max(abs(cand[n]), abs(mine[n]), 1e-30)
                      for n in widths if n in cand)
            if best is None or dev < best[0]:
                best = (dev, cand)
        if best is None or best[0] <= _W_RTOL:
            continue
        n_bad += 1
        dev, cand = best
        name = max((n for n in widths if n in cand),
                   key=lambda n: abs(mine[n] - cand[n]) / max(abs(cand[n]), abs(mine[n]), 1e-30))
        # How many of its own sigmas apart; a width with no stated sigma counts in full.
        sig = max((abs(mine[n] - cand[n]) / sigma[n] if sigma.get(n, 0) > 0 else np.inf)
                  for n in widths if n in cand and abs(mine[n] - cand[n]) > 0)
        worst_sigmas = max(worst_sigmas, sig)
        n_matter += sig >= _W_SIGMA_FRACTION
        if dev > worst[0]:
            worst = (dev, name, er)
        if len(examples) < _ITEMS:
            examples.append({"er": er, "parameter": name, "mf32": mine[name], "mf2": cand[name]})
    if n_bad:
        matters = n_matter > 0
        out.append(CovarianceFinding(
            "widths_differ_from_mf2", WARN if matters else NOTE,
            f"{n_bad} of {n_res} resonances have widths that differ from MF2's (worst {worst[1]} "
            f"at {worst[2]:.6g} eV, off by {worst[0]:.2g} of its value; "
            + (f"{n_matter} by {_W_SIGMA_FRACTION:g} sigma or more): the covariance is of a "
               "parameter set the cross sections are not built from" if matters else
               f"all within {_W_SIGMA_FRACTION:g} of their sigma): a copy with fewer digits, "
               "which changes nothing the covariance says"), loc,
            {"n": n_bad, "of": n_res, "max_relative_difference": worst[0],
             "n_beyond_sigma_fraction": n_matter,
             "max_in_sigmas": None if not np.isfinite(worst_sigmas) else worst_sigmas,
             "examples": examples}))


def _spin_uncertainty(rng, loc, out) -> None:
    """An uncertainty on the resonance spin J, which is a quantum number.

    LCOMP=0 carries four J terms per resonance (DJDN, DJDG, DJDF, DJ2) that
    §32.2.1 calls null and §32.3 (Other procedures, 2) says to treat as null if
    they are not; LCOMP=2 has a slot under AJ where §32.2.3 gives 0.0 "instead
    of DAJ". kika's decoder drops both (FUDGE's refuses the file): a warning,
    since what the evaluator meant by a non-zero one is lost.
    """
    body = rng.body
    kind = type(body).__name__
    hits = 0
    if kind == "LCOMP0Body":
        for block in body.l_blocks:
            count = int(block.n2)
            raw = np.asarray(block.values[:18 * count], dtype=float)
            if raw.size == 18 * count and count:
                hits += int(np.count_nonzero(np.any(raw.reshape(count, 18)[:, 13:17] != 0,
                                                    axis=1)))
    elif kind == "LCOMP2Body" and body.parameters is not None and int(rng.lrf) in (1, 2, 3):
        count = int(body.parameters.n2)
        raw = np.asarray(body.parameters.values[:12 * count], dtype=float)
        if raw.size == 12 * count and count:
            hits = int(np.count_nonzero(raw.reshape(count, 12)[:, 7]))
    if hits:
        out.append(CovarianceFinding(
            "spin_uncertainty", WARN,
            f"{hits} resonance(s) state an uncertainty on the spin J, a quantum number; "
            "ENDF-6 §32 has J carry none, and kika drops it", loc, {"n": hits}))


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _names(form) -> List[str]:
    names: List[str] = []
    for link in form.parameters:
        start, width = int(link.matrixStartIndex), int(link.nParameters)
        pn = list(link.parameterNames or [])
        names.extend((pn + ["?"] * width)[:width])
    return names


def _labels(form) -> List[str]:
    labels: List[str] = []
    for link in form.parameters:
        labels.extend([link.label] * int(link.nParameters))
    return labels


def _where(labels, names, i) -> str:
    return f"{labels[i]}/{names[i]}" if i < len(labels) and i < len(names) else f"row {i}"


def _check_form(form, ndigit, rng, loc, out) -> None:
    from .covariances import (ASYM_DEFECT, ASYM_NOTE, PSD_DEFECT, PSD_NOTE, RHO_DEFECT,
                              RHO_QUANTUM_SLACK, RHO_ROUNDING, _half_ulp_norm)

    matrix = np.asarray(form.matrix, dtype=float)
    n = matrix.shape[0]
    if n == 0:
        return
    names, labels = _names(form), _labels(form)
    d = np.diag(matrix).copy()

    neg = np.flatnonzero(d < 0)
    if neg.size:
        k = int(neg[np.argmin(d[neg])])
        out.append(CovarianceFinding(
            "negative_variance", DEFECT,
            f"{neg.size} of {n} variances are negative (worst {d[k]:.3e}, "
            f"{_where(labels, names, k)})", loc,
            {"n": int(neg.size), "worst": float(d[k]),
             "parameters": [_where(labels, names, int(i)) for i in neg[:_ITEMS]]}))

    zero_rows = ~np.any(matrix != 0, axis=1)
    if zero_rows.any():
        idx = np.flatnonzero(zero_rows)
        out.append(CovarianceFinding(
            "inert_rows", NOTE, f"{idx.size} of {n} parameters have no stated uncertainty",
            loc, {"n": int(idx.size), "of": n,
                  "parameters": [_where(labels, names, int(i)) for i in idx[:_ITEMS]]}))
    lonely = np.flatnonzero((d == 0) & ~zero_rows)
    if lonely.size:
        out.append(CovarianceFinding(
            "covariance_without_variance", DEFECT,
            f"{lonely.size} parameters have zero variance but non-zero covariances", loc,
            {"n": int(lonely.size),
             "parameters": [_where(labels, names, int(i)) for i in lonely[:_ITEMS]]}))

    scale = float(np.max(np.abs(matrix)))
    if scale > 0:
        asym = float(np.max(np.abs(matrix - matrix.T))) / scale
        if asym > ASYM_NOTE:
            out.append(CovarianceFinding(
                "asymmetric_self_block", DEFECT if asym > ASYM_DEFECT else NOTE,
                f"the matrix is not symmetric: max|C - C^T| / max|C| = {asym:.2e}", loc,
                {"relative_asymmetry": asym}))

    live = np.flatnonzero(d > 0)
    if live.size >= 2:
        sub = matrix[np.ix_(live, live)]
        sub = 0.5 * (sub + sub.T)
        s = np.sqrt(np.diag(sub))
        corr = sub / np.outer(s, s)
        a = np.abs(corr)
        np.fill_diagonal(a, 0.0)
        worst = float(a.max())
        if worst > 1 + RHO_ROUNDING:
            i, j = np.unravel_index(int(np.argmax(a)), a.shape)
            level = DEFECT if worst > 1 + RHO_DEFECT else NOTE
            n_bad = int(np.sum(np.triu(a, 1) > 1 + RHO_DEFECT))
            out.append(CovarianceFinding(
                "correlation_out_of_bounds", level,
                (f"|rho| up to {worst:.4g} ({n_bad} pairs above 1+{RHO_DEFECT:g})"
                 if level == DEFECT else
                 f"|rho| up to {worst:.7g}, within rounding of the stored values"), loc,
                {"max_abs_rho": worst, "n_above": n_bad,
                 "at": [_where(labels, names, int(live[i])), _where(labels, names, int(live[j]))]}))
        del a
        if live.size > EIG_MAX:
            out.append(CovarianceFinding(
                "psd_not_evaluated", NOTE,
                f"{live.size} parameters with an uncertainty: above {EIG_MAX}, the eigenvalues "
                "are not computed", loc, {"n": int(live.size)}))
        else:
            np.fill_diagonal(corr, 1.0)
            _psd_on_correlation(corr, ndigit, rng, loc, out,
                                PSD_NOTE, PSD_DEFECT, RHO_QUANTUM_SLACK, _half_ulp_norm)

    _magnitude(form, names, labels, d, loc, out)


def _psd_on_correlation(corr, ndigit, rng, loc, out, psd_note, psd_defect, slack,
                        half_ulp_norm) -> None:
    from .covariances import _clipping_impact, _grade_by_impact

    ev = np.linalg.eigvalsh(corr)
    lam_min, lam_max = float(ev[0]), float(ev[-1])
    if lam_min >= 0:
        return
    ratio = -lam_min / lam_max
    level = NOTE if ratio < psd_note else (WARN if ratio <= psd_defect else DEFECT)
    evidence = {"lambda_min": lam_min, "lambda_max": lam_max, "ratio": ratio,
                "n_negative": int(np.sum(ev < 0)), "on": "correlation matrix",
                "sigma_change_if_clipped": _clipping_impact(*np.linalg.eigh(corr), corr)}
    reason = ""
    if ndigit:
        # INTG keeps NDIGIT digits: each stored rho is off by up to q/2, and one that
        # rounds to zero is dropped. As for quantised MF33 records, the spectral norm
        # of that error is about 2 (q / sqrt 12) sqrt(n).
        q = 10.0 ** (-ndigit)
        bound = slack * 2.0 * (q / math.sqrt(12.0)) * math.sqrt(corr.shape[0])
        evidence.update({"rho_quantum": q, "quantised_rounding_bound": bound})
        if -lam_min <= bound:
            level = NOTE
            reason = (f"the correlations are stored with NDIGIT={ndigit} and that rounding can "
                      f"reach lambda_min (bound {bound:.2e})")
    else:
        ulp = half_ulp_norm(corr)
        evidence["rounding_bound"] = ulp
        if -lam_min <= ulp:
            level, reason = NOTE, "within the rounding of the values to 6 figures"
    level, reason = _grade_by_impact(level, reason, evidence["sigma_change_if_clipped"])
    summary = (f"correlation matrix: lambda_min/lambda_max = -{ratio:.2e} "
               f"(lambda_min {lam_min:.3g}, lambda_max {lam_max:.3g})")
    if reason:
        summary += f"; {reason}"
    out.append(CovarianceFinding("not_positive_semidefinite", level, summary, loc, evidence))


def _magnitude(form, names, labels, d, loc, out) -> None:
    sigma = np.sqrt(np.clip(d, 0.0, None))
    values = np.asarray(form.parameterValues, dtype=float)
    if values.size != sigma.size:
        return
    widths = np.array([nm not in ("ER", "AJ", "?") for nm in names[:sigma.size]])
    if form.isRelative:
        rel = sigma
        ok = widths & (sigma > 0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(values != 0, sigma / np.abs(values), np.inf)
        ok = widths & (sigma > 0)
    above = ok & (rel > 1.0)
    if not above.any():
        return
    idx = np.flatnonzero(above)
    finite = idx[np.isfinite(rel[idx])]
    k = int(finite[np.argmax(rel[finite])]) if finite.size else int(idx[0])
    by_name: Dict[str, int] = {}
    for i in idx:
        by_name[names[i]] = by_name.get(names[i], 0) + 1
    zero = int(np.sum(~np.isfinite(rel[idx])))
    out.append(CovarianceFinding(
        "relative_uncertainty_above_one", NOTE,
        f"sigma_rel above 100 % for {idx.size} widths ("
        + ", ".join(f"{v} {nm}" for nm, v in sorted(by_name.items()))
        + f"; worst {rel[k]:.3g}, {_where(labels, names, k)})"
        + (f", {zero} of them with a central value of 0" if zero else "")
        + "; widths are unbounded above and, in LRF=3/7, signed, so this is not a fault", loc,
        {"n": int(idx.size), "by_parameter": by_name, "worst_rel": float(rel[k]),
         "worst": _where(labels, names, k), "n_zero_central": zero}))

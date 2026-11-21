"""The HTML report of a spectrum adjustment: one page, with the report style of kika's other reports."""
from __future__ import annotations

import html
from typing import List, Tuple

import numpy as np

__all__ = ["CHI2_PER_DOF_WINDOW", "adjustment_html"]

#: Perey's (ORNL/TM-6062) bounds on chi2 per degree of freedom outside which the inputs are suspect.
CHI2_PER_DOF_WINDOW = (0.3, 2.0)

#: A group whose posterior standard deviation keeps this fraction of the prior's is reported as unseen.
UNSEEN_FRACTION = 0.99

_THRESHOLDS_EV = (1.0e5, 1.0e6)

_EXTRA_CSS = """
.kika-report svg.spectrum{width:100%;height:auto;max-width:760px;display:block;margin:.5em 0}
.kika-report svg.spectrum .prior{fill:none;stroke:var(--faint);stroke-width:1.5}
.kika-report svg.spectrum .post{fill:none;stroke:var(--accent);stroke-width:1.5}
.kika-report svg.spectrum .band{fill:var(--accent);opacity:.18;stroke:none}
.kika-report svg.spectrum text{font-size:11px;fill:var(--muted)}
.kika-report svg.spectrum .axis{stroke:var(--line)}
"""


def _esc(text) -> str:
    return html.escape(str(text))


def _num(value: float) -> str:
    return f"{value:.4g}"


def _pct(value: float) -> str:
    return f"{100 * value:.2g} %"


def _table(head: List[Tuple[str, bool]], rows: List[str]) -> str:
    th = "".join(f'<th class="n">{_esc(t)}</th>' if numeric else f"<th>{_esc(t)}</th>" for t, numeric in head)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _verdict(result) -> Tuple[str, str, str]:
    per_dof = result.chi2 / result.dof
    lo, hi = CHI2_PER_DOF_WINDOW
    if lo <= per_dof <= hi:
        return ("pass", f"Consistent: chi2/dof = {per_dof:.3g}",
                f"The measured rates agree with the prior within the stated covariances "
                f"(Perey's window is {lo} to {hi}).")
    worst = result.names[int(np.argmax(result.chi2_drop))]
    side = "too large" if per_dof > hi else "too small"
    return ("warn", f"Check the inputs: chi2/dof = {per_dof:.3g}",
            f"chi2/dof is {side} for Perey's window of {lo} to {hi}: some covariance is "
            f"understated or overstated, or a measurement or the model is wrong. The dosimeter "
            f"that weighs most on chi2 is {worst}.")


def _dosimeter_rows(result) -> List[str]:
    rows = []
    for i, name in enumerate(result.names):
        before = result.measured_rates[i] / result.prior_rates[i]
        after = result.measured_rates[i] / result.adjusted_rates[i]
        z = result.standardized_residuals[i]
        hot = ' class="n hot"' if abs(z) > 2 else ' class="n"'
        rows.append(f"<tr><td>{_esc(name)}</td><td class=\"n\">{_num(result.measured_rates[i])}</td>"
                    f"<td class=\"n\">{_num(result.prior_rates[i])}</td><td class=\"n\">{before:.4f}</td>"
                    f"<td class=\"n\">{_num(result.adjusted_rates[i])}</td><td class=\"n\">{after:.4f}</td>"
                    f"<td{hot}>{z:+.2f}</td><td class=\"n\">{result.chi2_drop[i]:.3g}</td></tr>")
    return rows


def _integral_rows(result) -> List[str]:
    rows = []
    cases = [("All groups", np.ones(result.flux.size))]
    for e in _THRESHOLDS_EV:
        where = np.flatnonzero(np.isclose(result.edges, e, rtol=1e-9, atol=0.0))
        if where.size == 1 and where[0] < result.flux.size:
            cases.append((f"Above {e / 1e6:g} MeV", (np.arange(result.flux.size) >= where[0]).astype(float)))
    for label, w in cases:
        p, sp = result.prior_integral(w)
        a, sa = result.integral(w)
        if p <= 0:
            continue
        rows.append(f"<tr><td>{_esc(label)}</td><td class=\"n\">{_num(p)}</td><td class=\"n\">{_pct(sp / p)}</td>"
                    f"<td class=\"n\">{_num(a)}</td><td class=\"n\">{_pct(sa / a)}</td>"
                    f"<td class=\"n\">{a / p:.4f}</td></tr>")
    return rows


def _energy_label(e: float) -> str:
    for scale, unit in ((1e6, "MeV"), (1e3, "keV"), (1.0, "eV")):
        if e >= scale:
            return f"{e / scale:g} {unit}"
    return f"{e:g} eV"


def _step_svg(lo, hi, lines, band, log_y: bool, caption: str) -> str:
    """Group values as steps on a log energy axis; *lines* are ``(css class, values)``, *band* ``(low, high)``."""
    values = np.concatenate([v for _, v in lines] + list(band))
    transform = np.log10 if log_y else (lambda v: np.asarray(v, dtype=float))
    y0, y1 = float(np.min(transform(values))), float(np.max(transform(values)))
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    x0, x1 = np.log10(lo.min()), np.log10(hi.max())
    W, H, L, B = 760, 260, 60, 30

    def px(e):
        return L + (np.log10(e) - x0) / (x1 - x0) * (W - L - 10)

    def py(v):
        return H - B - (transform(v) - y0) / (y1 - y0) * (H - B - 24)

    def steps(values):
        return " ".join(f"{px(a):.1f},{py(v):.1f} {px(b):.1f},{py(v):.1f}" for a, b, v in zip(lo, hi, values))

    upper = steps(band[1])
    lower = " ".join(f"{px(b):.1f},{py(v):.1f} {px(a):.1f},{py(v):.1f}"
                     for a, b, v in zip(lo[::-1], hi[::-1], band[0][::-1]))
    ticks = "".join(
        f'<line class="axis" x1="{px(10.0 ** k):.1f}" y1="{H - B}" x2="{px(10.0 ** k):.1f}" y2="20"/>'
        f'<text x="{px(10.0 ** k):.1f}" y="{H - 10}" text-anchor="middle">{_energy_label(10.0 ** k)}</text>'
        for k in range(int(np.ceil(x0)), int(np.floor(x1)) + 1))
    if not log_y and y0 < 1.0 < y1:
        ticks += f'<line class="axis" x1="{L}" y1="{py(1.0):.1f}" x2="{W - 10}" y2="{py(1.0):.1f}"/>'
    for level in (y0, y1):
        value = 10.0 ** level if log_y else level
        ticks += f'<text x="{L - 4}" y="{py(value) + 4:.1f}" text-anchor="end">{value:.3g}</text>'
    curves = "".join(f'<polyline class="{css}" points="{steps(v)}"/>' for css, v in lines)
    return (f'<svg class="spectrum" viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(caption)}">{ticks}'
            f'<polygon class="band" points="{upper} {lower}"/>{curves}'
            f'<text x="{L}" y="14">{_esc(caption)}</text></svg>')


def _spectrum_svg(result) -> str:
    keep = result.prior_flux > 0
    lo, hi = result.edges[:-1][keep], result.edges[1:][keep]
    width = np.log(hi / lo)
    prior = result.prior_flux[keep] / width
    post = result.flux[keep] / width
    sd = np.sqrt(np.clip(np.diag(result.log_covariance)[keep], 0.0, None))
    prior_sd = np.sqrt(np.clip(np.diag(result.prior_log_covariance)[keep], 0.0, None))
    ratio = post / prior
    spectrum = _step_svg(lo, hi, [("prior", prior), ("post", post)], (post * np.exp(-sd), post * np.exp(sd)), True,
                         "Flux per unit lethargy (log scale): grey prior, blue adjusted with its ±1σ band")
    change = _step_svg(lo, hi, [("prior", np.exp(prior_sd)), ("prior", np.exp(-prior_sd)), ("post", ratio)],
                       (ratio * np.exp(-sd), ratio * np.exp(sd)), False,
                       "Adjusted / prior with its ±1σ band; grey lines: the prior's ±1σ")
    return spectrum + change


def _cross_section_rows(result) -> List[str]:
    rows = []
    for name, factor in result.cross_section_factors.items():
        sd = np.sqrt(np.clip(np.diag(result.cross_section_log_covariance[name]), 0.0, None))
        change = np.max(np.abs(factor - 1.0))
        rows.append(f"<tr><td>{_esc(name)}</td><td class=\"n\">{factor.size}</td>"
                    f"<td class=\"n\">{_pct(change)}</td><td class=\"n\">{_pct(np.max(sd))}</td></tr>")
    return rows


def _unseen(result) -> str:
    keep = result.prior_flux > 0
    prior_sd = np.sqrt(np.clip(np.diag(result.prior_log_covariance), 0.0, None))
    post_sd = np.sqrt(np.clip(np.diag(result.log_covariance), 0.0, None))
    unseen = keep & (prior_sd > 0) & (post_sd >= UNSEEN_FRACTION * prior_sd)
    if not unseen.any():
        return "<p>Every group with prior flux is constrained by at least one dosimeter.</p>"
    index = np.flatnonzero(unseen)
    lo, hi = result.edges[index[0]], result.edges[index[-1] + 1]
    return (f"<p>{index.size} of {int(keep.sum())} groups keep at least {UNSEEN_FRACTION:.0%} of their prior "
            f"uncertainty, between {lo:.4g} and {hi:.4g} eV: no dosimeter sees them, and their adjusted "
            f"values are the prior's, moved only through its correlations.</p>")


def adjustment_html(result) -> str:
    """The report of *result* (a :class:`~kika.dosimetry.SpectrumAdjustment`) as one HTML page.

    Parameters
    ----------
    result : SpectrumAdjustment
        What :func:`~kika.dosimetry.adjust_spectrum` returned.

    Returns
    -------
    str
        A complete HTML document.
    """
    from kika.endf.reconstruction_report import _CSS

    status, headline, lead = _verdict(result)
    mark = {"pass": "✓", "warn": "!"}[status]
    one_step = np.max(np.abs(np.log(result.flux[result.prior_flux > 0] / result.one_step_flux[result.prior_flux > 0])))
    notes = "".join(f"<li>{_esc(n)}</li>" for n in result.notes)
    parts = [f'<h2 id="ka-notes">Notes</h2><ul class="notes">{notes}</ul>'] if notes else []
    parts += [
        '<h2 id="ka-dosimeters">Dosimeters</h2><p class="lead">M/C is the measured rate over the rate '
        'computed from the spectrum, before and after the adjustment. The pull is the residual of the '
        'measurement against the prior, linearised at the result, over its predictive standard '
        'deviation (flux, cross section and measurement), net of the fitted factor when the '
        'normalization is free; '
        'Δχ² is what chi2 loses without that dosimeter.</p>'
        + _table([("Dosimeter", False), ("Measured", True), ("Prior rate", True), ("M/C before", True),
                  ("Adjusted rate", True), ("M/C after", True), ("Pull", True), ("Δχ²", True)],
                 _dosimeter_rows(result)),
        '<h2 id="ka-integrals">Integrals</h2>'
        + _table([("Flux", False), ("Prior", True), ("Prior 1σ", True), ("Adjusted", True),
                  ("Adjusted 1σ", True), ("Adjusted / prior", True)], _integral_rows(result)),
        f'<h2 id="ka-spectrum">Spectrum</h2>{_spectrum_svg(result)}{_unseen(result)}',
    ]
    if result.cross_section_factors:
        parts.append('<h2 id="ka-xs">Cross sections</h2><p class="lead">The cross sections are adjusted '
                     'with the flux, on the intervals of their own covariance.</p>'
                     + _table([("Reaction", False), ("Intervals", True), ("Largest change", True),
                               ("Largest posterior 1σ", True)], _cross_section_rows(result)))
    parts.append(f'<h2 id="ka-method">Method</h2><p>Generalized least squares on ln φ and ln σ, iterated '
                 f'(Gauss-Newton) to the maximum of the posterior: {result.iterations} iterations. The '
                 f'first iteration alone, the linear one-step adjustment, differs from the result by '
                 f'{one_step:.2g} in ln φ at most. Adjusted values are posterior medians; '
                 f'chi2 = {result.chi2:.4g} with {result.dof} degrees of freedom.</p>')
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("ka-dosimeters", "Dosimeters"), ("ka-integrals", "Integrals"),
                                                       ("ka-spectrum", "Spectrum"), ("ka-method", "Method")))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark"><title>Spectrum adjustment</title></head>'
            f'<body style="margin:0;background:Canvas"><style>{_CSS}{_EXTRA_CSS}</style><div class="kika-report">'
            '<p class="eyebrow">kika · spectrum adjustment</p>'
            f'<h1>Spectrum adjustment <span class="muted">· {len(result.names)} dosimeters</span></h1>'
            f'<div class="verdict {status}"><span class="mark">{mark}</span><div><p class="big">{_esc(headline)}</p>'
            f'<p>{_esc(lead)}</p></div></div><nav class="toc">{nav}</nav>' + "".join(parts) + "</div></body></html>\n")

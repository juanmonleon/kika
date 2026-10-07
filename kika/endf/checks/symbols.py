"""Greek letters and math signs for the texts people read.

The checks write their summaries, and :mod:`.descriptions` its legend, in
ASCII (``|rho|``, ``lambda_min``, ``sigma_rel``, ``<=``): that is what
``__str__``, ``to_dataframe`` and the TSV of ``write`` give, because a Windows
console (cp1252) and Excel opening a TSV both mangle anything else. The
outputs meant to be read -- ``to_dict``, ``to_markdown``, ``to_html`` -- pass
the same texts through :func:`to_symbols` here. One spelling at the source,
one conversion, and the two can never disagree.

Only whole words are replaced: an evidence key quoted in a summary
(``max_abs_rho``, ``sigma_change_if_clipped``) is left as it is.
"""
from __future__ import annotations

import html
import re
from typing import List, Tuple

__all__ = ["to_symbols", "to_html_symbols"]

# (ASCII, symbol). The subscripted forms are listed so they keep their
# subscript; a bare word comes after them.
_WORDS: List[Tuple[str, str]] = [
    ("sigma-bar", "σ̄"), ("sigma_bar", "σ̄"), ("nu-bar", "ν̄"),
    ("sigma_rel", "σ_rel"), ("sigma_abs", "σ_abs"), ("sigma_i", "σ_i"), ("sigma_j", "σ_j"),
    ("lambda_min", "λ_min"), ("lambda_max", "λ_max"), ("lambda_k", "λ_k"),
    ("sigma", "σ"), ("lambda", "λ"), ("rho", "ρ"), ("mu", "μ"), ("chi", "χ"),
]
_WORD_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(a) for a, _ in _WORDS) + r")(?![\w])")
_WORD_MAP = dict(_WORDS)

_SIGNS = [("<=", "≤"), (">=", "≥"), ("!=", "≠"), ("+-", "±")]


def to_symbols(text: str) -> str:
    """*text* with Greek letters and math signs in place of their ASCII spelling."""
    if not text:
        return text
    out = _WORD_RE.sub(lambda m: _WORD_MAP[m.group(1)], text)
    for ascii_, sign in _SIGNS:
        out = out.replace(ascii_, sign)
    return out


_SUB_RE = re.compile(r"(?<![\w])([σλa])_(rel|abs|min|max|l1|[ijkl0])\b")


def to_html_symbols(text: str) -> str:
    """:func:`to_symbols`, HTML-escaped, with ``σ_rel``, ``λ_min``, ``a_l``… as subscripts."""
    return _SUB_RE.sub(r"\1<sub>\2</sub>", html.escape(to_symbols(text)))

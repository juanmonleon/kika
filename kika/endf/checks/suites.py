"""The families of checks a report can hold, and what the pages call each.

One renderer (:mod:`.export`) draws every report; a report's ``kind`` picks
its title, the meaning written under each level, the method section and the
JSON ``kind``. Check ids are unique across families, so the legend of any
report reads from one table (:func:`described`).

* ``"covariance"`` -- :func:`~kika.endf.checks.check_covariances` (MF31-35, MF40).
* ``"data"`` -- :func:`~kika.endf.checks.data.check_data`: everything else.
* ``"evaluation"`` -- both, as :func:`kika.check` runs them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .findings import DEFECT, NOTE, WARN

__all__ = ["Suite", "suite", "described", "all_checks", "KINDS"]


@dataclass(frozen=True)
class Suite:
    kind: str
    #: "Covariance check": the page title is "<title> of <tape>".
    title: str
    #: The small line above the title, for one tape.
    eyebrow: str
    json_kind: str
    json_library_kind: str
    #: How a tape with nothing this family checks is shown ("No covariances").
    none_word: str
    #: "with covariances checked", in the library status line.
    checked_phrase: str
    #: One line under each count card.
    means: Dict[str, str]
    method: Callable[[], List[Tuple[str, str, list]]]


def _covariance_method():
    from .export import method_sections

    return method_sections()


def _data_method():
    from .data.method import method_sections

    return method_sections()


def _evaluation_method():
    data = _data_method()
    cov = [(f"Covariances: {heading}", intro, rules) for heading, intro, rules in _covariance_method()]
    return data + cov


_COV_MEANS = {
    DEFECT: "Not what ENDF-6 says, or not a covariance",
    WARN: "Probably a fault, or one in some uses",
    NOTE: "True and worth knowing, not a fault",
}
_DATA_MEANS = {
    DEFECT: "Not what ENDF-6 or physics allows",
    WARN: "Can change a result; needs judgement",
    NOTE: "True and worth knowing, not a fault",
}

_SUITES = {
    "covariance": Suite("covariance", "Covariance check", "kika · covariance check",
                        "kika.covariance_check", "kika.covariance_library_check",
                        "No covariances", "with covariances checked", _COV_MEANS,
                        _covariance_method),
    "data": Suite("data", "Data check", "kika · data check",
                  "kika.data_check", "kika.data_library_check",
                  "Nothing to check", "checked", _DATA_MEANS, _data_method),
    "evaluation": Suite("evaluation", "Evaluation check", "kika · evaluation check",
                        "kika.evaluation_check", "kika.evaluation_library_check",
                        "Nothing to check", "checked", _DATA_MEANS, _evaluation_method),
}
KINDS = tuple(_SUITES)


def suite(kind: Optional[str]) -> Suite:
    """The :class:`Suite` of *kind*; a report without one is a covariance report."""
    return _SUITES[kind or "covariance"]


_ALL = None


def all_checks() -> Dict[str, object]:
    """Every check of every family, by id: ``{check: CheckDescription}``."""
    global _ALL
    if _ALL is None:
        from .data.descriptions import DATA_CHECKS
        from .descriptions import CHECKS

        clash = set(CHECKS) & set(DATA_CHECKS)
        if clash:  # a test keeps this from ever happening; say which if it does
            raise RuntimeError(f"check ids used by both families: {sorted(clash)}")
        _ALL = {**CHECKS, **DATA_CHECKS}
    return _ALL


def described(check: str):
    """The description of *check*, from whichever family it belongs to; ``None`` if it has none."""
    return all_checks().get(check)

"""One evaluation, every check: the data checks and the covariance checks together.

:func:`check_evaluation` is what :func:`kika.check` runs on a file: the data
checks (:func:`~kika.endf.checks.data.check_data`) and the covariance checks
(:func:`~kika.endf.checks.check_covariances`) on one read of the tape, merged
into one report of kind ``"evaluation"``. Either part can be run alone
(``what=("data",)``, ``what=("covariances",)``), and ``mf=`` narrows both to
some files -- the report's kind then says which family it holds.

The tape is read once and lazily; only the files the asked checks need are
parsed, so ``mf=[3]`` on a JENDL-5 tape costs MF3, not its 16 MB of MF6.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .findings import CheckReport

__all__ = ["check_evaluation", "check_evaluation_file", "WHAT", "normalise_what", "kind_of"]

#: The families :func:`check_evaluation` can run.
WHAT = ("data", "covariances")
_ALIASES = {"data": "data", "covariances": "covariances", "covariance": "covariances",
            "cov": "covariances", "all": None, "evaluation": None}


def normalise_what(what: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
    """``("data", "covariances")`` from what a caller wrote (``"all"``, ``"cov"``, a list...)."""
    if what is None:
        return WHAT
    items = [what] if isinstance(what, str) else list(what)
    out = []
    for item in items:
        key = str(item).lower()
        if key not in _ALIASES:
            raise ValueError(f"what= takes 'data', 'covariances' or 'all', got {item!r}")
        if _ALIASES[key] is None:
            return WHAT
        if _ALIASES[key] not in out:
            out.append(_ALIASES[key])
    if not out:
        raise ValueError("what= names no family of checks")
    return tuple(w for w in WHAT if w in out)


def kind_of(what: Sequence[str]) -> str:
    """The report kind of a run of *what*: ``"data"``, ``"covariance"`` or ``"evaluation"``."""
    what = tuple(what)
    if what == ("data",):
        return "data"
    if what == ("covariances",):
        return "covariance"
    return "evaluation"


def _plan(present, what, mf):
    """(data MF, covariance MF, MF to read): what each family checks and what must be parsed."""
    from .covariances import CHECKED_MF
    from .data import DATA_MF
    from .library import _SUPPORT_MF

    asked = None if mf is None else {int(m) for m in mf}
    data_mf = tuple(m for m in DATA_MF if m in present and (asked is None or m in asked)) \
        if "data" in what else ()
    cov_mf = tuple(m for m in CHECKED_MF if m in present and (asked is None or m in asked)) \
        if "covariances" in what else ()
    read = set(data_mf) | set(cov_mf)
    for m in cov_mf:
        read |= set(_SUPPORT_MF[m])
    return data_mf, cov_mf, sorted(m for m in read if m in present)


def check_evaluation(source, *, what=WHAT, mf: Optional[Sequence[int]] = None) -> CheckReport:
    """Run the data checks, the covariance checks or both on one evaluation.

    Parameters
    ----------
    source : path or ENDF
        An ENDF-6 tape on disk, or one already read.
    what : str or sequence
        ``"data"``, ``"covariances"`` or both (the default; ``"all"`` too).
    mf : sequence of int, optional
        Only these files: ``mf=[3, 33]`` checks MF3's data and MF33's
        covariances, and reads what those are compared against.

    Returns
    -------
    CheckReport
        Of kind ``"evaluation"``, or ``"data"``/``"covariance"`` when only one
        family was asked for.
    """
    from kika.endf import open_endf

    from .covariances import check_covariances
    from .data import check_data

    what = normalise_what(what)
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        endf = open_endf(path)
    else:
        endf, path = source, getattr(source, "source_path", None)
    data_mf, cov_mf, _ = _plan(set(endf.files), what, mf)
    findings = ()
    mat = getattr(endf, "mat", None)
    if data_mf:
        rep = check_data(endf, mf=data_mf)
        findings += rep.findings
        mat = rep.mat if rep.mat is not None else mat
    if cov_mf:
        rep = check_covariances(endf, mf=cov_mf)
        findings += rep.findings
        mat = mat if mat is not None else rep.mat
    return CheckReport(findings, source=path, mat=mat, mf=tuple(sorted(set(data_mf) | set(cov_mf))),
                       kind=kind_of(what))


@dataclass(frozen=True)
class TapeResult:
    """What :func:`check_evaluation_file` made of one tape: a report, or why none."""

    report: Optional[CheckReport]
    error: Optional[str]
    has_anything: bool
    read_seconds: float
    check_seconds: float
    mf: Tuple[int, ...]


def check_evaluation_file(path, *, what=WHAT, mf: Optional[Sequence[int]] = None) -> TapeResult:
    """:func:`check_evaluation` on a file, for a library walk: never raises.

    Reading (the index and every file the checks need, parsed up front so the
    timings say where the time went) and checking are timed apart; a failure of
    either is the result's ``error``.
    """
    from kika.endf import open_endf

    from .covariances import check_covariances
    from .data import check_data

    what = normalise_what(what)
    t0 = time.perf_counter()
    try:
        endf = open_endf(str(path))
        present = set(endf.files)
        data_mf, cov_mf, read = _plan(present, what, mf)
        if not present:
            return TapeResult(None, "read: no ENDF section found in the file", False,
                              time.perf_counter() - t0, 0.0, ())
        if not data_mf and not cov_mf:
            return TapeResult(None, None, False, time.perf_counter() - t0, 0.0, ())
        if what == ("covariances",):
            # Only what the covariance checks see, as check_covariance_library does.
            endf.files = {m: endf.files[m] for m in read}
        else:
            for m in read:
                try:
                    endf.files[m]
                except Exception:  # noqa: BLE001 - the data checks report it as file_unreadable
                    pass
    except Exception as exc:  # noqa: BLE001 - recorded, the walk goes on
        return TapeResult(None, f"read: {type(exc).__name__}: {exc}", False,
                          time.perf_counter() - t0, 0.0, ())
    t1 = time.perf_counter()
    try:
        findings = ()
        mat = getattr(endf, "mat", None)
        if data_mf:
            rep = check_data(endf, mf=data_mf)
            findings += rep.findings
        if cov_mf:
            rep = check_covariances(endf, mf=cov_mf)
            findings += rep.findings
            mat = mat if mat is not None else rep.mat
        report = CheckReport(findings, source=str(path), mat=mat,
                             mf=tuple(sorted(set(data_mf) | set(cov_mf))), kind=kind_of(what))
    except Exception as exc:  # noqa: BLE001
        return TapeResult(None, f"check: {type(exc).__name__}: {exc}", True, t1 - t0,
                          time.perf_counter() - t1, tuple(sorted(set(data_mf) | set(cov_mf))))
    return TapeResult(report, None, True, t1 - t0, time.perf_counter() - t1, report.mf)

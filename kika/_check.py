"""``kika.check()`` -- one call to check evaluations, whatever is checked.

The door in front of every check kika has on evaluated data: the data checks
(:mod:`kika.endf.checks.data`: MF1-MF28, format, mathematics and physics) and
the covariance checks (:func:`kika.endf.checks.check_covariances`: MF31-35,
MF40). Both by default, either alone with ``what=``, some files only with
``mf=``::

    report = kika.check("n-026_Fe_056.endf")                   # everything
    report = kika.check("n-026_Fe_056.endf", what="data")      # no covariances
    report = kika.check("n-026_Fe_056.endf", mf=[3, 33])       # MF3 and MF33
    library = kika.check("endfb81/neutrons/", library="endfb81")
    report.to_html("fe56.check.html"); print(report)

A file gives a :class:`~kika.endf.checks.CheckReport`; a folder, or a list of
files, a :class:`~kika.endf.checks.LibraryReport` with one row per tape. Both
write the same HTML, Markdown and JSON as the covariance reports always have.

Nothing is changed and nothing is refused: every finding only informs, at the
level its consequence deserves (``note``, ``warn``, ``defect``). A GNDS file is
read into kika's model and written as ENDF-6 in memory first, and the report
says so: what is checked is then kika's translation of it.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = ["check"]


def check(source, *, what: Union[str, Sequence[str]] = "all", mf: Optional[Sequence[int]] = None,
          deep: bool = False, library: Optional[str] = None, recursive: bool = False, progress=True,
          on_tape=None, should_stop=None, on_result=None):
    """Check one evaluation, a folder of them, or a list of files.

    Parameters
    ----------
    source : path, folder, or iterable of paths
        An ENDF-6 tape or GNDS file; a folder (every tape in it, as
        :data:`kika.endf.checks.TAPE_PATTERNS` finds them); or the files.
    what : str or sequence
        ``"all"`` (default), ``"data"`` or ``"covariances"``.
    mf : sequence of int, optional
        Only these files, and what they are checked against.
    deep : bool
        Also reconstruct the cross sections and check what codes will use
        (positivity, sums, 1/v): a reconstruction per tape.
    library : str, optional
        A name for a folder's report (``"endfb81"``).
    recursive : bool
        Look in subfolders too.
    progress, on_tape, should_stop, on_result
        For a folder: as :func:`kika.endf.checks.check_covariance_library` takes them.

    Returns
    -------
    CheckReport or LibraryReport
    """
    from kika.endf.checks.evaluation import check_evaluation, normalise_what
    from kika.endf.checks.library import check_library

    what = normalise_what(what)
    if isinstance(source, (str, os.PathLike)):
        path = Path(os.fspath(source))
        if path.is_dir():
            return check_library(path, what=what, mf=mf, deep=deep, recursive=recursive, library=library,
                                 progress=progress, on_tape=on_tape, should_stop=should_stop,
                                 on_result=on_result)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if _is_gnds(path):
            return _check_gnds(path, what, mf, deep)
        return check_evaluation(path, what=what, mf=mf, deep=deep)
    if hasattr(source, "files"):  # an ENDF object already read
        return check_evaluation(source, what=what, mf=mf, deep=deep)
    return check_library(list(source), what=what, mf=mf, deep=deep, library=library, progress=progress,
                         on_tape=on_tape, should_stop=should_stop, on_result=on_result)


def _is_gnds(path: Path) -> bool:
    try:
        from ._identify import identify

        return identify(path) == "gnds"
    except Exception:  # noqa: BLE001 - not identified: read it as ENDF and let that say why
        return False


def _check_gnds(path: Path, what, mf, deep):
    """A GNDS evaluation: read into the model, written as ENDF-6, checked as such."""
    import kika
    from kika.endf.checks.evaluation import check_evaluation
    from kika.endf.checks.findings import NOTE, CheckReport, Finding, Location
    from kika.endf.writers.assemble import writeEndfTape

    suite = kika.read(str(path), format="gnds", covariances="covariances" in what)
    with tempfile.TemporaryDirectory(prefix="kika-check-") as tmp:
        tape = Path(tmp) / (path.stem + ".endf")
        writeEndfTape(suite, tape)
        report = check_evaluation(tape, what=what, mf=mf, deep=deep)
    note = Finding("gnds_translated", NOTE,
                   "a GNDS evaluation: checked as the ENDF-6 tape kika writes from it, so a finding "
                   "may come from the translation as well as from the evaluation",
                   Location(mat=report.mat, mf=1, mt=451), {"source": str(path)})
    return CheckReport((note,) + report.findings, source=str(path), mat=report.mat, mf=report.mf,
                       kind=report.kind)

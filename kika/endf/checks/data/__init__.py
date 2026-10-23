"""Checks of an evaluation's data, everything but the covariances (layer 1).

:func:`check_data` reports what is wrong with a tape's MF1-MF28 -- format,
mathematics and physics -- without changing anything, in the vocabulary of
the covariance checks (:mod:`kika.endf.checks.findings`): a finding has a
check id, a level (``note``/``warn``/``defect``), a location and its evidence,
and every check id has one text in :data:`DATA_CHECKS`.

The checks come in families, one module each, and each family says which
files it reads; ``mf=`` runs only the families that look at those files::

    from kika.endf.checks.data import check_data
    report = check_data("n-026_Fe_056.endf")           # everything
    report = check_data("n-026_Fe_056.endf", mf=[3])   # MF3 only
    print(report); report.to_html("fe56_data.html")

The checks are *static* by default: they read the tape and nothing else, no
reconstruction and no processing, so a tape costs what reading it costs.
``deep=True`` adds the checks that need the cross sections reconstructed
(:mod:`.deep`), at the cost of a reconstruction.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from ..findings import DEFECT, NOTE, WARN, CheckReport
from .context import Out, Tape

__all__ = ["check_data", "FAMILIES", "DATA_MF", "DATA_CHECKS"]


def _families():
    from .distributions import check_distributions
    from .general import check_general
    from .photons import check_photons
    from .decay import check_decay
    from .photoatomic import check_photoatomic
    from .resonances import check_resonances
    from .thermal import check_thermal
    from .xs import check_xs

    # (name, the files it reads, the check). A family runs when ``mf`` names
    # any of its files; it reads the others it needs on its own.
    return (
        ("general", (1,), check_general),
        ("resonances", (2,), check_resonances),
        ("cross sections", (3,), check_xs),
        ("distributions", (4, 5, 6), check_distributions),
        ("photons", (12, 13, 14, 15), check_photons),
        ("thermal scattering", (7,), check_thermal),
        ("decay and yields", (8,), check_decay),
        ("photo-atomic", (23, 27, 28), check_photoatomic),
    )


#: Every file a data check looks at; covariances (MF30-40) are not data.
DATA_MF = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 23, 26, 27, 28)


def check_data(source, *, mf: Optional[Sequence[int]] = None, deep: bool = False) -> CheckReport:
    """Check the data of one evaluation; returns a :class:`CheckReport` of kind ``"data"``.

    Parameters
    ----------
    source : path or ENDF
        A tape on disk (read lazily: only the files the checks need are
        parsed) or an :class:`~kika.endf.classes.endf.ENDF` already read.
    mf : sequence of int, optional
        Check only these files (and nothing about the others); all by default.
    deep : bool
        Also reconstruct the cross sections and check what codes will use
        (:mod:`.deep`): positivity, sums and 1/v. Needs MF3 asked for.
    """
    from kika.endf import open_endf

    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        endf = open_endf(path)
    else:
        endf, path = source, getattr(source, "source_path", None)
    tape = Tape(endf, path)
    wanted = tuple(sorted(set(DATA_MF if mf is None else mf) & set(DATA_MF)))
    out = Out(tape.mat)
    for name, files, run in _families():
        if not set(files) & set(wanted):
            continue
        run(tape, out)
    if deep and 3 in wanted:
        from .deep import check_deep

        check_deep(tape, out)
    _parse_errors(tape, out, wanted)
    present = [m for m in tape.present if m in wanted]
    return CheckReport(tuple(out.findings), source=path, mat=tape.mat, mf=tuple(present), kind="data")


def _parse_errors(tape: Tape, out: Out, wanted) -> None:
    for mf in wanted:
        if not tape.has(mf):
            continue
        tape.mf(mf)  # parse it, if no check did, to see whether it can be read
        if mf in tape.parse_failures:
            out.add("file_unreadable", DEFECT,
                    f"kika could not read MF{mf} at all ({tape.parse_failures[mf]}); none of it was checked",
                    mf=mf, error=tape.parse_failures[mf])
            continue
        for mt, msg in sorted(tape.parse_errors(mf).items()):
            out.add("section_unreadable", DEFECT,
                    f"MF{mf}/MT{mt} could not be read and was left out ({msg})",
                    mf=mf, mt=mt, error=str(msg))


def __getattr__(name):
    if name == "DATA_CHECKS":
        from .descriptions import DATA_CHECKS

        return DATA_CHECKS
    if name == "FAMILIES":
        return _families()
    raise AttributeError(name)

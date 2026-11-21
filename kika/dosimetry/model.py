"""A :class:`~kika.dosimetry.Dosimeter` from an evaluation read into the model.

A reaction (MF3 with its MF33) or one final state of a production (MF10 with its
MF40, which is how dosimetry libraries state isomer activation). The cross section
is the tabulated form, with its own interpolation laws; the covariance is the
diagonal section of the same quantity in the suite's ``covarianceSuite``, on the
grid the evaluation states. An absolute covariance is made relative with the cross
section averaged over each interval with a 1/E weight, as kika's MF33 reading does.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from kika._covariance_forms import require_single_matrix
from kika.algebra import group_averages, interval_laws

from .response import Dosimeter

__all__ = ["dosimeter_from_suite"]


def _endf_mf_mt(link) -> Optional[tuple]:
    text = getattr(link, "ENDF_MFMT", None)
    if not text:
        return None
    parts = str(text).replace("/", ",").split(",")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def _diagonal_form(suite, mf: int, mt: int, label: Optional[str] = None):
    """The one-matrix form of the diagonal section of ``(mf, mt)``; for MF40, the one whose href names *label*."""
    covariances = getattr(suite, "covarianceSuite", None)
    if covariances is None:
        return None
    for section in covariances.covarianceSections:
        if section.columnData is not None or _endf_mf_mt(section.rowData) != (mf, mt):
            continue
        if label is not None and f"[@label='{label}']" not in (section.rowData.href or ""):
            continue
        return require_single_matrix(section.form, f"the MF{mf} section {section.label!r}")
    return None


def _level(suite, pid: str) -> int:
    particle = suite.PoPs.particles.get(pid) if getattr(suite, "PoPs", None) is not None else None
    level = getattr(particle, "nuclearLevel", None)
    if level is None:
        match = re.search(r"_e(\d+)$", pid)
        level = int(match.group(1)) if match else 0
    return int(level)


def _production(suite, mt: int, lfs: int):
    found = [p for p in suite.productions
             if p.label.startswith(f"MT{int(mt)} ")
             and any(product.pid != "photon" and _level(suite, product.pid) == int(lfs)
                     for product in p.outputChannel.products)]
    if len(found) != 1:
        names = [p.label for p in suite.productions if p.label.startswith(f"MT{int(mt)} ")]
        raise KeyError(f"MT{mt} LFS={lfs} names {len(found)} productions (those of MT{mt}: {names}); "
                       f"it must name exactly one")
    return found[0]


def _relative(matrix, edges, x, y, laws):
    weighted = edges[:-1] > 0
    average = np.zeros(edges.size - 1)
    if weighted.any():
        k = int(np.argmax(weighted))
        if k:
            average[:k] = group_averages(x, y, laws, edges[:k + 1])
        average[k:] = group_averages(x, y, laws, edges[k:], "1/x")
    else:
        average = group_averages(x, y, laws, edges)
    average = np.nan_to_num(average, nan=0.0)
    scale = np.where(average > 0, 1.0 / np.where(average > 0, average, 1.0), 0.0)
    return matrix * np.outer(scale, scale)


def dosimeter_from_suite(suite, mt: int, lfs: Optional[int] = None, name: Optional[str] = None,
                         form: str = "eval") -> Dosimeter:
    """The dosimeter of one reaction, or of one final state of a production, and its covariance.

    Parameters
    ----------
    suite
        A reaction suite from :func:`kika.read`, with its ``covarianceSuite``
        attached (the default of the reader).
    mt : int
        The ENDF MT number of the reaction.
    lfs : int, optional
        The final state of the product (0 the ground state, 1 the first isomer...).
        None reads the reaction (MF3 and MF33); a value reads the production of that
        state (MF10 and MF40).
    name : str, optional
        The dosimeter's label; ``"<target> MT<mt>"``, or the production's label, by
        default.
    form : str
        The label of the cross-section form: ``'eval'`` for the evaluated table,
        ``'recon'`` for one reconstructed from resonance parameters.

    Returns
    -------
    Dosimeter
        The cross section in eV and b with its interpolation laws, and the relative
        covariance on the evaluation's own intervals; no covariance when the suite
        holds no diagonal section for it.

    Raises
    ------
    KeyError
        If the reaction or the form is absent, or *mt* and *lfs* do not name exactly
        one production.
    TypeError
        If the form is not a tabulated function (reconstruct resonances first).
    ValueError
        If the cross section states units other than eV and b, the production is not
        an MF10 table, the covariance section does not state one matrix (a GNDS
        ``mixed`` or ``sum``), or *lfs* is None and *mt* has no MF33 covariance but
        has productions (the final state has to be named).
    """
    if lfs is None:
        suite.cross_section(mt, form)
        x, y, pairs = suite.reactionByENDF_MT(mt).crossSection[form].toEndfRegions()
        label = name if name is not None else f"{getattr(suite, 'target', '?')} MT{mt}"
        matrix = _diagonal_form(suite, 33, mt)
        productions = [p.label for p in getattr(suite, "productions", ()) if p.label.startswith(f"MT{int(mt)} ")]
        if matrix is None and productions:
            raise ValueError(f"MT{mt} has no MF33 covariance but has productions {productions} (MF10, "
                             f"with their MF40): a dosimetry reaction to a final state is one of those; "
                             f"pass lfs= to read it")
    else:
        production = _production(suite, mt, lfs)
        chosen = production.crossSection[form]
        if type(chosen).__name__ == "Reference":
            raise ValueError(f"{production.label} is an MF9 production (a multiplicity times the "
                             f"reaction's cross section), not an MF10 table")
        x, y, pairs = chosen.toEndfRegions()
        label = name if name is not None else production.label
        matrix = _diagonal_form(suite, 40, mt, production.label)
    laws = interval_laws(len(x), pairs)

    if matrix is None:
        return Dosimeter(label, x, y, laws)
    edges = np.asarray(matrix.rowGrid, dtype=float)
    covariance = np.asarray(matrix.matrix, dtype=float)
    if not matrix.isRelative:
        covariance = _relative(covariance, edges, x, y, laws)
    return Dosimeter(label, x, y, laws, edges, covariance)

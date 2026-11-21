"""A :class:`~kika.dosimetry.Dosimeter` from an evaluation read into the model.

The cross section is the reaction's tabulated form, with its own interpolation
laws; the covariance is the diagonal MF33 section of the same reaction in the
suite's ``covarianceSuite``, on the grid the evaluation states. An absolute
covariance is made relative with the cross section averaged over each interval.
Production cross sections (MF10) and their covariances (MF40) are not read here:
the covariance decoder does not bring MF40 into the model yet.
"""
from __future__ import annotations

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


def _diagonal_mf33_form(suite, mt: int):
    covariances = getattr(suite, "covarianceSuite", None)
    if covariances is None:
        return None
    for section in covariances.covarianceSections:
        row = _endf_mf_mt(section.rowData)
        column = row if section.columnData is None else _endf_mf_mt(section.columnData)
        if row == (33, mt) and column == (33, mt):
            return require_single_matrix(section.form, f"the MF33 section of MT{mt}")
    return None


def dosimeter_from_suite(suite, mt: int, name: Optional[str] = None, form: str = "eval") -> Dosimeter:
    """The dosimeter of one reaction of an evaluation: its cross section and its MF33 covariance.

    Parameters
    ----------
    suite
        A reaction suite from :func:`kika.read`, with its ``covarianceSuite``
        attached (the default of the reader).
    mt : int
        The ENDF MT number of the reaction.
    name : str, optional
        The dosimeter's label; ``"<target> MT<mt>"`` by default.
    form : str
        The label of the cross-section form: ``'eval'`` for the evaluated table,
        ``'recon'`` for one reconstructed from resonance parameters.

    Returns
    -------
    Dosimeter
        The cross section in eV and b with its interpolation laws, and the relative
        covariance on the evaluation's own intervals; no covariance when the suite
        holds no diagonal MF33 section for *mt*.

    Raises
    ------
    KeyError
        If the reaction or the form is absent.
    TypeError
        If the form is not a tabulated function (reconstruct resonances first).
    ValueError
        If the cross section states units other than eV and b, or the covariance
        section does not state one matrix (a GNDS ``mixed`` or ``sum``).
    """
    suite.cross_section(mt, form)
    x, y, pairs = suite.reactionByENDF_MT(mt).crossSection[form].toEndfRegions()
    laws = interval_laws(len(x), pairs)
    label = name if name is not None else f"{getattr(suite, 'target', '?')} MT{mt}"

    matrix = _diagonal_mf33_form(suite, mt)
    if matrix is None:
        return Dosimeter(label, x, y, laws)
    edges = np.asarray(matrix.rowGrid, dtype=float)
    covariance = np.asarray(matrix.matrix, dtype=float)
    if not matrix.isRelative:
        average = group_averages(x, y, laws, edges)
        scale = np.where(average > 0, 1.0 / np.where(average > 0, average, 1.0), 0.0)
        covariance = covariance * np.outer(scale, scale)
    return Dosimeter(label, x, y, laws, edges, covariance)

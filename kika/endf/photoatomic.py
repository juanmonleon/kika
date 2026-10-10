"""Photo-atomic data of an ENDF tape, in plain arrays.

The photo-atomic sublibrary (NSUB=3) is parsed into MF23 (cross sections) and
MF27 (form factors) by :mod:`kika.endf.parsers.parse_photoatomic`; the atomic
relaxation sublibrary (NSUB=6) into MF28. Those classes keep every record so
a tape can be written back. This module is the other door: what a user who
wants to *look* at the data asks for, as numpy arrays in the file's units.

    >>> endf = read_endf("photoat-026_Fe_000.endf")
    >>> photoatomic_summary(endf)["z"]
    26
    >>> E, sigma = photoatomic_cross_sections(endf)[502]      # eV, barns
    >>> x, F = form_factors(endf)[502]                          # 1/Å, electrons

Nothing here converts units. ENDF writes eV and barns, and the MF27 abscissa
is ``sin(θ/2)/λ`` in 1/Å for MT 502/504 and the photon energy in eV for MT
505/506; the summary says which per MT so a caller labelling an axis does not
have to know.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .classes.mf23.base import photoatomic_mt_name

__all__ = [
    "photoatomic_summary",
    "photoatomic_cross_sections",
    "form_factors",
    "is_photoatomic",
]

#: MF1/451 NSUB of the photo-atomic and atomic-relaxation sublibraries.
NSUB_PHOTOATOMIC = 3
NSUB_ATOMIC_RELAXATION = 6


def _mf(endf, number: int):
    files = getattr(endf, "files", None) or {}
    return files.get(number)


def _nsub(endf) -> Optional[int]:
    mf1 = _mf(endf, 1)
    section = mf1.sections.get(451) if mf1 is not None else None
    value = getattr(section, "_nsub", None) if section is not None else None
    return int(value) if value is not None else None


def is_photoatomic(endf) -> bool:
    """True when the tape carries MF23, MF27 or MF28, whatever MF1 says."""
    return any(_mf(endf, n) is not None for n in (23, 27, 28))


def _z_and_za(endf) -> Tuple[Optional[int], Optional[float]]:
    """Z from the first HEAD that has a ZA. Photo-atomic ZA is ``1000·Z``."""
    for number in (23, 27, 28):
        mf = _mf(endf, number)
        if mf is None:
            continue
        for section in mf.sections.values():
            za = getattr(section, "_za", None)
            if za is not None:
                return int(round(za)) // 1000, float(za)
    return None, None


def photoatomic_summary(endf) -> dict:
    """What the photo-atomic files of *endf* hold, without the arrays.

    Returns
    -------
    dict
        ``z`` (int or None), ``za`` (float or None), ``nsub`` (MF1/451 NSUB or
        None), and three lists, each entry a dict:

        * ``cross_sections`` (MF23): ``mt``, ``name``, ``n_points``,
          ``e_min``/``e_max`` (eV), and for a subshell ``subshell``, ``epe``
          (binding energy, eV) and ``efl`` (fluorescence yield, eV);
        * ``form_factors`` (MF27): ``mt``, ``name``, ``n_points``, ``x_unit``
          (``"1/angstrom"`` or ``"eV"``), ``x_min``/``x_max``;
        * ``relaxation`` (MF28): ``subshell``, ``designator``,
          ``binding_energy`` (eV), ``electrons``, ``n_transitions``.

        ``parse_errors`` maps ``MF -> {MT: reason}`` for any section the
        parser could not read, so an MT that is *missing* can be told from
        one that was *lost*.
    """
    z, za = _z_and_za(endf)
    out = {"z": z, "za": za, "nsub": _nsub(endf),
           "cross_sections": [], "form_factors": [], "relaxation": [],
           "parse_errors": {}}

    mf23 = _mf(endf, 23)
    if mf23 is not None:
        for mt in sorted(mf23.sections):
            s = mf23.sections[mt]
            entry = {"mt": mt, "name": photoatomic_mt_name(mt), "n_points": len(s.table.x),
                     "e_min": float(s.table.x[0]) if s.table.x else None,
                     "e_max": float(s.table.x[-1]) if s.table.x else None}
            if s.is_subshell:
                entry.update(subshell=s.subshell, epe=float(s.epe), efl=float(s.efl))
            out["cross_sections"].append(entry)

    mf27 = _mf(endf, 27)
    if mf27 is not None:
        for mt in sorted(mf27.sections):
            s = mf27.sections[mt]
            out["form_factors"].append({
                "mt": mt, "name": photoatomic_mt_name(mt), "n_points": len(s.table.x),
                "x_unit": s.x_unit,
                "x_min": float(s.table.x[0]) if s.table.x else None,
                "x_max": float(s.table.x[-1]) if s.table.x else None,
            })

    mf28 = _mf(endf, 28)
    if mf28 is not None and 533 in mf28.sections:
        for shell in mf28.sections[533].subshells:
            out["relaxation"].append({
                "subshell": shell.name, "designator": shell.subi,
                "binding_energy": float(shell.binding_energy),
                "electrons": float(shell.electrons),
                "n_transitions": len(shell.transitions),
            })

    for number in (23, 27, 28):
        mf = _mf(endf, number)
        if mf is not None and mf.parse_errors:
            out["parse_errors"][number] = dict(mf.parse_errors)
    return out


def _tables(endf, number: int, mts: Optional[Iterable[int]]):
    mf = _mf(endf, number)
    if mf is None:
        return {}
    wanted = sorted(mf.sections) if mts is None else [m for m in mts if m in mf.sections]
    return {mt: (np.asarray(mf.sections[mt].table.x, dtype=float),
                 np.asarray(mf.sections[mt].table.y, dtype=float))
            for mt in wanted}


def photoatomic_cross_sections(endf, mts: Optional[Iterable[int]] = None
                               ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """``{mt: (E in eV, σ in barns)}`` for the MF23 sections of *endf*.

    *mts* narrows the result; an MT the tape lacks is left out, not an error,
    because the subshells present differ from element to element. Each MT
    keeps its own grid: the evaluations are not on a common one, and
    interpolating them onto one would hide the photoelectric edges.
    """
    return _tables(endf, 23, mts)


def form_factors(endf, mts: Optional[Iterable[int]] = None
                 ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """``{mt: (x, value)}`` for the MF27 sections of *endf*.

    ``x`` is ``sin(θ/2)/λ`` in 1/Å for MT 502 (coherent form factor F) and
    504 (incoherent scattering function S); it is the photon energy in eV for
    505 and 506 (imaginary and real anomalous scattering factors).
    """
    return _tables(endf, 27, mts)

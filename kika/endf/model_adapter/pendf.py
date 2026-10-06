"""A PENDF's pointwise cross sections into the model, as a ``recon`` style.

GNDS §9.1's own example is a ``crossSection`` holding a
``resonancesWithBackground`` form labelled ``eval`` and an ``XYs1d`` labelled
``recon``, with a ``crossSectionReconstructed`` style saying ``recon`` was
derived from ``eval``. An ENDF tape read into the model has only the first: its
MF3 is the background, and in the resonance region that is not the cross
section. A PENDF from NJOY RECONR is the second, and this module puts it where
§9 says it goes.

**Why a separate step and not part of** ``kika.read``: reconstruction is
processing, it needs a processor (NJOY), and the processor's choices — the
tolerance, what it does in the unresolved region — belong in the report of the
suite that carries the result, not hidden inside a reader.

**Only 0 K.** RECONR writes unbroadened σ, and the style says
"reconstructed", not "heated". A PENDF that went through BROADR (``TEMP > 0``
in MF1/451) is refused: GNDS has a separate ``heated`` style for it, and its
first consumer here (G4NDL, which Geant4 Doppler-broadens itself) would count
the broadening twice.

This is roadmap G4NDL Phase 9's first half (ENDF → reconstruction → model);
the second half is ``kika.write(suite, root, format="g4ndl")``, which writes
the ``recon`` form and knows nothing about ENDF.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport, CrossSectionReconstructed, Regions1d, XYs1d, crossSectionAxes,
)
from kika.nuclear_data.model.enums import Interpolation

__all__ = ["attachReconstruction", "readReconstructed", "pendfHeader",
           "RECONSTRUCTED_LABEL"]

RECONSTRUCTED_LABEL = "recon"

_ENDF_FLOAT = re.compile(r"^([+-]?\d*\.?\d*)([+-]\d+)$")


def _endfFloat(field: str) -> float:
    s = field.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = _ENDF_FLOAT.match(s)
        if not m:
            raise ValueError(f"not an ENDF number: {field!r}") from None
        return float(f"{m.group(1)}e{m.group(2)}")


def pendfHeader(path) -> dict:
    """``MAT``, ``ZA``, ``AWR``, ``TEMP`` and ``ERR`` from a PENDF's MF1/451.

    Read off the raw records because the MF1 parser does not keep ``ERR`` (the
    reconstruction tolerance RECONR writes beside ``TEMP``, in the fourth
    record for ENDF-6), and the tolerance is the one number that says how close the
    pointwise σ is to the resonance formula.
    """
    lines = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                lines.append(line)
                if len(lines) == 4:
                    break
    if len(lines) < 4:
        raise ValueError(f"{path}: no MF1/MT451 header; is this an ENDF/PENDF tape?")
    # ENDF-6 (NFOR=6, the last field of the second record) has AWI/EMAX as the
    # third record and TEMP/ERR as the fourth; ENDF-5 and earlier, as the third.
    nfor = int(lines[1][55:66].strip() or 0)
    temps = lines[3] if nfor >= 6 else lines[2]
    return {
        "mat": int(lines[0][66:70]),
        "za": int(_endfFloat(lines[0][0:11])),
        "awr": _endfFloat(lines[0][11:22]),
        "nfor": nfor,
        "temp": _endfFloat(temps[0:11]),
        "err": _endfFloat(temps[11:22]),
    }


def _linlin(form, mt: int) -> XYs1d:
    """The PENDF MF3 table as one lin-lin ``XYs1d``; RECONR writes nothing else."""
    xs, ys, pairs = form.toEndfRegions()
    codes = {int(c) for _, c in pairs}
    if codes != {2}:
        raise ValueError(f"MT{mt}: the PENDF table has interpolation codes {sorted(codes)}, "
                         f"not lin-lin; it did not come out of RECONR")
    return XYs1d(xs=np.array(xs, dtype=float), ys=np.array(ys, dtype=float),
                 interpolation=Interpolation.linlin, axes=crossSectionAxes(),
                 label=RECONSTRUCTED_LABEL)


def attachReconstruction(suite, pendf, *, label: str = RECONSTRUCTED_LABEL,
                         mts: Optional[Iterable[int]] = None,
                         report: Optional[ConversionReport] = None) -> ConversionReport:
    """Add ``pendf``'s MF3 to ``suite`` as ``label`` forms, and the style that says so.

    Parameters
    ----------
    suite
        A suite decoded from the evaluation the PENDF was made from. Its
        ``eval`` style is what the new style is ``derivedFrom``.
    pendf
        Path to a PENDF tape (RECONR output, ``TEMP = 0``).
    mts
        Reactions to attach; default every MF3 section that is also a reaction
        of the suite. A PENDF section without a reaction (MT1, MT301…) is
        skipped and listed in the report.

    Raises
    ------
    ValueError
        If the PENDF is for another material (MAT or ZA differ), is heated
        (``TEMP > 0``), or a table is not lin-lin.

    The suite is modified in place, on purpose: the reconstruction is now part
    of the evaluation's description, which is what §9 styles are for.
    """
    from kika.endf.model_adapter.decode import decodeMF3MT
    from kika.endf.read_endf import read_endf

    report = report if report is not None else ConversionReport()
    path = Path(pendf)
    header = pendfHeader(path)
    prov = suite.provenance
    if getattr(prov, "mat", None) is not None and prov.mat != header["mat"]:
        raise ValueError(f"{path.name} is MAT {header['mat']}, the suite is MAT {prov.mat}")
    if getattr(prov, "za", None) is not None and prov.za != header["za"]:
        raise ValueError(f"{path.name} is ZA {header['za']}, the suite is ZA {prov.za}")
    if header["temp"] != 0.0:
        raise ValueError(
            f"{path.name} is at TEMP={header['temp']!r} K: a Doppler-broadened PENDF is a "
            f"'heated' style, not a reconstruction, and broadening it again downstream "
            f"(Geant4 does, for G4NDL) would count the Doppler effect twice. Use the "
            f"RECONR output, before BROADR")

    pendfTape = read_endf(str(path), mf_numbers=[3])
    mf3 = pendfTape.mf.get(3)
    if mf3 is None or not mf3.mt:
        raise ValueError(f"{path.name} has no MF3")
    wanted = set(int(m) for m in mts) if mts is not None else None
    present = set(suite.reactions.ENDF_MTs)
    attached, skipped = [], []
    for mt, section in sorted(mf3.mt.items()):
        mt = int(mt)
        if wanted is not None and mt not in wanted:
            continue
        if mt not in present:
            skipped.append(mt)
            continue
        reaction, _ = decodeMF3MT(section)
        suite.reactions[mt].crossSection[label] = _linlin(reaction.crossSection["eval"], mt)
        attached.append(mt)
    if wanted is not None and wanted - set(attached):
        raise ValueError(f"MT {sorted(wanted - set(attached))} not attached: absent "
                         f"from the PENDF or from the suite")

    if label not in suite.styleLabels():
        suite.styles.add(CrossSectionReconstructed(label=label, derivedFrom="eval"))
    report.approximated(
        f"'{label}': {len(attached)} cross sections reconstructed by NJOY RECONR to "
        f"ERR={header['err']:g} (relative), at 0 K, from {path.name}")
    resonances = getattr(suite, "resonances", None)
    if resonances is not None and resonances.unresolved is not None:
        u = resonances.unresolved
        report.approximated(
            f"'{label}': between {u.domainMin:g} and {u.domainMax:g} eV (unresolved "
            f"region) the reconstructed sigma is RECONR's infinitely dilute average; "
            f"no self-shielding or probability tables are carried")
    if skipped:
        report.warn(f"PENDF MF3 sections without a reaction in the suite, not attached: "
                    f"MT {skipped}")
    return report


def readReconstructed(endf, *, pendf=None, njoy=None, tolerance: float = 0.001,
                      cache_dir=None):
    """Read an ENDF tape and attach its RECONR reconstruction: ``(suite, report)``.

    Give either ``pendf`` (an existing RECONR output for this tape) or
    ``njoy`` (the executable): then RECONR runs at ``tolerance`` through
    :func:`kika.processing.njoy_pendf_cache.get_or_create_pendf`, which caches
    the PENDF by the tape's SHA-256.

        >>> suite, report = readReconstructed("n_82-Pb-208g.jeff", njoy="njoy.exe")
        >>> kika.write(suite, "G4NDL-mine", format="g4ndl")
    """
    import kika

    endf = Path(endf)
    if pendf is None:
        if njoy is None:
            raise ValueError("give pendf= (a RECONR output) or njoy= (the executable)")
        from kika.processing.njoy_pendf_cache import get_or_create_pendf
        pendf = get_or_create_pendf(endf, tolerance=tolerance, njoy_exe=njoy,
                                    cache_dir=cache_dir)
    suite = kika.read(str(endf), format="endf")
    report = attachReconstruction(suite, pendf)
    if getattr(suite, "report", None) is not None:
        suite.report.extend(report)
    return suite, report

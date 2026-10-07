"""Perturb the cross sections the resonance region actually has, not its background.

An ENDF tape with LRP=1 states the resolved (and possibly unresolved) resonance
region twice over: MF2 holds the resonance parameters and MF3 holds only a
*background* -- often zero, sometimes negative -- that the reconstructed
resonance cross section is added to. ENDF/B-VIII.1 Fe-56 has MF3 MT1, MT2 and
MT102 at zero or below for every energy under 850 keV. A factor applied to
that MF3 multiplies the background and leaves the resonances alone, so a
realisation built that way is, below the resonance boundary, essentially the
evaluation. That is what ``perturbFromModel`` did until 2026-10-07 whenever it
was given an ENDF tape rather than a PENDF; ``perturb_PENDF_files`` and SANDY
never had the problem, because they perturb NJOY RECONR's output.

The fix is the same idea, put where the model can use it:

1. **RECONR once**, at 0 K, through the PENDF cache
   (:func:`~kika.processing.njoy_pendf_cache.get_or_create_pendf`).
2. **A base tape** (:func:`reconstructedBase`): the original ENDF with its MF3
   replaced by RECONR's and MF1/451's LRP set to 2. ENDF-6 defines LRP=2 as
   exactly this -- "resonance parameters are given in File 2 for information;
   the cross sections in File 3 are complete" -- and NJOY honours it (RECONR
   with LRP/=1 adds no resonance contribution; ``reconr.f90``). MF2 is kept:
   PURR and UNRESR still read the unresolved parameters from it. The model is
   decoded from this tape, so what it perturbs is the cross section, and every
   ENDF it writes (``endf-delta``, ``endf-tape``) is a complete statement of
   the realisation.
3. **ACE from two tapes**, as the legacy pipeline and SANDY do: the perturbed
   PENDF (:func:`spliceMF3` of the sample's MF3 into RECONR's PENDF) and the
   original ENDF carrying the sample's other perturbations (MF1, MF4, MF5 ...)
   with its own MF3 and LRP, through
   :func:`~kika.njoy.run_njoy.run_njoy_with_pendf`, which skips RECONR. The
   two-tape route is not optional: PURR reads MF3 from the ENDF tape *as a
   background* for an unresolved range with LSSF=0, so handing it a complete
   MF3 would count that range twice.

The perturbation is applied at 0 K and Doppler-broadened afterwards, the order
both other pipelines use.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

__all__ = ["needsReconstruction", "reconstructedBase", "resonanceRanges",
           "setLRP", "spliceMF3"]


def resonanceRanges(endfObj) -> List[Tuple[int, float, float]]:
    """``[(LRU, EL, EH), ...]`` of every resolved/unresolved range in MF2/151."""
    mf = getattr(endfObj, "mf", None) or {}
    if 2 not in mf or 151 not in getattr(mf[2], "mt", {}):
        return []
    ranges = []
    for isotope in getattr(mf[2].mt[151], "_isotopes", ()) or ():
        for energyRange in getattr(isotope, "energy_ranges", ()) or ():
            lru = int(getattr(energyRange, "lru", 0) or 0)
            if lru in (1, 2):
                ranges.append((lru, float(energyRange.el), float(energyRange.eh)))
    return ranges


def _lrp(endfObj) -> Optional[int]:
    mf = getattr(endfObj, "mf", None) or {}
    section = getattr(mf.get(1), "mt", {}).get(451) if 1 in mf else None
    value = getattr(section, "_lrp", None)
    return None if value is None else int(value)


def needsReconstruction(endfObj) -> bool:
    """LRP=1 and MF2 states a resolved or unresolved range: MF3 is a background there."""
    return _lrp(endfObj) == 1 and bool(resonanceRanges(endfObj))


# ----------------------------------------------------------------------
# Text surgery on tapes: whole MF3 blocks and one header field
# ----------------------------------------------------------------------

def _readLines(path: Path) -> Tuple[List[str], str]:
    from kika.endf.utils import line_ending

    newline = line_ending(str(path))
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    return text.splitlines(), newline


def _mfOf(line: str) -> Optional[int]:
    try:
        return int(line[70:72])
    except (ValueError, IndexError):
        return None


def _mf3Span(lines: List[str]) -> Tuple[int, int]:
    """``[first, end)`` of the MF3 block, its FEND included; ``(i, i)`` if absent."""
    first = next((i for i, line in enumerate(lines) if _mfOf(line) == 3), None)
    if first is None:
        # Where MF3 would go: before the first MF above 3, else before MEND.
        for i, line in enumerate(lines):
            mf = _mfOf(line)
            if mf is not None and mf > 3:
                return i, i
        return len(lines) - 2, len(lines) - 2
    end = first
    while end < len(lines) and _mfOf(lines[end]) == 3:
        end += 1
    # The FEND that closes MF3 is MF=0 MT=0 right after it.
    if end < len(lines) and _mfOf(lines[end]) == 0:
        end += 1
    return first, end


def _write(path: Path, lines: Iterable[str], newline: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(newline.join(lines) + newline)


def spliceMF3(target: Path, donor: Path, out: Path) -> Path:
    """*target* with its whole MF3 replaced by *donor*'s, written to *out*.

    Lines keep *target*'s record width and line ending; the MF1 directory is
    rebuilt, since the MT list and the line counts change.
    """
    from kika.endf.utils import record_width
    from kika.endf.writers.update_directory import update_mf1_directory

    lines, newline = _readLines(Path(target))
    donorLines, _ = _readLines(Path(donor))
    width = record_width(lines)
    a, b = _mf3Span(lines)
    c, d = _mf3Span(donorLines)
    block = [line[:width] if width else line for line in donorLines[c:d]]
    if c == d:
        raise ValueError(f"{donor} has no MF3 to splice")
    out = Path(out)
    _write(out, lines[:a] + block + lines[b:], newline)
    update_mf1_directory(str(out))
    return out


def setLRP(path: Path, value: int) -> None:
    """Set MF1/451's LRP (L1 of its HEAD record) in place."""
    lines, newline = _readLines(Path(path))
    for i, line in enumerate(lines):
        if line[70:75] == " 1451":
            lines[i] = line[:22] + f"{int(value):11d}" + line[33:]
            break
    else:
        raise ValueError(f"{path} has no MF1/451")
    _write(Path(path), lines, newline)


def reconstructedBase(source: Path, pendf: Path, out: Path) -> Path:
    """The tape the model perturbs: *source*, MF3 from RECONR's *pendf*, LRP=2."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    spliceMF3(Path(source), Path(pendf), tmp)
    setLRP(tmp, 2)
    shutil.move(str(tmp), str(out))
    return out

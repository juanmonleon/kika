"""The thermal cross sections of a TSL tape, computed by NJOY THERMR.

kika computes the two elastic parts of a thermal scattering law in closed form
(:mod:`kika.nuclear_data.model.thermal_scattering`) but not the incoherent
inelastic cross section, which is S(α,β) integrated over angle and outgoing
energy (roadmap E4c, deferred). Until it does, :func:`run_thermr` has NJOY do
it, in the chain every processing of a TSL material uses::

    moder → reconr → broadr → thermr → moder

on the free-atom neutron evaluation of the principal atom (its PENDF is what
THERMR adds the bound-atom reactions to) and the TSL tape. The result is the
tape's thermal cross sections at the temperatures asked for, as THERMR writes
them in MF3: the inelastic under ``mtref`` and the elastic, when the tape has
one, under ``mtref + 1``; and the free-atom elastic (MT2) of the same PENDF,
which is what the bound-atom data replaces below ``emax``.

The cross sections are THERMR's, with its normalisation: for a compound with
more than one principal atom (``natom``, B(6) of MF7/MT4: 2 for H in H₂O)
they are what NJOY's own processing of the material writes.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

__all__ = ["ThermrResult", "ThermrCancelled", "thermr_deck", "run_thermr", "read_thermr_pendf"]

#: THERMR writes the inelastic here and the elastic at +1 (221 is free gas).
DEFAULT_MTREF = 222


@dataclass
class ThermrResult:
    """The cross sections THERMR wrote, per temperature, in eV and barns."""

    temperatures: List[float]
    mtref: int
    #: ``{temperature: {mt: (energies, sigma)}}`` for MT1, MT2, mtref, mtref+1.
    sections: Dict[float, Dict[int, tuple]] = field(default_factory=dict)
    deck: str = ""
    listing: Optional[str] = None
    emax: float = 0.0
    natom: int = 1


def _header(path: Path):
    from kika.library_index import _endf_header, _head

    header = _endf_header(_head(path))
    if header is None:
        raise ValueError(f"{path.name}: no MF1/MT451 header")
    return header


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{v:g}" for v in values)


def thermr_deck(tsl_mat: int, neutron_mat: int, temperatures: Sequence[float], *,
                icoh: int, natom: int, mtref: int = DEFAULT_MTREF, emax: float = 4.0,
                tol: float = 0.001, nbin: int = 16) -> str:
    """The NJOY input: the neutron tape on unit 20, the TSL tape on 24, the result on 26."""
    temps = _fmt(temperatures)
    return "\n".join([
        "moder", " 20 -21/",
        # All four tolerances written out, as kika's own templates do: the
        # decimal-locale guard (kika.njoy.locale_guard) checks NJOY's echo of
        # them, and it can only check numbers that were sent.
        "reconr", " -21 -22/", " 'pendf for thermr'/", f" {neutron_mat} 0 0/", " 0.001 0 0.01 5e-08/", " 0/",
        "broadr", " -21 -22 -23/", f" {neutron_mat} {len(temperatures)} 0 0 0/", " 0.001 1e+06 0.01 5e-08/",
        f" {temps}/", " 0/",
        "thermr", " 24 -23 -25/",
        f" {tsl_mat} {neutron_mat} {nbin} {len(temperatures)} 2 {icoh} 0 {natom} {mtref} 0/",
        f" {temps}/", f" {tol} {emax}/",
        "moder", " -25 26/",
        "stop", "",
    ])


def read_thermr_pendf(path: Union[str, Path], mts: Sequence[int]) -> Dict[float, Dict[int, tuple]]:
    """MF3 sections *mts* of each temperature block of an ASCII PENDF.

    Each temperature is a material block of its own, its TEMP in the fourth
    record of MF1/MT451. Read here directly: a multi-temperature PENDF is
    several materials in one tape.
    """
    import numpy as np
    from kika.endf.utils import parse_tab1

    wanted = set(mts)
    out: Dict[float, Dict[int, tuple]] = {}
    block_mf1: List[str] = []
    current: Dict[int, List[str]] = {}

    def flush():
        if not block_mf1:
            return
        temperature = _endf_float(block_mf1[3][0:11]) if len(block_mf1) > 3 else 0.0
        sections = {}
        for mt, lines in current.items():
            _, _, xs, ys, _ = parse_tab1(lines, 1)
            sections[mt] = (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        out[round(temperature, 3)] = sections

    with Path(path).open("r", encoding="latin-1") as handle:
        for line in handle:
            try:
                mat, mf, mt = int(line[66:70]), int(line[70:72]), int(line[72:75])
            except ValueError:
                continue
            if mat == 0 and mf == 0 and mt == 0:          # MEND: a temperature block ends
                flush()
                block_mf1, current = [], {}
                continue
            if mf == 1 and mt == 451:
                block_mf1.append(line)
            elif mf == 3 and mt in wanted:
                current.setdefault(mt, []).append(line)
    flush()
    return out


def _endf_float(text: str) -> float:
    from kika.library_index import _endf_float as parse
    return parse(text)


class _Misread(RuntimeError):
    """NJOY read numbers other than the deck's (the decimal-locale bug)."""


class ThermrCancelled(RuntimeError):
    """The run was stopped by its caller; NJOY was killed."""


#: Re-runs when NJOY misread its input (``kika.njoy.locale_guard``). More than
#: RECONR's 8: a misread here is caught within a second or two of the start
#: (the listing is unbuffered), and misreads come in runs on an affected binary.
_MAX_LOCALE_ATTEMPTS = 12
_MODULES = ("moder", "reconr", "broadr", "thermr")


def _run_once(njoy_exe: str, work: Path, deck: str, on_stage, should_stop, tol: float) -> None:
    """One NJOY run, killed when *should_stop* says so; *on_stage* hears each module start."""
    import subprocess
    import threading
    import time
    from kika.njoy.launcher import build_njoy_command

    # gfortran holds NJOY's listing ("output") in a buffer until the run ends,
    # so a misread echoed at the top of it would only be seen once BROADR had
    # spent minutes on a grid with no thinning. Unbuffered, it is seen at once.
    env = {**os.environ, "GFORTRAN_UNBUFFERED_ALL": "y"}
    process = subprocess.Popen(build_njoy_command(njoy_exe), cwd=work, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    process.stdin.write(deck.encode("utf-8"))
    process.stdin.close()

    def read():
        for raw in process.stdout:
            word = raw.decode("latin-1", errors="replace").strip().split(" ")[0].split(".")[0].lower()
            if word in _MODULES and on_stage is not None:
                on_stage(word)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    from kika.njoy.locale_guard import read_listing

    last_check = 0.0
    while process.poll() is None:
        if should_stop is not None and should_stop():
            process.kill()
            process.wait()
            raise ThermrCancelled("THERMR run cancelled")
        # A misread shows in RECONR's echo seconds into the run; a run on
        # misread tolerances can then spend minutes broadening on a grid with
        # no thinning. Stop it there and let the caller run it again.
        now = time.monotonic()
        if now - last_check > 1.0:
            last_check = now
            # Only whole lines: the listing is read while NJOY writes it, and
            # half an echo ("0.0" of "0.001") would look like a misread.
            listing = read_listing(work) or ""
            if _misread(deck, listing[: listing.rfind("\n") + 1], tol):
                process.kill()
                process.wait()
                raise _Misread()
        time.sleep(0.25)
    reader.join(timeout=2)


_THERMR_TOL = re.compile(r"thermr\.\.\.compute thermal.*?tolerance \.+\s*([-+0-9.eEdD]+)", re.DOTALL)


def _misread(deck: str, listing: Optional[str], tol: float) -> bool:
    """Any of RECONR, BROADR or THERMR echoed a number other than the deck's.

    ``locale_guard.check_run`` checks whichever of RECONR and BROADR reads
    first; the misread can start at any later read, so this chain checks the
    three modules that read a tolerance. An echo not written yet is no misread.
    """
    from kika.njoy.locale_guard import (check_broadr_listing, check_reconr_listing,
                                        parse_broadr_card, parse_reconr_card)

    if not listing:
        return False
    reconr = parse_reconr_card(deck)
    if reconr and check_reconr_listing(listing, err=reconr[0], errmax=reconr[1], errint=reconr[2]).corrupted:
        return True
    broadr = parse_broadr_card(deck)
    if broadr and check_broadr_listing(listing, errthn=broadr[0], errmax=broadr[1], errint=broadr[2]).corrupted:
        return True
    return not _thermr_read(listing, tol)


def _thermr_read(listing: Optional[str], tol: float) -> bool:
    """THERMR echoed the tolerance it was sent (the locale bug reads 0.001 as 0)."""
    if not listing:
        return True
    match = _THERMR_TOL.search(listing)
    if match is None:
        return True
    try:
        echoed = float(match.group(1).replace("D", "E"))
    except ValueError:
        return True
    return abs(echoed - tol) <= 1e-6 * max(1.0, tol) or abs(echoed - tol) / tol < 1e-3


def run_thermr(njoy_exe: Union[str, Path], tsl_path: Union[str, Path],
               neutron_path: Union[str, Path], temperatures: Sequence[float], *,
               emax: float = 4.0, mtref: int = DEFAULT_MTREF, tol: float = 0.001,
               workdir: Optional[Union[str, Path]] = None, on_stage=None,
               should_stop=None) -> ThermrResult:
    """Run THERMR for *tsl_path* over the free-atom evaluation *neutron_path*.

    *temperatures* must be ones the TSL tape tabulates (THERMR does not
    interpolate S(α,β) in temperature). *tsl_path* may also be a GNDS TSL
    file: it is written to ENDF-6 by :func:`kika.write` first, since THERMR
    reads only ENDF. Raises ``RuntimeError`` with NJOY's
    listing tail when the run writes no output tape, and
    :class:`ThermrCancelled` when *should_stop* turns true (NJOY is killed).
    *on_stage* is called with each NJOY module as it starts.
    """
    from kika.njoy.locale_guard import read_listing
    from .library import read_tsl_header

    tsl_path, neutron_path = Path(tsl_path), Path(neutron_path)
    converted = None
    from .read import detect_tsl_format
    if detect_tsl_format(tsl_path) == "gnds":
        import kika
        converted = Path(tempfile.mkdtemp(prefix="kika-thermr-gnds-")) / "tsl.endf"
        kika.write(kika.read(str(tsl_path), format="gnds", covariances=False), str(converted), format="endf")
        tsl_path = converted
    tape = read_tsl_header(tsl_path)
    if tape.nsub != 12:
        raise ValueError(f"{tsl_path.name} is not a thermal scattering tape (NSUB={tape.nsub})")
    _, _, neutron_mat, _, neutron_nsub, _ = _header(neutron_path)
    if neutron_nsub not in (10, 0):
        raise ValueError(f"{neutron_path.name} is not an incident-neutron evaluation (NSUB={neutron_nsub})")
    known = tape.temperatures or tape.elastic_temperatures
    missing = [t for t in temperatures if not any(abs(t - k) < 0.05 for k in known)]
    if missing:
        raise ValueError(f"{tsl_path.name} does not tabulate {missing} K; it has {known}")
    if not temperatures:
        raise ValueError("no temperature asked for")

    icoh = 1 if tape.elastic else 0
    natom = max(1, int(round(tape.principal_atoms or 1)))
    deck = thermr_deck(tape.mat, neutron_mat, temperatures, icoh=icoh, natom=natom,
                       mtref=mtref, emax=emax, tol=tol)

    own = workdir is None
    work = Path(tempfile.mkdtemp(prefix="kika-thermr-")) if own else Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(neutron_path, work / "tape20")
        shutil.copyfile(tsl_path, work / "tape24")
        (work / "input").write_text(deck, encoding="utf-8")
        for attempt in range(_MAX_LOCALE_ATTEMPTS):
            # A killed run leaves its listing and tapes behind; the next one
            # must not be judged on them before NJOY rewrites them.
            for name in ("output", "tape21", "tape22", "tape23", "tape25", "tape26"):
                (work / name).unlink(missing_ok=True)
            try:
                _run_once(str(njoy_exe), work, deck, on_stage, should_stop, tol)
            except _Misread:
                continue
            listing = read_listing(work)
            if not _misread(deck, listing, tol):
                break
        else:
            raise RuntimeError(
                f"NJOY misread its input in {_MAX_LOCALE_ATTEMPTS} runs in a row (the decimal-"
                f"separator problem of NJOY builds on a comma-decimal Windows locale)")
        output = work / "tape26"
        if not output.is_file() or output.stat().st_size == 0:
            tail = "\n".join((listing or "").splitlines()[-25:])
            raise RuntimeError(f"NJOY THERMR wrote no output tape.\n{tail}")
        sections = read_thermr_pendf(output, [1, 2, mtref, mtref + 1])
        return ThermrResult(temperatures=sorted(sections), mtref=mtref, sections=sections,
                            deck=deck, listing=listing, emax=emax, natom=natom)
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
        if converted is not None:
            shutil.rmtree(converted.parent, ignore_errors=True)

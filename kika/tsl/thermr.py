"""The thermal cross sections of a TSL tape, computed by NJOY THERMR.

kika computes the two elastic parts of a thermal scattering law in closed form
(``kika.nuclear_data.model.thermal_scattering``) and the incoherent
inelastic tables with its own THERMR (:mod:`kika.processing.thermal`).
:func:`run_thermr` runs NJOY's instead, as the reference to compare with, in
the chain every NJOY processing of a TSL material uses::

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
    """The cross sections THERMR wrote, per temperature, in eV and barns.

    Returned by :func:`run_thermr`.

    Attributes
    ----------
    temperatures
        The temperatures of the PENDF, in K, in increasing order.
    mtref
        The MT under which THERMR wrote the inelastic cross section; the elastic
        one, when the tape has it, is ``mtref + 1``.
    sections
        ``{temperature: {mt: (energies, sigma)}}``, energies in eV and σ in
        barns, for those of MT1, MT2, ``mtref`` and ``mtref + 1`` that the PENDF
        holds. The temperature keys are in K, rounded to three decimals.
    deck
        The NJOY input that was run.
    listing
        NJOY's output listing of the run, or None if it could not be read.
    emax
        The upper energy of the thermal treatment, in eV, as asked for.
    natom
        The number of principal atoms given to THERMR (at least 1).
    """

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
    """The NJOY input: the neutron tape on unit 20, the TSL tape on 24, the result on 26.

    The RECONR and BROADR cards are fixed: tolerance 0.001, maximum error 0.01
    and integration error 5e-8 in both, and BROADR's thinning limit of 1e6 eV.
    The temperatures and the arguments below set the THERMR card.

    Parameters
    ----------
    tsl_mat, neutron_mat
        MAT of the TSL tape and of the free-atom neutron evaluation.
    temperatures
        In K, written to BROADR and THERMR with six significant digits.
    icoh
        THERMR's ICOH, the elastic option. :func:`run_thermr` gives 1 when the
        tape has an elastic section and 0 when it has none.
    natom
        THERMR's NATOM, the number of principal atoms in the compound.
    mtref
        The MT THERMR writes the inelastic cross section under.
    emax
        The upper energy of the thermal treatment, in eV.
    tol
        THERMR's tolerance (the first value of its energy card).
    nbin
        The number of equiprobable angles (cosine bins) of THERMR.

    Returns
    -------
    str
        The input, one NJOY card per line, ending with ``stop``.
    """
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

    Parameters
    ----------
    path
        An ASCII PENDF.
    mts
        The MF3 sections to keep.

    Returns
    -------
    dict
        ``{temperature: {mt: (energies, sigma)}}``: energies in eV and σ in barns
        as arrays, the interpolation laws of the section not kept. The key is the
        block's TEMP in K, rounded to three decimals (0.0 when its MF1/MT451 is
        shorter than four lines). A section of *mts* a block lacks is missing from
        its dict, and a block with no MF1/MT451 is not returned.
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

    THERMR does not interpolate S(α,β) in temperature, so the temperatures
    must be ones the TSL tape tabulates. The deck is that of :func:`thermr_deck`
    with ``icoh`` 1 when the tape has an elastic section, ``natom`` the tape's
    number of principal atoms (B(6) of MF7/MT4, rounded, at least 1) and the
    default ``nbin`` of 16. NJOY is run again, up to 12 times, when its output
    shows it misread the numbers of the input (the decimal-separator problem of
    some Windows builds).

    Parameters
    ----------
    njoy_exe
        The NJOY executable.
    tsl_path
        The TSL tape: ENDF-6 with NSUB 12, or a GNDS TSL file, which THERMR
        cannot read and which is first written to ENDF-6 with :func:`kika.write`
        in a temporary folder that is removed afterwards.
    neutron_path
        The free-atom neutron evaluation of the principal atom, an ENDF-6 tape
        with NSUB 10 (or 0).
    temperatures
        In K. Each must match a temperature of the tape (its S(α,β), or its
        elastic part when it has none) within 0.05 K; the temperatures given, not
        the tape's, go into the deck.
    emax
        The upper energy of the thermal treatment, in eV, written unchanged into
        THERMR's energy card. By NJOY's convention, below it the bound-atom data
        replaces the free-atom elastic.
    mtref
        The MT the inelastic cross section is written under; the elastic goes to
        ``mtref + 1``.
    tol
        THERMR's tolerance.
    workdir
        The folder NJOY runs in. None uses a temporary folder, removed when the
        call ends. A folder given is created if needed and kept, with the tapes
        (``tape20`` to ``tape26``), ``input`` and ``output`` in it. Before each
        run ``output`` and ``tape21``, ``tape22``, ``tape23``, ``tape25`` and
        ``tape26`` are deleted, and ``tape20``, ``tape24`` and ``input`` are
        overwritten.
    on_stage
        A callable called with ``"moder"``, ``"reconr"``, ``"broadr"`` or
        ``"thermr"`` when NJOY's output reaches that module. The same name can
        come more than once.
    should_stop
        A callable with no arguments, polled about every 0.25 s while NJOY runs.
        When it returns true NJOY is killed and :class:`ThermrCancelled` is
        raised.

    Returns
    -------
    ThermrResult
        The MF3 sections MT1, MT2, ``mtref`` and ``mtref + 1`` of each
        temperature of the PENDF NJOY wrote, with the deck and the listing.

    Raises
    ------
    ValueError
        If *temperatures* is empty, if one of them is not tabulated by the tape,
        if *tsl_path* is not a TSL tape (NSUB other than 12) or *neutron_path* is
        not an incident-neutron evaluation, or if either file has no MF1/MT451
        header.
    RuntimeError
        If NJOY misread its input in 12 runs in a row, or wrote no output tape
        (the message ends with the last 25 lines of its listing).
    ThermrCancelled
        If *should_stop* turned true.
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

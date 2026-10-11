"""What the writer takes from NJOY until kika computes it (roadmap D7).

The URR probability tables (PURR), the KERMA -- total, MT 301, and with the
photons deposited locally, MT 901 -- (HEATR, twice) and the damage energy
(HEATR, MT 444) are not computed by kika yet (``openmc_hdf5_roadmap.md``
§2.1, H8). They come from NJOY run on the **same ENDF tape**, with the cards
OpenMC's ``make_ace`` uses (``openmc/data/njoy.py``):

    reconr (err) → broadr (T) → heatr local (``4 0 0 1 0 ed``, 302 318 402 444)
    → heatr (``4 0 0 0 0 ed``) → purr (``1 20 64``)

one deck per temperature. ACER is not run: the probability tables are read
from PURR's MF2/MT153 and converted the way ACER converts them
(``acefc.f90:5945-5985``), which saves most of the run. They reach kika's grid
**without mixing NJOY's cross sections into kika's**:

- KERMA and damage are read off the HEATR PENDF tapes (MF3/MT 1, 301, 318,
  444) and turned into a **number per collision**, ``k(E)/σ_t(E)`` on NJOY's
  grid, the fission KERMA (MT 318) removed. The writer interpolates it onto
  kika's grid, multiplies by kika's σ_t -- how ``IncidentNeutron.from_ace``
  builds MT 301 from the ACE heating number -- and adds the fission part as
  ``from_njoy`` does (``neutron.py:786-842``): ``(EFR + EB) σ_f`` for 301,
  ``(EFR + EGP + EGD + EB) σ_f`` for 901, with kika's σ_f.
- The probability tables are factors on the smooth cross sections
  (``multiply_smooth = 1``, as in the official libraries), so they combine with
  kika's σ as they are.

**Which NJOY.** A MinGW NJOY on a comma-decimal Windows can misread the ENDF
tape itself after its input card was read correctly (``kika/njoy/
locale_guard.py``) and die in RECONR's ``lunion`` with an "ill-behaved
threshold" the evaluation does not have: on ENDF/B-VIII.1 U-235 it does on
every run (6/6, 2026-10-11) where the same NJOY 2016.78 built for Linux runs
clean. :func:`wslNjoy` runs a Linux NJOY through ``wsl.exe``, in the WSL's own
file system (a run on ``/mnt/c`` is several times slower), copying back only
the tapes read here.

Without NJOY the writer raises :class:`MissingNJOYPieceError` naming the
pieces; leaving one out is a request (``omit=``), and the report says so.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .exceptions import MissingNJOYPieceError
from .names import EV_PER_MEV

__all__ = ["NJOY_PIECES", "TemperaturePieces", "NjoyPieces", "runNjoyPieces", "perCollision",
           "wslNjoy", "deckFor", "readMT153", "LOCALE_SYMPTOM"]

#: The pieces D7 takes from NJOY.
NJOY_PIECES = ("urr", "kerma", "damage")

#: What a Windows NJOY prints when the decimal-locale misread hits an ENDF read.
LOCALE_SYMPTOM = "ill-behaved threshold"

#: The tapes the deck writes and this module reads.
TAPES = ("tape23", "tape24", "tape25")


@dataclass
class TemperaturePieces:
    """What one temperature's NJOY run gave, on NJOY's grids.

    ``kerma``/``kermaLocal``/``damage``: ``(E, number per collision)`` in eV
    (the fission KERMA removed from both KERMAs). ``urr``: a
    :class:`~kika.nuclear_data.model.ProbabilityTables` or ``None``.
    """

    temperature: float
    kerma: Optional[Tuple[np.ndarray, np.ndarray]] = None
    kermaLocal: Optional[Tuple[np.ndarray, np.ndarray]] = None
    damage: Optional[Tuple[np.ndarray, np.ndarray]] = None
    urr: Optional[object] = None


@dataclass
class NjoyPieces:
    temperatures: Dict[float, TemperaturePieces] = field(default_factory=dict)
    njoy: Optional[str] = None
    decks: Dict[float, str] = field(default_factory=dict)


def deckFor(mat: int, temperature: float, error: float = 0.001, ed: float = 0.0,
            urr: bool = True, library: str = "kika") -> str:
    """One temperature's NJOY input, card for card as OpenMC's ``make_ace`` writes it.

    Two departures, both stated: RECONR's card carries a positive ``errmax``
    and ``errint`` (``10*err`` and 5e-8, NJOY's own defaults when ``err`` is
    0.001) so kika's locale guard can tell a misread card from a default; and
    there is no ACER (see the module docstring).
    """
    lines = [
        "reconr", "20 21", f"'{library} PENDF'/", f"{mat} 2/",
        f"{error} 0. {10 * error} 5.0e-8/", "'kika'/", "'Processed by NJOY for kika'/", "0/",
        "broadr", "20 21 22", f"{mat} 1 0 0 0. /", f"{error}/", f"{temperature}", "0/",
        "heatr", "20 22 23 /", f"{mat} 4 0 0 1 0 {ed}/", "302 318 402 444 /",
        "heatr", "20 22 24 /", f"{mat} 4 0 0 0 0 {ed}/", "302 318 402 444 /",
    ]
    if urr:
        lines += ["purr", "20 24 25 /", f"{mat} 1 1 20 64 /", f"{temperature}", "1.e10", "0/"]
    lines.append("stop")
    return "\n".join(lines) + "\n"


def perCollision(energy, kerma, total):
    """``kerma/σ_t`` where σ_t > 0, else 0: a number of eV per collision."""
    energy = np.asarray(energy, dtype=float)
    out = np.zeros_like(energy)
    np.divide(kerma, total, out=out, where=np.asarray(total) > 0)
    return energy, out


def _section(tape, mt):
    from kika.endf.read_endf import read_endf

    data = read_endf(str(tape), mf_numbers=[3])
    sections = data.mf[3].mt if 3 in data.mf else {}
    if mt not in sections:
        return None
    s = sections[mt]
    return np.asarray(s.energies, dtype=float), np.asarray(s.cross_sections, dtype=float)


def _onGrid(energy, pair):
    if pair is None:
        return np.zeros_like(energy)
    return np.interp(energy, pair[0], pair[1], left=0.0, right=0.0)


def _heating(tape):
    total = _section(tape, 1)
    kerma = _section(tape, 301)
    if total is None or kerma is None:
        raise MissingNJOYPieceError(f"HEATR wrote no MF3/MT1 or MT301 on {tape}", ("kerma",))
    energy, sigma = total
    k301 = _onGrid(energy, kerma)
    k318 = _onGrid(energy, _section(tape, 318))
    k444 = _section(tape, 444)
    damage = None if k444 is None else perCollision(energy, _onGrid(energy, k444), sigma)
    return perCollision(energy, k301 - k318, sigma), damage


def _endfFloat(field: str) -> float:
    text = field.strip()
    if not text:
        return 0.0
    for i in range(1, len(text)):
        if text[i] in "+-" and text[i - 1] not in "eE":
            text = text[:i] + "e" + text[i:]
            break
    return float(text)


def _sigfig(value: float, digits: int = 7) -> float:
    """NJOY's ``sigfig(x, digits, 0)``: round to that many significant figures."""
    return float(f"{value:.{digits - 1}e}") if value else 0.0


def readMT153(tape) -> Optional[object]:
    """PURR's MF2/MT153 → :class:`ProbabilityTables`, converted as ACER converts it.

    MT153 is a HEAD ``(ZA, AWR, IINEL, IABSO, INT, NBIN)`` and a LIST
    ``(T, 0, LSSF, 0, NW, NE)`` whose body is, per energy, ``E`` and six blocks
    of NBIN values: probability, total, elastic, fission, capture, heating
    (``purr.f90:472-560``). ACER accumulates the probabilities into a CDF
    rounded to seven figures with the last bin set to 1, keeps the factors as
    they are, and writes the energies in MeV to seven figures
    (``acefc.f90:5945-5985``). OpenMC then multiplies the heating row by 1e6
    whatever it holds (``urr.py:212``), which the official files show; that is
    reproduced too.
    """
    from kika.nuclear_data.model import ProbabilityTables

    lines = [l for l in Path(tape).read_text(errors="replace").splitlines()
             if len(l) >= 75 and l[70:72].strip() == "2" and l[72:75].strip() == "153"]
    if not lines:
        return None
    head = [_endfFloat(lines[0][i:i + 11]) for i in range(0, 66, 11)]
    iinel, iabso, nbin = int(head[2]), int(head[3]), int(head[5])
    lst = [_endfFloat(lines[1][i:i + 11]) for i in range(0, 66, 11)]
    lssf, nw, ne = int(lst[2]), int(lst[4]), int(lst[5])
    values: List[float] = []
    for line in lines[2:]:
        for i in range(0, 66, 11):
            if len(values) < nw:
                values.append(_endfFloat(line[i:i + 11]))
    if iinel == 0:
        # Old competition flags, as ACER reads them.
        iinel, iabso = -1, -1
    data = np.asarray(values, dtype=float).reshape(ne, 1 + 6 * nbin)
    energies = np.array([_sigfig(e / EV_PER_MEV) * EV_PER_MEV for e in data[:, 0]])
    table = np.zeros((ne, 6, nbin))
    for k in range(ne):
        block = data[k, 1:].reshape(6, nbin)
        cdf = np.array([_sigfig(c) for c in np.cumsum(block[0])])
        cdf[-1] = 1.0
        table[k, 0] = cdf
        table[k, 1:5] = block[1:5]
        heating = block[5] if lssf == 1 else block[5] / EV_PER_MEV
        table[k, 5] = heating * EV_PER_MEV
    return ProbabilityTables(energies=energies, table=table, interpolation=2,
                             inelastic=iinel, absorption=iabso, multiplySmooth=lssf == 1)


def wslNjoy(linuxExecutable: str, distro: str = "Ubuntu", libraryPath: Optional[str] = None):
    """The command that runs a Linux NJOY through ``wsl.exe``, in the WSL's ``/tmp``.

    The deck arrives on standard input. The working directory (a Windows
    folder holding ``tape20``) is forwarded by ``wsl.exe``; the run happens in
    a fresh ``mktemp`` directory and only NJOY's listing and the tapes this
    module reads are copied back. ``libraryPath`` sets ``LD_LIBRARY_PATH`` for
    a binary linked against a private toolchain.
    """
    env = "LC_ALL=C LC_NUMERIC=C LANG=C" + (f" LD_LIBRARY_PATH='{libraryPath}'" if libraryPath else "")
    # Every tape present before the run is an input; every other tape and the
    # listing are copied back.
    script = (f"set -e; here=\"$PWD\"; work=$(mktemp -d); inputs=$(ls tape* 2>/dev/null || true); "
              f"for f in $inputs; do cp \"$f\" \"$work\"/; done; cd \"$work\"; "
              f"env {env} '{linuxExecutable}' || true; "
              f"for f in output tape*; do case \" $inputs \" in *\" $f \"*) ;; "
              f"*) [ -f \"$f\" ] && cp \"$f\" \"$here\"/ ;; esac; done; "
              f"cd /; rm -rf \"$work\"")
    return ["wsl.exe", "-d", distro, "--exec", "bash", "-c", script]


def _runCommand(command, folder: Path, deck: str):
    return subprocess.run(list(command), cwd=folder, input=deck.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def runNjoyPieces(endf, temperatures: Iterable[float], *, mat: int, error: float = 0.001,
                  ed: float = 0.0, urr: bool = True, njoy=None, workdir=None,
                  keep: bool = False) -> NjoyPieces:
    """Run NJOY once per temperature and collect the pieces.

    ``njoy`` is the executable (default: ``$NJOY_EXECUTABLE``, then ``njoy`` on
    PATH), or a whole command line as a list -- :func:`wslNjoy` builds the one
    that runs a Linux NJOY from Windows.
    """
    from kika.njoy.run_njoy import _run_njoy_deck
    from kika.processing.njoy_pendf_cache import NjoyNotFoundError, find_njoy_executable

    command = None
    if isinstance(njoy, (list, tuple)):
        command = list(njoy)
        executable = "WSL NJOY"
    else:
        try:
            executable = find_njoy_executable(njoy, why="the OpenMC HDF5 writer's URR tables, "
                                                        "KERMA and damage (roadmap D7)")
        except NjoyNotFoundError as exc:
            raise MissingNJOYPieceError(
                f"{exc}\nWithout NJOY these pieces cannot be produced: "
                f"{', '.join(NJOY_PIECES)}. Write without them only by asking for it: "
                f"omit=(...)", NJOY_PIECES) from exc
    out = NjoyPieces(njoy=" ".join(command) if command else str(executable))
    base = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="kika_njoy_"))
    try:
        for temperature in temperatures:
            folder = base / f"T{temperature:g}"
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(endf, folder / "tape20")
            deck = deckFor(mat, temperature, error, ed, urr)
            out.decks[float(temperature)] = deck
            (folder / "input").write_text(deck)
            result = (_runCommand(command, folder, deck) if command
                      else _run_njoy_deck(executable, folder, deck))
            (folder / "output.txt").write_bytes(result.stdout or b"")
            printed = (result.stdout or b"").decode("utf-8", "replace")
            if LOCALE_SYMPTOM in printed and command is None and os.name == "nt":
                raise MissingNJOYPieceError(
                    f"NJOY ({executable}) died in RECONR with '{LOCALE_SYMPTOM}' at "
                    f"{temperature} K. On Windows this is the known decimal-locale misread "
                    f"of the ENDF tape (kika/njoy/locale_guard.py), not a defect of the "
                    f"evaluation; run a Linux NJOY instead, njoy=wslNjoy(...). Listing: "
                    f"{folder / 'output'}", NJOY_PIECES)
            for tape in ("tape23", "tape24"):
                if not (folder / tape).is_file() or (folder / tape).stat().st_size == 0:
                    raise MissingNJOYPieceError(
                        f"NJOY did not write {tape} at {temperature} K; see {folder / 'output.txt'}",
                        ("kerma", "damage"))
            kermaLocal, _ = _heating(folder / "tape23")
            kerma, damage = _heating(folder / "tape24")
            pieces = TemperaturePieces(temperature=float(temperature), kerma=kerma,
                                       kermaLocal=kermaLocal, damage=damage)
            if urr and (folder / "tape25").is_file():
                pieces.urr = readMT153(folder / "tape25")
            out.temperatures[float(temperature)] = pieces
    finally:
        if not keep and workdir is None:
            shutil.rmtree(base, ignore_errors=True)
    return out

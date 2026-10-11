"""Running NJOY for what kika still takes from it: the thermal inelastic tables (D7, H9c).

The neutron files no longer need NJOY: their URR tables, KERMA and damage are
kika's (``kika.processing.resonances.probability_tables``,
``kika.processing.heating``; roadmap H8). What remains is THERMR + ACER for a
thermal evaluation's incoherent inelastic tables, run by
:mod:`kika.hdf5.thermal_processing`.

**Which NJOY.** A MinGW NJOY on a comma-decimal Windows can misread the ENDF
tape itself after its input card was read correctly (``kika/njoy/
locale_guard.py``) and die in RECONR's ``lunion`` with an "ill-behaved
threshold" the evaluation does not have: on ENDF/B-VIII.1 U-235 it does on
every run (6/6, 2026-10-11) where the same NJOY 2016.78 built for Linux runs
clean. :func:`wslNjoy` runs a Linux NJOY through ``wsl.exe``, in the WSL's own
file system (a run on ``/mnt/c`` is several times slower), copying back only
NJOY's listing and the tapes it wrote.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

__all__ = ["wslNjoy", "runCommand", "LOCALE_SYMPTOM"]

#: What a Windows NJOY prints when the decimal-locale misread hits an ENDF read.
LOCALE_SYMPTOM = "ill-behaved threshold"


def wslNjoy(linuxExecutable: str, distro: str = "Ubuntu", libraryPath: Optional[str] = None):
    """The command that runs a Linux NJOY through ``wsl.exe``, in the WSL's ``/tmp``.

    The deck arrives on standard input. The working directory (a Windows
    folder holding the input tapes) is forwarded by ``wsl.exe``; the run
    happens in a fresh ``mktemp`` directory and only NJOY's listing and the
    tapes it wrote are copied back. ``libraryPath`` sets ``LD_LIBRARY_PATH``
    for a binary linked against a private toolchain.
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


def runCommand(command, folder: Path, deck: str):
    """Run ``command`` (a whole command line) in ``folder`` with ``deck`` on standard input."""
    return subprocess.run(list(command), cwd=folder, input=deck.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

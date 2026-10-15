"""A run that dies on the decimal-locale misread is run again, wherever it dies.

The listing check reads RECONR's or BROADR's echo; a misread that stops a later
module (an ACER card: "Error during floating point read") left that echo clean,
and the run was not repeated. It is now, and a genuine failure still returns.
A run whose echo shows the misread is stopped while it runs: with ``err = 0``
RECONR may never finish.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from types import SimpleNamespace

from kika.njoy import locale_guard

# ``kika.njoy.run_njoy`` is also the name of a function the package re-exports.
run_njoy = importlib.import_module("kika.njoy.run_njoy")


def _fake_runs(monkeypatch, outcomes):
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        code, text = outcomes[min(len(calls), len(outcomes)) - 1]
        return SimpleNamespace(returncode=code, stdout=text.encode())

    monkeypatch.setattr(run_njoy, "_watchedRun", fake)
    monkeypatch.setattr(locale_guard, "check_run",
                        lambda deck, listing: SimpleNamespace(corrupted=False, detail=""))
    monkeypatch.setattr(locale_guard, "read_listing", lambda workdir: None)
    return calls


def test_a_floating_point_read_crash_is_run_again(monkeypatch, tmp_path):
    calls = _fake_runs(monkeypatch, [
        (2, "At line 295 of file acer.f90\nFortran runtime error: Error during floating point read"),
        (0, "acer... done"),
    ])
    result = run_njoy._run_njoy_deck("njoy.exe", tmp_path, "acer\n")
    assert result.returncode == 0 and len(calls) == 2


def test_another_failure_is_returned_at_once(monkeypatch, tmp_path):
    calls = _fake_runs(monkeypatch, [(2, "***error in acer***bad material")])
    result = run_njoy._run_njoy_deck("njoy.exe", tmp_path, "acer\n")
    assert result.returncode == 2 and len(calls) == 1


_DECK = "moder\n 20 -21/\nreconr\n -21 -22/\n 'title'/\n 125 0 0/\n 0.001 0 0.01 5e-08/\n 0/\nstop\n"


def _fake_njoy(monkeypatch, err_echo: str, then: str):
    """An "NJOY" that echoes RECONR's tolerance into its listing, then *then*."""
    script = (
        "import sys, time\n"
        "sys.stdin.read()\n"
        "open('output', 'w').write(' reconstruction tolerance ............. "
        f"{err_echo}\\n')\n"
        f"{then}\n"
    )
    monkeypatch.setattr(run_njoy, "build_njoy_command", lambda exe: [sys.executable, "-c", script])


def test_a_run_that_misreads_its_tolerance_is_stopped_while_it_runs(monkeypatch, tmp_path):
    # err read as 0: the real RECONR may never finish (Pu-239, 47 min of CPU).
    _fake_njoy(monkeypatch, "0.000", "time.sleep(600)")
    start = time.monotonic()
    result = run_njoy._watchedRun("njoy.exe", tmp_path, _DECK)
    assert time.monotonic() - start < 30
    assert result.returncode != 0
    assert locale_guard.check_run(_DECK, locale_guard.read_listing(tmp_path)).corrupted


def test_a_run_that_reads_its_tolerance_is_left_to_finish(monkeypatch, tmp_path):
    _fake_njoy(monkeypatch, "0.001", "time.sleep(1.2); sys.exit(0)")
    result = run_njoy._watchedRun("njoy.exe", tmp_path, _DECK)
    assert result.returncode == 0

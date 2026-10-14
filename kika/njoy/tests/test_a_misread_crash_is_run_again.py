"""A run that dies on the decimal-locale misread is run again, wherever it dies.

The listing check reads RECONR's or BROADR's echo; a misread that stops a later
module (an ACER card: "Error during floating point read") left that echo clean,
and the run was not repeated. It is now, and a genuine failure still returns.
"""
from __future__ import annotations

import importlib
import subprocess
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

    monkeypatch.setattr(subprocess, "run", fake)
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

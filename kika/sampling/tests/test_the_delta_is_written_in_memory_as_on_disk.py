"""Writing a sample in memory gives the bytes the chain of files gave.

``endf-delta`` used to cost a sample its tape's size in reads and writes several
times over -- a copy, a read and a write per replaced section, another pair for
the directory -- and the ACE emitter added two MF3 splices and an LRP edit,
each one a file read back and written again. ``_Tape`` holds the source once
and composes the same steps in memory, writing each file once. Nothing about
that is allowed to show in the output, so the gate is the strongest available:
the file-based chain (still there, for a tape that is not plain ASCII) is run
on the same realisation and the bytes are compared, for a CRLF tape and for the
same tape with LF endings.

The micro-tape has a resonance region, and perturbing MF33 there reconstructs
σ(E) through NJOY RECONR, so every test here spawns NJOY: each requests the
root ``njoy_exe`` fixture (which marks it ``njoy`` and skips it where no binary
is found, as on CI) and points ``$NJOY_EXECUTABLE`` at that same binary, so the
library runs the one the fixture resolved rather than whatever is on PATH.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kika.sampling.model_perturbation as mp
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = DATA / "micro_fe56_xs_and_angular.endf"
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}


@pytest.fixture(autouse=True)
def _njoy(njoy_exe, monkeypatch):
    monkeypatch.setenv("NJOY_EXECUTABLE", str(njoy_exe))


def _tape(tmp_path: Path, newline: bytes) -> Path:
    raw = TAPE.read_bytes().replace(b"\r\n", b"\n")
    path = tmp_path / f"tape_{len(newline)}.endf"
    path.write_bytes(raw.replace(b"\n", newline))
    return path


def _run(tape: Path, out: Path):
    mp._TAPES.clear()
    return perturbFromModel(str(tape), REQUEST, 2, seed=20260906, outputDir=out,
                            formats=("endf-delta",))


@pytest.mark.parametrize("newline", [b"\r\n", b"\n"], ids=["crlf", "lf"])
def test_the_delta_matches_the_file_based_chain(tmp_path, monkeypatch, newline):
    tape = _tape(tmp_path, newline)
    assert mp._tape(tape).inMemory
    inMemory = _run(tape, tmp_path / "memory")

    monkeypatch.setattr(mp._Tape, "__init__", _onDisk(mp._Tape.__init__))
    onDisk = _run(tape, tmp_path / "disk")

    for a, b in zip(inMemory.samples, onDisk.samples):
        assert Path(a["files"]["endf-delta"]).read_bytes() == \
            Path(b["files"]["endf-delta"]).read_bytes()


@pytest.mark.parametrize("newline", [b"\r\n", b"\n"], ids=["crlf", "lf"])
def test_the_njoy_inputs_match_the_file_based_splices(tmp_path, monkeypatch, newline):
    """``_njoyInputs``: perturbed PENDF and NJOY's ENDF, in memory and on disk.

    Any tape with an MF3 stands in for RECONR's PENDF; the splice does not
    look at what the numbers mean.
    """
    tape = _tape(tmp_path, newline)
    run = _run(tape, tmp_path / "run")
    delta = Path(run.samples[0]["files"]["endf-delta"])
    pendf = DATA / "micro_fe56_mf33.endf"

    mp._TAPES.clear()
    memory = mp._njoyInputs(delta, tape, pendf, tmp_path / "memory")
    mp._TAPES.clear()
    monkeypatch.setattr(mp._Tape, "__init__", _onDisk(mp._Tape.__init__))
    disk = mp._njoyInputs(delta, tape, pendf, tmp_path / "disk")

    for a, b in zip(memory, disk):
        assert a.read_bytes() == b.read_bytes(), a.name


def _onDisk(init):
    def forced(self, raw):
        init(self, raw)
        self.inMemory = False
    return forced

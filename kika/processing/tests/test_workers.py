"""The workers count: the caller's, else KIKA_WORKERS, else each stage's default."""
import multiprocessing
import os
from pathlib import Path

import pytest

from kika.processing import workers as w

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(w.ENV, raising=False)


def test_the_callers_count_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(w.ENV, "16")
    assert w.resolve(3, default=4) == 3
    assert w.resolve(None, default=4) == 16


def test_without_either_the_stage_default_is_used():
    assert w.resolve(None, default=4) == 4
    assert w.verification_default() == max(1, min(4, (os.cpu_count() or 1) // 2))
    assert w.pool_default() == max(1, min(8, os.cpu_count() or 1))


def test_the_defaults_are_not_capped_when_asked_for_more(monkeypatch):
    # A server asks for more than the conservative defaults; nothing clips it.
    monkeypatch.setenv(w.ENV, "64")
    from kika.processing.resonances.parallel import default_workers
    from kika.hdf5.processing import _poolWorkers
    assert default_workers() == 64
    assert _poolWorkers(None, jobs=100) == 64 and _poolWorkers(48, jobs=100) == 48
    assert _poolWorkers(None, jobs=5) == 5


@pytest.mark.parametrize("raw", ["0", "-2", "four", "2.5"])
def test_a_mistyped_environment_value_is_an_error(monkeypatch, raw):
    monkeypatch.setenv(w.ENV, raw)
    with pytest.raises(ValueError, match="KIKA_WORKERS"):
        w.resolve(None, default=4)


def test_an_empty_environment_value_is_unset(monkeypatch):
    monkeypatch.setenv(w.ENV, " ")
    assert w.resolve(None, default=2) == 2


def test_a_daemonic_process_runs_its_pools_in_process(monkeypatch):
    class Daemon:
        daemon = True
    monkeypatch.setattr(multiprocessing, "current_process", lambda: Daemon())
    from kika.hdf5.processing import _poolWorkers
    assert not w.can_spawn() and _poolWorkers(8, jobs=100) == 1


def test_reconstruct_endf_hands_its_workers_to_the_reconstruction(monkeypatch):
    from kika.endf.reconstruction import reconstruct_endf
    from kika.processing.resonances import suite as rs
    seen = []
    real = rs.reconstruct_suite

    def spy(*args, workers=None, **kwargs):
        seen.append(workers)
        return real(*args, workers=workers, **kwargs)
    monkeypatch.setattr(rs, "reconstruct_suite", spy)
    import kika.processing.resonances as res
    monkeypatch.setattr(res, "reconstruct_suite", spy)
    rec = reconstruct_endf(TAPE, workers=6)
    assert seen == [6] and rec.workers == 6

"""A refusal raised in a worker process reaches the caller whole.

The processing broadens, tabulates URR tables and computes damage in worker
processes. An exception that pickle cannot rebuild arrives as BrokenProcessPool,
with its reason lost: He-3 of ENDF/B-VIII.1, refused by the Doppler stage for its
conversion losses, came back that way with more than one worker.
"""
from __future__ import annotations

import pickle

import pytest

from kika.endf.process import ProcessingOutputError
from kika.processing.doppler.core import DopplerConvergenceError, DopplerInputError, UnsupportedDopplerError


@pytest.mark.parametrize("error", [
    DopplerInputError("invalid-quantity", "temperature", energy=1.0, budget=2.0),
    UnsupportedDopplerError("conversion-not-clean", "cross-section conversion losses"),
    DopplerConvergenceError("budget", "not converged", energy=3.0),
    ProcessingOutputError("pendf", "could not be written"),
])
def test_a_processing_refusal_survives_pickle_with_its_fields(error):
    back = pickle.loads(pickle.dumps(error))
    assert type(back) is type(error)
    assert str(back) == str(error)
    assert back.category == error.category
    for field in ("energy", "budget"):
        assert getattr(back, field, None) == getattr(error, field, None)


def test_every_processing_exception_survives_pickle():
    import importlib
    import inspect
    import pkgutil

    import kika.hdf5
    import kika.processing

    failures = []
    for package in (kika.processing, kika.hdf5):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if ".tests" in info.name:
                continue
            module = importlib.import_module(info.name)
            for name, cls in vars(module).items():
                if not (inspect.isclass(cls) and issubclass(cls, Exception) and cls.__module__ == info.name):
                    continue
                required = [p for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
                            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                error = cls(*(["x"] * max(1, len(required))))
                try:
                    pickle.loads(pickle.dumps(error))
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{info.name}.{name}: {exc}")
    assert not failures, failures

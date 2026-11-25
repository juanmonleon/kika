"""A GNDS source with resonances is perturbed to a tape when its MF2 can be derived.

The tape a GNDS source is written to needs MF2: without it MF3 in the
resonance region is only the background. MF2 is now derived from a
provenance-free model (Reich-Moore, Breit-Wigner, R-Matrix Limited and the
unresolved range), so the refusal is the deriver's own verdict: a range it
cannot state refuses, with its reason; a declared approximation (LAD) does not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kika
from kika.endf import read_endf
from kika.sampling.model_perturbation import _resonanceRefusal, perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
ANGULAR = {34: {"mt": [2], "index": [1, 2, 3]}}


def _viaGnds(name, tmp_path):
    path = tmp_path / f"{name}.gnds.xml"
    kika.write(kika.read(str(DATA / name)), str(path), format="gnds")
    return path


@pytest.mark.parametrize("name", ["micro_mn55_mf2.endf", "micro_ne20_covcheck.endf"])
def test_a_range_the_deriver_writes_is_not_refused(name, tmp_path):
    suite = kika.read(str(_viaGnds(name, tmp_path)))
    assert suite.resonances.resolved
    assert _resonanceRefusal(suite, 2525) is None


def test_a_range_the_deriver_cannot_write_is_refused_with_its_reason(tmp_path):
    from kika.nuclear_data.model.resonances import RadiusPolicy

    suite = kika.read(str(_viaGnds("micro_ne20_covcheck.endf", tmp_path)))
    formalism = suite.resonances.resolved[0].formalism
    formalism.radiusPolicy = RadiusPolicy(channelMode="constant", channelRadius=4.0)
    why = _resonanceRefusal(suite, 1025)
    assert why is not None and "NAPS=2" in why


def test_a_gnds_source_with_resonances_is_written_as_a_tape(tmp_path):
    gnds = _viaGnds("micro_fe56_xs_and_angular.endf", tmp_path)
    run = perturbFromModel(str(gnds), ANGULAR, 1, seed=5, outputDir=tmp_path / "out",
                           formats=("endf-tape",))
    tape = run.samples[0]["files"]["endf-tape"]
    assert 2 in read_endf(str(tape), mf_numbers=[2]).mf

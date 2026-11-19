"""The resonance region is rebuilt by kika or by NJOY, and NJOY can be missing.

``perturbFromModel(reconstructionEngine=...)`` picks who rebuilds σ(E) at 0 K
before the draw: kika (the default) or NJOY RECONR. NJOY that is missing or
fails gives way to kika for the whole run, with a warning, so an ensemble never
mixes engines. The draw does not depend on the engine: both give the same
factors, and their reconstructed cross sections agree within the tolerance.

The full tape is needed (the micro-tapes keep no MF2 to rebuild); the
equivalence also needs NJOY.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_perturbation import perturbFromModel

ZA = 26056
TOLERANCE = 1e-3


def _values(path, mt, energies):
    section = read_endf(str(path), mf_numbers=[3]).get_file(3).sections[mt]
    return np.interp(energies, np.asarray(section.energies, float),
                     np.asarray(section.cross_sections, float))


def _run(tape, out, **kwargs):
    run = perturbFromModel(str(tape), {33: [2]}, 1, seed=3, outputDir=out,
                           formats=("endf-delta",), **kwargs)
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    return run, meta, Path(meta["resonanceRegion"]["base"])


@pytest.fixture(scope="module")
def kikaRun(fe56_b81_tape, tmp_path_factory):
    return _run(fe56_b81_tape, tmp_path_factory.mktemp("kika"))


@pytest.mark.slow
def test_kika_rebuilds_the_region_by_default(kikaRun):
    run, meta, base = kikaRun
    assert meta["resonanceRegion"]["reconstructed"] is True
    assert meta["resonanceRegion"]["engine"] == "kika"
    assert run.log.of("reconstructed")[0].payload["engine"] == "kika"
    assert read_endf(str(base), mf_numbers=[1]).mf[1].mt[451]._lrp == 2
    reconstructed = _values(base, 2, np.array([1.0e3, 3.0e4, 4.0e5]))
    assert np.all(reconstructed > 1.0), reconstructed


@pytest.mark.slow
def test_the_resonance_cross_section_moves_by_its_factor(kikaRun):
    run, _meta, base = kikaRun
    sample = run.samples[0]
    key = ComponentKey(ZA, 33, 2)
    factors = np.asarray(sample["set"].factors[key], float)
    edges = np.asarray(sample["set"].binEdges[key], float)
    mids = np.sqrt(edges[:-1] * edges[1:])
    inside = (mids > 1.0) & (mids < 8.0e5)
    ratio = _values(run.paths("endf-delta")[0], 2, mids[inside]) / _values(base, 2, mids[inside])
    assert np.allclose(ratio, factors[inside], rtol=2e-6)


@pytest.mark.slow
def test_a_missing_njoy_gives_way_to_kika_with_a_warning(fe56_b81_tape, tmp_path, monkeypatch):
    from kika.processing import njoy_pendf_cache

    def nowhere(*_args, **_kwargs):
        raise njoy_pendf_cache.NjoyNotFoundError("no NJOY here")

    monkeypatch.setattr(njoy_pendf_cache, "find_njoy_executable", nowhere)
    run, meta, _base = _run(fe56_b81_tape, tmp_path, reconstructionEngine="njoy")
    assert meta["resonanceRegion"]["engine"] == "kika"
    fallbacks = [e for e in run.log.of("warning") if e.payload.get("code") == "njoy_fallback"]
    assert len(fallbacks) == 1 and "kika reconstructed it instead" in fallbacks[0].message


@pytest.mark.slow
def test_both_engines_draw_the_same_and_agree_within_the_tolerance(
        fe56_b81_tape, kikaRun, tmp_path):
    from kika.processing.njoy_pendf_cache import NjoyNotFoundError, find_njoy_executable

    try:
        find_njoy_executable()
    except NjoyNotFoundError:
        pytest.skip("needs NJOY ($NJOY_EXECUTABLE or njoy on PATH)")
    kika, _meta, kikaBase = kikaRun
    njoy, meta, njoyBase = _run(fe56_b81_tape, tmp_path, reconstructionEngine="njoy")
    assert meta["resonanceRegion"]["engine"] == "njoy"
    key = ComponentKey(ZA, 33, 2)
    np.testing.assert_array_equal(np.asarray(kika.samples[0]["set"].factors[key]),
                                  np.asarray(njoy.samples[0]["set"].factors[key]))
    energies = np.geomspace(1.0e-3, 8.0e5, 400)
    a, b = _values(kikaBase, 2, energies), _values(njoyBase, 2, energies)
    # Each table is within the tolerance of the same σ(E), so of each other within twice it.
    assert np.all(np.abs(a - b) <= 2 * (TOLERANCE * np.abs(b) + 1e-8))

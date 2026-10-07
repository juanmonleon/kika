"""The resonance region is perturbed as the cross section, not as MF3's background.

On a tape with LRP=1, MF3 below the resonance boundary is the background the
reconstructed resonances are added to. ENDF/B-VIII.1 Fe-56's MT2 there is
5e-4 b at 1 keV in MF3 and 9.3 b reconstructed (measured 2026-10-07), so a
factor applied to MF3 left the realisation essentially unperturbed below
850 keV -- which is what ``perturbFromModel`` did on an ENDF tape until that
day. See :mod:`kika.sampling.resonance_region` for the fix: RECONR once, the
model decoded from the tape with RECONR's MF3 and LRP=2, ACE from the
perturbed PENDF.

These need NJOY and the full tape; the section-sliced micro-tapes cannot be
reconstructed (RECONR stops at their orphan FEND records).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.resonance_region import needsReconstruction

ZA = 26056


def _njoy():
    from kika.processing.njoy_pendf_cache import NjoyNotFoundError, find_njoy_executable

    try:
        return find_njoy_executable()
    except NjoyNotFoundError:
        pytest.skip("needs NJOY ($NJOY_EXECUTABLE or njoy on PATH)")


def _values(path, mt, energies):
    section = read_endf(str(path), mf_numbers=[3]).get_file(3).sections[mt]
    return np.interp(energies, np.asarray(section.energies, float),
                     np.asarray(section.cross_sections, float))


def _sections(path):
    """``(MF, MT) -> text`` of every section, sequence numbers dropped."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            mf, mt = int(line[70:72]), int(line[72:75])
        except ValueError:
            continue
        if mt:
            out.setdefault((mf, mt), []).append(line[:75])
    return out


@pytest.fixture(scope="module")
def elasticRun(fe56_b81_tape, tmp_path_factory):
    _njoy()
    out = tmp_path_factory.mktemp("recon")
    run = perturbFromModel(str(fe56_b81_tape), {33: [2]}, 1, seed=3,
                           outputDir=out, formats=("endf-delta",))
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    return run, meta, Path(meta["resonanceRegion"]["base"])


@pytest.mark.slow
def test_the_tape_states_a_background_and_the_base_the_cross_section(
        fe56_b81_tape, elasticRun):
    run, meta, base = elasticRun
    assert needsReconstruction(read_endf(str(fe56_b81_tape)))
    assert meta["resonanceRegion"]["reconstructed"] is True
    assert any("reconstructed by NJOY RECONR" in note for note in run.notes)
    header = read_endf(str(base), mf_numbers=[1]).mf[1].mt[451]
    assert header._lrp == 2, "the base states its MF3 as complete"
    energies = np.array([1.0e3, 3.0e4, 4.0e5])
    evaluated = _values(fe56_b81_tape, 2, energies)
    reconstructed = _values(base, 2, energies)
    assert np.all(evaluated < 1e-2), evaluated
    assert np.all(reconstructed > 1.0), reconstructed


@pytest.mark.slow
def test_the_resonance_cross_section_moves_by_its_factor(elasticRun):
    """Inside the resolved range, MT2 written / MT2 reconstructed = the factor."""
    run, _meta, base = elasticRun
    sample = run.samples[0]
    key = ComponentKey(ZA, 33, 2)
    factors = np.asarray(sample["set"].factors[key], float)
    edges = np.asarray(sample["set"].binEdges[key], float)
    delta = run.paths("endf-delta")[0]
    mids = np.sqrt(edges[:-1] * edges[1:])
    inside = (mids > 1.0) & (mids < 8.0e5)
    assert inside.sum() >= 3, "no covariance bin inside the resolved range"
    ratio = _values(delta, 2, mids[inside]) / _values(base, 2, mids[inside])
    assert np.allclose(ratio, factors[inside], rtol=2e-6), (
        f"max {np.max(np.abs(ratio / factors[inside] - 1)):.2e}")
    assert np.any(np.abs(factors[inside] - 1.0) > 1e-3), "a draw of exactly 1"


@pytest.mark.slow
def test_the_delta_differs_from_its_base_only_where_it_should(elasticRun):
    """MT2, MT1 re-derived above it, and MF1/451's directory: nothing else."""
    run, _meta, base = elasticRun
    before, after = _sections(base), _sections(run.paths("endf-delta")[0])
    moved = {key for key in set(before) | set(after)
             if before.get(key) != after.get(key)}
    assert moved <= {(3, 2), (3, 1), (1, 451)}, sorted(moved)
    assert {(3, 2), (3, 1)} <= moved


@pytest.mark.slow
def test_njoy_gets_the_perturbed_pendf_and_the_original_endf(
        fe56_b81_tape, elasticRun, tmp_path):
    """The two inputs of ``run_njoy_with_pendf`` for a reconstructed sample."""
    from kika.sampling.model_perturbation import _njoyInputs

    run, meta, _base = elasticRun
    delta = run.paths("endf-delta")[0]
    endf, pendf = _njoyInputs(delta, Path(fe56_b81_tape),
                              Path(meta["resonanceRegion"]["pendf"]), tmp_path)
    original, sample = _sections(fe56_b81_tape), _sections(delta)
    forNjoy, perturbed = _sections(endf), _sections(pendf)
    mf3 = lambda sections: {k: v for k, v in sections.items() if k[0] == 3}
    assert mf3(perturbed) == mf3(sample), "the PENDF carries the sample's MF3"
    assert mf3(forNjoy) == mf3(original), "PURR reads the evaluation's MF3"
    assert read_endf(str(endf), mf_numbers=[1]).mf[1].mt[451]._lrp == 1
    for key, text in sample.items():
        if key[0] not in (1, 3):
            assert forNjoy.get(key) == text, f"MF{key[0]}/MT{key[1]} lost"


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("KIKA_RUN_NJOY_ACE"),
                    reason="runs NJOY to ACE twice (~20 min); set KIKA_RUN_NJOY_ACE=1")
def test_the_ace_carries_the_perturbation_in_the_resolved_range(fe56_b81_tape, tmp_path):
    """The perturbed ACE over the unperturbed one, through the same NJOY chain.

    Wherever the factor is the same 0.5 % either side (the Doppler width at
    293.6 K is about 0.1 % at 1 keV, so neighbouring factors cannot mix), MT2's
    ratio is the factor the delta tape carries. Measured 2026-10-07 on 6918
    such points below 850 keV: median deviation 1e-7, 99th percentile 4e-4,
    the worst 5.8e-3 beside a resonance at 24 keV. And the unperturbed ACE
    made this way equals the library's own (standard template, RECONR
    inside) to 4.9e-7 on all 41999 of its MT2 points.
    """
    from kika.ace.parsers.parse_ace import read_ace
    from kika.njoy.run_njoy import run_njoy_with_pendf
    from kika.sampling.model_perturbation import AceOptions

    exe = str(_njoy())
    run = perturbFromModel(str(fe56_b81_tape), {33: [2]}, 1, seed=3,
                           outputDir=tmp_path / "run", formats=("endf-delta", "ace"),
                           ace=AceOptions(temperatures=(293.6,), njoyExe=exe,
                                          libraryName="endfb81"))
    assert not run.aceFailures(), run.aceFailures()
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text("utf-8"))
    reference = run_njoy_with_pendf(exe, fe56_b81_tape, meta["resonanceRegion"]["pendf"],
                                    293.6, "endfb81", tmp_path / "ref")
    perturbed = read_ace(str(run.samples[0]["files"]["ace"][0])).get_cross_section(2)
    unperturbed = read_ace(str(reference["ace_file"])).get_cross_section(2)
    ep = perturbed.iloc[:, 0].to_numpy(float) * 1e6              # MeV -> eV
    er = unperturbed.iloc[:, 0].to_numpy(float) * 1e6
    xp = perturbed.iloc[:, 1].to_numpy(float)
    xr = unperturbed.iloc[:, 1].to_numpy(float)

    base = meta["resonanceRegion"]["base"]
    delta = run.paths("endf-delta")[0]

    def factor(e):
        return _values(delta, 2, e) / _values(base, 2, e)

    energy = er[(er > 1.0) & (er < 8.5e5)]
    f0 = factor(energy)
    flat = ((np.abs(factor(energy * 0.995) - f0) < 1e-7)
            & (np.abs(factor(energy * 1.005) - f0) < 1e-7))
    deviation = np.abs(np.interp(energy[flat], ep, xp)
                       / np.interp(energy[flat], er, xr) / f0[flat] - 1)
    assert flat.sum() > 1000
    assert np.median(deviation) < 1e-5
    assert np.percentile(deviation, 99) < 1e-3

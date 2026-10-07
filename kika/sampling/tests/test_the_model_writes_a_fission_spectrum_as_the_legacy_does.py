"""Q1 of the PFNS migration: the model and the shipped pipeline are one operation.

Q0 (``test_the_model_draws_a_fission_spectrum_as_the_legacy_does.py``) made the
model *draw* MF35 the way ``generate_pfns_samples`` does, on the micro-tape.
This file carries that to the end of the chain and to full evaluations:

* **Q1.1, the draw** -- bit-identical on whole tapes, and the one place the two
  are *meant* to differ: a request that also perturbs a relative quantity moves
  the spectrum up the seed ladder;
* **Q1.2, the tape** -- for one seed, the MF5/MT18 records ``perturb_pfns_files``
  writes and the ones ``perturbFromModel(formats=("endf-delta",))`` writes are
  the same bytes, and nothing outside MF5 moves on either side.

Two differences are pinned rather than hidden, so nobody "fixes" one by
accident:

* **PF-1** (legacy): asked for a subset of bands, ``perturb_pfns_partial`` puts
  no shoulder below the lowest one, so the factor ramps across the incident
  interval under it. The model steps there. Only the whole-band request -- the
  only one the legacy driver ever makes -- is equivalent;
* **PF-5** (model emitter): ``endf-delta`` re-encodes the whole MF it touched,
  so the *other* MF5 sections (MT455 on Cf-252) come back with the same values
  and new sequence numbers and SEND records. The legacy writer replaces MT18
  alone. The values are gated here; the bytes are not.

The plan is ``kika-workspace/docs/pfns/pfns_mf5_mf35_roadmap.md``, "Phase 2", Q1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.core import BLOCK_SEED_STRIDE
from kika.sampling.mf35_sampling import build_pfns_covariance, generate_pfns_samples
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.pfns_perturbation import perturb_pfns_files

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
CF252_MICRO = DATA / "micro_cf252_pfns.endf"


def _legacyDraw(tape, n, seed):
    endf = read_endf(str(tape), mf_numbers=[5, 35])
    suite, _mf5, _bands = build_pfns_covariance(endf, mt=18)
    drawn, _info = generate_pfns_samples(suite, n, seed=seed, verbose=False)
    return {key[-1]: value for key, value in drawn.items()}


def _modelSpectrumDeltas(run):
    """``{band: (nSamples, nGroups)}``, MF35 components only."""
    rows = [{component.index: np.asarray(values, dtype=float)
             for component, values in sample["set"].factors.items()
             if component.mf == 35}
            for sample in run.samples]
    return {band: np.vstack([row[band] for row in rows]) for band in rows[0]}


def _records(path):
    """``{(mf, mt): [line, ...]}`` of the data records, control records dropped.

    SEND/FEND/MEND/TEND carry no data and are where the two writers differ in
    padding (blank against zero-filled), so they are left out of every
    comparison here; a directory or count error would still show in MF1/451.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        mf, mt = int(line[70:72]), int(line[72:75])
        if mt:
            out.setdefault((mf, mt), []).append(line)
    return out


def _number(field):
    """One ENDF field as a value: ``9.223300+4`` and ``92233.0000`` are one number."""
    text = field.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    for sign in "+-":
        cut = text.rfind(sign)
        if cut > 0:
            try:
                return float(f"{text[:cut]}e{text[cut:]}")
            except ValueError:
                break
    return text


def _values(lines):
    """The six fields of every record, read as numbers. U-233 B-VIII.1 spells
    MT455's ZA ``92233.0000`` and the encoder writes ``9.223300+4``."""
    return [tuple(_number(line[i:i + 11]) for i in range(0, 66, 11))
            for line in lines]


# ======================================================================
# Q1.1 -- the draw, on whole evaluations
# ======================================================================

def _assertSameDraw(tape):
    legacy = _legacyDraw(tape, 8, seed=11)
    run = perturbFromModel(str(tape), {35: None}, nSamples=8, seed=11,
                           dryRun=True)
    model = _modelSpectrumDeltas(run)
    assert set(model) == set(legacy)
    for band in legacy:
        assert np.array_equal(model[band], legacy[band]), (
            f"band {band}: max |Δ| = "
            f"{np.max(np.abs(model[band] - legacy[band])):.3e}")


def test_the_draw_is_the_legacy_draw_on_jeff40_u235(u235_tape):
    """Eight bands, ranks 97 down to 16: the case the thesis ensemble drew."""
    _assertSameDraw(u235_tape)


def test_the_draw_is_the_legacy_draw_on_endfb81_cf252(cf252_b81_tape):
    _assertSameDraw(cf252_b81_tape)


def test_a_mixed_request_draws_the_spectrum_further_up_the_seed_ladder(
        cf252_b81_tape):
    """Not equal to the MF35-only draw, and on purpose.

    ``_drawEverything`` continues one ladder across the relative and the
    absolute blocks, so a relative block drawn first pushes every band up by
    ``BLOCK_SEED_STRIDE``. Restarting the absolute draw at *seed* would hand
    band 0 the stream MF31 already used -- a correlation between ν̄ and χ that
    nothing would report. So the spectrum of a mixed run is the legacy draw at
    the shifted seed, exactly.
    """
    run = perturbFromModel(str(cf252_b81_tape), {31: None, 35: None},
                           nSamples=4, seed=11, dryRun=True)
    relative = [key for key in run.diagnostics if key[1] != "MF35"]
    assert len(relative) == 1, relative

    shifted = _legacyDraw(cf252_b81_tape, 4, seed=11 + BLOCK_SEED_STRIDE)
    model = _modelSpectrumDeltas(run)
    for band in shifted:
        assert np.array_equal(model[band], shifted[band]), f"band {band}"

    unshifted = _legacyDraw(cf252_b81_tape, 4, seed=11)
    assert not np.array_equal(model[0], unshifted[0])


# ======================================================================
# Q1.2 -- the tape
# ======================================================================

def _assertSameTape(tape, tmp_path, nSamples=1):
    legacy = perturb_pfns_files(str(tape), nSamples, generate_ace=False,
                                seed=11, output_dir=str(tmp_path / "legacy"))
    assert legacy["errors"] == []
    run = perturbFromModel(str(tape), {35: None}, nSamples=nSamples, seed=11,
                           outputDir=tmp_path / "model", formats=("endf-delta",))

    source = _records(tape)
    legacyTapes = sorted((tmp_path / "legacy").glob("*_pfns_*.endf"))
    for number, (legacyTape, modelTape) in enumerate(
            zip(legacyTapes, run.paths("endf-delta"))):
        old, new = _records(legacyTape), _records(modelTape)
        assert set(old) == set(new) == set(source)

        assert old[(5, 18)] == new[(5, 18)], (
            f"sample {number}: MF5/MT18 differs between the two writers")
        assert old[(5, 18)] != source[(5, 18)], "the sample moved nothing"
        assert old[(1, 451)] == new[(1, 451)], "the directories disagree"

        for key in source:
            if key[0] == 5 and key != (5, 18):
                assert old[key] == source[key]
                assert _values(new[key]) == _values(source[key]), (
                    f"MF{key[0]}/MT{key[1]}: the model re-encoded a section it "
                    f"did not perturb and changed its values (PF-5)")
            elif key[0] != 5 and key != (1, 451):
                assert old[key] == source[key] == new[key], (
                    f"MF{key[0]}/MT{key[1]} moved")


def test_the_two_writers_write_the_same_spectrum_on_the_micro_tape(tmp_path):
    _assertSameTape(CF252_MICRO, tmp_path, nSamples=2)


def test_the_two_writers_write_the_same_spectrum_on_jeff40_u235(
        u235_tape, tmp_path):
    _assertSameTape(u235_tape, tmp_path)


def test_the_two_writers_write_the_same_spectrum_on_endfb81_cf252(
        cf252_b81_tape, tmp_path):
    _assertSameTape(cf252_b81_tape, tmp_path)


@pytest.mark.slow
def test_the_two_writers_write_the_same_spectrum_on_endfb81_u235(
        u235_b81_tape, tmp_path):
    """The reference evaluation, which the model could not perturb at all (PF-6).

    Its MT18 is stated in MF5 *and* in MF6 (``JP=11``, a ``LAW=0`` neutron), and
    the decoder let MF6's ``unspecified`` replace MF5's spectrum. MF6 must come
    back byte for byte as well, which the "every other MF" clause checks.
    """
    _assertSameTape(u235_b81_tape, tmp_path)


@pytest.mark.slow
def test_a_rounded_correlation_is_drawn_as_the_legacy_draws_it(
        pu240_b71_tape, tmp_path):
    """PF-7: B-VII.1 Pu-240 states correlations above 1 by up to 1.2e-5.

    The pre-flight refused the band; the legacy clipped it and drew. Rounding
    in the file is a note now, so the model clips and draws the same thing.

    The tapes are not byte-equal, and that is PF-1 rather than PF-7: the one
    band is [0.5, 0.5] MeV, so it does not start at the lowest incident node.
    The legacy puts no shoulder under it and ramps the factor from 1e-5 eV up
    to 0.5 MeV; the model steps at 0.4995 MeV. From the band up, every
    incident table is the same.
    """
    _assertSameDraw(pu240_b71_tape)

    legacy = perturb_pfns_files(str(pu240_b71_tape), 1, generate_ace=False,
                                seed=11, output_dir=str(tmp_path / "legacy"))
    assert legacy["errors"] == []
    run = perturbFromModel(str(pu240_b71_tape), {35: None}, nSamples=1,
                           seed=11, outputDir=tmp_path / "model",
                           formats=("endf-delta",))

    def partial(path):
        return read_endf(str(path), mf_numbers=[5]).mf[5].mt[18].partials[0]

    source = partial(pu240_b71_tape)
    old = partial(next((tmp_path / "legacy").glob("*_pfns_*.endf")))
    new = partial(run.paths("endf-delta")[0])
    nodes = np.asarray(source.incident_energies)
    assert np.array_equal(old.incident_energies, nodes)
    assert np.array_equal(new.incident_energies,
                          np.insert(nodes, 1, 4.995e5)), "the PF-1 shoulder"
    for energy in nodes:
        a, b = old.evaluate_at_incident(energy), new.evaluate_at_incident(energy)
        assert all(np.array_equal(x, y) for x, y in zip(a, b)), energy
    assert any(not np.array_equal(x, y)
               for x, y in zip(old.evaluate_at_incident(2.5e5),
                               new.evaluate_at_incident(2.5e5))), \
        "below the band the legacy ramps and the model does not"


@pytest.mark.slow
def test_a_band_that_states_no_variance_draws_no_perturbation(
        pu242_j40_tape, tmp_path):
    """PF-8: JEFF-4.0 Pu-242 states its [20, 200] MeV band as a 1x1 zero.

    The pre-flight refused the whole run over it; the legacy drew δ ≡ 0 there.
    """
    _assertSameDraw(pu242_j40_tape)
    legacy = _legacyDraw(pu242_j40_tape, 8, seed=11)
    assert not np.any(legacy[4])
    _assertSameTape(pu242_j40_tape, tmp_path)


def test_a_band_subset_is_where_the_two_appliers_part_on_purpose():
    """PF-1: the legacy applier leaves the step below the lowest band unwritten.

    It skips ``bands[1:]`` when inserting incident shoulders, which is right only
    because the legacy driver always passes every band. Handed the top band
    alone it inserts none, and the factor ramps from the last node below the
    band to the first node in it. The model inserts the shoulder the step
    needs. Equivalence is claimed for the whole-band request only.
    """
    from kika.endf.model_adapter.energy import decodeMF5MT
    from kika.nuclear_data.model.perturbation import applySpectrumFactors
    from kika.sampling.mf35_sampling import (band_grids, perturb_pfns_partial,
                                             pfns_ratio_rule)

    endf = read_endf(str(CF252_MICRO), mf_numbers=[5, 35])
    suite, mf5, bands = build_pfns_covariance(endf, mt=18)
    grids = band_grids(suite)
    top = len(bands) - 1
    delta = _legacyDraw(CF252_MICRO, 1, seed=11)[top][0]

    _partial, legacy = perturb_pfns_partial(
        read_endf(str(CF252_MICRO), mf_numbers=[5]).mf[5].mt[18].partials[0],
        {0: delta}, [bands[top]], [grids[top]])
    form, _provenance, _report = decodeMF5MT(mf5)
    _perturbed, model = applySpectrumFactors(
        form, {top: bands[top]}, {top: grids[top]}, pfns_ratio_rule({top: delta}))

    assert legacy["n_incident_inserted"] == 0
    assert model["n_outer_inserted"] == 1


# ======================================================================
# Q1.3 -- NJOY takes the model's tape
# ======================================================================

@pytest.mark.slow
def test_njoy_makes_ace_from_the_models_perturbed_spectrum(
        cf252_b81_tape, njoy_exe, tmp_path):
    """The legacy NJOY gate, on the model's output.

    Same tape, same question: does ACER accept an MF5/MT18 that grew by the
    shoulders the applier inserts. The unperturbed tape goes first, so a
    failure that is the environment's is reported as a skip and not blamed on
    the perturbation -- the reason is in
    ``test_pfns_perturbation.test_njoy_regenerates_ace_from_a_perturbed_pfns_tape``.
    """
    from kika.njoy.run_njoy import run_njoy
    from kika.sampling.model_perturbation import AceOptions

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    result = run_njoy(njoy_exe=str(njoy_exe), endf_path=cf252_b81_tape,
                      temperature=293.6, library_name="endfb81",
                      output_dir=baseline, njoy_version="NJOY 2016.78",
                      ace_dir=baseline, xsdir_dir=baseline)
    if int(result.get("returncode", -1)) != 0 or not result.get("ace_file"):
        pytest.skip(f"NJOY cannot process the unperturbed Cf-252 tape here "
                    f"(return code {result.get('returncode')})")

    run = perturbFromModel(
        str(cf252_b81_tape), {35: None}, nSamples=1, seed=1,
        outputDir=tmp_path / "model", formats=("endf-delta", "ace"),
        ace=AceOptions(temperatures=(293.6,), njoyExe=str(njoy_exe),
                       libraryName="endfb81"))
    produced = run.samples[0]["files"].get("ace")
    assert produced, (
        "NJOY processed the original tape but not the model's, so the "
        "perturbed MF5 is the difference; the listing is under "
        f"{tmp_path / 'model' / '0000' / 'njoy'}")
    assert all(Path(path).stat().st_size > 0 for path in produced)

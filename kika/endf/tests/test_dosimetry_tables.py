"""Dosimetry ACE tables (``.NNy``) from kika's own processing and from a pointwise tape.

* ``EndfProcessing.write_dosimetry_ace``: every processed cross section comes
  out as processed (same nodes, same values), a level sum the tape leaves out
  is added, gas production is written, and an unprocessed temperature is
  refused.
* ``dosimetry_ace``: a tape without resonance parameters is written as it is
  stated (MF 10 with its own laws), an MF 9 production is the multiplicity
  times the cross section to the processing tolerance, and a tape with
  resonance parameters is refused.
* With IRDFF-II beside the repo (``../NuclearData/dosimetry/irdff2``): Mg, Al-27
  and Ti come out as the IAEA's ACE library writes them, MT 5 products numbered
  by their product and the tape's blank-exponent numbers read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.dosimetry import DosimetryAce, split_production_mt
from kika.endf import ProcessingOutputError, ProcessingPlan, dosimetry_ace, process_endf

DATA = Path(__file__).resolve().parent / "data"
STRUCTURAL = DATA / "micro_fe56_structural.endf"
ACTIVATION = DATA / "micro_co59_activation.endf"


@pytest.fixture(scope="module")
def run():
    return process_endf(STRUCTURAL, ProcessingPlan(temperatures=(293.6,), workers=2))


@pytest.fixture(scope="module")
def table(run, tmp_path_factory):
    out = tmp_path_factory.mktemp("dosimetry")
    paths = run.write_dosimetry_ace(out)
    return out, paths


def test_one_table_per_temperature_with_its_suffix(table):
    out, paths = table
    assert [p.name for p in paths] == ["26056.02y"]
    assert "26056.02y" in (out / "xsdir").read_text()
    t = read_ace(str(paths[0]))
    assert isinstance(t, DosimetryAce) and t.repeated_numbers == []
    assert abs(t.header.temperature - 2.53e-8) < 1e-11


def test_processed_cross_sections_come_out_as_processed(run, table):
    t = read_ace(str(table[1][0]))
    for mt, (e, sigma) in run.cross_sections(293.6).items():
        if not np.any(sigma != 0.0):
            continue
        r = t.reaction(mt)
        assert not r.interpolation
        np.testing.assert_allclose(r.energies, e, rtol=1e-11)
        np.testing.assert_allclose(r.cross_section, sigma, rtol=1e-11)


def test_a_level_sum_the_tape_leaves_out_is_added(run, table):
    from kika.hdf5.processing import LEVEL_SUMS

    xs = run.cross_sections(293.6)
    t = read_ace(str(table[1][0]))
    added = [mt for mt, levels in LEVEL_SUMS.items() if mt not in xs and any(c in xs for c in levels)]
    for mt in added:
        r = t.reaction(mt)
        grid = r.energies
        want = sum(np.interp(grid, *xs[c], left=0.0, right=0.0) for c in LEVEL_SUMS[mt] if c in xs)
        np.testing.assert_allclose(r.cross_section, want, rtol=1e-11, atol=1e-30)


def test_an_unprocessed_temperature_is_refused(run, tmp_path):
    with pytest.raises(ProcessingOutputError) as error:
        run.write_dosimetry_ace(tmp_path, temperatures=600.0)
    assert error.value.category == "ace-temperature-not-processed"


def test_a_pointwise_tape_is_written_as_stated(tmp_path):
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.nuclear_data.model.cross_section_forms import Reference

    path = dosimetry_ace(ACTIVATION, tmp_path, extension=".00y")
    t = read_ace(str(path))
    suite, _ = decodeReactionSuite(read_endf(str(ACTIVATION)))
    mf10 = mf9 = 0
    for r in t.reactions:
        split = split_production_mt(r.mt)
        if split is None:
            reaction = suite.reactionByENDF_MT(r.mt) if hasattr(suite, "reactionByENDF_MT") else None
            continue
        mt, lfs = split
        production = next(p for p in suite.productions if p.ENDF_MT == mt
                          and _lfs(suite, p) == lfs)
        form = next(iter(production.crossSection.forms.values()))
        if isinstance(form, Reference):
            mf9 += 1
            mid = 0.5 * (r.energies[:-1] + r.energies[1:])
            mult = production.outputChannel.products[0].multiplicity.form
            host = suite.reactionByENDF_MT(mt).crossSection
            sigma = next(iter(host.forms.values()))
            want = np.asarray(mult.evaluate(mid, outOfRange="zero")) * np.asarray(sigma.evaluate(mid, outOfRange="zero"))
            got = 0.5 * (r.cross_section[:-1] + r.cross_section[1:])
            wide = r.energies[1:] > r.energies[:-1]                  # a step has no inside
            assert np.all((np.abs(got - want) <= 1e-3 * np.abs(want) + 1e-30)[wide])
        else:
            mf10 += 1
            x, y, regions = form.toEndfRegions()
            np.testing.assert_allclose(r.energies, x, rtol=1e-11)
            np.testing.assert_allclose(r.cross_section, y, rtol=1e-11)
            assert (r.breakpoints, r.interpolation) in (([], []), ([n for n, _ in regions], [i for _, i in regions]))
    assert mf10 > 0 and mf9 > 0


def _lfs(suite, production):
    from kika.endf.model_adapter.productions import _levelOf

    product = next(p for p in production.outputChannel.products if p.pid != "photon")
    return _levelOf(suite, product.pid)[0]


def test_a_pointwise_tape_takes_its_temperature_suffix(tmp_path):
    # 0 K is the suffix of 0.1 K, as for every table kika names (kika._constants.K_TO_SUFFIX)
    assert dosimetry_ace(ACTIVATION, tmp_path).name == "27059.01y"


def test_a_tape_with_resonance_parameters_is_refused(tmp_path):
    with pytest.raises(ProcessingOutputError, match="resonance parameters"):
        dosimetry_ace(STRUCTURAL, tmp_path, extension=".02y")


IRDFF2 = Path(__file__).resolve().parents[5] / "NuclearData" / "dosimetry" / "irdff2"


def _irdffMaterial(mat, out):
    lines = (IRDFF2 / "endf" / "IRDFF-II.endf").read_text().splitlines(keepends=True)
    body = [l for l in lines[1:] if l[66:70].strip() == str(mat)]
    path = out / f"irdff2-{mat}.endf"
    end = " " * 66 + "   0 0  0    0\n" + " " * 66 + "  -1 0  0    0\n"
    path.write_text(lines[0] + "".join(body) + end)
    return path


@pytest.mark.parametrize("mat", [1200, 1325, 2200])
def test_irdff2_agrees_with_the_iaea_ace_library(mat, tmp_path):
    if not (IRDFF2 / "endf" / "IRDFF-II.endf").exists():
        pytest.skip("no IRDFF-II in ../NuclearData/dosimetry/irdff2")
    mine = read_ace(str(dosimetry_ace(_irdffMaterial(mat, tmp_path), tmp_path, xsdir=False)))
    ref = read_ace(str(IRDFF2 / "ace" / "IRDFF-II" / f"dos-irdff2-{mat}.acef"))
    assert set(ref.mt_numbers) <= set(mine.mt_numbers) and set(mine.mt_numbers) - set(ref.mt_numbers) <= {1}
    for r in ref.reactions:
        m = mine.reaction(r.mt)
        assert (m.breakpoints, m.interpolation) == (r.breakpoints, r.interpolation)
        np.testing.assert_allclose(m.energies, r.energies, rtol=1e-6)
        np.testing.assert_allclose(m.cross_section, r.cross_section, rtol=6e-7, atol=0)

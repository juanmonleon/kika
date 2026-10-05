"""A tabulated MF4 perturbed end to end: request, draw, apply, write, read back.

T5 and T7.3 of ``docs/library/mf4_tabulated_perturbation_roadmap.md``.

The tables are real: ``micro_u238_ltt2_mf34.endf`` is U-238 of JEFF-4.0, whose
MF4/MT2 is tabulated (LTT=2) at 39 energies with 90-91 cosines. Its own MF34 is
**not** used here: as stated it cannot be sampled -- 172 correlations outside
[-1, 1] (worst 4.8), not PSD, relative variances up to 1.7e4 -- and the
pre-flight refuses it (``docs/library/jeff40_mf34_covariance_report.md``).
Al-27, the only other evaluation to hand with LTT=2 and MF34, fails the same
way, and no other library to hand has an LTT=2 section with MF34 at all.

So the covariance is taken from another library, and of two kinds:

* **real** -- MF34/MT2 of U-238 from ENDF/B-VIII.1 (``micro_u238_mf34_l0.endf``),
  the same nuclide and reaction, which passes the pre-flight. L=1 and 2 on 44
  bins from 1e-5 eV to 30 MeV, plus an L=0 placeholder that the request drops.
  The statistics are checked against it.
* **synthetic** -- :mod:`l0_fixture`, orders 1 and 2 on 0.1-20 MeV, for the one
  thing the real one cannot show: a range with no covariance at all, where the
  written table has to be the evaluation bit for bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import l0_fixture
from kika.endf import read_endf
from kika.nuclear_data.model.angular_tables import legendreMoments
from kika.sampling.model_perturbation import perturbFromModel

TAPE = (Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
        / "micro_u238_ltt2_mf34.endf")
GRID = [1.0e5, 1.0e6, 5.0e6, 2.0e7]
N = 8

pytestmark = pytest.mark.skipif(not TAPE.is_file(), reason=f"{TAPE} missing")


@pytest.fixture(scope="module")
def covariance(tmp_path_factory):
    return l0_fixture.write(tmp_path_factory.mktemp("cov") / "u238_cov.endf",
                            orders=(1, 2), za=92238, awr=236.0058, mat=9237,
                            grid=GRID)


@pytest.fixture(scope="module")
def run(covariance, tmp_path_factory):
    return perturbFromModel(TAPE, {34: [2]}, N, seed=5,
                            outputDir=tmp_path_factory.mktemp("run"),
                            formats=("endf-delta", "endf-tape", "gnds"),
                            covarianceSource=covariance)


REAL_COVARIANCE = TAPE.with_name("micro_u238_mf34_l0.endf")


@pytest.fixture(scope="module")
def realRun(tmp_path_factory):
    return perturbFromModel(TAPE, {34: [2]}, N, seed=5,
                            outputDir=tmp_path_factory.mktemp("real"),
                            formats=("endf-delta",),
                            covarianceSource=REAL_COVARIANCE)


@pytest.fixture(scope="module")
def base():
    return read_endf(str(TAPE)).get_file(4).sections[2]


def _section(path):
    return read_endf(str(path)).get_file(4).sections[2]


def test_every_format_is_written_and_mf4_stays_tabulated(run):
    sample = run.samples[0]
    assert set(sample["files"]) >= {"endf-delta", "endf-tape", "gnds"}
    for fmt in ("endf-delta", "endf-tape"):
        section = _section(sample["files"][fmt])
        assert type(section).__name__ == "MF4MTTabulated" and section._ltt == 2
        # 39 energies, plus one twin per bin edge: all four edges of the grid
        # are energies the evaluation tabulates.
        assert all(edge in section.energies for edge in GRID)
        assert len(section.energies) == 39 + len(GRID)
    assert any("tabulated and perturbed as a table" in note for note in run.notes)


def test_outside_the_covariance_the_written_table_is_the_evaluation(run, base):
    """Below 0.1 MeV no bin applies: the read-back tables are the base, bit for bit."""
    out = _section(run.samples[0]["files"]["endf-delta"])
    for energy, mu, p in zip(base.energies, base.cosines, base.probabilities):
        if energy >= GRID[0]:
            break
        at = out.energies.index(energy)
        assert out.cosines[at] == mu and out.probabilities[at] == p, energy


def _binOf(energy):
    return int(np.searchsorted(GRID, energy, side="right") - 1)


def test_the_drawn_factors_come_back_out_of_the_written_tables(run, base):
    """T7.3: sample - base, projected, is (c_l - 1) a_l for l = 1, 2 and ~0 above.

    At every base energy strictly inside a bin, over all N samples, read back
    from the written tape. What separates the two is the writer's seven
    significant figures and the lin-lin interpolation of the polynomial
    correction between 90 nodes: 5.5e-5 on the perturbed orders and the bound
    below on the others, measured 2026-10-05, against shifts of order 1e-2.
    """
    worstPerturbed, worstOther = 0.0, 0.0
    for sample in run.samples:
        factors = {c.index: np.asarray(v) for c, v in sample["set"].factors.items()}
        out = _section(sample["files"]["endf-delta"])
        for energy, mu, p in zip(base.energies, base.cosines, base.probabilities):
            k = _binOf(energy)
            if not (0 <= k < len(GRID) - 1) or energy in GRID or len(mu) < 3:
                continue
            at = out.energies.index(energy)
            assert out.cosines[at] == mu
            before = legendreMoments(mu, p, range(1, 7))
            after = legendreMoments(mu, out.probabilities[at], range(1, 7))
            for order in (1, 2):
                expected = (factors[order][k] - 1.0) * before[order]
                worstPerturbed = max(worstPerturbed,
                                     abs(after[order] - before[order] - expected))
            for order in range(3, 7):
                worstOther = max(worstOther, abs(after[order] - before[order]))
    assert worstPerturbed < 1e-4, worstPerturbed
    assert worstOther < 2e-3, worstOther


def test_the_realisation_records_what_the_tables_did(run):
    for sample in run.samples:
        for component, info in sample["applied"].items():
            tables = info["tables"]
            assert set(tables) >= {"min_p", "n_negative_nodes",
                                   "max_integral_change", "n_repaired"}
            assert tables["n_repaired"] == 0
            assert tables["max_integral_change"] < 1e-3


def test_repair_leaves_no_negative_node(covariance):
    """On the default ``report`` a negative node stays; ``repair`` removes it."""
    run = perturbFromModel(TAPE, {34: [2]}, 4, seed=5, dryRun=True,
                           covarianceSource=covariance,
                           angularPositivity="repair")
    for sample in run.samples:
        for info in sample["applied"].values():
            assert info["tables"]["min_p"] > -1e-10 or info["tables"]["n_repaired"]


def test_a_real_covariance_from_another_library_comes_back_out_of_the_tables(realRun, base):
    """T7.3 with ENDF/B-VIII.1's MF34 on JEFF-4.0's tables, N=8, read back.

    Measured 2026-10-05: 1.2e-5 on the perturbed orders, 1e-7 on the others.
    """
    assert {c.index for c in realRun.samples[0]["set"].components()} == {1, 2}
    assert any("states L=0" in note for note in realRun.notes)
    worstPerturbed, worstOther = 0.0, 0.0
    for sample in realRun.samples:
        blocks = {c.index: (np.asarray(v), np.asarray(sample["set"].binEdges[c]))
                  for c, v in sample["set"].factors.items()}
        out = _section(sample["files"]["endf-delta"])
        for energy, mu, p in zip(base.energies, base.cosines, base.probabilities):
            grid = blocks[1][1]
            if len(mu) < 3 or energy in grid or not grid[0] <= energy < grid[-1]:
                continue
            at = out.energies.index(energy)
            before = legendreMoments(mu, p, range(1, 7))
            after = legendreMoments(mu, out.probabilities[at], range(1, 7))
            for order in (1, 2):
                factors, edges = blocks[order]
                k = int(np.searchsorted(edges, energy, side="right") - 1)
                expected = (factors[k] - 1.0) * before[order]
                worstPerturbed = max(worstPerturbed,
                                     abs(after[order] - before[order] - expected))
            for order in range(3, 7):
                worstOther = max(worstOther, abs(after[order] - before[order]))
    assert worstPerturbed < 1e-4, worstPerturbed
    assert worstOther < 1e-5, worstOther

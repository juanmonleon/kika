"""MF3's sum rules on a realisation: partials govern, sums follow, nothing else moves.

The rule (decision 3 of ``kika-workspace/docs/library/perturbation_model_roadmap.md``,
completed 2026-10-07) in three parts, each pinned here:

1. a partial with its own block is perturbed by it -- partials govern;
2. one without rides the nearest perturbed sum above it -- MT1's block moves
   MT2 when MT2 has none, MT4's block beats MT1's for MT51-91;
3. every sum with a moved partial is re-derived as ``S + sum (p' - p)``, and a
   sum's own block is never applied to the sum.

And the property a reader of the tape relies on: **a sum moves by exactly what
its partials moved, and nowhere else.** Outside the factor blocks' energy range
the sum is the evaluation's, value for value -- the delta form is what keeps a
tape whose own sum residual is not zero (JEFF-4.0 U-235 MT1 sits 2.3 % off its
partials) from being "repaired" everywhere by a perturbation of one reaction.

Why it is not optional (measured 2026-10-07, ENDF/B-VIII.1 Fe-56 through
RECONR): NJOY discards the MT1 and MT4 a tape states and rebuilds them from the
partials. A block put on MT1 alone therefore never reached the ACE, and MT27 and
MT101, which RECONR does not rebuild, would reach it stale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._constants import MF3_SUM_RULES
from kika.endf import read_endf
from kika.endf.model_adapter.decode import _summationMTs
from kika.nuclear_data.model.functions import Regions1d, XYs1d
from kika.sampling.cross_section_sums import (planCrossSectionSums, rederiveSum,
                                              sumTree)
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.perturbation_set import PerturbationSet

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56_XS = DATA / "micro_fe56_xs_and_angular.endf"
FE56_STRUCTURAL = DATA / "micro_fe56_structural.endf"
ZA = 26056


def _key(mt):
    return ComponentKey(ZA, 33, mt)


# ----------------------------------------------------------------------
# The plan: which block moves which MT
# ----------------------------------------------------------------------

def test_the_sum_rules_are_a_tree():
    """No MT has two parents, so "the nearest perturbed sum above it" is definite."""
    everything = set(MF3_SUM_RULES)
    for parts, _ in MF3_SUM_RULES.values():
        everything |= {int(str(part).lstrip("@")) for part in parts}
    sumTree(everything, MF3_SUM_RULES)          # raises on a second parent


def test_partials_govern_and_the_rest_ride_the_nearest_sum():
    present = {1, 2, 4, 16, 51, 52, 102}
    sums = _summationMTs(present)
    assert sums == {1, 4}
    plan = planCrossSectionSums({1: _key(1), 4: _key(4), 51: _key(51)},
                                present, sums)

    assert plan.leafControl[51] == _key(51), "a partial's own block governs"
    assert plan.leafControl[52] == _key(4), "MT4 is nearer to MT52 than MT1 is"
    for leaf in (2, 16, 102):
        assert plan.leafControl[leaf] == _key(1)
    assert plan.rederive == (4, 1), "deepest first"
    assert plan.ownBlockReached == {1: (2, 16, 102), 4: (52,)}


def test_a_sum_whose_partials_all_carry_blocks_has_its_own_discarded():
    present = {4, 51, 52}
    plan = planCrossSectionSums({4: _key(4), 51: _key(51), 52: _key(52)},
                                present, _summationMTs(present))
    assert plan.ownBlockReached == {4: ()}
    assert plan.rederive == (4,)


def test_a_partial_alone_rebuilds_every_sum_above_it_and_nothing_else():
    present = {1, 2, 4, 51, 52, 102}
    plan = planCrossSectionSums({51: _key(51)}, present, _summationMTs(present))
    assert plan.leafControl == {51: _key(51)}
    assert plan.rederive == (4, 1)
    assert plan.movedUnder == {4: (51,), 1: (51,)}


def test_a_sum_stated_only_through_its_partials_still_moves_them():
    """ENDF/B-VIII.1 Fe-56: MF33 for MT103, MF3 for MT600-649 and no MT103.

    The block goes to the partials -- what ``apply_factors_to_pendf_mf3`` does
    for a composite the PENDF lacks -- and MT1 above them is rebuilt. Before
    this, the model pipeline could not run ``{33: None}`` on that tape at all.
    """
    present = {1, 2, 102, 600, 601, 649}
    plan = planCrossSectionSums({103: _key(103)}, present, _summationMTs(present))
    assert plan.virtual == (103,)
    assert {leaf: c.mt for leaf, c in plan.leafControl.items()} == {
        600: 103, 601: 103, 649: 103}
    assert plan.rederive == (1,)


def test_a_cross_section_the_tape_does_not_state_is_still_refused():
    with pytest.raises(KeyError, match="does not state"):
        planCrossSectionSums({16: _key(16)}, {1, 2, 102}, {1})


# ----------------------------------------------------------------------
# The arithmetic: S' = S + sum (p' - p), exactly
# ----------------------------------------------------------------------

def _xs(xs, ys, law="lin-lin"):
    from kika.nuclear_data.model.enums import Interpolation
    return XYs1d(xs=np.asarray(xs, float), ys=np.asarray(ys, float),
                 interpolation=Interpolation(law))


def _perturbed(form, factors, edges):
    pset = PerturbationSet(label="r", factors={_key(51): np.asarray(factors, float)},
                           binEdges={_key(51): np.asarray(edges, float)})
    return pset.apply(form, _key(51))[0]


def test_the_sum_moves_by_exactly_what_its_partial_moved():
    """Lin-lin everywhere, so the identity holds at every point, steps included."""
    grid = np.linspace(1.0, 10.0, 10)
    p1, p2 = _xs(grid, 2.0 + grid), _xs(grid, 5.0 - 0.3 * grid)
    total = _xs(grid, (2.0 + grid) + (5.0 - 0.3 * grid))
    edges = [2.5, 6.5, 8.0]
    p1after = _perturbed(p1, [1.2, 0.7], edges)

    rebuilt, info = rederiveSum(total, [(p1, p1after, edges[0], edges[-1])])

    probes = np.unique(np.concatenate([rebuilt.xs, np.linspace(1, 10, 181)]))
    probes = probes[(probes != 2.5) & (probes != 6.5) & (probes != 8.0)]
    expected = p1after.evaluate(probes) + p2.evaluate(probes)
    assert np.allclose(rebuilt.evaluate(probes), expected, rtol=1e-14, atol=0)
    # The steps at the block's edges are in the sum, as ENDF's repeated abscissae.
    for edge in edges:
        assert np.count_nonzero(rebuilt.xs == edge) == 2, f"no step at {edge}"
    assert info["n_negative"] == 0


def test_outside_the_block_the_sum_is_the_evaluations_value_for_value():
    """Even where the evaluation's sum is not the sum of its parts.

    The total here sits 3 % above its partial (a residual the evaluation has
    and the perturbation must not repair), and below 4 and above 7 every one of
    its values comes back as the same float.
    """
    grid = np.linspace(1.0, 10.0, 19)
    part = _xs(grid, 1.0 + grid ** 2)
    total = _xs(grid, 1.03 * (1.0 + grid ** 2) + 0.4)
    edges = [4.0, 7.0]
    after = _perturbed(part, [1.5], edges)

    rebuilt, _ = rederiveSum(total, [(part, after, 4.0, 7.0)])
    outside = (rebuilt.xs < 4.0) | (rebuilt.xs > 7.0)
    original = dict(zip(total.xs, total.ys))
    for x, y in zip(rebuilt.xs[outside], rebuilt.ys[outside]):
        assert y == original[x], f"the sum moved at {x}, outside the block"
    inside = (grid > 4.0) & (grid < 7.0)
    assert np.allclose(rebuilt.evaluate(grid[inside]),
                       total.evaluate(grid[inside])
                       + 0.5 * part.evaluate(grid[inside]), rtol=1e-14)


def test_the_sums_regions_and_laws_survive():
    """A two-region total keeps both regions, both laws and its own step."""
    xs = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 8.0])
    ys = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0])
    total = Regions1d.fromEndfRegions(xs, ys, [(4, 2), (7, 5)])
    part = _xs([1.0, 8.0], [0.5, 0.5])
    after = _perturbed(part, [2.0], [5.0, 7.0])

    rebuilt, _ = rederiveSum(total, [(part, after, 5.0, 7.0)])
    _xs_, _ys_, pairs = rebuilt.toEndfRegions()
    assert [code for _nbt, code in pairs] == [2, 5]
    assert np.count_nonzero(_xs_ == 3.0) == 2, "the evaluation's own step went"
    assert rebuilt.evaluate(2.5) == total.evaluate(2.5)
    assert rebuilt.evaluate(6.0) == pytest.approx(total.evaluate(6.0) + 0.5)


# ----------------------------------------------------------------------
# On the model, and through the tape
# ----------------------------------------------------------------------

def _sectionValues(path, mt, energies):
    section = read_endf(str(path), mf_numbers=[3]).get_file(3).sections[mt]
    return np.interp(energies, np.asarray(section.energies, float),
                     np.asarray(section.cross_sections, float))


def test_a_total_perturbed_alone_reaches_its_partials(tmp_path):
    """MT1's block, drawn on its own, moves MT2 and MT102 by the same factor.

    This is the case RECONR makes essential: it throws the tape's MT1 away, so a
    block that stayed on MT1 would not reach the ACE at all.
    """
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.nuclear_data.model import EVAL_LABEL

    suite, _ = decodeReactionSuite(read_endf(str(FE56_STRUCTURAL)))
    edges = np.array([1.0e-5, 1.0e6, 2.0e7])
    pset = PerturbationSet(label="r", factors={_key(1): np.array([1.1, 0.9])},
                           binEdges={_key(1): edges})
    applied = pset.applyToSuite(suite)

    assert set(applied) == {_key(1), _key(2), _key(102)}
    assert applied[_key(2)]["factor_from"] == 1
    assert applied[_key(1)]["own_block"] == (2, 102)
    for mt in (2, 102):
        cross = suite.reactionByENDF_MT(mt).crossSection
        probe = np.array([9.5e5, 5.0e6])   # above the resolved range
        ratio = cross["r"].evaluate(probe) / cross[EVAL_LABEL].evaluate(probe)
        assert np.allclose(ratio, [1.1, 0.9], rtol=1e-12)


def test_the_written_total_moved_by_what_its_partials_moved(tmp_path):
    """Through the delta tape: MT1_out - MT1_in = MT2_out - MT2_in, to ENDF's digits.

    On this tape MT1 is far above MT2 + MT102 (it is a full Fe-56 total over
    three surviving sections), which is the case a fresh resummation would get
    wrong and the delta form gets right.
    """
    run = perturbFromModel(str(FE56_XS), {33: None}, 1, seed=3,
                           outputDir=tmp_path, formats=("endf-delta",))
    delta = run.paths("endf-delta")[0]
    energies = np.geomspace(1.0e-4, 1.9e7, 4000)
    moved1 = _sectionValues(delta, 1, energies) - _sectionValues(FE56_XS, 1, energies)
    moved2 = _sectionValues(delta, 2, energies) - _sectionValues(FE56_XS, 2, energies)
    scale = _sectionValues(FE56_XS, 1, energies)
    assert np.max(np.abs(moved2)) > 0, "MT2 did not move"
    # Seven significant digits on both sections, and np.interp across the steps.
    assert np.all(np.abs(moved1 - moved2) <= 2e-6 * scale + 1e-12), (
        f"max mismatch {np.max(np.abs(moved1 - moved2) / scale):.2e} of MT1")
    assert any("re-derived" in note for note in run.notes)


def test_the_switch_restores_the_old_behaviour(tmp_path):
    """``crossSectionSums=False``: MT1 is left as written, and the run says so."""
    run = perturbFromModel(str(FE56_XS), {33: None}, 1, seed=3,
                           outputDir=tmp_path, formats=("endf-delta",),
                           crossSectionSums=False)
    delta = run.paths("endf-delta")[0]
    before = read_endf(str(FE56_XS), mf_numbers=[3]).get_file(3).sections[1]
    after = read_endf(str(delta), mf_numbers=[3]).get_file(3).sections[1]
    assert np.array_equal(before.cross_sections, after.cross_sections)
    assert any("NOT the sum of their parts" in note for note in run.notes)


@pytest.mark.slow
def test_every_sum_holds_on_a_whole_evaluation(fe56_b81_tape, tmp_path):
    """ENDF/B-VIII.1 Fe-56, every cross section its MF33 states, one sample.

    The request names sums and partials together (MF33 there carries MT1, MT4
    and partials of both), so this is rules 1-3 at once on a real tree. Checked
    per re-derived sum, against the leaves under it, on the sum's own grid.
    """
    from kika.endf.model_adapter.decode import _summationMTs as summation
    from kika.sampling.cross_section_sums import _leavesUnder

    run = perturbFromModel(str(fe56_b81_tape), {33: None}, 1, seed=7,
                           outputDir=tmp_path, formats=("endf-delta",))
    delta = run.paths("endf-delta")[0]
    source = read_endf(str(fe56_b81_tape), mf_numbers=[3]).get_file(3).sections
    written = read_endf(str(delta), mf_numbers=[3]).get_file(3).sections
    present = set(source)
    children, _ = sumTree(present, summation(present))
    rebuilt = [c.mt for c, info in run.samples[0]["applied"].items()
               if "rederived_from" in info]
    assert rebuilt, "nothing was re-derived on a tape that states MT1 and MT4"
    for component, info in run.samples[0]["applied"].items():
        assert not info.get("n_negative"), (
            f"MT{component.mt} went negative where no partial is: {info}")

    def values(sections, mt, energies):
        s = sections[mt]
        return np.interp(energies, np.asarray(s.energies, float),
                         np.asarray(s.cross_sections, float), left=0, right=0)

    pset = run.samples[0]["set"]
    stepEdges = np.unique(np.concatenate(
        [np.asarray(e, float) for e in pset.binEdges.values()]))

    for total in rebuilt:
        leaves = _leavesUnder(total, children)
        grid = np.unique(np.concatenate(
            [np.asarray(source[total].energies, float)]
            + [np.asarray(source[leaf].energies, float) for leaf in leaves]))
        grid = grid[(grid > float(source[total].energies[0]))
                    & (grid < float(source[total].energies[-1]))]
        # On a factor step np.interp picks one side arbitrarily; and the
        # partials' own steps (thresholds) likewise. Everywhere else the
        # identity is exact up to the seven digits ENDF stores.
        near = np.min(np.abs(grid[:, None] / stepEdges[None, :] - 1.0), axis=1) < 1e-9
        energies = grid[~near]
        movedTotal = values(written, total, energies) - values(source, total, energies)
        movedParts = sum(values(written, leaf, energies) - values(source, leaf, energies)
                         for leaf in leaves)
        # Absolute values: inside the resolved range MF3 holds backgrounds,
        # which can be negative, and a signed sum of them can cancel to zero.
        scale = np.abs(values(source, total, energies)) + sum(
            np.abs(values(written, leaf, energies)) for leaf in leaves)
        mismatch = np.abs(movedTotal - movedParts) / np.maximum(scale, 1e-30)
        assert np.max(mismatch) < 1e-5, (
            f"MT{total}: the sum and the {len(leaves)} partials under it moved "
            f"apart by {np.max(mismatch):.2e} at "
            f"{energies[np.argmax(mismatch)]:.6e} eV")


def test_a_negative_background_is_not_an_alarm_and_a_short_sum_is():
    """MF3 may state a negative background inside the resolved range.

    ENDF/B-VIII.1 Fe-56 has MT2 below zero at 15 points in 501-850 keV while
    its MT1 there is zero, so moving MT2 legitimately takes the re-derived MT1
    negative; RECONR adds the resonances back. Only a sum that goes negative
    while every partial under it is non-negative -- an evaluation stating the
    sum below its own parts -- is counted.
    """
    grid = np.linspace(1.0, 10.0, 10)
    total = _xs(grid, np.zeros(grid.size))
    background = _xs(grid, np.full(grid.size, -0.5))
    _, info = rederiveSum(total, [(background, _perturbed(background, [1.5], [1.0, 10.0]),
                                   1.0, 10.0)])
    assert info["n_negative"] == 0

    short = _xs(grid, np.full(grid.size, 0.2))
    part = _xs(grid, np.full(grid.size, 1.0))
    _, info = rederiveSum(short, [(part, _perturbed(part, [0.5], [1.0, 10.0]),
                                   1.0, 10.0)])
    assert info["n_negative"] > 0

"""MF3's sum rules on a realisation: partials govern, sums follow, nothing else moves.

The rule (decision 3 of ``kika-workspace/docs/library/perturbation_model_roadmap.md``,
settled with Juan on 2026-10-07) in four parts, each pinned here:

1. a partial with its own block is perturbed by it -- partials govern;
2. every sum with a moved partial is re-derived as ``S + sum (p' - p)``, and a
   sum's own block is never applied to the sum;
3. a sum the file decomposes (something under it carries a covariance) is
   perturbed through those sections only; its block is discarded and the
   partials without a covariance stay as evaluated -- ``sumBlocks="fill"``
   carries it to them instead, as a stated assumption;
4. a sum the file does not decompose moves all its partials by its own factor
   -- ``sumBlocks="never"`` drops that.

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

from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest

from kika._constants import MF3_SUM_RULES
from kika.endf import read_endf
from kika.endf.model_adapter.decode import _summationMTs
from kika.nuclear_data.model.functions import Regions1d, XYs1d
from kika.sampling.cross_section_sums import (Remainder, planCrossSectionSums,
                                              rederiveSum, screenSumClaims,
                                              sumTree)
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.perturbation_set import PerturbationSet

# The micro-tapes keep Fe-56's resonance region (MF2, LRP=1) but are cut from
# the full tape section by section, and NJOY cannot read them (RECONR stops at
# their orphan FEND records, "illegal TAB1 for mf/mt = 3/0"). So they are
# perturbed as stated -- resonanceRegion="evaluated" -- which is what these
# tests are about; the reconstruction itself is tested on full tapes, in
# test_the_resonance_region_is_perturbed_as_reconstructed.py.
perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56_XS = DATA / "micro_fe56_xs_and_angular.endf"
FE56_STRUCTURAL = DATA / "micro_fe56_structural.endf"
ZA = 26056


def _key(mt):
    return ComponentKey(ZA, 33, mt)


def _retag(lines, first, last, mt):
    out = []
    for offset, line in enumerate(lines[first:last + 1]):
        if offset == 1:
            assert line[33:44] == "          2", "the subsection names MT2"
            line = line[:33] + f"{mt:11d}" + line[44:]
        out.append(line[:72] + f"{mt:3d}" + line[75:])
    return out


@pytest.fixture
def fe56OnlyTotalCovariance(tmp_path):
    """``micro_fe56_xs_and_angular`` with its MF33 block restated as MT1's alone.

    MF3 has MT1, MT2 and MT102; MF33 has MT1 and nothing under it: a total the
    file does not decompose, which is ENDF/B-VIII.1 Fe-56's MT4 in miniature.
    """
    from kika.endf.writers.update_directory import update_mf1_directory

    lines = FE56_XS.read_text(encoding="utf-8").splitlines(keepends=True)
    section = [i for i, line in enumerate(lines)
               if line[70:72] == "33" and line[72:75] == "  2"]
    first, last = section[0], section[-1]
    out = tmp_path / "fe56_only_mt1_covariance.endf"
    out.write_text("".join(lines[:first] + _retag(lines, first, last, 1)
                           + lines[last + 1:]), encoding="utf-8", newline="")
    update_mf1_directory(str(out))
    return out


@pytest.fixture
def fe56WithTotalCovariance(tmp_path):
    """``micro_fe56_xs_and_angular`` with MT2's MF33 block also stated as MT1's.

    No committed micro-tape carries a covariance for a sum over partials it
    also states, and that is the case the strict rule is about: MF3 has MT1,
    MT2 and MT102; MF33 then has MT1 and MT2, and nothing for MT102.
    """
    lines = FE56_XS.read_text(encoding="utf-8").splitlines(keepends=True)
    section = [i for i, line in enumerate(lines)
               if line[70:72] == "33" and line[72:75] == "  2"]
    first, last = section[0], section[-1]
    copy = []
    for offset, line in enumerate(lines[first:last + 1]):
        if offset == 1:
            assert line[33:44] == "          2", "the subsection names MT2"
            line = line[:33] + "          1" + line[44:]
        copy.append(line[:72] + "  1" + line[75:])
    send = lines[last + 1]
    assert send[70:72] == "33" and send[72:75] == "  0"
    copy.append(send[:72] + "  0" + send[75:])
    out = tmp_path / "fe56_with_mt1_covariance.endf"
    out.write_text("".join(lines[:first] + copy + lines[first:]), encoding="utf-8",
                   newline="")
    from kika.endf.writers.update_directory import update_mf1_directory

    update_mf1_directory(str(out))
    return out


# ----------------------------------------------------------------------
# The plan: which block moves which MT
# ----------------------------------------------------------------------

def test_the_sum_rules_are_a_tree():
    """No MT has two parents, so "the nearest perturbed sum above it" is definite."""
    everything = set(MF3_SUM_RULES)
    for parts, _ in MF3_SUM_RULES.values():
        everything |= {int(str(part).lstrip("@")) for part in parts}
    sumTree(everything, MF3_SUM_RULES)          # raises on a second parent


def test_a_decomposed_sums_block_reaches_none_of_its_partials():
    """MT4 and MT51 carry blocks; MT52 does not, and so it does not move.

    With MT51 stated apart the file decomposes MT4, and MT4's block -- the
    uncertainty of MT51 + MT52 together -- no longer says what MT52 alone
    carries. MT1 is decomposed too (MT4 and MT51 under it are claimed).
    """
    present = {1, 2, 4, 16, 51, 52, 102}
    sums = _summationMTs(present)
    assert sums == {1, 4}
    plan = planCrossSectionSums({1: _key(1), 4: _key(4), 51: _key(51)},
                                present, sums)

    assert plan.leafControl == {51: _key(51)}, "only a partial's own block moves it"
    assert plan.rederive == (4, 1), "deepest first"
    assert plan.movedUnder == {4: (51,), 1: (51,)}
    assert plan.ownBlockReached == {1: (), 4: ()}, "both sums' blocks discarded"


def test_the_screen_sorts_the_sums_by_whether_the_file_decomposes_them():
    present = {1, 2, 4, 16, 51, 52, 102}
    screen = screenSumClaims({1, 2, 4}, present, _summationMTs(present))
    assert {k: set(v) for k, v in screen.discarded.items()} == {1: {2, 4}}, (
        "MT2 and the smaller sum MT4 under MT1 carry their own")
    assert screen.undecomposed == {4: (51, 52)}, "nothing under MT4 does"


def test_an_undecomposed_sum_moves_all_its_partials_alike():
    """MT4's block alone: the covariance is there, just not broken down."""
    present = {1, 2, 4, 16, 51, 52, 102}
    plan = planCrossSectionSums({4: _key(4)}, present, _summationMTs(present))
    assert plan.leafControl == {51: _key(4), 52: _key(4)}
    assert plan.rederive == (4, 1)
    assert plan.ownBlockReached == {4: (51, 52)}


def test_a_decomposed_total_leaves_an_undecomposed_branch_to_its_own_sum():
    """MT1, MT2 and MT4 claimed: MT4 still moves MT51-52; MT16, MT102 stay."""
    present = {1, 2, 4, 16, 51, 52, 102}
    plan = planCrossSectionSums({1: _key(1), 2: _key(2), 4: _key(4)},
                                present, _summationMTs(present))
    assert plan.leafControl == {2: _key(2), 51: _key(4), 52: _key(4)}
    assert plan.ownBlockReached == {1: (), 4: (51, 52)}


def test_the_leaves_of_each_sum_come_from_mf3_virtual_ones_included():
    """What the app's builder shows beside a sum: its partials, from MF3."""
    from kika.sampling.cross_section_sums import sumLeaves

    present = {1, 2, 4, 51, 52, 102, 600, 601}
    leaves = sumLeaves(present, _summationMTs(present), claims={103, 851})
    assert leaves[4] == (51, 52)
    assert set(leaves[1]) == {2, 51, 52, 102, 600, 601}
    assert leaves[103] == (600, 601), "a virtual sum counts its MF3 partials"
    assert 851 not in leaves, "a lumped MTL is not a sum of MF3 sections"
    assert sumLeaves(present, None, claims={103}) == leaves, (
        "deriving the sums from MF3 alone gives the decoder's tree")


@pytest.mark.parametrize("claims", [{1}, {4}, {1, 2, 4}])
def test_never_refuses_a_sum_block_with_nowhere_to_go(claims):
    """``sumBlocks="never"``: an undecomposed sum's block would be lost."""
    present = {1, 2, 4, 16, 51, 52, 102}
    with pytest.raises(ValueError, match="sumBlocks='never'"):
        planCrossSectionSums({mt: _key(mt) for mt in claims}, present,
                             _summationMTs(present), mode="never")


def test_fill_rides_the_nearest_sum_as_before():
    """``mode="fill"``: the explicit assumption, MT4 beating MT1 for MT52."""
    present = {1, 2, 4, 16, 51, 52, 102}
    plan = planCrossSectionSums({1: _key(1), 4: _key(4), 51: _key(51)},
                                present, _summationMTs(present), mode="fill")

    assert plan.leafControl[51] == _key(51), "a partial's own block governs"
    assert plan.leafControl[52] == _key(4), "MT4 is nearer to MT52 than MT1 is"
    for leaf in (2, 16, 102):
        assert plan.leafControl[leaf] == _key(1)
    assert plan.rederive == (4, 1)
    assert plan.ownBlockReached == {1: (2, 16, 102), 4: (52,)}


def test_a_lump_moves_its_members_as_their_own_block():
    """B-VIII.1 U-238: MT851 (MTL) is the covariance of MT52-91, stated for them.

    The members carry no block, their MF33 points at the lump, so the lump's
    factor moves them -- in every mode, ``never`` included: it is not a sum's
    block standing in for them, it is the one the file gives them. MT51 keeps
    its own, MT4 = MT51 + MT52-53 is decomposed and re-derived.
    """
    present = {1, 2, 4, 51, 52, 53}
    for mode in ("undecomposed", "never"):
        plan = planCrossSectionSums({51: _key(51), 851: _key(851)}, present,
                                    _summationMTs(present), mode=mode,
                                    lumped={851: (52, 53, 54)})
        assert plan.leafControl == {51: _key(51), 52: _key(851), 53: _key(851)}
        assert plan.lumpOf == {52: 851, 53: 851}
        assert plan.rederive == (4, 1)


def test_a_remainder_anchors_its_total():
    """JENDL-5 Fe-56: MF33 states MT2 as MT1 minus the rest.

    MT1 is decomposed (MT102 carries its own), so without the statement its
    block would be discarded; with it, MT1 is anchored, MT2 takes the
    difference, and MT1 is rebuilt only after it.
    """
    present = {1, 2, 4, 51, 52, 102}
    statement = Remainder(total=1, minus=(4, 102), lo=1e-5, hi=2e7)
    screen = screenSumClaims({1, 102}, present, _summationMTs(present),
                             remainders={2: (statement,)})
    assert screen.anchored == {1: (2,)}
    assert screen.discarded == {}
    plan = planCrossSectionSums({1: _key(1), 102: _key(102)}, present,
                                _summationMTs(present),
                                remainders={2: (statement,)})
    assert plan.leafControl == {102: _key(102)}, "MT2 does not ride MT1"
    assert plan.remainders == {2: (statement,)}
    assert plan.rederive[plan.remainderAt:] == (1,)


def test_a_remainder_waits_for_its_total_to_be_drawn():
    """MT102 asked for alone: MT1 is not drawn, so MT2 is left as evaluated."""
    present = {1, 2, 102}
    plan = planCrossSectionSums(
        {102: _key(102)}, present, _summationMTs(present),
        remainders={2: (Remainder(total=1, minus=(102,), lo=1e-5, hi=2e7),)})
    assert plan.remainders == {}
    assert plan.leafControl == {102: _key(102)}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="sumBlocks must be one of"):
        planCrossSectionSums({4: _key(4)}, {4, 51}, {4}, mode="distribute")


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


def test_a_sum_stated_only_through_its_partials():
    """ENDF/B-VIII.1 Fe-56: MF33 for MT103, MF3 for MT600-649 and no MT103.

    Its partials come from MF3 like any sum's, and none carries a block, so it
    is undecomposed: its block moves them -- what ``apply_factors_to_pendf_mf3``
    does for a composite the PENDF lacks -- and MT1 above them is rebuilt.
    """
    present = {1, 2, 102, 600, 601, 649}
    with pytest.raises(ValueError, match="MT103 over MT600-601, 649"):
        planCrossSectionSums({103: _key(103)}, present, _summationMTs(present),
                             mode="never")
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


def test_a_total_alone_is_refused_on_the_model_under_never():
    """MT1's block alone with ``sumBlocks="never"``: nowhere it may go."""
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _ = decodeReactionSuite(read_endf(str(FE56_STRUCTURAL)))
    pset = PerturbationSet(label="r", factors={_key(1): np.array([1.1, 0.9])},
                           binEdges={_key(1): np.array([1.0e-5, 1.0e6, 2.0e7])})
    with pytest.raises(ValueError, match="sumBlocks='never'"):
        pset.applyToSuite(suite, sumBlocks="never")


def test_an_undecomposed_total_reaches_its_partials():
    """MT1's block alone moves MT2 and MT102 by the same factor (the default).

    The case RECONR makes the only way to get a total's block to the ACE: it
    throws the tape's MT1 away, so a block that stayed on MT1 would not reach
    it.
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


def test_a_remainder_takes_the_difference_and_the_total_keeps_its_own_move():
    """MT1 and MT102 drawn, MT2 stated as MT1 - MT102.

    MT2' = MT2 + (MT1' - MT1) - (MT102' - MT102), so the total rebuilt from its
    partials is MT1 times MT1's own factor: the covariance the file states for
    the total is the one the realisation carries, and MT2's is the one the
    file defines for it. Checked above the resolved range, where MF3 is the
    cross section and not a background.
    """
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.nuclear_data.model import EVAL_LABEL

    suite, _ = decodeReactionSuite(read_endf(str(FE56_STRUCTURAL)))
    edges = np.array([1.0e-5, 1.0e6, 2.0e7])
    pset = PerturbationSet(
        label="r",
        factors={_key(1): np.array([1.10, 0.95]), _key(102): np.array([0.7, 1.3])},
        binEdges={_key(1): edges, _key(102): edges})
    statement = Remainder(total=1, minus=(102,), lo=1e-5, hi=2e7)
    applied = pset.applyToSuite(suite, remainders={2: (statement,)})

    assert applied[_key(2)]["remainder_of"] == 1
    assert applied[_key(1)]["own_block"] == "remainder"
    probe = np.array([9.0e5, 3.0e6, 1.2e7])
    form = {mt: suite.reactionByENDF_MT(mt).crossSection for mt in (1, 2, 102)}
    total = form[1]["r"].evaluate(probe) / form[1][EVAL_LABEL].evaluate(probe)
    assert np.allclose(total, [1.10, 0.95, 0.95], rtol=1e-12)
    expected2 = (form[2][EVAL_LABEL].evaluate(probe)
                 + (total - 1.0) * form[1][EVAL_LABEL].evaluate(probe)
                 - (form[102]["r"].evaluate(probe)
                    - form[102][EVAL_LABEL].evaluate(probe)))
    assert np.allclose(form[2]["r"].evaluate(probe), expected2, rtol=1e-12)


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


def test_a_sums_own_block_is_not_drawn_where_its_partials_carry_theirs(
        fe56WithTotalCovariance, tmp_path):
    """MF33 states MT1 and MT2: MT2 governs, MT1 follows, MT102 stays put."""
    tape = fe56WithTotalCovariance
    run = perturbFromModel(str(tape), {33: None}, 1, seed=3,
                           outputDir=tmp_path, formats=("endf-delta",))
    delta = run.paths("endf-delta")[0]
    assert {c.mt for c in run.samples[0]["set"].components()} == {2}
    assert any("MT1's own covariance was not drawn" in n for n in run.notes)
    energies = np.geomspace(1.0e-4, 1.9e7, 4000)
    assert np.array_equal(_sectionValues(delta, 102, energies),
                          _sectionValues(tape, 102, energies)), "MT102 moved"
    moved1 = _sectionValues(delta, 1, energies) - _sectionValues(tape, 1, energies)
    moved2 = _sectionValues(delta, 2, energies) - _sectionValues(tape, 2, energies)
    assert np.all(np.abs(moved1 - moved2)
                  <= 2e-6 * _sectionValues(tape, 1, energies) + 1e-12)
    meta = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["sumBlocks"]["mode"] == "undecomposed"
    assert meta["sumBlocks"]["discarded"] == {"1": [2]}
    assert meta["sumBlocks"]["undecomposed"] == {}


def test_naming_a_decomposed_sum_names_what_decomposes_it(fe56WithTotalCovariance):
    """``{33: [1]}`` with MF33 for MT1 and MT2: the total, through MT2.

    Juan, 2026-10-07: asking for the inelastic where MT51-52 are stated apart
    is asking for MT51-52. MT2 is added to the request and said to be.
    """
    run = perturbFromModel(str(fe56WithTotalCovariance), {33: [1]}, 1, seed=3,
                           dryRun=True)
    assert {c.mt for c in run.samples[0]["set"].components()} == {2}
    assert any("MT1 was asked for, and the file decomposes it: MT2" in n
               for n in run.notes), run.notes
    assert any("MT1's own covariance was not drawn" in n for n in run.notes)


def test_naming_an_undecomposed_sum_moves_its_partials(fe56OnlyTotalCovariance):
    """``{33: [1]}`` with MF33 for MT1 alone: MT2 and MT102 ride it."""
    run = perturbFromModel(str(fe56OnlyTotalCovariance), {33: [1]}, 1, seed=3,
                           dryRun=True)
    assert {c.mt for c in run.samples[0]["set"].components()} == {1}
    riders = {c.mt for c, info in run.samples[0]["applied"].items()
              if info.get("factor_from") == 1}
    assert riders == {2, 102}
    assert any("moved its unperturbed partials MT2, 102 by one common factor" in n
               for n in run.notes), run.notes


def test_never_refuses_a_named_undecomposed_sum(fe56OnlyTotalCovariance):
    with pytest.raises(ValueError, match="the request names the sum"):
        perturbFromModel(str(fe56OnlyTotalCovariance), {33: [1]}, 1, seed=3,
                         dryRun=True, sumBlocks="never")
    with pytest.raises(ValueError, match="nothing is left to perturb"):
        perturbFromModel(str(fe56OnlyTotalCovariance), {33: None}, 1, seed=3,
                         dryRun=True, sumBlocks="never")


def test_a_sum_and_its_partial_named_together_is_the_partial(
        fe56WithTotalCovariance):
    """``{33: [1, 2]}``: partials govern, so this is MT2 and a re-derived MT1."""
    run = perturbFromModel(str(fe56WithTotalCovariance), {33: [1, 2]}, 1, seed=3,
                           dryRun=True)
    assert {c.mt for c in run.samples[0]["set"].components()} == {2}
    assert any("MT1's own covariance was not drawn" in n for n in run.notes)


def test_a_named_partial_without_a_covariance_is_refused():
    """MT102 has no block of its own, and MT1's does not stand in for it."""
    with pytest.raises(ValueError, match="MT102: asked for"):
        perturbFromModel(str(FE56_XS), {33: [2, 102]}, 1, seed=3, dryRun=True)
    run = perturbFromModel(str(FE56_XS), {33: [2, 102]}, 1, seed=3, dryRun=True,
                           onMissing="skip")
    assert {c.mt for c in run.samples[0]["set"].components()} == {2}
    assert any("MT102 was asked for" in note for note in run.notes)


def test_fill_from_the_pipeline_names_its_riders(fe56WithTotalCovariance):
    """``sumBlocks="fill"``: MT1 is drawn and MT102 rides it, said aloud."""
    run = perturbFromModel(str(fe56WithTotalCovariance), {33: None}, 1, seed=3,
                           dryRun=True, sumBlocks="fill")
    assert {c.mt for c in run.samples[0]["set"].components()} == {1, 2}
    assert any("moved its unperturbed partials MT102" in note
               and "assumption" in note for note in run.notes)


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
    and partials of both), so this is the whole rule at once on a real tree.
    Checked per re-derived sum, against the leaves under it, on the sum's own
    grid. MT4 and MT103 carry blocks over partials that carry none (MT51-91,
    MT600-649), so they are undecomposed and move those partials; MT1's is
    discarded because MT2, MT4, MT5, MT16 and MT102 under it carry their own
    (measured 2026-10-07).
    """
    from kika.endf.model_adapter.decode import _summationMTs as summation
    from kika.sampling.cross_section_sums import _leavesUnder

    run = perturbFromModel(str(fe56_b81_tape), {33: None}, 1, seed=7,
                           outputDir=tmp_path, formats=("endf-delta",),
                           resonanceRegion="reconstructed")
    delta = run.paths("endf-delta")[0]
    # The tape the realisation was made from: the evaluation with its resonance
    # region reconstructed (MF3 from RECONR, LRP=2).
    base = json.loads((tmp_path / "run_metadata.json").read_text(
        encoding="utf-8"))["resonanceRegion"]["base"]
    source = read_endf(base, mf_numbers=[3]).get_file(3).sections
    written = read_endf(str(delta), mf_numbers=[3]).get_file(3).sections
    present = set(source)
    children, _ = sumTree(present, summation(present))
    rebuilt = [c.mt for c, info in run.samples[0]["applied"].items()
               if "rederived_from" in info]
    assert rebuilt, "nothing was re-derived on a tape that states MT1 and MT4"
    drawn = {c.mt for c in run.samples[0]["set"].components() if c.mf == 33}
    sums = summation(present) | {103}
    assert drawn & sums == {4, 103}, sorted(drawn & sums)
    assert any("MT1's own covariance was not drawn" in n for n in run.notes)
    riders = {}
    for component, info in run.samples[0]["applied"].items():
        if "factor_from" in info:
            riders.setdefault(info["factor_from"], set()).add(component.mt)
    assert set(riders) == {4, 103}, riders
    assert riders[4] == set(range(51, 92)) & present
    assert riders[103] == set(range(600, 650)) & present
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


@pytest.mark.slow
def test_jendl5_fe56_elastic_is_the_remainder_it_is_stated_as(fe56_jendl_tape, tmp_path):
    """JENDL-5 Fe-56 states MT2's covariance as MT1 minus every other channel.

    Before 2026-10-07 the model dropped that statement (an NC subsection) and
    the elastic was not perturbed at all. Now MT1 is drawn, MT2 takes the
    difference, and the written MT1 is the evaluation's times MT1's own factor
    wherever MT1's block reaches -- to the seven digits ENDF stores.
    """
    run = perturbFromModel(str(fe56_jendl_tape), {33: None}, 1, seed=11,
                           outputDir=tmp_path, formats=("endf-delta",),
                           resonanceRegion="reconstructed")
    base = json.loads((tmp_path / "run_metadata.json").read_text(
        encoding="utf-8"))["resonanceRegion"]["base"]
    sample = run.samples[0]
    assert sample["applied"][ComponentKey(ZA, 33, 2)]["remainder_of"] == 1
    assert any("MT2 took MT1's move" in note for note in run.notes), run.notes

    key = ComponentKey(ZA, 33, 1)
    factors = np.asarray(sample["set"].factors[key], float)
    edges = np.asarray(sample["set"].binEdges[key], float)
    delta = run.paths("endf-delta")[0]
    mids = np.sqrt(edges[:-1] * edges[1:])
    keep = mids > 1.0e6                       # above the resolved range
    ratio = (_sectionValues(delta, 1, mids[keep])
             / _sectionValues(base, 1, mids[keep]))
    assert np.allclose(ratio, factors[keep], rtol=2e-6), (
        f"max {np.max(np.abs(ratio / factors[keep] - 1)):.2e}")

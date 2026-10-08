"""MF34 in LAB, MF4 in CM: the factors act on the LAB coefficients.

ENDF-6 §34.1 allows MF34's LCT to differ from MF4's. ENDF/B-VIII.1 does this for
U-235 and U-238, and both ENDF/B-VIII.1 and JEFF-4.0 do it for Th-232, Mn-55 and
W-180…186. In all of them the covariance is of LAB coefficients and MF4 is in
CM. A drawn factor then describes a LAB coefficient. Applying it to the CM
coefficient perturbs a different quantity. `kika._legendre_frames` applies it in
LAB and brings the change back to CM: option A of
``docs/library/cov_checks_roadmap.md``, N2.

**The synthetic tape** is ``micro_fe56_structural.endf`` (JEFF-4.0 Fe-56,
MF4/MT2 in CM, LTT=3, MF34 L=1..6 on three grids) with every MF34 LCT rewritten
to 1. The tape keeps its numbers and changes only what the covariance claims
about its frame.

**What is checked independently.** The LAB coefficients are computed by
``kika.cov.multigroup.collapse.cm_to_lab_legendre``, which has its own
quadrature and its own cosine map (``kika.exfor``'s), not the module under test.
The factors each node received are read off the same run without conversion
(perturbed over baseline, order by order), so no private helper of the applier
enters the check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._legendre_frames import (CM, LAB, FrameConversion, normaliseFrame,
                                   scaleInOtherFrame, transferMatrix,
                                   twoBodyGamma)
from kika.cov.multigroup.collapse import cm_to_lab_legendre
from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF4MT
from kika.nuclear_data.model.perturbation import (_legendreRegions,
                                                  applyLegendreFactors)
from kika.sampling.endf_perturbation import (_apply_factors_to_mf4_legendre,
                                             _frame_conversion_for,
                                             _frames_from_index,
                                             _parameter_mapping_from_index)
from kika.sampling.model_blocks import legendre_covariance_index, mf34_frames
from kika.sampling.perturbation_set import (PerturbationSet,
                                            readFactorsTable,
                                            writeFactorsTable)

TAPE = str(Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
           / "micro_fe56_structural.endf")
MT = 2
ZA = 26056


@pytest.fixture(scope="module")
def labTape(tmp_path_factory):
    """The micro-tape with MF34 claiming LAB (LCT=1) throughout."""
    from kika.endf.writers.endf_writer import ENDFWriter

    endf = read_endf(TAPE)
    rewritten = 0
    for subsection in endf.get_file(34).sections[MT].subsections:
        for subSubsection in subsection.sub_subsections:
            subSubsection.lct = 1
            rewritten += 1
    assert rewritten > 0
    out = tmp_path_factory.mktemp("lab") / "fe56_mf34_lab.endf"
    assert ENDFWriter(TAPE).replace_mf_section(endf.files[34], str(out))
    return str(out)


def _entry(tape):
    suite, _report = decodeCovarianceSuite(read_endf(tape))
    (entry,) = legendre_covariance_index(suite, relative=True).values()
    return suite, entry


def _labOf(vector, awr, order_max=12, q=0.0, energy=1.0e6):
    """LAB coefficients of a CM vector, by the collapse module's transform."""
    gamma = twoBodyGamma(energy, awr, q)
    # cm_to_lab_legendre uses alpha = gamma; it returns the orders it is given.
    coeffs = {k: np.array([vector[k] if k < len(vector) else 0.0])
              for k in range(max(order_max + 1, len(vector)))}
    lab = cm_to_lab_legendre(coeffs, gamma, n_quad=256)
    return np.array([lab[k][0] for k in range(order_max + 1)])


def _labOfMany(vectors, awr, order_max):
    """:func:`_labOf` for many elastic nodes at once (γ = 1/A at every one)."""
    width = max(order_max + 1, max(len(v) for v in vectors))
    stacked = np.zeros((width, len(vectors)))
    for at, vector in enumerate(vectors):
        stacked[:len(vector), at] = vector
    lab = cm_to_lab_legendre({k: stacked[k] for k in range(width)}, 1.0 / awr,
                             n_quad=256)
    return {k: lab[k] for k in range(order_max + 1)}


# ----------------------------------------------------------------------
# The numerics
# ----------------------------------------------------------------------

def test_the_transfer_matrix_agrees_with_the_collapse_transform():
    rng = np.random.default_rng(7)
    a = np.concatenate(([1.0], 0.3 * rng.standard_normal(9)))
    for awr in (1.9, 11.9, 55.45, 236.0):
        gamma = 1.0 / awr
        mine = transferMatrix(gamma, 14, a.size, "CM->LAB") @ a
        coeffs = {k: np.array([a[k] if k < a.size else 0.0]) for k in range(14)}
        theirs = cm_to_lab_legendre(coeffs, gamma, n_quad=256)
        np.testing.assert_allclose(mine, [theirs[k][0] for k in range(14)],
                                   rtol=0, atol=1e-13)


def test_the_two_directions_are_inverse():
    for gamma in (1 / 236.0, 1 / 12.0, 0.4):
        there = transferMatrix(gamma, 60, 12, "CM->LAB")
        back = transferMatrix(gamma, 12, 60, "LAB->CM")
        np.testing.assert_allclose(back @ there, np.eye(12), rtol=0, atol=1e-10)


def test_gamma_is_one_over_a_for_elastic_and_infinite_at_threshold():
    assert twoBodyGamma(1.0e6, 55.45) == pytest.approx(1 / 55.45, rel=1e-15)
    q = -846.8e3                                  # Fe-56 first level
    threshold = -q * (55.45 + 1) / 55.45
    assert twoBodyGamma(threshold, 55.45, q) == float("inf")
    assert 0 < twoBodyGamma(2 * threshold, 55.45, q) < 1


def test_unit_factors_return_the_evaluation_bit_for_bit():
    a = np.array([1.0, 0.31, -0.04, 7e-3, 1.2e-4])
    out, _info = scaleInOtherFrame(a, {l: 1.0 for l in range(1, 7)}, 1 / 55.45,
                                   "CM->LAB")
    assert out.shape == a.shape and np.array_equal(out, a)


@pytest.mark.parametrize("awr", [11.9, 55.45, 236.0])
def test_the_lab_coefficients_take_exactly_the_drawn_factors(awr):
    """The orders the covariance states move by their factors; the rest stay put."""
    a = np.array([1.0, 0.31, -0.04, 7e-3, 1.2e-4])
    factors = {1: 1.08, 2: 0.9, 3: 1.2, 4: 0.85, 5: 1.1, 6: 0.95}
    out, info = scaleInOtherFrame(a, factors, 1 / awr, "CM->LAB")
    before, after = _labOf(a, awr, 20), _labOf(out, awr, 20)
    expected = before.copy()
    for order, factor in factors.items():
        expected[order] *= factor
    np.testing.assert_allclose(after[:7], expected[:7], rtol=1e-12, atol=1e-15)
    # Orders 7+ are not in the covariance. What the CM correction leaks there
    # is the pad's tolerance, far below anything ENDF writes.
    assert np.abs(after[7:] - before[7:]).max() < 1e-11
    assert info["condition"] < 2.0


def test_applying_in_cm_is_not_the_same_thing():
    """The defect this fixes, measured: in CM, the LAB a_1 of U-238 at 1 keV
    hardly moves, because it is almost all kinematics (2/3A) and not evaluation."""
    awr, a = 236.0, np.array([1.0, 2.0e-4, 1.0e-5])
    lab = _labOf(a, awr, 4)
    wrong = a.copy()
    wrong[1] *= 1.10
    moved = _labOf(wrong, awr, 4)[1] / lab[1] - 1.0
    right, _ = scaleInOtherFrame(a, {1: 1.10}, 1 / awr, "CM->LAB")
    assert _labOf(right, awr, 4)[1] / lab[1] - 1.0 == pytest.approx(0.10, rel=1e-10)
    assert moved < 0.01        # a tenth of the drawn 10 %, at most


def test_a_node_at_threshold_is_left_alone():
    a = np.array([1.0, 0.0, 0.0])
    out, info = scaleInOtherFrame(a, {1: 1.1}, float("inf"), "CM->LAB")
    assert out is None and info["gamma"] == float("inf")


def test_the_other_direction_lab_distribution_cm_covariance():
    """MF4 in LAB and MF34 in CM: no tape in the census does it, and the module
    covers it the same way."""
    gamma = 1 / 30.0
    a_lab = np.array([1.0, 0.25, 0.05, 0.01])
    factors = {1: 0.9, 2: 1.15, 3: 1.05}
    out, _ = scaleInOtherFrame(a_lab, factors, gamma, "LAB->CM")
    to_cm = transferMatrix(gamma, 8, 40, "LAB->CM")
    pad = lambda v: np.pad(v, (0, 40 - v.size))  # noqa: E731
    before, after = to_cm @ pad(a_lab), to_cm @ pad(out)
    for order, factor in factors.items():
        assert after[order] == pytest.approx(factor * before[order], rel=1e-11)


def test_frames_normalise_from_every_spelling():
    assert normaliseFrame("same-as-MF4") is None and normaliseFrame(None) is None
    assert normaliseFrame("LAB") == LAB and normaliseFrame("lab") == LAB
    assert normaliseFrame("centerOfMass") == CM and normaliseFrame(2) == CM
    with pytest.raises(ValueError):
        normaliseFrame("rest")
    assert FrameConversion.between("CM", "same-as-MF4", awr=55.45) is None
    assert FrameConversion.between("CM", "CM", awr=55.45) is None
    with pytest.raises(ValueError, match="two-body"):
        FrameConversion.between("CM", "LAB", awr=55.45, mt=16)


# ----------------------------------------------------------------------
# The frame reaches the index
# ----------------------------------------------------------------------

def test_the_shipped_tape_states_no_frame(labTape):
    suite, entry = _entry(TAPE)
    assert mf34_frames(suite) == {}
    assert entry["frames"] == {} and _frames_from_index(entry) == {}


def test_the_lab_tape_carries_lab_to_every_triplet(labTape):
    suite, entry = _entry(labTape)
    assert mf34_frames(suite) == {(ZA, MT): LAB}
    assert set(entry["frames"]) == set(entry["triplets"])
    assert set(entry["frames"].values()) == {LAB}
    assert _frames_from_index(entry) == {MT: LAB}


# ----------------------------------------------------------------------
# The two appliers on the synthetic tape
# ----------------------------------------------------------------------

def _factorVector(size: int) -> np.ndarray:
    return 1.0 + 0.05 * np.cos(np.arange(size, dtype=float))


def _cutPerOrder(factors, entry):
    stride = entry["stride"]
    byOrder, edges = {}, {}
    for position, triplet in enumerate(entry["triplets"]):
        order = int(triplet[2])
        grid = np.asarray(entry["grids"][triplet], dtype=float)
        byOrder[order] = np.asarray(factors[position * stride:
                                            position * stride + len(grid) - 1])
        edges[order] = grid
    return byOrder, edges


def _table(node):
    energies, vectors = [], []
    for _c, _p, region in _legendreRegions(node):
        for inner in region.function1ds:
            energies.append(float(inner.outerDomainValue))
            vectors.append(np.asarray(inner.coefficients, dtype=float))
    return np.asarray(energies), vectors


@pytest.fixture(scope="module")
def modelRuns(labTape):
    _suite, entry = _entry(labTape)
    param_mapping, _grids = _parameter_mapping_from_index(entry)
    factors = _factorVector(len(param_mapping))
    byOrder, edges = _cutPerOrder(factors, entry)
    unit = {order: np.ones_like(values) for order, values in byOrder.items()}
    distribution, provenance, _ = decodeMF4MT(read_endf(labTape).get_file(4)
                                              .sections[MT])
    awr = read_endf(labTape).get_file(4).sections[MT]._awr
    conversion = FrameConversion.between(distribution.productFrame, LAB,
                                         awr=awr, mt=MT)
    assert conversion is not None and conversion.direction == "CM->LAB"
    angular = distribution.angular
    base, _ = applyLegendreFactors(angular, unit, edges, coverageEdges="ramp")
    inCM, _ = applyLegendreFactors(angular, byOrder, edges, coverageEdges="ramp")
    inLAB, diagnostics = applyLegendreFactors(angular, byOrder, edges,
                                              coverageEdges="ramp",
                                              frameConversion=conversion)
    return {"base": _table(base), "inCM": _table(inCM), "inLAB": _table(inLAB),
            "diagnostics": diagnostics, "awr": awr, "factors": factors,
            "entry": entry, "edges": edges}


def test_the_model_applier_moves_the_lab_coefficients_by_the_drawn_factors(modelRuns):
    energies, base = modelRuns["base"]
    _, inCM = modelRuns["inCM"]
    energiesLAB, inLAB = modelRuns["inLAB"]
    assert np.array_equal(energies, energiesLAB)
    awr = modelRuns["awr"]
    before, after = _labOfMany(base, awr, 8), _labOfMany(inLAB, awr, 8)
    checked = 0
    for at, (energy, b, c) in enumerate(zip(energies, base, inCM)):
        # The factor each order received at this node, read off the CM run.
        factors = {order: c[order] / b[order] for order in range(1, min(7, b.size))
                   if b[order] != 0.0}
        if not factors or all(abs(f - 1.0) < 1e-15 for f in factors.values()):
            continue
        for order, factor in factors.items():
            assert after[order][at] == pytest.approx(
                factor * before[order][at], rel=1e-10, abs=1e-14), (energy, order)
        checked += 1
    assert checked > 1000


def test_the_model_applier_reports_the_conversion(modelRuns):
    frame = modelRuns["diagnostics"]["frame"]
    assert frame["direction"] == "CM->LAB" and frame["n_degenerate"] == 0
    assert frame["n_converted"] > 1000 and frame["max_condition"] < 1.1
    assert modelRuns["diagnostics"]["orders_absent"] == []


def test_without_a_conversion_nothing_changes(modelRuns, labTape):
    """`frameConversion=None` is the old applier, bit for bit."""
    entry, edges = modelRuns["entry"], modelRuns["edges"]
    byOrder, _ = _cutPerOrder(modelRuns["factors"], entry)
    distribution, _p, _r = decodeMF4MT(read_endf(TAPE).get_file(4).sections[MT])
    again, _ = applyLegendreFactors(distribution.angular, byOrder, edges,
                                    coverageEdges="ramp")
    _, vectors = _table(again)
    for mine, theirs in zip(vectors, modelRuns["inCM"][1]):
        assert np.array_equal(mine, theirs)


def test_the_legacy_applier_agrees_with_the_model_off_the_bin_edges(modelRuns, labTape):
    """`perturb_ENDF_files`' applier, with the frames from the index.

    Compared off the bin edges: at an edge the legacy applier keeps the
    baseline of every order whose own grid has no edge there (D32 in
    ``library-gaps.md``), with or without a conversion.
    """
    entry = modelRuns["entry"]
    param_mapping, energy_grids = _parameter_mapping_from_index(entry)
    endf = read_endf(labTape)
    mtData = endf.get_file(4).sections[MT]
    conversion = _frame_conversion_for(endf, mtData, _frames_from_index(entry)[MT])
    assert conversion == FrameConversion(CM, LAB, mtData._awr, 0.0, MT)
    _apply_factors_to_mf4_legendre(mtData, modelRuns["factors"], param_mapping,
                                   energy_grids, verbose=False,
                                   frame_conversion=conversion)
    legacyE = np.asarray(mtData._energies, dtype=float)
    energies, inLAB = modelRuns["inLAB"]
    assert np.array_equal(legacyE, energies)
    edges = np.unique(np.concatenate(list(modelRuns["edges"].values())))
    compared = 0
    for energy, model, legacy in zip(energies, inLAB, mtData._legendre_coeffs):
        if np.any(np.isclose(edges, energy, rtol=0, atol=1e-10)):
            continue
        np.testing.assert_allclose(np.concatenate(([1.0], legacy)), model,
                                   rtol=0, atol=1e-15)
        compared += 1
    assert compared > 1000


# ----------------------------------------------------------------------
# The perturbation set carries the frame
# ----------------------------------------------------------------------

def test_a_perturbation_set_carries_the_frame_through_disk(modelRuns, tmp_path):
    entry = modelRuns["entry"]
    index = {(ZA, "MF34", tuple(entry["triplets"])): {
        "triplets": entry["triplets"], "stride": entry["stride"],
        "grids": entry["grids"], "widths": entry["widths"],
        "frames": entry["frames"]}}
    drawn = _factorVector(entry["dimension"])
    first = PerturbationSet.fromDraw(drawn, index, label="realization-0000")
    assert set(first.componentFrames.values()) == {LAB}
    assert len(first.componentFrames) == len(entry["triplets"])
    assert PerturbationSet.from_dict(first.to_dict()).componentFrames == first.componentFrames
    writeFactorsTable([first], tmp_path)
    assert readFactorsTable(tmp_path, 0).componentFrames == first.componentFrames


def test_a_perturbation_set_refuses_a_frame_on_a_cross_section(modelRuns):
    from kika.sampling.joint_blocks import ComponentKey

    key = ComponentKey(ZA, 33, 2, 0)
    with pytest.raises(ValueError, match="MF34 components only"):
        PerturbationSet(label="x", factors={key: np.ones(1)},
                        binEdges={key: np.array([1.0, 2.0])},
                        componentFrames={key: LAB})


def test_the_joint_path_hands_the_frames_to_the_mf4_applier():
    """`perturb_joint_mf33_mf34` writes MF4 on a base tape with no MF34 on it, so
    the frame has to arrive with the factors."""
    from kika.sampling.joint_perturbation import _split_factors

    index = {"n_sigma": 0, "triplets": [(ZA, MT, 1)], "stride": 2,
             "grids": {(ZA, MT, 1): np.array([1.0, 2.0, 3.0])},
             "widths": {(ZA, MT, 1): 2}, "frames": {MT: LAB}}
    endf_pre, _ = _split_factors(np.ones((3, 2)), index, "mf34")
    assert endf_pre["frames"] == {MT: LAB}
    index["frames"] = {}
    endf_pre, _ = _split_factors(np.ones((3, 2)), index, "mf34")
    assert endf_pre["frames"] == {}

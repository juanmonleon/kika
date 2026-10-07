"""``flatAngularDistribution`` reads LTT and the frame from the form, not only from an MF4 header.

The only caller used to be ``AngularDistribution.from_endf``, which hands it the
MF4 section's own provenance (``ltt``, ``lct`` in ``headerFields``). Given any
other provenance -- the suite's, or one decoded from G4NDL -- it fell back to
"Legendre in the centre of mass": an LTT=3 section crashed (the table's
``XYs1d`` reached the Legendre densifier) and a LAB section came out as CM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.nuclear_data.angular_distribution import AngularDistribution
from kika.nuclear_data.model.enums import Frame
from kika.nuclear_data.model.interop import flatAngularDistribution

DATA = Path(kika.__file__).parent / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"


def _from_tape():
    return AngularDistribution.from_endf(kika.read_endf(str(FE56), mf_numbers=[4]).files[4].mt[2])


def _distribution(suite):
    return suite.reactions[2].outputChannel.products.byPid("n")[0].distribution["eval"]


def _same(a: AngularDistribution, b: AngularDistribution):
    assert a.representation == b.representation and a.frame == b.frame
    np.testing.assert_array_equal(a.energies, b.energies)
    assert sorted(a.coefficients) == sorted(b.coefficients)
    for order in a.coefficients:
        np.testing.assert_array_equal(a.coefficients[order], b.coefficients[order])
    mu = np.linspace(-1, 1, 41)
    for e in (1e3, 1e6, 3e6, 4.5e7, 1e8):
        np.testing.assert_allclose(a.evaluate_pdf(e, mu)[1], b.evaluate_pdf(e, mu)[1], rtol=1e-12)


def test_a_mixed_section_with_the_suite_provenance():
    suite = kika.read(str(FE56), format="endf")
    flat = AngularDistribution(**flatAngularDistribution(_distribution(suite), suite.provenance, 2))
    reference = _from_tape()
    assert reference.representation == "mixed"
    _same(flat, reference)


def test_a_section_read_back_from_g4ndl(tmp_path):
    import kika.g4ndl as g4ndl
    from kika.endf.model_adapter.pendf import attachReconstruction

    suite = kika.read(str(FE56), format="endf")
    attachReconstruction(suite, FE56)
    kika.write(suite, tmp_path, format="g4ndl")
    back = g4ndl.open(tmp_path).read("Fe56")
    flat = AngularDistribution(**flatAngularDistribution(_distribution(back), back.provenance, 2))
    _same(flat, _from_tape())


def test_the_frame_comes_from_the_form():
    suite = kika.read(str(FE56), format="endf")
    dist = _distribution(suite)
    assert flatAngularDistribution(dist, suite.provenance, 2)["frame"] == "CM"
    dist.productFrame = Frame.lab
    try:
        assert flatAngularDistribution(dist, suite.provenance, 2)["frame"] == "LAB"
    finally:
        dist.productFrame = Frame.centerOfMass

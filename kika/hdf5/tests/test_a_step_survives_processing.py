"""A step in a 0 K cross section is still a step at temperature (D36).

ENDF writes a discontinuity as a repeated energy. ``processSuite`` built each
temperature's union grid with ``np.unique`` of the raw abscissae, which keeps
one knot per energy, so the two faces of a step collapsed into one: the value
at the step was the left face and the right face was only reached at the next
grid point, a ramp of one interval. Measured on ENDF/B-VIII.1 Au-197 at
293.6 K, the elastic at the top of the URR went 9.9007 b at 100 000 eV to
9.7934 b at 100 250 eV, where the official OpenMC file (and NJOY) has the jump
in place, 9.9007 -> 9.7986 b; MT51 was off by 8.95 % in the middle of that
ramp. On the committed Fe-56 slice the right face of the 850 keV step came out
1.4407 b (MT2) and 0.004762 b (MT102) against 1.3959 and 0.0055107.

The grid now takes each form's abscissae with the repeated energy nudged up
first, as ACER does, so both faces are knots.
"""
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeReactionSuite
from kika.hdf5.processing import _flatten, processSuite

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"


@pytest.fixture(scope="module")
def suites():
    suite, _ = decodeReactionSuite(read_endf(str(TAPE)))
    processed = processSuite(suite, [293.6], endf=TAPE, omit=("urr", "kerma", "damage"),
                             thinning=None, workers=1)
    return suite, processed


def _steps(suite):
    """``{MT: [(E, left, right)]}`` of every 0 K step with a real jump."""
    out = {}
    for reaction in suite.reactions:
        x, y = _flatten(reaction.crossSection["eval"])
        at = np.flatnonzero((np.diff(x) == 0) & (y[:-1] != y[1:]))
        if at.size:
            out[reaction.ENDF_MT] = [(x[i], y[i], y[i + 1]) for i in at]
    return out


@pytest.mark.slow
def test_both_faces_of_a_step_are_on_the_processed_grid(suites):
    suite, processed = suites
    steps = _steps(suite)
    # The slice's step is 850 keV in MT2 and MT102, above the broadening
    # ceiling, where the 0 K values are carried unchanged.
    assert set(steps) == {2, 102}, steps
    for mt, faces in steps.items():
        form = processed.reactionByENDF_MT(mt).crossSection
        label = next(label for label in form.forms if label.startswith("heated"))
        x, y = np.asarray(form[label].xs), np.asarray(form[label].ys)
        for energy, _left, right in faces:
            # Two knots, one on each face. (The left face is not compared: below
            # 850 keV MT2 and MT102 are the reconstructed resonances, not MF3's
            # background, which is what the 0 K step's left value is.)
            at = int(np.searchsorted(x, energy))
            assert x[at] == energy and 0 < x[at + 1] / energy - 1 < 1e-11, (mt, x[at:at + 2])
            # Just past the step: the right face, not a point on a ramp.
            assert np.interp(energy * (1 + 1e-9), x, y) == pytest.approx(right, rel=1e-6), (
                mt, energy)

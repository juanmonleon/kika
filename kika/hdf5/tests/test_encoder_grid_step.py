"""The HDF5 encoder takes a grid with a step, and still refuses one that goes down.

kika's own ACE keeps a discontinuity as the same energy twice (ENDF/B-VIII.1
Fe-54 has 14), so a suite read from it reached ``encodeNeutron`` with a
repeated energy and was refused as "not strictly increasing". The encoder now
nudges the right face up by one part in 1e12, as ``processSuite`` does.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kika.hdf5.exceptions import HDF5FormatError
from kika.hdf5.neutron_encode import _grid


class _Form:
    def __init__(self, xs):
        self.xs = np.asarray(xs, float)

    def __len__(self):
        return len(self.xs)


def _suite(xs):
    reaction = SimpleNamespace(crossSection=SimpleNamespace(forms={'294K': _Form(xs)}))
    return SimpleNamespace(reactions=[reaction], fissionComponents=[], sums=[],
                           productions=[], incompleteReactions=[])


def test_a_step_is_nudged_up_and_the_rest_kept():
    grid = _grid(_suite([1.0, 2.0, 2.0, 3.0]), '294K')
    assert np.all(np.diff(grid) > 0)
    assert grid[1] == 2.0
    assert grid[2] == pytest.approx(2.0, rel=2e-12) and grid[2] > 2.0
    assert grid[[0, 3]].tolist() == [1.0, 3.0]


def test_a_grid_that_goes_down_is_still_refused():
    with pytest.raises(HDF5FormatError, match="not strictly increasing"):
        _grid(_suite([1.0, 3.0, 2.0]), '294K')

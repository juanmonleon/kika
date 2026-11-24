"""A photon continuum read back from GNDS as a Regions1d (a step in E') flattens to LAW=1's points."""
import numpy as np

from kika.endf.model_adapter.photon_lines import _spectrumPoints
from kika.nuclear_data.model import Regions1d, XYs1d


def test_the_step_keeps_both_values_at_the_shared_energy():
    node = Regions1d(function1ds=[
        XYs1d(xs=np.array([0.0, 1.0, 2.0]), ys=np.array([1.0, 2.0, 3.0])),
        XYs1d(xs=np.array([2.0, 3.0]), ys=np.array([0.0, 0.0])),
    ])
    xs, ys = _spectrumPoints(node)
    assert list(xs) == [0.0, 1.0, 2.0, 2.0, 3.0]
    assert list(ys) == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_a_plain_table_is_its_own_points():
    node = XYs1d(xs=np.array([0.0, 1.0]), ys=np.array([4.0, 5.0]))
    xs, ys = _spectrumPoints(node)
    assert list(xs) == [0.0, 1.0] and list(ys) == [4.0, 5.0]

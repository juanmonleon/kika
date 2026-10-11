"""Thinning rows on one grid keeps every dropped point within the tolerance."""
import numpy as np
import pytest

from kika.algebra import thin_common_grid


def test_every_row_is_reproduced_at_every_dropped_point():
    x = np.geomspace(1e-5, 2e7, 20001)
    ys = np.vstack([1 / np.sqrt(x) + 1, np.where(x > 1e6, (x - 1e6) / 1e6, 0.0), 3 + np.sin(np.log(x))])
    kept = thin_common_grid(x, ys, rtol=1e-3, atol=1e-12)
    assert kept[0] == 0 and kept[-1] == len(x) - 1
    assert len(kept) < len(x) // 10
    for y in ys:
        back = np.interp(x, x[kept], y[kept])
        assert np.all(np.abs(back - y) <= 1e-3 * np.abs(y) + 1e-12)


def test_mandatory_points_stay_and_values_are_untouched():
    x = np.linspace(0, 1, 101)
    ys = np.vstack([x])
    kept = thin_common_grid(x, ys, keep=[37, 50])
    assert set([0, 37, 50, 100]) <= set(kept)
    assert list(kept) == [0, 37, 50, 100]


def test_a_kink_is_kept():
    x = np.linspace(-1, 1, 201)
    kept = thin_common_grid(x, np.abs(x)[None, :], rtol=1e-6)
    assert 100 in kept

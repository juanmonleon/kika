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


def test_a_per_point_tolerance_is_the_tolerance_of_the_chord_that_starts_there():
    x = np.geomspace(1e-3, 10.0, 20001)
    y = 1 / np.sqrt(x)
    rtol = np.where(x < 1.0, 2e-4, 1e-3)
    kept = thin_common_grid(x, y[None, :], rtol=rtol)
    back = np.interp(x, x[kept], y[kept])
    left = kept[np.searchsorted(kept, np.arange(len(x)), side="right") - 1]
    error = np.abs(back - y) / y
    assert np.all(error <= rtol[left])
    # the loose chords use more than the tight tolerance, so the tight one is what holds
    assert error[x[left] >= 1.0].max() > 2e-4

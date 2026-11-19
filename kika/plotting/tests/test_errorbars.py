import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting import draw_errorbars


def _render(draw, log_x=True):
    fig, ax = plt.subplots(figsize=(4, 3))
    if log_x:
        ax.set_xscale("log")
    draw(ax)
    ax.set_xlim(5e-4, 2.0)
    ax.set_ylim(0.0, 2.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=100)
    plt.close(fig)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _data(n=60):
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(1e-3, 1.0, n))
    y = 1.0 + rng.random(n)
    return x, y, 0.05 * y, 0.02 * x


@pytest.mark.parametrize("capsize", [0, 2])
@pytest.mark.parametrize("which", ["x", "y", "both"])
def test_same_pixels_as_errorbar_when_opaque(capsize, which):
    x, y, ye, xe = _data()
    yerr = ye if which in ("y", "both") else None
    xerr = xe if which in ("x", "both") else None
    ref = _render(lambda ax: ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="none", ecolor="C1",
                                         elinewidth=1.5, capsize=capsize))
    new = _render(lambda ax: draw_errorbars(ax, x, y, yerr=yerr, xerr=xerr, color="C1",
                                            linewidth=1.5, capsize=capsize))
    differing = np.count_nonzero(np.any((ref.reshape(-1, 4) != new.reshape(-1, 4)), axis=1))
    # Anti-aliasing along a joined path differs from separate segments by a few edge pixels.
    assert differing / (ref.size // 4) < 1e-3


def test_asymmetric_and_nan_errors_match_errorbar_limits():
    x, y, ye, _ = _data(10)
    ye = ye.copy()
    ye[3] = np.nan
    asym = np.vstack([ye, 2 * ye])
    fig, ax = plt.subplots()
    draw_errorbars(ax, x, y, yerr=asym)
    ours = ax.dataLim.bounds
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=asym, fmt="none")
    theirs = ax.dataLim.bounds
    plt.close(fig)
    np.testing.assert_allclose(ours, theirs)


def test_rejects_errors_of_the_wrong_length():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_errorbars(ax, [1, 2, 3], [1, 2, 3], yerr=[0.1, 0.2])
    plt.close(fig)


def test_no_points_draws_nothing():
    fig, ax = plt.subplots()
    assert draw_errorbars(ax, [], [], yerr=[]) == []
    plt.close(fig)

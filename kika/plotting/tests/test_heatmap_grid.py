"""heatmap_grid: several covariance heatmaps in one figure, on one scale.

The data are built by hand: what is under test is the layout and the shared
normalisation, not any file's numbers.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from kika.plotting import HeatmapBuilder, heatmap_grid, shared_norm  # noqa: E402
from kika.plotting.plot_data import CovarianceHeatmapData  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data(n, scale=1.0, matrix_type="corr"):
    rng = np.random.default_rng(n)
    a = rng.normal(size=(n, n))
    cov = a @ a.T * scale
    if matrix_type == "corr":
        s = np.sqrt(np.diag(cov))
        cov = cov / np.outer(s, s)
    return CovarianceHeatmapData(
        matrix_data=cov, matrix_type=matrix_type, zaid=92235,
        energy_grid=np.logspace(0, 7, n + 1), scale="log",
        uncertainty_data={18: np.full(n, 3.0)}, mt_labels=["MT18"],
    )


def _builder(data):
    builder = HeatmapBuilder(figsize=(4, 4), dpi=50)
    lim = (float(data.energy_grid[0]), float(data.energy_grid[-1]))
    builder.set_limits(x_lim=lim, y_lim=lim)
    return builder.add_heatmap(data, show_block_labels=False)


def test_panels_with_different_orders_share_one_figure():
    """640 and 64 groups, as ENDF/B-VIII.1 and JEFF-4.0 give U-235's PFNS."""
    fig = heatmap_grid([_builder(_data(40)), _builder(_data(8))], title="t")
    assert len(fig.subfigs) == 2
    assert fig.get_size_inches()[0] == pytest.approx(8.0)


def test_every_colorbar_stays_inside_its_cell():
    """A whole figure lets its colorbar hang past the edge; a cell cannot,
    because the next cell is drawn over it."""
    fig = heatmap_grid([_builder(_data(10)) for _ in range(3)])
    fig.canvas.draw()
    for cell in fig.subfigs:
        cell_box = cell.bbox
        for ax in cell.axes:
            box = ax.get_window_extent()
            assert box.x1 <= cell_box.x1 + 1.0
            assert box.x0 >= cell_box.x0 - 1.0


def test_four_panels_default_to_two_by_two():
    fig = heatmap_grid([_builder(_data(5)) for _ in range(4)])
    assert len(fig.subfigs) == 4
    assert tuple(fig.get_size_inches()) == pytest.approx((8.0, 9.6))   # 2 x 2, tall cells


def test_shared_norm_keeps_correlations_symmetric():
    norm = shared_norm([_data(6), _data(9)])
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vmin == -norm.vmax and norm.vcenter == 0.0


def test_shared_norm_covers_every_covariance_range():
    small, large = _data(6, 1.0, "cov"), _data(6, 50.0, "cov")
    norm = shared_norm([small, large])
    assert norm.vmax == pytest.approx(np.nanmax(large.matrix_data))
    assert norm.vmin <= np.nanmin(small.matrix_data)


def test_shared_norm_refuses_to_mix_correlation_and_covariance():
    assert shared_norm([_data(6, matrix_type="corr"), _data(6, matrix_type="cov")]) is None


def test_the_grid_puts_every_panel_on_the_shared_scale():
    builders = [_builder(_data(6, 1.0, "cov")), _builder(_data(6, 50.0, "cov"))]
    heatmap_grid(builders)
    limits = {(b._heatmap_styling_overrides["norm"].vmin,
               b._heatmap_styling_overrides["norm"].vmax) for b in builders}
    assert len(limits) == 1


def test_a_builder_without_a_heatmap_is_named():
    with pytest.raises(ValueError, match=r"builders \[1\]"):
        heatmap_grid([_builder(_data(4)), HeatmapBuilder()])

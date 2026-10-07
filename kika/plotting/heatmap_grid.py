"""Several covariance heatmaps in one figure, on one colour scale.

Comparing a covariance between libraries means reading the same structure on
matrices that do not share a grid: ENDF/B-VIII.1 gives U-235's PFNS on 640
outgoing groups where JEFF-4.0 gives 64, and their MF31 grids differ as well.
They cannot be overlaid, so they are put side by side, and the one thing that
must be shared for the comparison to be read at a glance is the colour scale:
two panels each normalised to their own extremes make a weak correlation and a
strong one look alike.

Each panel is a :class:`HeatmapBuilder` drawn into a matplotlib
:class:`~matplotlib.figure.SubFigure`, so it keeps everything a single heatmap
has (uncertainty panel, energy ticks, its own colorbar) without a second
implementation of the layout.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from .styles import _isolated_rc
from .heatmap_builder import HeatmapBuilder

__all__ = ["heatmap_grid", "shared_norm"]


def _limits(norm) -> Optional[Tuple[float, float]]:
    vmin, vmax = getattr(norm, "vmin", None), getattr(norm, "vmax", None)
    if vmin is None or vmax is None or not np.isfinite([vmin, vmax]).all():
        return None
    return float(vmin), float(vmax)


def shared_norm(datas: Sequence) -> Optional[Normalize]:
    """One normalisation covering every panel's own range.

    Correlations stay symmetric about zero, as a single correlation heatmap is;
    covariances take the union of the ranges, centred on zero when it spans it.
    ``None`` when the panels mix correlations and covariances, which no single
    scale describes.
    """
    kinds = {getattr(d, "matrix_type", None) for d in datas}
    if len(kinds) != 1:
        return None
    ranges = [r for r in (_limits(getattr(d, "norm", None)) for d in datas) if r]
    if not ranges:
        return None
    vmin = min(r[0] for r in ranges)
    vmax = max(r[1] for r in ranges)
    if kinds == {"corr"}:
        absmax = max(abs(vmin), abs(vmax)) or 1.0
        return TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def heatmap_grid(
    builders: Sequence[HeatmapBuilder],
    *,
    ncols: Optional[int] = None,
    shared_scale: bool = True,
    title: Optional[str] = None,
    title_fontsize: Optional[float] = None,
) -> plt.Figure:
    """Lay out ``builders`` (each with ``add_heatmap`` already called) in a grid.

    Parameters
    ----------
    builders : sequence of HeatmapBuilder
        One per panel, in reading order. Each keeps its own figure size, which
        becomes the size of its cell; the first one's size and dpi are used for
        every cell so the grid stays regular.
    ncols : int, optional
        Columns. Default: all panels on one row up to three, then two columns
        for four, three beyond.
    shared_scale : bool
        Override each panel's normalisation with one covering them all
        (:func:`shared_norm`). A panel given an explicit ``norm=`` keeps it.
    title : str, optional
        A title over the whole grid.
    title_fontsize : float, optional
        Its size; default ``"x-large"``. Each panel's own title follows its
        builder (``set_font_sizes(title=...)``).
    """
    builders = list(builders)
    if not builders:
        raise ValueError("heatmap_grid needs at least one builder")
    missing = [i for i, b in enumerate(builders) if getattr(b, "_heatmap_data", None) is None]
    if missing:
        raise ValueError(f"builders {missing} have no heatmap; call add_heatmap first")

    n = len(builders)
    if ncols is None:
        ncols = n if n <= 3 else (2 if n == 4 else 3)
    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)

    if shared_scale:
        norm = shared_norm([b._heatmap_data for b in builders])
        if norm is not None:
            for builder in builders:
                overrides = builder._heatmap_styling_overrides
                if overrides.get("norm") is None:
                    # A fresh instance per panel: a Normalize autoscales in
                    # place, and a shared one would let panels reach each other.
                    overrides["norm"] = type(norm)(**_norm_kwargs(norm))

    first = builders[0]
    width, height = getattr(first, "_figsize_user", None) or first.figsize
    dpi = getattr(first, "_dpi_user", None) or first.dpi
    tall = any(_shows_uncertainties(b) for b in builders)
    cell = (width, height * (1.2 if tall else 1.0))

    with _isolated_rc(first._heatmap_rc()):
        fig = plt.figure(figsize=(cell[0] * ncols, cell[1] * nrows), dpi=dpi)
        cells = fig.subfigures(nrows, ncols, squeeze=False)
    for builder, target in zip(builders, cells.flat):
        builder.build(target=target)
    if title:
        # Above the cells, not inside them: each cell already has its own
        # title at the top, and ``bbox_inches="tight"`` keeps what lies outside.
        # A panel title sits at 1.05 of its cell, so the grid's clears it by
        # a little more than that, in figure coordinates.
        fig.suptitle(title, y=1.0 + 0.1 / nrows, va="bottom",
                     fontsize=title_fontsize if title_fontsize is not None else "x-large")
    return fig


def _norm_kwargs(norm) -> dict:
    if isinstance(norm, TwoSlopeNorm):
        return dict(vmin=norm.vmin, vcenter=norm.vcenter, vmax=norm.vmax)
    return dict(vmin=norm.vmin, vmax=norm.vmax)


def _shows_uncertainties(builder: HeatmapBuilder) -> bool:
    data = builder._heatmap_data
    return bool(builder._heatmap_show_uncertainties
                and getattr(data, "uncertainty_data", None))

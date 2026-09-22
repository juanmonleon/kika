"""Plots of a SINBAD benchmark: the measured profiles, C/E, and the correlations.

Three figures cover what an entry is usually opened for. Each takes the
benchmark (or a piece of it), returns the :class:`matplotlib.axes.Axes` it drew
on, and draws on one the caller passes if there is one -- so they compose into
a bigger figure instead of owning it.

matplotlib is imported inside the functions, like everywhere else in kika:
reading a file must not need a display stack.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["plot_profile", "plot_ce", "plot_correlation"]


def _axes(ax, **kwargs):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is not None:
        return ax
    _, ax = plt.subplots(**kwargs)
    return ax


def plot_profile(
    benchmark,
    convention: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    **kwargs,
):
    """
    Measured values against shield thickness, one series per data object.

    Parameters
    ----------
    benchmark : SinbadBenchmark
    convention : str, optional
        Bring every table to this ``valueConvention`` first -- without it,
        tables stored in different conventions are plotted as stored, which is
        rarely what is wanted.
    labels : sequence of str, optional
        Which data objects to draw. Default: every measured table.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to :meth:`~matplotlib.axes.Axes.errorbar`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> from kika.sinbad import plot_profile              # doctest: +SKIP
    >>> plot_profile(b, convention="backgroundSubtracted")   # doctest: +SKIP
    """
    ax = _axes(ax, figsize=(9, 6))
    objects = (
        [benchmark.data[label] for label in labels] if labels
        else [o for o in benchmark.measurements if o.kind == "table"]
    )
    style = {"fmt": "o", "capsize": 3, "markersize": 5}
    style.update(kwargs)
    for obj in objects:
        table = obj.table
        if "shieldThickness" not in table:
            continue
        x = table["shieldThickness"]
        y = obj.corrected(convention) if convention else obj.values
        relative = obj.uncertainty
        yerr = y * relative if relative is not None else None
        ax.errorbar(x, y, yerr=yerr, label=obj.reaction or obj.label, **style)
    ax.set_yscale("log")
    ax.set_xlabel("Shield thickness [cm]")
    unit = ""
    if objects:
        values = objects[0].table.value_columns
        unit = values[0].unit if values else ""
    ax.set_ylabel(f"Measured value [{unit}]" if unit else "Measured value")
    normalisation = objects[0].normalisation if objects else None
    title = benchmark.short_code or benchmark.id
    if normalisation is not None:
        title += f"  --  per {normalisation.basis}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_ce(benchmark, reaction: Optional[str] = None, ax=None, **kwargs):
    """
    Published C/E against shield thickness, one line per library and code.

    Parameters
    ----------
    benchmark : SinbadBenchmark
    reaction : str, optional
        Draw only this reaction, e.g. ``"S32(n,p)P32"``. Without it every
        comparison is drawn, which is readable only for a small entry.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to :meth:`~matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, figsize=(9, 6))
    table = benchmark.ce()
    if table.empty:
        raise ValueError(f"{benchmark.short_code or benchmark.id} has no comparisons")
    if reaction is not None:
        table = table[table["reaction"] == reaction]
    style = {"marker": "o", "linestyle": ":", "markersize": 5}
    style.update(kwargs)
    for (source, series), group in table.groupby(["calculations", "series"], sort=False):
        group = group.sort_values("shieldThickness")
        ax.plot(group["shieldThickness"], group["value"], label=f"{source} {series}", **style)
    ax.axhline(1.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Shield thickness [cm]")
    ax.set_ylabel("C/E")
    ax.set_title(f"{benchmark.short_code or benchmark.id}" + (f"  --  {reaction}" if reaction else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_correlation(benchmark, labels: Optional[Sequence[str]] = None, ax=None, **kwargs):
    """
    The correlation between the measured points, built from the uncertainty budgets.

    The block structure is the point of the figure: the diagonal blocks are the
    within-detector components, the background is what ``within-entry``
    correlates -- in the pilot, the 8 % on the fission-plate power that every
    point carries.

    Parameters
    ----------
    benchmark : SinbadBenchmark
    labels : sequence of str, optional
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to :meth:`~matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ax = _axes(ax, figsize=(7, 6))
    matrix, index = benchmark.correlation(labels)
    style = {"vmin": 0.0, "vmax": 1.0, "cmap": "viridis", "origin": "upper"}
    style.update(kwargs)
    image = ax.imshow(matrix, **style)
    boundaries, last = [], None
    for position, (label, _) in enumerate(index):
        if label != last:
            boundaries.append((position, label))
            last = label
    for position, _ in boundaries[1:]:
        ax.axhline(position - 0.5, color="white", linewidth=0.8)
        ax.axvline(position - 0.5, color="white", linewidth=0.8)
    ticks = [p for p, _ in boundaries]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    names = [label for _, label in boundaries]
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"{benchmark.short_code or benchmark.id} -- correlation of the measured points")
    plt.colorbar(image, ax=ax, label="correlation")
    return ax

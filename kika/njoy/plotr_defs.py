"""PLOTR data model — dataclasses for tape-driven plot definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .viewr_defs import (
    Orientation,
    FontStyle,
    PlotType,
    GridStyle,
    LegendStyle,
    CurveColor,
    PageColor,
    SymbolType,
    DashType,
    PageSetup,
    AxisRange,
    CurveStyle,
    DataPoint2D,
)


@dataclass
class EndfSource:
    """Card 8 of PLOTR: the ENDF, PENDF or GENDF tape data one curve is drawn from.

    When ``iverf`` is 0 the curve is drawn from the points of ``PlotrCurve.data_2d``
    instead of from a tape.

    Attributes
    ----------
    iverf
        NJOY's ENDF-version field, written unchanged. kika tests only whether it
        is 0, which selects the user data of ``PlotrCurve.data_2d`` over the tape.
        In NJOY, 1 is ENDF-6 format.
    nin
        Input tape unit; in NJOY a negative unit is a binary tape.
    matd, mfd, mtd
        MAT, MF and MT of the data to plot.
    temper
        NJOY's ``temper`` field, written unchanged. In NJOY it is the temperature
        in K of Doppler-broadened data, and 0 takes the first one on the tape.
    nth, ntp, nkh
        NJOY's ``nth``, ``ntp`` and ``nkh`` fields, written unchanged. In NJOY 0
        is the default of each.
    """

    iverf: int = 1
    """ENDF version: 0=user data, 1=ENDF-6 format (default)."""
    nin: int = 0
    """Input tape unit number (negative = binary)."""
    matd: int = 0
    """MAT number."""
    mfd: int = 0
    """MF (file) number."""
    mtd: int = 0
    """MT (reaction) number."""
    temper: float = 0.0
    """Temperature for Doppler-broadened data (default 0.0 = first temp)."""
    nth: int = 0
    """Thermal index (0 = default)."""
    ntp: int = 0
    """Sub-particle index (0 = default)."""
    nkh: int = 0
    """Hollerith index (0 = default)."""


@dataclass
class PlotrCurve:
    """One curve of a ``PlotrPlot``: a tape source, a style and a legend.

    Attributes
    ----------
    source
        The tape data to draw (card 8).
    style
        How the curve is drawn (card 9, 2D plots only).
    legend
        Legend text (card 10), written when the plot's legend style is not
        ``LegendStyle.NONE``.
    data_2d
        User-supplied points, written when ``source.iverf`` is 0 and ``nform`` is 0.
    nform
        Data format of the user data (card 12): 0 is ``x, y`` with optional errors.
    """

    source: EndfSource = field(default_factory=EndfSource)
    """Card 8 — tape source for this curve."""
    style: CurveStyle = field(default_factory=CurveStyle)
    """Card 9 — visual style (2D only)."""
    legend: str = ""
    """Legend label (Card 10)."""
    data_2d: list[DataPoint2D] = field(default_factory=list)
    """User-supplied 2D data points (used when source.iverf == 0)."""
    nform: int = 0
    """Data format for user data: 0 = x,y[,errors] (Card 13)."""


@dataclass
class PlotrPlot:
    """One plot of a ``PlotrJob``: a set of axes with its curves.

    The fields are those of ``kika.njoy.Plot`` plus the legend-box position
    (``xtag``, ``ytag``) and a window rotation (``wr``). The first curve carries
    the window and axis cards; each further curve is drawn on the same axes. A
    plot with no curves writes nothing.

    Attributes
    ----------
    title1, title2
        The two title lines.
    plot_type
        Linear or logarithmic axes.
    alt_axis_type
        The ``jtype`` field of card 4: 0 for no alternate axis, 1 for a linear
        and 2 for a logarithmic alternate y axis. A non-zero value makes the
        module write the ``z_range`` and ``z_label`` cards.
    grid
        Frame decoration.
    legend_style
        How the curves are labelled. With ``TAG_LABELS`` every curve gets a tag
        position of 0 0 0.
    xtag, ytag
        Position of the legend box, written only with ``LegendStyle.LEGEND_BOX``
        and when either is not 0.
    x_range, y_range
        Limits and step of the x and y axes.
    x_label, y_label
        Axis labels.
    z_range
        Limits of the alternate y axis; None writes an all-zero range.
    z_label
        Label of that axis.
    new_page
        True starts a new page (``iplot`` 1); False places the plot as a subplot of
        the current page (``iplot`` -1).
    window_color
        Background colour of the plot window.
    factx, facty, xll, yll, ww, wh
        The window fields of card 2, as NJOY reads them.
    wr
        Window rotation, in degrees.
    curves
        The curves, drawn on the same axes in order.
    """

    title1: str = ""
    title2: str = ""
    plot_type: PlotType = PlotType.LIN_LIN
    alt_axis_type: int = 0
    """jtype: 0=none, 1=lin alt-y, 2=log alt-y."""
    grid: GridStyle = GridStyle.TIC_MARKS_INSIDE
    legend_style: LegendStyle = LegendStyle.NONE
    xtag: float = 0.0
    """Card 4 legend box x-position (used when legend_style == LEGEND_BOX)."""
    ytag: float = 0.0
    """Card 4 legend box y-position (used when legend_style == LEGEND_BOX)."""

    x_range: AxisRange = field(default_factory=AxisRange)
    x_label: str = ""
    y_range: AxisRange = field(default_factory=AxisRange)
    y_label: str = ""
    z_range: AxisRange | None = None
    z_label: str = ""

    # Window / subplot positioning
    new_page: bool = True
    """True -> iplot=1 (new page); False -> iplot=-1 (subplot)."""
    window_color: PageColor = PageColor.WHITE
    factx: float = 1.0
    facty: float = 1.0
    xll: float = 0.0
    yll: float = 0.0
    ww: float = 0.0
    wh: float = 0.0
    wr: float = 0.0
    """Window rotation angle in degrees (PLOTR-specific)."""

    curves: list[PlotrCurve] = field(default_factory=list)


@dataclass
class PlotrJob:
    """The plot commands of a PLOTR step, built by hand.

    Pass it to ``InputDeck.plotr(plot_def=...)`` to write the commands in the deck
    after the module card. The commands end with the ``99 /`` card.

    Attributes
    ----------
    nplt, nplt0
        Not read: ``InputDeck.plotr`` writes its own ``nplt`` and ``nplt0``
        arguments on the module card.
    page
        Page settings (card 1).
    plots
        The plots, in the order they are written.
    """

    nplt: int = 0
    """Output plot-command tape unit (filled by deck builder)."""
    nplt0: int = 0
    """Input plot-command tape unit (0 = none)."""
    page: PageSetup = field(default_factory=PageSetup)
    plots: list[PlotrPlot] = field(default_factory=list)

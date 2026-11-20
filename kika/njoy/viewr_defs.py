"""VIEWR data model — dataclasses and enums for custom plot definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ── Enums (IntEnum so they serialize as NJOY integer values) ─────────


class Orientation(IntEnum):
    """Page orientation, the first field of VIEWR and PLOTR card 1.

    The integer value is what NJOY reads: ``PORTRAIT`` is 0 and ``LANDSCAPE`` is 1.
    """

    PORTRAIT = 0
    LANDSCAPE = 1


class FontStyle(IntEnum):
    """Font of the plot text, the second field of VIEWR and PLOTR card 1.

    The integer value is what NJOY reads: ``ROMAN`` is 1 and ``SWISS`` is 2.
    """

    ROMAN = 1
    SWISS = 2


class PlotType(IntEnum):
    """Linear or logarithmic axes of a plot, the ``itype`` field of card 4.

    The first word is the x axis and the second the y axis (``LIN_LOG`` is a linear
    x axis and a logarithmic y axis). The values are the positive ``itype`` codes
    of NJOY, which are the 2D plots: the module writes a curve-style card only
    when the value is positive.
    """

    LIN_LIN = 1
    LIN_LOG = 2
    LOG_LIN = 3
    LOG_LOG = 4


class GridStyle(IntEnum):
    """Frame decoration of a plot, the ``igrid`` field of card 4.

    The integer value is what NJOY reads: no decoration (0), grid lines (1), tic
    marks outside the frame (2) or inside it (3).
    """

    NONE = 0
    GRID_LINES = 1
    TIC_MARKS_OUTSIDE = 2
    TIC_MARKS_INSIDE = 3


class LegendStyle(IntEnum):
    """How the curves of a plot are labelled, the ``ileg`` field of card 4.

    ``NONE`` writes no legend text. ``LEGEND_BOX`` and ``TAG_LABELS`` make the
    module write the legend text of every curve; ``TAG_LABELS`` also writes a tag
    position for it.
    """

    NONE = 0
    LEGEND_BOX = 1
    TAG_LABELS = 2


class CurveColor(IntEnum):
    """Colour of a curve, the ``iccol`` field of the curve-style card (card 9).

    The integer value is what NJOY reads, from ``BLACK`` (0) to ``ORANGE`` (8).
    """

    BLACK = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    MAGENTA = 4
    CYAN = 5
    BROWN = 6
    PURPLE = 7
    ORANGE = 8


class PageColor(IntEnum):
    """Background colour of the page (card 1) or of a plot window (``iwcol``, card 2).

    The integer value is what NJOY reads: ``WHITE`` is 0 and the seven pale
    colours are 1 to 7.
    """

    WHITE = 0
    VERY_PALE_GREEN = 1
    VERY_PALE_CYAN = 2
    VERY_PALE_YELLOW = 3
    VERY_PALE_MAGENTA = 4
    VERY_PALE_SALMON = 5
    VERY_PALE_GREY = 6
    VERY_PALE_BLUE = 7


class SymbolType(IntEnum):
    """Marker drawn at the data points of a curve, the ``isym`` field of card 9.

    The integer value is what NJOY reads, from ``SQUARE`` (0) to ``EXED_DIAMOND``
    (24). By NJOY's convention for ``CurveStyle.icon``, a curve with ``icon`` 0 is
    a line with no marker.
    """

    SQUARE = 0
    OCTAGON = 1
    TRIANGLE_UP = 2
    TRIANGLE_DOWN = 3
    DIAMOND = 4
    CROSS = 5
    X_MARK = 6
    STAR1 = 7
    STAR2 = 8
    FILLED_SQUARE = 9
    FILLED_OCTAGON = 10
    FILLED_TRIANGLE_UP = 11
    FILLED_TRIANGLE_DOWN = 12
    FILLED_DIAMOND = 13
    PLUS_SQUARE = 14
    PLUS_OCTAGON = 15
    PLUS_TRIANGLE_UP = 16
    PLUS_TRIANGLE_DOWN = 17
    PLUS_DIAMOND = 18
    HOURGLASS = 19
    FILLED_HOURGLASS = 20
    BOWTIE = 21
    FILLED_BOWTIE = 22
    EXED_SQUARE = 23
    EXED_DIAMOND = 24


class DashType(IntEnum):
    """Line pattern of a curve, the ``idash`` field of card 9.

    The integer value is what NJOY reads, from ``SOLID`` (0) to ``INVISIBLE`` (5).
    """

    SOLID = 0
    DASHED = 1
    CHAIN_DASH = 2
    CHAIN_DOT = 3
    DOT = 4
    INVISIBLE = 5


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class PageSetup:
    """Card 1 of VIEWR and PLOTR: the page settings, written once per job.

    Held by ``ViewrPlot.page`` and ``PlotrJob.page``.

    Attributes
    ----------
    orientation
        Page orientation.
    font_style
        Font of the plot text.
    size
        The size field of the card, as NJOY reads it (1.0 by default).
    page_color
        Background colour of the page.
    """

    orientation: Orientation = Orientation.PORTRAIT
    font_style: FontStyle = FontStyle.SWISS
    size: float = 1.0
    page_color: PageColor = PageColor.WHITE


@dataclass
class AxisRange:
    """Limits and step of one axis, as written on cards 5, 6 and 7.

    Attributes
    ----------
    min, max
        Lower and upper limit of the axis, in the units of the plotted quantity.
    step
        Axis step, as NJOY reads it, in the same units.
    """

    min: float = 0.0
    max: float = 0.0
    step: float = 0.0


@dataclass
class CurveStyle:
    """Card 9 of VIEWR and PLOTR: how one curve is drawn.

    The module writes this card only for the 2D plot types (a positive
    ``PlotType``).

    Attributes
    ----------
    icon
        0 draws a line with no marker; a positive value a marker and a line; a
        negative value markers only.
    isym
        Marker.
    idash
        Line pattern.
    iccol
        Curve colour.
    ithick
        Line thickness, as NJOY reads it.
    ishade
        The shading field, written unchanged and never read by kika. In NJOY, 0
        is open, 1 to 7 are cross-hatch patterns and 100 or more is a colour
        fill.
    """

    icon: int = 0
    """0=line only, positive=symbol+line, negative=symbol only."""
    isym: SymbolType = SymbolType.SQUARE
    idash: DashType = DashType.SOLID
    iccol: CurveColor = CurveColor.BLACK
    ithick: int = 1
    ishade: int = 0
    """0=open, 1-7 = cross-hatch patterns, 100+ = color fill."""


@dataclass
class DataPoint2D:
    """One point of a user-supplied 2D curve, a line of card 13.

    The point is written as ``x y`` when the four error fields are all 0, and
    with the four errors after them otherwise.

    Attributes
    ----------
    x, y
        Coordinates, in the units of the plotted quantity.
    yerr_upper, yerr_lower, xerr_upper, xerr_lower
        Error-bar extents on each side of the point, in the units of the axis.
        0 is no error bar.
    """

    x: float = 0.0
    y: float = 0.0
    yerr_upper: float = 0.0
    yerr_lower: float = 0.0
    xerr_upper: float = 0.0
    xerr_lower: float = 0.0


@dataclass
class View3D:
    """Card 11 of VIEWR: the viewpoint of a 3D plot.

    Written only for a curve of a plot whose ``PlotType`` value is negative.

    Attributes
    ----------
    xv, yv, zv
        The first three fields of the card (viewpoint), as NJOY reads them.
    x3, y3, z3
        The last three fields of the card, as NJOY reads them.
    """

    xv: float = 15.0
    yv: float = -15.0
    zv: float = 15.0
    x3: float = 2.5
    y3: float = 6.5
    z3: float = 2.5


@dataclass
class Curve:
    """One curve of a ``Plot``, with its style, legend and data.

    Attributes
    ----------
    style
        How the curve is drawn (card 9).
    legend
        Legend text (card 10), written when the plot's legend style is not
        ``LegendStyle.NONE``.
    xtag, ytag
        Tag position (card 10a), written only with ``LegendStyle.TAG_LABELS``.
    view3d
        Viewpoint (card 11), written only for a 3D plot and when it is not None.
    nform
        Data format (card 12): 0 for the free-format 2D points of ``data_2d``,
        1 for the 3D families of ``data_3d``. Any other value writes no data.
    data_2d
        The points, used when ``nform`` is 0.
    data_3d
        Each element is ``(x_value, [(y1, z1), (y2, z2), ...])``; used when
        ``nform`` is 1.
    """

    style: CurveStyle = field(default_factory=CurveStyle)
    legend: str = ""
    """Legend label (Card 10) for LEGEND_BOX or TAG_LABELS mode."""
    xtag: float = 0.0
    """Tag x-position (Card 10a) — used only with TAG_LABELS."""
    ytag: float = 0.0
    """Tag y-position (Card 10a) — used only with TAG_LABELS."""
    view3d: View3D | None = None
    """3D viewpoint — set only for 3D plots (negative PlotType)."""
    nform: int = 0
    """Data format: 0 = free-format 2D (Card 13), 1 = 3D families (Card 14)."""
    data_2d: list[DataPoint2D] = field(default_factory=list)
    """2D data points (used when nform=0)."""
    data_3d: list[tuple[float, list[tuple[float, float]]]] = field(
        default_factory=list
    )
    """3D family data (used when nform=1).

    Each element is ``(x_value, [(y1, z1), (y2, z2), ...])``.
    A family is terminated by an empty list.
    """


@dataclass
class Plot:
    """One plot of a ``ViewrPlot``: a set of axes with its curves.

    The first curve carries the window and axis cards (2 to 7); each further
    curve is written on the same axes. A plot with no curves writes nothing.

    Attributes
    ----------
    title1, title2
        The two title lines (cards 3 and 3a).
    plot_type
        Linear or logarithmic axes.
    alt_axis_type
        The ``jtype`` field of card 4, written unchanged. kika tests only whether
        it is 0: a non-zero value makes the module write the ``z_range`` and
        ``z_label`` cards. In NJOY, 0 is no alternate axis, 1 a linear and 2 a
        logarithmic alternate y axis, and a negative value is the z axis of a 3D
        plot.
    grid
        Frame decoration.
    legend_style
        How the curves are labelled.
    x_range, y_range
        Limits and step of the x and y axes.
    x_label, y_label
        Axis labels.
    z_range
        Limits of the alternate y axis or of the z axis; None writes an all-zero
        range.
    z_label
        Label of that axis.
    new_page
        True starts a new page (``iplot`` 1); False places the plot as a subplot of
        the current page (``iplot`` -1).
    window_color
        Background colour of the plot window.
    factx, facty, xll, yll, ww, wh
        The window fields of card 2, as NJOY reads them.
    curves
        The curves, drawn on the same axes in order.
    """

    title1: str = ""
    title2: str = ""
    plot_type: PlotType = PlotType.LIN_LIN
    alt_axis_type: int = 0
    """jtype: 0=none, 1=lin alt-y, 2=log alt-y.  Negative for 3D z-axis."""
    grid: GridStyle = GridStyle.TIC_MARKS_INSIDE
    legend_style: LegendStyle = LegendStyle.NONE

    x_range: AxisRange = field(default_factory=AxisRange)
    x_label: str = ""
    y_range: AxisRange = field(default_factory=AxisRange)
    y_label: str = ""
    z_range: AxisRange | None = None
    """Set for alt-y or 3D z-axis (Card 7/7a). None = omit."""
    z_label: str = ""

    # Window / subplot positioning
    new_page: bool = True
    """True → iplot=1 (new page); False → iplot=-1 (subplot on same page)."""
    window_color: PageColor = PageColor.WHITE
    factx: float = 1.0
    facty: float = 1.0
    xll: float = 0.0
    yll: float = 0.0
    ww: float = 0.0
    wh: float = 0.0

    curves: list[Curve] = field(default_factory=list)


@dataclass
class ViewrPlot:
    """The plot commands of a VIEWR step, built by hand.

    Pass it to ``InputDeck.viewr(plot_def=...)`` to write the commands in the deck
    after the module card; the deck then reads them from standard input. The
    commands end with the ``99 /`` card.

    Attributes
    ----------
    page
        Page settings (card 1).
    plots
        The plots, in the order they are written.
    """

    page: PageSetup = field(default_factory=PageSetup)
    plots: list[Plot] = field(default_factory=list)

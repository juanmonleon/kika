"""
Plotting styles for kika.

A :class:`Style` is everything that makes a figure recognisable: the categorical
palette (which colour the n-th series gets), a sequential and a diverging colour map,
and the matplotlib rcParams for fonts, spines, grid, ticks and legend.

Styles live in a registry. ``list_styles()`` gives them in display order and
``get_style(name)`` resolves a name. ``'light'`` and ``'dark'``, the only two names
kika accepted before the registry existed, are aliases of ``'classic'`` and
``'classic-dark'``, which reproduce the old look exactly.

Applying a style never touches global matplotlib state. :meth:`Style.context` is a
context manager that starts from matplotlib's defaults, applies the style, and
restores the caller's rcParams on exit; ``PlotBuilder`` renders inside it.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _setup_notebook_backend,
    _configure_figure_interactivity
)


ColormapSpec = Union[str, Tuple[str, ...]]


@dataclass(frozen=True)
class Style:
    """
    A named visual style for kika figures.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``'signature'``.
    label : str
        Human-readable name for selectors.
    description : str
        One sentence saying what the style is for.
    palette : tuple of str
        Categorical colours, assigned to series in this fixed order. The order is
        part of the design: neighbouring slots were chosen to stay distinguishable
        under protanopia and deuteranopia.
    sequential, diverging : str or tuple of str
        Colour maps for ordered series and for signed quantities (correlation
        matrices). Either a matplotlib colormap name or a list of colour stops.
    rc : mapping
        rcParams that define the look (backgrounds, ink, spines, grid, legend).
        Font sizes and layout are shared by every style (see :meth:`rc_params`).
    font_family : str
        Generic family used when the caller does not choose a font.
    dark : bool
        Whether the style draws on a dark background.
    light_variant : str, optional
        For dark styles: the light style used for figures that are always drawn on
        white (heatmaps).
    band_alpha : float
        Suggested opacity for uncertainty bands.
    neutral : str
        Colour for reference lines and "other" elements.
    """
    name: str
    label: str
    description: str
    palette: Tuple[str, ...]
    sequential: ColormapSpec
    diverging: ColormapSpec
    rc: Mapping[str, Any] = field(default_factory=dict)
    font_family: str = 'sans-serif'
    dark: bool = False
    light_variant: Optional[str] = None
    band_alpha: float = 0.25
    neutral: str = '#6b7280'

    # -- colours ---------------------------------------------------------------

    def color(self, index: int) -> str:
        """Colour of the ``index``-th series (the palette repeats past its end)."""
        return self.palette[index % len(self.palette)]

    def cmap(self, kind: str = 'sequential') -> mpl.colors.Colormap:
        """The style's ``'sequential'`` or ``'diverging'`` colour map."""
        if kind not in ('sequential', 'diverging'):
            raise ValueError(f"kind must be 'sequential' or 'diverging', got {kind!r}")
        spec = self.sequential if kind == 'sequential' else self.diverging
        if isinstance(spec, str):
            return plt.get_cmap(spec)
        return mpl.colors.LinearSegmentedColormap.from_list(f'kika-{self.name}-{kind}', list(spec))

    def cmap_name(self, kind: str = 'sequential') -> str:
        """A name for :meth:`cmap` that matplotlib or kika can resolve."""
        spec = self.sequential if kind == 'sequential' else self.diverging
        return spec if isinstance(spec, str) else f'kika-{self.name}-{kind}'

    def colors_along(self, n: int, kind: str = 'sequential') -> List[str]:
        """``n`` colours sampled evenly from a colour map, as hex strings.

        Useful for ordered series (a distribution at increasing energies), which
        read better on a ramp than with categorical colours.
        """
        cmap = self.cmap(kind)
        positions = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
        return [mpl.colors.to_hex(cmap(p)) for p in positions]

    # -- rcParams --------------------------------------------------------------

    def rc_value(self, key: str, default: Any = None) -> Any:
        """A single rcParam as this style sets it (falls back to matplotlib's default)."""
        if key in self.rc:
            return self.rc[key]
        return mpl.rcParamsDefault.get(key, default)

    def rc_params(
        self,
        notebook_mode: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        font_family: Optional[str] = None,
        projection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        The complete set of rcParams for a figure in this style.

        Parameters
        ----------
        notebook_mode : bool
            Slightly smaller type and lines, for notebooks.
        figsize, dpi : optional
            Written to ``figure.figsize`` / ``figure.dpi`` when given.
        font_family : str, optional
            Overrides the style's font (a generic family such as ``'serif'`` or a
            font name such as ``'Arial'``).
        projection : str, optional
            ``'3d'`` disables constrained layout, which does not work with 3-D axes.
        """
        rc: Dict[str, Any] = {
            'font.size': 11 if notebook_mode else 12,
            'axes.labelsize': 12 if notebook_mode else 14,
            'axes.titlesize': 13 if notebook_mode else 14,
            'xtick.labelsize': 10 if notebook_mode else 12,
            'ytick.labelsize': 10 if notebook_mode else 12,
            'legend.fontsize': 10 if notebook_mode else 12,
            'axes.linewidth': 1.0 if notebook_mode else 1.2,
            'lines.linewidth': 1.0 if notebook_mode else 1.2,
            'lines.markersize': 6 if notebook_mode else 8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'xtick.major.width': 1.0 if notebook_mode else 1.2,
            'ytick.major.width': 1.0 if notebook_mode else 1.2,
            'xtick.minor.width': 0.8 if notebook_mode else 1.0,
            'ytick.minor.width': 0.8 if notebook_mode else 1.0,
            'xtick.major.size': 4.0 if notebook_mode else 5.0,
            'ytick.major.size': 4.0 if notebook_mode else 5.0,
            'xtick.minor.size': 2.5 if notebook_mode else 3.0,
            'ytick.minor.size': 2.5 if notebook_mode else 3.0,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'figure.constrained_layout.use': projection != '3d',
            # Vector output keeps its text as text: TrueType embedding in PDF/PS
            # (matplotlib's default, Type 3, turns every glyph into a drawing).
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
        rc.update(self.rc)
        rc['font.family'] = font_family or self.font_family
        rc['axes.prop_cycle'] = cycler(color=list(self.palette))
        if figsize is not None:
            rc['figure.figsize'] = figsize
        if dpi is not None:
            rc['figure.dpi'] = dpi
        return rc

    @contextmanager
    def context(self, **rc_kwargs) -> Iterator['Style']:
        """
        Render in this style without touching the caller's rcParams.

        Starts from matplotlib's defaults (so a user's own rc settings do not leak
        into kika figures), applies :meth:`rc_params` with ``rc_kwargs``, and
        restores everything on exit.

        >>> with get_style('signature').context():
        ...     fig, ax = plt.subplots()
        ...     ax.plot(x, y)
        """
        with _isolated_rc(self.rc_params(**rc_kwargs)):
            yield self

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable description, for user interfaces.

        Colour maps are given as 11 evenly spaced stops so a client can draw a
        swatch or rebuild the map without matplotlib.
        """
        return {
            'name': self.name,
            'label': self.label,
            'description': self.description,
            'dark': self.dark,
            # Whether the style asks for a grid at all. `journal` does not, and a
            # client that always sends an explicit "show grid" flag overrides that
            # without ever knowing it existed.
            # `rc_value` cannot answer this: the True comes from the base dict in
            # `rc_params`, which a style's own rc then overrides. Read it the same
            # way that merge does.
            'axesGrid': bool(self.rc.get('axes.grid', True)),
            'palette': list(self.palette),
            'sequential': self.colors_along(11, 'sequential'),
            'diverging': self.colors_along(11, 'diverging'),
            'background': _hex(self.rc_value('axes.facecolor')),
            'figureBackground': _hex(self.rc_value('figure.facecolor')),
            'foreground': _hex(self.rc_value('text.color')),
            'muted': _hex(self.rc_value('xtick.color')),
            'grid': _hex(self.rc_value('grid.color')),
            'fontFamily': self.font_family,
            'fonts': list(self.rc_value(f'font.{self.font_family}', []) or []),
            'bandAlpha': self.band_alpha,
            'neutral': self.neutral,
            'lightVariant': self.light_variant,
        }


def _hex(color: Any) -> str:
    try:
        return mpl.colors.to_hex(color)
    except (ValueError, TypeError):
        return str(color)


@contextmanager
def _isolated_rc(rc: Mapping[str, Any]) -> Iterator[None]:
    """matplotlib defaults + ``rc`` for the duration of the block, then the caller's rcParams back.

    ``style.use('default')`` resets everything except the keys matplotlib keeps out of
    styles (backend, interactive mode, ...), which is what kika's style application
    did before, minus the global side effects.
    """
    with mpl.rc_context():
        plt.style.use('default')
        mpl.rcParams.update(rc)
        yield


# -----------------------------------------------------------------------------
# The styles
# -----------------------------------------------------------------------------

_SANS = ['Inter', 'Segoe UI', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
_HUMANIST = ['Gill Sans MT', 'Gill Sans', 'Segoe UI', 'Helvetica Neue', 'DejaVu Sans']
_SERIF_STIX = ['STIX Two Text', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif']

# matplotlib's tab10, spelled out: the old 'dark' style read it from rcParams, which
# made the palette depend on whatever the user had configured.
_TAB10 = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

# Seaborn "colorblind": kika's palette before the registry existed.
CLASSIC_PALETTE = ('#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
                   '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9')


def _open_axes(ink: str, muted: str, grid: str, background: str = 'white',
               figure_background: Optional[str] = None) -> Dict[str, Any]:
    """Left and bottom spines only, a quiet solid grid, a frameless legend.

    ``grid`` is the colour drawn at ``grid.alpha = 0.3`` (the alpha kika and the app
    apply by default), so it is darker than the tint it produces on the page.
    """
    fig_bg = figure_background or background
    return {
        'figure.facecolor': fig_bg, 'savefig.facecolor': fig_bg, 'axes.facecolor': background,
        'text.color': ink, 'axes.labelcolor': ink, 'axes.titlecolor': ink,
        'axes.edgecolor': muted, 'axes.linewidth': 0.9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': muted, 'ytick.color': muted,
        'xtick.labelcolor': ink, 'ytick.labelcolor': ink,
        'axes.axisbelow': True,
        'grid.color': grid, 'grid.linewidth': 0.8, 'grid.linestyle': '-', 'grid.alpha': 0.3,
        'legend.frameon': False,
        'axes.titlelocation': 'left', 'axes.titleweight': 'semibold',
    }


_REGISTRY: Dict[str, Style] = {}
_ALIASES: Dict[str, str] = {
    # The two names PlotBuilder accepted before the registry.
    'light': 'classic',
    'dark': 'classic-dark',
    # Names older helpers documented but never implemented; they always rendered
    # the light style.
    'default': 'classic',
    'paper': 'classic',
    'publication': 'classic',
    'presentation': 'classic',
}


def register_style(style: Style, *, replace: bool = False) -> Style:
    """Add ``style`` to the registry, so every builder and ``get_style`` can use it."""
    key = style.name.lower()
    if not replace and (key in _REGISTRY or key in _ALIASES):
        raise ValueError(f"A style named {style.name!r} already exists")
    _REGISTRY[key] = style
    # Colour maps given as stops are registered with matplotlib under
    # 'kika-<style>-<kind>', so the name works anywhere a cmap name does.
    for kind in ('sequential', 'diverging'):
        if not isinstance(getattr(style, kind), str):
            name = style.cmap_name(kind)
            if name in mpl.colormaps:
                mpl.colormaps.unregister(name)
            mpl.colormaps.register(style.cmap(kind), name=name)
    return style


def get_style(style: Union[str, Style, None] = None) -> Style:
    """
    Resolve a style name (or alias) to a :class:`Style`.

    ``None`` gives the default style, ``'classic'``. A :class:`Style` instance is
    returned unchanged, so functions can accept either.
    """
    if isinstance(style, Style):
        return style
    key = 'classic' if style is None else str(style).lower()
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        names = ', '.join(repr(s.name) for s in list_styles())
        raise ValueError(f"Unknown style {style!r}. Available styles: {names}") from None


def list_styles() -> List[Style]:
    """Every registered style, in the order a selector should show them."""
    return list(_REGISTRY.values())


def style_names() -> List[str]:
    """Names of every registered style (aliases not included)."""
    return [s.name for s in _REGISTRY.values()]


register_style(Style(
    name='classic', label='Classic',
    description="kika's original look: colour-blind palette, serif type, boxed axes, dashed grid.",
    palette=CLASSIC_PALETTE,
    sequential='viridis', diverging='RdYlGn',
    font_family='serif',
    rc={
        'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': 'black',
        'legend.framealpha': 0.9,
    },
    band_alpha=0.3, neutral='#949494',
))

register_style(Style(
    name='signature', label='KIKA',
    description="KIKA's own look: the blue of the brand, open axes, clean sans-serif type.",
    palette=('#036bb9', '#d9480f', '#14a0a0', '#548106', '#6331a0', '#c78b09', '#c6367d'),
    sequential=('#0b2545', '#0a4f8f', '#0f7fb8', '#14a0a0', '#5fb04a', '#a9cf2f'),
    diverging=('#0b3c73', '#036bb9', '#8dbde6', '#f5f4f0', '#f0a27f', '#d9480f', '#8a2a05'),
    font_family='sans-serif',
    rc={'font.sans-serif': _SANS, **_open_axes(ink='#0b1d29', muted='#5b6b78', grid='#afbcca')},
    band_alpha=0.22, neutral='#5b6b78',
))

register_style(Style(
    name='journal', label='Journal',
    description='Physics-journal figure: STIX type to match LaTeX, boxed axes with inward ticks, no grid.',
    palette=('#1a1a1a', '#c1272d', '#0a5fa8', '#1b8a5a', '#c77c11', '#7b3f9e', '#2aa1b3'),
    sequential=('#0d1b2a', '#1b4f72', '#2e86ab', '#6fb3a6', '#c9b458'),
    diverging=('#053061', '#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b', '#67001f'),
    font_family='serif',
    rc={
        'font.serif': _SERIF_STIX, 'mathtext.fontset': 'stix',
        'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.linewidth': 0.8, 'axes.grid': False,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.frameon': False,
    },
    band_alpha=0.2, neutral='#555555',
))

register_style(Style(
    name='isotope', label='Isotope',
    description='Deep jewel tones (indigo, rose, cerulean, ochre, jade) with open axes and humanist type.',
    palette=('#284ba4', '#da6182', '#1f99c7', '#b5771c', '#047252', '#9b51ac', '#ba3535'),
    sequential=('#1b1f4b', '#284ba4', '#1f99c7', '#4fb99f', '#c9d66b'),
    diverging=('#1b2a6b', '#284ba4', '#9db4e3', '#f6f4f1', '#eaa3b5', '#c23a62', '#7a1233'),
    font_family='sans-serif',
    rc={'font.sans-serif': _HUMANIST, **_open_axes(ink='#1c1b29', muted='#6a687a', grid='#bcb6cd')},
    band_alpha=0.22, neutral='#6a687a',
))

register_style(Style(
    name='mineral', label='Mineral',
    description='Muted earth tones on warm paper; calm with many series and in long reports.',
    palette=('#006a9e', '#ac8b26', '#10989f', '#883500', '#cc5e6b', '#763671', '#3a874f'),
    sequential=('#22303c', '#006a9e', '#10989f', '#7fb38a', '#d8c47a'),
    diverging=('#0b4a6f', '#006a9e', '#9cc3d5', '#faf8f3', '#dcae8a', '#a9561d', '#6b2c00'),
    font_family='sans-serif',
    rc={'font.sans-serif': _SANS,
        **_open_axes(ink='#2a2926', muted='#77746c', grid='#baaf95', background='#fdfcf9')},
    band_alpha=0.22, neutral='#77746c',
))

register_style(Style(
    name='classic-dark', label='Classic dark',
    description="kika's original dark style: black background, white ink, matplotlib's default colours.",
    palette=_TAB10,
    sequential='viridis', diverging='RdYlGn',
    font_family='serif',
    rc={
        'axes.facecolor': 'black', 'figure.facecolor': 'black', 'savefig.facecolor': 'black',
        'axes.edgecolor': 'white', 'axes.labelcolor': 'white',
        'xtick.color': 'white', 'ytick.color': 'white', 'text.color': 'white',
        'legend.fancybox': True, 'legend.framealpha': 0.9,
    },
    dark=True, light_variant='classic', band_alpha=0.3, neutral='#7f7f7f',
))

register_style(Style(
    name='signature-dark', label='KIKA dark',
    description='The KIKA look on the navy of the website, for slides and dark themes.',
    palette=('#4abaf4', '#fc7756', '#b1ef4a', '#9468c2', '#b17000', '#6be5de', '#d75f94'),
    sequential=('#1f6fae', '#35a0e0', '#58c8e6', '#6be5de', '#9aeb86', '#d3f76a'),
    diverging=('#9fdcff', '#4abaf4', '#1f6fae', '#0b1d29', '#a8431f', '#fc7756', '#ffc2ad'),
    font_family='sans-serif',
    rc={'font.sans-serif': _SANS,
        **_open_axes(ink='#edf8fd', muted='#9bb1bc', grid='#335c72',
                     background='#0b1d29', figure_background='#07151f')},
    dark=True, light_variant='signature', band_alpha=0.28, neutral='#9bb1bc',
))


# -----------------------------------------------------------------------------
# Internal helpers used by the builders and the legacy plotting functions
# -----------------------------------------------------------------------------

def _get_color_palette(style: Union[str, Style]) -> list[str]:
    """Categorical palette of ``style`` (a name, an alias or a :class:`Style`)."""
    return list(get_style(style).palette)


def _get_linestyles() -> list:
    """
    Get the default linestyle cycle.

    Returns
    -------
    list
        List of linestyles for cycling
    """
    return ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def _apply_style_to_rcparams(
    style: Union[str, Style],
    notebook_mode: bool,
    figsize: Tuple[float, float],
    dpi: int,
    font_family: Optional[str],
    projection: Optional[str] = None
) -> None:
    """
    Apply a style to the **global** matplotlib rcParams.

    Kept for the legacy plotting functions that draw with pyplot after calling it
    (``setup_plot_style``, the deprecated MF4 helpers). The builders do not use it:
    they render inside :meth:`Style.context` and leave global state alone.

    Parameters
    ----------
    style : str or Style
        Style name, alias or instance
    notebook_mode : bool
        Whether running in notebook mode
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Dots per inch for figure resolution
    font_family : str, optional
        Font family for text elements (None: the style's own)
    projection : str, optional
        Matplotlib projection type (e.g., '3d' for 3D plots)
    """
    rc = get_style(style).rc_params(
        notebook_mode=notebook_mode, figsize=figsize, dpi=dpi,
        font_family=font_family, projection=projection,
    )
    # Reset matplotlib settings to avoid style contamination
    plt.close('all')
    mpl.rcdefaults()
    plt.style.use('default')
    plt.rcParams.update(rc)


def _adjust_figsize_for_notebook(
    figsize: Tuple[float, float],
    interactive: bool
) -> Tuple[float, float]:
    """
    Adjust figure size for notebook environments.
    
    Parameters
    ----------
    figsize : tuple
        Original figure size (width, height) in inches
    interactive : bool
        Whether using interactive backend
        
    Returns
    -------
    tuple
        Adjusted figure size
    """
    # More aggressive figure size reduction for notebooks
    if figsize[0] > 10 or figsize[1] > 7:
        scale = min(8/figsize[0], 5/figsize[1])
        return (figsize[0] * scale, figsize[1] * scale)
    elif figsize[0] > 8 or figsize[1] > 6:
        scale = min(7/figsize[0], 5/figsize[1])
        return (figsize[0] * scale, figsize[1] * scale)
    return figsize


def _adjust_dpi_for_notebook(dpi: int, interactive: bool) -> int:
    """
    Adjust DPI for notebook environments.
    
    Parameters
    ----------
    dpi : int
        Original DPI value
    interactive : bool
        Whether using interactive backend
        
    Returns
    -------
    int
        Adjusted DPI value
    """
    if dpi > 120:
        return 90 if interactive else 120
    return dpi


def format_energy_axis_ticks(ax: plt.Axes) -> None:
    """
    Label a log-scale energy axis so the ticks stay readable.

    How many ticks fit depends on how many decades are on the axis, and a nuclear
    data plot routinely spans ten of them. The tick density therefore follows the
    span: decades only when the axis is wide, 1-2-5 within a decade when it is
    narrow. Labels are powers of ten in mathtext ($10^{3}$) rather than ``1e+03``,
    which is both the journal convention and half the width -- the old fixed
    ``subs=(1, 2, 5)`` plus ``%.3g`` produced a row of overlapping ``5e+04``
    labels on any axis narrower than six decades, which is most of them.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes with log-scale energy x-axis
    """
    xmin, xmax = ax.get_xlim()
    if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin <= 0 or xmax <= xmin:
        return
    decades = np.log10(xmax / xmin)

    if decades >= 4:
        subs = (1.0,)
    elif decades >= 2:
        subs = (1.0, 3.0)
    else:
        subs = (1.0, 2.0, 5.0)

    ax.xaxis.set_major_locator(
        mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=12)
    )
    if subs == (1.0,):
        # Decade ticks: 10^3 reads better and is narrower than 1e+03.
        ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation(base=10.0))
    else:
        # Inside a decade, plain numbers are clearer than powers of ten.
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _pos: f"{value:.3g}")
        )
    ax.xaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.1), numticks=100)
    )
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def format_axes(
    ax: plt.Axes,
    style: str = 'light',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    use_log_scale: bool = False,
    is_energy_axis: bool = False,
    use_y_log_scale: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> plt.Axes:
    """
    Format axes with labels, title, legend, scales, and limits.
    
    This is a centralized function to apply consistent formatting to plot axes.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to format
    style : str
        Style name ('light' or 'dark')
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    title : str, optional
        Plot title (suppressed for 'paper' and 'publication' styles)
    legend_loc : str, optional
        Legend location ('best', 'upper right', etc.)
    use_log_scale : bool
        Whether to use logarithmic scale for x-axis
    is_energy_axis : bool
        Whether x-axis is an energy axis (applies special tick formatting)
    use_y_log_scale : bool
        Whether to use logarithmic scale for y-axis
    x_min, x_max : float, optional
        X-axis limits
    y_min, y_max : float, optional
        Y-axis limits
        
    Returns
    -------
    plt.Axes
        Formatted axes object
    """
    # Set axis scales
    if use_log_scale:
        ax.set_xscale('log')
        if is_energy_axis:
            format_energy_axis_ticks(ax)
    
    if use_y_log_scale:
        ax.set_yscale('log')
    
    # Set labels
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    
    # Set title (suppress for paper/publication styles)
    if title and style not in ('paper', 'publication'):
        ax.set_title(title)
    
    # Set limits
    if x_min is not None or x_max is not None:
        current_xlim = ax.get_xlim()
        ax.set_xlim(
            x_min if x_min is not None else current_xlim[0],
            x_max if x_max is not None else current_xlim[1]
        )
    
    if y_min is not None or y_max is not None:
        current_ylim = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_ylim[0],
            y_max if y_max is not None else current_ylim[1]
        )
    
    # Add legend if requested
    if legend_loc:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Style-specific legend formatting
            if style == 'dark':
                ax.legend(
                    loc=legend_loc,
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    facecolor='black',
                    edgecolor='white'
                )
            else:
                ax.legend(
                    loc=legend_loc,
                    frameon=True,
                    fancybox=True,
                    shadow=False
                )
    
    # Grid configuration
    ax.grid(True, alpha=0.3, linestyle='--', which='major')
    if use_log_scale or use_y_log_scale:
        ax.grid(True, alpha=0.15, linestyle=':', which='minor')
    
    return ax


def setup_plot_style(style: str = 'light', figsize: Tuple[float, float] = (8, 6), ax: Optional[plt.Axes] = None):
    """
    Legacy function for backward compatibility with old plotting code.
    
    This function creates a figure and axes with the specified style settings.
    For new code, use PlotBuilder instead.
    
    Parameters
    ----------
    style : str
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple
        Figure size in inches (width, height)
    ax : plt.Axes, optional
        Existing axes to use. If provided, no new figure is created.
        
    Returns
    -------
    dict
        Dictionary with 'fig' and 'ax' keys
    """
    if ax is not None:
        return {'fig': ax.figure, 'ax': ax}
    
    # Apply style using the internal function with default parameters
    _apply_style_to_rcparams(
        style=style,
        notebook_mode=_is_notebook(),
        figsize=figsize,
        dpi=100,
        font_family='sans-serif'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    return {'fig': fig, 'ax': ax}


def finalize_plot(fig: plt.Figure, ax: plt.Axes, show: bool = True):
    """
    Legacy function for backward compatibility with old plotting code.
    
    Finalizes a plot by adjusting layout and optionally displaying it.
    For new code, use PlotBuilder instead.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to finalize
    ax : plt.Axes
        Axes to finalize
    show : bool
        Whether to call plt.show()
    """
    fig.tight_layout()
    if show:
        plt.show()

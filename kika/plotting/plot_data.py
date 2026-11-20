"""
Base classes for representing plottable data.

These classes encapsulate the data and basic styling information for plot elements,
separating data representation from the actual plotting logic.
"""

from typing import Optional, Dict, Any, Union, List, Tuple, Iterator
import numpy as np
from dataclasses import dataclass, field


#: What a curve's numbers are, beyond the file they came from. Used in labels and
#: to warn when curves that should not be compared share an axis.
PROVENANCE_STATES = (
    'evaluated',      # the evaluation as written (ENDF MF3 without resonances, MF4, ...)
    'background',     # ENDF MF3 of a tape whose resonances are in MF2: not sigma(E)
    'reconstructed',  # pointwise from the resonance parameters (PENDF, NJOY RECONR)
    'heated',         # Doppler broadened to a temperature (ACE)
    'projected',      # derived by projection (Legendre moments of a tabulated f(mu))
    'multigroup',     # group-averaged
    'measured',       # experimental (EXFOR)
)


@dataclass(frozen=True)
class Provenance:
    """
    Where a curve comes from and what state its numbers are in.

    Every field is optional. :meth:`describe` turns it into the suffix kika puts
    in legends, so two curves of the same reaction are told apart by what makes
    them different (evaluation, reconstruction, temperature, projection).

    Attributes
    ----------
    format : str, optional
        ``'endf'``, ``'pendf'``, ``'ace'``, ``'gnds'``, ``'g4ndl'``, ``'exfor'``,
        ``'covariance'``.
    evaluation : str, optional
        Library or experiment, e.g. ``'JEFF-4.0'``, ``'ENDF/B-VIII.1'``.
    nuclide : str, optional
        e.g. ``'Fe56'``.
    reaction : int, optional
        ENDF MT number.
    temperature : float, optional
        Kelvin.
    state : str, optional
        One of ``'evaluated'``, ``'background'``, ``'reconstructed'``,
        ``'heated'``, ``'projected'``, ``'multigroup'`` or ``'measured'``.
    frame : str, optional
        ``'CM'`` or ``'LAB'``, for angular quantities.
    source : str, optional
        File the data was read from.
    detail : str, optional
        Anything else a reader needs (``'folded, TOF 27 m / 5 ns'``).
    """
    format: Optional[str] = None
    evaluation: Optional[str] = None
    nuclide: Optional[str] = None
    reaction: Optional[int] = None
    temperature: Optional[float] = None
    state: Optional[str] = None
    frame: Optional[str] = None
    source: Optional[str] = None
    detail: Optional[str] = None

    def describe(self) -> str:
        """The part of a legend entry that says what state the data is in.

        Empty for plain evaluated data, so the common case stays short.
        """
        parts = []
        if self.state == 'background':
            parts.append('MF3 background')
        elif self.state == 'reconstructed':
            parts.append('reconstructed')
        elif self.state == 'projected':
            parts.append('projected')
        elif self.state == 'multigroup':
            parts.append('multigroup')
        if self.format == 'ace':
            parts.append(f'ACE {self.temperature:.1f} K' if self.temperature else 'ACE')
        elif self.format == 'g4ndl':
            parts.append('G4NDL')
        elif self.temperature and self.state == 'heated':
            parts.append(f'{self.temperature:.1f} K')
        if self.detail:
            parts.append(self.detail)
        return ', '.join(parts)


@dataclass
class PlotData:
    """
    Base class for plottable data.
    
    This encapsulates the data to plot along with styling information,
    allowing flexible composition of plots without recreating plotting logic.
    
    Attributes
    ----------
    x, y : array_like
        The abscissae and ordinates, of one length, stored as numpy arrays. Their
        units are ``x_unit`` and ``y_unit``; nothing in the class converts them.
    label : str, optional
        Label for legend
    color : str or tuple, optional
        Line/marker color
    linestyle : str, optional
        Line style ('-', '--', ':', '-.', etc.)
    linewidth : float, optional
        Line width
    marker : str, optional
        Marker style ('o', 's', '^', etc.)
    markersize : float, optional
        Marker size
    alpha : float, optional
        Transparency (0-1)
    plot_type : str
        Type of plot: 'line', 'step', 'scatter', 'errorbar'
    drawstyle : str, optional
        Matplotlib drawstyle for line plots. If set to 'steps-pre', 'steps-post',
        or 'steps-mid', the plot will use step rendering even if plot_type is 'line'.
        This provides an alternative way to specify step plots that integrates with
        frontend plotting libraries (e.g., Plotly's line_shape parameter).
    metadata : dict
        Additional metadata about the data (e.g., isotope, MT, order)
    quantity : str, optional
        Key of :data:`~kika.plotting.QUANTITIES` that says what the curve is.
        Keyword-only. :func:`~kika.plotting.plottable` fills it in; a curve
        without it is drawn as given.
    x_unit, y_unit : str, optional
        Unit of ``x`` and of ``y``, keyword-only. An adapter of
        :func:`~kika.plotting.plottable` may set them; where it sets none,
        ``plottable`` fills in the canonical unit of the quantity (eV, barn,
        b/sr, ...). None means unknown::class:`~kika.plotting.PlotBuilder` then leaves that axis
        unconverted.
    provenance : Provenance, optional
        Where the curve comes from and what state it is in, keyword-only.
        :func:`~kika.plotting.plottable` builds the default legend entry from it,
        and :class:`~kika.plotting.PlotBuilder` warns when curves of different
        frames share an axis.
    interpolation : str, optional
        Law between the points, keyword-only (``'lin-lin'``, ``'log-log'``, ...).
        :class:`~kika.plotting.ComparisonBuilder` reads the reference's one when
        it has no ``interpolation`` of its own.
    """
    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    color: Optional[Union[str, Tuple]] = None
    linestyle: Optional[str] = '-'
    linewidth: Optional[float] = None
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    plot_type: str = 'line'
    drawstyle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # What the numbers mean. Keyword-only and optional, so every existing
    # constructor call (positional ones included) keeps working. Filled in by
    # kika.plotting.plottable(); PlotBuilder uses them for unit conversion, axis
    # labels and mixed-quantity warnings.
    quantity: Optional[str] = field(default=None, kw_only=True)
    x_unit: Optional[str] = field(default=None, kw_only=True)
    y_unit: Optional[str] = field(default=None, kw_only=True)
    provenance: Optional[Provenance] = field(default=None, kw_only=True)
    interpolation: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self):
        """Validate and convert data to numpy arrays."""
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have the same length. Got x: {len(self.x)}, y: {len(self.y)}")
    
    def apply_styling(self, **kwargs) -> 'PlotData':
        """
        Create a copy with updated styling.
        
        Parameters
        ----------
        **kwargs : dict
            Styling attributes to update (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        PlotData
            New PlotData object with updated styling. It is a shallow copy: the
            arrays and ``metadata`` are shared with this object.

        Raises
        ------
        ValueError
            If a keyword is not an attribute of the object.
        """
        # Create a copy of the current object
        import copy
        new_data = copy.copy(self)
        
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(new_data, key):
                setattr(new_data, key, value)
            else:
                raise ValueError(f"Unknown styling attribute: {key}")
        
        return new_data
    
    def get_plot_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for matplotlib plotting functions.
        
        Returns
        -------
        dict
            Dictionary of kwargs suitable for ax.plot(), ax.step(), etc.
        """
        kwargs = {}
        
        if self.label is not None:
            kwargs['label'] = self.label
        if self.color is not None:
            kwargs['color'] = self.color
        if self.linestyle is not None:
            kwargs['linestyle'] = self.linestyle
        if self.linewidth is not None:
            kwargs['linewidth'] = self.linewidth
        if self.marker is not None:
            kwargs['marker'] = self.marker
        if self.markersize is not None:
            kwargs['markersize'] = self.markersize
        if self.alpha is not None:
            kwargs['alpha'] = self.alpha
            
        return kwargs


@dataclass
class LegendreCoeffPlotData(PlotData):
    """
    Plottable data for Legendre coefficients.

    A :class:`PlotData` whose ``x`` is the incident energy and ``y`` the
    coefficient of ``order``. The ``to_plot_data`` methods of the ENDF, ACE and
    covariance classes return it.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    plot_type, metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'<isotope> - L=<order>'``, or
        ``'L=<order>'`` when there is no isotope.
    order : int
        Legendre polynomial order.
    isotope : str, optional
        Isotope identifier.
    mt : int, optional
        MT reaction number.
    energy_range : tuple, optional
        ``(min, max)`` energy range of the data, in the unit of ``x``. Stored
        in ``metadata``; it does not clip the data.
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    energy_range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['energy_range'] = self.energy_range
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class LegendreUncertaintyPlotData(PlotData):
    """
    Plottable data for Legendre coefficient uncertainties.

    A :class:`PlotData` whose ``y`` is the uncertainty of the coefficient of
    ``order`` at each ``x`` (an energy). Passed as the ``uncertainty`` of
    :meth:`PlotBuilder.add_data`, it becomes a band around the nominal curve.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'<isotope> - L=<order>'``, or
        ``'L=<order>'`` when there is no isotope.
    plot_type : str
        As in :class:`PlotData`, except that the default ``'line'`` becomes
        ``'step'``.
    order : int
        Legendre polynomial order.
    isotope : str, optional
        Isotope identifier.
    mt : int, optional
        MT reaction number.
    uncertainty_type : {'relative', 'absolute'}
        ``'relative'``: ``y`` is a percentage (the band divides it by 100).
        ``'absolute'``: ``y`` is the half-width in the units of the nominal
        curve. Any other string is read as ``'absolute'`` by the band.
    sigma : float
        The number of standard deviations ``y`` stands for (1.0 is one sigma).
        Stored for the caller: the band built from this object does not apply
        it again.
    energy_bins : array_like, optional
        Energy bin boundaries, stored as an array. The builders do not read it.
    step_where : {'pre', 'post', 'mid'}
        Where the steps change, as in ``matplotlib.axes.Axes.step``. It is read
        only when the curve is drawn as steps, and a ``drawstyle`` containing
        ``'steps'`` (``'steps-pre'``, ``'steps-mid'``, ``'steps-post'``) overrides
        it.
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    sigma: float = 1.0
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class AngularDistributionPlotData(PlotData):
    """
    Plottable data for angular distributions.

    A :class:`PlotData` whose ``x`` is the cosine of the scattering angle and
    ``y`` the distribution at one incident ``energy``.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    plot_type, metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'E = <energy> MeV'`` with the energy in
        scientific notation and two decimals, whatever the unit of ``energy``.
    energy : float
        Incident energy of this distribution. Its unit is the caller's: the
        default label writes the number followed by MeV.
    isotope : str, optional
        Isotope identifier.
    mt : int, optional
        MT reaction number.
    distribution_type : str, optional
        Kind of distribution: ``'legendre'``, ``'tabulated'`` or ``'mixed'``.
        Stored; the class does not check it.
    """
    energy: float = 0.0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    distribution_type: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['energy'] = self.energy
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['distribution_type'] = self.distribution_type

        # Default label if not provided
        if self.label is None and self.energy is not None:
            self.label = f"E = {self.energy:.2e} MeV"


@dataclass
class CrossSectionPlotData(PlotData):
    """
    Plottable data for cross sections (energy-dependent).

    This class represents cross section data as a function of energy,
    typically from EXFOR experimental data or ENDF evaluations.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    plot_type, metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'<isotope> MT=<mt>'``, or ``'MT=<mt>'``
        without an isotope; it stays None without an ``mt``.
    isotope : str, optional
        Isotope identifier (e.g., 'Fe56')
    mt : int, optional
        MT reaction number (e.g., 1 for total, 2 for elastic)
    energy_range : tuple of float, optional
        ``(min, max)`` energy range of the data, in the unit of ``x`` (the
        ENDF producer sets it to the range of its own ``x``). Stored in
        ``metadata``; it does not clip the data.
    data_source : str, optional
        Source of data: 'exfor', 'endf', 'ace', etc.
    """
    isotope: Optional[str] = None
    mt: Optional[int] = None
    energy_range: Optional[Tuple[float, float]] = None
    data_source: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['energy_range'] = self.energy_range
        self.metadata['data_source'] = self.data_source

        # Default label if not provided
        if self.label is None and self.isotope and self.mt is not None:
            self.label = f"{self.isotope} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class DifferencePlotData(PlotData):
    """
    Plottable data representing a difference or ratio between two datasets.

    :func:`~kika.plotting.compute_difference` returns one as the ``difference``
    of its result, with no units, quantity or provenance set.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    plot_type, metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'(<comparison> − <reference>) / <reference>'``
        for a relative difference and ``'<comparison> − <reference>'`` for an
        absolute one, with ``'comp'`` and ``'ref'`` for a missing label.
    difference_type : {'relative', 'absolute'}
        ``'relative'``: ``(comparison - reference) / reference``, in per cent
        or as a fraction as the producer chose. ``'absolute'``: ``comparison -
        reference``, in the units of the two datasets. Used for the default
        label and stored in ``metadata``.
    reference_label : str, optional
        Label of the reference dataset
    comparison_label : str, optional
        Label of the comparison dataset
    """
    difference_type: str = 'relative'
    reference_label: Optional[str] = None
    comparison_label: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.metadata['difference_type'] = self.difference_type
        self.metadata['reference_label'] = self.reference_label
        self.metadata['comparison_label'] = self.comparison_label

        if self.label is None:
            ref = self.reference_label or 'ref'
            comp = self.comparison_label or 'comp'
            if self.difference_type == 'relative':
                self.label = f"({comp} \u2212 {ref}) / {ref}"
            else:
                self.label = f"{comp} \u2212 {ref}"


class UncertaintyBand:
    """
    Represents an uncertainty band for a plot.

    This can store either absolute bounds or relative uncertainties.
    When relative uncertainties are provided, they will be converted to
    absolute bounds by PlotBuilder using the nominal y values.

    :func:`~kika.plotting.plottable` returns one in a :class:`PlotItem`: a
    relative band for evaluated data (the relative standard deviation from the
    covariance, times ``sigma``) and an absolute one with ``style='errorbar'``
    for measurements (the reported errors times ``sigma``, as ``y`` minus and
    plus that).

    Parameters
    ----------
    x : array_like
        Abscissae of the band, the same as those of the curve it belongs to, in
        the same unit.
    y_lower, y_upper : array_like, optional
        Absolute bounds, in the unit of the curve's ``y``, of the length of
        ``x``. ``sigma`` is not applied to them. Give both or use
        ``relative_uncertainty``.
    relative_uncertainty : array_like, optional
        One standard deviation as a fraction of the curve's ``y`` (0.05 for
        5 %), of the length of ``x``. The band is then ``y`` plus and minus
        ``abs(y) * sigma * relative_uncertainty``, computed by
        :meth:`to_absolute`. Cannot be combined with ``y_lower`` and ``y_upper``.
    sigma : float
        Number of standard deviations that ``relative_uncertainty`` is
        multiplied by (default 1.0, a one-sigma band). It has no effect on
        absolute bounds.
    color : str or tuple, optional
        Fill color
    alpha : float
        Transparency (default: 0.2)
    label : str, optional
        Label for legend
    style : {'band', 'errorbar'}
        Rendering style: ``'band'`` for ``fill_between`` (default) or
        ``'errorbar'`` for vertical error bars via ``ax.errorbar()``.
    capsize : float
        Cap size for error bars (default: 2.5). Only used when
        ``style='errorbar'``.
    elinewidth : float
        Line width for error bars (default: 1.5). Only used when
        ``style='errorbar'``.

    Raises
    ------
    ValueError
        If ``style`` is not one of the two above; if ``relative_uncertainty`` is
        given with ``y_lower`` or ``y_upper``; if neither
        ``relative_uncertainty`` nor both bounds are given; or if an array does
        not have the length of ``x``.
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        relative_uncertainty: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        color: Optional[Union[str, Tuple]] = None,
        alpha: float = 0.2,
        label: Optional[str] = None,
        style: str = 'band',
        capsize: float = 2.5,
        elinewidth: float = 1.5,
    ):
        self.x = np.asarray(x)
        self.sigma = sigma
        self.color = color
        self.alpha = alpha
        self.label = label

        if style not in ('band', 'errorbar'):
            raise ValueError(f"style must be 'band' or 'errorbar', got '{style}'")
        self.style = style
        self.capsize = capsize
        self.elinewidth = elinewidth

        # Store either absolute or relative uncertainties
        if relative_uncertainty is not None:
            if y_lower is not None or y_upper is not None:
                raise ValueError("Cannot specify both relative_uncertainty and y_lower/y_upper")
            self.relative_uncertainty = np.asarray(relative_uncertainty)
            self.y_lower = None
            self.y_upper = None
            # Validate
            if len(self.x) != len(self.relative_uncertainty):
                raise ValueError("x and relative_uncertainty must have the same length")
        elif y_lower is not None and y_upper is not None:
            self.y_lower = np.asarray(y_lower)
            self.y_upper = np.asarray(y_upper)
            self.relative_uncertainty = None
            # Validate
            if not (len(self.x) == len(self.y_lower) == len(self.y_upper)):
                raise ValueError("x, y_lower, and y_upper must have the same length")
        else:
            raise ValueError("Must provide either (y_lower, y_upper) or relative_uncertainty")
    
    def is_relative(self) -> bool:
        """Check if this band stores relative uncertainties."""
        return self.relative_uncertainty is not None
    
    def to_absolute(self, y_nominal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert relative uncertainties to absolute bounds.
        
        Parameters
        ----------
        y_nominal : array_like
            Nominal y values to apply uncertainties to, of the length of ``x``.
            Ignored by a band that already holds absolute bounds.

        Returns
        -------
        tuple of numpy.ndarray
            ``(y_lower, y_upper)``. For a relative band,
            ``y_nominal -/+ abs(y_nominal) * sigma * relative_uncertainty``,
            symmetric about ``y_nominal``. For an absolute band, the stored
            bounds, unchanged.

        Raises
        ------
        ValueError
            If the band is relative and ``y_nominal`` does not have the length
            of ``relative_uncertainty``.
        """
        if not self.is_relative():
            return self.y_lower, self.y_upper
        
        y_nominal = np.asarray(y_nominal)
        if len(y_nominal) != len(self.relative_uncertainty):
            raise ValueError("y_nominal must have same length as relative_uncertainty")
        
        abs_unc = np.abs(y_nominal) * self.sigma * self.relative_uncertainty
        y_lower = y_nominal - abs_unc
        y_upper = y_nominal + abs_unc
        
        return y_lower, y_upper
    
    def get_fill_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for ax.fill_between()."""
        kwargs = {'alpha': self.alpha}

        if self.color is not None:
            kwargs['color'] = self.color
        if self.label is not None:
            kwargs['label'] = self.label

        return kwargs

    def get_errorbar_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for ax.errorbar() when style='errorbar'."""
        kwargs = {
            'fmt': ' ',
            'capsize': self.capsize,
            'elinewidth': self.elinewidth,
        }
        if self.color is not None:
            kwargs['ecolor'] = self.color
        if self.label is not None:
            kwargs['label'] = self.label
        return kwargs


@dataclass
class MultigroupCrossSectionPlotData(PlotData):
    """
    Plottable data for multigroup cross sections.

    A :class:`PlotData` drawn as steps, with ``x`` the group energies. Also
    exported as ``MultigroupXSPlotData``, the same class.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'<symbol> MT=<mt>'`` from the ZAID, or
        ``'MT=<mt>'`` without one; it stays None without an ``mt``.
    plot_type : str
        As in :class:`PlotData`, except that the default ``'line'`` becomes
        ``'step'``.
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    energy_bins : array_like, optional
        Energy bin boundaries, stored as an array. The builders do not read it.
    step_where : {'pre', 'post', 'mid'}
        Where the steps change, as in ``matplotlib.axes.Axes.step``. It is read
        only when the curve is drawn as steps, and a ``drawstyle`` containing
        ``'steps'`` (``'steps-pre'``, ``'steps-mid'``, ``'steps-post'``) overrides
        it.
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for cross sections
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class MultigroupUncertaintyPlotData(PlotData):
    """
    Plottable data for multigroup cross section uncertainties.

    A :class:`PlotData` drawn as steps, with ``x`` the group energies and ``y``
    the uncertainty of the group cross section. Passed as the ``uncertainty``
    of :meth:`PlotBuilder.add_data`, it becomes a band around the nominal curve.

    Attributes
    ----------
    x, y, color, linestyle, linewidth, marker, markersize, alpha, drawstyle
        As in :class:`PlotData`.
    metadata, quantity, x_unit, y_unit, provenance, interpolation
        As in :class:`PlotData`.
    label : str, optional
        Legend entry. None becomes ``'<symbol> MT=<mt>'`` from the ZAID, or
        ``'MT=<mt>'`` without one; it stays None without an ``mt``.
    plot_type : str
        As in :class:`PlotData`, except that the default ``'line'`` becomes
        ``'step'``.
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    uncertainty_type : {'relative', 'absolute'}
        ``'relative'``: ``y`` is a percentage (the band divides it by 100).
        ``'absolute'``: ``y`` is the half-width in the units of the nominal
        curve. Any other string is read as ``'absolute'`` by the band.
    energy_bins : array_like, optional
        Energy bin boundaries, stored as an array. The builders do not read it.
    step_where : {'pre', 'post', 'mid'}
        Where the steps change, as in ``matplotlib.axes.Axes.step``. It is read
        only when the curve is drawn as steps, and a ``drawstyle`` containing
        ``'steps'`` (``'steps-pre'``, ``'steps-mid'``, ``'steps-post'``) overrides
        it.
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class HeatmapPlotData:
    """
    Base class for 2D heatmap data.
    
    This encapsulates data for heatmap/image plots including matrix data,
    colormap configuration, and colorbar settings.
    
    Attributes
    ----------
    matrix_data : np.ndarray
        2D matrix to display
    extent : tuple of float, optional
        (xmin, xmax, ymin, ymax) for imshow extent parameter
    x_edges : np.ndarray, optional
        Bin edges for x-axis (for pcolormesh, preferred over imshow)
    y_edges : np.ndarray, optional
        Bin edges for y-axis (for pcolormesh)
    cmap : str or colormap
        Colormap name or instance
    norm : matplotlib.colors.Normalize, optional
        Custom normalization (e.g., TwoSlopeNorm for diverging colormaps).
        If not provided, auto-scaling is used.
    colorbar_label : str, optional
        Label for the colorbar
    colorbar_position : str
        Position for colorbar: 'right', 'bottom', 'top', 'left'
    mask_value : float, optional
        Value to mask (e.g., 0.0 for correlation matrices)
    mask_color : str
        Color for masked regions
    label : str, optional
        Label for plot identification
    metadata : dict
        Additional metadata
    """
    matrix_data: np.ndarray
    extent: Optional[Tuple[float, float, float, float]] = None
    x_edges: Optional[np.ndarray] = None
    y_edges: Optional[np.ndarray] = None
    cmap: Union[str, Any] = "viridis"
    norm: Optional[Any] = None
    colorbar_label: Optional[str] = None
    colorbar_position: str = "right"
    mask_value: Optional[float] = None
    mask_color: str = "#F0F0F0"
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate heatmap data."""
        # Nobody chose a colour map: the builder may replace it with the style's.
        if isinstance(self.cmap, str) and self.cmap == "viridis":
            self.metadata.setdefault('auto_cmap', True)
        self.matrix_data = np.asarray(self.matrix_data)
        if self.matrix_data.ndim != 2:
            raise ValueError(f"matrix_data must be 2D array, got shape {self.matrix_data.shape}")
        
        if self.x_edges is not None:
            self.x_edges = np.asarray(self.x_edges)
        if self.y_edges is not None:
            self.y_edges = np.asarray(self.y_edges)
    
    def get_masked_data(self) -> np.ndarray:
        """
        Get masked array with mask_value and NaN values masked.

        Returns
        -------
        np.ma.MaskedArray
            Masked version of data where NaN values and mask_value (if set) are masked
        """
        # Always mask NaN values so they render as grey via set_bad()
        mask = ~np.isfinite(self.matrix_data)
        if self.mask_value is not None:
            mask = mask | (self.matrix_data == self.mask_value)
        return np.ma.masked_where(mask, self.matrix_data)


@dataclass
class CovarianceHeatmapData(HeatmapPlotData):
    """
    Covariance/correlation matrix heatmap data.

    The ``to_heatmap_data`` methods of the covariance classes return it, and
    :class:`HeatmapBuilder` draws it.

    Attributes
    ----------
    matrix_data, extent, x_edges, y_edges, norm, colorbar_position, mask_value
        As in :class:`HeatmapPlotData`. With no ``norm`` given, a correlation
        matrix gets a norm symmetric about zero, ``TwoSlopeNorm`` from minus to
        plus the largest absolute unmasked value (1 if that is below 1e-10).
        A covariance matrix gets ``TwoSlopeNorm`` over the unmasked range when
        it spans zero and ``Normalize`` over it otherwise.
    mask_color, label, metadata
        As in :class:`HeatmapPlotData`.
    cmap : str or colormap
        As in :class:`HeatmapPlotData`, except that the default ``'viridis'``
        becomes ``'RdYlGn'`` for a correlation matrix.
    colorbar_label : str, optional
        None becomes ``'Correlation'`` or ``'Covariance'`` from ``matrix_type``.
    matrix_type : {'cov', 'corr'}
        Covariance or correlation. It selects the default norm, colour map and
        colorbar label above.
    zaid : int or list of int, optional
        Isotope identifier(s): one ZAID for a single-isotope heatmap, a list for a
        multi-isotope one.
    block_info : dict, optional
        The block structure of the matrix, with the keys below.

        For a single isotope:

        * ``'mts'``: list of MT numbers.
        * ``'G'``: number of energy groups per MT.
        * ``'ranges'``: list of ``(start, end)`` indices, one per MT block.
        * ``'energy_ranges'``: dict from MT to ``(start, end)`` coordinates.

        For several isotopes (``'is_multi_isotope'`` is True):

        * ``'blocks'``: list of ``(zaid, mt)`` tuples.
        * ``'zaids'``: list of ZAIDs.
        * ``'mts'``: list of MT numbers.
        * ``'G'``: number of energy groups per block.
        * ``'ranges'``: dict from ``(zaid, mt)`` to ``(start, end)`` indices.
        * ``'energy_ranges'``: dict from ``(zaid, mt)`` to ``(start, end)``
          coordinates.
        * ``'is_multi_isotope'``: True.
    uncertainty_data : dict, optional
        Uncertainty values for the optional panels above the heatmap, in
        percent: ``{mt: array}`` for a single isotope, ``{(zaid, mt): array}``
        for several.
    energy_grid : numpy.ndarray, optional
        Energy bin boundaries (length G+1) for the energy ticks, in the unit the
        producer used. The ticks print the values as they are, with no unit.
    mt_labels : list of str, optional
        Custom labels for blocks. For multi-isotope, typically "Symbol-MT#" format.
    is_diagonal : bool
        Whether this represents diagonal blocks (vs off-diagonal)
    scale : str
        ``'log'`` (default) lays the energy axis out on log10 of ``energy_grid``;
        any other value lays it out linearly.
    """
    matrix_type: str = "corr"
    zaid: Optional[Union[int, List[int]]] = None
    block_info: Optional[Dict[str, Any]] = None
    uncertainty_data: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None
    energy_grid: Optional[np.ndarray] = None
    mt_labels: Optional[List[str]] = None
    is_diagonal: bool = True
    scale: str = "log"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Auto-configure for correlation matrices with diverging colormap
        if self.matrix_type == "corr" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm
            # Use masked data to exclude mask_value from absmax calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                # compressed() returns only non-masked values
                valid_data = masked_data.compressed()
                absmax = np.nanmax(np.abs(valid_data)) if len(valid_data) > 0 else 1.0
            else:
                absmax = np.nanmax(np.abs(masked_data))
            if absmax < 1e-10:
                absmax = 1.0  # Avoid division by zero
            self.norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

        # Auto-configure for covariance matrices with proper normalization
        # This ensures masked/NaN values are properly displayed as grey
        if self.matrix_type == "cov" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm, Normalize
            # Use masked data to exclude NaN values from range calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                valid_data = masked_data.compressed()
                if len(valid_data) > 0:
                    vmin = np.nanmin(valid_data)
                    vmax = np.nanmax(valid_data)
                else:
                    vmin, vmax = -1.0, 1.0
            else:
                vmin = np.nanmin(masked_data)
                vmax = np.nanmax(masked_data)

            # Use TwoSlopeNorm if data spans zero, otherwise use regular Normalize
            if vmin < 0 < vmax:
                self.norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                self.norm = Normalize(vmin=vmin, vmax=vmax)

        if self.matrix_type == "corr" and isinstance(self.cmap, str) and self.cmap == "viridis":
            self.cmap = "RdYlGn"  # Diverging colormap for correlations
        
        # Auto-generate colorbar label if not provided
        if self.colorbar_label is None:
            self.colorbar_label = "Correlation" if self.matrix_type == "corr" else "Covariance"
        
        # Store metadata
        self.metadata['matrix_type'] = self.matrix_type
        self.metadata['zaid'] = self.zaid
        self.metadata['is_diagonal'] = self.is_diagonal
        self.metadata['scale'] = self.scale
        
        # Convert energy_grid if provided
        if self.energy_grid is not None:
            self.energy_grid = np.asarray(self.energy_grid)


@dataclass
class LegendreHeatmapData(HeatmapPlotData):
    """
    Angular distribution (MF34) covariance heatmap data.

    These heatmaps can be more complex because different Legendre coefficients
    may have different energy group structures. Also exported as
    ``MF34HeatmapData``, the same class.

    Attributes
    ----------
    matrix_data, extent, x_edges, y_edges, norm, colorbar_position, mask_value
        As in :class:`HeatmapPlotData`. With no ``norm`` given, the default is
        the one of :class:`CovarianceHeatmapData` for the ``matrix_type``.
    mask_color, label, metadata
        As in :class:`HeatmapPlotData`.
    cmap : str or colormap
        As in :class:`HeatmapPlotData`, except that the default ``'viridis'``
        becomes ``'RdYlGn'`` for a correlation matrix.
    colorbar_label : str, optional
        None becomes ``'Correlation'`` or ``'Covariance'`` from ``matrix_type``.
    isotope : int
        Isotope identifier (ZAID)
    mt : int
        MT reaction number
    legendre_coeffs : list of int
        Legendre polynomial orders included
    matrix_type : {'cov', 'corr'}
        Covariance or correlation.
    scale : str
        ``'log'`` (default) lays the energy axis out on log10 of the energy
        grids; any other value lays it out linearly.
    block_info : dict, optional
        Per-Legendre block structure information, with the keys:

        * ``'legendre_coeffs'``: list of L values.
        * ``'G_per_L'``: dict ``{L: G_L}`` of energy groups per L.
        * ``'ranges'``: dict ``{L: (start, end)}`` of index ranges.
    uncertainty_data : dict, optional
        Uncertainty values for the optional panels, in percent:
        ``{L: array of G_L values}``.
    energy_grids : dict, optional
        Energy bin boundaries per Legendre coefficient for the energy ticks,
        ``{L: array of G_L + 1 values}``, in the unit the producer used. The
        ticks print the values as they are, with no unit.
    is_diagonal : bool
        Whether diagonal blocks (L vs L) or off-diagonal (L1 vs L2)
    """
    isotope: int = 0
    mt: int = 0
    legendre_coeffs: List[int] = field(default_factory=list)
    matrix_type: str = "corr"
    scale: str = "log"
    block_info: Optional[Dict[str, Any]] = None
    uncertainty_data: Optional[Dict[int, np.ndarray]] = None
    energy_grids: Optional[Dict[int, np.ndarray]] = None
    is_diagonal: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        
        # Auto-configure for correlation matrices
        if self.matrix_type == "corr" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm
            # Use masked data to exclude mask_value from absmax calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                # compressed() returns only non-masked values
                valid_data = masked_data.compressed()
                absmax = np.nanmax(np.abs(valid_data)) if len(valid_data) > 0 else 1.0
            else:
                absmax = np.nanmax(np.abs(masked_data))
            if absmax < 1e-10:
                absmax = 1.0
            self.norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

        # Auto-configure for covariance matrices with proper normalization
        # This ensures masked/NaN values are properly displayed as grey
        if self.matrix_type == "cov" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm, Normalize
            # Use masked data to exclude NaN values from range calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                valid_data = masked_data.compressed()
                if len(valid_data) > 0:
                    vmin = np.nanmin(valid_data)
                    vmax = np.nanmax(valid_data)
                else:
                    vmin, vmax = -1.0, 1.0
            else:
                vmin = np.nanmin(masked_data)
                vmax = np.nanmax(masked_data)

            # Use TwoSlopeNorm if data spans zero, otherwise use regular Normalize
            if vmin < 0 < vmax:
                self.norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                self.norm = Normalize(vmin=vmin, vmax=vmax)

        if self.matrix_type == "corr" and isinstance(self.cmap, str) and self.cmap == "viridis":
            self.cmap = "RdYlGn"

        # Auto-generate colorbar label
        if self.colorbar_label is None:
            self.colorbar_label = "Correlation" if self.matrix_type == "corr" else "Covariance"

        # Store metadata
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['legendre_coeffs'] = self.legendre_coeffs
        self.metadata['matrix_type'] = self.matrix_type
        self.metadata['scale'] = self.scale
        self.metadata['is_diagonal'] = self.is_diagonal
        
        # Convert energy_grids if provided
        if self.energy_grids is not None:
            self.energy_grids = {L: np.asarray(grid) for L, grid in self.energy_grids.items()}


# Backward compatibility aliases
MultigroupXSPlotData = MultigroupCrossSectionPlotData
MF34HeatmapData = LegendreHeatmapData


@dataclass
class PlotItem:
    """
    What :func:`kika.plotting.plottable` returns: a curve and its uncertainty.

    Always this shape, whatever the source, so a caller never special-cases
    tuples, lists or ``None``. It unpacks like the ``(data, band)`` tuples the
    older ``to_plot_data`` methods return, and ``PlotBuilder.add_data`` takes it
    directly.

    Attributes
    ----------
    data : PlotData
        The curve, with ``quantity``, ``x_unit``, ``y_unit`` and ``provenance``
        filled in. Its units are the ones the adapter set, and the canonical
        unit of the quantity (eV, barn, b/sr, ...) where it set none.
    band : UncertaintyBand, optional
        Its uncertainty: a relative band for evaluated data, error bars for
        measurements. ``None`` when the source carries none.
    """
    data: PlotData
    band: Optional[UncertaintyBand] = None

    def __iter__(self) -> Iterator[Any]:
        yield self.data
        yield self.band

    @property
    def label(self) -> Optional[str]:
        """The legend entry of the curve, ``data.label`` (None when it has none)."""
        return self.data.label

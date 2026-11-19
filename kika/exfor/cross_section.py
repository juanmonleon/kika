"""
EXFOR cross section class with unit conversion support.

This module provides ExforCrossSection, a specialized class for working
with experimental cross section data from EXFOR (quantity codes SIG, CS).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import json

import numpy as np
import pandas as pd

from kika.plotting import PlotData
from kika.exfor.exfor_entry import ExforEntry
from kika.exfor._constants import (
    SCHEMA_VERSION,
    ENERGY_TO_MEV,
    PLOTTING_ENERGY_TOLERANCE,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Cross section unit conversion factors (to barns)
XS_TO_B = {
    "b": 1.0,
    "B": 1.0,
    "mb": 1e-3,
    "MB": 1e-3,
    "ub": 1e-6,
    "UB": 1e-6,
    "mub": 1e-6,
    "MUB": 1e-6,
    "nb": 1e-9,
    "NB": 1e-9,
}


@dataclass
class ExforCrossSection(ExforEntry):
    """
    EXFOR cross-section against incident energy, with unit conversion and interpolation.

    Returned by the database queries for cross-section quantities, or built from a
    dict or a JSON file with :meth:`from_dict` and :meth:`from_json`. The data are
    a table of energy, cross-section and error.

    Attributes
    ----------
    entry, subentry, quantity, citation, reaction, facility, method, detector, sample
        As in :class:`~kika.exfor.ExforEntry`.
    units : dict of str
        Units of the table, with the keys ``"energy"`` (``"eV"``, ``"keV"`` or
        ``"MeV"``) and ``"cross_section"`` (``"b"``, ``"mb"``, ``"ub"`` or
        ``"nb"``). The default is MeV and b. The conversion methods update it.
        Objects built from the database get ``"b/sr"`` when the dataset's unit
        contains ``/SR``. That string is not one of the four, so conversions
        treat it as a factor of 1.0: it neither scales to nor from ``"b/sr"``.

    Notes
    -----
    The table has the columns ``energy`` and ``cross_section`` in the units above
    and ``error``, the uncertainty in the cross-section unit. The ``data``
    property returns a copy of it.

    Examples
    --------
    >>> from kika.exfor import X4ProDatabase
    >>> db = X4ProDatabase()
    >>> exp = db.load_experiment("10571002")  # Cross section data
    >>> print(exp.label)
    Kinney et al. (1970)
    >>> print(exp.energies())
    [1.0, 1.5, 2.0, ...]
    >>> df = exp.to_dataframe(energy=(1.0, 5.0))  # Data in range
    """

    units: Dict[str, str] = field(
        default_factory=lambda: {"energy": "MeV", "cross_section": "b"}
    )
    _data: Optional[pd.DataFrame] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize empty DataFrame if not provided."""
        if self._data is None:
            self._data = pd.DataFrame(columns=["energy", "cross_section", "error"])

    # =========================================================================
    # Data Access Methods
    # =========================================================================

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the cross section data as DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the data with columns: energy, cross_section, error
        """
        return self._data.copy()

    @property
    def num_data_points(self) -> int:
        """Return the number of data points."""
        return len(self._data)

    def energies(
        self,
        unit: str = "MeV",
        bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Get available energy values.

        Parameters
        ----------
        unit : str
            Output unit: 'eV', 'keV', or 'MeV' (default: 'MeV')
        bounds : Tuple[float, float], optional
            Filter to [min, max] bounds (in requested units)

        Returns
        -------
        np.ndarray
            Sorted array of unique energy values
        """
        if self._data.empty:
            return np.array([])

        # Get raw energies
        energies = self._data["energy"].values

        # Convert to requested units
        from_unit = self.units.get("energy", "MeV")
        factor = ENERGY_TO_MEV.get(from_unit, 1.0) / ENERGY_TO_MEV.get(unit, 1.0)
        energies = energies * factor

        # Apply bounds filter
        if bounds is not None:
            mask = (energies >= bounds[0]) & (energies <= bounds[1])
            energies = energies[mask]

        return np.sort(np.unique(energies))

    def to_dataframe(
        self,
        energy: Optional[Union[float, Tuple[float, float]]] = None,
        tolerance: float = 0.05,
        energy_unit: str = "MeV",
        cross_section_unit: str = "b",
    ) -> pd.DataFrame:
        """
        Convert EXFOR data to DataFrame with optional filtering.

        Parameters
        ----------
        energy : float or Tuple[float, float], optional
            - float: Select data at specific energy (with tolerance)
            - tuple: Select data in energy range [min, max]
            - None: Include all energies
        tolerance : float
            Relative tolerance for energy matching (default: 0.05 = 5%)
        energy_unit : str
            Output energy unit (default: 'MeV')
        cross_section_unit : str
            Output cross-section unit (default: 'b')

        Returns
        -------
        pd.DataFrame
            Columns: [energy, cross_section, error]
        """
        if self._data.empty:
            return pd.DataFrame(columns=["energy", "cross_section", "error"])

        df = self._data.copy()

        # Unit conversion factors
        from_energy_unit = self.units.get("energy", "MeV")
        from_xs_unit = self.units.get("cross_section", "b")

        energy_factor = ENERGY_TO_MEV.get(from_energy_unit, 1.0) / ENERGY_TO_MEV.get(energy_unit, 1.0)
        xs_factor = XS_TO_B.get(from_xs_unit, 1.0) / XS_TO_B.get(cross_section_unit, 1.0)

        # Apply energy filter
        if energy is not None:
            if isinstance(energy, tuple):
                # Range filter (convert bounds to internal units)
                internal_factor = ENERGY_TO_MEV.get(energy_unit, 1.0) / ENERGY_TO_MEV.get(from_energy_unit, 1.0)
                min_e = energy[0] * internal_factor
                max_e = energy[1] * internal_factor
                df = df[(df["energy"] >= min_e) & (df["energy"] <= max_e)]
            else:
                # Single value with tolerance
                internal_factor = ENERGY_TO_MEV.get(energy_unit, 1.0) / ENERGY_TO_MEV.get(from_energy_unit, 1.0)
                target_e = energy * internal_factor
                tol = abs(target_e) * tolerance if target_e != 0 else tolerance
                df = df[(df["energy"] >= target_e - tol) & (df["energy"] <= target_e + tol)]

        # Convert units
        df["energy"] = df["energy"] * energy_factor
        df["cross_section"] = df["cross_section"] * xs_factor
        if "error" in df.columns:
            df["error"] = df["error"] * xs_factor

        return df

    def filter(
        self,
        energy: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Rows of the table at an energy or in an energy range, in MeV and b.

        It is :meth:`to_dataframe` with its defaults, so the energy is in MeV with a
        5 % relative tolerance for a single value, and the cross-section is in b.

        Parameters
        ----------
        energy : float or tuple of float, optional
            An energy in MeV, or ``(minimum, maximum)``. None keeps every row.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        pandas.DataFrame
            Columns ``energy`` (MeV), ``cross_section`` and ``error`` (b).
        """
        return self.to_dataframe(energy=energy)

    # =========================================================================
    # Unit Conversion Methods
    # =========================================================================

    def convert_energy(
        self, to: str, inplace: bool = False
    ) -> Optional["ExforCrossSection"]:
        """
        Convert energy units.

        Parameters
        ----------
        to : str
            Target unit ('eV', 'keV', or 'MeV')
        inplace : bool
            If True, modify this object; if False, return new object

        Returns
        -------
        ExforCrossSection or None
            Converted object (if inplace=False) or None
        """
        from_unit = self.units.get("energy", "MeV")
        if from_unit == to:
            if inplace:
                return None
            return deepcopy(self)

        factor = ENERGY_TO_MEV.get(from_unit, 1.0) / ENERGY_TO_MEV.get(to, 1.0)
        target_obj = self if inplace else deepcopy(self)

        target_obj._data["energy"] = target_obj._data["energy"] * factor
        target_obj.units["energy"] = to

        if inplace:
            return None
        return target_obj

    def convert_cross_section(
        self, to: str, inplace: bool = False
    ) -> Optional["ExforCrossSection"]:
        """
        Convert cross section units.

        Parameters
        ----------
        to : str
            Target unit ('b', 'mb', 'ub', 'nb')
        inplace : bool
            If True, modify this object; if False, return new object

        Returns
        -------
        ExforCrossSection or None
            Converted object (if inplace=False) or None
        """
        from_unit = self.units.get("cross_section", "b")
        if from_unit.lower() == to.lower():
            if inplace:
                return None
            return deepcopy(self)

        factor = XS_TO_B.get(from_unit, 1.0) / XS_TO_B.get(to, 1.0)
        target_obj = self if inplace else deepcopy(self)

        target_obj._data["cross_section"] = target_obj._data["cross_section"] * factor
        if "error" in target_obj._data.columns:
            target_obj._data["error"] = target_obj._data["error"] * factor
        target_obj.units["cross_section"] = to

        if inplace:
            return None
        return target_obj

    # =========================================================================
    # Interpolation Methods
    # =========================================================================

    def interpolate(
        self,
        energies: Union[float, np.ndarray],
        method: str = "linear",
        energy_unit: str = "MeV",
        cross_section_unit: str = "b",
    ) -> Union[float, np.ndarray]:
        """
        Interpolate cross section at given energy(ies).

        Parameters
        ----------
        energies : float or array-like
            Energy value(s) at which to interpolate
        method : str
            Interpolation method: 'linear', 'log-log', 'lin-log' (default: 'linear')
        energy_unit : str
            Unit of input energies (default: 'MeV')
        cross_section_unit : str
            Unit of output cross sections (default: 'b')

        Returns
        -------
        float or np.ndarray
            Interpolated cross section value(s)
        """
        from scipy.interpolate import interp1d

        df = self.to_dataframe(energy_unit=energy_unit, cross_section_unit=cross_section_unit)
        if df.empty:
            if np.isscalar(energies):
                return np.nan
            return np.full_like(np.asarray(energies), np.nan, dtype=float)

        # Sort by energy
        df = df.sort_values("energy")
        x = df["energy"].values
        y = df["cross_section"].values

        # Apply log transforms if requested
        if method == "log-log":
            # Avoid log(0)
            mask = (x > 0) & (y > 0)
            x = np.log10(x[mask])
            y = np.log10(y[mask])
            interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
            energies_arr = np.atleast_1d(energies)
            result = 10 ** interp_func(np.log10(energies_arr))
        elif method == "lin-log":
            mask = y > 0
            y = np.log10(y[mask])
            x = x[mask]
            interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
            energies_arr = np.atleast_1d(energies)
            result = 10 ** interp_func(energies_arr)
        else:
            interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
            energies_arr = np.atleast_1d(energies)
            result = interp_func(energies_arr)

        if np.isscalar(energies):
            return float(result[0])
        return result

    # =========================================================================
    # PlotBuilder Integration
    # =========================================================================

    def to_plot_data(
        self,
        energy: Optional[Union[float, Tuple[float, float]]] = None,
        *,
        energy_unit: str = "MeV",
        cross_section_unit: str = "b",
        uncertainty: bool = True,
        connect_points: bool = True,
        label: Optional[str] = None,
        include_natural_tag: bool = True,
        **styling_kwargs,
    ) -> Union["PlotData", Tuple["PlotData", None], None]:
        """
        Extract cross section data for plotting with PlotBuilder.

        This method returns data in a format compatible with kika's PlotBuilder,
        enabling easy visualization and comparison of EXFOR experimental data
        with theoretical calculations from ENDF files.

        Parameters
        ----------
        energy : float, tuple, or None
            Energy selection, in *energy_unit*:
            - float: Select data at specific energy (with a 5 % relative tolerance)
            - tuple: Select data in energy range (min, max)
            - None: Include all data (default)
        energy_unit : {'eV', 'keV', 'MeV'}
            Energy unit of the x-axis, and of *energy*.
        cross_section_unit : {'b', 'mb', 'ub', 'nb'}
            Cross-section unit of the y-axis.
        uncertainty : bool, default True
            If True, include error bars in the plot
        connect_points : bool, default True
            If True, connect data points with lines
        label : str or None, default None
            Custom label for legend. If None, auto-generates from author and year.
        include_natural_tag : bool, default True
            If True and target is natural element, append '[nat]' to label.
        **styling_kwargs
            Additional styling parameters passed to PlotData:
            - color: Line/marker color
            - marker: Marker style (default: 'o')
            - markersize: Marker size (default: 5)
            - capsize: Error bar cap size (default: 2)

        Returns
        -------
        ~kika.plotting.CrossSectionPlotData, tuple or None
            The points sorted by energy, in *energy_unit* and *cross_section_unit*.
            With ``uncertainty=True`` and at least one positive error, the absolute
            errors are in ``metadata["yerr"]``.

            - None if no row matches *energy*.
            - ``uncertainty=True``: the tuple ``(plot_data, None)``. The second item
              is always None.
            - ``uncertainty=False``: the plot data alone.

        Examples
        --------
        >>> from kika.exfor import X4ProDatabase
        >>> from kika.plotting import PlotBuilder
        >>>
        >>> db = X4ProDatabase()
        >>> exp = db.query_cross_sections("Fe56", mt=1)[0]
        >>>
        >>> plot_data = exp.to_plot_data(energy=(1.0, 10.0))
        >>> if plot_data is not None:
        ...     fig = PlotBuilder().add_data(plot_data).set_labels(
        ...         x_label='Energy (MeV)',
        ...         y_label='Cross Section (b)'
        ...     ).build()
        """
        from kika.plotting import CrossSectionPlotData

        # Get data in requested units
        df = self.to_dataframe(
            energy=energy,
            energy_unit=energy_unit,
            cross_section_unit=cross_section_unit,
        )

        if df.empty:
            return None

        # Sort by energy
        df = df.sort_values("energy")

        # Extract arrays
        energies = df["energy"].values
        cross_sections = df["cross_section"].values
        errors = df["error"].values if "error" in df.columns else np.zeros_like(cross_sections)

        # Generate label if not provided
        if label is None:
            base_label = self.label
            if include_natural_tag and self.is_natural_target:
                label = f"{base_label} [nat]"
            else:
                label = base_label

        # Determine MT from reaction metadata
        mt = None
        if self.reaction:
            # Try to extract MT from notation or process
            notation = self.reaction.get("notation", "")
            if "TOT" in notation.upper() or self.process == "TOT":
                mt = 1
            elif "EL" in notation.upper() or self.process == "EL":
                mt = 2

        # Build metadata
        metadata = {
            "exfor_entry": self.entry,
            "exfor_subentry": self.subentry,
            "target": self.target,
            "zaid": self.zaid,
            "process": self.process,
            "source": "EXFOR",
            "energy_unit": energy_unit,
            "cross_section_unit": cross_section_unit,
        }

        # Styling defaults
        marker = styling_kwargs.pop("marker", "o")
        markersize = styling_kwargs.pop("markersize", 5)
        linestyle = styling_kwargs.pop("linestyle", None)
        capsize = styling_kwargs.pop("capsize", 2)

        if linestyle is None:
            linestyle = "-" if connect_points else "none"

        has_errors = np.any(errors > 0)

        if uncertainty and has_errors:
            plot_data = CrossSectionPlotData(
                x=energies,
                y=cross_sections,
                isotope=self.target,
                mt=mt,
                label=label,
                plot_type="errorbar",
                marker=marker,
                markersize=markersize,
                linestyle=linestyle,
                **styling_kwargs,
            )
            plot_data.metadata["yerr"] = errors
            plot_data.metadata["capsize"] = capsize
        else:
            plot_type = "line" if connect_points else "scatter"
            plot_data = CrossSectionPlotData(
                x=energies,
                y=cross_sections,
                isotope=self.target,
                mt=mt,
                label=label,
                plot_type=plot_type,
                marker=marker if not connect_points else None,
                markersize=markersize,
                linestyle=linestyle if connect_points else None,
                **styling_kwargs,
            )

        plot_data.metadata.update(metadata)

        if uncertainty:
            return plot_data, None
        return plot_data

    # =========================================================================
    # Representation Methods
    # =========================================================================

    def summary(self) -> str:
        """
        Return a formatted summary of the experiment.

        Returns
        -------
        str
            Multi-line summary string
        """
        energies = self.energies()
        n_points = self.num_data_points

        e_min = f"{energies.min():.4g}" if len(energies) > 0 else "N/A"
        e_max = f"{energies.max():.4g}" if len(energies) > 0 else "N/A"

        lines = [
            f"ExforCrossSection",
            f"  Entry:       {self.entry}{self.subentry}",
            f"  Label:       {self.label}",
            f"  Quantity:    {self.quantity}",
            f"  Target:      {self.target} (ZAID: {self.zaid})",
            f"  Projectile:  {self.projectile}",
            f"  Data points: {n_points}",
            f"  Energy:      {e_min} - {e_max} {self.units.get('energy', 'MeV')}",
            f"  Units:       {self.units}",
            f"  URL:         {self.exfor_url}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return summary representation."""
        return self.summary()

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = super().to_dict()
        result["units"] = self.units
        result["data"] = self._data.to_dict(orient="records")
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExforCrossSection":
        """
        Build a cross-section from a dict with the layout of the JSON files.

        ``units`` is taken as written, with no normalisation; missing, it is MeV and
        b. ``data`` is a list of records with the keys ``energy``,
        ``cross_section`` and ``error``, in those units. Missing or empty, the table
        is empty.

        Parameters
        ----------
        data : dict
            The values, as in a JSON file.

        Returns
        -------
        ExforCrossSection
            The cross-section, with the table and units of *data*.
        """
        fields = cls._parse_base_fields(data)
        fields["units"] = data.get("units", {"energy": "MeV", "cross_section": "b"})

        data_records = data.get("data", [])
        if data_records:
            fields["_data"] = pd.DataFrame(data_records)
        else:
            fields["_data"] = pd.DataFrame(columns=["energy", "cross_section", "error"])

        return cls(**fields)

    @classmethod
    def from_json(cls, filepath: str) -> "ExforCrossSection":
        """
        Load from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file

        Returns
        -------
        ExforCrossSection
            Loaded instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

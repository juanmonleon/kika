"""
General EXFOR experiment class for any quantity type.

This module provides ExforExperiment, a flexible class that can hold data
for any EXFOR quantity type (CS, DA, FY, DE, etc.) with full metadata
and data access capabilities.

Quantity codes (from EXFOR database):
- CS: Cross section data
- DA: Differential data with respect to angle
- FY: Fission product yields
- DE: Differential data with respect to energy
- PY: Product yields
- RI: Resonance integrals
- RP: Resonance parameters
- POL: Polarization data
See EXFOR_QUANTITY_CODES in _constants.py for full list.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from kika.exfor.exfor_entry import ExforEntry
from kika.exfor._constants import EXFOR_QUANTITY_CODES


@dataclass
class ExforExperiment(ExforEntry):
    """
    General EXFOR experiment class for any quantity type.

    Unlike ExforAngularDistribution which is specialized for DA data,
    this class can hold cross sections (CS), fission yields (FY),
    energy spectra (DE), and any other EXFOR quantity.

    Attributes
    ----------
    entry, subentry, quantity, citation, reaction, facility, method, detector, sample
        As in :class:`~kika.exfor.ExforEntry`.
    quantity_description : str
        Human-readable description of the quantity. Empty, it is taken from
        ``EXFOR_QUANTITY_CODES``.
    independent_vars : List[str]
        Names of independent variables (e.g., ["energy"], ["energy", "angle"])
    dependent_var : str
        Name of dependent variable (e.g., "cross_section", "yield")
    units : Dict[str, str]
        Unit of each variable. In objects made by the database's general loader,
        energies are in MeV and cross-sections in b, b/sr or b/MeV, the error
        following its value; other quantities (transmission, yields, angles) keep
        the unit string and values of the source.

    Examples
    --------
    >>> # Load any experiment type from database
    >>> from kika.exfor import X4ProDatabase
    >>> db = X4ProDatabase()
    >>> exp = db.load_experiment_general("10571002")
    >>> print(exp.quantity)
    CS
    >>> print(exp.data.head())
         energy     value     error
    0     1.0e6  0.001234  0.000012
    """

    quantity_description: str = ""
    independent_vars: List[str] = field(default_factory=list)
    dependent_var: str = "value"
    units: Dict[str, str] = field(default_factory=dict)
    _data: Optional[pd.DataFrame] = field(default=None, repr=False)

    def __post_init__(self):
        """Set quantity description from constants if not provided."""
        if not self.quantity_description:
            self.quantity_description = EXFOR_QUANTITY_CODES.get(
                self.quantity, f"Unknown quantity: {self.quantity}"
            )
        if self._data is None:
            self._data = pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the experiment data as DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the data with columns for each variable
        """
        return self._data.copy()

    @property
    def num_data_points(self) -> int:
        """
        Return the number of data points.

        Returns
        -------
        int
            Number of rows in the data
        """
        return len(self._data)

    def get_unique_values(self, variable: str) -> np.ndarray:
        """
        Get unique values for an independent variable.

        Parameters
        ----------
        variable : str
            Name of the variable (e.g., "energy", "angle")

        Returns
        -------
        np.ndarray
            Sorted array of unique values

        Raises
        ------
        ValueError
            If variable is not in the data columns
        """
        if variable not in self._data.columns:
            raise ValueError(
                f"Variable '{variable}' not in data. Available: {list(self._data.columns)}"
            )
        return np.sort(self._data[variable].dropna().unique())

    def filter(self, **kwargs) -> pd.DataFrame:
        """
        Filter data by variable values.

        Parameters
        ----------
        **kwargs
            Variable name and value/range pairs.
            - Single value: filter to that value (with 1% tolerance)
            - Tuple (min, max): filter to range

        Returns
        -------
        pd.DataFrame
            Filtered data

        Examples
        --------
        >>> exp.filter(energy=1.5)  # Data at E=1.5 MeV
        >>> exp.filter(energy=(1.0, 3.0))  # Data in range
        >>> exp.filter(energy=(1.0, 3.0), angle=90.0)  # Combined filter
        """
        df = self._data.copy()
        for var, value in kwargs.items():
            if var not in df.columns:
                continue
            if isinstance(value, tuple):
                df = df[(df[var] >= value[0]) & (df[var] <= value[1])]
            else:
                tol = abs(value) * 0.01 if value != 0 else 1e-6
                df = df[(df[var] >= value - tol) & (df[var] <= value + tol)]
        return df

    def to_dataframe(self, **filters) -> pd.DataFrame:
        """
        Return data as DataFrame, optionally filtered.

        Parameters
        ----------
        **filters
            Optional filters to apply (see filter() method)

        Returns
        -------
        pd.DataFrame
            Data, optionally filtered
        """
        if filters:
            return self.filter(**filters)
        return self.data

    def summary(self) -> str:
        """
        Return a formatted summary of the experiment.

        Returns
        -------
        str
            Multi-line summary string
        """
        lines = [
            f"EXFOR Experiment: {self.entry}.{self.subentry}",
            f"  Label:       {self.label}",
            f"  Quantity:    {self.quantity} ({self.quantity_description})",
            f"  Target:      {self.target} (ZAID: {self.zaid})",
            f"  Projectile:  {self.projectile}",
            f"  Data points: {self.num_data_points}",
            f"  Variables:   {list(self._data.columns)}",
            f"  Units:       {self.units}",
            f"  URL:         {self.exfor_url}",
        ]
        if self.independent_vars:
            for var in self.independent_vars:
                if var in self._data.columns:
                    vals = self._data[var].dropna()
                    if len(vals) > 0:
                        lines.append(f"  {var} range: {vals.min():.4g} - {vals.max():.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return summary representation."""
        return self.summary()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all metadata and data
        """
        result = super().to_dict()
        result["quantity_description"] = self.quantity_description
        result["independent_vars"] = self.independent_vars
        result["dependent_var"] = self.dependent_var
        result["units"] = self.units
        result["data"] = self._data.to_dict(orient="records")
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExforExperiment":
        """
        Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with experiment data

        Returns
        -------
        ExforExperiment
            New instance
        """
        fields = cls._parse_base_fields(data)
        fields["quantity_description"] = data.get("quantity_description", "")
        fields["independent_vars"] = data.get("independent_vars", [])
        fields["dependent_var"] = data.get("dependent_var", "value")
        fields["units"] = data.get("units", {})

        data_records = data.get("data", [])
        if data_records:
            fields["_data"] = pd.DataFrame(data_records)
        else:
            fields["_data"] = pd.DataFrame()

        return cls(**fields)

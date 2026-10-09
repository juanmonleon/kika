from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from kika.ace.classes.angular_distribution.types import AngularDistributionType
from kika._utils import create_repr_section


@dataclass
class AngularDistribution:
    """Base class for angular distributions."""
    mt: int = None  # MT number for this reaction
    _energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energy grid (view of xss_data)
    distribution_type: AngularDistributionType = AngularDistributionType.ISOTROPIC
    
    def __post_init__(self):
        """Initialize after creation, ensuring values are properly stored."""
        # Store the MT number as a plain int
        if self.mt is not None:
            self.mt = int(self.mt)
    
    @property
    def energies(self) -> List[float]:
        """Get energy values as floats."""
        return [float(e) for e in self._energies]
    
    @property
    def is_isotropic(self) -> bool:
        """Check if the distribution is isotropic."""
        return self.distribution_type == AngularDistributionType.ISOTROPIC

    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert angular distribution to a pandas DataFrame.
        
        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate when interpolating, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid (True) or return original points (False)
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'energy', 'cosine', and 'pdf' columns
            Returns None if pandas is not available
        """
        try:
            import pandas as pd
            
            # For isotropic distribution, return simple DataFrame
            if interpolate:
                cosines = np.linspace(-1, 1, num_points)
                return pd.DataFrame({
                    'energy': np.full_like(cosines, energy, dtype=float),
                    'cosine': cosines,
                    'pdf': np.ones_like(cosines) * 0.5
                })
            else:
                return pd.DataFrame({
                    'energy': [energy, energy],
                    'cosine': [-1.0, 1.0],
                    'pdf': [0.5, 0.5]
                })
            
        except ImportError:
            return None

    def __repr__(self) -> str:
        """
        Returns a user-friendly string representation focusing on raw ACE data.
        
        Returns
        -------
        str
            Formatted string representation showing the raw ACE data
        """
        header_width = 85
        header = "=" * header_width + "\n"
        mt_value = int(self.mt)
        header += f"{'Angular Distribution for MT=' + str(mt_value):^{header_width}}\n"
        header += f"{self.distribution_type.name:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description focused on ACE data
        description = (
            f"This object contains angular distribution data for reaction MT={mt_value}.\n"
            f"Distribution Type: {self.distribution_type.name}\n\n"
        )
        
        # Energy grid information from raw ACE data
        if hasattr(self, "energies") and self.energies:
            description += "ENERGY GRID STRUCTURE:\n"
            description += "-" * header_width + "\n"
            description += f"Number of energy points: {len(self.energies)}\n"
            
            # Show the first few energy points
            max_display = min(5, len(self.energies))
            description += f"First {max_display} energy points (MeV):\n"
            for i in range(max_display):
                e_value = self.energies[i]
                description += f"  Energy[{i}] = {e_value:.6g}\n"
            
            # If there are more than max_display points, show the last one too
            if len(self.energies) > max_display:
                e_value = self.energies[-1]
                description += f"  ...\n"
                description += f"  Energy[{len(self.energies)-1}] = {e_value:.6g}\n"
            
            description += "\n"
        else:
            description += "No energy grid data available (isotropic at all energies)\n\n"
        
        # Add a note that this is the base representation
        description += (
            "NOTE: This is the base representation. For detailed distribution data,\n"
            "access the derived class attributes directly.\n\n"
        )
        
        # Add property descriptions (only public attributes)
        properties = {
            ".energies": "List of incident energy points (MeV)",
            ".mt": "MT number of the reaction",
            ".distribution_type": "Type of angular distribution",
            ".is_isotropic": "Boolean indicating if distribution is isotropic"
        }
        
        property_col_width = 35
        properties_section = create_repr_section(
            "Public Properties:", 
            properties, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Create a section for available methods but keep it minimal
        methods = {
            ".to_dataframe(energy, interpolate=False)": "Get distribution at a specific energy as DataFrame",
        }

        methods_section = create_repr_section(
            "Methods to Access Data:",
            methods,
            total_width=header_width,
            method_col_width=property_col_width
        )

        return header + description + properties_section + "\n" + methods_section

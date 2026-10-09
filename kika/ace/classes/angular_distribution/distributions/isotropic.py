from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.types import AngularDistributionType
from kika._utils import create_repr_section

@dataclass
class IsotropicAngularDistribution(AngularDistribution):
    """Angular distribution for isotropic scattering."""
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.ISOTROPIC
    
    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert isotropic angular distribution to a pandas DataFrame.
        
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
            
        # For isotropic distribution, the PDF is constant (0.5) for all cosines
        if interpolate:
            cosines = np.linspace(-1, 1, num_points)
            return pd.DataFrame({
                'energy': np.full_like(cosines, energy, dtype=float),
                'cosine': cosines,
                'pdf': np.full_like(cosines, 0.5, dtype=float)
            })
        else:
            # Just return points at the ends of the cosine range for efficiency
            return pd.DataFrame({
                'energy': [energy, energy],
                'cosine': [-1.0, 1.0],
                'pdf': [0.5, 0.5]
            })

    def __repr__(self) -> str:

        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Isotropic Angular Distribution Details':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        description = (
            "This object represents isotropic angular scattering, where the cosine of the\n"
            "scattering angle is uniformly distributed between -1 and 1. This means the\n"
            "probability density is constant at 0.5 across the entire range.\n\n"
            "Data Structure Overview:\n"
            "- The ACE file may indicate isotropic scattering in two ways:\n"
            "  * By setting the LOCB value to 0 in the locator table\n"
            "  * By storing a distribution table with NE=0 (number of energy points)\n"
            "- No actual distribution data is stored for isotropic scattering\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3
        
        info_table = "Data Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # MT number
        mt_value = int(self.mt)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "MT Number", f"{mt_value}", width1=property_col_width, width2=value_col_width)
        
        # Distribution properties
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Type", "Isotropic (uniform)",
            width1=property_col_width, width2=value_col_width)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "PDF Function", "P(μ) = 0.5 for all μ ∈ [-1, 1]",
            width1=property_col_width, width2=value_col_width)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Sampling Method", "μ = 2*ξ - 1 where ξ ∈ [0, 1] is random",
            width1=property_col_width, width2=value_col_width)
        
        # Energy grid information
        if self.energies:
            num_energies = len(self.energies)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Energy Points", num_energies,
                width1=property_col_width, width2=value_col_width)
            
            min_energy = self.energies[0]  # Now directly a float
            max_energy = self.energies[-1]  # Now directly a float
            energy_range = f"{min_energy:.6g} - {max_energy:.6g} MeV"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Range", energy_range,
                width1=property_col_width, width2=value_col_width)
        else:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Dependence", "None (same for all energies)",
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Raw data properties section
        properties = {
            ".mt": "MT number of the reaction (int)",
            ".energies": "List of incident energy points as float values (List[float])"
        }
        
        properties_section = create_repr_section(
            "Raw Data Properties (Direct from ACE file):", 
            properties, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Methods section
        methods = {
            ".sample_mu(energy, random_value)": "Sample a cosine μ = 2*random_value - 1",
            ".to_dataframe(energy, num_points)": "Convert to a pandas DataFrame with uniform probability",
        }

        methods_section = create_repr_section(
            "Calculation Methods:",
            methods,
            total_width=header_width,
            method_col_width=property_col_width
        )

        # Add example for directly accessing property
        example = (
            "Example:\n"
            "--------\n"
            "# Access the MT number\n"
            "mt_value = int(distribution.mt)\n\n"
            "# Sample a cosine for any energy (will always be uniform)\n"
            "mu = distribution.sample_mu(energy=1.0, random_value=0.5)  # Returns 0.0\n"
        )

        return header + description + info_table + properties_section + "\n" + methods_section + "\n" + example
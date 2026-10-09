from typing import List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.types import AngularDistributionType
from kika._utils import create_repr_section


@dataclass
class TabulatedAngularDistribution(AngularDistribution):
    """Angular distribution for tabulated scattering."""
    interpolation: List[int] = field(default_factory=list)  # Interpolation flag for each energy
    _cosine_grid: List[np.ndarray] = field(default_factory=list)  # Cosine grid for each energy (views of xss_data)
    _pdf: List[np.ndarray] = field(default_factory=list)  # PDF for each energy (views of xss_data)
    _cdf: List[np.ndarray] = field(default_factory=list)  # CDF for each energy (views of xss_data)
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.TABULATED
    
    @property
    def cosine_grid(self) -> List[List[float]]:
        """Get cosine grid values as lists of floats."""
        return [[float(c) for c in cosine_list] for cosine_list in self._cosine_grid]
    
    @property
    def pdf(self) -> List[List[float]]:
        """Get PDF values as lists of floats."""
        return [[float(p) for p in pdf_list] for pdf_list in self._pdf]
    
    @property
    def cdf(self) -> List[List[float]]:
        """Get CDF values as lists of floats."""
        return [[float(c) for c in cdf_list] for cdf_list in self._cdf]
    
    
    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert tabulated angular distribution to a pandas DataFrame.
        
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
            DataFrame with 'energy', 'cosine', 'pdf', and potentially 'cdf' columns
            Returns None if pandas is not available
        """
            
        # If no energies in this distribution, return isotropic for all directions
        if len(self._energies) == 0:
            # For specific energy, return isotropic distribution
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
                    'pdf': [0.5, 0.5],
                    'cdf': [0.0, 1.0]
                })
        
        # If energy is outside our range, return uniform distribution
        if energy < self._energies[0] or energy > self._energies[-1]:
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
                    'pdf': [0.5, 0.5],
                    'cdf': [0.0, 1.0]
                })
        
        # Find bounding energy indices
        energy_values = self.energies
        idx = np.searchsorted(energy_values, energy)
        
        # Get appropriate PDF/CDF values based on energy
        if idx == 0:
            cosine_values = self.cosine_grid[0]
            pdf_values = self.pdf[0]
            cdf_values = self.cdf[0]
        elif idx >= len(energy_values):
            cosine_values = self.cosine_grid[-1]
            pdf_values = self.pdf[-1]
            cdf_values = self.cdf[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            # Get values at the two bounding energies
            cosines_low = self.cosine_grid[idx-1]
            cosines_high = self.cosine_grid[idx]
            pdf_low = self.pdf[idx-1]
            pdf_high = self.pdf[idx]
            cdf_low = self.cdf[idx-1]
            cdf_high = self.cdf[idx]
            
            # If cosine grids are different between the two energy points,
            # interpolate onto a common grid to ensure consistent lengths
            if len(cosines_low) != len(cosines_high):
                # For simplicity, use a common grid based on the lower energy point
                cosine_values = cosines_low
                # Interpolate PDF and CDF from the high energy point onto the lower energy grid
                pdf_high_interp = np.interp(
                    cosine_values, cosines_high, pdf_high, 
                    left=pdf_high[0], right=pdf_high[-1]
                )
                cdf_high_interp = np.interp(
                    cosine_values, cosines_high, cdf_high, 
                    left=0.0, right=1.0
                )
                # Linear interpolation between the two energy points
                pdf_values = [(1-frac)*pl + frac*ph for pl, ph in zip(pdf_low, pdf_high_interp)]
                cdf_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cdf_low, cdf_high_interp)]
            else:
                # Simple linear interpolation when grids are the same
                cosine_values = cosines_low  # Use the lower energy cosine grid
                pdf_values = [(1-frac)*pl + frac*ph for pl, ph in zip(pdf_low, pdf_high)]
                cdf_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cdf_low, cdf_high)]
        
        if not interpolate:
            # Convert values to numpy arrays to ensure consistency and check lengths
            cosine_np = np.array(cosine_values)
            pdf_np = np.array(pdf_values)
            cdf_np = np.array(cdf_values)
            
            # Make sure all arrays have the same length - if not, use the shortest length
            min_length = min(len(cosine_np), len(pdf_np), len(cdf_np))
            if min_length < len(cosine_np):
                cosine_np = cosine_np[:min_length]
            if min_length < len(pdf_np):
                pdf_np = pdf_np[:min_length]
            if min_length < len(cdf_np):
                cdf_np = cdf_np[:min_length]
            
            # Create DataFrame with consistent array lengths
            df = pd.DataFrame({
                'energy': np.full_like(cosine_np, energy, dtype=float),
                'cosine': cosine_np,
                'pdf': pdf_np,
                'cdf': cdf_np
            })
            return df
        
        # If interpolation requested, create a regular grid and interpolate
        cosines = np.linspace(-1, 1, num_points)
        
        # Get interpolation type for this energy
        interp_type = None
        if idx < len(self.interpolation):
            interp_type = self.interpolation[idx]
        elif self.interpolation:
            interp_type = self.interpolation[-1]

        # Generate PDF values based on interpolation type
        if interpolate:
            pdf_values_interp = np.zeros_like(cosines)

            if interp_type == 0:  # Histogram (JJ=0 per ACE Table 20)
                # For histogram, each point gets the value of the left edge of its bin
                for i, mu in enumerate(cosines):
                    bin_idx = np.searchsorted(cosine_values, mu)
                    if bin_idx == 0:
                        pdf_values_interp[i] = pdf_values[0]
                    else:
                        pdf_values_interp[i] = pdf_values[bin_idx-1]
            else:  # Linear-linear (JJ=1 per ACE Table 20)
                # Use linear interpolation between tabulated points
                pdf_values_interp = np.interp(cosines, cosine_values, pdf_values, left=0.0, right=0.0)
        
        return pd.DataFrame({
            'energy': np.full_like(cosines, energy, dtype=float),
            'cosine': cosines,
            'pdf': pdf_values_interp
        })

    def __repr__(self) -> str:
        """
        Returns a user-friendly string representation with detailed structure information.
        
        Returns
        -------
        str
            Formatted string representation showing the distribution structure
        """
        header_width = 85
        header = "=" * header_width + "\n"
        mt_value = int(self.mt)
        header += f"{'Tabulated Angular Distribution for MT=' + str(mt_value):^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Detailed description of tabulated format
        description = (
            f"This object contains tabulated angular distribution data for reaction MT={mt_value}.\n\n"
            f"DISTRIBUTION STRUCTURE:\n"
            f"The tabulated format stores the angular distribution as explicit probability\n"
            f"density functions (PDFs) and cumulative distribution functions (CDFs) for a set of\n"
            f"incident energy points. The data is organized as follows:\n\n"
            f"1. Energy Grid: A set of incident neutron energies (E₁, E₂, ..., Eₙ)\n"
            f"2. For each energy point, the distribution includes:\n"
            f"   a. Interpolation flag (0=histogram, 1=linear-linear per ACE Table 20)\n"
            f"   b. Set of cosine values (μ) ranging from -1 to 1\n"
            f"   c. PDF values (probability density function) for each cosine\n"
            f"   d. CDF values (cumulative distribution function) for each cosine\n\n"
            f"INTERPOLATION METHODS:\n"
            f"- Between incident energy points: Linear interpolation of PDF values\n"
            f"- Within a cosine grid (μ values):\n"
            f"  * Histogram (flag=0): PDF value is constant within each cosine bin\n"
            f"  * Linear-linear (flag=1): Linear interpolation between cosine points\n\n"
            f"CALCULATION EXAMPLE:\n"
            f"To calculate the PDF at incident energy E and scattering cosine μ:\n"
            f"1. Find bounding energy points: E₁ ≤ E ≤ E₂\n"
            f"2. Calculate interpolation factor: f = (E - E₁)/(E₂ - E₁)\n"
            f"3. Get PDFs at μ for both energies: PDF₁(μ), PDF₂(μ)\n"
            f"4. Interpolate: PDF(E,μ) = (1-f) × PDF₁(μ) + f × PDF₂(μ)\n\n"
        )
        
        # Energy grid information
        if hasattr(self, "energies") and self.energies:
            description += "ENERGY GRID:\n"
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
        
        # Interpolation information
        if hasattr(self, "interpolation") and self.interpolation:
            interp_types = set(self.interpolation)
            interp_desc = {
                0: "Histogram",
                1: "Linear-Linear"
            }
            interp_str = ", ".join(interp_desc.get(i, f"Type {i}") for i in interp_types)
            
            description += "INTERPOLATION FLAGS:\n"
            description += "-" * header_width + "\n"
            description += f"Interpolation type(s) used: {interp_str}\n\n"
        
        # Information about cosine grid structure at first energy
        if hasattr(self, "cosine_grid") and self.cosine_grid and len(self.cosine_grid) > 0:
            first_cosines = self.cosine_grid[0]
            num_points = len(first_cosines)
            
            description += "COSINE GRID STRUCTURE:\n"
            description += "-" * header_width + "\n"
            description += f"Number of points in first energy's cosine grid: {num_points}\n"
            if num_points > 0:
                description += f"Cosine range: [{first_cosines[0]:.4f}, {first_cosines[-1]:.4f}]\n\n"
        
        # Add property descriptions (only public attributes)
        properties = {
            ".energies": "List of incident energy points (MeV)",
            ".interpolation": "List of interpolation flags for each energy",
            ".cosine_grid": "List of cosine grids for each energy",
            ".pdf": "List of PDF values for each energy",
            ".cdf": "List of CDF values for each energy"
        }
        
        property_col_width = 35
        properties_section = create_repr_section(
            "Public Properties:", 
            properties, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Create a section for available methods
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

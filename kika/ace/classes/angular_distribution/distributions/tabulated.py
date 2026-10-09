from typing import List, Optional
import numpy as np
from kika.ace.classes.energy_distribution import tabular_math
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

        Between two incident energies the two tables are mixed with weights
        ``1-f`` and ``f`` (the table is chosen at random, as MCNP samples it), on
        the union of their cosine grids. Outside the tabulated range the end
        table is used. JJ follows ENDF (1 histogram, 2 lin-lin), which NJOY writes;
        the 0 of manual Table 20 is read as histogram too.

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
            DataFrame with 'energy', 'cosine', 'pdf', and (on the tabulated points) 'cdf' columns
        """
        if len(self._energies) == 0:
            cosines = np.linspace(-1, 1, num_points) if interpolate else np.array([-1.0, 1.0])
            df = pd.DataFrame({'energy': np.full_like(cosines, energy, dtype=float),
                               'cosine': cosines, 'pdf': np.full_like(cosines, 0.5)})
            if not interpolate:
                df['cdf'] = [0.0, 1.0]
            return df

        i, frac = tabular_math.bracket(self._energies, energy)
        members = [(i, 1.0)] if frac == 0.0 else [(i, 1.0 - frac), (i + 1, frac)]

        def table(k):
            jj = self.interpolation[k] if k < len(self.interpolation) else 2
            return (np.asarray(self._cosine_grid[k], dtype=float),
                    np.asarray(self._pdf[k], dtype=float),
                    np.asarray(self._cdf[k], dtype=float), 1 if jj in (0, 1) else 2)

        if interpolate:
            cosines = np.linspace(-1, 1, num_points)
        else:
            cosines = np.unique(np.concatenate([table(k)[0] for k, _ in members]))
        pdf = np.zeros_like(cosines)
        cdf = np.zeros_like(cosines)
        for k, w in members:
            x, p, c, jj = table(k)
            pdf += w * tabular_math.pdf_on_grid(x, p, jj, cosines)
            cdf += w * tabular_math.cdf_on_grid(x, p, c, jj, cosines)
        df = pd.DataFrame({'energy': np.full_like(cosines, energy, dtype=float),
                           'cosine': cosines, 'pdf': pdf})
        if not interpolate:
            df['cdf'] = cdf
        return df

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

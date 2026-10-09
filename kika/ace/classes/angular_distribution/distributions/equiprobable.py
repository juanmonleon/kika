from typing import List, Optional, Tuple
import numpy as np
from kika.ace.classes.energy_distribution import tabular_math
import pandas as pd
from dataclasses import dataclass, field
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.types import AngularDistributionType
from kika._utils import create_repr_section


@dataclass
class EquiprobableAngularDistribution(AngularDistribution):
    """Angular distribution for 32 equiprobable bin scattering."""
    _cosine_bins: List[np.ndarray] = field(default_factory=list)  # 33 cosines for each energy (views of xss_data)
    
    def __post_init__(self):
        super().__post_init__()
        self.distribution_type = AngularDistributionType.EQUIPROBABLE
    
    @property
    def cosine_bins(self) -> List[List[float]]:
        """Get cosine bin values as lists of floats."""
        return [[float(c) for c in cosine_list] for cosine_list in self._cosine_bins]

    def to_dataframe(self, energy: float, num_points: int = 100, interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert equiprobable bin distribution to a pandas DataFrame.

        Each table is 32 bins of probability 1/32. Between two incident energies
        the two tables are mixed with weights ``1-f`` and ``f`` (the table is
        chosen at random, as MCNP samples it), on the union of their bin edges.
        Outside the tabulated range the end table is used.

        Parameters
        ----------
        energy : float
            Incident energy to evaluate the distribution at
        num_points : int, optional
            Number of angular points to generate when interpolating, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid (True) or return the bins (False)

        Returns
        -------
        pandas.DataFrame or None
            With ``interpolate=False``: one row per bin, 'cosine' at its centre,
            its density in 'pdf' and its edges in 'bin_low'/'bin_high'.
            Otherwise 'energy', 'cosine' and 'pdf' on a regular grid.
        """
        if len(self._energies) == 0:
            cosines = np.linspace(-1, 1, num_points) if interpolate else np.array([-1.0, 1.0])
            return pd.DataFrame({'energy': np.full_like(cosines, energy, dtype=float),
                                 'cosine': cosines, 'pdf': np.full_like(cosines, 0.5)})

        i, frac = tabular_math.bracket(self._energies, energy)
        members = [(i, 1.0)] if frac == 0.0 else [(i, 1.0 - frac), (i + 1, frac)]

        def density(k, grid):
            edges = np.asarray(self._cosine_bins[k], dtype=float)
            widths = np.diff(edges)
            dens = np.divide(1.0 / (len(edges) - 1), widths, out=np.zeros_like(widths), where=widths > 0)
            return tabular_math.pdf_on_grid(edges, np.append(dens, 0.0), 1, grid)

        if interpolate:
            cosines = np.linspace(-1, 1, num_points)
            pdf = sum(w * density(k, cosines) for k, w in members)
            return pd.DataFrame({'energy': np.full_like(cosines, energy, dtype=float),
                                 'cosine': cosines, 'pdf': pdf})

        edges = np.unique(np.concatenate([np.asarray(self._cosine_bins[k], dtype=float) for k, _ in members]))
        centres = 0.5 * (edges[:-1] + edges[1:])
        pdf = sum(w * density(k, centres) for k, w in members)
        return pd.DataFrame({'energy': np.full_like(centres, energy, dtype=float),
                             'cosine': centres, 'pdf': pdf,
                             'bin_low': edges[:-1], 'bin_high': edges[1:]})

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
        header += f"{'Equiprobable Angular Distribution for MT=' + str(mt_value):^{header_width}}\n"
        header += f"{self.distribution_type.name:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Detailed description of equiprobable format
        description = (
            f"This object contains equiprobable bin angular distribution data for reaction MT={mt_value}.\n\n"
            f"DISTRIBUTION STRUCTURE:\n"
            f"The equiprobable format divides the cosine range [-1,1] into 32 bins such that each\n"
            f"bin has the same probability (1/32 = 0.03125). This creates a non-uniform bin width\n"
            f"distribution where narrower bins indicate higher probability density regions.\n\n"
            f"The data is organized as follows:\n\n"
            f"1. Energy Grid: A set of incident neutron energies (E₁, E₂, ..., Eₙ)\n"
            f"2. For each energy point, the distribution includes:\n"
            f"   - 33 cosine values defining the boundaries of 32 equal-probability bins\n"
            f"   - The first value is always -1 and the last is +1\n\n"
            f"INTERPOLATION METHODS:\n"
            f"- Between incident energy points: Linear interpolation of bin boundaries\n"
            f"- Within each bin: Uniform probability (constant PDF within each bin)\n\n"
            f"CALCULATION EXAMPLE:\n"
            f"To calculate the PDF at incident energy E and scattering cosine μ:\n"
            f"1. Find bounding energy points: E₁ ≤ E ≤ E₂\n"
            f"2. Calculate interpolation factor: f = (E - E₁)/(E₂ - E₁)\n"
            f"3. Interpolate bin boundaries between energies\n"
            f"4. Find which bin contains μ: bin where μᵢ ≤ μ < μᵢ₊₁\n"
            f"5. Calculate PDF = 0.03125/(μᵢ₊₁ - μᵢ)   (probability/bin width)\n\n"
            f"SAMPLING ALGORITHM:\n"
            f"1. Select a random bin (1-32) with equal probability\n"
            f"2. Draw a random value uniformly within that bin's cosine range\n\n"
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
        
        # Information about cosine bin structure at first energy
        if hasattr(self, "cosine_bins") and self.cosine_bins and len(self.cosine_bins) > 0:
            first_bins = self.cosine_bins[0]
            num_bins = len(first_bins) - 1  # Subtract 1 because there are n+1 boundaries for n bins
            
            description += "COSINE BIN STRUCTURE:\n"
            description += "-" * header_width + "\n"
            description += f"Number of equiprobable bins: {num_bins}\n"
            description += f"Number of bin boundaries: {len(first_bins)}\n"
            
            # Show bin width variation
            if len(first_bins) > 2:
                # Calculate bin widths for the first energy
                bin_widths = [first_bins[i+1] - first_bins[i] for i in range(len(first_bins)-1)]
                description += f"Bin width statistics (showing non-uniformity):\n"
                description += f"  Min width: {min(bin_widths):.6g}\n"
                description += f"  Max width: {max(bin_widths):.6g}\n"
                description += f"  Mean width: {sum(bin_widths)/len(bin_widths):.6g}\n"
                
                # Show example bins
                description += "\nExample bins from first energy point:\n"
                # Show first bin
                description += f"  Bin 1:  [{first_bins[0]:.6f}, {first_bins[1]:.6f}], "
                description += f"width = {first_bins[1]-first_bins[0]:.6f}, "
                description += f"PDF = {0.03125/(first_bins[1]-first_bins[0]):.6f}\n"
                
                # Show a middle bin
                mid = num_bins // 2
                description += f"  Bin {mid}: [{first_bins[mid-1]:.6f}, {first_bins[mid]:.6f}], "
                description += f"width = {first_bins[mid]-first_bins[mid-1]:.6f}, "
                description += f"PDF = {0.03125/(first_bins[mid]-first_bins[mid-1]):.6f}\n"
                
                # Show last bin
                description += f"  Bin {num_bins}: [{first_bins[-2]:.6f}, {first_bins[-1]:.6f}], "
                description += f"width = {first_bins[-1]-first_bins[-2]:.6f}, "
                description += f"PDF = {0.03125/(first_bins[-1]-first_bins[-2]):.6f}\n"
            
            description += "\n"
        
        # Add property descriptions (only public attributes)
        properties = {
            ".energies": "List of incident energy points (MeV)",
            ".cosine_bins": "List of cosine bin boundaries for each energy"
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
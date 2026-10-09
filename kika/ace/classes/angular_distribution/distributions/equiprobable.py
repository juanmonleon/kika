from typing import List, Optional, Tuple
import numpy as np
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
            DataFrame with 'energy', 'cosine', and 'pdf' columns,
            optionally with bin boundary columns if not interpolating
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
                    'pdf': [0.5, 0.5]
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
                    'pdf': [0.5, 0.5]
                })
        
        # Find bounding energy indices
        energy_values = self.energies
        idx = np.searchsorted(energy_values, energy)
        
        # Get appropriate cosine bins based on energy
        if idx == 0:
            bin_values = self.cosine_bins[0]
        elif idx >= len(energy_values):
            bin_values = self.cosine_bins[-1]
        else:
            # Interpolate between energy points
            e_low = energy_values[idx-1]
            e_high = energy_values[idx]
            frac = (energy - e_low) / (e_high - e_low)
            
            # Get the cosine values at the two bounding energies
            cosines_low = self.cosine_bins[idx-1]
            cosines_high = self.cosine_bins[idx]
            
            # Interpolate cosine bin boundaries
            bin_values = [(1-frac)*cl + frac*ch for cl, ch in zip(cosines_low, cosines_high)]
        
        if not interpolate:
            # Return the actual bin boundaries and their probabilities
            # For equiprobable bins, each bin has probability 1/32
            prob_per_bin = 1.0 / 32.0
            
            # Calculate probability density for each bin (constant within bin)
            pdf_values = []
            bin_centers = []
            bin_lows = []
            bin_highs = []
            energy_values = []
            
            for i in range(len(bin_values) - 1):
                bin_width = bin_values[i+1] - bin_values[i]
                if bin_width > 0:
                    pdf = prob_per_bin / bin_width
                else:
                    pdf = 0.0
                
                # Use bin center as the cosine value
                bin_center = (bin_values[i] + bin_values[i+1]) / 2
                
                bin_centers.append(bin_center)
                pdf_values.append(pdf)
                bin_lows.append(bin_values[i])
                bin_highs.append(bin_values[i+1])
                energy_values.append(energy)
            
            # Verify all arrays have the same length
            array_lengths = [len(bin_centers), len(pdf_values), len(bin_lows), len(bin_highs), len(energy_values)]
            if len(set(array_lengths)) > 1:
                # If lengths don't match, truncate to the shortest length
                min_length = min(array_lengths)
                bin_centers = bin_centers[:min_length]
                pdf_values = pdf_values[:min_length]
                bin_lows = bin_lows[:min_length]
                bin_highs = bin_highs[:min_length]
                energy_values = energy_values[:min_length]
            
            return pd.DataFrame({
                'energy': energy_values,
                'cosine': bin_centers,
                'pdf': pdf_values,
                'bin_low': bin_lows,
                'bin_high': bin_highs
            })
        
        # If interpolation requested, use the existing code
        # Generate a fine cosine grid
        cosines = np.linspace(-1, 1, num_points)
        
        # Calculate PDF (should be constant within each bin)
        # For a 32-bin equiprobable distribution, each bin has probability of 1/32
        prob_per_bin = 1.0 / 32.0
        
        # Initialize PDF array
        pdf_values = np.zeros_like(cosines)
        
        # Assign PDF values based on bin membership
        for i, mu in enumerate(cosines):
            # Find which bin the cosine falls into
            bin_idx = 0
            while bin_idx < 32 and bin_values[bin_idx] <= mu:
                bin_idx += 1
            
            if bin_idx > 0 and bin_idx <= 32:
                bin_width = bin_values[bin_idx] - bin_values[bin_idx-1]
                if bin_width > 0:
                    pdf_values[i] = prob_per_bin / bin_width
        
        return pd.DataFrame({
            'energy': np.full_like(cosines, energy, dtype=float),
            'cosine': cosines,
            'pdf': pdf_values
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
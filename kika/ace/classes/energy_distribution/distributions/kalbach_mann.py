# Law 44: Kalbach-Mann distribution

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class KalbachMannDistribution(EnergyDistribution):
    """
    Law 44: Kalbach-Mann Correlated Energy-Angle Distribution.
    
    From ENDF-6 MF=6 LAW=1, LANG=2, this represents a correlated energy-angle
    distribution using the Kalbach-87 formalism.
    
    Data format (Table 43 and 44):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Location of distributions
    
    For each incident energy:
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - R(l): Precompound fraction r
    - A(l): Angular distribution slope value a
    
    Angular distribution is given by:
    p(μ, E_in, E_out) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]
    """
    law: int = 44
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy-angle distribution
    # Each entry is a dictionary containing the distribution data
    distributions: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data with keys:
            - 'intt': Interpolation type
            - 'n_discrete': Number of discrete lines
            - 'n_points': Number of points
            - 'e_out': Outgoing energy grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            - 'r': Precompound fraction
            - 'a': Angular distribution slope
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.distributions):
            return self.distributions[energy_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated distribution for a specific incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the interpolated distribution data
        """
        # Incident energies as floats for comparison
        incident_energy_values = [float(e) for e in self.incident_energies]
        
        # Find the bracketing incident energies
        if not incident_energy_values or incident_energy <= incident_energy_values[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= incident_energy_values[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.distributions) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(incident_energy_values, incident_energy, side='right') - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.get_distribution(idx)
        dist_high = self.get_distribution(idx + 1)
        
        if not dist_low or not dist_high:
            return dist_low if dist_low else dist_high
            
        # Interpolate the distributions based on the incident energy
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        
        # Determine the interpolation scheme from the regions
        # For simplicity, assume linear-linear interpolation
        # In a full implementation, we would determine this from the NBT and INT arrays
        
        # Interpolation fraction
        frac = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Create common grid for interpolation (use outgoing energy points from low distribution)
        e_out = dist_low['e_out']
        
        # Interpolate PDF values
        pdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['pdf'])
        pdf_interp = (1.0 - frac) * np.array(dist_low['pdf']) + frac * pdf_high_interp
        
        # Interpolate CDF values
        cdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['cdf'])
        cdf_interp = (1.0 - frac) * np.array(dist_low['cdf']) + frac * cdf_high_interp
        
        # Interpolate r values (precompound fraction)
        r_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['r'])
        r_interp = (1.0 - frac) * np.array(dist_low['r']) + frac * r_high_interp
        
        # Interpolate a values (angular distribution slope)
        a_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['a'])
        a_interp = (1.0 - frac) * np.array(dist_low['a']) + frac * a_high_interp
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out),
            'e_out': e_out,
            'pdf': pdf_interp.tolist(),
            'cdf': cdf_interp.tolist(),
            'r': r_interp.tolist(),
            'a': a_interp.tolist()
        }
        
        return interp_dist

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the KalbachMannDistribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Kalbach-Mann Energy-Angle Distribution (Law 44)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents a correlated energy-angle distribution using the\n"
            "Kalbach-Mann formalism. The angular part of the distribution is given by:\n"
            "p(μ, E_in, E_out) = (1/2)*(a/sinh(a))*[cosh(aμ) + r*sinh(aμ)]\n\n"
            "where 'a' is the angular distribution slope parameter and 'r' is the precompound\n"
            "fraction. Both parameters depend on the outgoing energy.\n\n"
            "This distribution is commonly used for reactions involving pre-equilibrium effects,\n"
            "like (n,p), (n,α), and other particle emission reactions.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Distribution Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Law Number", self.law, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", self.n_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", self.n_interp_regions, 
            width1=property_col_width, width2=value_col_width)
        
        # If we have incident energies, show the range
        incident_energy_values = [float(e) for e in self.incident_energies]
        if incident_energy_values:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", 
                f"{min(incident_energy_values):.6g} - {max(incident_energy_values):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # Information about distributions
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Distributions", len(self.distributions), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_distribution(energy_idx)": "Get the distribution for a specific incident energy index",
            ".get_interpolated_distribution(incident_energy)": "Get an interpolated distribution for a specific incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add example section
        example = (
            "Example:\n"
            "--------\n"
            "# Get the interpolated distribution at 5 MeV incident energy\n"
            "dist = distribution.get_interpolated_distribution(incident_energy=5.0)\n"
            "\n"
            "# Access parameters for this distribution\n"
            "outgoing_energies = dist['e_out']\n"
            "r_values = dist['r']  # precompound fraction\n"
            "a_values = dist['a']  # angular distribution slope\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
# Law 1, 4: Tabular distributions

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section


@dataclass
class TabularEnergyDistribution(EnergyDistribution):
    """
    Law 1: Tabular energy distribution.
    
    This is a tabular function of outgoing energy E' and incident energy E.
    """
    law: int = 1
    interpolation: int = 0  # Interpolation scheme (1=histogram, 2=lin-lin)
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Incident energy values (view of xss_data)
    
    # For each incident energy, there's a tabular distribution of outgoing energies
    # Each distribution has a number of points, an interpolation scheme, and energy-pdf pairs
    distribution_data: List[Dict] = field(default_factory=list)
    
    def get_outgoing_energy_distribution(self, incident_energy: float) -> Tuple[List[float], List[float]]:
        """
        Get the outgoing energy distribution for a given incident energy.
        
        For Law 1, this involves interpolating between the tabular distributions
        at the closest incident energies.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple of (energies, probabilities)
        """
        # Extract incident energy values
        incident_energy_values = [float(e) for e in self.incident_energies]
        
        # If energy is outside the tabulated range or we don't have enough data, return empty lists
        if not incident_energy_values or not self.distribution_data:
            return [], []
            
        if incident_energy <= incident_energy_values[0]:
            dist = self.distribution_data[0]
            return [float(e) for e in dist['e_out']], [float(p) for p in dist['pdf']]
            
        if incident_energy >= incident_energy_values[-1]:
            dist = self.distribution_data[-1]
            return [float(e) for e in dist['e_out']], [float(p) for p in dist['pdf']]
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(incident_energy_values, incident_energy) - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.distribution_data[idx]
        dist_high = self.distribution_data[idx + 1]
        
        # Calculate interpolation factor
        energy_low = incident_energy_values[idx]
        energy_high = incident_energy_values[idx + 1]
        factor = (incident_energy - energy_low) / (energy_high - energy_low)
        
        e_out_low = [float(e) for e in dist_low['e_out']]
        pdf_low = [float(p) for p in dist_low['pdf']]
        e_out_high = [float(e) for e in dist_high['e_out']]
        pdf_high = [float(p) for p in dist_high['pdf']]
        
        # For simplicity, we'll use the energy grid from the lower distribution
        # and interpolate the PDF values 
        pdf_high_interp = np.interp(e_out_low, e_out_high, pdf_high)
        pdf_interp = [(1.0 - factor) * p_low + factor * p_high 
                      for p_low, p_high in zip(pdf_low, pdf_high_interp)]
        
        return e_out_low, pdf_interp
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the TabularEnergyDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Tabular Energy Distribution (Law 1)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution represents the outgoing energy distribution for a secondary particle\n"
            "as a tabular function of both incident energy and outgoing energy.\n"
            "It corresponds to Law 1 in the ACE format.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Tabular)", 
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Interpolation Scheme", f"{self.interpolation} " + 
            f"({('Histogram' if self.interpolation == 1 else 'Lin-Lin' if self.interpolation == 2 else 'Unknown')})",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", f"{self.n_incident_energies}",
            width1=property_col_width, width2=value_col_width)
        
        # Show energy ranges if available
        if len(self.incident_energies) > 0:
            try:
                min_energy = self.incident_energies[0]
                max_energy = self.incident_energies[-1]
                properties += "{:<{width1}} {:<{width2}}\n".format(
                    "Incident Energy Range", f"{min_energy:.4e} - {max_energy:.4e} MeV",
                    width1=property_col_width, width2=value_col_width)
            except (IndexError, AttributeError):
                pass
        
        # Count distributions and points
        total_points = 0
        for dist in self.distribution_data:
            if 'e_out' in dist:
                total_points += len(dist['e_out'])
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Total Distribution Points", f"{total_points}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_outgoing_energy_distribution(incident_energy)": 
                "Get energies and probabilities for a given incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Data origin information
        data_origin = (
            "\nData Source:\n"
            "This data is parsed from the ACE-formatted nuclear data file and includes\n"
            "the incident energy grid and corresponding tabular distributions for outgoing energies.\n"
        )
        
        return header + description + properties + methods_section + data_origin


@dataclass
class ContinuousTabularDistribution(EnergyDistribution):
    """
    Law 4: Continuous tabular energy distribution.
    
    From ENDF-6 Law 1, this represents a fully tabulated energy distribution.
    The distribution may be discrete, continuous, or a combination.
    
    Data format (Table 35 and 36):
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
    """
    law: int = 4
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy distribution
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
            Dictionary containing the distribution data
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
        # Find the bracketing incident energies
        if len(self.incident_energies) == 0 or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.get_distribution(idx)
        dist_high = self.get_distribution(idx + 1)
        
        if not dist_low or not dist_high:
            return None
            
        # Interpolate the distributions based on the incident energy
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        
        # Determine the interpolation scheme from the regions
        # For simplicity, assume linear-linear interpolation
        # In a full implementation, we would determine this from the NBT and INT arrays
        
        # Interpolation fraction
        frac = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Interpolate the distribution arrays
        e_out_interp = np.interp(
            dist_low['e_out'],  # x-coordinates from first distribution
            dist_high['e_out'],  # x-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        pdf_interp = np.interp(
            dist_low['pdf'],  # y-coordinates from first distribution
            dist_high['pdf'],  # y-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        cdf_interp = np.interp(
            dist_low['cdf'],  # y-coordinates from first distribution
            dist_high['cdf'],  # y-coordinates from second distribution
            frac  # interpolation fraction
        )
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out_interp),
            'e_out': e_out_interp,
            'pdf': pdf_interp,
            'cdf': cdf_interp
        }
        
        return interp_dist
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the ContinuousTabularDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Continuous Tabular Distribution (Law 4)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution represents a fully tabulated continuous energy distribution.\n"
            "It corresponds to Law 4 in the ACE format (ENDF-6 Law 1).\n"
            "The distribution may represent discrete lines, continuous spectra, or a combination.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Continuous Tabular)", 
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", f"{self.n_interp_regions}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", f"{self.n_energies}",
            width1=property_col_width, width2=value_col_width)
        
        # Show energy ranges if available
        if len(self.incident_energies) >= 2:
            properties += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", f"{self.incident_energies[0]:.4e} - {self.incident_energies[-1]:.4e} MeV",
                width1=property_col_width, width2=value_col_width)
        
        # Count distribution points
        dist_count = len(self.distributions)
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Tabulated Distributions", f"{dist_count}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_distribution(energy_idx)": 
                "Get the distribution for a specific incident energy index",
            ".get_interpolated_distribution(incident_energy)": 
                "Get interpolated distribution for a specific incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Data origin information
        data_origin = (
            "\nData Source:\n"
            "This data is parsed from the ACE-formatted nuclear data file and includes the\n"
            "incident energy grid, interpolation parameters, and tabulated distributions for\n"
            "outgoing energies, including their probability density and cumulative distribution functions.\n"
        )
        
        return header + description + properties + methods_section + data_origin
# Law 1, 4: Tabular distributions

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution import tabular_math
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

        Each table holds the NET boundaries of NET-1 equally likely bins. Between
        two incident energies the boundaries are interpolated (lin-lin, or the
        lower table for INT=1), and the result is again NET-1 equally likely bins.
        Outside the tabulated range the end table is used.

        Parameters
        ----------
        incident_energy : float
            The incident neutron energy

        Returns
        -------
        Tuple[List[float], List[float]]
            Bin boundaries and the histogram density on each (0 at the last one)
        """
        if len(self.incident_energies) == 0 or not self.distribution_data:
            return [], []
        i, frac = tabular_math.bracket(self.incident_energies, incident_energy)
        bounds = np.asarray(self.distribution_data[i]['e_out'], dtype=float)
        if frac > 0.0 and tabular_math.interval_scheme(self.nbt, self.interp, i) != 1:
            upper = np.asarray(self.distribution_data[i + 1]['e_out'], dtype=float)
            bounds = (1.0 - frac) * bounds + frac * upper
        widths = np.diff(bounds)
        density = np.divide(1.0 / len(widths), widths, out=np.zeros_like(widths), where=widths > 0)
        return bounds.tolist(), np.append(density, 0.0).tolist()

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
    
    def get_interpolated_distribution(self, incident_energy: float) -> Optional[Dict]:
        """
        Outgoing-energy distribution at an incident energy.

        Between two tables the lin-lin rule of the ACE format applies (random
        choice of table plus unit-base scaling of the continuous part, INT=1
        taking the lower table); see :mod:`kika.ace.classes.energy_distribution.tabular_math`.
        Outside the tabulated range the end table is returned.

        Returns
        -------
        dict or None
            ``e_out``, ``pdf`` and ``cdf`` of the continuous part, and
            ``discrete_energies`` / ``discrete_probabilities`` of the lines.
        """
        return tabular_math.distribution_at(
            self.incident_energies, self.distributions, self.nbt, self.interp, incident_energy)

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
# Law 61, 67: Angle-energy distributions

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution import tabular_math
from kika._utils import create_repr_section


@dataclass
class TabulatedAngleEnergyDistribution(EnergyDistribution):
    """
    Law 61: Tabulated Angle-Energy Distribution.
    
    Similar to LAW=44 but with tabular angular distributions instead of the Kalbach-Mann formalism.
    
    Data format (Table 45 and 46):
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
    - LC(l): Location of angular distribution tables
    
    For each angular distribution (Table 47):
    - JJ: Interpolation flag
    - N_p: Number of points
    - CosOut(j): Cosine scattering angular grid
    - PDF(j): Probability density function
    - CDF(j): Cumulative density function
    """
    law: int = 61
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy-angle distribution
    # Each entry is a dictionary containing the distribution data
    distributions: List[Dict] = field(default_factory=list)
    
    # Store the angular distribution tables
    # Each table is a dictionary with 'jj', 'n_points', 'cosines', 'pdf', and 'cdf' keys
    angular_tables: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the energy distribution for a specific incident energy index.
        
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
            - 'lc': Location of angular distribution tables
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.distributions):
            return self.distributions[energy_idx]
        return None
    
    def get_angular_table(self, table_idx: int) -> Dict:
        """
        Get the angular distribution table with the specified index.
        
        Parameters
        ----------
        table_idx : int
            Index of the angular table
            
        Returns
        -------
        Dict
            Dictionary containing the angular distribution with keys:
            - 'jj': Interpolation flag
            - 'n_points': Number of points
            - 'cosines': Cosine scattering angular grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            or None if index is invalid
        """
        if 0 <= table_idx < len(self.angular_tables):
            return self.angular_tables[table_idx]
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

    def angular_pdf(self, incident_energy: float, cosines) -> np.ndarray:
        """
        Angular density at an incident energy, marginal over the outgoing
        energy (in the frame of the reaction, the centre of mass for a
        negative TY). LC=0 points are isotropic.

        Parameters
        ----------
        incident_energy : float
            Incident energy in MeV
        cosines : array_like
            Cosines at which to evaluate the density

        Returns
        -------
        numpy.ndarray
            Density in mu
        """
        mu = np.asarray(cosines, dtype=float)
        by_lc = {int(t["lc"]): t for t in self.angular_tables}
        return tabular_math.angular_at(
            self.incident_energies, self.distributions, self.nbt, self.interp, incident_energy,
            lambda table: tabular_math.tabular_angular_table(table, by_lc, mu))

    # Removed sampling methods:
    # - sample_angular_distribution
    # - sample_outgoing_energy_angle
    # - sample_outgoing_energy
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the TabulatedAngleEnergyDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Tabulated Angle-Energy Distribution (Law 61)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution provides correlated angle-energy distributions using tabular\n"
            "angular distributions. It corresponds to Law 61 in the ACE format.\n"
            "Similar to LAW=44 but uses tabular angular distributions instead of Kalbach-Mann formalism.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Tabulated Angle-Energy)", 
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
        
        # Count distributions and angular tables
        energy_dist_count = len(self.distributions)
        angular_table_count = len(self.angular_tables)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Distributions", f"{energy_dist_count}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Angular Tables", f"{angular_table_count}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods - UPDATED to remove sampling methods
        methods = {
            ".get_distribution(energy_idx)": 
                "Get energy distribution for a specific incident energy index",
            ".get_angular_table(table_idx)": 
                "Get angular distribution table for a specific index",
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
            "This data is parsed from the ACE-formatted nuclear data file and includes the incident\n"
            "energy grid, energy distributions, and angular distribution tables. Each energy point\n"
            "has a set of outgoing energies, and each outgoing energy has an associated angular\n"
            "distribution that describes the probability of scattering at different angles.\n"
        )
        
        return header + description + properties + methods_section + data_origin


@dataclass
class LaboratoryAngleEnergyDistribution(EnergyDistribution):
    """
    Law 67: Laboratory Angle-Energy Distribution.
    
    From ENDF-6 MF=6 LAW=7, this represents a distribution that directly 
    specifies the angular and energy distributions in the laboratory frame.
    
    Data format (Table 49, 50, 51):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Locations of distributions
    
    For each incident energy:
    - INTMU: Interpolation scheme for angles
    - NMU: Number of secondary cosines
    - XMU(l): Secondary cosines
    - LMU(l): Locations of secondary energy distributions
    
    For each cosine:
    - INTEP: Interpolation parameter for secondary energies
    - NPEP: Number of secondary energies
    - E_p(l): Secondary energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    """
    law: int = 67
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each angle-energy distribution
    # Each entry corresponds to one incident energy
    # Format: List of dictionaries with keys:
    # - 'intmu': Interpolation scheme for angles
    # - 'n_cosines': Number of secondary cosines
    # - 'cosines': Array of secondary cosines
    # - 'energy_dist_locations': Locations of energy distributions for each cosine
    # - 'energy_distributions': List of energy distributions for each cosine
    angle_energy_distributions: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the angle-energy distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if index is invalid
        """
        if 0 <= energy_idx < len(self.angle_energy_distributions):
            return self.angle_energy_distributions[energy_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated distribution for a specific incident energy.
        
        For Law 67, we find the bracketing incident energies and use the
        distribution for the lower energy. In a full implementation,
        we would interpolate between the distributions.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if not available
        """
        # Find the bracketing incident energies
        if len(self.incident_energies) == 0 or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the distribution for the lower incident energy
        # In a full implementation, we would interpolate between distributions
        return self.get_distribution(idx)

    # Removed sampling methods:
    # - sample_outgoing_angle_energy
    # - sample_outgoing_energy
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the LaboratoryAngleEnergyDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Laboratory Angle-Energy Distribution (Law 67)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution directly specifies the angular and energy distributions in the\n"
            "laboratory frame. It corresponds to Law 67 in the ACE format (ENDF-6 MF=6 LAW=7).\n"
            "For each incident energy, the distribution provides a set of emission angles and\n"
            "corresponding energy distributions for each angle.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Laboratory Angle-Energy)", 
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
        
        # Count angle-energy distributions
        dist_count = len(self.angle_energy_distributions)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Angle-Energy Distributions", f"{dist_count}",
            width1=property_col_width, width2=value_col_width)
        
        # Count total number of cosines across all distributions
        total_cosines = 0
        for dist in self.angle_energy_distributions:
            if 'cosines' in dist:
                total_cosines += len(dist['cosines'])
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Total Number of Angular Points", f"{total_cosines}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods - UPDATED to remove sampling methods
        methods = {
            ".get_distribution(energy_idx)": 
                "Get distribution for a specific incident energy index",
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
            "This data is parsed from the ACE-formatted nuclear data file and includes the incident\n"
            "energy grid and corresponding angle-energy distributions. For each incident energy,\n"
            "there is a set of cosines, and for each cosine, there is an energy distribution that\n"
            "specifies the probability of different outgoing energies at that angle.\n"
        )
        
        return header + description + properties + methods_section + data_origin

from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.types import AngularDistributionType
from kika._utils import create_repr_section
from kika.ace.classes.angular_distribution.utils import Law44DataError


@dataclass
class KalbachMannAngularDistribution(AngularDistribution):
    """
    Angular distribution using Law=44 (Kalbach-Mann) from the DLW/DLWH Block.
    This distribution is correlated with energy and uses the Kalbach-Mann formalism.
    The actual angular distribution data is stored in the energy-angle distribution
    section of the ACE file (DLW/DLWH blocks).
    
    IMPORTANT: This distribution requires Law=44 data from the energy distribution
    section to calculate angular probabilities. The ACE object must be provided to
    methods that calculate or sample angular distributions.
    """
    # Reference to the reaction index in the DLW/DLWH block
    reaction_index: int = -1
    # Whether this is a particle production reaction
    is_particle_production: bool = False
    # Particle index (only used if is_particle_production=True)
    particle_idx: int = -1
    # Flag to indicate this distribution requires Law=44 data
    requires_law44_data: bool = True
    
    def __post_init__(self):
        self.distribution_type = AngularDistributionType.KALBACH_MANN
    
    def _find_law44_distribution(self, ace):
        """
        Find the correlated energy-angle laws of this reaction.

        LOCB=-1 says the angle is given with the energy in the DLW block, by
        LAW=44 (Kalbach-87) or LAW=61 (tabular in angle); both are accepted.

        Returns
        -------
        list of EnergyDistribution
            The laws of the reaction, in DLW order

        Raises
        ------
        Law44DataError
            If the ACE object is not provided or holds no correlated law for this MT
        """
        mt_value = int(self.mt)
        if ace is None:
            raise Law44DataError(
                f"ACE object must be provided for the correlated angular distribution (MT={mt_value})"
            )
        if ace.energy_distributions is None:
            raise Law44DataError(
                f"Energy distributions missing in ACE object for MT={mt_value}"
            )
        if self.is_particle_production:
            if (self.particle_idx < 0 or
                    self.particle_idx >= len(ace.energy_distributions.particle_production)):
                raise Law44DataError(
                    f"Particle index {self.particle_idx} out of bounds for MT={mt_value}"
                )
            distributions = ace.energy_distributions.get_particle_distribution(
                self.particle_idx, mt_value)
        else:
            distributions = ace.energy_distributions.get_neutron_distribution(mt_value)

        where = f"MT={mt_value}" + (f", particle={self.particle_idx}" if self.is_particle_production else "")
        if not distributions:
            raise Law44DataError(f"No energy distributions found for {where}")
        if not all(d.law in (44, 61) for d in distributions):
            laws = [d.law for d in distributions]
            raise Law44DataError(f"{where} has LOCB=-1 but laws {laws}; only LAW=44 and 61 carry the angle")
        return list(distributions)

    def angular_pdf(self, energy: float, ace, cosines) -> np.ndarray:
        """
        Angular density at an incident energy, marginal over the outgoing energy.

        With several laws, law ``j`` is used with probability ``P_j(E)`` times
        the probability that no earlier law was (manual Table 25), the last law
        taking what remains.

        Parameters
        ----------
        energy : float
            Incident energy in MeV
        ace : Ace
            ACE object holding the DLW data
        cosines : array_like
            Cosines at which to evaluate the density

        Returns
        -------
        numpy.ndarray
            Density in mu (centre of mass for a negative TY)
        """
        mu = np.asarray(cosines, dtype=float)
        laws = self._find_law44_distribution(ace)
        out = np.zeros_like(mu)
        remaining = 1.0
        for j, law in enumerate(laws):
            w = remaining if j == len(laws) - 1 else remaining * law.get_applicability_probability(energy)
            out += w * law.angular_pdf(energy, mu)
            remaining -= w
        return out

    def to_dataframe(self, energy: Optional[float] = None, ace=None, num_points: int = 100, interpolate: bool = True) -> Optional[pd.DataFrame]:
        """
        Convert the correlated angular distribution to a pandas DataFrame.

        The density is the marginal over the outgoing energy of the LAW=44 or
        LAW=61 data (see :meth:`angular_pdf`), on a regular cosine grid.

        Parameters
        ----------
        energy : float, optional
            Incident energy to evaluate the distribution at. If None, five
            incident energies spread over the tables of the law are returned.
        ace : Ace
            ACE object containing the DLW data
        num_points : int, optional
            Number of angular points to generate, defaults to 100
        interpolate : bool, optional
            Kept for a common signature; the result is always on a regular grid

        Returns
        -------
        pandas.DataFrame
            'energy', 'cosine' and 'pdf' columns

        Raises
        ------
        Law44DataError
            If the ACE object is not provided or the correlated data is missing
        """
        cosines = np.linspace(-1, 1, num_points)
        if energy is None:
            laws = self._find_law44_distribution(ace)
            incident = np.asarray(laws[0].incident_energies, dtype=float)
            if len(incident) == 0:
                raise Law44DataError(f"No incident energies in the DLW data of MT={int(self.mt)}")
            picks = np.unique(np.linspace(0, len(incident) - 1, min(5, len(incident))).astype(int))
            energies = incident[picks]
        else:
            energies = [energy]
        frames = [pd.DataFrame({'energy': np.full_like(cosines, e, dtype=float),
                                'cosine': cosines,
                                'pdf': self.angular_pdf(float(e), ace, cosines)})
                  for e in energies]
        return pd.concat(frames, ignore_index=True)

    def __str__(self) -> str:
        """Human-readable string representation."""
        mt_value = int(self.mt)
        particle_info = f", particle={self.particle_idx}" if self.is_particle_production else ""
        return (f"Kalbach-Mann Angular Distribution (MT={mt_value}{particle_info})\n"
                f"REQUIRES: Law=44 data from energy distribution section\n"
                f"NOTE: Must provide ACE object when sampling or plotting this distribution")
    
    def __repr__(self) -> str:
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Kalbach-Mann Angular Distribution Details':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        description = (
            "This object represents an angular distribution using the Kalbach-Mann formalism.\n"
            "The Kalbach-Mann model correlates energy and angle distributions, with parameters\n"
            "R (precompound fraction) and A (angular slope) that vary with outgoing energy.\n\n"
            "Data Structure Overview:\n"
            "- In the ACE file, a LOCB value of -1 indicates a Kalbach-Mann distribution\n"
            "- The actual angular distribution parameters (R and A) are stored in the\n"
            "  energy distribution section as a Law=44 distribution\n"
            "- This object stores reference indices to locate the Law=44 data when needed\n\n"
            "IMPORTANT: This distribution REQUIRES Law=44 data from the energy distribution\n"
            "section (DLW/DLWH blocks). The ACE object must be provided to all methods that\n"
            "calculate or sample angular distributions. Without this data, methods will raise\n"
            "a Law44DataError exception.\n\n"
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
            "Distribution Type", "Kalbach-Mann (Law=44)",
            width1=property_col_width, width2=value_col_width)
        
        # Law 44 requirement
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Requires Law=44 Data", "Yes",
            width1=property_col_width, width2=value_col_width)
        
        # ACE requirement
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Requires ACE Object", "Yes",
            width1=property_col_width, width2=value_col_width)
        
        # Reaction index information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Reaction Index", self.reaction_index,
            width1=property_col_width, width2=value_col_width)
        
        # Particle production information
        if self.is_particle_production:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", f"Yes (particle index: {self.particle_idx})",
                width1=property_col_width, width2=value_col_width)
        else:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", "No (incident neutron reaction)",
                width1=property_col_width, width2=value_col_width)
        
        # Kalbach-Mann formula
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Kalbach-Mann Formula", "p(μ) = (a/2)/sinh(a) * [cosh(aμ) + r*sinh(aμ)]",
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Raw data properties section
        properties = {
            ".mt": "MT number of the reaction (int)",
            ".reaction_index": "Index of the reaction in the energy distribution table (int)",
            ".is_particle_production": "Whether this is a particle production reaction (bool)",
            ".particle_idx": "Particle type index if particle production (int)"
        }
        
        properties_section = create_repr_section(
            "Raw Data Properties (Reference data from ACE file):", 
            properties, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Error handling section
        error_section = "Error Handling:\n"
        error_section += "-" * header_width + "\n"
        error_section += (
            "If Law=44 data is required but not available, methods will raise Law44DataError.\n"
            "This can happen when:\n"
            "  - ACE object is not provided to methods\n"
            "  - ACE object doesn't contain energy distribution data\n"
            "  - No Law=44 distribution is found for this reaction\n"
            "  - Distribution data is incomplete or invalid\n"
        )
        error_section += "-" * header_width + "\n\n"
        
        # Methods section
        methods = {
            ".to_dataframe(energy, ace, num_points)": "Convert to a pandas DataFrame at a specific energy",
        }

        methods_section = create_repr_section(
            "Calculation Methods (All require ACE object):",
            methods,
            total_width=header_width,
            method_col_width=property_col_width
        )

        # Add example for using this specific distribution type
        example = (
            "Example:\n"
            "--------\n"
            "# Access reference properties\n"
            "mt_value = int(distribution.mt)\n"
            "reaction_idx = distribution.reaction_index\n"
            "is_particle = distribution.is_particle_production\n\n"
            "# Get data as DataFrame at 14 MeV\n"
            "try:\n"
            "    df = distribution.to_dataframe(energy=14.0, ace=ace_object)\n"
            "except Law44DataError as e:\n"
            "    print(f\"Error: {e}\")\n"
        )

        return header + description + info_table + properties_section + "\n" + error_section + methods_section + "\n" + example



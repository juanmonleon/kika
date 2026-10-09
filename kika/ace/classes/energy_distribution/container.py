from dataclasses import dataclass, field
from typing import List, Dict, Optional
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section


@dataclass
class EnergyDistributionContainer:
    """Container for energy distributions from DLW, DLWP, DLWH, and DNED blocks."""
    # Neutron reaction energy distributions (MT → list of distributions)
    incident_neutron: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    
    # Photon production energy distributions (MT → list of distributions)
    photon_production: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    
    # Particle production energy distributions (particle index → MT → list of distributions)
    particle_production: List[Dict[int, List[EnergyDistribution]]] = field(default_factory=list)
    
    # Delayed neutron energy distributions (precursor group → distribution)
    delayed_neutron: List[EnergyDistribution] = field(default_factory=list)
    
    # Energy-dependent neutron yields (MT → yield)
    neutron_yields: Dict[int, EnergyDistribution] = field(default_factory=dict)
    
    # Energy-dependent photon yields (MT → yield)
    photon_yields: Dict[int, EnergyDistribution] = field(default_factory=dict)
    
    # Energy-dependent particle yields (particle index → MT → yield)
    particle_yields: List[Dict[int, EnergyDistribution]] = field(default_factory=list)
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction energy distributions are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production energy distributions are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production energy distributions are available."""
        return len(self.particle_production) > 0
    
    @property
    def has_delayed_neutron_data(self) -> bool:
        """Check if delayed neutron energy distributions are available."""
        return len(self.delayed_neutron) > 0
    
    @property
    def has_neutron_yields(self) -> bool:
        """Check if energy-dependent neutron yields are available."""
        return len(self.neutron_yields) > 0
    
    @property
    def has_photon_yields(self) -> bool:
        """Check if energy-dependent photon yields are available."""
        return len(self.photon_yields) > 0
    
    @property
    def has_particle_yields(self) -> bool:
        """Check if energy-dependent particle yields are available."""
        return len(self.particle_yields) > 0
    
    def get_neutron_reaction_mt_numbers(self) -> List[int]:
        """Get a list of all MT numbers with neutron reaction energy distributions."""
        return sorted(int(mt) for mt in self.incident_neutron.keys())
    
    def get_photon_production_mt_numbers(self) -> List[int]:
        """Get a list of all MT numbers with photon production energy distributions."""
        return sorted(int(mt) for mt in self.photon_production.keys())
    
    def get_particle_production_mt_numbers(self, particle_idx: int = 0) -> List[int]:
        """
        Get a list of all MT numbers with particle production energy distributions for a specific particle.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        List[int]
            List of MT numbers sorted in ascending order, or empty list if particle_idx is invalid
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return []
        return sorted(list(self.particle_production[particle_idx].keys()))
    
    def get_neutron_distribution(self, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a neutron reaction with the specified MT number.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        if mt in self.incident_neutron:
            return self.incident_neutron[mt]
        
        # Not found
        return None
    
    def get_photon_distribution(self, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a photon production reaction with the specified MT number.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        if mt in self.photon_production:
            return self.photon_production[mt]
        
        # Not found
        return None
    
    def get_particle_distribution(self, particle_idx: int, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a particle production reaction.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
            
        # Get the particle's dictionary
        particle_dict = self.particle_production[particle_idx]
        
        if mt in particle_dict:
            return particle_dict[mt]
        
        # Not found
        return None
    
    def get_delayed_neutron_distribution(self, group: int) -> Optional[EnergyDistribution]:
        """
        Get the energy distribution for a delayed neutron precursor group.
        
        Parameters
        ----------
        group : int
            Precursor group index (0-based)
            
        Returns
        -------
        Optional[EnergyDistribution]
            Energy distribution for the precursor group, or None if not found
        """
        if group < 0 or group >= len(self.delayed_neutron):
            return None
        return self.delayed_neutron[group]
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the EnergyDistributionContainer object.
    
        This representation provides an overview of available energy distributions by type and MT number.
        
        Returns
        -------
        str
            Formatted string representation of the EnergyDistributionContainer
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Energy Distribution Container':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of energy distributions
        description = (
            "This container holds energy distributions for secondary particles produced in nuclear reactions.\n"
            "Distributions are organized by particle type (neutron, photon, other) and reaction (MT number).\n"
            "Each reaction can have multiple distribution laws that apply in different energy ranges.\n\n"
        )
        
        # Create a summary table of available data
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        data_summary = "Available Energy Distribution Data:\n"
        data_summary += "-" * header_width + "\n"
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Type", "Status", width1=property_col_width, width2=value_col_width)
        data_summary += "-" * header_width + "\n"
        
        # Neutron distributions
        n_neutron_mt = len(self.incident_neutron)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Incident Neutron Distributions", f"{'Available' if n_neutron_mt > 0 else 'None'} ({n_neutron_mt} MT numbers)", 
            width1=property_col_width, width2=value_col_width)
        
        # Photon production
        n_photon_mt = len(self.photon_production)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production Distributions", f"{'Available' if n_photon_mt > 0 else 'None'} ({n_photon_mt} MT numbers)", 
            width1=property_col_width, width2=value_col_width)
        
        # Particle production
        n_particles = len(self.particle_production)
        has_particles = n_particles > 0 and any(self.particle_production)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production Distributions", f"{'Available' if has_particles else 'None'} ({n_particles} particle types)", 
            width1=property_col_width, width2=value_col_width)
        
        # Delayed neutron
        n_delayed = len(self.delayed_neutron)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Delayed Neutron Distributions", f"{'Available' if n_delayed > 0 else 'None'} ({n_delayed} groups)", 
            width1=property_col_width, width2=value_col_width)
        
        # Energy-dependent yields
        n_neutron_yields = len(self.neutron_yields)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Energy-Dependent Neutron Yields", f"{'Available' if n_neutron_yields > 0 else 'None'} ({n_neutron_yields} MT numbers)", 
            width1=property_col_width, width2=value_col_width)
        
        n_photon_yields = len(self.photon_yields)
        data_summary += "{:<{width1}} {:<{width2}}\n".format(
            "Energy-Dependent Photon Yields", f"{'Available' if n_photon_yields > 0 else 'None'} ({n_photon_yields} MT numbers)", 
            width1=property_col_width, width2=value_col_width)
        
        data_summary += "-" * header_width + "\n\n"
        
        # Summary of distribution types
        distribution_types = {}
        
        # Count distribution types for incident neutron
        for mt, dist_list in self.incident_neutron.items():
            for dist in dist_list:
                dist_type = dist.__class__.__name__
                if dist_type not in distribution_types:
                    distribution_types[dist_type] = 0
                distribution_types[dist_type] += 1
                
        # Count distribution types for photon production
        for mt, dist_list in self.photon_production.items():
            for dist in dist_list:
                dist_type = dist.__class__.__name__
                if dist_type not in distribution_types:
                    distribution_types[dist_type] = 0
                distribution_types[dist_type] += 1
        
        # Count distribution types for particle production
        for particle_dict in self.particle_production:
            for mt, dist_list in particle_dict.items():
                for dist in dist_list:
                    dist_type = dist.__class__.__name__
                    if dist_type not in distribution_types:
                        distribution_types[dist_type] = 0
                    distribution_types[dist_type] += 1
        
        # Count distribution types for delayed neutron
        for dist in self.delayed_neutron:
            dist_type = dist.__class__.__name__
            if dist_type not in distribution_types:
                distribution_types[dist_type] = 0
            distribution_types[dist_type] += 1
        
        # Display distribution type summary
        if distribution_types:
            dist_summary = "Distribution Type Summary:\n"
            dist_summary += "-" * header_width + "\n"
            dist_summary += "{:<{width1}} {:<{width2}}\n".format(
                "Distribution Type", "Count", width1=property_col_width, width2=value_col_width)
            dist_summary += "-" * header_width + "\n"
            
            for dist_type, count in sorted(distribution_types.items(), key=lambda x: x[1], reverse=True):
                dist_summary += "{:<{width1}} {:<{width2}}\n".format(
                    dist_type, count, width1=property_col_width, width2=value_col_width)
            
            dist_summary += "-" * header_width + "\n\n"
        else:
            dist_summary = ""
            
        # Create a section for data access methods
        data_access = {
            ".get_neutron_reaction_mt_numbers()": "Get list of MT numbers with neutron distributions",
            ".get_photon_production_mt_numbers()": "Get list of MT numbers with photon distributions",
            ".get_particle_production_mt_numbers(particle_idx)": "Get MT numbers for a specific particle type",
            ".get_neutron_distribution(mt)": "Get neutron distribution for a specific MT number",
            ".get_photon_distribution(mt)": "Get photon distribution for a specific MT number",
            ".get_particle_distribution(particle_idx, mt)": "Get particle distribution for specific type and MT",
            ".get_delayed_neutron_distribution(group)": "Get delayed neutron distribution for a precursor group",
            ".print_distribution_info()": "Print detailed information about distribution types"
        }
        
        data_access_section = create_repr_section(
            "Data Access Methods:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + data_summary + dist_summary + data_access_section
    
    def print_distribution_info(self) -> None:
        """
        Print detailed information about the energy distribution types for each reaction.
        
        This method provides a user-friendly display of what energy distribution types
        are stored for each reaction category (neutron, photon, particle, delayed).
        """
        header_width = 100
        print("=" * header_width)
        print(f"{'Energy Distribution Type Information':^{header_width}}")
        print("=" * header_width)
        
        # Function to print distribution info table
        def print_distribution_table(title, data_dict, getter_method, particle_idx=None):
            if not data_dict:
                return
            
            # Add particle index to title if applicable
            if particle_idx is not None:
                title = f"{title} (Particle Index: {particle_idx})"
            
            print(f"\n{title}")
            print("-" * header_width)
            print(f"{'MT':<8} {'Law':<8} {'Distribution Type':<30} {'Access Method':<50}")
            print("-" * header_width)
            
            for mt, dist_list in sorted(data_dict.items()):
                mt_value = mt
                
                # Determine the access method string
                if particle_idx is not None:
                    access = f"{getter_method}({particle_idx}, {mt_value})"
                else:
                    access = f"{getter_method}({mt_value})"
                
                for i, dist in enumerate(dist_list):
                    dist_type = dist.__class__.__name__
                    
                    # Get law if available
                    law = ""
                    if hasattr(dist, 'law'):
                        law = str(dist.law)
                    
                    if i == 0:
                        print(f"{mt_value:<8} {law:<8} {dist_type:<30} {access:<50}")
                    else:
                        print(f"{'':<8} {law:<8} {dist_type:<30} {'':<50}")
        
        # Print neutron distributions
        if self.incident_neutron:
            print_distribution_table("Incident Neutron Energy Distributions", 
                                    self.incident_neutron, 
                                    ".get_neutron_distribution")
        
        # Print photon distributions
        if self.photon_production:
            print_distribution_table("Photon Production Energy Distributions", 
                                    self.photon_production, 
                                    ".get_photon_distribution")
        
        # Print particle distributions for each particle type
        for idx, particle_dict in enumerate(self.particle_production):
            if particle_dict:
                print_distribution_table("Particle Production Energy Distributions", 
                                        particle_dict, 
                                        ".get_particle_distribution", 
                                        idx)
        
        # Print delayed neutron distributions
        if self.delayed_neutron:
            print("\nDelayed Neutron Energy Distributions")
            print("-" * header_width)
            print(f"{'Group':<8} {'Distribution Type':<30} {'Access Method':<50}")
            print("-" * header_width)
            
            for i, dist in enumerate(self.delayed_neutron):
                dist_type = dist.__class__.__name__
                access = f".get_delayed_neutron_distribution({i})"
                print(f"{i:<8} {dist_type:<30} {access:<50}")
        
        # Print energy-dependent yields information if available
        yield_categories = [
            ("Neutron", self.neutron_yields, "neutron_yields"),
            ("Photon", self.photon_yields, "photon_yields")
        ]
        
        for yield_name, yield_dict, attr_name in yield_categories:
            if yield_dict:
                print(f"\nEnergy-Dependent {yield_name} Yields")
                print("-" * header_width)
                print(f"{'MT':<8} {'Distribution Type':<30} {'Access Method':<50}")
                print("-" * header_width)
                
                for mt, dist in sorted(yield_dict.items()):
                    mt_value = mt
                    dist_type = dist.__class__.__name__
                    access = f".{attr_name}[{mt_value}]"
                    print(f"{mt_value:<8} {dist_type:<30} {access:<50}")
        
        # Print particle yield information
        for idx, particle_dict in enumerate(self.particle_yields):
            if particle_dict:
                print(f"\nEnergy-Dependent Particle Yields (Particle Index: {idx})")
                print("-" * header_width)
                print(f"{'MT':<8} {'Distribution Type':<30} {'Access Method':<50}")
                print("-" * header_width)
                
                for mt, dist in sorted(particle_dict.items()):
                    mt_value = mt
                    dist_type = dist.__class__.__name__
                    access = f".particle_yields[{idx}][{mt_value}]"
                    print(f"{mt_value:<8} {dist_type:<30} {access:<50}")
        
        print("\n" + "=" * header_width)


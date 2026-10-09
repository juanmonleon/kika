from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import numpy as np
from kika._utils import create_repr_section

@dataclass
class PhotonProductionData:
    """
    Container for photon production data from the GPD block.
    
    The GPD Block contains the total photon production cross section tabulated
    on the energy grid from the ESZ Block. It may also include outgoing photon
    energies in an obsolete 30×20 matrix format used in older datasets.
    """
    total_xs: np.ndarray = field(default_factory=lambda: np.empty(0))  # Total photon production cross section (view of xss_data)
    outgoing_energies: Optional[List[np.ndarray]] = None  # 30x20 matrix of outgoing photon energies (views of xss_data)
    neutron_energy_boundaries: Optional[List[float]] = None  # 30 neutron energy group boundaries
    
    @property
    def has_data(self) -> bool:
        """Check if any photon production data is available."""
        return len(self.total_xs) > 0
    
    @property
    def has_outgoing_energies(self) -> bool:
        """Check if outgoing photon energy data is available."""
        return (self.outgoing_energies is not None and 
                len(self.outgoing_energies) > 0 and 
                self.neutron_energy_boundaries is not None)
    
    def get_interpolated_xs(self, energy: float, energy_grid: List[float]) -> Optional[float]:
        """
        Get an interpolated photon production cross section value at a given energy.
        
        Parameters
        ----------
        energy : float
            Energy point (in MeV)
        energy_grid : List[float]
            Full energy grid
            
        Returns
        -------
        float or None
            Interpolated photon production cross section value, or None if not available
        """
        if not self.has_data or len(self.total_xs) != len(energy_grid):
            return None
        
        # Check if energy is out of range
        if energy < energy_grid[0] or energy > energy_grid[-1]:
            return None
        
        xs_values = [float(entry) for entry in self.total_xs]
        
        # Use numpy for efficient interpolation
        return np.interp(energy, energy_grid, xs_values)
    
    def get_photon_energy_group(self, neutron_energy: float) -> Optional[List[float]]:
        """
        Get the appropriate group of 20 outgoing photon energies for a given incident neutron energy.

        The 30 boundaries are read as upper bounds, as manual Table 47 heads
        them. Table 48 instead lays group i out as EN(i) <= E < EN(i+1), which
        would shift every energy one group down. No table in Lib81 or JEFF-4.0
        carries this obsolete matrix (it needs JXS(13)=0), so neither reading
        can be checked against data.
        
        Parameters
        ----------
        neutron_energy : float
            Incident neutron energy (MeV)
            
        Returns
        -------
        List[float] or None
            20 equiprobable outgoing photon energies, or None if not available
        """
        if not self.has_outgoing_energies:
            return None
            
        # Find which neutron energy group this energy falls into
        group_idx = 0
        for i, boundary in enumerate(self.neutron_energy_boundaries):
            if neutron_energy < boundary:
                group_idx = i
                break
                
        # If energy is greater than all boundaries, use the last group
        if neutron_energy >= self.neutron_energy_boundaries[-1]:
            group_idx = len(self.neutron_energy_boundaries) - 1
            
        # Return the corresponding group of photon energies as floats
        if 0 <= group_idx < len(self.outgoing_energies):
            return [float(entry) for entry in self.outgoing_energies[group_idx]]
        
        return None
    
    def get_photon_energy_distribution(self, neutron_energy: float) -> Optional[Tuple[List[float], List[float]]]:
        """
        Get a probability distribution for outgoing photon energies at a given neutron energy.
        
        Parameters
        ----------
        neutron_energy : float
            Incident neutron energy (MeV)
            
        Returns
        -------
        Tuple[List[float], List[float]] or None
            Tuple of (energies, probabilities), or None if not available
        """
        energies = self.get_photon_energy_group(neutron_energy)
        if energies is None:
            return None
            
        # The energies are equiprobable, so create a uniform probability distribution
        n_points = len(energies)
        probabilities = [1.0/n_points] * n_points
        
        return (energies, probabilities)
    
    def plot_xs(self, energy_grid: List[float], ax=None, **kwargs):
        """
        Plot total photon production cross section.
        
        Parameters
        ----------
        energy_grid : List[float]
            Full energy grid
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        if self.has_data and len(self.total_xs) == len(energy_grid):
            xs_values = [float(entry) for entry in self.total_xs]
            
            ax.plot(energy_grid, xs_values, label="Total photon production", **kwargs)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Cross Section (barns)')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def plot_outgoing_energies(self, neutron_energy: Optional[float] = None, ax=None, **kwargs):
        """
        Plot the outgoing photon energy distribution for a specific neutron energy.
        
        Parameters
        ----------
        neutron_energy : float, optional
            Incident neutron energy (MeV). If None, multiple distributions will be shown.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        if not self.has_outgoing_energies:
            return ax
            
        if neutron_energy is not None:
            # Plot for a specific neutron energy
            distribution = self.get_photon_energy_distribution(neutron_energy)
            if distribution:
                energies, probs = distribution
                ax.step(energies, probs, where='post', 
                       label=f"E_neutron = {neutron_energy:.2e} MeV", **kwargs)
        else:
            # Plot for a few representative neutron energies
            sample_indices = [0, 5, 10, 15, 20, 25, 29]  # Select a few groups to display
            for idx in sample_indices:
                if idx < len(self.neutron_energy_boundaries) and idx < len(self.outgoing_energies):
                    e_neutron = self.neutron_energy_boundaries[idx]
                    energies = [float(e) for e in self.outgoing_energies[idx]]
                    probs = [1.0/len(energies)] * len(energies)
                    ax.step(energies, probs, where='post', 
                           label=f"E_neutron = {e_neutron:.2e} MeV", **kwargs)
        
        ax.set_xscale('log')
        ax.set_xlabel('Photon Energy (MeV)')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the PhotonProductionData object.
        
        This representation provides an overview of the photon production data.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Photon Production Data (GPD Block)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description
        description = (
            "The GPD Block contains the total photon production cross section tabulated\n"
            "on the energy grid from the ESZ Block. It may also include outgoing photon\n"
            "energies in an obsolete 30×20 matrix format used in older datasets.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3
        
        info_table = "Data Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add data about total production cross section
        xs_status = "Available" if self.has_data else "Not available"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Total Photon Production Cross Section", xs_status,
            width1=property_col_width, width2=value_col_width)
        
        if self.has_data:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Cross Section Points", len(self.total_xs),
                width1=property_col_width, width2=value_col_width)
        
        # Add data about outgoing photon energies
        energy_status = "Available (obsolete 30×20 matrix format)" if self.has_outgoing_energies else "Not available"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Outgoing Photon Energies", energy_status,
            width1=property_col_width, width2=value_col_width)
        
        if self.has_outgoing_energies:
            num_groups = len(self.outgoing_energies)
            points_per_group = len(self.outgoing_energies[0]) if num_groups > 0 else 0
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Matrix Size", f"{num_groups} neutron groups × {points_per_group} photon energies",
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create sections for methods
        methods = {
            ".get_interpolated_xs(energy, energy_grid)": "Get interpolated photon production cross section",
            ".get_photon_energy_group(neutron_energy)": "Get outgoing photon energies for a neutron energy",
            ".get_photon_energy_distribution(neutron_energy)": "Get probability distribution for outgoing photons",
            ".plot_xs(energy_grid, ax=None, **kwargs)": "Plot photon production cross section",
            ".plot_outgoing_energies(neutron_energy=None, ax=None, **kwargs)": "Plot outgoing photon energy distribution"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Combine all sections
        return header + description + info_table + methods_section

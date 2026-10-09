from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from kika._utils import create_repr_section
from kika.ace.classes.nubar.nubar_repr import nudata_repr, nucontainer_repr
import pandas as pd
import numpy as np

@dataclass
class NuPolynomial:
    """Polynomial form of nubar data."""
    coefficients: np.ndarray = field(default_factory=lambda: np.empty(0))  # view of xss_data
    
    def evaluate(self, energy: float) -> float:
        """Evaluate the polynomial at the given energy.
        
        :param energy: Energy in MeV
        :type energy: float
        :returns: The nubar value at the given energy
        :rtype: float
        """
        result = 0.0
        for i, coef in enumerate(self.coefficients):
            result += coef * (energy ** i)
        return result

@dataclass
class NuTabulated:
    """Tabulated form of nubar data."""
    interpolation_regions: List[Tuple[int, int]] = field(default_factory=list)  # (NBT, INT) pairs
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energy points (view of xss_data)
    nubar_values: np.ndarray = field(default_factory=lambda: np.empty(0))  # nubar values (view of xss_data)
    
    def evaluate(self, energy: float) -> float:
        """Evaluate the tabulated data at the given energy using interpolation.
        
        :param energy: Energy in MeV
        :type energy: float
        :returns: The nubar value at the given energy
        :rtype: float
        """
        # Simple linear interpolation for now
        if energy <= self.energies[0]:
            return self.nubar_values[0]
        
        if energy >= self.energies[-1]:
            return self.nubar_values[-1]
        
        # Find the bracketing energy points
        for i in range(len(self.energies) - 1):
            if self.energies[i] <= energy <= self.energies[i + 1]:
                # Linear interpolation
                x1, x2 = self.energies[i], self.energies[i + 1]
                y1, y2 = self.nubar_values[i], self.nubar_values[i + 1]
                return y1 + (y2 - y1) * (energy - x1) / (x2 - x1)
        
        # Shouldn't reach here, but just in case
        return self.nubar_values[-1]

@dataclass
class NuData:
    """Container for either polynomial or tabulated nubar data."""
    format: str = ""  # "polynomial" or "tabulated"
    polynomial: Optional[NuPolynomial] = None
    tabulated: Optional[NuTabulated] = None
    
    @property
    def energies(self) -> Optional[List[float]]:
        """Get the energy grid for this nubar data.
        
        For tabulated data, returns the actual energy grid.
        For polynomial data, returns None as there is no explicit energy grid.
        
        :returns: List of energy points or None if polynomial format
        :rtype: list of float or None
        """
        if self.format == "tabulated" and self.tabulated is not None:
            energy_points = [float(e) for e in self.tabulated.energies]
            return energy_points
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the nubar data to a pandas DataFrame.
        
        For polynomial data, creates a linspace of 100 points from 1e-5 to 20 MeV.
        For tabulated data, uses the energy points in the table.
        
        :returns: DataFrame with columns 'energy' and 'nubar'
        :rtype: pandas.DataFrame
        """
        if self.format == "polynomial" and self.polynomial is not None:
            # For polynomial, generate default energy points
            energy_points = np.linspace(1e-5, 20.0, 100)
            
            # Evaluate polynomial at each energy point
            nubar_values = [self.polynomial.evaluate(e) for e in energy_points]
            
        elif self.format == "tabulated" and self.tabulated is not None:
            # For tabulated, use the existing energy grid
            energy_points = [float(e) for e in self.tabulated.energies]  
            nubar_values = [float(n) for n in self.tabulated.nubar_values]  
        else:
            return pd.DataFrame(columns=['energy', 'nubar'])
        
        return pd.DataFrame({
            'energy': energy_points,
            'nubar': nubar_values
        })
    
    def plot(self, ax=None, title=None, **kwargs) -> tuple:
        """Create a plot of the nubar data.
        
        :param ax: Matplotlib axes to plot on. If None, new figure and axes are created.
        :type ax: matplotlib.axes.Axes, optional
        :param title: Title for the plot. If None, a default title is used.
        :type title: str, optional
        :param kwargs: Additional keyword arguments passed to the plot function.
        :type kwargs: dict, optional
        :returns: Tuple containing the figure and axes objects
        :rtype: tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        import matplotlib.pyplot as plt
        
        # Get the data to plot
        df = self.to_dataframe()
        
        # Set default plot parameters if not specified
        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 2
        if 'color' not in kwargs:
            kwargs['color'] = 'blue'
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        
        # Plot the data
        ax.plot(df['energy'], df['nubar'], **kwargs)
        
        # Set labels and title
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Nubar (neutrons/fission)')
        
        if title is None:
            title = f'Nubar Data ({self.format.capitalize()})'
        ax.set_title(title)
        
        # Set log scale for x-axis (common for nuclear data)
        ax.set_xscale('log')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def __repr__(self):
        return nudata_repr(self)
    

@dataclass
class NuContainer:
    """Container for all nubar data (prompt, total, delayed)."""
    has_nubar: bool = False  # True if NU block is present
    has_both_nu_types: bool = False  # True if both prompt and total nu-bar are given
    prompt: Optional[NuData] = None  # Prompt nu-bar data
    total: Optional[NuData] = None  # Total nu-bar data
    has_delayed: bool = False  # True if DNU block is present
    delayed: Optional[NuData] = None  # Delayed nu-bar data
    
    def get_nubar(self, energy: float, nubar_type: str) -> Optional[float]:
        """Get the nubar value for the given energy and type.
        
        :param energy: Energy in MeV
        :type energy: float
        :param nubar_type: Type of nubar to evaluate: 'prompt', 'total', or 'delayed'
        :type nubar_type: str
        :returns: The nubar value, or None if the requested type is not available
        :rtype: float or None
        """
        nubar_obj = None
        
        if nubar_type.lower() == 'prompt':
            nubar_obj = self.prompt
        elif nubar_type.lower() == 'total':
            nubar_obj = self.total
        elif nubar_type.lower() == 'delayed':
            nubar_obj = self.delayed
        else:
            raise ValueError(f"Unknown nubar type: {nubar_type}")
        
        if nubar_obj is None:
            return None
        
        if nubar_obj.format == "polynomial" and nubar_obj.polynomial is not None:
            return nubar_obj.polynomial.evaluate(energy)
        elif nubar_obj.format == "tabulated" and nubar_obj.tabulated is not None:
            return nubar_obj.tabulated.evaluate(energy)
        
        return None
    
    def to_dataframe(self, nubar_types: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """Convert nubar data to a pandas DataFrame.
        
        :param nubar_types: Which nubar types to include: 'prompt', 'total', 'delayed'.
                            Can be a single string or a list of strings.
                            If None, includes all available types.
        :type nubar_types: str or list of str, optional
        :returns: DataFrame with columns 'energy' and one column for each nubar type
        :rtype: pandas.DataFrame
        """
        # Convert single string to list if provided
        if isinstance(nubar_types, str):
            nubar_types = [nubar_types]
            
        # Default to all available types if not specified
        if nubar_types is None:
            nubar_types = []
            if self.prompt is not None:
                nubar_types.append('prompt')
            if self.total is not None:
                nubar_types.append('total')
            if self.delayed is not None:
                nubar_types.append('delayed')
        
        # Get DataFrames for each requested type
        dfs = []
        for ntype in nubar_types:
            nubar_obj = None
            if ntype.lower() == 'prompt':
                nubar_obj = self.prompt
            elif ntype.lower() == 'total':
                nubar_obj = self.total
            elif ntype.lower() == 'delayed':
                nubar_obj = self.delayed
            
            if nubar_obj is not None:
                df = nubar_obj.to_dataframe()
                # Use type name directly as column name without 'nubar_' prefix
                df.rename(columns={'nubar': ntype}, inplace=True)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame(columns=['energy'])
        
        # Merge all dataframes on energy
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='energy', how='outer')
        
        # Sort by energy
        result.sort_values('energy', inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        return result
    
    def plot(self, nubar_types: Optional[Union[str, List[str]]] = None, 
             ax=None, title=None, colors=None, styles=None, **kwargs) -> tuple:
        """Create a plot of nubar data for specified types.
        
        :param nubar_types: Which nubar types to plot: 'prompt', 'total', 'delayed'.
                           Can be a single string or a list of strings.
                           If None, plots all available types.
        :type nubar_types: str or list of str, optional
        :param ax: Matplotlib axes to plot on. If None, new figure and axes are created.
        :type ax: matplotlib.axes.Axes, optional
        :param title: Title for the plot. If None, a default title is used.
        :type title: str, optional
        :param colors: Dictionary mapping nubar types to colors, or list of colors.
                      If None, default colors are used.
        :type colors: dict or list, optional
        :param styles: Dictionary mapping nubar types to line styles, or list of styles.
                      If None, default styles are used.
        :type styles: dict or list, optional
        :param kwargs: Additional keyword arguments passed to the plot function.
        :type kwargs: dict, optional
        :returns: Tuple containing the figure and axes objects
        :rtype: tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        import matplotlib.pyplot as plt
        
        # Get DataFrame with all the data
        df = self.to_dataframe(nubar_types)
        
        # If the dataframe is empty, return empty plot
        if df.empty or df.shape[1] <= 1:  # Only energy column, no data
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5))
            else:
                fig = ax.figure
            
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Nubar (neutrons/fission)')
            ax.set_title('No Nubar Data Available' if title is None else title)
            return fig, ax
        
        # Get list of nubar types that are present in the DataFrame
        available_types = [col for col in df.columns if col != 'energy']
        
        # Define default colors and styles if not provided
        default_colors = {'prompt': 'blue', 'total': 'red', 'delayed': 'green'}
        default_styles = {'prompt': '-', 'total': '--', 'delayed': '-.'}
        
        # Process colors parameter
        if colors is None:
            plot_colors = default_colors
        elif isinstance(colors, dict):
            plot_colors = colors
        elif isinstance(colors, list) or isinstance(colors, tuple):
            plot_colors = {t: c for t, c in zip(available_types, colors)}
        else:
            plot_colors = default_colors
        
        # Process styles parameter
        if styles is None:
            plot_styles = default_styles
        elif isinstance(styles, dict):
            plot_styles = styles
        elif isinstance(styles, list) or isinstance(styles, tuple):
            plot_styles = {t: s for t, s in zip(available_types, styles)}
        else:
            plot_styles = default_styles
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot each type of nubar data
        for nubar_type in available_types:
            # Prepare plot parameters for this type
            plot_params = kwargs.copy()
            if nubar_type in plot_colors:
                plot_params['color'] = plot_colors[nubar_type]
            if nubar_type in plot_styles:
                plot_params['linestyle'] = plot_styles[nubar_type]
                
            # Plot this type
            ax.plot(df['energy'], df[nubar_type], label=nubar_type.capitalize(), **plot_params)
        
        # Set labels and title
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Nubar (neutrons/fission)')
        
        if title is None:
            title = 'Nubar Data'
        ax.set_title(title)
        
        # Set log scale for x-axis (common for nuclear data)
        ax.set_xscale('log')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax
    
    def __repr__(self):
        return nucontainer_repr(self)

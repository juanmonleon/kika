from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from kika.ace.classes.energy_distribution import tabular_math
from kika.ace.classes.delayed_neutron.delayed_neutron_repr import precursor_repr, delayed_neutron_data_repr

@dataclass
class DelayedNeutronPrecursor:
    """Data for a single delayed neutron precursor group."""
    decay_constant: Optional[float] = None  # Decay constant for the group
    interpolation_regions: List[Tuple[int, int]] = field(default_factory=list)  # (NBT, INT) pairs
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energy points (view of xss_data)
    probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))  # Corresponding probabilities
    
    def evaluate(self, energy: float) -> float:
        """
        Evaluate the precursor probability at the given energy using interpolation.
        
        Parameters
        ----------
        energy : float
            Energy in MeV
            
        Returns
        -------
        float
            The probability value at the given energy
        """
        if len(self.energies) == 0 or len(self.probabilities) == 0:
            return 0.0
        nbt = [int(b) for b, _ in self.interpolation_regions]
        interp = [int(i) for _, i in self.interpolation_regions]
        return tabular_math.tab1(self.energies, self.probabilities, nbt, interp, energy)
    # Define repr explicitly as a method to ensure it's picked up correctly
    def __repr__(self):
        return precursor_repr(self)

@dataclass
class DelayedNeutronData:
    """Container for all delayed neutron precursor groups."""
    has_delayed_neutron_data: bool = False  # True if BDD block is present
    precursors: List[DelayedNeutronPrecursor] = field(default_factory=list)
    
    def get_precursor_probability(self, group_idx: int, energy: float) -> Optional[float]:
        """
        Get the probability for a specific precursor group at the given energy.
        
        Parameters
        ----------
        group_idx : int
            Index of the precursor group (0-based)
        energy : float
            Energy in MeV
            
        Returns
        -------
        float or None
            The probability value, or None if the group doesn't exist
        """
        if not self.has_delayed_neutron_data or group_idx < 0 or group_idx >= len(self.precursors):
            return None
        
        return self.precursors[group_idx].evaluate(energy)
    
    def get_decay_constant(self, group_idx: int) -> Optional[float]:
        """
        Get the decay constant for a specific precursor group.
        
        Parameters
        ----------
        group_idx : int
            Index of the precursor group (0-based)
            
        Returns
        -------
        float or None
            The decay constant, or None if the group doesn't exist
        """
        if not self.has_delayed_neutron_data or group_idx < 0 or group_idx >= len(self.precursors):
            return None
        
        precursor = self.precursors[group_idx]
        return precursor.decay_constant if precursor.decay_constant is not None else None
        
    # Define repr explicitly as a method to ensure it's picked up correctly
    def __repr__(self):
        return delayed_neutron_data_repr(self)

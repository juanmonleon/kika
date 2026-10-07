"""Explicit incident-neutron conventions; energies eV, radii fm, outputs barns."""
from dataclasses import dataclass
import math

from kika._constants import NEUTRON_MASS_AMU, NEUTRON_MASS_MEV


@dataclass(frozen=True)
class NeutronContext:
    """Target/neutron mass ratio and target spin (in hbar).

    The caller supplies these from the model's particle data. No ENDF provenance
    is used to infer physics. Constants are recorded and may be matched to an
    external processor in a validation run.
    """

    atomic_weight_ratio: float
    target_spin: float
    neutron_mass_mev: float = NEUTRON_MASS_MEV
    neutron_mass_amu: float = NEUTRON_MASS_AMU
    hbar_c_mev_fm: float = 197.3269804

    def __post_init__(self):
        for name in ("atomic_weight_ratio", "neutron_mass_mev", "neutron_mass_amu",
                     "hbar_c_mev_fm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not math.isfinite(self.target_spin) or self.target_spin < 0
                or not float(2 * self.target_spin).is_integer()):
            raise ValueError("target_spin must be a nonnegative integer or half-integer")

    @property
    def k_squared_per_ev(self):
        """Lab energy includes both reduced-mass and lab-to-CM factors."""
        ratio = self.atomic_weight_ratio / (1 + self.atomic_weight_ratio)
        return 2 * self.neutron_mass_mev * 1e-6 * ratio**2 / self.hbar_c_mev_fm**2

    @property
    def mass_channel_radius_fm(self):
        """ENDF-102 D.0, using target mass in atomic mass units."""
        return 1.23 * (self.neutron_mass_amu * self.atomic_weight_ratio)**(1/3) + 0.8

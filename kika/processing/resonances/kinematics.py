"""Two-body relativistic channel kinematics, with explicit mass/Q consistency."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RelativisticPair:
    incident_mass: float
    target_mass: float
    mass_a: float
    mass_b: float
    q_ev: float
    hbar_c: float

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.incident_mass,self.target_mass,self.mass_a,self.mass_b,self.hbar_c)) or not np.isfinite(self.q_ev):
            raise ValueError('relativistic pair requires finite positive masses and finite Q')
        expected=self.incident_mass+self.target_mass-self.q_ev*1e-6
        if not np.isclose(self.mass_a+self.mass_b,expected,rtol=2e-13,atol=1e-10):
            raise ValueError('relativistic pair masses and Q do not conserve rest energy')

    @property
    def threshold(self):
        total=self.incident_mass+self.target_mass
        return -self.q_ev*(2*total-self.q_ev*1e-6)/(2*self.target_mass)

    def energy(self,lab_ev):
        total=self.incident_mass+self.target_mass
        lab=np.asarray(lab_ev,dtype=float)
        invariant=np.sqrt(total*total+2*self.target_mass*lab*1e-6)
        # The difference from the declared threshold avoids cancellation at
        # tiny E_cm and enforces its exact floating-point zero.
        return 2*self.target_mass*(lab-self.threshold)/(invariant+total-self.q_ev*1e-6)

    def k_squared(self,channel_ev):
        t=np.asarray(channel_ev,dtype=float)*1e-6
        a,b=self.mass_a,self.mass_b;total=a+b
        if np.any(t<=-2*min(a,b)):
            raise ValueError('closed relativistic continuation crosses a pair branch point')
        return t*(t+2*total)*(t+2*a)*(t+2*b)/(4*(t+total)**2*self.hbar_c**2)

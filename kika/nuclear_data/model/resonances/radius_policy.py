"""Physical radius policy, independent of the source-format flags."""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from . import ScatteringRadius


@dataclass
class RadiusPolicy:
    """Channel radius for P/S and phase radius for hard-sphere scattering.

    ``channelMode`` is mass, phase, or constant. A phase function is local to
    its region; ``None`` inherits the formalism/region scattering radius.
    All lengths are fm. This can express ENDF NAPS=2 without overloading a bool.
    """
    channelMode: str = "phase"
    channelRadius: Optional[float] = None
    phaseRadius: Optional["ScatteringRadius"] = None

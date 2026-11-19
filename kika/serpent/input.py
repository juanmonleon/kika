"""Data models for a parsed Serpent input file.

Provides dataclasses that represent the key cards of a Serpent input:
materials, detectors, energy grids, time grids, and sensitivity configuration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnergyGrid:
    """Energy binning definition (``ene`` card).

    A user gets it from ``SerpentInput.energy_grids``.

    Attributes
    ----------
    name : str
        Name given on the card.
    grid_type : int
        The type number of the card: 1 arbitrary edges, 2 uniform linear, 3 uniform
        logarithmic, 4 predefined. Any other number is read as arbitrary edges.
    boundaries : numpy.ndarray
        Bin edges. For type 1 (and any type other than 2, 3 and 4), the numbers on the
        card as written. For types 2 and 3, the ``N + 1`` edges that kika computes
        with ``numpy.linspace`` or ``numpy.geomspace`` from ``N``, ``Emin`` and
        ``Emax`` of the card. Empty for type 4: the parser stores no edges for it.
        Serpent writes energies in MeV; kika does not convert them.
    """
    name: str
    grid_type: int  # 1=arbitrary, 2=uniform linear, 3=uniform log, 4=predefined
    boundaries: np.ndarray  # energy bin edges in MeV

    @property
    def n_bins(self) -> int:
        """Number of bins, one less than the number of edges (0 if there are none)."""
        return max(0, len(self.boundaries) - 1)

    def __repr__(self) -> str:
        return f"EnergyGrid('{self.name}', type={self.grid_type}, bins={self.n_bins})"


@dataclass
class TimeGrid:
    """Time binning definition (``tme`` card).

    A user gets it from ``SerpentInput.time_grids``.

    Attributes
    ----------
    name : str
        Name given on the card.
    grid_type : int
        Card type: 1 arbitrary edges, 2 uniform linear, 3 uniform logarithmic. Any
        other type is read as arbitrary edges.
    boundaries : numpy.ndarray
        Bin edges. For type 1 (and any other type), the numbers on the card as
        written. For types 2 and 3, the ``N + 1`` edges that kika computes with
        ``numpy.linspace`` or ``numpy.geomspace`` from ``N`` and the two limits of
        the card. Serpent writes times in seconds; kika does not convert them.
    """
    name: str
    grid_type: int  # 1=arbitrary, 2=uniform linear, 3=uniform log
    boundaries: np.ndarray  # seconds

    @property
    def n_bins(self) -> int:
        """Number of bins, one less than the number of edges (0 if there are none)."""
        return max(0, len(self.boundaries) - 1)

    def __repr__(self) -> str:
        return f"TimeGrid('{self.name}', type={self.grid_type}, bins={self.n_bins})"


@dataclass
class DetectorDefinition:
    """Detector definition (``det`` card), with its core options only.

    A user gets it from ``SerpentInput.detectors``.

    Attributes
    ----------
    name : str
        Detector name.
    particle : str
        ``"n"`` (neutron) or ``"p"`` (photon); ``"n"`` unless the card names a photon.
    energy_grid : str or None
        Name of the ``ene`` grid (the ``de`` option); a key of
        ``SerpentInput.energy_grids``.
    time_grid : str or None
        Name of the ``tme`` grid (the ``di`` option); a key of
        ``SerpentInput.time_grids``.
    volume : float or None
        The ``dv`` option.
    materials : list of str
        Material names of the ``dm`` options.
    cells : list of int
        Cell numbers of the ``dc`` options.
    response_mt : list of int
        MT numbers of the ``dr`` option.
    extra_options : dict
        Reserved for options the parser does not interpret. The parser skips them and
        leaves this empty.
    """
    name: str
    particle: str = "n"
    energy_grid: Optional[str] = None   # name of ``ene`` grid (de option)
    time_grid: Optional[str] = None     # name of ``tme`` grid (di option)
    volume: Optional[float] = None      # dv option
    materials: List[str] = field(default_factory=list)   # dm options
    cells: List[int] = field(default_factory=list)       # dc options
    response_mt: List[int] = field(default_factory=list) # dr option MT numbers
    extra_options: Dict[str, Any] = field(default_factory=dict)  # unparsed options

    def __repr__(self) -> str:
        parts = [f"DetectorDefinition('{self.name}'"]
        if self.energy_grid:
            parts.append(f"ene='{self.energy_grid}'")
        if self.time_grid:
            parts.append(f"tme='{self.time_grid}'")
        if self.materials:
            parts.append(f"mats={self.materials}")
        return ", ".join(parts) + ")"


@dataclass
class SensResponse:
    """A single ``sens resp`` sub-card.

    A user gets it from ``SensitivityConfig.responses``.

    Attributes
    ----------
    response_type : str
        ``"keff"``, ``"leff"``, ``"beff"``, ``"lambda"``, ``"void"`` or ``"ratio"``
        (``ratio`` and ``detratio`` cards both give ``"ratio"``).
    name : str or None
        Name given on a ratio card.
    det1, det2 : str or None
        Names of the two detectors of a ratio card.
    material : str or None
        Material name of a void card.
    """
    response_type: str  # "keff", "leff", "beff", "lambda", "void", "ratio"
    name: Optional[str] = None       # user-given name (for ratio)
    det1: Optional[str] = None       # detector 1 (for ratio)
    det2: Optional[str] = None       # detector 2 (for ratio)
    material: Optional[str] = None   # material name (for void)

    def __repr__(self) -> str:
        if self.response_type == "ratio" and self.det1 and self.det2:
            return f"SensResponse('ratio', {self.name}: {self.det1}/{self.det2})"
        if self.response_type == "void" and self.material:
            return f"SensResponse('void', mat='{self.material}')"
        return f"SensResponse('{self.response_type}')"


@dataclass
class SensPerturbation:
    """A single ``sens pert`` sub-card.

    A user gets it from ``SensitivityConfig.perturbations``.

    Attributes
    ----------
    pert_type : str
        ``"xs"``, ``"chi"``, ``"nubar"``, ``"eleg"``, ``"temperature"``, ``"elamu"``
        or ``"inlmu"``.
    mode : str or None
        For ``xs``, the token after ``xs`` in lower case, whatever it is (it is not
        checked to be a mode word); ``"all"`` when there is no token. None for the
        other types.
    nuclides : list of str
        For ``xs``, the nuclide names after the mode, up to the next card keyword.
    legendre_order : int or None
        For ``eleg``, the order on the card (0 when absent). None for the other types.
    """
    pert_type: str  # "xs", "chi", "nubar", "eleg", "temperature", "elamu", "inlmu"
    mode: Optional[str] = None        # for xs: "all", "allmt", "realist"
    nuclides: List[str] = field(default_factory=list)
    legendre_order: Optional[int] = None  # for eleg

    def __repr__(self) -> str:
        if self.pert_type == "xs":
            return f"SensPerturbation('xs', mode='{self.mode}', nuclides={len(self.nuclides)})"
        return f"SensPerturbation('{self.pert_type}')"


@dataclass
class SensitivityConfig:
    """All ``sens`` cards of an input combined.

    A user gets it from ``SerpentInput.sensitivity``.

    Attributes
    ----------
    responses : list of SensResponse
        From the ``sens resp`` cards, in file order.
    perturbations : list of SensPerturbation
        From the ``sens pert`` cards, in file order.
    energy_grid : str or None
        Name of the ``ene`` grid given by ``sens opt egrid``.
    """
    responses: List[SensResponse] = field(default_factory=list)
    perturbations: List[SensPerturbation] = field(default_factory=list)
    energy_grid: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"SensitivityConfig(responses={len(self.responses)}, "
            f"perturbations={len(self.perturbations)})"
        )


@dataclass
class SerpentInput:
    """A parsed Serpent input file.

    A user gets it from :func:`read_serpent_input`.

    Attributes
    ----------
    materials
        The materials of the deck, as read by the library's Serpent material parser
        (a ``MaterialCollection``); None when built by hand.
    detectors : dict of str to DetectorDefinition
        By detector name.
    energy_grids : dict of str to EnergyGrid
        By grid name.
    time_grids : dict of str to TimeGrid
        By grid name.
    sensitivity : SensitivityConfig or None
        None when no response and no perturbation was parsed: no ``sens resp`` or
        ``sens pert`` card, one of an unrecognised type, or one that raised an
        ``IndexError`` or ``ValueError`` (the reader skips it). A deck with only
        ``sens opt egrid`` gives None, so its energy grid is not returned.
    title : str or None
        Not filled by the reader.
    metadata : dict
        Not filled by the reader.
    """
    materials: Any = None  # MaterialCollection from kika.materials
    detectors: Dict[str, DetectorDefinition] = field(default_factory=dict)
    energy_grids: Dict[str, EnergyGrid] = field(default_factory=dict)
    time_grids: Dict[str, TimeGrid] = field(default_factory=dict)
    sensitivity: Optional[SensitivityConfig] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = ["SerpentInput("]
        if self.materials:
            parts.append(f"  materials={len(self.materials)}")
        parts.append(f"  detectors={len(self.detectors)}")
        parts.append(f"  energy_grids={len(self.energy_grids)}")
        parts.append(f"  time_grids={len(self.time_grids)}")
        if self.sensitivity:
            parts.append(f"  sensitivity={self.sensitivity}")
        parts.append(")")
        return "\n".join(parts)

    def summary(self) -> str:
        """Compact text summary."""
        lines = []
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.materials:
            lines.append(f"Materials: {len(self.materials)}")
        if self.detectors:
            lines.append(f"Detectors: {', '.join(self.detectors.keys())}")
        if self.energy_grids:
            for name, eg in self.energy_grids.items():
                lines.append(f"Energy grid '{name}': {eg.n_bins} bins, type {eg.grid_type}")
        if self.time_grids:
            for name, tg in self.time_grids.items():
                lines.append(f"Time grid '{name}': {tg.n_bins} bins, type {tg.grid_type}")
        if self.sensitivity:
            lines.append(f"Sensitivity: {len(self.sensitivity.responses)} responses, "
                         f"{len(self.sensitivity.perturbations)} perturbation types")
        return "\n".join(lines)

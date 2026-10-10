"""A photo-atomic ACE table (``.84p``, ``.12p``, ``.14p``...), decoded.

A photo-atomic table is not a neutron table with fewer reactions: its NXS and
JXS arrays mean different things (NXS(2) is Z, NXS(3) the photon grid
length; JXS(2) points at the incoherent scattering function, not at MTR), so
it gets its own class and :func:`kika.ace.read_ace` returns this one for a
table whose ZAID ends in ``p``.

**Units.** ACE stores photon energies in MeV, most of them as natural logs;
every energy here has been converted to **eV** so it can be laid next to an
ENDF MF23 section without a second thought. Cross sections are in **barns**.
The form-factor abscissae are ``v = sin(θ/2)/λ`` in 1/Å, as in MF27, and the
Compton profiles' ``pz`` is the projected electron momentum in units of
m_e·c, both as the table writes them.

**Two layouts, one class.** The old MCPLIB tables (``.01p`` to ``.84p``)
tabulate the scattering functions on fixed grids that the file does not
carry; the EPRDATA tables (``.12p``, ``.14p``) write their own grids and add
subshell and relaxation blocks (Hughes, LA-UR-13-27377, Tables 1-4). NXS(6)
says which: 0 for MCPLIB, nonzero for EPRDATA. ``layout`` keeps the answer.

**What is not decoded.** The electron blocks of EPRDATA (JXS(19)-JXS(26):
electron cross sections, excitation, elastic angular tables,
electroionisation, bremsstrahlung) are photon-transport irrelevant and are
listed in ``unparsed_blocks`` instead of being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

#: MeV → eV.
EV_PER_MEV = 1.0e6

#: ENDF subshell designators (1 = K). The same list as
#: :data:`kika.endf.classes.mf23.base.SUBSHELL_NAMES`, repeated so that the ACE
#: package does not import the ENDF one for a table of names.
SUBSHELL_NAMES = (
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11",
    "Q1", "Q2", "Q3",
)


def subshell_name(designator: int) -> Optional[str]:
    if 1 <= designator <= len(SUBSHELL_NAMES):
        return SUBSHELL_NAMES[designator - 1]
    return None


@dataclass
class ComptonProfile:
    """One shell's Compton profile J(pz) for Doppler broadening (SWD block)."""

    interpolation: int
    pz: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray


@dataclass
class RelaxationTransitionAce:
    """One XPROB entry: fill from ``designator_1``, eject from ``designator_2`` (0: a photon)."""

    designator_1: int
    designator_2: int
    energy: float            # eV
    cumulative_probability: float


@dataclass
class PhotoatomicAce:
    """The decoded blocks of a photo-atomic ACE table. See the module docstring for units."""

    filename: Optional[str] = None
    header: object = None
    xss_data: Optional[np.ndarray] = None
    #: ``"mcplib"`` (fixed form-factor grids) or ``"eprdata"`` (grids in the file).
    layout: str = "mcplib"
    #: NXS(6) as written: 0 for MCPLIB, the EPRDATA format flag otherwise.
    format_flag: int = 0
    z: int = 0

    # ESZG (energies and the four partial cross sections) and LHNM
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))
    incoherent: np.ndarray = field(default_factory=lambda: np.empty(0))
    coherent: np.ndarray = field(default_factory=lambda: np.empty(0))
    photoelectric: np.ndarray = field(default_factory=lambda: np.empty(0))
    pair_production: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Average heating number per collision, eV (LHNM).
    heating: np.ndarray = field(default_factory=lambda: np.empty(0))

    # JINC and JCOH
    incoherent_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    incoherent_scattering_function: np.ndarray = field(default_factory=lambda: np.empty(0))
    coherent_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: The cumulative (integrated) coherent form factor, what MCNP samples from.
    coherent_integrated_form_factor: np.ndarray = field(default_factory=lambda: np.empty(0))
    coherent_form_factor: np.ndarray = field(default_factory=lambda: np.empty(0))

    # JFLO: the "old method" fluorescence data, NXS(4) edges
    fluorescence_edge_energies: np.ndarray = field(default_factory=lambda: np.empty(0))   # eV
    fluorescence_ejection_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    fluorescence_yields: np.ndarray = field(default_factory=lambda: np.empty(0))
    fluorescence_energies: np.ndarray = field(default_factory=lambda: np.empty(0))        # eV

    # Compton Doppler broadening (JXS(6)-JXS(10)), NXS(5) shells
    shell_electrons: np.ndarray = field(default_factory=lambda: np.empty(0))
    shell_binding_energies: np.ndarray = field(default_factory=lambda: np.empty(0))       # eV
    #: LPIPS as written. Differential in MCPLIB04 and EPRDATA12, cumulative in
    #: MCPLIB63/84 (LA-UR-13-27377 Note 3), so it is not normalised here.
    shell_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    compton_profiles: List[ComptonProfile] = field(default_factory=list)

    # EPRDATA only: SUBSH (JXS(11)-(15)), SPHEL (16), RELO/XPROB (17-18)
    subshell_designators: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    subshell_electrons: np.ndarray = field(default_factory=lambda: np.empty(0))
    subshell_binding_energies: np.ndarray = field(default_factory=lambda: np.empty(0))   # eV
    subshell_vacancy_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    subshell_num_transitions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    #: ENDF designator (1 = K) → σ(E) in barns on ``energies``.
    subshell_photoelectric: Dict[int, np.ndarray] = field(default_factory=dict)
    #: ENDF designator → the transitions that fill a vacancy there.
    relaxation: Dict[int, List[RelaxationTransitionAce]] = field(default_factory=dict)

    #: Blocks the table carries that are not decoded, by their manual name.
    unparsed_blocks: List[str] = field(default_factory=list)

    @property
    def zaid(self) -> Optional[int]:
        return getattr(self.header, "zaid", None)

    @property
    def extension(self) -> Optional[str]:
        return getattr(self.header, "extension", None)

    @property
    def total(self) -> np.ndarray:
        """The sum of the four partials. ACE carries no total; MCNP sums them too."""
        return self.incoherent + self.coherent + self.photoelectric + self.pair_production

    def cross_sections(self, subshells: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """``{name: (E in eV, σ in barns)}`` on the table's common grid.

        Names: ``total``, ``coherent``, ``incoherent``, ``photoelectric`` and
        ``pair_production`` (pair and triplet together, which ACE does not
        separate). With *subshells*, EPRDATA's per-subshell photoelectric
        cross sections are added as ``photoelectric_K``, ``photoelectric_L1``...
        """
        E = self.energies
        out = {
            "total": (E, self.total),
            "coherent": (E, self.coherent),
            "incoherent": (E, self.incoherent),
            "photoelectric": (E, self.photoelectric),
            "pair_production": (E, self.pair_production),
        }
        if subshells:
            for designator, sigma in self.subshell_photoelectric.items():
                out[f"photoelectric_{subshell_name(designator) or designator}"] = (E, sigma)
        return out

    def __repr__(self) -> str:
        ext = self.extension or ""
        return (f"PhotoatomicAce({self.zaid}{ext}, Z={self.z}, layout={self.layout!r}, "
                f"NES={self.energies.size}, shells={len(self.compton_profiles)}, "
                f"subshells={len(self.subshell_photoelectric)})")

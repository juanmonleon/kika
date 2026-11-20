"""A photonuclear ACE table (``.NNu``), decoded.

A photonuclear table describes a photon hitting a nucleus: the reaction cross
sections on one energy grid, their sum, a heating number, and for each kind of
particle the reactions emit (neutron, photon, proton, deuteron, triton,
helium-3, alpha) its production cross section, its heating, and per reaction
its yield, angular distribution and energy law. There are no resonance, ν̄ or
delayed blocks, which is why it is not an :class:`~kika.ace.classes.ace.Ace`
with fewer blocks: its NXS and JXS mean other things.

**Layout** (NJOY2016 ``acepn.f90``, ``phnout``).

- NXS(1) the length of XSS, NXS(2) ZA, NXS(3) NES the grid size, NXS(4) NTR
  the reactions, NXS(5) NTYPE the particle kinds, NXS(6) = 2 and NXS(7) = 12
  the size of an IXS entry, NXS(9) the isomeric state, NXS(10) Z and NXS(11) A
  (zero in tables older than NJOY2016), NXS(16) = 1.
- JXS(1) ESZ, the energies (MeV), followed by the total; JXS(2) TOT; JXS(3) NON
  the nonelastic cross section (equal to TOT when there is no elastic);
  JXS(4) ELS the elastic (0 if none); JXS(5) THN the heating number
  (MeV per collision); JXS(6) MTR, JXS(7) LQR (Q, MeV), JXS(8) LSIG, JXS(9) SIG
  (each reaction IE, NE, σ(NE) from grid point IE, located by LSIG relative to
  SIG); JXS(10) IXSA, NTYPE entries of 12 words; JXS(11) the word after them.
- Each IXS entry: IPT (1 n, 2 γ, 9 p, 31 d, 32 t, 33 ³He, 34 α), NTRP, then the
  absolute locations of PXS and PHN (each IE, NE and NE values on the main
  grid), MTRP, TYRP (+1 lab, −1 centre of mass), LSIGP, SIGP, LANDP, ANDP, LDLWP
  and DLWP. Inside SIGP, ANDP and DLWP the locators are relative to the block,
  as in neutron tables.
- A SIGP record is MFTYPE, then for MFTYPE 6, 12 or 16 MTMULT, NR, NBT, INT, NE,
  E(NE), Y(NE), and for MFTYPE 13 IE, NE, σ(NE).
- LANDP is 0 for isotropic, −1 when the energy law carries the angle, else
  the location of an angular record (NE, E, LC, tables) as in neutron ANDH.
- DLWP holds a law chain per reaction, as neutron DLWH.

**Units.** The grid, Q values and heating numbers here are in **eV** (eV per
collision for heating), cross sections in **barns**. The angular records and
energy laws are kika's neutron-table objects and keep the table's MeV, as they
do in an :class:`~kika.ace.classes.ace.Ace`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

#: MeV → eV.
EV_PER_MEV = 1.0e6

#: IPT of each particle a photonuclear table can describe, and its name in kika.
PARTICLE_IPT = {1: "n", 2: "photon", 9: "H1", 31: "H2", 32: "H3", 33: "He3", 34: "He4"}

#: Words in one IXS entry (NXS(7)).
IXS_ENTRY_WORDS = 12


@dataclass
class PhotonuclearReaction:
    """One reaction: σ on the main grid from point ``first_index`` (1-based) on.

    Attributes
    ----------
    mt : int
        The reaction number.
    q_value : float
        The Q-value in eV.
    first_index : int
        The 1-based point of the main grid (``PhotonuclearAce.energies``) where
        ``cross_section`` starts; the cross section is zero below it.
    cross_section : numpy.ndarray
        Cross sections in barns, for the grid points from ``first_index`` on.
    """

    mt: int
    q_value: float                 # eV
    first_index: int
    cross_section: np.ndarray      # b

    def on_grid(self, n: int) -> np.ndarray:
        """σ on all ``n`` points of the main grid, zero below the first.

        Parameters
        ----------
        n : int
            The number of points of the main grid.

        Returns
        -------
        numpy.ndarray
            ``n`` cross sections in barns: zero below ``first_index``, then
            ``cross_section``.

        Raises
        ------
        ValueError
            If ``cross_section`` runs past the ``n`` points (numpy refuses the
            assignment with a shape mismatch).
        """
        out = np.zeros(n)
        out[self.first_index - 1:self.first_index - 1 + self.cross_section.size] = self.cross_section
        return out


@dataclass
class ProductYield:
    """A SIGP record.

    MFTYPE 6, 12 or 16: the multiplicity of the product in reaction ``mt_mult``,
    a table of ``energies`` (eV) and ``values`` with ``breakpoints`` and
    ``interpolation`` (both empty for one lin-lin region). MFTYPE 13: a
    production cross section, ``values`` (b) from main-grid point
    ``first_index`` on.

    Attributes
    ----------
    mftype : int
        6, 12, 13 or 16.
    mt_mult : int
        The reaction whose multiplicity this is (MFTYPE 6, 12, 16); 0 for 13.
    breakpoints, interpolation : list of int
        NBT and INT of the table (MFTYPE 6, 12, 16); empty otherwise.
    energies : numpy.ndarray or None
        Incident energies in eV (MFTYPE 6, 12, 16); ``None`` for 13.
    values : numpy.ndarray or None
        The multiplicities (dimensionless) for MFTYPE 6, 12, 16, or the
        production cross section in barns for 13.
    first_index : int
        The 1-based main-grid point where ``values`` starts (MFTYPE 13 only; 0
        otherwise).
    """

    mftype: int
    mt_mult: int = 0
    breakpoints: List[int] = field(default_factory=list)
    interpolation: List[int] = field(default_factory=list)
    energies: Optional[np.ndarray] = None   # eV
    values: Optional[np.ndarray] = None
    first_index: int = 0


@dataclass
class ProductReaction:
    """How one reaction emits the particle: frame, yield, angle and energy law.

    ``angular_locator`` is the table's LANDP word kind: 0 isotropic, −1 the
    angle is in the energy law, 1 a record (``angular``).

    Attributes
    ----------
    mt : int
        The reaction number.
    frame : int
        The table's TYRP: +1 for the laboratory frame, -1 for the centre of mass.
        The decoder refuses any other value.
    yield_ : ProductYield
        The product's yield record (SIGP).
    angular_locator : int
        0, -1 or 1 as above.
    angular : object or None
        The angular distribution record when ``angular_locator`` is 1 (as a
        neutron table's angular distribution, with its energies in MeV); ``None``
        otherwise.
    laws : list
        The chain of energy laws of the reaction, as in a neutron table's DLW
        (energies in MeV).
    """

    mt: int
    frame: int                     # +1 lab, −1 centre of mass
    yield_: ProductYield
    angular_locator: int
    angular: object = None
    laws: list = field(default_factory=list)


@dataclass
class ParticleProduction:
    """What a table says about one kind of emitted particle.

    Attributes
    ----------
    ipt : int
        The particle type: 1 neutron, 2 photon, 9 proton, 31 deuteron, 32 triton,
        33 helium-3, 34 alpha.
    production_first_index : int
        The 1-based main-grid point where ``production`` starts.
    production : numpy.ndarray
        The production cross section in barns, from ``production_first_index`` on.
    heating_first_index : int
        The 1-based main-grid point where ``heating`` starts.
    heating : numpy.ndarray
        The heating number in eV per collision, from ``heating_first_index`` on.
    reactions : list of ProductReaction
        The reactions that emit the particle, in the order of the table.
    """

    ipt: int
    production_first_index: int
    production: np.ndarray         # b, on the main grid from the first index
    heating_first_index: int
    heating: np.ndarray            # eV per collision
    reactions: List[ProductReaction] = field(default_factory=list)

    @property
    def particle(self) -> str:
        """The kika name of the particle.

        One of ``n``, ``photon``, ``H1``, ``H2``, ``H3``, ``He3`` and ``He4``; an IPT
        outside these gives ``"IPT <number>"``.
        """
        return PARTICLE_IPT.get(self.ipt, f"IPT {self.ipt}")

    @property
    def mt_numbers(self) -> List[int]:
        """The reaction numbers of ``reactions``, in order."""
        return [r.mt for r in self.reactions]


@dataclass
class PhotonuclearAce:
    """The decoded contents of a photonuclear ACE table. See the module docstring for units.

    :func:`~kika.ace.read_ace` returns one for a file whose ZAID suffix ends in ``u``.

    Attributes
    ----------
    filename : str or None
        The path the table was read from.
    header : object or None
        The parsed ACE header.
    xss_data : numpy.ndarray or None
        The XSS array as read (MeV and barns, 1-based).
    xss_int_words : numpy.ndarray or None
        The integer-spelling mask of the XSS.
    za : int or None
        1000·Z + A of the target (NXS(2)).
    isomeric_state : int
        NXS(9).
    energies : numpy.ndarray or None
        The main energy grid in eV.
    total : numpy.ndarray or None
        The total cross section in barns, on the main grid.
    nonelastic : numpy.ndarray or None
        The nonelastic cross section in barns; set only when the table has an
        elastic block, else ``None``.
    elastic : numpy.ndarray or None
        The elastic cross section in barns; ``None`` when the table has none.
    heating : numpy.ndarray or None
        The heating number in eV per collision, on the main grid; ``None`` when
        the table has none.
    reactions : list of PhotonuclearReaction
        The reactions in the order of the table.
    particles : list of ParticleProduction
        One block per emitted particle kind.
    """

    filename: Optional[str] = None
    header: object = None
    xss_data: Optional[np.ndarray] = None
    xss_int_words: Optional[np.ndarray] = None
    za: Optional[int] = None
    isomeric_state: int = 0
    energies: Optional[np.ndarray] = None      # eV
    total: Optional[np.ndarray] = None         # b
    nonelastic: Optional[np.ndarray] = None    # b, only when the table has an elastic
    elastic: Optional[np.ndarray] = None       # b
    heating: Optional[np.ndarray] = None       # eV per collision
    reactions: List[PhotonuclearReaction] = field(default_factory=list)
    particles: List[ParticleProduction] = field(default_factory=list)

    @property
    def mt_numbers(self) -> List[int]:
        """The reaction numbers of ``reactions``, in order."""
        return [r.mt for r in self.reactions]

    def reaction(self, mt: int) -> PhotonuclearReaction:
        """The reaction numbered ``mt``.

        Parameters
        ----------
        mt : int
            A reaction number.

        Returns
        -------
        PhotonuclearReaction
            The first reaction with that number.

        Raises
        ------
        KeyError
            If the table has no such reaction.
        """
        for r in self.reactions:
            if r.mt == int(mt):
                return r
        raise KeyError(f"no reaction {mt} in the table")

    def particle(self, name_or_ipt) -> ParticleProduction:
        """The block of one emitted particle, by kika name (``'n'``, ``'H1'``…) or IPT.

        Parameters
        ----------
        name_or_ipt : str or int
            The kika name (``ParticleProduction.particle``) or the IPT number.

        Returns
        -------
        ParticleProduction
            The first block that matches.

        Raises
        ------
        KeyError
            If the table emits no such particle.
        """
        for p in self.particles:
            if p.ipt == name_or_ipt or p.particle == name_or_ipt:
                return p
        raise KeyError(f"the table emits no {name_or_ipt}")

    def __repr__(self) -> str:
        name = f"{self.header.zaid}{self.header.extension}" if self.header is not None else "?"
        kinds = ", ".join(p.particle for p in self.particles)
        return f"PhotonuclearAce({name}, {len(self.reactions)} reactions, emits {kinds})"

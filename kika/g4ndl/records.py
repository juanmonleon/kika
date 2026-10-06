"""What one elastic G4NDL file says, field by field, before any physics.

These are the intermediate records the parsers in :mod:`kika.g4ndl.parse`
produce and the model adapter (roadmap Phase 4) consumes. They are **not** a
second public representation of the data: they keep every token the consumer
reads — bookkeeping integers, temperatures, ``tempdep``, both frame flags of
``repFlag=0`` — and interpret none of them, so that nothing is lost between
the file and the decision of what it means.

Two things they never do, because the files need them not to:

* **Collapse or sort energies.** Repeated energies are real (print collisions
  in the cross sections, repeated incident energies in the final states;
  ``G4NDL_token_spec.md`` §7); every array is kept in file order.
* **Merge interpolation into data.** A region table (``NBT``, ``INT``) stays
  beside the values it governs, exactly as the file has it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "AngularBlock", "CrossSectionRecord", "ElasticFSRecord", "Interpolation",
    "LegendreRecord", "TabulatedRecord",
    "REP_ISOTROPIC", "REP_LEGENDRE", "REP_TABULATED", "REP_MIXED",
    "FRAME_LAB", "FRAME_CM",
]

REP_ISOTROPIC, REP_LEGENDRE, REP_TABULATED, REP_MIXED = 0, 1, 2, 3
FRAME_LAB, FRAME_CM = 1, 2


@dataclass(frozen=True)
class Interpolation:
    """One ``G4InterpolationManager`` record: ``NR``, then ``NR`` pairs ``(NBT, INT)``.

    ``nbt`` is **cumulative**, ENDF style: region ``i`` ends at point
    ``nbt[i]`` (1-based). Codes follow ENDF: 1 histogram, 2 lin-lin,
    3 lin-log (y linear in ln x), 4 log-lin (ln y linear in x), 5 log-log.
    """

    nbt: Tuple[int, ...]
    codes: Tuple[int, ...]

    @property
    def nRegions(self) -> int:
        return len(self.nbt)


@dataclass(frozen=True, eq=False)
class CrossSectionRecord:
    """``Elastic/CrossSection/<name>``: bookkeeping, then ``N`` pairs ``(E, σ)``.

    ``energy`` in eV, ``sigma`` in barn, pointwise at 0 K, in file order with
    repeated energies kept. The file carries no interpolation record; the
    consumer interpolates lin-lin.
    """

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    bookkeeping: Tuple[int, int]
    energy: np.ndarray
    sigma: np.ndarray

    def __len__(self) -> int:
        return len(self.energy)


@dataclass(frozen=True, eq=False)
class LegendreRecord:
    """One incident energy of a Legendre block.

    ``coefficients`` are ``a_1 … a_NL`` exactly as stored; ``a_0 = 1`` is
    implicit in the file and **not** prepended here. The distribution is
    ``p(μ) = Σ_l (2l+1)/2 a_l P_l(μ)``.
    """

    temperature: float
    energy: float
    tempdep: int
    coefficients: np.ndarray


@dataclass(frozen=True, eq=False)
class TabulatedRecord:
    """One incident energy of a tabulated block: ``p(μ)`` on a μ grid.

    ``probability`` is a density in μ (not in θ, not a differential cross
    section), with its own interpolation record in μ.
    """

    temperature: float
    energy: float
    tempdep: int
    interpolation: Interpolation
    mu: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True, eq=False)
class AngularBlock:
    """``NE``, an interpolation record in incident energy, then ``NE`` records."""

    interpolation: Interpolation
    records: Tuple = ()

    @property
    def energies(self) -> np.ndarray:
        """Incident energies in file order, repeats kept."""
        return np.array([r.energy for r in self.records], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True, eq=False)
class ElasticFSRecord:
    """``Elastic/FS/<name>``, as ``G4ParticleHPElasticFS::Init`` reads it.

    ``repFlag`` selects the blocks present: 0 neither (isotropic), 1
    ``legendre``, 2 ``tabulated``, 3 both, Legendre first. ``targetMass`` is
    the consumer's mass ratio (AWR), not the mass number. ``frameFlag`` is
    1 laboratory, 2 centre of mass; ``frameFlag2`` is the second frame flag
    the consumer reads for ``repFlag=0`` only (``:209``). It **overwrites**
    the first there (``theData >> frameFlag``), so for an isotropic file
    ``frameFlag2`` is the one Geant4 uses; both are kept.
    """

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    repFlag: int
    targetMass: float
    frameFlag: int
    frameFlag2: Optional[int] = None
    legendre: Optional[AngularBlock] = None
    tabulated: Optional[AngularBlock] = None

    @property
    def transitionEnergy(self) -> Optional[float]:
        """For ``repFlag=3``, the last Legendre energy.

        Geant4 uses Legendre for ``E <= transitionEnergy`` and the table
        above it.
        """
        if self.repFlag != REP_MIXED:
            return None
        return self.legendre.records[-1].energy

    @property
    def temperatures(self) -> np.ndarray:
        return np.array([r.temperature for b in self._blocks() for r in b.records])

    @property
    def tempdeps(self) -> np.ndarray:
        return np.array([r.tempdep for b in self._blocks() for r in b.records], dtype=int)

    def _blocks(self):
        return [b for b in (self.legendre, self.tabulated) if b is not None]

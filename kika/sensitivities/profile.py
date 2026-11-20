"""Format-neutral sensitivity profiles used by UQ calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple

import numpy as np


EnergyUnit = Literal["eV", "MeV"]


def _validated_vector(values: Iterable[float], name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be a one-dimensional vector of length {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass
class SensitivityReaction:
    """The groupwise sensitivities of one nuclide and reaction, with their uncertainty.

    Built by hand, or taken from :attr:`SensitivityProfile.reactions`. It is not
    checked until a :class:`SensitivityProfile` is made from it.

    Attributes
    ----------
    zaid : int
        ZAID of the nuclide.
    mt : int
        MT reaction number.
    sensitivity : numpy.ndarray
        Relative sensitivity of each group, per group (not per unit lethargy), in the
        ascending order of the profile's grid.
    uncertainty : numpy.ndarray, optional
        **Absolute** one-sigma standard deviation of each group, non-negative. None
        means the uncertainty is unknown, which is not the same as zero.
    label : str, optional
        Free label, for example the reaction name.
    """

    zaid: int
    mt: int
    sensitivity: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    label: Optional[str] = None

    def validated(self, n_groups: int) -> "SensitivityReaction":
        """A checked copy, with float arrays of ``n_groups`` finite values.

        Parameters
        ----------
        n_groups
            Number of groups the arrays must have.

        Returns
        -------
        SensitivityReaction
            With ``zaid`` and ``mt`` as integers and copies of the arrays.

        Raises
        ------
        ValueError
            If *sensitivity* or *uncertainty* is not a one-dimensional vector of
            ``n_groups`` finite values, or *uncertainty* has a negative value.
        """
        sensitivity = _validated_vector(self.sensitivity, "sensitivity", n_groups)
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = _validated_vector(self.uncertainty, "uncertainty", n_groups)
            if np.any(uncertainty < 0.0):
                raise ValueError("uncertainty must contain non-negative absolute standard deviations")
        return SensitivityReaction(
            zaid=int(self.zaid),
            mt=int(self.mt),
            sensitivity=sensitivity,
            uncertainty=uncertainty,
            label=self.label,
        )

    @property
    def key(self) -> Tuple[int, int]:
        """The pair ``(zaid, mt)`` that names the profile within a profile."""
        return self.zaid, self.mt


@dataclass
class SensitivityProfile:
    """Validated sensitivity profiles on one ascending energy grid.

    A user gets one from
    :meth:`~kika.sensitivities.sdf.SDFData.to_sensitivity_profile` or builds one by
    hand. Sensitivities are dimensionless relative coefficients, per group, and
    ``uncertainty`` is the absolute one-sigma uncertainty on each of them.
    :func:`~kika.sensitivities.condensation.condense_sensitivity_profile` coarsens the
    grid.
    Construction checks everything below and stores copies.

    Attributes
    ----------
    energy_grid : numpy.ndarray
        Group boundaries in *energy_unit*, ``n + 1`` finite, non-negative values,
        strictly increasing.
    reactions : tuple of SensitivityReaction
        Validated profiles, each with ``n`` values, and no two with the same
        ``(zaid, mt)``.
    energy_unit : {"eV", "MeV"}
        Unit of *energy_grid*.
    response : float, optional
        The response the sensitivities refer to, such as k-eff. Finite when given.
    response_uncertainty : float, optional
        **Absolute** one-sigma uncertainty of *response*, finite and non-negative.
    label : str, optional
        Free label.
    metadata : mapping
        Free information, copied to a ``dict``.

    Raises
    ------
    ValueError
        If *energy_grid* is not a one-dimensional strictly increasing grid of at
        least two finite, non-negative values; *energy_unit* is not ``"eV"`` or
        ``"MeV"``; a profile fails :meth:`SensitivityReaction.validated`; two profiles
        share ``(zaid, mt)``; or *response* or *response_uncertainty* is not valid.
    """

    energy_grid: np.ndarray
    reactions: Tuple[SensitivityReaction, ...]
    energy_unit: EnergyUnit = "MeV"
    response: Optional[float] = None
    response_uncertainty: Optional[float] = None
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = np.asarray(self.energy_grid, dtype=float)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("energy_grid must be a one-dimensional array with at least two boundaries")
        if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
            raise ValueError("energy_grid must contain finite non-negative boundaries")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("energy_grid must be strictly increasing")
        if self.energy_unit not in ("eV", "MeV"):
            raise ValueError("energy_unit must be 'eV' or 'MeV'")

        n_groups = len(grid) - 1
        validated = tuple(reaction.validated(n_groups) for reaction in self.reactions)
        keys = [reaction.key for reaction in validated]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate sensitivity reaction keys: {duplicates}")

        if self.response is not None and not np.isfinite(self.response):
            raise ValueError("response must be finite when provided")
        if self.response_uncertainty is not None:
            if not np.isfinite(self.response_uncertainty) or self.response_uncertainty < 0.0:
                raise ValueError("response_uncertainty must be a finite non-negative absolute sigma")

        self.energy_grid = grid.copy()
        self.reactions = validated
        self.metadata = dict(self.metadata)

    @property
    def n_groups(self) -> int:
        """Number of energy groups, one fewer than the boundaries of ``energy_grid``."""
        return len(self.energy_grid) - 1

    @property
    def reaction_map(self) -> Dict[Tuple[int, int], SensitivityReaction]:
        """The profiles keyed by ``(zaid, mt)``, built anew at each access."""
        return {reaction.key: reaction for reaction in self.reactions}

    def grid_in(self, unit: EnergyUnit) -> np.ndarray:
        """The group boundaries in *unit*, as a new array.

        Parameters
        ----------
        unit : {"eV", "MeV"}
            Unit to convert to, by a factor of ``1e6``.

        Returns
        -------
        numpy.ndarray
            ``n + 1`` boundaries, ascending.

        Raises
        ------
        ValueError
            If *unit* is not ``"eV"`` or ``"MeV"``.
        """
        if unit not in ("eV", "MeV"):
            raise ValueError("unit must be 'eV' or 'MeV'")
        if unit == self.energy_unit:
            return self.energy_grid.copy()
        factor = 1.0e6 if self.energy_unit == "MeV" else 1.0e-6
        return self.energy_grid * factor

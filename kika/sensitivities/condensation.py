"""Explicit, format-neutral condensation of multigroup sensitivities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .profile import EnergyUnit, SensitivityProfile, SensitivityReaction


class CondensationError(ValueError):
    """Raised when a sensitivity profile cannot be condensed exactly."""


@dataclass(frozen=True)
class CondensationReport:
    """What one condensation did: the grids, the matched boundaries and the assumptions.

    The ``report`` of a :class:`CondensationResult`.

    Attributes
    ----------
    source_groups, target_groups : int
        Number of groups before and after.
    source_energy_grid, target_energy_grid : tuple of float
        The boundaries of the two grids, in their own units, ascending.
    source_energy_unit, target_energy_unit : {"eV", "MeV"}
        Units of those grids.
    boundary_indices : tuple of int
        For each target boundary, the index of the matching source boundary:
        ``target_groups + 1`` strictly increasing indices. Target group ``k`` sums the
        source groups ``boundary_indices[k]`` up to ``boundary_indices[k + 1] - 1``.
    energy_rtol : float
        Relative tolerance with which boundaries were taken as equal.
    uncertainty_method : str
        ``"independent source-group quadrature"`` when any profile had an uncertainty,
        otherwise ``"not available"``.
    assumptions : tuple of str
        The statistical assumption made, empty when no profile had an uncertainty.
    max_integral_drift : float
        Largest absolute difference, over the profiles, between the sum of the
        condensed values and the sum of the source values. Zero up to rounding.
    """

    source_groups: int
    target_groups: int
    source_energy_grid: Tuple[float, ...]
    target_energy_grid: Tuple[float, ...]
    source_energy_unit: EnergyUnit
    target_energy_unit: EnergyUnit
    boundary_indices: Tuple[int, ...]
    energy_rtol: float
    uncertainty_method: str
    assumptions: Tuple[str, ...]
    max_integral_drift: float


@dataclass(frozen=True)
class CondensationResult:
    """A condensed profile with the matrix and the report that produced it.

    Returned by
    :func:`~kika.sensitivities.condensation.condense_sensitivity_profile`.

    Attributes
    ----------
    profile : SensitivityProfile
        The condensed profile, on the target grid.
    operator : numpy.ndarray
        Matrix of shape ``(target_groups, source_groups)`` with entries 1 where a
        source group lies in a target group and 0 elsewhere. The condensed
        sensitivities of a profile are ``operator @ sensitivity``.
    report : CondensationReport
        What was done.
    """

    profile: SensitivityProfile
    operator: np.ndarray
    report: CondensationReport


def _validated_target_grid(
    target_grid: Iterable[float],
    target_energy_unit: EnergyUnit,
) -> np.ndarray:
    grid = np.asarray(target_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise CondensationError(
            "target_grid must be a one-dimensional array with at least two boundaries"
        )
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise CondensationError(
            "target_grid must contain finite non-negative boundaries"
        )
    if np.any(np.diff(grid) <= 0.0):
        raise CondensationError("target_grid must be strictly increasing")
    if target_energy_unit not in ("eV", "MeV"):
        raise CondensationError("target_energy_unit must be 'eV' or 'MeV'")
    return grid


def _grid_to_mev(grid: np.ndarray, unit: EnergyUnit) -> np.ndarray:
    return grid if unit == "MeV" else grid * 1.0e-6


def _exact_boundary_indices(
    source_grid_mev: np.ndarray,
    target_grid_mev: np.ndarray,
    *,
    energy_rtol: float,
) -> Tuple[int, ...]:
    if energy_rtol < 0.0 or not np.isfinite(energy_rtol):
        raise CondensationError("energy_rtol must be finite and non-negative")

    if not np.isclose(
        target_grid_mev[0], source_grid_mev[0], rtol=energy_rtol, atol=0.0
    ) or not np.isclose(
        target_grid_mev[-1], source_grid_mev[-1], rtol=energy_rtol, atol=0.0
    ):
        raise CondensationError(
            "Exact condensation requires the target grid to cover the same energy "
            "range as the source grid"
        )

    indices = []
    missing = []
    for boundary in target_grid_mev:
        matches = np.flatnonzero(
            np.isclose(source_grid_mev, boundary, rtol=energy_rtol, atol=0.0)
        )
        if len(matches) == 0:
            missing.append(float(boundary))
            continue
        nearest = matches[np.argmin(np.abs(source_grid_mev[matches] - boundary))]
        indices.append(int(nearest))

    if missing:
        preview = ", ".join(f"{value:.8g}" for value in missing[:5])
        if len(missing) > 5:
            preview += f", ... ({len(missing)} total)"
        raise CondensationError(
            "Exact condensation requires every target boundary to exist on the "
            f"source grid after unit conversion. Missing boundaries in MeV: {preview}. "
            "A non-nested grid requires a separate, explicitly requested projection."
        )

    if len(indices) != len(target_grid_mev) or np.any(np.diff(indices) <= 0):
        raise CondensationError(
            "Target boundaries do not define a strictly coarser subset of the source grid"
        )
    return tuple(indices)


def _exact_condensation_operator(
    source_groups: int,
    boundary_indices: Tuple[int, ...],
) -> np.ndarray:
    operator = np.zeros((len(boundary_indices) - 1, source_groups), dtype=float)
    for target_group, (start, stop) in enumerate(
        zip(boundary_indices[:-1], boundary_indices[1:])
    ):
        operator[target_group, start:stop] = 1.0
    return operator


def condense_sensitivity_profile(
    profile: SensitivityProfile,
    target_grid: Iterable[float],
    *,
    target_energy_unit: Optional[EnergyUnit] = None,
    energy_rtol: float = 1.0e-5,
) -> CondensationResult:
    """Condense a sensitivity profile onto an exactly nested coarser grid.

    Group-wise sensitivity coefficients are integral contributions, so source
    groups inside each target group are summed. Absolute one-sigma statistical
    uncertainties are combined in quadrature, which assumes independent source-
    group estimates. Unknown uncertainties remain unknown.

    The operation is deliberately strict: the target grid must span the same
    range and every target boundary must be present in the source grid after
    explicit unit conversion. Non-nested grid projection is not performed.

    There is no weight: no flux or lethargy enters, and nothing is averaged. A target
    group's sensitivity is the plain sum of its source groups' and its uncertainty is
    the square root of the sum of their squared uncertainties. A profile without an
    uncertainty keeps None. The response, its uncertainty and the label are copied; the
    metadata of the result gains an entry in ``"condensation_history"``.

    Parameters
    ----------
    profile
        The source profile.
    target_grid
        Boundaries of the coarser grid, strictly ascending, in *target_energy_unit*.
    target_energy_unit : {"eV", "MeV"}, optional
        Unit of ``target_grid`` and of the returned profile. None is the unit of
        *profile*.
    energy_rtol
        Relative tolerance (no absolute one) with which a target boundary is taken as
        equal to a source boundary, after both are put in MeV. The default covers the
        rounding of the boundaries that an SDF file keeps (six decimals).

    Returns
    -------
    CondensationResult
        The condensed profile, the 0/1 operator and the report.

    Raises
    ------
    TypeError
        If *profile* is not a
        :class:`~kika.sensitivities.profile.SensitivityProfile`.
    CondensationError
        If *target_grid* is not a one-dimensional strictly increasing grid of at
        least two finite, non-negative values; *target_energy_unit* is not ``"eV"`` or
        ``"MeV"``; *energy_rtol* is negative or not finite; the end points of the
        target grid are not those of the source; or a target boundary is not a source
        boundary.
    """
    if not isinstance(profile, SensitivityProfile):
        raise TypeError("profile must be a SensitivityProfile")

    output_unit = (
        profile.energy_unit if target_energy_unit is None else target_energy_unit
    )
    target = _validated_target_grid(target_grid, output_unit)
    source_mev = profile.grid_in("MeV")
    target_mev = _grid_to_mev(target, output_unit)
    boundary_indices = _exact_boundary_indices(
        source_mev,
        target_mev,
        energy_rtol=energy_rtol,
    )
    operator = _exact_condensation_operator(profile.n_groups, boundary_indices)

    reactions = []
    max_integral_drift = 0.0
    has_uncertainty = False
    for reaction in profile.reactions:
        sensitivity = operator @ reaction.sensitivity
        uncertainty = None
        if reaction.uncertainty is not None:
            has_uncertainty = True
            uncertainty = np.sqrt((operator * operator) @ (reaction.uncertainty**2))

        source_integral = float(np.sum(reaction.sensitivity))
        target_integral = float(np.sum(sensitivity))
        max_integral_drift = max(
            max_integral_drift,
            abs(target_integral - source_integral),
        )
        reactions.append(
            SensitivityReaction(
                zaid=reaction.zaid,
                mt=reaction.mt,
                sensitivity=sensitivity,
                uncertainty=uncertainty,
                label=reaction.label,
            )
        )

    uncertainty_method = (
        "independent source-group quadrature" if has_uncertainty else "not available"
    )
    assumptions = (
        (
            "Source-group sensitivity estimates were treated as statistically "
            "independent when propagating absolute one-sigma uncertainties."
        ),
    ) if has_uncertainty else ()

    metadata = dict(profile.metadata)
    history = list(metadata.get("condensation_history", ()))
    history.append(
        {
            "method": "exact_nested_sum",
            "source_groups": profile.n_groups,
            "target_groups": len(target) - 1,
            "source_energy_unit": profile.energy_unit,
            "target_energy_unit": output_unit,
            "energy_rtol": energy_rtol,
        }
    )
    metadata["condensation_history"] = history

    condensed = SensitivityProfile(
        energy_grid=target,
        energy_unit=output_unit,
        reactions=tuple(reactions),
        response=profile.response,
        response_uncertainty=profile.response_uncertainty,
        label=profile.label,
        metadata=metadata,
    )
    report = CondensationReport(
        source_groups=profile.n_groups,
        target_groups=condensed.n_groups,
        source_energy_grid=tuple(float(value) for value in profile.energy_grid),
        target_energy_grid=tuple(float(value) for value in target),
        source_energy_unit=profile.energy_unit,
        target_energy_unit=output_unit,
        boundary_indices=boundary_indices,
        energy_rtol=energy_rtol,
        uncertainty_method=uncertainty_method,
        assumptions=assumptions,
        max_integral_drift=max_integral_drift,
    )
    return CondensationResult(profile=condensed, operator=operator, report=report)


__all__ = [
    "CondensationError",
    "CondensationReport",
    "CondensationResult",
    "condense_sensitivity_profile",
]

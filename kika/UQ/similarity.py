"""Covariance-weighted similarity of integral-response sensitivity profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.UQ.alignment import (
    AlignmentReport,
    LegendreCovarianceInput,
    ParameterIndex,
    ParameterKey,
    PreparedCovariance,
    CovarianceInput,
    align_sensitivity_covariance,
)
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


class ZeroSimilarityVarianceError(ValueError):
    """Raised when c-k is undefined because either covariance norm is non-positive."""


@dataclass(frozen=True)
class ReactionSimilarity:
    """Non-additive c-k diagnostic from one diagonal reaction block.

    The terms use only the covariance block of one parameter with itself, so they do
    not add up to the total of :class:`SimilarityResult`. Found in
    ``SimilarityResult.reaction_similarity``.

    Attributes
    ----------
    key
        The parameter (nuclide, MT and, for Legendre, order).
    value
        The c-k of the block alone; None when either variance is not positive.
    variance_a, variance_b
        ``S.T @ C @ S`` of profile a and of profile b restricted to this parameter
        (relative variance, dimensionless).
    cross_covariance
        The same product with one vector from each profile.
    """

    key: ParameterKey
    value: Optional[float]
    variance_a: float
    variance_b: float
    cross_covariance: float


@dataclass
class SimilarityResult:
    """Covariance-weighted similarity result and alignment provenance.

    Returned by :func:`similarity_ck`.

    Attributes
    ----------
    value
        The c-k coefficient ``S_a.T @ C @ S_b / sqrt(var_a * var_b)``, in [-1, 1]
        (also the property ``ck``).
    variance_a, variance_b
        ``S.T @ C @ S`` of each profile with the relative covariance: the relative
        variance (dimensionless, fraction squared) of its response due to the
        covariance.
    cross_covariance
        ``S_a.T @ C @ S_b``, the relative covariance between the two responses.
    reaction_similarity
        One :class:`ReactionSimilarity` per aligned parameter, in the order of
        *parameter_keys*.
    index
        Meaning of each element of the flat vector the sums run over: parameters in
        sorted order, groups in ascending energy.
    parameter_keys
        The aligned parameters.
    alignment_report
        The :class:`AlignmentReport` of the alignment, with the filters of
        :func:`similarity_ck` added to its exclusions and assumptions.
    """

    value: float
    variance_a: float
    variance_b: float
    cross_covariance: float
    reaction_similarity: Tuple[ReactionSimilarity, ...]
    index: Tuple[ParameterIndex, ...]
    parameter_keys: Tuple[ParameterKey, ...]
    alignment_report: AlignmentReport

    @property
    def ck(self) -> float:
        """Covariance-weighted similarity coefficient."""
        return self.value


def _as_profile(value: Union[SensitivityProfile, SDFData]) -> SensitivityProfile:
    if isinstance(value, SensitivityProfile):
        return value
    if isinstance(value, SDFData):
        return value.to_sensitivity_profile()
    raise TypeError("similarity profiles must be SensitivityProfile or SDFData")


def _nubar_mode_for(mode, zaid: int) -> str:
    return mode.get(zaid, "total") if isinstance(mode, dict) else mode


def _prepare_profile(
    value: Union[SensitivityProfile, SDFData],
    include: Optional[Iterable[Tuple[int, int]]],
    exclude: Optional[Iterable[Tuple[int, int]]],
    include_mt1: bool,
    nubar_mode,
) -> SensitivityProfile:
    profile = _as_profile(value)
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or ())
    selected = [
        reaction for reaction in profile.reactions
        if (include_mt1 or reaction.mt != 1)
        and (
            reaction.mt != 0
            or (include_set is not None and (reaction.zaid, 0) in include_set)
        )
        and (include_set is None or (reaction.zaid, reaction.mt) in include_set)
        and (reaction.zaid, reaction.mt) not in exclude_set
    ]

    by_zaid = {}
    for reaction in selected:
        if reaction.mt in (452, 455, 456):
            by_zaid.setdefault(reaction.zaid, set()).add(reaction.mt)
    kept = []
    for reaction in selected:
        mts = by_zaid.get(reaction.zaid, set())
        if 452 not in mts or not ({455, 456} & mts):
            kept.append(reaction)
            continue
        mode = _nubar_mode_for(nubar_mode, reaction.zaid)
        complete = {455, 456}.issubset(mts)
        if mode == "components" and complete:
            if reaction.mt != 452:
                kept.append(reaction)
        elif reaction.mt not in (455, 456):
            kept.append(reaction)

    if not kept:
        raise ValueError("No sensitivity reactions remain after applying similarity filters")
    return SensitivityProfile(
        energy_grid=profile.energy_grid,
        energy_unit=profile.energy_unit,
        reactions=tuple(kept),
        response=profile.response,
        response_uncertainty=profile.response_uncertainty,
        label=profile.label,
        metadata=profile.metadata,
    )


def _checked_ck(cross: float, variance_a: float, variance_b: float) -> float:
    if not np.isfinite(variance_a) or not np.isfinite(variance_b):
        raise ZeroSimilarityVarianceError("c_k is undefined because a covariance norm is non-finite")
    if variance_a <= 0.0 or variance_b <= 0.0:
        raise ZeroSimilarityVarianceError(
            "c_k is undefined because S.T @ C @ S is zero or negative for at least one profile"
        )
    value = float(cross / np.sqrt(variance_a * variance_b))
    if not np.isfinite(value):
        raise ZeroSimilarityVarianceError("c_k is undefined because its normalized value is non-finite")
    if abs(value) > 1.0 + 1.0e-10:
        raise ValueError(
            f"c_k={value:.12g} lies outside [-1, 1]; verify covariance symmetry and definiteness"
        )
    return float(np.clip(value, -1.0, 1.0))


def similarity_ck(
    profile_a: Union[SensitivityProfile, SDFData],
    profile_b: Union[SensitivityProfile, SDFData],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    alias_policy: str = "exact",
    missing: str = "error",
    include: Optional[Sequence[Tuple[int, int]]] = None,
    exclude: Optional[Sequence[Tuple[int, int]]] = None,
    include_mt1: bool = False,
    nubar_mode: Union[str, dict] = "total",
    energy_tolerance: float = 1.0e-6,
    prepared_covariance: Optional[PreparedCovariance] = None,
) -> SimilarityResult:
    """Calculate covariance-weighted c-k between two sensitivity profiles.

    Both profiles are filtered, aligned with the covariance by
    :func:`align_sensitivity_covariance` (a parameter in only one profile has zero
    sensitivity in the other), and compared as ``S_a.T @ C @ S_b`` over the
    relative covariance, normalised by the two variances. The sensitivities are
    relative coefficients and the covariance is relative, so the result is
    dimensionless.

    Parameters
    ----------
    profile_a, profile_b
        Sensitivity profiles on the same energy grid as the covariance.
    covariance, legendre_covariance
        Cross-section and Legendre covariances, as in
        :func:`align_sensitivity_covariance`.
    alias_policy : {"exact", "tsurfer"}
        As in :func:`align_sensitivity_covariance`.
    missing : {"error", "drop"}
        As in :func:`align_sensitivity_covariance`.
    include
        Pairs ``(zaid, mt)`` to keep; every other reaction is removed. None keeps all.
        A reaction with MT 0 (pseudo-total) is kept only when ``(zaid, 0)`` is listed.
        Legendre reactions are matched by their sensitivity MT (4000 plus the order).
    exclude
        Pairs ``(zaid, mt)`` to remove, applied after *include*.
    include_mt1
        Keep MT 1 reactions. False removes them, which avoids counting the total
        together with its partial reactions.
    nubar_mode : {"total", "components"} or dict
        For a nuclide with MT 452 and at least one of MT 455 and 456, which view of
        nu-bar to keep. ``"components"`` keeps MT 455 and 456 and drops MT 452, only when
        both components are present; otherwise (and for ``"total"``) MT 452 is kept and
        the components are dropped. A dict maps ZAID to a mode; a nuclide not in it uses
        ``"total"``.
    energy_tolerance
        Relative tolerance of the grid comparisons (``energy_rtol`` of
        :func:`align_sensitivity_covariance`).
    prepared_covariance
        Normalised covariance from :func:`prepare_covariance`, used in place of
        *covariance* and *legendre_covariance*.

    Returns
    -------
    SimilarityResult
        The coefficient, the two variances, their covariance, the per-parameter
        diagnostics and the alignment report.

    Raises
    ------
    ValueError
        If no reaction of a profile remains after the filters, or the computed c-k lies
        outside [-1, 1] by more than 1e-10 (the covariance is then not symmetric or not
        positive semi-definite). Values within the tolerance are clipped. Errors of
        :func:`align_sensitivity_covariance` also propagate.
    ZeroSimilarityVarianceError
        If either variance is not finite or is zero or negative, so that c-k is
        undefined.
    AlignmentError
        As in :func:`align_sensitivity_covariance`, including
        :class:`MissingCovarianceError` when ``missing="error"``.
    TypeError
        If a profile is neither a :class:`~kika.sensitivities.profile.SensitivityProfile`
        nor an :class:`~kika.sensitivities.sdf.SDFData`.
    """
    raw_a = _as_profile(profile_a)
    raw_b = _as_profile(profile_b)
    a = _prepare_profile(raw_a, include, exclude, include_mt1, nubar_mode)
    b = _prepare_profile(raw_b, include, exclude, include_mt1, nubar_mode)
    aligned = align_sensitivity_covariance(
        [a, b],
        covariance=covariance,
        legendre_covariance=legendre_covariance,
        alias_policy=alias_policy,
        missing=missing,
        energy_rtol=energy_tolerance,
        prepared_covariance=prepared_covariance,
    )
    for profile_index, (raw, selected) in enumerate(((raw_a, a), (raw_b, b))):
        raw_keys = {ParameterKey.from_sensitivity(r.zaid, r.mt) for r in raw.reactions}
        selected_keys = {
            ParameterKey.from_sensitivity(r.zaid, r.mt) for r in selected.reactions
        }
        exclusions = sorted(raw_keys - selected_keys)
        if exclusions:
            existing = aligned.report.policy_exclusions.get(profile_index, [])
            aligned.report.policy_exclusions[profile_index] = sorted(
                set(existing).union(exclusions)
            )
    if not include_mt1 and any(
        reaction.mt == 1 for profile in (raw_a, raw_b) for reaction in profile.reactions
    ):
        aligned.report.assumptions.append("MT=1 excluded from c_k")
    if any(
        reaction.mt == 0 for profile in (raw_a, raw_b) for reaction in profile.reactions
    ):
        aligned.report.assumptions.append(
            "MT=0 pseudo-total excluded from c_k unless explicitly included"
        )
    aligned.report.assumptions.append(
        f"nu-bar redundancy resolved with nubar_mode={nubar_mode!r}"
    )

    s_a, s_b = aligned.sensitivity_vectors
    matrix = aligned.covariance
    variance_a = float(s_a @ matrix @ s_a)
    variance_b = float(s_b @ matrix @ s_b)
    cross = float(s_a @ matrix @ s_b)
    value = _checked_ck(cross, variance_a, variance_b)

    diagnostics: List[ReactionSimilarity] = []
    for index, key in enumerate(aligned.parameter_keys):
        start, length = aligned.reaction_spans[index]
        stop = start + length
        block = matrix[start:stop, start:stop]
        sub_a = s_a[start:stop]
        sub_b = s_b[start:stop]
        var_a = float(sub_a @ block @ sub_a)
        var_b = float(sub_b @ block @ sub_b)
        sub_cross = float(sub_a @ block @ sub_b)
        local = None
        if var_a > 0.0 and var_b > 0.0:
            local = _checked_ck(sub_cross, var_a, var_b)
        diagnostics.append(ReactionSimilarity(key, local, var_a, var_b, sub_cross))

    return SimilarityResult(
        value=value,
        variance_a=variance_a,
        variance_b=variance_b,
        cross_covariance=cross,
        reaction_similarity=tuple(diagnostics),
        index=aligned.index,
        parameter_keys=aligned.parameter_keys,
        alignment_report=aligned.report,
    )

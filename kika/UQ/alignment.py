"""Strict, format-neutral alignment of sensitivities and covariance blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


AliasPolicy = Literal["exact", "tsurfer"]
MissingPolicy = Literal["error", "drop"]


@dataclass(frozen=True, order=True)
class ParameterKey:
    """A nuclear-data parameter independent of its source file format.

    Built by hand or taken from ``AlignmentResult.parameter_keys``. Instances are
    frozen and ordered by ``(kind, zaid, mt, order)``, which is the order of the aligned
    vectors (``"legendre"`` sorts before ``"xs"``).

    Attributes
    ----------
    kind
        ``"xs"`` for a multigroup cross section, ``"legendre"`` for a Legendre moment of
        the angular distribution.
    zaid
        Nuclide identifier, ``1000 * Z + A``.
    mt
        ENDF reaction number.
    order
        Legendre order. None for ``"xs"``.
    """

    kind: Literal["xs", "legendre"]
    zaid: int
    mt: int
    order: Optional[int] = None

    @classmethod
    def from_sensitivity(cls, zaid: int, mt: int) -> "ParameterKey":
        """Key of a sensitivity reaction.

        Parameters
        ----------
        zaid
            Nuclide identifier, ``1000 * Z + A``.
        mt
            Reaction number of the sensitivity. 4000 or more means Legendre order
            ``mt - 4000`` of MT 2.

        Returns
        -------
        ParameterKey
            A ``"legendre"`` key for ``mt >= 4000``, an ``"xs"`` key otherwise.
        """
        if mt >= 4000:
            return cls("legendre", int(zaid), 2, int(mt) - 4000)
        return cls("xs", int(zaid), int(mt), None)

    @property
    def label(self) -> str:
        """Text such as ``"ZAID=26056 MT=2"``, with `` P<order>`` appended for a Legendre key."""
        suffix = f" P{self.order}" if self.kind == "legendre" else ""
        return f"ZAID={self.zaid} MT={self.mt}{suffix}"


@dataclass(frozen=True)
class ParameterIndex:
    """Meaning of one element of an aligned flat vector.

    Attributes
    ----------
    key
        The parameter the element belongs to.
    group
        Zero-based group number within the parameter, counted in ascending energy.
    energy_low, energy_high
        Lower and upper boundary of the group, in the unit of *energy_unit*.
    energy_unit
        Always ``"MeV"``: the aligned grid is converted to MeV.
    """

    key: ParameterKey
    group: int
    energy_low: float
    energy_high: float
    energy_unit: str = "MeV"


@dataclass(frozen=True)
class AliasMapping:
    """A sensitivity parameter matched to a covariance parameter under another key.

    Attributes
    ----------
    profile
        Zero-based position of the sensitivity profile in the input.
    source
        The key found in the sensitivity profile.
    target
        The key of the covariance block it was matched to.
    reason
        Free text naming the rule that applied (bound or natural ZAID, MT 101/102
        equivalence, MCNP MT -2 to capture, or a combination).
    """

    profile: int
    source: ParameterKey
    target: ParameterKey
    reason: str


@dataclass
class AlignmentReport:
    """Complete record of non-exact decisions made while aligning data.

    It is found on ``AlignmentResult.report``, on ``MissingCovarianceError.report`` and,
    for the propagation functions, on the ``alignment_report`` of their result.

    An exact alignment has empty ``aliases``, ``zeros_inserted``,
    ``zero_covariance_blocks``, ``missing_covariance``, ``dropped`` and
    ``policy_exclusions``, and coverages of 1.0. ``assumptions`` can still hold
    entries (an energy grid in eV, a covariance without ``is_relative`` flags).

    Attributes
    ----------
    aliases
        Parameters matched under another key (``alias_policy="tsurfer"`` only).
    zeros_inserted
        Per profile (zero-based position), the aligned keys the profile does not have
        and for which a zero sensitivity was used.
    zero_covariance_blocks
        Pairs ``(row, col)`` of aligned keys with no covariance block, taken as zero.
        Each pair is listed once, with ``row`` at or before ``col`` in the aligned
        order.
    missing_covariance
        Sensitivity keys with no diagonal covariance block.
    dropped
        The keys of *missing_covariance* that were removed (``missing="drop"``).
    policy_exclusions
        Per profile, keys removed by a filter of the calling function (or by an alias
        collision resolved in favour of the closer key).
    assumptions
        One sentence per assumption made: unit conversions of an energy grid,
        covariance blocks treated as relative or converted from absolute,
        symmetrisation of single-precision blocks, and the filters of the calling
        function.
    parameter_coverage
        One value per profile: the fraction of the profile's reactions, as passed to
        the alignment, that were kept (1.0 for an empty profile). Only covariance and
        alias losses lower it; removals made beforehand by a calling function (filters
        of :func:`~kika.UQ.sandwich.sandwich_uncertainty_propagation` or
        :func:`~kika.UQ.similarity.similarity_ck`) are not counted.
    sensitivity_coverage
        One value per profile: the fraction of the sum of ``|S|`` over all its groups
        and reactions, as passed to the alignment, that was kept (1.0 when that sum
        is zero). The same removals are not counted.
    """

    aliases: List[AliasMapping] = field(default_factory=list)
    zeros_inserted: Dict[int, List[ParameterKey]] = field(default_factory=dict)
    zero_covariance_blocks: List[Tuple[ParameterKey, ParameterKey]] = field(default_factory=list)
    missing_covariance: List[ParameterKey] = field(default_factory=list)
    dropped: List[ParameterKey] = field(default_factory=list)
    policy_exclusions: Dict[int, List[ParameterKey]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    parameter_coverage: List[float] = field(default_factory=list)
    sensitivity_coverage: List[float] = field(default_factory=list)


class AlignmentError(ValueError):
    """Base error for a sensitivity/covariance alignment failure."""

    def __init__(self, message: str, report: Optional[AlignmentReport] = None):
        super().__init__(message)
        self.report = report


class MissingCovarianceError(AlignmentError):
    """Raised when sensitivity parameters lack covariance data."""


@dataclass
class AlignmentResult:
    """Aligned sensitivity vectors, absolute sigmas, relative covariance and index.

    Returned by :func:`align_sensitivity_covariance`. All arrays share one flat
    parameter axis of ``n_parameters = len(parameter_keys) * n_groups`` elements: the
    parameters in ``parameter_keys`` order, each with its ``n_groups`` groups in
    ascending energy.

    Attributes
    ----------
    sensitivity_vectors : numpy.ndarray
        Shape ``(n_profiles, n_parameters)``. Relative sensitivity coefficients
        (dimensionless), zero where a profile has no reaction for the parameter.
    sensitivity_uncertainties : numpy.ndarray
        Same shape. Absolute one-sigma uncertainty of each sensitivity coefficient (in
        the unit of the sensitivity); NaN where the profile's reaction has no
        uncertainty, zero where a zero sensitivity was inserted.
    covariance : numpy.ndarray
        Shape ``(n_parameters, n_parameters)``, symmetric. Relative covariance of the
        cross-section and Legendre-moment parameters, zero for pairs with no block
        (see *covariance* in :func:`prepare_covariance` for how blocks are read as
        relative).
    index : tuple of ParameterIndex
        One entry per element of the flat axis.
    parameter_keys : tuple of ParameterKey
        The aligned parameters, in sorted order: the union of the profiles' keys after
        aliasing and dropping.
    reaction_spans : dict
        Maps the position of a parameter in *parameter_keys* (not its MT) to
        ``(start, n_groups)`` on the flat axis.
    profiles : tuple of SensitivityProfile
        The input profiles as format-neutral objects, before any alias or drop.
    report : AlignmentReport
        What was aliased, inserted, dropped or assumed.
    """

    sensitivity_vectors: np.ndarray
    sensitivity_uncertainties: np.ndarray
    covariance: np.ndarray
    index: Tuple[ParameterIndex, ...]
    parameter_keys: Tuple[ParameterKey, ...]
    reaction_spans: Dict[int, Tuple[int, int]]
    profiles: Tuple[SensitivityProfile, ...]
    report: AlignmentReport


@dataclass
class PreparedCovariance:
    """Normalized covariance blocks reusable for profiles on one exact grid.

    Returned by :func:`prepare_covariance` and accepted as *prepared_covariance* by
    :func:`align_sensitivity_covariance` and
    :func:`~kika.UQ.similarity.similarity_ck`, to avoid normalising the same
    covariance again for each call.

    Attributes
    ----------
    energy_grid_mev : numpy.ndarray
        Group boundaries in MeV, strictly increasing: the grid of the profile it was
        prepared from.
    blocks : dict
        Maps ``(row, col)`` pairs of :class:`ParameterKey` to relative covariance
        matrices of shape ``(n_groups, n_groups)``. Both ``(row, col)`` and
        ``(col, row)`` are stored.
    assumptions : tuple of str
        The assumptions made while normalising, copied into the report of each
        alignment that uses the object.
    """

    energy_grid_mev: np.ndarray
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]
    assumptions: Tuple[str, ...] = ()


CovarianceInput = Union[CrossSectionCovariance, Sequence[CrossSectionCovariance]]
LegendreCovarianceInput = Union[
    MultigroupLegendreCovariance, Sequence[MultigroupLegendreCovariance]
]


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _as_profile(value: Union[SensitivityProfile, SDFData]) -> SensitivityProfile:
    if isinstance(value, SensitivityProfile):
        return value
    if isinstance(value, SDFData):
        return value.to_sensitivity_profile()
    raise TypeError("profiles must contain SensitivityProfile or SDFData objects")


def _grid_to_mev(grid: Iterable[float], unit: str) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    if unit == "MeV":
        return values
    if unit == "eV":
        return values * 1.0e-6
    raise AlignmentError(f"unsupported energy unit {unit!r}; expected 'eV' or 'MeV'")


def _require_grid_match(
    expected: np.ndarray,
    actual: Iterable[float],
    unit: str,
    *,
    rtol: float,
    source: str,
) -> None:
    candidate = _grid_to_mev(actual, unit)
    if candidate.ndim != 1 or np.any(np.diff(candidate) <= 0.0):
        raise AlignmentError(f"{source} energy grid must be one-dimensional and strictly increasing")
    if candidate.shape != expected.shape or not np.allclose(
        candidate, expected, rtol=rtol, atol=0.0
    ):
        raise AlignmentError(
            f"{source} energy grid does not exactly match the sensitivity grid after unit "
            "conversion. Partial alignment is not performed; explicitly condense the "
            "sensitivity profile or covariance grid first."
        )


def _store_block(
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray],
    row: ParameterKey,
    col: ParameterKey,
    matrix: np.ndarray,
    source: str,
) -> bool:
    """Store one block and return whether float32 symmetry noise was removed."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AlignmentError(f"non-square covariance block {row.label} x {col.label} in {source}")
    if not np.all(np.isfinite(matrix)):
        raise AlignmentError(f"non-finite covariance block {row.label} x {col.label} in {source}")

    symmetrized = False
    if row == col and not np.array_equal(matrix, matrix.T):
        # Binary COVERX stores single-precision values.  Some real SCALE
        # libraries contain independently rounded upper/lower triangles; the
        # observed differences are below 6e-8 in absolute covariance and are
        # sometimes larger than a few ULPs for coefficients close to zero.
        # Accept only that representational noise; materially asymmetric
        # covariance still fails loudly.
        if not np.allclose(matrix, matrix.T, rtol=5.0e-7, atol=2.0e-8):
            raise AlignmentError(
                f"incompatible asymmetric diagonal covariance block {row.label} in {source}"
            )
        matrix = 0.5 * (matrix + matrix.T)
        symmetrized = True

    for key, value in (((row, col), matrix), ((col, row), matrix.T)):
        previous = blocks.get(key)
        if previous is not None and not np.allclose(previous, value, rtol=1e-12, atol=1e-15):
            raise AlignmentError(
                f"incompatible duplicate covariance block {key[0].label} x {key[1].label}"
            )
        blocks[key] = value.copy()
    return symmetrized


def _relative_xs_blocks(
    covariance: Sequence[CrossSectionCovariance],
    expected_grid: np.ndarray,
    report: AlignmentReport,
    energy_rtol: float,
) -> Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]:
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray] = {}
    for source_index, cov in enumerate(covariance):
        source = f"cross-section covariance[{source_index}]"
        if cov.energy_unit != "MeV":
            report.assumptions.append(
                f"{source} energy grid converted from {cov.energy_unit} to MeV"
            )
        if cov.energy_grid is None:
            raise AlignmentError(f"{source} has no shared multigroup energy grid")
        _require_grid_match(
            expected_grid, cov.energy_grid, cov.energy_unit, rtol=energy_rtol, source=source
        )
        if cov.is_relative and len(cov.is_relative) != len(cov.matrices):
            raise AlignmentError(f"{source} has an incomplete is_relative block index")
        unmarked = not cov.is_relative
        if unmarked:
            note = f"{source} has no is_relative flags; blocks were treated as relative"
            if note not in report.assumptions:
                report.assumptions.append(note)

        for block_index, (zaid_r, mt_r, zaid_c, mt_c, raw) in enumerate(
            zip(
                cov.isotope_rows,
                cov.reaction_rows,
                cov.isotope_cols,
                cov.reaction_cols,
                cov.matrices,
            )
        ):
            row = ParameterKey("xs", int(zaid_r), int(mt_r))
            col = ParameterKey("xs", int(zaid_c), int(mt_c))
            matrix = np.asarray(raw, dtype=float)
            if matrix.shape != (len(expected_grid) - 1, len(expected_grid) - 1):
                raise AlignmentError(
                    f"{source} block {block_index} shape {matrix.shape} does not match the energy grid"
                )
            is_relative = True if unmarked else bool(cov.is_relative[block_index])
            if not is_relative:
                xs_row = cov.cross_sections.get((row.zaid, row.mt))
                xs_col = cov.cross_sections.get((col.zaid, col.mt))
                if xs_row is None or xs_col is None:
                    raise AlignmentError(
                        f"absolute covariance block {row.label} x {col.label} requires "
                        "row and column cross sections for conversion to relative covariance"
                    )
                xs_row = np.asarray(xs_row, dtype=float)
                xs_col = np.asarray(xs_col, dtype=float)
                if xs_row.shape != (len(expected_grid) - 1,) or xs_col.shape != xs_row.shape:
                    raise AlignmentError("cross-section vectors do not match the covariance grid")
                scale = np.outer(xs_row, xs_col)
                if np.any(scale == 0.0):
                    raise AlignmentError(
                        f"absolute covariance block {row.label} x {col.label} cannot be "
                        "converted because a required cross section is zero"
                    )
                matrix = matrix / scale
                report.assumptions.append(
                    f"{source} block {row.label} x {col.label} converted from absolute "
                    "to relative covariance using its row and column cross sections"
                )
            if _store_block(blocks, row, col, matrix, source):
                note = (
                    f"{source} diagonal covariance blocks were symmetrized within "
                    "float32 rounding tolerance"
                )
                if note not in report.assumptions:
                    report.assumptions.append(note)
    return blocks


def _relative_legendre_blocks(
    covariance: Sequence[MultigroupLegendreCovariance],
    expected_grid: np.ndarray,
    report: AlignmentReport,
    energy_rtol: float,
) -> Dict[Tuple[ParameterKey, ParameterKey], np.ndarray]:
    blocks: Dict[Tuple[ParameterKey, ParameterKey], np.ndarray] = {}
    for source_index, cov in enumerate(covariance):
        source = f"Legendre covariance[{source_index}]"
        if cov.energy_unit != "MeV":
            report.assumptions.append(
                f"{source} energy grid converted from {cov.energy_unit} to MeV"
            )
        _require_grid_match(
            expected_grid, cov.energy_grid, cov.energy_unit, rtol=energy_rtol, source=source
        )
        lengths = {
            len(cov.isotope_rows), len(cov.reaction_rows), len(cov.l_rows),
            len(cov.isotope_cols), len(cov.reaction_cols), len(cov.l_cols),
            len(cov.relative_matrices),
        }
        if len(lengths) != 1:
            raise AlignmentError(f"{source} has inconsistent block metadata lengths")
        for values in zip(
            cov.isotope_rows, cov.reaction_rows, cov.l_rows,
            cov.isotope_cols, cov.reaction_cols, cov.l_cols,
            cov.relative_matrices,
        ):
            zaid_r, mt_r, order_r, zaid_c, mt_c, order_c, matrix = values
            row = ParameterKey("legendre", int(zaid_r), int(mt_r), int(order_r))
            col = ParameterKey("legendre", int(zaid_c), int(mt_c), int(order_c))
            if np.asarray(matrix).shape != (len(expected_grid) - 1, len(expected_grid) - 1):
                raise AlignmentError(f"{source} block shape does not match the energy grid")
            if _store_block(blocks, row, col, matrix, source):
                note = (
                    f"{source} diagonal covariance blocks were symmetrized within "
                    "float32 rounding tolerance"
                )
                if note not in report.assumptions:
                    report.assumptions.append(note)
    return blocks


_ZAID_ALIASES = {4309: 4009, 4509: 4009, 1801: 1001, 1901: 1001, 6312: 6000}


def _alias_candidates(key: ParameterKey) -> List[Tuple[ParameterKey, str]]:
    candidates: List[Tuple[ParameterKey, str]] = []
    mapped_zaid = _ZAID_ALIASES.get(key.zaid)
    if mapped_zaid is not None:
        candidates.append((ParameterKey(key.kind, mapped_zaid, key.mt, key.order), "bound/natural ZAID"))
    if key.kind == "xs" and key.mt in (101, 102):
        other = 102 if key.mt == 101 else 101
        candidates.append((ParameterKey("xs", key.zaid, other), "MT 101/102 equivalence"))
        if mapped_zaid is not None:
            candidates.append((ParameterKey("xs", mapped_zaid, other), "ZAID and MT 101/102 equivalence"))
    if key.kind == "xs" and key.mt == -2:
        candidates.append((ParameterKey("xs", key.zaid, 101), "MCNP MT -2 to capture"))
        if mapped_zaid is not None:
            candidates.append((ParameterKey("xs", mapped_zaid, 101), "MCNP MT -2 and ZAID alias"))
    return candidates


def _resolve_key(
    source: ParameterKey,
    available: set,
    alias_policy: AliasPolicy,
) -> Tuple[Optional[ParameterKey], Optional[str]]:
    if source in available:
        return source, None
    if source.mt < 0 and source.mt != -2:
        raise AlignmentError(f"unsupported negative reaction identifier: {source.label}")
    if alias_policy == "tsurfer":
        matches = [(candidate, reason) for candidate, reason in _alias_candidates(source) if candidate in available]
        unique = {candidate for candidate, _ in matches}
        if len(unique) > 1:
            raise AlignmentError(
                f"ambiguous TSURFER alias for {source.label}: "
                + ", ".join(candidate.label for candidate in sorted(unique))
            )
        if matches:
            return matches[0]
    return None, None


def prepare_covariance(
    profile: Union[SensitivityProfile, SDFData],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    energy_rtol: float = 1.0e-8,
) -> PreparedCovariance:
    """Normalise covariance blocks once for repeated calculations on one grid.

    Both orientations of each block are stored, and a slightly asymmetric diagonal
    block (single-precision rounding) is symmetrised.

    Parameters
    ----------
    profile
        Gives the energy grid the covariance must match (converted to MeV).
    covariance
        One cross-section covariance or a sequence of them. The result is read as a
        relative covariance in the unit the covariance object stores; nothing checks
        or converts percent versus fraction. Each block follows the ``is_relative``
        flags of the object: a block flagged absolute is divided by the outer product
        of the cross sections of its row and column parameters (per group); a block
        flagged relative is used as it is; a covariance with no flags is taken as
        relative, which is recorded in the report's ``assumptions``. A flag list that
        is non-empty and has a different length from the list of blocks raises
        ``AlignmentError``, as does a cross section that is missing, of the wrong
        shape or zero.
    legendre_covariance
        One multigroup Legendre covariance or a sequence of them. Its blocks are read
        from ``relative_matrices`` and are relative by construction.
    energy_rtol
        Relative tolerance of the element-wise comparison between the covariance
        grids and the grid of *profile*. The grids must have the same number of
        boundaries; no condensation or partial alignment is done.

    Returns
    -------
    PreparedCovariance
        The relative blocks, keyed by pairs of :class:`ParameterKey`.

    Raises
    ------
    AlignmentError
        If a covariance has no shared energy grid, its grid (after conversion to MeV)
        is not strictly increasing or does not match that of *profile*, an energy
        unit is not eV or MeV, the metadata of a Legendre covariance has inconsistent
        lengths, a block is non-square, not finite, of the wrong shape, materially
        asymmetric on the diagonal or inconsistent with a duplicate of itself, the
        ``is_relative`` flags are of the wrong length, an absolute block lacks a
        cross section, has one of the wrong shape or a zero one, or no block is
        given at all.
    TypeError
        If *profile* is neither a
        :class:`~kika.sensitivities.profile.SensitivityProfile` nor an
        :class:`~kika.sensitivities.sdf.SDFData`.
    """
    neutral = _as_profile(profile)
    expected_grid = neutral.grid_in("MeV")
    report = AlignmentReport()
    blocks = _relative_xs_blocks(_as_list(covariance), expected_grid, report, energy_rtol)
    legendre_blocks = _relative_legendre_blocks(
        _as_list(legendre_covariance), expected_grid, report, energy_rtol
    )
    for key, matrix in legendre_blocks.items():
        if key in blocks:
            raise AlignmentError(f"duplicate covariance block across covariance types: {key}")
        blocks[key] = matrix
    if not blocks:
        raise AlignmentError("at least one covariance block is required")
    return PreparedCovariance(expected_grid.copy(), blocks, tuple(report.assumptions))

def align_sensitivity_covariance(
    profiles: Sequence[Union[SensitivityProfile, SDFData]],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    alias_policy: AliasPolicy = "exact",
    missing: MissingPolicy = "error",
    energy_rtol: float = 1.0e-8,
    prepared_covariance: Optional[PreparedCovariance] = None,
) -> AlignmentResult:
    """Align one or more sensitivity profiles with relative covariance blocks.

    Profile keys are combined by union. A missing key in one profile is an
    explicit zero sensitivity; a key missing from covariance is an error unless
    ``missing="drop"`` is requested.

    The result puts every profile and the covariance on one flat axis: the selected
    parameters (a :class:`ParameterKey` each: nuclide, MT, and Legendre order where
    it applies) in sorted order, each with its energy groups in ascending energy. The
    sandwich rule ``S @ C @ S`` can then be applied directly to the rows of
    ``sensitivity_vectors``. The covariance of a pair of parameters with no block is
    zero and is listed in the report. A covariance block for a parameter that no
    profile has is ignored, without a report entry. All profiles must share the grid
    of the first one, and the covariance must match it; nothing is condensed.

    Parameters
    ----------
    profiles
        One or more sensitivity profiles on the same energy grid, whose sensitivities
        are relative coefficients and whose uncertainties are absolute. A profile
        reaction with MT 4000 or more is Legendre order ``MT - 4000`` of MT 2.
    covariance
        One cross-section covariance or a sequence of them. How its blocks are read
        as relative (the ``is_relative`` flags) is described under *covariance* in
        :func:`prepare_covariance`.
    legendre_covariance
        One multigroup Legendre covariance or a sequence of them (relative; see
        :func:`prepare_covariance`).
    alias_policy : {"exact", "tsurfer"}
        ``"exact"`` matches a sensitivity key only to the identical covariance key.
        ``"tsurfer"`` also accepts, for a key with no block: the natural or bound ZAID
        for ZAIDs 4309 and 4509 (to 4009), 1801 and 1901 (to 1001) and 6312 (to 6000);
        MT 101 for MT 102 and the reverse; MT 101 for MCNP MT -2. Each use is
        recorded in ``report.aliases``. When two sensitivity keys of one profile map
        to the same key, the one needing fewer substitutions is kept (the other goes
        to ``report.policy_exclusions``).
    missing : {"error", "drop"}
        What to do with a sensitivity key that has no diagonal covariance block (after
        aliasing). ``"error"`` raises; ``"drop"`` removes it from every profile and
        lists it in ``report.missing_covariance`` and ``report.dropped``.
    energy_rtol
        Relative tolerance of the element-wise grid comparisons (profiles with each
        other and with the covariance). Energies in eV are converted to MeV first.
    prepared_covariance
        Normalised covariance from :func:`prepare_covariance`, used in place of
        *covariance* and *legendre_covariance*. Its grid must match that of the
        profiles.

    Returns
    -------
    AlignmentResult
        The aligned vectors, covariance, index and report.

    Raises
    ------
    ValueError
        If *alias_policy* or *missing* is not one of the strings above, *profiles* is
        empty, or *prepared_covariance* is given together with *covariance* or
        *legendre_covariance*.
    MissingCovarianceError
        If a sensitivity key has no covariance and ``missing="error"``. Its ``report``
        carries the keys.
    AlignmentError
        If a grid does not match, an energy unit is not eV or MeV, a covariance block
        is invalid (see :func:`prepare_covariance`), a sensitivity MT is negative and
        not -2 (and not itself in the covariance), an alias is ambiguous or collides
        with an equally close key, no parameter remains after dropping, or the
        assembled covariance is not symmetric.
    TypeError
        If an element of *profiles* is neither a
        :class:`~kika.sensitivities.profile.SensitivityProfile` nor an
        :class:`~kika.sensitivities.sdf.SDFData`.
    """
    if alias_policy not in ("exact", "tsurfer"):
        raise ValueError("alias_policy must be 'exact' or 'tsurfer'")
    if missing not in ("error", "drop"):
        raise ValueError("missing must be 'error' or 'drop'")
    if not profiles:
        raise ValueError("at least one sensitivity profile is required")

    neutral = tuple(_as_profile(profile) for profile in profiles)
    expected_grid = neutral[0].grid_in("MeV")
    profile_energy_conversions = [
        f"sensitivity profile[{index}] energy grid converted from {profile.energy_unit} to MeV"
        for index, profile in enumerate(neutral)
        if profile.energy_unit != "MeV"
    ]
    for index, profile in enumerate(neutral[1:], 1):
        _require_grid_match(
            expected_grid,
            profile.energy_grid,
            profile.energy_unit,
            rtol=energy_rtol,
            source=f"sensitivity profile[{index}]",
        )

    report = AlignmentReport()
    report.assumptions.extend(profile_energy_conversions)
    if prepared_covariance is not None:
        if covariance is not None or legendre_covariance is not None:
            raise ValueError("pass covariance inputs or prepared_covariance, not both")
        _require_grid_match(
            expected_grid,
            prepared_covariance.energy_grid_mev,
            "MeV",
            rtol=energy_rtol,
            source="prepared covariance",
        )
        blocks = prepared_covariance.blocks
        report.assumptions.extend(prepared_covariance.assumptions)
    else:
        prepared_covariance = prepare_covariance(
            neutral[0], covariance, legendre_covariance, energy_rtol=energy_rtol
        )
        blocks = prepared_covariance.blocks
        report.assumptions.extend(prepared_covariance.assumptions)

    available = {row for row, col in blocks if row == col}
    resolved_maps: List[Dict[ParameterKey, ParameterKey]] = []
    missing_keys = set()
    source_maps: List[Dict[ParameterKey, object]] = []
    for profile_index, profile in enumerate(neutral):
        reaction_map = {
            ParameterKey.from_sensitivity(reaction.zaid, reaction.mt): reaction
            for reaction in profile.reactions
        }
        source_maps.append(reaction_map)
        resolved: Dict[ParameterKey, ParameterKey] = {}
        target_owners: Dict[ParameterKey, Tuple[ParameterKey, Optional[str]]] = {}
        for source in sorted(reaction_map):
            target, reason = _resolve_key(source, available, alias_policy)
            if target is None:
                missing_keys.add(source)
                continue
            previous = target_owners.get(target)
            if previous is not None:
                previous_source = previous[0]
                # Prefer the source requiring the fewest semantic changes:
                # exact key, then same reaction with a ZAID alias, then an MT
                # alias.  This resolves the common DICE MT 101/102 duplicate
                # without summing it twice, including for bound ZAIDs.  Equal-
                # specificity collisions remain ambiguous and fail loudly.
                score = (source != target, source.mt != target.mt, source.zaid != target.zaid)
                previous_score = (
                    previous_source != target,
                    previous_source.mt != target.mt,
                    previous_source.zaid != target.zaid,
                )
                if score != previous_score:
                    if score < previous_score:
                        resolved.pop(previous_source)
                        excluded = previous_source
                        target_owners[target] = (source, reason)
                        resolved[source] = target
                    else:
                        excluded = source
                    report.policy_exclusions.setdefault(profile_index, []).append(excluded)
                    note = (
                        f"sensitivity profile[{profile_index}] closest source parameter "
                        "was preferred over a competing TSURFER alias"
                    )
                    if note not in report.assumptions:
                        report.assumptions.append(note)
                    continue
                raise AlignmentError(
                    f"alias collision in sensitivity profile[{profile_index}]: multiple "
                    f"source parameters map to {target.label}"
                )
            target_owners[target] = (source, reason)
            resolved[source] = target
        for source, target in resolved.items():
            reason = target_owners[target][1]
            if reason is not None:
                report.aliases.append(AliasMapping(profile_index, source, target, reason))
        resolved_maps.append(resolved)

    report.missing_covariance = sorted(missing_keys)
    if missing_keys and missing == "error":
        labels = ", ".join(key.label for key in sorted(missing_keys))
        raise MissingCovarianceError(
            "No covariance data are available for: " + labels + ". "
            "To continue explicitly with the covered subset, repeat the call with "
            'missing="drop". The returned AlignmentReport will record the loss.',
            report,
        )
    report.dropped = sorted(missing_keys)

    selected_keys = sorted({target for resolved in resolved_maps for target in resolved.values()})
    if not selected_keys:
        raise AlignmentError("no sensitivity parameters remain after covariance alignment", report)

    n_profiles = len(neutral)
    n_groups = len(expected_grid) - 1
    n_parameters = len(selected_keys) * n_groups
    vectors = np.zeros((n_profiles, n_parameters))
    uncertainties = np.zeros_like(vectors)
    spans: Dict[int, Tuple[int, int]] = {}

    for key_index, key in enumerate(selected_keys):
        spans[key_index] = (key_index * n_groups, n_groups)
    for profile_index, (profile, reaction_map, resolved) in enumerate(
        zip(neutral, source_maps, resolved_maps)
    ):
        reverse = {target: source for source, target in resolved.items()}
        zeros = []
        for key_index, key in enumerate(selected_keys):
            start = key_index * n_groups
            source = reverse.get(key)
            if source is None:
                zeros.append(key)
                continue
            reaction = reaction_map[source]
            vectors[profile_index, start:start + n_groups] = reaction.sensitivity
            if reaction.uncertainty is None:
                uncertainties[profile_index, start:start + n_groups] = np.nan
            else:
                uncertainties[profile_index, start:start + n_groups] = reaction.uncertainty
        if zeros:
            report.zeros_inserted[profile_index] = zeros

        original_count = len(reaction_map)
        kept_sources = len(resolved)
        report.parameter_coverage.append(
            kept_sources / original_count if original_count else 1.0
        )
        total_abs = sum(float(np.sum(np.abs(r.sensitivity))) for r in reaction_map.values())
        kept_abs = sum(
            float(np.sum(np.abs(reaction_map[source].sensitivity))) for source in resolved
        )
        report.sensitivity_coverage.append(kept_abs / total_abs if total_abs else 1.0)

    full_covariance = np.zeros((n_parameters, n_parameters))
    for row_index, row in enumerate(selected_keys):
        for col_index, col in enumerate(selected_keys):
            block = blocks.get((row, col))
            if block is None:
                if row_index <= col_index:
                    report.zero_covariance_blocks.append((row, col))
                continue
            r0 = row_index * n_groups
            c0 = col_index * n_groups
            full_covariance[r0:r0 + n_groups, c0:c0 + n_groups] = block
    if not np.allclose(full_covariance, full_covariance.T, rtol=1e-12, atol=1e-15):
        raise AlignmentError("assembled covariance matrix is not symmetric", report)
    full_covariance = 0.5 * (full_covariance + full_covariance.T)

    flat_index = tuple(
        ParameterIndex(
            key=key,
            group=group,
            energy_low=float(expected_grid[group]),
            energy_high=float(expected_grid[group + 1]),
        )
        for key in selected_keys
        for group in range(n_groups)
    )
    return AlignmentResult(
        sensitivity_vectors=vectors,
        sensitivity_uncertainties=uncertainties,
        covariance=full_covariance,
        index=flat_index,
        parameter_keys=tuple(selected_keys),
        reaction_spans=spans,
        profiles=neutral,
        report=report,
    )

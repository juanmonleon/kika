"""Thin benchmark-database adapters for Kika's format-neutral UQ APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from kika.UQ.alignment import AlignmentError, prepare_covariance
from kika.UQ.sandwich import UncertaintyResult, sandwich_uncertainty_propagation
from kika.UQ.similarity import SimilarityResult, similarity_ck as uq_similarity_ck
from kika.benchmarks.database import BenchmarksDatabase
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


@dataclass(frozen=True)
class BenchmarkSimilarity:
    """One deterministic row in a benchmark c-k ranking.

    Returned in a list by :func:`rank_benchmarks_by_ck`.

    Attributes
    ----------
    benchmark_id : str
        The benchmark id.
    profile_id : int
        The profile of that benchmark that was compared.
    ck : float
        The covariance-weighted similarity coefficient between the application and
        the profile, dimensionless, in [-1, 1].
    n_parameters : int
        Number of aligned parameters (nuclide and MT, and Legendre order for
        Legendre parameters) the coefficient sums over.
    application_parameter_coverage, benchmark_parameter_coverage : float
        Fraction (0 to 1) of the reactions of the application, and of the
        benchmark profile, that the alignment with the covariance kept. Only
        losses from missing covariance or aliasing lower it; removals by
        ``include``, ``exclude``, ``include_mt1`` and ``nubar_mode`` are not
        counted.
    application_sensitivity_coverage, benchmark_sensitivity_coverage : float
        Fraction (0 to 1) of the sum of ``|S|`` over all groups and reactions of
        the application, and of the profile, that the alignment kept; the same
        removals are not counted.
    code, library, group_structure : str or None
        Transport code, nuclear data library and group structure of the profile,
        as stored.
    result : kika.UQ.similarity.SimilarityResult or None
        The full result of the comparison, with the variances, the per-parameter
        diagnostics and the alignment report.
    """

    benchmark_id: str
    profile_id: int
    ck: float
    n_parameters: int
    application_parameter_coverage: float
    benchmark_parameter_coverage: float
    application_sensitivity_coverage: float
    benchmark_sensitivity_coverage: float
    code: Optional[str] = None
    library: Optional[str] = None
    group_structure: Optional[str] = None
    result: Optional[SimilarityResult] = None


def benchmark_uncertainty(
    profile_id: int,
    covariance=None,
    legendre_covariance=None,
    *,
    db_path: Optional[str] = None,
    **kwargs,
) -> UncertaintyResult:
    """Run sandwich propagation for one SQLite benchmark profile.

    The profile is read with
    :meth:`~kika.benchmarks.BenchmarksDatabase.get_sensitivity_profile` and
    passed to :func:`~kika.UQ.sandwich.sandwich_uncertainty_propagation`. The
    sensitivities are relative coefficients and the covariance is read as
    relative, so the result is a relative uncertainty (a fraction of the
    response, not a percentage or pcm).

    Parameters
    ----------
    profile_id
        The ``profile_id`` of a profile of the database. Its k-eff must be
        stored and non-zero, which a profile built from the 30-group ABBN tables
        does not satisfy.
    covariance
        One cross-section covariance or a sequence of them, on the profile's
        energy grid. Given to ``cov_mat`` of the propagation.
    legendre_covariance
        One multigroup Legendre covariance or a sequence of them. Given to
        ``legendre_cov_mat``. At least one of the two covariances is required.
    db_path : str, optional
        Database path; None resolves it as :func:`~kika.benchmarks.get_db_path`
        does.
    **kwargs
        Passed on to the propagation: ``reaction_filter``, ``nubar_mode``,
        ``energy_tolerance``, ``verbose``, ``bootstrap``, ``n_bootstrap``,
        ``ci_level``, ``bootstrap_seed``, ``alias_policy`` and ``missing``. The
        bootstrap is on by default and needs a stored uncertainty for every
        sensitivity, so a profile without stored errors needs
        ``bootstrap=False``. A profile of the compact TSUNAMI-1D dialect stores
        zeros as its uncertainties, so the bootstrap runs on it without
        complaint and draws no variation.

    Returns
    -------
    ~kika.UQ.sandwich.UncertaintyResult
        ``total_uncertainty`` and ``relative_uncertainty`` hold the relative
        one-sigma uncertainty of the profile's k-eff, ``total_variance`` the
        relative variance, and ``contributions`` one entry per parameter.

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If ``profile_id`` is not in the database.
    ValueError
        If neither covariance is given, the profile's k-eff is missing or zero,
        or the propagation refuses its other inputs.
    ~kika.UQ.alignment.AlignmentError
        If the profile and the covariance cannot be aligned.
    """
    with BenchmarksDatabase(db_path) as database:
        profile = database.get_sensitivity_profile(profile_id)
    return sandwich_uncertainty_propagation(
        profile,
        cov_mat=covariance,
        legendre_cov_mat=legendre_covariance,
        **kwargs,
    )


def similarity_ck(
    application: Union[SensitivityProfile, SDFData],
    benchmark_profile_id: int,
    covariance=None,
    legendre_covariance=None,
    *,
    db_path: Optional[str] = None,
    **kwargs,
) -> SimilarityResult:
    """Calculate c-k between an application and one SQLite benchmark profile.

    The profile is read with
    :meth:`~kika.benchmarks.BenchmarksDatabase.get_sensitivity_profile` and
    compared with ``application`` by
    :func:`~kika.UQ.similarity.similarity_ck`. The sensitivities are relative
    coefficients and the covariance is read as relative, so c-k is
    dimensionless.

    Parameters
    ----------
    application
        The application's sensitivity profile or SDF data (converted to a
        profile), on the same energy grid as the covariance.
    benchmark_profile_id
        The ``profile_id`` of the benchmark profile in the database.
    covariance
        One cross-section covariance or a sequence of them.
    legendre_covariance
        One multigroup Legendre covariance or a sequence of them.
    db_path : str, optional
        Database path; None resolves it as :func:`~kika.benchmarks.get_db_path`
        does.
    **kwargs
        Passed on to the comparison: ``alias_policy``, ``missing``, ``include``,
        ``exclude``, ``include_mt1``, ``nubar_mode``, ``energy_tolerance`` and
        ``prepared_covariance``. ``include`` and ``exclude`` filter both
        profiles.

    Returns
    -------
    ~kika.UQ.similarity.SimilarityResult
        ``value`` (also ``ck``) is the coefficient, in [-1, 1]; ``variance_a``
        and ``variance_b`` are the relative variances of the application and of
        the benchmark response, ``cross_covariance`` their relative covariance.

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If ``benchmark_profile_id`` is not in the database.
    ValueError
        If no reaction of a profile remains after the filters, or c-k is outside
        [-1, 1] (it is clipped within 1e-10 of the bounds).
    ~kika.UQ.similarity.ZeroSimilarityVarianceError
        If either variance is not finite or is not positive.
    ~kika.UQ.alignment.AlignmentError
        If the profiles and the covariance cannot be aligned.
    TypeError
        If ``application`` is neither a sensitivity profile nor SDF data.
    """
    with BenchmarksDatabase(db_path) as database:
        benchmark = database.get_sensitivity_profile(benchmark_profile_id)
    return uq_similarity_ck(
        application,
        benchmark,
        covariance,
        legendre_covariance,
        **kwargs,
    )


def rank_benchmarks_by_ck(
    application: Union[SensitivityProfile, SDFData],
    covariance=None,
    legendre_covariance=None,
    *,
    benchmark_ids: Optional[Sequence[str]] = None,
    preferred_only: bool = True,
    limit: Optional[int] = 100,
    rank_by_absolute: bool = False,
    db_path: Optional[str] = None,
    alias_policy: str = "exact",
    missing: str = "error",
    include=None,
    exclude=None,
    include_mt1: bool = False,
    nubar_mode="total",
    energy_tolerance: float = 1.0e-6,
) -> List[BenchmarkSimilarity]:
    """Rank SQLite benchmark profiles by covariance-weighted c-k.

    Covariance normalization is performed once. In strict mode any candidate
    that cannot be aligned aborts the ranking with its benchmark/profile label;
    no failed candidate is silently omitted.

    Each candidate profile is compared with ``application`` by
    :func:`~kika.UQ.similarity.similarity_ck`. The sensitivities are relative
    coefficients and the covariance is read as relative, so every ``ck`` is
    dimensionless, in [-1, 1].

    The rows are sorted by ``ck`` from largest to smallest (so the most similar
    benchmark comes first and a negative ``ck`` comes last); with
    ``rank_by_absolute`` by ``abs(ck)`` from largest to smallest, so a ``ck`` of
    -0.9 ranks above 0.5. Ties are broken by ``benchmark_id`` and then
    ``profile_id``, both ascending. ``limit`` cuts the list after sorting. No
    candidate is excluded by the sort: the candidates are the profiles of
    ``benchmark_ids`` (every benchmark when None), only the preferred profile of
    each unless ``preferred_only`` is false.

    Parameters
    ----------
    application
        The application's sensitivity profile, on the same energy grid as the
        covariance and the benchmark profiles.
    covariance
        One cross-section covariance or a sequence of them. It is normalised
        once for all candidates.
    legendre_covariance
        One multigroup Legendre covariance or a sequence of them.
    benchmark_ids
        Restrict the candidates to these benchmarks; None uses all of them and
        an empty sequence gives an empty ranking.
    preferred_only
        True compares only the preferred profile of each benchmark; False
        compares every profile, so a benchmark can appear more than once.
    limit
        Maximum number of rows kept after sorting; None keeps all, and 0 gives
        an empty list.
    rank_by_absolute
        Sort by ``abs(ck)`` instead of ``ck``.
    db_path : str, optional
        Database path; None resolves it as :func:`~kika.benchmarks.get_db_path`
        does.
    alias_policy, missing, include, exclude, include_mt1, nubar_mode
        As in :func:`~kika.UQ.similarity.similarity_ck`. ``include`` and
        ``exclude`` filter the application and every benchmark profile alike.
    energy_tolerance
        Relative tolerance of the energy-grid comparisons, for the covariance
        and for each comparison.

    Returns
    -------
    list of BenchmarkSimilarity
        One row per compared profile, in the order above, with ``ck``, the
        parameter counts and coverages (fractions, 0 to 1) and the full
        ``result``.

    Raises
    ------
    ValueError
        If ``limit`` is negative, or a comparison raises it (no reaction left
        after the filters, or c-k outside [-1, 1]). A
        :class:`~kika.UQ.similarity.ZeroSimilarityVarianceError` (c-k undefined)
        from any candidate also aborts the ranking.
    TypeError
        If ``application`` is neither a sensitivity profile nor SDF data.
    ~kika.UQ.alignment.AlignmentError
        If the covariance does not match the application's energy grid, or a
        candidate cannot be aligned (the message names its benchmark and
        profile).
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    neutral_application = (
        application.to_sensitivity_profile() if isinstance(application, SDFData) else application
    )
    if not isinstance(neutral_application, SensitivityProfile):
        raise TypeError("application must be SDFData or SensitivityProfile")
    prepared = prepare_covariance(
        neutral_application,
        covariance,
        legendre_covariance,
        energy_rtol=energy_tolerance,
    )

    rows: List[BenchmarkSimilarity] = []
    with BenchmarksDatabase(db_path) as database:
        candidates = database.list_similarity_profiles(
            list(benchmark_ids) if benchmark_ids is not None else None,
            preferred_only=preferred_only,
        )
        for candidate in candidates:
            profile = database.get_sensitivity_profile(candidate["profile_id"])
            try:
                result = uq_similarity_ck(
                    neutral_application,
                    profile,
                    alias_policy=alias_policy,
                    missing=missing,
                    include=include,
                    exclude=exclude,
                    include_mt1=include_mt1,
                    nubar_mode=nubar_mode,
                    energy_tolerance=energy_tolerance,
                    prepared_covariance=prepared,
                )
            except AlignmentError as error:
                raise AlignmentError(
                    f"Cannot rank benchmark {candidate['benchmark_id']} "
                    f"(profile_id={candidate['profile_id']}): {error}",
                    getattr(error, "report", None),
                ) from error
            report = result.alignment_report
            rows.append(
                BenchmarkSimilarity(
                    benchmark_id=candidate["benchmark_id"],
                    profile_id=candidate["profile_id"],
                    ck=result.value,
                    n_parameters=len(result.parameter_keys),
                    application_parameter_coverage=report.parameter_coverage[0],
                    benchmark_parameter_coverage=report.parameter_coverage[1],
                    application_sensitivity_coverage=report.sensitivity_coverage[0],
                    benchmark_sensitivity_coverage=report.sensitivity_coverage[1],
                    code=candidate["code"],
                    library=candidate["library"],
                    group_structure=candidate["group_structure"],
                    result=result,
                )
            )

    rows.sort(
        key=lambda row: (
            -(abs(row.ck) if rank_by_absolute else row.ck),
            row.benchmark_id,
            row.profile_id,
        )
    )
    return rows if limit is None else rows[:limit]

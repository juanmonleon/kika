"""
Public screening API for the benchmarks database.

Find and rank ICSBEP/DICE benchmarks by their sensitivity to a given isotope,
reaction, and energy region — the core "which benchmarks matter for nuclide X"
question.
"""

from typing import List, Optional, Union

from kika._constants import MT_TO_REACTION
from kika._utils import symbol_to_zaid
from kika.benchmarks._constants import REACTION_ALIASES
from kika.benchmarks.database import BenchmarksDatabase

# Reverse of MT_TO_REACTION for name -> MT resolution (lowercased canonical labels).
_REACTION_NAME_TO_MT = {v.lower(): k for k, v in MT_TO_REACTION.items()}


def _normalize_isotope(isotope: Union[str, int]) -> int:
    """Normalize an isotope to a ZAID. Accepts 26056, 'Fe56', or 'Fe-56'."""
    if isinstance(isotope, int):
        return isotope
    if isinstance(isotope, str):
        return symbol_to_zaid(isotope.replace("-", "").strip())
    raise TypeError(f"isotope must be int or str, got {type(isotope).__name__}")


def _normalize_reaction(reaction: Optional[Union[str, int]]) -> Optional[int]:
    """Normalize a reaction to an MT integer. Accepts 2, 'elastic', '(n,el)', None."""
    if reaction is None:
        return None
    if isinstance(reaction, int):
        return reaction
    if isinstance(reaction, str):
        key = reaction.strip().lower()
        if key.isdigit():
            return int(key)
        if key in REACTION_ALIASES:
            return REACTION_ALIASES[key]
        if key in _REACTION_NAME_TO_MT:
            return _REACTION_NAME_TO_MT[key]
        raise ValueError(f"Unrecognized reaction: {reaction!r}")
    raise TypeError(f"reaction must be int, str or None, got {type(reaction).__name__}")


def find_sensitive_benchmarks(
    isotope: Union[str, int],
    reaction: Optional[Union[str, int]] = None,
    energy_region: str = "total",
    sensitivity_threshold: float = 0.0,
    fraction_threshold: Optional[float] = None,
    preferred_only: bool = True,
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[dict]:
    """
    Find benchmarks sensitive to an isotope/reaction in an energy region.

    Parameters
    ----------
    isotope : str or int
        Isotope as ZAID (26056), symbol ('Fe56') or hyphenated symbol ('Fe-56').
    reaction : str or int, optional
        Reaction as MT (2), alias ('elastic', 'capture') or canonical label
        ('(n,el)'). None matches any reaction of the isotope.
    energy_region : str
        'thermal', 'epithermal', 'fast' or 'total' (default).
    sensitivity_threshold : float
        Minimum absolute integral sensitivity in the region.
    fraction_threshold : float, optional
        Minimum ``|region sensitivity| / sum(|sensitivity|)`` for the reaction.
    preferred_only : bool
        Only consider the preferred profile per benchmark (default True).
    limit : int
        Maximum number of results.
    db_path : str, optional
        Explicit database path (otherwise resolved from config/env/default).

    Returns
    -------
    list of dict
        One row per matching benchmark reaction, ranked by absolute sensitivity
        (descending). Each row includes benchmark id/category/spectrum, profile
        code/library, keff, the region sensitivity and its fraction of the total.
    """
    zaid = _normalize_isotope(isotope)
    mt = _normalize_reaction(reaction)
    with BenchmarksDatabase(db_path) as db:
        return db.screen(
            zaid=zaid,
            mt=mt,
            region=energy_region,
            sensitivity_threshold=sensitivity_threshold,
            fraction_threshold=fraction_threshold,
            preferred_only=preferred_only,
            limit=limit,
        )


def rank_benchmarks(
    isotope: Union[str, int],
    reaction: Optional[Union[str, int]] = None,
    energy_region: str = "total",
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[dict]:
    """Rank benchmarks by absolute sensitivity to an isotope/reaction/region.

    Convenience wrapper over :func:`find_sensitive_benchmarks` with no thresholds.
    The ranking sorts by the absolute value of the integral sensitivity in the
    region, largest first, over the preferred profile of each benchmark and its
    system-total reactions. A reaction with an integral of exactly zero in the
    region is not returned.

    Parameters
    ----------
    isotope : str or int
        ZAID (26056), symbol (``"Fe56"``) or hyphenated symbol (``"Fe-56"``).
    reaction : str or int, optional
        An MT, an alias (``"elastic"``, ``"capture"``) or a canonical label
        (``"(n,el)"``). None ranks every reaction of the isotope, MT 0 (the
        pseudo-total) excepted, so a benchmark can appear once per reaction.
    energy_region : {"thermal", "epithermal", "fast", "total"}
        The region whose integral is ranked: up to 0.625 eV, up to 100 keV,
        above 100 keV, and all groups.
    limit
        Maximum number of rows.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    list of dict
        The rows of :meth:`~kika.benchmarks.BenchmarksDatabase.screen`: the integral
        sensitivity ``s_region`` is the dimensionless relative sensitivity
        coefficient summed over the groups of the region, ``fraction`` is
        ``abs(s_region)`` over the sum of the absolute group values of the same
        reaction.

    Raises
    ------
    ValueError
        If ``energy_region`` is not one of the four, or ``reaction`` is a string
        that matches no alias, label or integer.
    TypeError
        If ``isotope`` is not an int or a str, or ``reaction`` is not an int, a
        str or None.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    return find_sensitive_benchmarks(
        isotope,
        reaction=reaction,
        energy_region=energy_region,
        sensitivity_threshold=0.0,
        fraction_threshold=None,
        limit=limit,
        db_path=db_path,
    )


def get_benchmark(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return a benchmark's metadata and its list of profile variants.

    Parameters
    ----------
    benchmark_id
        The id as stored, e.g. ``"HEU-COMP-THERM-021-079"``.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_benchmark`: the
        benchmark's stored row and ``profiles``, the list of its variants.

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If the id is not in the database.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_benchmark(benchmark_id)


def get_benchmark_dashboard(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return the whole per-benchmark record (metadata, keff, vectors, spectrum,
    balance, input codes) in a single bundle for the app dashboard.

    Parameters
    ----------
    benchmark_id
        The id as stored.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_benchmark_dashboard`:
        the dict of :func:`get_benchmark` plus ``experimental_keff``,
        ``preferred_profile_id``, ``profile_vector`` (of the preferred profile),
        ``spectrum``, ``balance`` and ``input_codes``.

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If the id is not in the database.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_benchmark_dashboard(benchmark_id)


def get_profile_vector(
    profile_id: int,
    zaid: Optional[int] = None,
    mt: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Return a profile's full per-group sensitivity vectors (SDFSensitivityData shape).

    Parameters
    ----------
    profile_id
        The ``profile_id`` of a profile of the database.
    zaid, mt
        Keep only the reactions with this ZAID (1000 Z + A) or this ENDF MT.
        None applies no filter.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_profile_vector`:
        ``pert_energies`` (group boundaries in MeV, increasing) and
        ``reactions``, the system-total reactions with their ``sensitivity``
        (dimensionless relative coefficient per group, in increasing energy)
        and ``error`` (absolute one-sigma uncertainty of each coefficient;
        ``None`` when not stored and for ABBN tables, all zeros for the compact
        TSUNAMI-1D dialect, which has none).

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If ``profile_id`` is not in the database.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_profile_vector(profile_id, zaid=zaid, mt=mt)


def get_sensitivity_profile(profile_id: int, db_path: Optional[str] = None):
    """Return a profile as a validated format-neutral sensitivity object.

    Parameters
    ----------
    profile_id
        The ``profile_id`` of a profile of the database.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    ~kika.sensitivities.SensitivityProfile
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_sensitivity_profile`:
        energy boundaries in MeV, dimensionless relative sensitivities with
        absolute one-sigma uncertainties (``None`` when not stored and for ABBN
        tables, all zeros for the compact TSUNAMI-1D dialect), and the profile's
        k-eff and its absolute uncertainty as ``response`` and
        ``response_uncertainty`` (0.0 when the file's k-eff line has no ``+/-``;
        both ``None`` for an ABBN table).

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If ``profile_id`` is not in the database.
    ValueError
        If the stored vectors fail the profile's validation.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_sensitivity_profile(profile_id)


def get_experimental_keff(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return a benchmark's experimental keff ``{'keff', 'keff_unc'}`` (empty if unknown).

    Parameters
    ----------
    benchmark_id
        The id as stored.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_experimental_keff`:
        the experimental k-eff and its uncertainty as the benchmark workbook
        gives them (kika does not convert them), or an empty dict when none was
        matched to the benchmark.

    Raises
    ------
    ~kika.benchmarks.BenchmarkNotFoundError
        If the id is not in the database.
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_experimental_keff(benchmark_id)


def get_spectrum(benchmark_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return a benchmark's 299-group flux spectrum ``{'energies', 'flux'}`` or None.

    Parameters
    ----------
    benchmark_id
        The id as stored.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict or None
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_spectrum`: lists of
        the upper energy edge of each group in MeV and of the group flux, in the
        order of the DICE file; ``None`` when no spectrum is stored.

    Raises
    ------
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_spectrum(benchmark_id)


def get_balance(benchmark_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return a benchmark's reaction-rate balance ``{'summary', 'text'}`` or None.

    Parameters
    ----------
    benchmark_id
        The id as stored.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    dict or None
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_balance`: ``summary``
        (those of ``keff``, ``leakage`` and ``n_zones`` found in the file) and
        ``text`` (the whole balance file); ``None`` when none is stored.

    Raises
    ------
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_balance(benchmark_id)


def get_input_deck(
    benchmark_id: str, code: Optional[str] = None, db_path: Optional[str] = None
) -> list:
    """Return a benchmark's raw input deck(s) as ``[{'code', 'text'}]`` (verbatim).

    Parameters
    ----------
    benchmark_id
        The id as stored.
    code
        ``"KENO"`` or ``"MCNP"`` to return that deck only (exact match); None
        returns every stored deck.
    db_path : str, optional
        Database path; None resolves it as :func:`get_db_path` does.

    Returns
    -------
    list of dict
        As :meth:`~kika.benchmarks.BenchmarksDatabase.get_input_deck`: one dict
        per deck with its ``code`` and ``text``; empty when nothing matches.

    Raises
    ------
    ~kika.benchmarks.DatabaseNotConfiguredError
        If the database file does not exist.
    """
    with BenchmarksDatabase(db_path) as db:
        return db.get_input_deck(benchmark_id, code=code)

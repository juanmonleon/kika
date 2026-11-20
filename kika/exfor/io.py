"""
EXFOR data loading functions.

This module provides the main I/O functions for loading EXFOR data from
JSON files or the X4Pro database, and organizing datasets by energy.

Supports multiple data sources:
- JSON files: Local JSON files in standardized format
- Database: X4Pro SQLite database (full 2025 version with JSON schema)
- Auto: Database with JSON fallback for missing entries
"""

import os
import glob
import math
from typing import Dict, List, Optional, Tuple, Union

from kika.exfor._constants import ENERGY_MATCH_ABS_TOL
from kika.exfor.config import get_db_path
from kika.exfor.angular_distribution import ExforAngularDistribution

def read_exfor(filepath: str) -> "ExforAngularDistribution":
    """
    Read an EXFOR angular distribution from a JSON file.

    The file is read whatever its ``quantity`` field says, as an angular
    distribution. Both the standardised v1.0 schema and the older one (reaction
    given as a string, energies under ``E``, points under ``result``) are read. The
    frame, units and values are kept as the file gives them, with no conversion.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    ExforAngularDistribution
        The distribution, in the ``angle_frame`` and ``units`` of the file. On it,
        ``sigma_sys_scalar_relative`` and ``sigma_sys_indep_relative`` are set to
        0.0 and ``uncertainty_manifest_flag`` to ``"default"``: reading a file
        applies no systematic uncertainty.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is not valid JSON. A valid file with missing or unexpected
        fields may raise other errors.

    Example:
        >>> from kika.exfor import read_exfor
        >>> exfor = read_exfor('/path/to/27673002.json')
        >>> print(exfor.label)
        Gkatis et al. (2025)
        >>> print(exfor.energies())
        [1.0098, 1.0202, ...]
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXFOR file not found: {filepath}")

    ad = ExforAngularDistribution.from_json(filepath)
    # Manifest application is pipeline-specific and lives in
    # scripts/uncertainty_manifest.py. The kika library returns the raw
    # ExforAngularDistribution; the pipeline (e.g. build_exfor_cache_from_objects)
    # calls apply_manifest_to_exfor() before building the cache.
    # JSON path: no raw EXFOR columns. The pipeline resolver treats an empty
    # list the same as None and synthesizes a DATA-ERR component from the
    # per-point uncertainty_stat -- see apply_manifest_to_exfor in
    # scripts/uncertainty_manifest.py.
    ad._raw_uncertainty_components = []
    ad.sigma_sys_scalar_relative = 0.0
    ad.sigma_sys_indep_relative = 0.0
    ad.uncertainty_manifest_flag = "default"
    return ad


def read_all_exfor(
    target: Union[str, int, List[str], List[int]] = None,
    mt: int = None,
    source: str = "database",
    directory: str = None,
    pattern: str = "*.json",
    group_by_energy: bool = True,
    db_path: str = None,
    projectile: str = "n",
    energy_range: Tuple[float, float] = None,
    supplementary_json_files: List[str] = None,
    return_load_status: bool = False,
    exclude_experiments: List[str] = None,
    # Deprecated parameter for backwards compatibility
    target_zaid: Union[int, List[int]] = None,
) -> Union[Dict[float, List], Dict[str, "ExforAngularDistribution"], Tuple[Dict, Dict]]:
    """
    Read EXFOR angular distributions from the X4Pro database, JSON files, or both.

    From the database only angular distributions are read (quantity ``DA``, ENDF
    file MF 4); JSON files are read as angular distributions whatever their
    quantity. A dataset is identified by its entry and subentry numbers joined (``"27673002"``).
    When two sources give the same identifier, the first one read is kept, in the
    order database, JSON directory, supplementary files.

    Parameters
    ----------
    target : str, int, list of str or list of int, optional
        Target nucleus, or a list of them: ``"Fe56"``, ``"Fe-56"``, ``"26-FE-56"``,
        a ZAID such as ``26056`` (``26000`` is natural iron), or a bare symbol
        such as ``"Fe"``, which is natural iron. It selects database datasets
        only; None applies no target filter. JSON files are never filtered by it.
    mt : int, optional
        ENDF MT number to select in the database (2 for elastic scattering). None
        applies no MT filter. Not used for JSON files.
    source : {"database", "json", "auto", "both"}
        Where to read, case-insensitive. ``"database"`` reads the X4Pro database
        and raises if it cannot. ``"json"`` reads the files of *directory* only.
        ``"auto"`` and ``"both"`` behave the same: the database first, then the
        files of *directory* if one is given. If the database is missing or its
        query fails, a warning is printed and the JSON files are used.
    directory : str, optional
        Directory of JSON files. If it holds no ``.json`` file, its ``data``
        subdirectory is tried, then the ``data`` subdirectory of its parent. Only
        read for ``"json"``, ``"auto"`` and ``"both"``; with ``"json"`` and no
        directory nothing is read from it, and the call raises unless
        *supplementary_json_files* loaded something.
    pattern : str
        Glob pattern of the files in *directory*.
    group_by_energy : bool
        If True, key the result by incident energy. If False, key it by dataset
        identifier.
    db_path : str, optional
        Path of the X4Pro database. Defaults to the one set with
        :func:`~kika.exfor.configure`, then to the ``KIKA_X4PRO_DB_PATH``
        environment variable.
    projectile : str
        Projectile, selected in the database (``"n"`` for neutrons). Not used for
        JSON files.
    energy_range : tuple of float, optional
        ``(minimum, maximum)`` incident energy in MeV. A database dataset is kept
        when its energy span overlaps this window, and it is kept whole: its
        points outside the window stay. Datasets with no energies are dropped.
        Not used for JSON files or supplementary files.
    supplementary_json_files : list of str, optional
        JSON files to add after the main source, for experiments the database
        lacks. A file whose identifier is already loaded is skipped, and a
        file that is missing or cannot be read is recorded as failed without
        raising.
    return_load_status : bool
        If True, also return the status of *supplementary_json_files*.
    exclude_experiments : list of str, optional
        Experiments to drop from everything read, supplementary files included.
        ``"20743"`` (up to five characters) drops every subentry of entry 20743;
        ``"20743002"`` and ``"20743/002"`` drop one subentry. The last form joins
        the two parts as written, so ``"20743/2"`` does not match ``"20743002"``.
    target_zaid : int or list of int, optional
        Same as *target*, and used only when *target* is None.

    Returns
    -------
    dict
        With ``group_by_energy=True``, a dict from incident energy in MeV to the
        list of :class:`~kika.exfor.ExforAngularDistribution` that have a point
        at that energy. Energies within 1e-6 MeV of each other share the key of
        the first one met, and keys are not sorted. A dataset with several
        energies is listed under each of them and keeps all its energies. With
        ``group_by_energy=False``, a dict from dataset identifier to the
        distribution. The distributions keep the frame and units of their
        source: database data is in MeV and b/sr.

        With ``return_load_status=True`` a tuple ``(data, status)``. *status* is a
        dict with the lists ``"loaded"`` (entries ``file``, ``id``, ``label``,
        ``n_energies``), ``"skipped"`` (``file``, ``id``, ``reason``) and
        ``"failed"`` (``file``, ``error``). It covers the supplementary files
        only. Files of *directory* that cannot be read are printed as warnings.

    Raises
    ------
    ValueError
        If *source* is not one of the four values.
    FileNotFoundError
        If nothing was read from any source, if *exclude_experiments* removed
        everything, if ``source="database"`` and the database path is missing or
        wrong, or if ``source="json"`` and *directory* holds no file matching
        *pattern*.

    Examples
    --------
    >>> # Load Fe-56 elastic scattering data
    >>> data = read_all_exfor(target="Fe56", mt=2)

    >>> # Load using ZAID
    >>> data = read_all_exfor(target=26056, mt=2)

    >>> # Load both Fe-56 and natural iron (recommended for better data coverage)
    >>> data = read_all_exfor(target=[26056, 26000], mt=2)

    >>> # Load only from your curated JSON files
    >>> data = read_all_exfor(source="json", directory="/path/to/data/")

    >>> # Load with energy filter
    >>> data = read_all_exfor(target="Fe56", energy_range=(1.0, 3.0))

    >>> # Load from database plus a supplementary JSON file
    >>> data = read_all_exfor(
    ...     target="Fe56", mt=2,
    ...     supplementary_json_files=["C:/EXFOR/27673002.json"]
    ... )
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    source = source.lower()
    valid_sources = {"json", "database", "auto", "both"}
    if source not in valid_sources:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {valid_sources}")

    # Handle backwards compatibility: target_zaid -> target
    if target_zaid is not None and target is None:
        target = target_zaid

    all_datasets: List[ExforAngularDistribution] = []
    loaded_ids: set = set()

    # Load from database if requested
    if source in ("database", "auto", "both"):
        try:
            db_datasets = _load_from_database(
                db_path=db_path,
                target=target,
                projectile=projectile,
                mt=mt,
                energy_range=energy_range,
            )
            for ds in db_datasets:
                ds_id = f"{ds.entry}{ds.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(ds)
                    loaded_ids.add(ds_id)
        except FileNotFoundError as e:
            if source == "database":
                raise
            # For auto/both, continue to JSON fallback
            print(f"Warning: Database not available, falling back to JSON: {e}")
        except Exception as e:
            if source == "database":
                raise
            print(f"Warning: Database query failed: {e}")

    # Load from JSON files if requested
    if source in ("json", "auto", "both"):
        if directory is not None:
            try:
                json_datasets = _load_from_json_files(directory, pattern)
                for ds in json_datasets:
                    ds_id = f"{ds.entry}{ds.subentry}"
                    if ds_id not in loaded_ids:
                        all_datasets.append(ds)
                        loaded_ids.add(ds_id)
            except FileNotFoundError as e:
                if source == "json":
                    raise
                # For auto/both, database data is sufficient
                if not all_datasets:
                    raise FileNotFoundError(
                        f"No data available. Database not found and {e}"
                    )

    # Load supplementary JSON files (for experiments not in database)
    supplementary_status = {'loaded': [], 'failed': [], 'skipped': []}

    if supplementary_json_files:
        for json_file in supplementary_json_files:
            if not os.path.exists(json_file):
                supplementary_status['failed'].append({
                    'file': json_file,
                    'error': 'File not found'
                })
                continue
            try:
                exfor = read_exfor(json_file)
                ds_id = f"{exfor.entry}{exfor.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(exfor)
                    loaded_ids.add(ds_id)
                    supplementary_status['loaded'].append({
                        'file': json_file,
                        'id': ds_id,
                        'label': getattr(exfor, 'label', 'unknown'),
                        'n_energies': len(exfor.energies()),
                    })
                else:
                    supplementary_status['skipped'].append({
                        'file': json_file,
                        'id': ds_id,
                        'reason': 'Already in database'
                    })
            except Exception as e:
                supplementary_status['failed'].append({
                    'file': json_file,
                    'error': str(e)
                })

    if not all_datasets:
        raise FileNotFoundError("No EXFOR data found from any source")

    # Filter out excluded experiments
    if exclude_experiments:
        all_datasets = _filter_excluded_experiments(all_datasets, exclude_experiments)

    if not all_datasets:
        raise FileNotFoundError("All EXFOR data was excluded by exclude_experiments filter")

    # Organize results
    if not group_by_energy:
        result = {f"{ds.entry}{ds.subentry}": ds for ds in all_datasets}
        if return_load_status:
            return result, supplementary_status
        return result

    # Group by energy
    energy_data: Dict[float, List[ExforAngularDistribution]] = {}
    for exfor in all_datasets:
        energies_mev = exfor.energies(unit='MeV')
        for energy in energies_mev:
            key = _resolve_energy_key(energy, energy_data)
            if key not in energy_data:
                energy_data[key] = []
            if exfor not in energy_data[key]:
                energy_data[key].append(exfor)

    if return_load_status:
        return energy_data, supplementary_status
    return energy_data


def _load_from_database(
    db_path: str = None,
    target: Union[str, int, List[str], List[int]] = None,
    projectile: str = "n",
    mt: int = None,
    energy_range: Tuple[float, float] = None,
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from X4Pro database.

    Parameters
    ----------
    db_path : str, optional
        Path to database file
    target : str, int, List[str], or List[int], optional
        Target (accepts "Fe56", 26056, "Fe-56", [26056, 26000], etc.)
    projectile : str, optional
        Projectile (default: "n")
    mt : int, optional
        ENDF MT number
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from database
    """
    from kika.exfor.database import X4ProDatabase

    with X4ProDatabase(db_path) as db:
        # Normalize target(s) to database format
        target_pattern = db._normalize_target(target) if target else None

        return db.query_angular_distributions(
            target=target_pattern,
            projectile=projectile,
            mt=mt,
            energy_range=energy_range,
        )


def _load_from_json_files(
    directory: str,
    pattern: str = "*.json",
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from JSON files in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing JSON files
    pattern : str, optional
        Glob pattern for file matching

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from JSON files
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    search_dir = _find_data_directory(directory)
    if search_dir is None:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    json_files = glob.glob(os.path.join(search_dir, pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {search_dir}")

    datasets = []
    for json_file in json_files:
        try:
            exfor = read_exfor(json_file)
            datasets.append(exfor)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return datasets


def _find_data_directory(directory: str) -> Optional[str]:
    """
    Find directory containing JSON files.

    Checks the given directory, then looks for 'data' subdirectory.

    Parameters:
        directory: Starting directory path

    Returns:
        Path to directory with JSON files, or None if not found
    """
    # Check if directory exists and has JSON files
    if os.path.isdir(directory):
        if glob.glob(os.path.join(directory, "*.json")):
            return directory
        # Try 'data' subdirectory
        data_subdir = os.path.join(directory, "data")
        if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
            return data_subdir

    # Try parent directory with 'data' subdirectory
    parent_dir = os.path.dirname(directory.rstrip("/\\"))
    data_subdir = os.path.join(parent_dir, "data")
    if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
        return data_subdir

    return None


def _resolve_energy_key(energy: float, energy_dict: Dict[float, List]) -> float:
    """
    Find existing energy key within tolerance, or return the new energy.

    Parameters:
        energy: Energy value to match
        energy_dict: Existing dictionary with energy keys

    Returns:
        Existing key if within tolerance, otherwise the input energy
    """
    for existing in energy_dict.keys():
        if math.isclose(existing, energy, rel_tol=0.0, abs_tol=ENERGY_MATCH_ABS_TOL):
            return existing
    return energy


def _filter_excluded_experiments(
    datasets: List["ExforAngularDistribution"],
    exclude_experiments: List[str],
) -> List["ExforAngularDistribution"]:
    """
    Filter out experiments matching exclusion patterns.

    Parameters
    ----------
    datasets : List[ExforAngularDistribution]
        List of datasets to filter
    exclude_experiments : List[str]
        List of experiment IDs to exclude. Accepts:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above

    Returns
    -------
    List[ExforAngularDistribution]
        Filtered list with excluded experiments removed
    """
    if not exclude_experiments:
        return datasets

    # Parse exclusion patterns
    exclusion_set = set()
    prefix_exclusions = set()

    for item in exclude_experiments:
        item = item.strip()
        if not item:
            continue

        # Handle "entry/subentry" format
        if "/" in item:
            parts = item.split("/")
            entry = parts[0].strip()
            subentry = parts[1].strip() if len(parts) > 1 else ""
            exclusion_set.add(entry + subentry)
        elif len(item) <= 5:
            # Short ID - treat as entry prefix (matches all subentries)
            prefix_exclusions.add(item)
        else:
            # Full dataset ID
            exclusion_set.add(item)

    # Filter datasets
    filtered = []
    for ds in datasets:
        ds_id = f"{ds.entry}{ds.subentry}"

        # Check exact match
        if ds_id in exclusion_set:
            continue

        # Check prefix match
        excluded_by_prefix = False
        for prefix in prefix_exclusions:
            if ds_id.startswith(prefix):
                excluded_by_prefix = True
                break

        if not excluded_by_prefix:
            filtered.append(ds)

    return filtered

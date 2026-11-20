"""
Angular Distribution Analysis Utilities

.. deprecated::
    This module is deprecated. Please use the new class-based API instead:

    - load_exfor_data() -> use read_exfor() and ExforAngularDistribution
    - load_all_exfor_data() -> use read_all_exfor()
    - cos_cm_from_cos_lab() -> use kika.exfor.transforms.cos_cm_from_cos_lab()
    - transform_lab_to_cm() -> use kika.exfor.transforms.transform_lab_to_cm()

    Example:
        from kika.exfor import read_exfor
        exfor = read_exfor('/path/to/file.json')
        exfor_cm = exfor.convert_to_cm()

This module provides functions for analyzing and plotting angular distributions
from both ACE nuclear data files and experimental EXFOR data.

Functions:
    ACE Data Processing:
    - extract_angular_distribution: Extract angular distribution data from ACE files
    - calculate_differential_cross_section: Calculate dσ/dΩ using ACE data
    - cosine_to_angle_degrees, angle_degrees_to_cosine: Convert between μ and θ

    EXFOR Data Processing:
    - cos_cm_from_cos_lab: Convert LAB to CM frame cosines
    - jacobian_cm_to_lab: Calculate transformation Jacobian
    - transform_lab_to_cm: Transform LAB to CM frame
    - load_exfor_data: Load and process EXFOR JSON data
    - extract_experiment_info: Extract author and year from metadata

    Plotting:
    - plot_angular_distribution: Plot ACE data only
    - plot_combined_angular_distribution: Plot ACE data with multiple experimental datasets

Author: KIKA Development Team
Date: September 2025
"""

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Optional, Any


def _deprecation_warning(old_name: str, new_name: str, stacklevel: int = 3):
    """Issue deprecation warning for legacy functions."""
    warnings.warn(
        f"{old_name}() is deprecated. Use {new_name} instead. "
        f"See 'from kika.exfor import read_exfor' for the new API.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


ENERGY_MATCH_ABS_TOL = 1e-6


# ============================================================================
# ACE Data Processing Functions
# ============================================================================

def extract_angular_distribution(ace_data, energy: float, mt: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosines and probability density of an ACE angular distribution at one energy.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``.

    The distribution is ``ace_data.angular_distributions.elastic`` for MT 2 and
    ``ace_data.angular_distributions.incident_neutron[mt]`` otherwise. Its table
    at *energy* is taken as the ACE distribution gives it with
    ``to_dataframe(energy, interpolate=False)``: no regular grid is imposed, but
    the points depend on the table type. For a tabulated distribution at an energy
    between two table energies, they are the union of the two cosine grids with
    the tables mixed; an equiprobable table gives bin centres. The frame is the
    ACE table's and is not checked: for MT 2 it is CM (ACE's convention), for
    other MTs it is the reaction's own (TYR), which this function does not
    consult. Nothing is converted.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``) with angular distributions.
    energy : float
        Incident energy in MeV.
    mt : int
        ENDF MT number: 2 for elastic scattering, or a key of
        ``angular_distributions.incident_neutron``.

    Returns
    -------
    cosines : numpy.ndarray
        Cosines of the scattering angle, in the frame of the table.
    pdf : numpy.ndarray
        The density ``f(mu, E)`` per unit cosine, as the table gives it. The
        function ``calculate_differential_cross_section`` takes it as normalised
        to 1 over the cosine.

    Raises
    ------
    ValueError
        If *mt* has no angular distribution, or the distribution gives no table.
    """
    
    # Get the appropriate angular distribution
    if mt == 2:  # Elastic scattering
        angular_dist = ace_data.angular_distributions.elastic
    else:
        # Handle other MT numbers if needed
        if mt in ace_data.angular_distributions.incident_neutron:
            angular_dist = ace_data.angular_distributions.incident_neutron[mt]
        else:
            raise ValueError(f"MT={mt} angular distribution not found")
    
    if angular_dist is None:
        raise ValueError(f"No angular distribution found for MT={mt}")
    
    # Get the DataFrame with angular distribution data at specified energy
    df = angular_dist.to_dataframe(energy, interpolate=False)
    
    if df is None:
        raise ValueError("Could not extract angular distribution data")
    
    # Extract cosines and PDF
    cosines = df['cosine'].values
    pdf = df['pdf'].values
    
    return cosines, pdf


def calculate_differential_cross_section(ace_data, energy: float, mt: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Differential cross-section dσ/dΩ = σ(E)/(2π) × f(μ,E) of an ACE reaction.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``.

    ``f`` is the density per unit cosine from :func:`extract_angular_distribution`,
    taken as integrating to 1 over the cosine, so dividing by 2π gives a density per
    steradian. ``σ(E)`` is the cross-section of reaction *mt*, interpolated
    linearly in energy (``numpy.interp``). Outside the tabulated energies it is
    the end value, and a warning is printed. The frame is the ACE table's and is
    not checked: for MT 2 it is CM (ACE's convention), for other MTs the
    reaction's own (TYR), which is not consulted. Nothing is converted.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``).
    energy : float
        Incident energy in MeV.
    mt : int
        ENDF MT number: 2 for elastic scattering.

    Returns
    -------
    cosines : numpy.ndarray
        Cosines of the scattering angle.
    dsigma_domega : numpy.ndarray
        dσ/dΩ in b/sr at each cosine.
    sigma_total : float
        The cross-section ``σ(E)`` of reaction *mt* at *energy*, in b. It is the
        total cross-section only if *mt* is 1.

    Raises
    ------
    ValueError
        If the cross-section of reaction *mt* is not available in *ace_data*
        (raised by ``get_cross_section``), or from
        :func:`extract_angular_distribution`.
    TypeError
        If *mt* is not an integer (raised by ``get_cross_section``).
    """
    
    # Get the total cross section at this energy
    xs_data = ace_data.get_cross_section(mt)
    
    # Interpolate cross section at the requested energy
    if energy < xs_data['Energy'].min() or energy > xs_data['Energy'].max():
        print(f"Warning: Energy {energy} MeV is outside cross section range "
              f"({xs_data['Energy'].min():.2e} - {xs_data['Energy'].max():.2e} MeV)")
    
    # Interpolate the cross section
    sigma_total = np.interp(energy, xs_data['Energy'], xs_data[f'MT={mt}'])
    
    # Get angular distribution data
    cosines, pdf = extract_angular_distribution(ace_data, energy, mt)
    
    # Calculate differential cross section: dσ/dΩ = σ/(2π) × f(μ,E)
    dsigma_domega = (sigma_total / (2 * np.pi)) * pdf
    
    return cosines, dsigma_domega, sigma_total


def cosine_to_angle_degrees(cosines: np.ndarray) -> np.ndarray:
    """
    Scattering angles in degrees from their cosines, θ = arccos(μ).

    Kept for existing scripts; new code uses the classes of ``kika.exfor``. The
    frame is whatever the cosines are in.

    Parameters
    ----------
    cosines : array_like
        Cosines μ of the scattering angle. Values outside [-1, 1] are clipped to
        the nearest end.

    Returns
    -------
    numpy.ndarray
        Angles in degrees, in [0, 180], with the shape of *cosines*.
    """
    # θ = arccos(μ)
    angles_rad = np.arccos(np.clip(cosines, -1, 1))  # clip to handle numerical errors
    angles_deg = np.degrees(angles_rad)
    return angles_deg


def angle_degrees_to_cosine(angles_deg: np.ndarray) -> np.ndarray:
    """
    Cosines of scattering angles given in degrees, μ = cos(θ).

    Kept for existing scripts; new code uses the classes of ``kika.exfor``. The
    frame is whatever the angles are in.

    Parameters
    ----------
    angles_deg : array_like
        Scattering angles in degrees.

    Returns
    -------
    numpy.ndarray
        Cosines of the angles, with the shape of *angles_deg*.
    """
    angles_rad = np.radians(angles_deg)
    cosines = np.cos(angles_rad)
    return cosines


# ============================================================================
# EXFOR Data Processing Functions
# ============================================================================

def cos_cm_from_cos_lab(mu_L: np.ndarray, alpha: float) -> np.ndarray:
    """
    Convert cos(theta) from LAB to CM (forward branch).
    Inverse of mu_L = (mu_c + alpha) / sqrt(1 + 2 alpha mu_c + alpha^2).
    Valid and monotonic for alpha = m_proj/m_targ < 1 (e.g. n on Fe-56).
    
    Parameters:
    -----------
    mu_L : np.ndarray
        Cosines in LAB frame
    alpha : float
        Mass ratio m_proj/m_targ
        
    Returns:
    --------
    np.ndarray
        Cosines in CM frame
    """
    return -alpha*(1 - mu_L**2) + mu_L*np.sqrt(1 - alpha**2*(1 - mu_L**2))


def jacobian_cm_to_lab(mu_c: np.ndarray, alpha: float) -> np.ndarray:
    """
    Returns dOmega_CM/dOmega_LAB = (1 + alpha^2 + 2 alpha mu_c)^(3/2) / |1 + alpha mu_c|.
    Note name: this is the *inverse* of dOmega_LAB/dOmega_CM.
    To convert LAB -> CM cross section: (dσ/dΩ)_CM = (dσ/dΩ)_LAB * (dΩ_L/dΩ_C) = (dσ/dΩ)_LAB / (dΩ_C/dΩ_L).
    So divide LAB values by this function to get CM values.
    
    Parameters:
    -----------
    mu_c : np.ndarray
        Cosines in CM frame
    alpha : float
        Mass ratio m_proj/m_targ
        
    Returns:
    --------
    np.ndarray
        Jacobian for frame transformation
    """
    return (1 + alpha**2 + 2*alpha*mu_c)**1.5 / np.abs(1 + alpha*mu_c)


def transform_lab_to_cm(mu_L_arr: np.ndarray, dsdo_L_arr: np.ndarray, 
                       m_proj_u: float, m_targ_u: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform differential cross section from LAB to CM frame.
    
    Parameters:
    -----------
    mu_L_arr : np.ndarray
        LAB frame cosines
    dsdo_L_arr : np.ndarray
        LAB frame differential cross sections
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
        
    Returns:
    --------
    mu_c : np.ndarray
        CM frame cosines
    dsdo_c : np.ndarray
        CM frame differential cross sections
    """
    alpha = m_proj_u / m_targ_u
    mu_c = cos_cm_from_cos_lab(mu_L_arr, alpha)
    J = jacobian_cm_to_lab(mu_c, alpha)  # = dΩ_C/dΩ_L
    dsdo_c = dsdo_L_arr / J              # multiply by dΩ_L/dΩ_C
    return mu_c, dsdo_c


def load_exfor_data(json_file: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read an EXFOR angular-distribution JSON file into a table in MeV, degrees and b/sr.

    .. deprecated::
        Kept for existing scripts: ``load_exfor_data() is deprecated. Use
        read_exfor() instead``, as its warning says. New code uses
        :func:`~kika.exfor.read_exfor` and the classes.

    Only the older file layout is read: energy blocks under ``energies`` with the
    energy in ``E``, and points with ``angle``, ``result`` and ``error_stat``. The
    keys of the newer layout (``value``, ``cross_section``, ``uncertainty_stat``,
    ``units["cross_section"]``) are not read, and a file that uses only them gives
    zeros. Energies are converted to MeV when ``units["energy"]`` is ``keV`` or ``eV``
    (and kept for ``MeV`` or ``M``). The cross-section and ``error_stat`` are
    divided by 1000 for ``units["dsig"]`` equal to ``MB/SR`` and by 1e6 for
    ``MUB/SR``; they are kept for ``B/SR`` or when ``dsig`` is absent, and kept
    unchanged for any other value. Angles in ``COS`` become degrees, and ``ADEG``
    is kept. An unknown energy or angle unit prints a warning and is taken as MeV
    or degrees. Unit names are case-insensitive. The frame is not converted: it is
    only recorded.

    Parameters
    ----------
    json_file : str
        Path to the JSON file.

    Returns
    -------
    df : pandas.DataFrame
        One row per data point, with the columns ``energy`` (MeV), ``angle``
        (degrees), ``dsig`` (dσ/dΩ in b/sr), ``error_stat`` (absolute statistical
        uncertainty in b/sr) and ``frame`` (the file's ``angle_frame``, ``"LAB"``
        or ``"CM"``, or ``"Unknown"``). Systematic uncertainties are not read.
    meta : dict
        ``entry``, ``subentry``, ``citation``, ``reaction``, ``quantity``,
        ``units`` and ``angle_frame`` as in the file, ``energy_resolution_inputs``
        (``flight_path_m`` and ``time_resolution_ns`` from the file's
        ``energy_resolution_inputs``, or None) and ``filename``. ``units`` keeps the
        file's original units, not the converted ones.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the file is not valid JSON.
    """
    _deprecation_warning("load_exfor_data", "read_exfor()")
    with open(json_file, "r") as f:
        obj = json.load(f)
    
    # Extract metadata
    meta = {
        'entry': obj.get('entry', 'Unknown'),
        'subentry': obj.get('subentry', 'Unknown'),
        'citation': obj.get('citation', {}),
        'reaction': obj.get('reaction', 'Unknown'),
        'quantity': obj.get('quantity', 'Unknown'),
        'units': obj.get('units', {}),
        'angle_frame': obj.get('angle_frame', 'Unknown')
    }

    # Extract energy resolution inputs from method section (for TOF experiments)
    method = obj.get('method', {})
    energy_res_inputs = method.get('energy_resolution_inputs', {})
    if energy_res_inputs:
        distance_info = energy_res_inputs.get('distance', {})
        time_res_info = energy_res_inputs.get('time_resolution', {})
        meta['energy_resolution_inputs'] = {
            'flight_path_m': distance_info.get('value'),
            'time_resolution_ns': time_res_info.get('value'),
        }
    else:
        meta['energy_resolution_inputs'] = None

    # Parse data from all energies
    data_rows = []
    for energy_block in obj.get('energies', []):
        energy = energy_block.get('E', 0.0)
        for data_point in energy_block.get('data', []):
            row = {
                'energy': energy,
                'angle': data_point.get('angle', 0.0),
                'dsig': data_point.get('result', 0.0),
                'error_stat': data_point.get('error_stat', 0.0),
                'frame': meta['angle_frame'],
            }
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Convert units to be consistent with ACE data (energy in MeV, cross section in barns/sr, angle in degrees)
    
    # Handle energy units - convert to MeV if needed
    energy_unit = meta['units'].get('energy', 'MEV').upper()
    if energy_unit == 'KEV':
        # Convert keV to MeV
        df['energy'] = df['energy'] / 1000.0
        print(f"Note: Converted energy from keV to MeV in {json_file}")
    elif energy_unit == 'EV':
        # Convert eV to MeV
        df['energy'] = df['energy'] / 1e6
        print(f"Note: Converted energy from eV to MeV in {json_file}")
    elif energy_unit in ['MEV', 'M']:
        # Already in MeV, no conversion needed
        pass
    else:
        print(f"Warning: Unknown energy unit '{energy_unit}'. Expected 'MEV' or 'KEV'. Assuming MeV.")
    
    # Handle different cross section units
    dsig_unit = meta['units'].get('dsig', 'B/SR').upper()
    if dsig_unit == 'MB/SR':
        # Convert millibarns to barns
        df['dsig'] = df['dsig'] / 1000.0
        df['error_stat'] = df['error_stat'] / 1000.0
    elif dsig_unit == 'MUB/SR':
        # Convert microbarns to barns
        df['dsig'] = df['dsig'] / 1e6
        df['error_stat'] = df['error_stat'] / 1e6
    # B/SR is already correct
    
    # Handle angle units - convert to degrees if needed
    angle_unit = meta['units'].get('angle', 'ADEG').upper()
    if angle_unit == 'ADEG':
        # Already in degrees, no conversion needed
        pass
    elif angle_unit == 'COS':
        # Convert cosines to degrees
        # Clamp values to [-1, 1] to avoid numerical issues with arccos
        cosines = np.clip(df['angle'].values, -1.0, 1.0)
        df['angle'] = np.degrees(np.arccos(cosines))
    else:
        print(f"Warning: Unknown angle unit '{angle_unit}'. Expected 'ADEG' or 'COS'. Assuming degrees.")
    
    # Add filename to metadata for identification
    import os
    meta['filename'] = os.path.basename(json_file)
    
    return df, meta


def extract_experiment_info(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Legend label and year of an experiment, from the ``meta`` of an EXFOR file.

    .. deprecated::
        Kept for existing scripts: ``extract_experiment_info() is deprecated. Use
        ExforAngularDistribution.label instead``, as its warning says. New code
        uses the ``label`` property of the classes.

    The label is the surname of the first author, followed by ``et al.`` when there
    are several authors, and the year: ``"Kinney et al. (1976)"``. The surname is
    what follows the last period of the name, else its last word. With no author
    the label is built from ``meta["filename"]`` without ``.json`` and with
    underscores as spaces, or is ``"Experiment (year)"``.

    Parameters
    ----------
    meta : dict
        The metadata dict returned by :func:`load_exfor_data`. Only
        ``meta["citation"]`` (with ``authors`` and ``year``) and ``meta["filename"]``
        are read.

    Returns
    -------
    experiment_label : str
        The label, with the year in brackets.
    year : str
        The year as text, ``"Unknown"`` when the citation has none.
    """
    _deprecation_warning("extract_experiment_info", "ExforAngularDistribution.label")
    # Extract year from citation
    citation = meta.get('citation', {})
    experiment_year = str(citation.get('year', 'Unknown'))
    
    # Create a label from the first author's last name
    authors = citation.get('authors', [])
    if isinstance(authors, list) and len(authors) > 0:
        first_author = authors[0]
        # Extract last name (usually the part after the last period or the whole name)
        if '.' in first_author:
            # Handle format like "W.L.Rodgers" -> "Rodgers"
            last_name = first_author.split('.')[-1].strip()
        else:
            # If no period, take the last word as surname
            last_name = first_author.split()[-1] if ' ' in first_author else first_author
        
        if len(authors) > 1:
            experiment_label = f"{last_name} et al. ({experiment_year})"
        else:
            experiment_label = f"{last_name} ({experiment_year})"
    else:
        # Use filename if available, otherwise fall back to generic label
        filename = meta.get('filename', 'Unknown')
        if filename != 'Unknown':
            # Remove extension and clean up filename for display
            base_name = filename.replace('.json', '').replace('_', ' ')
            experiment_label = f"{base_name} ({experiment_year})"
        else:
            experiment_label = f"Experiment ({experiment_year})"
    
    return experiment_label, experiment_year


def load_all_exfor_data(directory: str) -> Tuple[Dict[float, List[Tuple[pd.DataFrame, Dict[str, Any]]]], List[float]]:
    """
    Read every EXFOR JSON file of a directory with :func:`load_exfor_data`, by energy.

    .. deprecated::
        Kept for existing scripts: ``load_all_exfor_data() is deprecated. Use
        read_all_exfor() instead``, as its warning says. New code uses
        :func:`~kika.exfor.read_all_exfor`.

    The files are those matching ``*.json`` in *directory*. If *directory* is not
    a folder, the ``data`` folder next to it is used; if it holds no JSON file, its
    own ``data`` subfolder is used. Each file is read as :func:`load_exfor_data`
    reads it (older layout only), so energies are in MeV and cross-sections in
    b/sr. A file that cannot be read is skipped with a printed warning. If no file
    is found, the result is empty and nothing is raised.

    Parameters
    ----------
    directory : str
        Directory of the JSON files.

    Returns
    -------
    energy_data : dict
        From incident energy in MeV to a list of ``(df, meta)`` pairs, one per file
        with points at that energy: ``df`` holds that file's rows at that energy
        (columns as in :func:`load_exfor_data`) and ``meta`` is its metadata dict.
        Energies within 1e-6 MeV of each other share the key of the first one met.
    sorted_energies : list of float
        The keys of *energy_data* in increasing order.
    """
    _deprecation_warning("load_all_exfor_data", "read_all_exfor()")
    import os
    import glob
    
    energy_data = {}
    all_energies = set()

    def _resolve_energy_key(value: float) -> float:
        for existing in energy_data.keys():
            if math.isclose(existing, value, rel_tol=0.0, abs_tol=ENERGY_MATCH_ABS_TOL):
                return existing
        return value
    
    # Find all JSON files in directory
    # First check if directory exists, if not check for 'data' subdirectory
    search_dir = directory
    if not os.path.isdir(directory):
        # Try parent directory with 'data' subdirectory
        parent_dir = os.path.dirname(directory.rstrip('/\\'))
        data_subdir = os.path.join(parent_dir, 'data')
        if os.path.isdir(data_subdir):
            search_dir = data_subdir
            print(f"Note: Using data subdirectory: {data_subdir}")
    elif not glob.glob(os.path.join(directory, "*.json")):
        # Directory exists but has no JSON files, try 'data' subdirectory
        data_subdir = os.path.join(directory, 'data')
        if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
            search_dir = data_subdir
            print(f"Note: JSON files found in data subdirectory: {data_subdir}")
    
    json_files = glob.glob(os.path.join(search_dir, "*.json"))
    
    for json_file in json_files:
        try:
            df, meta = load_exfor_data(json_file)
            
            # Group by energy
            for energy in df['energy'].unique():
                energy = float(energy)
                representative_energy = _resolve_energy_key(energy)

                if representative_energy not in energy_data:
                    energy_data[representative_energy] = []

                mask = np.isclose(
                    df['energy'].astype(float).values,
                    energy,
                    rtol=0.0,
                    atol=ENERGY_MATCH_ABS_TOL
                )
                df_energy = df.loc[mask].copy()

                if df_energy.empty:
                    continue

                energy_data[representative_energy].append((df_energy, meta))
                all_energies.add(representative_energy)
                
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    sorted_energies = sorted(list(all_energies))
    return energy_data, sorted_energies


def plot_all_energies_comparison(ace_data, exfor_directory: str, 
                                m_proj_u: float = 1.008665, m_targ_u: float = 55.93494,
                                mt: int = 2, max_plots_per_figure: int = 6,
                                figsize: Tuple[int, int] = (15, 10)):
    """
    One panel per EXFOR energy of a directory, comparing the ACE curve with the data.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``, with
    :func:`~kika.exfor.plot_multiple_energies`.

    The files are read with :func:`load_all_exfor_data`, so every energy found in
    them gets a panel, at most 3 panels per row and *max_plots_per_figure* per
    figure. A panel shows dσ/dΩ in b/sr against the CM scattering angle in degrees.
    The ACE curve is the one of :func:`calculate_differential_cross_section` at that
    energy, drawn as stored (the table's frame, not checked: CM for MT 2 by ACE's
    convention, the reaction's own for other MTs). Data in the LAB frame are
    converted to CM with ``m_proj_u`` and ``m_targ_u`` (cross-section and
    statistical error divided by the Jacobian dΩ_CM/dΩ_LAB); data in the CM frame
    or in an unrecorded frame are drawn as they are. The error bars are the
    statistical errors only. A panel whose ACE curve or data cannot be drawn shows
    the error text instead.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``).
    exfor_directory : str
        Directory of the EXFOR JSON files, searched as in :func:`load_all_exfor_data`.
    m_proj_u, m_targ_u : float
        Projectile and target masses in atomic mass units, for the LAB to CM
        conversion. Defaults: neutron (1.008665 u) and Fe-56 (55.93494 u).
    mt : int
        ENDF MT number of the ACE reaction: 2 for elastic scattering.
    max_plots_per_figure : int
        Number of energy panels in one figure.
    figsize : tuple of int
        Width and height of each figure, in inches.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per *max_plots_per_figure* energies, in increasing energy. Empty
        when no data are found.
    """
    # Load all experimental data
    energy_data, sorted_energies = load_all_exfor_data(exfor_directory)
    
    print(f"Found experimental data at {len(sorted_energies)} energies: {sorted_energies} MeV")
    
    figures = []
    
    # Create figures with subplots
    num_figures = (len(sorted_energies) + max_plots_per_figure - 1) // max_plots_per_figure
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_plots_per_figure
        end_idx = min(start_idx + max_plots_per_figure, len(sorted_energies))
        energies_this_fig = sorted_energies[start_idx:end_idx]
        
        n_plots = len(energies_this_fig)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, energy in enumerate(energies_this_fig):
            ax = axes[i]
            
            try:
                # Plot ACE data
                cosines_ace, dsigma_domega_ace, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)
                angles_deg_ace = cosine_to_angle_degrees(cosines_ace)
                
                ax.plot(angles_deg_ace, dsigma_domega_ace, 'b-', linewidth=2, 
                       label=f'ACE (σ={sigma_total:.3f}b)', zorder=10)
                ax.plot(angles_deg_ace, dsigma_domega_ace, 'bo', markersize=3, zorder=10)
                
                # Plot experimental data
                colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                color_idx = 0
                
                for df_exp, meta_exp in energy_data[energy]:
                    experiment_label, year = extract_experiment_info(meta_exp)
                    color = colors[color_idx % len(colors)]
                    
                    # Get data
                    angles = df_exp["angle"].values
                    dsig = df_exp["dsig"].values
                    err_stat = df_exp["error_stat"].values
                    data_frame = df_exp["frame"].iloc[0]
                    
                    # Convert to CM frame if needed
                    if data_frame.upper() == 'LAB':
                        mu_lab = np.cos(np.deg2rad(angles))
                        mu_cm, dsig_cm = transform_lab_to_cm(mu_lab, dsig, m_proj_u, m_targ_u)
                        ang_plot = np.rad2deg(np.arccos(np.clip(mu_cm, -1.0, 1.0)))
                        dsig_plot = dsig_cm
                        
                        alpha = m_proj_u / m_targ_u
                        J = jacobian_cm_to_lab(mu_cm, alpha)
                        err_plot = err_stat / J
                    else:
                        ang_plot = angles
                        dsig_plot = dsig
                        err_plot = err_stat
                    
                    # Plot experimental data
                    ax.errorbar(ang_plot, dsig_plot, yerr=err_plot, 
                              fmt='o', color=color, markersize=4, capsize=3, 
                              label=f'{experiment_label}', alpha=0.8)
                    
                    color_idx += 1
                
                # Formatting
                ax.set_title(f'E = {energy} MeV', fontsize=12)
                ax.set_xlabel('Angle (degrees, CM)', fontsize=10)
                ax.set_ylabel('dσ/dΩ (barns/sr)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlim(0, 180)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error at {energy} MeV:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=8)
                ax.set_title(f'E = {energy} MeV (Error)')
        
        # Remove empty subplots
        for i in range(n_plots, len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_angular_distribution(ace_data, energy: float, mt: int = 2, title: Optional[str] = None, 
                            figsize: Tuple[int, int] = (10, 6), log_scale: bool = False, 
                            show_total_xs: bool = True):
    """
    Plot the ACE differential cross-section dσ/dΩ in b/sr against the cosine.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``.

    The curve is the one of :func:`calculate_differential_cross_section`, drawn as
    stored in the ACE table (its frame is not checked: CM for MT 2 by ACE's
    convention, the reaction's own for other MTs). The x-axis is the
    cosine of the scattering angle, from -1 to 1.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``).
    energy : float
        Incident energy in MeV.
    mt : int
        ENDF MT number: 2 for elastic scattering.
    title : str, optional
        Title of the plot. None builds one with the energy, *mt* and, if
        *show_total_xs* is True, the cross-section ``σ(E)`` of reaction *mt* in b.
    figsize : tuple of int
        Width and height of the figure, in inches.
    log_scale : bool
        If True, the y-axis is logarithmic.
    show_total_xs : bool
        Whether the default title includes ``σ(E)``. Ignored when *title* is given.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    
    # Calculate differential cross section
    cosines, dsigma_domega, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data using cosines
    ax.plot(cosines, dsigma_domega, 'bo-', linewidth=2, markersize=4, label=f'ACE data (E = {energy} MeV)')
    
    # Set up the plot
    ax.set_xlabel('cos(θ) - Cosine of Scattering Angle', fontsize=12)
    ax.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (barns/steradian)', fontsize=12)
    
    if log_scale:
        ax.set_yscale('log')
    
    # Set title
    if title is None:
        if show_total_xs:
            title = f'Angular Distribution at {energy} MeV (MT={mt})\\n' + f'Total Cross Section: {sigma_total:.4f} barns'
        else:
            title = f'Angular Distribution at {energy} MeV (MT={mt})'
    
    ax.set_title(title, fontsize=14)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis limits and ticks for cosine
    ax.set_xlim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    
    plt.tight_layout()
    
    return fig, ax


def plot_combined_angular_distribution(ace_data, energy: float, exfor_files: List[str] = None,
                                     m_proj_u: float = 1.008665, m_targ_u: float = 55.93494,
                                     mt: int = 2, title: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 8), log_scale: bool = False,
                                     frame: str = 'CM'):
    """
    Plot the ACE dσ/dΩ in b/sr against the CM angle in degrees, with EXFOR files.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``, with
    :func:`~kika.exfor.plot_exfor_ace_comparison`.

    The ACE curve is the one of :func:`calculate_differential_cross_section`,
    drawn as stored (the table's frame, not checked: CM for MT 2 by ACE's
    convention, the reaction's own for other MTs). Each file is read with
    :func:`load_exfor_data` (older layout only) and its points within 0.1 MeV of
    *energy* are drawn, all together if the file has several energies in that
    window. Points in the LAB frame are converted to CM with *m_proj_u* and
    *m_targ_u* (cross-section and statistical error divided by the Jacobian
    dΩ_CM/dΩ_LAB); points in the CM frame are drawn as they are, and so are those
    of an unrecorded frame (a message is printed). The error bars are the
    statistical errors only. A file that cannot be read is skipped with a printed
    warning, and a file with no point in the window is left out.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``).
    energy : float
        Incident energy in MeV, for the ACE curve and the 0.1 MeV window.
    exfor_files : list of str, optional
        Paths of the EXFOR JSON files. None draws the ACE curve alone.
    m_proj_u, m_targ_u : float
        Projectile and target masses in atomic mass units, for the LAB to CM
        conversion. Defaults: neutron (1.008665 u) and Fe-56 (55.93494 u).
    mt : int
        ENDF MT number of the ACE reaction: 2 for elastic scattering.
    title : str, optional
        Title of the plot. None builds one with the energy, *mt*, the frame label
        and ``σ(E)``.
    figsize : tuple of int
        Width and height of the figure, in inches.
    log_scale : bool
        If True, the y-axis is logarithmic.
    frame : {'CM', 'LAB'}
        Only the label of the x-axis and of the default title. It converts
        nothing: LAB points are converted to CM whatever *frame* says, and with
        ``'LAB'`` the CM points are kept as CM and a warning is printed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    
    # Calculate ACE differential cross section
    cosines_ace, dsigma_domega_ace, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)
    angles_deg_ace = cosine_to_angle_degrees(cosines_ace)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ACE data
    ax.plot(angles_deg_ace, dsigma_domega_ace, 'b-', linewidth=2, 
            label=f'ACE data (E = {energy} MeV)', zorder=10)
    ax.plot(angles_deg_ace, dsigma_domega_ace, 'bo', markersize=4, zorder=10)
    
    # Plot experimental data if provided
    if exfor_files:
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        color_idx = 0
        
        for exfor_file in exfor_files:
            try:
                df, meta = load_exfor_data(exfor_file)
                experiment_label, year = extract_experiment_info(meta)
                
                # Filter data for the specific energy (within tolerance)
                df_filtered = df.copy()
                energy_tolerance = 0.1  # MeV
                df_energy = df_filtered[abs(df_filtered['energy'] - energy) <= energy_tolerance].copy()
                
                if len(df_energy) == 0:
                    continue  # No data at this energy
                
                color = colors[color_idx % len(colors)]
                
                # Get angles and cross sections
                angles = df_energy["angle"].values
                dsig = df_energy["dsig"].values
                err_stat = df_energy["error_stat"].values
                data_frame = df_energy["frame"].iloc[0]  # Frame should be consistent within experiment
                
                # Always convert to CM frame for comparison with ACE data
                if data_frame.upper() == 'LAB':
                    # Transform LAB to CM frame
                    mu_lab = np.cos(np.deg2rad(angles))
                    mu_cm, dsig_cm = transform_lab_to_cm(mu_lab, dsig, m_proj_u, m_targ_u)
                    ang_plot = np.rad2deg(np.arccos(np.clip(mu_cm, -1.0, 1.0)))
                    dsig_plot = dsig_cm
                    
                    # Transform errors
                    alpha = m_proj_u / m_targ_u
                    J = jacobian_cm_to_lab(mu_cm, alpha)
                    err_plot = err_stat / J
                elif data_frame.upper() == 'CM':
                    # Already in CM frame
                    ang_plot = angles
                    dsig_plot = dsig
                    err_plot = err_stat
                else:
                    print(f"Warning: Unknown frame '{data_frame}' in {exfor_file}, assuming CM")
                    ang_plot = angles
                    dsig_plot = dsig
                    err_plot = err_stat
                
                # If user wants LAB frame but data is CM, transform back
                if frame.upper() == 'LAB' and data_frame.upper() == 'CM':
                    # This is more complex and requires inverse transformation
                    # For now, we'll keep CM data and warn user
                    print(f"Warning: Requested LAB frame but experimental data is in CM. Keeping CM frame.")
                
                # Create label
                label = experiment_label
                
                # Plot with error bars
                ax.errorbar(ang_plot, dsig_plot, yerr=err_plot, 
                          fmt='o', color=color, markersize=6, capsize=4, 
                          label=label, alpha=0.8)
                
                color_idx += 1
                    
            except Exception as e:
                print(f"Warning: Could not load {exfor_file}: {e}")
                continue
    
    # Formatting
    ax.set_xlabel(f'Scattering Angle (degrees) - {frame} frame', fontsize=12)
    ax.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (barns/steradian)', fontsize=12)
    
    if log_scale:
        ax.set_yscale('log')
    
    # Title with formula
    if title is None:
        title = f'Differential Cross Section at {energy} MeV (MT={mt}) - {frame} Frame\n' + \
                r'$\frac{d\sigma}{d\Omega}(\mu,E) = \frac{\sigma(E)}{2\pi} \cdot f(\mu,E)$' + \
                f'\nACE Total σ = {sigma_total:.4f} barns'
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    
    plt.tight_layout()
    return fig, ax


def plot_individual_energy_comparisons(ace_data, exfor_directory: str, m_proj_u: float, m_targ_u: float,
                                     mt: int = 2, figsize: Tuple[int, int] = (12, 8)) -> List[plt.Figure]:
    """
    One figure per EXFOR energy of a directory: the ACE curve and the data at it.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``, with
    :func:`~kika.exfor.plot_multiple_energies`.

    The files are read with :func:`load_all_exfor_data`, and every energy found in
    them gets a figure. A figure shows dσ/dΩ in b/sr against the CM angle in
    degrees. The ACE curve is the one of :func:`calculate_differential_cross_section`
    at that energy, drawn as stored (the table's frame, not checked: CM for MT 2
    by ACE's convention, the reaction's own for other MTs), and omitted with a
    printed warning if it cannot be computed. Data whose file records the LAB frame
    are converted to CM with *m_proj_u* and *m_targ_u* (cross-section and
    statistical error divided by the Jacobian dΩ_CM/dΩ_LAB); data in the CM frame
    or in an unrecorded frame are drawn as they are. The error bars are the
    statistical errors only.

    Parameters
    ----------
    ace_data
        Loaded ACE object (``kika.ace``).
    exfor_directory : str
        Directory of the EXFOR JSON files, searched as in :func:`load_all_exfor_data`.
    m_proj_u, m_targ_u : float
        Projectile and target masses in atomic mass units, for the LAB to CM
        conversion. They have no default here.
    mt : int
        ENDF MT number of the ACE reaction: 2 for elastic scattering.
    figsize : tuple of int
        Width and height of each figure, in inches.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per energy, in increasing energy. Empty when no data are found.
    """
    # Load all experimental data
    energy_data, sorted_energies = load_all_exfor_data(exfor_directory)
    
    print(f"Found experimental data at {len(sorted_energies)} energies: {sorted_energies} MeV")
    
    figures = []
    
    for energy in sorted_energies:
        print(f"Creating plot for E = {energy} MeV...")
        
        # Create new figure for this energy
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ACE data
        try:
            cosines, dsigma_domega, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)
            angles_deg = cosine_to_angle_degrees(cosines)
            
            ax.plot(angles_deg, dsigma_domega, 
                   label=f'ACE (σ={sigma_total:.4f}b)', 
                   linewidth=2, color='blue')
                   
        except Exception as e:
            print(f"Warning: Could not calculate ACE data for E={energy} MeV: {e}")
            sigma_total = 0.0
        
        # Plot experimental data for this energy
        experiment_data = energy_data[energy]
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_data)))
        
        for i, (df_exp, meta_exp) in enumerate(experiment_data):
            try:
                exp_label, year = extract_experiment_info(meta_exp)
                
                # Get angles and cross sections
                angles = df_exp["angle"].values
                dsig = df_exp["dsig"].values
                err_stat = df_exp["error_stat"].values
                data_frame = meta_exp['angle_frame']
                
                # Always convert to CM frame for comparison with ACE data
                if data_frame.upper() == 'LAB':
                    # Transform LAB to CM frame
                    mu_lab = np.cos(np.deg2rad(angles))
                    mu_cm, dsig_cm = transform_lab_to_cm(mu_lab, dsig, m_proj_u, m_targ_u)
                    ang_plot = np.rad2deg(np.arccos(np.clip(mu_cm, -1.0, 1.0)))
                    dsig_plot = dsig_cm
                    
                    # Transform errors
                    alpha = m_proj_u / m_targ_u
                    J = jacobian_cm_to_lab(mu_cm, alpha)
                    err_plot = err_stat / J
                elif data_frame.upper() == 'CM':
                    # Already in CM frame
                    ang_plot = angles
                    dsig_plot = dsig
                    err_plot = err_stat
                else:
                    print(f"Warning: Unknown frame '{data_frame}', assuming CM")
                    ang_plot = angles
                    dsig_plot = dsig
                    err_plot = err_stat
                
                # Plot experimental data
                ax.errorbar(ang_plot, dsig_plot, yerr=err_plot,
                           fmt='o', color=colors[i], label=exp_label,
                           markersize=4, capsize=3, alpha=0.8)
                           
            except Exception as e:
                print(f"Warning: Could not process experiment {i+1} for E={energy} MeV: {e}")
                continue
        
        # Format the plot
        ax.set_xlabel('Scattering Angle (degrees, CM frame)', fontsize=12)
        ax.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (barns/steradian)', fontsize=12)
        
        title = f'Differential Cross Section at E = {energy} MeV (MT={mt})\\n' + \
                r'$\frac{d\sigma}{d\Omega}(\mu,E) = \frac{\sigma(E)}{4\pi} \cdot f(\mu,E)$'
        if sigma_total > 0:
            title += f'\\nACE Total σ = {sigma_total:.4f} barns'
        
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def plot_combined_angular_distribution_multi_ace(ace_dict: Dict[str, Any], energy: float,
                                                exfor_files: List[str] = None,
                                                m_proj_u: float = 1.008665, m_targ_u: float = 55.93494,
                                                mt: int = 2, title: Optional[str] = None,
                                                figsize: Tuple[int, int] = (12, 8), log_scale: bool = False,
                                                frame: str = 'CM'):
    """
    Plot several ACE dσ/dΩ curves in b/sr against the cosine, with EXFOR files.

    Kept for existing scripts; new code uses the classes of ``kika.exfor``, with
    :func:`~kika.exfor.plot_exfor_ace_comparison`.

    Each ACE curve is the one of :func:`calculate_differential_cross_section`,
    drawn as stored (the table's frame, not checked: CM for MT 2 by ACE's
    convention, the reaction's own for other MTs) and never converted; one that
    cannot be computed is skipped with a printed warning. Each file is read with
    :func:`load_exfor_data` (older layout only), and its points within 0.05 MeV of
    *energy* (strictly) are drawn at the cosine of their angle. Only when *frame*
    is ``'CM'`` and the file records the LAB frame are the points converted to CM,
    with *m_proj_u* and *m_targ_u* (cross-section and statistical error divided by
    the Jacobian dΩ_CM/dΩ_LAB). In every other case they are drawn as they are.
    The error bars are the statistical errors only.

    Parameters
    ----------
    ace_dict : dict
        ACE objects to draw, as ``{label: {"data": ace_object, "color": c,
        "linestyle": s}}``. ``color`` (default ``"blue"``) and ``linestyle``
        (default ``"-"``) are optional.
    energy : float
        Incident energy in MeV, for the ACE curves and the 0.05 MeV window.
    exfor_files : list of str, optional
        Paths of the EXFOR JSON files. None draws the ACE curves alone.
    m_proj_u, m_targ_u : float
        Projectile and target masses in atomic mass units, for the LAB to CM
        conversion. Defaults: neutron (1.008665 u) and Fe-56 (55.93494 u).
    mt : int
        ENDF MT number of the ACE reactions: 2 for elastic scattering.
    title : str, optional
        Title of the plot. None gives ``"Differential Cross Section at E = ... MeV"``.
    figsize : tuple of int
        Width and height of the figure, in inches.
    log_scale : bool
        If True, the y-axis is logarithmic.
    frame : {'CM', 'LAB'}
        With ``'CM'``, LAB points are converted to CM. With ``'LAB'`` nothing is
        converted and the points of the files are drawn in the frame the files
        record, next to ACE curves that are not converted either. The value is
        also written on the x-axis label.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ACE theoretical data for all provided datasets
    for ace_label, ace_config in ace_dict.items():
        ace_data = ace_config['data']
        color = ace_config.get('color', 'blue')
        linestyle = ace_config.get('linestyle', '-')
        
        try:
            cosines, dsigma_domega, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)
            # Use cosines directly instead of converting to angles
            ax.plot(cosines, dsigma_domega, 
                   label=f'{ace_label} (σ={sigma_total:.4f}b)', 
                   linewidth=2, color=color, linestyle=linestyle)
        except Exception as e:
            print(f"Warning: Could not calculate for {ace_label} at {energy} MeV: {e}")
    
    # Plot experimental data if provided
    if exfor_files:
        # Use a color palette that avoids conflicts with ACE data colors
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(exfor_files), 3)))
        
        for i, exfor_file in enumerate(exfor_files):
            try:
                df_exp, meta_exp = load_exfor_data(exfor_file)
                
                # Filter by energy (within tolerance)
                energy_tolerance = 0.05  # MeV
                df_energy = df_exp[abs(df_exp['energy'] - energy) < energy_tolerance]
                
                if df_energy.empty:
                    continue

                exp_label, year = extract_experiment_info(meta_exp)

                # Get data
                angles = df_energy["angle"].values
                dsig = df_energy["dsig"].values
                err_stat = df_energy["error_stat"].values
                data_frame = meta_exp['angle_frame']
                
                # Convert to requested frame if needed and get cosines for plotting
                if frame.upper() == 'CM' and data_frame.upper() == 'LAB':
                    mu_lab = np.cos(np.deg2rad(angles))
                    mu_cm, dsig_cm = transform_lab_to_cm(mu_lab, dsig, m_proj_u, m_targ_u)
                    mu_plot = mu_cm  # Use cosines directly
                    dsig_plot = dsig_cm
                    alpha = m_proj_u / m_targ_u
                    J = jacobian_cm_to_lab(mu_cm, alpha)
                    err_plot = err_stat / J
                elif frame.upper() == 'LAB' and data_frame.upper() == 'CM':
                    # For LAB frame display when data is in CM, would need inverse transformation
                    # For now, just use as-is and convert angles to cosines
                    mu_plot = np.cos(np.deg2rad(angles))
                    dsig_plot = dsig
                    err_plot = err_stat
                else:
                    # Data is already in correct frame, convert angles to cosines
                    mu_plot = np.cos(np.deg2rad(angles))
                    dsig_plot = dsig
                    err_plot = err_stat
                
                # Plot experimental data using cosines
                ax.errorbar(mu_plot, dsig_plot, yerr=err_plot,
                           fmt='o', color=colors[i], label=exp_label,
                           markersize=4, capsize=3, alpha=0.8)
                           
            except Exception as e:
                print(f"Warning: Could not load {exfor_file}: {e}")
    
    # Format plot
    ax.set_xlabel(f'cos(θ) - Cosine of Scattering Angle ({frame} frame)', fontsize=12)
    ax.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (barns/steradian)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Differential Cross Section at E = {energy} MeV', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-1, 1)  # Cosine range from -1 to 1
    ax.set_xticks(np.arange(-1, 1.1, 0.5))  # Tick marks at -1, -0.5, 0, 0.5, 1
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    return fig, ax
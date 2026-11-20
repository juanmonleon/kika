"""
EXFOR angular distribution class with frame/unit conversion support.

This module provides ExforAngularDistribution, the main class for working
with experimental angular distribution data from EXFOR.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import json

import numpy as np
import pandas as pd

from kika.exfor.exfor_entry import ExforEntry, _normalize_units
from kika.exfor._constants import (
    SCHEMA_VERSION,
    FRAME_LAB,
    FRAME_CM,
    NEUTRON_MASS_AMU,
    ENERGY_TO_MEV,
    XS_TO_B_SR,
    PLOTTING_ENERGY_TOLERANCE,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from kika.plotting import AngularDistributionPlotData, UncertaintyBand


@dataclass
class ExforAngularDistribution(ExforEntry):
    """
    EXFOR differential cross-sections dσ/dΩ against angle, at one or more energies.

    An instance is returned by :func:`~kika.exfor.read_exfor`, by
    :func:`~kika.exfor.read_all_exfor` and by the queries of
    :class:`~kika.exfor.X4ProDatabase`, or built by hand from the fields below.
    It converts between the LAB and CM frames and between units, and plots against
    an evaluation. The data are plain dicts and lists, as in the JSON file.

    Attributes
    ----------
    entry, subentry, quantity, citation, reaction, facility, method, detector, sample
        As in :class:`~kika.exfor.ExforEntry`.
    angle_frame : str
        ``"LAB"`` or ``"CM"``: the frame of every angle and cross-section in the
        object.
    units : dict of str
        Units of the stored values, with the keys ``"energy"`` (``"eV"``,
        ``"keV"`` or ``"MeV"``), ``"angle"`` (``"deg"`` for degrees or ``"cos"``
        for the cosine of the scattering angle) and ``"cross_section"``
        (``"b/sr"``, ``"mb/sr"`` or ``"ub/sr"``). The conversion methods update it.

    Notes
    -----
    Each energy block is a dict with ``value`` (the incident energy, in
    ``units["energy"]``), an optional ``uncertainty`` on that energy and ``data``,
    a list of points. A point has ``angle``, ``cross_section`` and the absolute
    uncertainties ``uncertainty_stat`` and ``uncertainty_sys``, in the units of
    the cross-section. Older files use ``E``, ``result``, ``error_stat`` and
    ``error_sys``; they are read as well.

    Example:
        >>> from kika.exfor import read_exfor
        >>> exfor = read_exfor('/path/to/27673002.json')
        >>> print(exfor.label)
        Gkatis et al. (2025)
        >>> print(exfor.energies())
        [1.0098, 1.0202, ...]
        >>> df = exfor.to_dataframe(energy=1.5)  # Data at E~1.5 MeV
        >>> df = exfor.to_dataframe(energy=(1.0, 2.0))  # Data in range
        >>> df = exfor.to_dataframe(angle=(0, 90))  # Forward angles only
    """

    angle_frame: str = FRAME_LAB
    units: Dict[str, str] = field(
        default_factory=lambda: {"energy": "MeV", "angle": "deg", "cross_section": "b/sr"}
    )
    _data_blocks: List[Dict[str, Any]] = field(default_factory=list)

    # =========================================================================
    # Static Transform Methods
    # =========================================================================

    @staticmethod
    def cos_cm_from_cos_lab(mu_lab: np.ndarray, alpha: float) -> np.ndarray:
        """
        Cosine of the CM scattering angle from that of the LAB angle (forward branch).

        The kinematics are non-relativistic two-body elastic scattering (no energy
        released), with the target at rest in the LAB frame. The angle is that of
        the scattered projectile. It inverts
        ``mu_lab = (mu_cm + alpha) / sqrt(1 + 2*alpha*mu_cm + alpha**2)``, which is
        one-to-one only for ``alpha < 1``. For ``alpha > 1`` the result is NaN
        where the root is undefined, and the forward branch is not the only one.

        Parameters
        ----------
        mu_lab : array_like
            Cosines of the LAB scattering angle, in [-1, 1].
        alpha : float
            Mass ratio ``m_projectile / m_target``, both in the same unit.

        Returns
        -------
        numpy.ndarray
            Cosines of the CM scattering angle, with the shape of *mu_lab*.
        """
        mu_lab = np.asarray(mu_lab)
        return -alpha * (1 - mu_lab**2) + mu_lab * np.sqrt(1 - alpha**2 * (1 - mu_lab**2))

    @staticmethod
    def cos_lab_from_cos_cm(mu_cm: np.ndarray, alpha: float) -> np.ndarray:
        """
        Cosine of the LAB scattering angle from the cosine of the CM angle.

        The kinematics are non-relativistic two-body elastic scattering (no energy
        released), with the target at rest in the LAB frame. The angle is that of
        the scattered projectile:
        ``mu_lab = (mu_cm + alpha) / sqrt(1 + 2*alpha*mu_cm + alpha**2)``.

        Parameters
        ----------
        mu_cm : array_like
            Cosines of the CM scattering angle, in [-1, 1].
        alpha : float
            Mass ratio ``m_projectile / m_target``, both in the same unit.

        Returns
        -------
        numpy.ndarray
            Cosines of the LAB scattering angle, with the shape of *mu_cm*.
        """
        mu_cm = np.asarray(mu_cm)
        denominator = np.sqrt(1 + 2 * alpha * mu_cm + alpha**2)
        return (mu_cm + alpha) / denominator

    @staticmethod
    def jacobian_cm_to_lab(mu_cm: np.ndarray, alpha: float) -> np.ndarray:
        """
        Solid-angle ratio ``dOmega_CM / dOmega_LAB`` at a given CM cosine.

        The value is ``(1 + alpha**2 + 2*alpha*mu_cm)**1.5 / abs(1 + alpha*mu_cm)``,
        for non-relativistic two-body elastic scattering with the target at rest.
        It is the factor between the differential cross-sections of the two
        frames at the same scattering event:
        ``(dsigma/dOmega)_LAB = (dsigma/dOmega)_CM * J`` and
        ``(dsigma/dOmega)_CM = (dsigma/dOmega)_LAB / J``. It diverges at
        ``alpha * mu_cm = -1``, which can happen only for ``alpha >= 1``.

        Parameters
        ----------
        mu_cm : array_like
            Cosines of the CM scattering angle.
        alpha : float
            Mass ratio ``m_projectile / m_target``, both in the same unit.

        Returns
        -------
        numpy.ndarray
            Dimensionless ``J``, with the shape of *mu_cm*.
        """
        mu_cm = np.asarray(mu_cm)
        return (1 + alpha**2 + 2 * alpha * mu_cm) ** 1.5 / np.abs(1 + alpha * mu_cm)

    @staticmethod
    def jacobian_lab_to_cm(mu_lab: np.ndarray, alpha: float) -> np.ndarray:
        """
        Solid-angle ratio ``dOmega_LAB / dOmega_CM`` at a given LAB cosine.

        It is ``1 / jacobian_cm_to_lab`` evaluated at the CM cosine that
        :meth:`cos_cm_from_cos_lab` gives for *mu_lab*, with the same kinematics
        (non-relativistic two-body elastic scattering, target at rest, forward
        branch, ``alpha < 1``). With ``K`` the returned value,
        ``(dsigma/dOmega)_CM = (dsigma/dOmega)_LAB * K`` and
        ``(dsigma/dOmega)_LAB = (dsigma/dOmega)_CM / K``.

        Parameters
        ----------
        mu_lab : array_like
            Cosines of the LAB scattering angle.
        alpha : float
            Mass ratio ``m_projectile / m_target``, both in the same unit.

        Returns
        -------
        numpy.ndarray
            Dimensionless ``K``, with the shape of *mu_lab*.
        """
        mu_cm = ExforAngularDistribution.cos_cm_from_cos_lab(mu_lab, alpha)
        return 1.0 / ExforAngularDistribution.jacobian_cm_to_lab(mu_cm, alpha)

    @staticmethod
    def transform_lab_to_cm(
        mu_lab: np.ndarray,
        dsdo_lab: np.ndarray,
        err_lab: np.ndarray,
        m_proj: float,
        m_targ: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform cosines, dσ/dΩ and uncertainties from the LAB to the CM frame.

        The kinematics are non-relativistic two-body elastic scattering with the
        target at rest, and ``alpha = m_proj / m_targ``. The cosines go through
        :meth:`cos_cm_from_cos_lab`. The cross-sections and the uncertainties are
        divided by ``jacobian_cm_to_lab`` (``dOmega_CM / dOmega_LAB``) at the CM
        cosine. The uncertainties are taken as absolute, in the unit of the
        cross-section, and scale like it. The energy is not changed.

        Parameters
        ----------
        mu_lab : array_like
            Cosines of the LAB scattering angle.
        dsdo_lab : array_like
            LAB differential cross-sections per steradian, in any unit.
        err_lab : array_like
            Absolute uncertainties of *dsdo_lab*, in the same unit.
        m_proj, m_targ : float
            Projectile and target masses in atomic mass units. Only their ratio
            is used, and it must be below 1.

        Returns
        -------
        mu_cm : numpy.ndarray
            Cosines of the CM scattering angle.
        dsdo_cm : numpy.ndarray
            CM differential cross-sections per steradian, in the unit of *dsdo_lab*.
        err_cm : numpy.ndarray
            Absolute uncertainties of *dsdo_cm*, in the same unit.
        """
        mu_lab = np.asarray(mu_lab)
        dsdo_lab = np.asarray(dsdo_lab)
        err_lab = np.asarray(err_lab)

        alpha = m_proj / m_targ

        # Convert angles
        mu_cm = ExforAngularDistribution.cos_cm_from_cos_lab(mu_lab, alpha)

        # Calculate Jacobian and transform cross section
        J = ExforAngularDistribution.jacobian_cm_to_lab(mu_cm, alpha)  # dOmega_CM/dOmega_LAB
        dsdo_cm = dsdo_lab / J
        err_cm = err_lab / J

        return mu_cm, dsdo_cm, err_cm

    @staticmethod
    def transform_cm_to_lab(
        mu_cm: np.ndarray,
        dsdo_cm: np.ndarray,
        err_cm: np.ndarray,
        m_proj: float,
        m_targ: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform cosines, dσ/dΩ and uncertainties from the CM to the LAB frame.

        The kinematics are non-relativistic two-body elastic scattering with the
        target at rest, and ``alpha = m_proj / m_targ``. The cosines go through
        :meth:`cos_lab_from_cos_cm`. The cross-sections and the uncertainties are
        multiplied by ``jacobian_cm_to_lab`` (``dOmega_CM / dOmega_LAB``) at the CM
        cosine. The uncertainties are taken as absolute, in the unit of the
        cross-section, and scale like it. The energy is not changed.

        Parameters
        ----------
        mu_cm : array_like
            Cosines of the CM scattering angle.
        dsdo_cm : array_like
            CM differential cross-sections per steradian, in any unit.
        err_cm : array_like
            Absolute uncertainties of *dsdo_cm*, in the same unit.
        m_proj, m_targ : float
            Projectile and target masses in atomic mass units. Only their ratio
            is used.

        Returns
        -------
        mu_lab : numpy.ndarray
            Cosines of the LAB scattering angle.
        dsdo_lab : numpy.ndarray
            LAB differential cross-sections per steradian, in the unit of *dsdo_cm*.
        err_lab : numpy.ndarray
            Absolute uncertainties of *dsdo_lab*, in the same unit.
        """
        mu_cm = np.asarray(mu_cm)
        dsdo_cm = np.asarray(dsdo_cm)
        err_cm = np.asarray(err_cm)

        alpha = m_proj / m_targ

        # Convert angles
        mu_lab = ExforAngularDistribution.cos_lab_from_cos_cm(mu_cm, alpha)

        # Calculate Jacobian and transform cross section
        J = ExforAngularDistribution.jacobian_cm_to_lab(mu_cm, alpha)  # dOmega_CM/dOmega_LAB
        dsdo_lab = dsdo_cm * J
        err_lab = err_cm * J

        return mu_lab, dsdo_lab, err_lab

    @staticmethod
    def angle_deg_to_cos(angles_deg: np.ndarray) -> np.ndarray:
        """
        Convert scattering angles in degrees to cosines.

        Parameters
        ----------
        angles_deg : np.ndarray
            Scattering angles in degrees

        Returns
        -------
        np.ndarray
            Cosines of scattering angles (mu = cos(theta))
        """
        return np.cos(np.radians(angles_deg))

    @staticmethod
    def cos_to_angle_deg(cosines: np.ndarray) -> np.ndarray:
        """
        Convert cosines of scattering angles to degrees.

        Parameters
        ----------
        cosines : np.ndarray
            Cosines of scattering angles (mu = cos(theta))

        Returns
        -------
        np.ndarray
            Scattering angles in degrees
        """
        return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))

    # =========================================================================
    # Data Access Methods
    # =========================================================================

    def energies(
        self,
        unit: str = "MeV",
        bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Get available energy values.

        Parameters
        ----------
        unit : str
            Output unit: 'eV', 'keV', or 'MeV' (default: 'MeV')
        bounds : Tuple[float, float], optional
            Filter to [min, max] bounds (in requested units)

        Returns
        -------
        np.ndarray
            Sorted array of unique energy values
        """
        # Get raw energies in internal units
        raw_energies = [block.get("value") or block.get("E", 0.0) for block in self._data_blocks]

        # Convert to requested units
        factor = ENERGY_TO_MEV.get(self.units["energy"], 1.0) / ENERGY_TO_MEV.get(unit, 1.0)
        energies = np.array([e * factor for e in raw_energies])

        # Apply bounds filter
        if bounds is not None:
            mask = (energies >= bounds[0]) & (energies <= bounds[1])
            energies = energies[mask]

        return np.sort(np.unique(energies))

    def angles(
        self,
        unit: str = "deg",
        energy: Optional[float] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Get available angle values.

        Parameters
        ----------
        unit : str
            Output unit: 'deg' or 'cos' (default: 'deg')
        energy : float or Tuple[float, float], optional
            Filter by energy (float with 5% tolerance, or (min, max) tuple) in MeV
        bounds : Tuple[float, float], optional
            Filter to [min, max] angle bounds (in requested units)

        Returns
        -------
        np.ndarray
            Sorted array of unique angle values
        """
        all_angles = []

        # Energy filtering setup
        if energy is not None:
            if isinstance(energy, tuple):
                e_min, e_max = energy
            else:
                e_min = energy * 0.95
                e_max = energy * 1.05

            # Convert bounds to internal units
            factor_to_internal = ENERGY_TO_MEV.get("MeV", 1.0) / ENERGY_TO_MEV.get(self.units["energy"], 1.0)
            e_min_internal = e_min * factor_to_internal
            e_max_internal = e_max * factor_to_internal

        for block in self._data_blocks:
            block_energy = block.get("value") or block.get("E", 0.0)

            # Apply energy filter if specified
            if energy is not None:
                if block_energy < e_min_internal or block_energy > e_max_internal:
                    continue

            # Extract angles from data points
            for point in block.get("data", []):
                angle_raw = point.get("angle", 0.0)

                # Convert to requested units
                if self.units["angle"] == "deg" and unit == "cos":
                    angle_out = float(np.cos(np.radians(angle_raw)))
                elif self.units["angle"] == "cos" and unit == "deg":
                    angle_out = float(np.degrees(np.arccos(np.clip(angle_raw, -1.0, 1.0))))
                else:
                    angle_out = angle_raw

                all_angles.append(angle_out)

        angles = np.array(all_angles)

        # Apply bounds filter
        if bounds is not None:
            mask = (angles >= bounds[0]) & (angles <= bounds[1])
            angles = angles[mask]

        return np.sort(np.unique(angles))

    def to_dataframe(
        self,
        energy: Optional[float] = None,
        angle: Optional[float] = None,
        tolerance: float = 0.05,
        energy_unit: str = "MeV",
        cross_section_unit: str = "b/sr",
        angle_unit: str = "deg",
        decompose_uncertainty: bool = False,
    ) -> pd.DataFrame:
        """
        Convert EXFOR data to DataFrame with optional filtering.

        Parameters
        ----------
        energy : float or Tuple[float, float], optional
            - float: Select data at specific energy (with tolerance)
            - tuple: Select data in energy range [min, max]
            - None: Include all energies
        angle : float or Tuple[float, float], optional
            - float: Select data at specific angle (with tolerance)
            - tuple: Select data in angle range [min, max]
            - None: Include all angles
        tolerance : float
            Absolute tolerance in MeV for energy matching (default: 0.05 MeV)
        energy_unit : str
            Output energy unit (default: 'MeV')
        cross_section_unit : str
            Output cross-section unit (default: 'b/sr')
        angle_unit : str
            Output angle unit (default: 'deg')
        decompose_uncertainty : bool, default False
            If True, return ``error_stat`` and ``error_sys`` columns separately
            (per-point uncorrelated noise vs per-point correlated systematic)
            instead of the combined ``error`` column.

        Returns
        -------
        pd.DataFrame
            Columns: [energy, angle, value, error] (default) or
            [energy, angle, value, error_stat, error_sys] when ``decompose_uncertainty=True``.

        Examples
        --------
        >>> df = exfor.to_dataframe()  # All data
        >>> df = exfor.to_dataframe(energy=1.5)  # Single energy (with 5% tolerance)
        >>> df = exfor.to_dataframe(energy=(1.0, 2.0))  # Energy range
        >>> df = exfor.to_dataframe(angle=(0, 90))  # Forward angles
        >>> df = exfor.to_dataframe(energy=1.5, angle=(0, 90))  # Combined filters
        """
        # Unit conversion factors
        # From internal units to MeV/b_sr/deg, then to requested units
        energy_factor = ENERGY_TO_MEV.get(self.units["energy"], 1.0) / ENERGY_TO_MEV.get(energy_unit, 1.0)
        xs_factor = XS_TO_B_SR.get(self.units["cross_section"], 1.0) / XS_TO_B_SR.get(cross_section_unit, 1.0)

        # Energy filtering: convert requested energy to internal units for comparison
        internal_energy_factor = ENERGY_TO_MEV.get(energy_unit, 1.0) / ENERGY_TO_MEV.get(self.units["energy"], 1.0)

        rows = []

        for block in self._data_blocks:
            block_energy = block.get("value") or block.get("E", 0.0)

            # Apply energy filter (in internal units)
            if energy is not None:
                if isinstance(energy, tuple):
                    # Range filter
                    min_e_internal = energy[0] * internal_energy_factor
                    max_e_internal = energy[1] * internal_energy_factor
                    if block_energy < min_e_internal or block_energy > max_e_internal:
                        continue
                else:
                    # Single value with absolute tolerance (tolerance is in MeV)
                    target_energy_internal = energy * internal_energy_factor
                    tolerance_internal = tolerance * internal_energy_factor  # Convert tolerance to internal units
                    if abs(block_energy - target_energy_internal) > tolerance_internal:
                        continue

            # Convert energy to output units
            energy_out = block_energy * energy_factor

            data_points = block.get("data", [])
            for point in data_points:
                # Get raw values
                angle_raw = point.get("angle", 0.0)
                xs_raw = point.get("cross_section") or point.get("result", 0.0)
                error_stat = point.get("uncertainty_stat") or point.get("error_stat", 0.0) or 0.0
                error_sys = point.get("uncertainty_sys") or point.get("error_sys", 0.0) or 0.0

                # Combine errors: sqrt(stat^2 + sys^2)
                total_error = np.sqrt(error_stat**2 + error_sys**2)

                # Handle angle unit conversion
                # Internal units are stored in self.units["angle"]
                if self.units["angle"] == "deg" and angle_unit == "cos":
                    angle_out = float(np.cos(np.radians(angle_raw)))
                elif self.units["angle"] == "cos" and angle_unit == "deg":
                    angle_out = float(np.degrees(np.arccos(np.clip(angle_raw, -1.0, 1.0))))
                else:
                    angle_out = angle_raw

                # Apply angle filter (in output units)
                if angle is not None:
                    if isinstance(angle, tuple):
                        # Range filter
                        if angle_out < angle[0] or angle_out > angle[1]:
                            continue
                    else:
                        # Single value with tolerance
                        if abs(angle_out - angle) > tolerance * max(abs(angle), 1e-10):
                            continue

                # Convert cross section and error
                xs_out = xs_raw * xs_factor
                error_out = total_error * xs_factor

                if decompose_uncertainty:
                    rows.append({
                        "energy": energy_out,
                        "angle": angle_out,
                        "value": xs_out,
                        "error_stat": error_stat * xs_factor,
                        "error_sys":  error_sys * xs_factor,
                    })
                else:
                    rows.append({
                        "energy": energy_out,
                        "angle": angle_out,
                        "value": xs_out,
                        "error": error_out,
                    })

        cols = ["energy", "angle", "value", "error_stat", "error_sys"] if decompose_uncertainty \
               else ["energy", "angle", "value", "error"]
        return pd.DataFrame(rows, columns=cols)

    # =========================================================================
    # Frame Conversion Methods
    # =========================================================================

    def convert_to_cm(
        self, m_proj: float = None, inplace: bool = False
    ) -> Optional["ExforAngularDistribution"]:
        """
        Convert every angle and cross-section from the LAB frame to the CM frame.

        Uses :meth:`transform_lab_to_cm`: non-relativistic two-body elastic
        scattering, ``alpha = m_proj / m_targ``, the target mass being
        :attr:`~kika.exfor.ExforEntry.target_mass`. Angles go to the CM angle
        in the unit they have (degrees, or cosines if ``units["angle"]`` is
        not ``"deg"``). Cross-sections and ``uncertainty_stat`` are divided by
        ``dOmega_CM / dOmega_LAB``. ``uncertainty_sys`` is not converted and keeps
        its LAB value. Energies and units are unchanged, and the points keep their
        order. The projectile must be lighter than the target.

        Parameters
        ----------
        m_proj : float, optional
            Projectile mass in atomic mass units. None is the neutron mass.
        inplace : bool
            If True, modify this object and return None.

        Returns
        -------
        ExforAngularDistribution or None
            A converted copy, or None when *inplace* is True. An object already in
            the CM frame is returned as an unchanged copy (or left as it is).
        """
        if self.angle_frame == FRAME_CM:
            if inplace:
                return None
            return deepcopy(self)

        if m_proj is None:
            m_proj = NEUTRON_MASS_AMU

        m_targ = self.target_mass
        target_obj = self if inplace else deepcopy(self)

        for block in target_obj._data_blocks:
            data_points = block.get("data", [])
            if not data_points:
                continue

            # Extract arrays
            angles_arr = np.array([p.get("angle", 0.0) for p in data_points])
            xs_arr = np.array([p.get("cross_section") or p.get("result", 0.0) for p in data_points])
            err_arr = np.array([p.get("uncertainty_stat") or p.get("error_stat", 0.0) for p in data_points])

            # Convert angles to cosines if in degrees
            if target_obj.units["angle"] == "deg":
                mu_lab = np.cos(np.radians(angles_arr))
            else:
                mu_lab = angles_arr

            # Transform to CM using static method
            mu_cm, xs_cm, err_cm = ExforAngularDistribution.transform_lab_to_cm(
                mu_lab, xs_arr, err_arr, m_proj, m_targ
            )

            # Convert back to degrees if needed
            if target_obj.units["angle"] == "deg":
                angles_cm = np.degrees(np.arccos(np.clip(mu_cm, -1.0, 1.0)))
            else:
                angles_cm = mu_cm

            # Update data points
            for i, point in enumerate(data_points):
                point["angle"] = float(angles_cm[i])
                point["cross_section"] = float(xs_cm[i])
                point["uncertainty_stat"] = float(err_cm[i])

        target_obj.angle_frame = FRAME_CM

        if inplace:
            return None
        return target_obj

    def convert_to_lab(
        self, m_proj: float = None, inplace: bool = False
    ) -> Optional["ExforAngularDistribution"]:
        """
        Convert every angle and cross-section from the CM frame to the LAB frame.

        Uses :meth:`transform_cm_to_lab`: non-relativistic two-body elastic
        scattering, ``alpha = m_proj / m_targ``, the target mass being
        :attr:`~kika.exfor.ExforEntry.target_mass`. Angles go to the LAB angle
        in the unit they have (degrees, or cosines if ``units["angle"]`` is
        not ``"deg"``). Cross-sections and ``uncertainty_stat`` are multiplied by
        ``dOmega_CM / dOmega_LAB``. ``uncertainty_sys`` is not converted and keeps
        its CM value. Energies and units are unchanged, and the points keep their
        order.

        Parameters
        ----------
        m_proj : float, optional
            Projectile mass in atomic mass units. None is the neutron mass.
        inplace : bool
            If True, modify this object and return None.

        Returns
        -------
        ExforAngularDistribution or None
            A converted copy, or None when *inplace* is True. An object already in
            the LAB frame is returned as an unchanged copy (or left as it is).
        """
        if self.angle_frame == FRAME_LAB:
            if inplace:
                return None
            return deepcopy(self)

        if m_proj is None:
            m_proj = NEUTRON_MASS_AMU

        m_targ = self.target_mass
        target_obj = self if inplace else deepcopy(self)

        for block in target_obj._data_blocks:
            data_points = block.get("data", [])
            if not data_points:
                continue

            # Extract arrays
            angles_arr = np.array([p.get("angle", 0.0) for p in data_points])
            xs_arr = np.array([p.get("cross_section") or p.get("result", 0.0) for p in data_points])
            err_arr = np.array([p.get("uncertainty_stat") or p.get("error_stat", 0.0) for p in data_points])

            # Convert angles to cosines if in degrees
            if target_obj.units["angle"] == "deg":
                mu_cm = np.cos(np.radians(angles_arr))
            else:
                mu_cm = angles_arr

            # Transform to LAB using static method
            mu_lab, xs_lab, err_lab = ExforAngularDistribution.transform_cm_to_lab(
                mu_cm, xs_arr, err_arr, m_proj, m_targ
            )

            # Convert back to degrees if needed
            if target_obj.units["angle"] == "deg":
                angles_lab = np.degrees(np.arccos(np.clip(mu_lab, -1.0, 1.0)))
            else:
                angles_lab = mu_lab

            # Update data points
            for i, point in enumerate(data_points):
                point["angle"] = float(angles_lab[i])
                point["cross_section"] = float(xs_lab[i])
                point["uncertainty_stat"] = float(err_lab[i])

        target_obj.angle_frame = FRAME_LAB

        if inplace:
            return None
        return target_obj

    # =========================================================================
    # Unit Conversion Methods
    # =========================================================================

    def convert_energy(
        self, to: str, inplace: bool = False
    ) -> Optional["ExforAngularDistribution"]:
        """
        Express the incident energies, and their uncertainties, in another unit.

        Angles, cross-sections and the frame are not changed.

        Parameters
        ----------
        to : {"eV", "keV", "MeV"}
            Target energy unit. It is stored in ``units["energy"]``.
        inplace : bool
            If True, modify this object and return None.

        Returns
        -------
        ExforAngularDistribution or None
            A converted copy, or None when *inplace* is True. When the unit is
            already *to*, an unchanged copy (or the object left as it is).

        Raises
        ------
        KeyError
            If *to*, or the current energy unit, is not one of the three above.
        """
        from_unit = self.units["energy"]
        if from_unit == to:
            if inplace:
                return None
            return deepcopy(self)

        factor = ENERGY_TO_MEV[from_unit] / ENERGY_TO_MEV[to]
        target_obj = self if inplace else deepcopy(self)

        for block in target_obj._data_blocks:
            # Handle both old (E) and new (value) formats
            if "value" in block:
                block["value"] *= factor
            elif "E" in block:
                block["E"] *= factor
            if block.get("uncertainty") is not None:
                block["uncertainty"] *= factor

        target_obj.units["energy"] = to

        if inplace:
            return None
        return target_obj

    def convert_cross_section(
        self, to: str, inplace: bool = False
    ) -> Optional["ExforAngularDistribution"]:
        """
        Express the cross-sections and their uncertainties in another unit.

        Every point's cross-section, ``uncertainty_stat`` and ``uncertainty_sys``
        are multiplied by the same factor. Angles, energies and the frame are not
        changed.

        Parameters
        ----------
        to : {"b/sr", "mb/sr", "ub/sr"}
            Target unit, per steradian. It is stored in ``units["cross_section"]``.
        inplace : bool
            If True, modify this object and return None.

        Returns
        -------
        ExforAngularDistribution or None
            A converted copy, or None when *inplace* is True. When the unit is
            already *to*, an unchanged copy (or the object left as it is).

        Raises
        ------
        KeyError
            If *to*, or the current cross-section unit, is not one of the three
            above.
        """
        from_unit = self.units["cross_section"]
        if from_unit == to:
            if inplace:
                return None
            return deepcopy(self)

        factor = XS_TO_B_SR[from_unit] / XS_TO_B_SR[to]
        target_obj = self if inplace else deepcopy(self)

        for block in target_obj._data_blocks:
            for point in block.get("data", []):
                if "cross_section" in point:
                    point["cross_section"] *= factor
                elif "result" in point:
                    point["result"] *= factor
                if "uncertainty_stat" in point:
                    point["uncertainty_stat"] *= factor
                elif "error_stat" in point:
                    point["error_stat"] *= factor
                if point.get("uncertainty_sys") is not None:
                    point["uncertainty_sys"] *= factor
                elif point.get("error_sys") is not None:
                    point["error_sys"] *= factor

        target_obj.units["cross_section"] = to

        if inplace:
            return None
        return target_obj

    def convert_angle(
        self, to: str, inplace: bool = False
    ) -> Optional["ExforAngularDistribution"]:
        """
        Express the scattering angles in degrees or as cosines.

        The frame is not changed (use :meth:`convert_to_cm` or
        :meth:`convert_to_lab` for that), nor are the cross-sections. A cosine
        becomes ``degrees(arccos(mu))`` with *mu* clipped to [-1, 1].

        Parameters
        ----------
        to : {"deg", "cos"}
            Target unit, stored in ``units["angle"]``. Any other string is stored
            as the unit and leaves the values as they were.
        inplace : bool
            If True, modify this object and return None.

        Returns
        -------
        ExforAngularDistribution or None
            A converted copy, or None when *inplace* is True. When the unit is
            already *to*, an unchanged copy (or the object left as it is).
        """
        from_unit = self.units["angle"]
        if from_unit == to:
            if inplace:
                return None
            return deepcopy(self)

        target_obj = self if inplace else deepcopy(self)

        for block in target_obj._data_blocks:
            for point in block.get("data", []):
                angle = point.get("angle", 0.0)
                if from_unit == "deg" and to == "cos":
                    point["angle"] = float(np.cos(np.radians(angle)))
                elif from_unit == "cos" and to == "deg":
                    point["angle"] = float(np.degrees(np.arccos(np.clip(angle, -1.0, 1.0))))

        target_obj.units["angle"] = to

        if inplace:
            return None
        return target_obj

    # =========================================================================
    # Energy Resolution Methods
    # =========================================================================

    def compute_energy_resolution(self, energy_mev: float) -> Optional[float]:
        """
        Time-of-flight energy resolution ``sigma_E`` of the beam at one energy, in MeV.

        The flight path ``L`` and the time resolution ``sigma_t`` come from
        ``method["energy_resolution_input"]`` (or ``"energy_resolution_inputs"``):
        ``distance.value`` is read as metres and ``time_resolution.value`` as
        nanoseconds, whatever their ``unit`` entries say. The neutron speed is
        non-relativistic, ``t = L / v`` is the time of flight, and
        ``sigma_E = 2 * E * sigma_t / t``. The time resolution is used as given, so
        ``sigma_E`` is a standard deviation or a full width as ``sigma_t`` is.
        Objects read from the X4Pro database always carry a geometry: the curated
        one of the experiment, or a default flight path of 27.037 m and a time
        resolution of 10 ns.

        Parameters
        ----------
        energy_mev : float
            Incident neutron energy in MeV, positive.

        Returns
        -------
        float or None
            ``sigma_E`` in MeV. None if the object has no flight path or no time
            resolution.
        """
        eri = self.method.get("energy_resolution_input") or self.method.get("energy_resolution_inputs")
        if eri is None:
            return None

        distance_info = eri.get("distance")
        time_info = eri.get("time_resolution")

        if distance_info is None or time_info is None:
            return None

        L = distance_info.get("value")  # Flight path in meters
        sigma_t = time_info.get("value")  # Time resolution in ns

        if L is None or sigma_t is None:
            return None

        from kika._constants import NEUTRON_MASS_MEV as m_n, SPEED_OF_LIGHT_M_NS as c

        # Convert energy to velocity (non-relativistic)
        v_over_c = np.sqrt(2 * energy_mev / m_n)
        v = v_over_c * c  # v in m/ns

        # Time of flight
        t = L / v  # ns

        # Energy resolution: sigma_E/E = 2 * sigma_t / t
        sigma_E_over_E = 2 * sigma_t / t
        sigma_E = sigma_E_over_E * energy_mev

        return sigma_E

    # =========================================================================
    # Representation Methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return a nicely formatted representation for notebooks."""
        energies = self.energies()
        n_energies = len(energies)
        n_points = sum(len(block.get("data", [])) for block in self._data_blocks)

        e_min = f"{energies.min():.4g}" if n_energies > 0 else "N/A"
        e_max = f"{energies.max():.4g}" if n_energies > 0 else "N/A"

        lines = [
            "ExforAngularDistribution",
            f"  Entry:       {self.entry}{self.subentry}",
            f"  Label:       {self.label}",
            f"  Target:      {self.target} (ZAID: {self.zaid})",
            f"  Reaction:    {self.reaction.get('notation', 'N/A')}",
            f"  Frame:       {self.angle_frame}",
            f"  Energies:    {n_energies} ({e_min} - {e_max} {self.units['energy']})",
            f"  Data points: {n_points}",
        ]
        return "\n".join(lines)

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = super().to_dict()
        result["angle_frame"] = self.angle_frame
        result["units"] = self.units
        result["energies"] = self._data_blocks  # Output as "energies" for JSON compatibility
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExforAngularDistribution":
        """
        Build a distribution from a dict with the layout of the JSON files.

        The keys are those of :meth:`to_dict`. A missing ``angle_frame`` is
        ``"LAB"`` and it is upper-cased. The unit strings are normalised to the
        spellings of the ``units`` attribute (``"ADEG"`` to ``"deg"``, ``"MEV"`` to
        ``"MeV"``, ``"MUB/SR"`` to ``"ub/sr"``), a missing entry meaning MeV,
        degrees and b/sr. The list under
        ``energies`` becomes the energy blocks without being copied. ``reaction``
        may be a dict or an EXFOR string such as ``"26-FE-56(N,EL)26-FE-56"``.

        Parameters
        ----------
        data : dict
            The values, as in a JSON file.

        Returns
        -------
        ExforAngularDistribution
            The distribution, in the frame and units *data* declares. Nothing is
            converted.
        """
        base_fields = cls._parse_base_fields(data)

        # Parse angle frame
        angle_frame = data.get("angle_frame", FRAME_LAB).upper()

        # Parse and normalize units
        units_data = data.get("units", {})
        units = _normalize_units(units_data)

        # Parse energy blocks (stored as "energies" in JSON)
        data_blocks = data.get("energies", [])

        return cls(
            **base_fields,
            angle_frame=angle_frame,
            units=units,
            _data_blocks=data_blocks,
        )

    @classmethod
    def from_json(cls, filepath: str) -> "ExforAngularDistribution":
        """
        Read a distribution from a JSON file, through :meth:`from_dict`.

        Unlike :func:`~kika.exfor.read_exfor`, it does not set the
        systematic-uncertainty attributes.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        ExforAngularDistribution
            The distribution, in the frame and units of the file.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file is not valid JSON.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot(self, energy: float, ax: "plt.Axes" = None, **kwargs) -> "plt.Axes":
        """
        Plot dσ/dΩ in b/sr against the scattering angle at one incident energy.

        This calls :func:`~kika.exfor.plot_exfor_angular`. The data stay in the
        frame of the object (``angle_frame``, written on the x-axis) and are not
        converted. The x-axis is the cosine and the y-axis is logarithmic, unless
        *kwargs* says otherwise. A warning is printed and the axes are returned
        empty when no energy matches.

        Parameters
        ----------
        energy : float
            Incident energy in MeV, matched to the energy blocks with the
            tolerance of :func:`~kika.exfor.plot_exfor_angular`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. None creates a new figure.
        **kwargs
            The other arguments of :func:`~kika.exfor.plot_exfor_angular`
            (``tolerance``, ``use_mev``, ``label``, ``color``, ``marker``,
            ``use_cos``, ``use_log``). Anything else goes to
            ``matplotlib.axes.Axes.errorbar``. With ``use_mev=False``, *energy* is
            in the unit of ``units["energy"]``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.
        """
        from kika.exfor.plotting import plot_exfor_angular

        return plot_exfor_angular(self, energy, ax=ax, **kwargs)

    def compare_with_ace(
        self, ace_data, energy: float, mt: int = 2, ax: "plt.Axes" = None, **kwargs
    ) -> "plt.Axes":
        """
        Plot the data over the ACE angular distribution of the same reaction.

        This calls :func:`~kika.exfor.plot_exfor_ace_comparison`. Both curves are
        dσ/dΩ in b/sr. By default LAB data are converted to the CM frame first,
        because the ACE curve is drawn as stored, with
        ``dσ/dΩ = σ(E) f(mu) / (2π)``, and is not converted. The ACE curve is
        evaluated at *energy*, not at the EXFOR energy that matched it.

        Parameters
        ----------
        ace_data
            ACE object whose angular distribution and cross-section of reaction
            *mt* are read.
        energy : float
            Incident energy in MeV, for the EXFOR data and the ACE curve.
        mt : int
            ENDF MT number of the ACE reaction (2 for elastic scattering).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. None creates a new figure.
        **kwargs
            The other arguments of :func:`~kika.exfor.plot_exfor_ace_comparison`:
            ``convert_to_cm`` (False plots the data in their own frame), ``m_proj``
            (projectile mass in atomic mass units, default neutron), ``tolerance``,
            ``use_cos``, ``use_log``, ``exfor_kwargs`` and ``ace_kwargs``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.
        """
        from kika.exfor.plotting import plot_exfor_ace_comparison

        return plot_exfor_ace_comparison(self, ace_data, energy, mt=mt, ax=ax, **kwargs)

    def _to_plot_data_single_energy(
        self,
        energy: float,
        frame: Optional[str],
        angle_unit: str,
        cross_section_unit: str,
        uncertainty: bool,
        connect_points: bool,
        label: Optional[str],
        include_natural_tag: bool = True,
        **styling_kwargs
    ) -> Union["AngularDistributionPlotData", Tuple["AngularDistributionPlotData", "UncertaintyBand"]]:
        """
        Extract plot data for a single energy.

        This is a helper method called by to_plot_data() for each energy value.

        Parameters
        ----------
        energy : float
            Energy value in MeV
        frame : str or None
            Output frame ('LAB', 'CM', or None for current)
        angle_unit : str
            Angle unit ('cos' or 'deg')
        cross_section_unit : str
            Cross section unit ('b/sr' or 'mb/sr')
        uncertainty : bool
            Whether to return uncertainty band
        connect_points : bool
            Whether to connect data points with lines
        label : str or None
            Custom label (auto-generated if None)
        include_natural_tag : bool, default True
            If True and target is natural, append '[nat]' to auto-generated label
        **styling_kwargs
            Additional styling parameters

        Returns
        -------
        AngularDistributionPlotData or tuple
            PlotData, or (PlotData, UncertaintyBand) if uncertainty=True
        """
        # Work on a copy if frame conversion is needed
        if frame is not None and frame.upper() != self.angle_frame:
            if frame.upper() == FRAME_CM:
                data_obj = self.convert_to_cm(inplace=False)
            elif frame.upper() == FRAME_LAB:
                data_obj = self.convert_to_lab(inplace=False)
            else:
                raise ValueError(f"Invalid frame '{frame}'. Must be 'LAB', 'CM', or None.")
            output_frame = frame.upper()
        else:
            data_obj = self
            output_frame = self.angle_frame

        # Extract data using to_dataframe
        # Use a very small tolerance here since 'energy' is already the exact value
        # from the dataset (selected by to_plot_data). We just need to match it precisely.
        df = data_obj.to_dataframe(
            energy=energy,
            tolerance=1e-9,  # Essentially exact match - energy is already selected
            energy_unit="MeV",
            cross_section_unit=cross_section_unit,
            angle_unit=angle_unit,
        )

        if df.empty:
            raise ValueError(
                f"No data found at energy {energy} MeV. "
                f"Available energies: {self.energies(unit='MeV')}"
            )

        # Get the actual energy value (may differ slightly due to tolerance)
        actual_energy = df["energy"].iloc[0]

        # Sort by angle for proper plotting
        df = df.sort_values("angle")

        # Extract arrays
        angles = df["angle"].values
        cross_sections = df["value"].values
        errors = df["error"].values

        # Generate label if not provided
        if label is None:
            base_label = self.label  # "Author et al. (Year)"
            # Add [nat] suffix if target is natural element
            if include_natural_tag and self.is_natural_target:
                label = f"{base_label} [nat]"
            else:
                label = base_label

        # Determine MT from process
        mt = None
        if self.process:
            process_upper = self.process.upper()
            if process_upper == "EL":
                mt = 2
            elif process_upper == "INL":
                mt = 4
            # Could add more mappings here

        # Build metadata
        metadata = {
            "exfor_entry": self.entry,
            "exfor_subentry": self.subentry,
            "target": self.target,
            "zaid": self.zaid,
            "process": self.process,
            "frame": output_frame,
            "source": "EXFOR",
        }

        # For experimental data with errors, use errorbar plot type
        # For experimental data without errors (or uncertainty=False), use scatter
        has_errors = np.any(errors > 0)

        # Get styling options
        marker = styling_kwargs.pop("marker", "o")
        markersize = styling_kwargs.pop("markersize", 5)
        linestyle = styling_kwargs.pop("linestyle", None)
        capsize = styling_kwargs.pop("capsize", 2)

        # Determine linestyle based on connect_points
        if linestyle is None:
            linestyle = "-" if connect_points else "none"

        if uncertainty and has_errors:
            # Use errorbar plot type for experimental data with uncertainties
            plot_data = AngularDistributionPlotData(
                x=angles,
                y=cross_sections,
                energy=actual_energy,
                isotope=self.target,
                mt=mt,
                distribution_type="experimental",
                label=label,
                plot_type="errorbar",
                marker=marker,
                markersize=markersize,
                linestyle=linestyle,
                **styling_kwargs,
            )
            # Store yerr in metadata for PlotBuilder to use
            plot_data.metadata["yerr"] = errors
            plot_data.metadata["capsize"] = capsize
        else:
            # Use scatter plot type (no error bars)
            plot_data = AngularDistributionPlotData(
                x=angles,
                y=cross_sections,
                energy=actual_energy,
                isotope=self.target,
                mt=mt,
                distribution_type="experimental",
                label=label,
                plot_type="scatter",
                marker=marker,
                markersize=markersize,
                # Don't pass linestyle to scatter - it doesn't support it
                **styling_kwargs,
            )

        plot_data.metadata.update(metadata)

        if not uncertainty:
            return plot_data

        # For errorbar type, we don't need a separate UncertaintyBand
        # The errors are already in metadata['yerr']
        # But return None as second element for API consistency
        return plot_data, None

    def to_plot_data(
        self,
        energy: Union[float, Tuple[float, float]] = None,
        *,
        tolerance: float = PLOTTING_ENERGY_TOLERANCE,
        select_all_in_range: bool = False,
        frame: Optional[str] = None,
        angle_unit: str = "cos",
        cross_section_unit: str = "b/sr",
        uncertainty: bool = True,
        connect_points: bool = False,
        label: Optional[str] = None,
        include_natural_tag: bool = True,
        **styling_kwargs
    ) -> Union[
        "AngularDistributionPlotData",
        Tuple["AngularDistributionPlotData", "UncertaintyBand"],
        List[Union["AngularDistributionPlotData", Tuple["AngularDistributionPlotData", "UncertaintyBand"]]],
        None
    ]:
        """
        Extract angular distribution data for plotting with PlotBuilder.

        This method returns data in a format compatible with kika's PlotBuilder,
        enabling easy visualization and comparison of EXFOR experimental data
        with theoretical calculations from ACE/ENDF files.

        Parameters
        ----------
        energy : float, tuple, or None
            Energy selection:
            - float: Select data at specific energy (with absolute tolerance matching)
            - tuple: Select data in energy range (min, max) in MeV
            - None: Return data for all available energies
        tolerance : float, default PLOTTING_ENERGY_TOLERANCE (0.01 MeV = 10 keV)
            Absolute tolerance in MeV for energy matching.
            Only used when energy is a single float value.
        select_all_in_range : bool, default False
            Controls behavior when energy is a single float:
            - False (default): Return only data at the single closest energy within tolerance
            - True: Return data at ALL energies within the tolerance range
        frame : str or None, default None
            Output reference frame:
            - 'LAB': Convert to laboratory frame
            - 'CM': Convert to center-of-mass frame
            - None: Keep current frame (no conversion)
        angle_unit : str, default 'cos'
            Angle unit for x-axis:
            - 'cos': Cosine of scattering angle (range [-1, 1])
            - 'deg': Degrees (range [0, 180])
        cross_section_unit : str, default 'b/sr'
            Cross section unit for y-axis:
            - 'b/sr': barns per steradian
            - 'mb/sr': millibarns per steradian
        uncertainty : bool, default True
            If True, include error bars in the plot (uses 'errorbar' plot type).
            If False, show only scatter points (uses 'scatter' plot type).
        connect_points : bool, default False
            If True, connect data points with lines.
            If False (default), show only markers (scatter plot style).
        label : str or None, default None
            Custom label for legend. If None, auto-generates from author and year
            (e.g., "Kinney et al. (1976)"). For natural targets, "[nat]" is appended
            if include_natural_tag is True.
        include_natural_tag : bool, default True
            If True and the experiment's target is a natural element (ZAID ends in 000),
            append '[nat]' to the label. This helps distinguish natural Fe from Fe-56.
        **styling_kwargs
            Additional styling parameters passed to PlotData:
            - color: Line/marker color
            - marker: Marker style (default: 'o')
            - markersize: Marker size (default: 5)
            - capsize: Error bar cap size (default: 2)
            - alpha: Transparency

        Returns
        -------
        ~kika.plotting.AngularDistributionPlotData, tuple, list or None
            The points are sorted by angle, in *angle_unit* and *cross_section_unit*,
            and the frame is on ``metadata["frame"]``. With ``uncertainty=True``
            and at least one positive error, the error bars are in
            ``metadata["yerr"]``, absolute and in the cross-section unit. They
            combine ``uncertainty_stat`` and ``uncertainty_sys`` in quadrature, and
            after a frame conversion ``uncertainty_sys`` is the unconverted one.
            Otherwise the plot data are plain points.

            - None if a single *energy* has no energy block within *tolerance*.
            - One energy with ``uncertainty=True``: the tuple
              ``(plot_data, None)``. The second item is always None.
            - One energy with ``uncertainty=False``: the plot data alone.
            - A range or no *energy*, or a single *energy* with
              ``select_all_in_range=True``: a list of the above, one per energy,
              except that when the selection holds exactly one energy the item
              itself is returned, not a list of one.

        Raises
        ------
        ValueError
            If *angle_unit*, *cross_section_unit* or *frame* is not one of the
            values above, if the object has no energies, or if the *energy* range
            contains none.

        Examples
        --------
        >>> from kika.exfor import X4ProDatabase
        >>> from kika.plotting import PlotBuilder
        >>>
        >>> db = X4ProDatabase()
        >>> exfor = db.load_experiment("10571002")  # Kinney Fe-56
        >>>
        >>> # Plot at single energy with uncertainty band
        >>> plot_data = exfor.to_plot_data(energy=5.0, frame='CM')
        >>> if plot_data is not None:
        ...     fig = PlotBuilder().add_data(plot_data).set_labels(
        ...         x_label=r'$\\cos(\\theta)$',
        ...         y_label=r'$d\\sigma/d\\Omega$ (b/sr)'
        ...     ).build()
        >>>
        >>> # Plot without uncertainty
        >>> plot_data = exfor.to_plot_data(energy=5.0, uncertainty=False)
        >>>
        >>> # Plot multiple energies in range
        >>> plot_list = exfor.to_plot_data(energy=(1.0, 8.0))
        >>> builder = PlotBuilder()
        >>> for pd in plot_list:
        ...     builder.add_data(pd)
        >>> fig = builder.build()
        """
        # Validate parameters
        if angle_unit not in ("cos", "deg"):
            raise ValueError(f"Invalid angle_unit '{angle_unit}'. Must be 'cos' or 'deg'.")
        if cross_section_unit not in ("b/sr", "mb/sr"):
            raise ValueError(f"Invalid cross_section_unit '{cross_section_unit}'. Must be 'b/sr' or 'mb/sr'.")
        if frame is not None and frame.upper() not in (FRAME_LAB, FRAME_CM):
            raise ValueError(f"Invalid frame '{frame}'. Must be 'LAB', 'CM', or None.")

        # Get all available energies
        available_energies = self.energies(unit="MeV")

        if len(available_energies) == 0:
            raise ValueError("No energy data available in this EXFOR entry.")

        # Determine which energies to process
        if energy is None:
            # All energies
            selected_energies = available_energies
        elif isinstance(energy, tuple):
            # Energy range
            e_min, e_max = energy
            mask = (available_energies >= e_min) & (available_energies <= e_max)
            selected_energies = available_energies[mask]
            if len(selected_energies) == 0:
                raise ValueError(
                    f"No data in energy range ({e_min}, {e_max}) MeV. "
                    f"Available energies: {available_energies}"
                )
        else:
            # Single energy with absolute tolerance matching
            energy_float = float(energy)
            # Find energies within absolute tolerance
            abs_diff = np.abs(available_energies - energy_float)
            within_tolerance = abs_diff <= tolerance
            if not np.any(within_tolerance):
                # No data within tolerance - return None for graceful skipping
                return None

            if select_all_in_range:
                # Return all energies within tolerance range
                selected_energies = available_energies[within_tolerance]
            else:
                # Only the closest energy within tolerance
                matching_energies = available_energies[within_tolerance]
                closest_idx = np.argmin(np.abs(matching_energies - energy_float))
                selected_energies = [matching_energies[closest_idx]]

        # Process each selected energy
        results = []
        for e in selected_energies:
            result = self._to_plot_data_single_energy(
                energy=e,
                frame=frame,
                angle_unit=angle_unit,
                cross_section_unit=cross_section_unit,
                uncertainty=uncertainty,
                connect_points=connect_points,
                label=label,
                include_natural_tag=include_natural_tag,
                **styling_kwargs
            )
            results.append(result)

        # Return single item if only one energy selected
        if len(results) == 1:
            return results[0]

        return results

"""In-memory model of a SCALE Sensitivity Data File (SDF) and its writer.

:class:`SDFData` holds one response and the sensitivity profiles
(:class:`SDFReactionData`) of its nuclides and reactions on one energy grid, in MeV
and per group. Uncertainties are absolute one-sigma standard deviations.
:func:`~kika.sensitivities.sdf_parser.read_sdf` reads a file into it and
:meth:`SDFData.write_file` writes it.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os
import numpy as np
from kika._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL
from kika.sensitivities.sensitivity import SensitivityData
from kika.energy_grids.utils import _identify_energy_grid
from kika.plotting import MultigroupCrossSectionPlotData, UncertaintyBand


def sensitivity_to_plot_data(
    pert_energies,
    sensitivity,
    error=None,
    zaid: int = None,
    mt: int = None,
    nuclide: str = None,
    reaction_name: str = None,
    per_lethargy: bool = True,
    uncertainty: bool = True,
    sigma: float = 1.0,
    uncertainty_style: str = 'errorbar',
    label: str = None,
    **styling_kwargs,
) -> Union['MultigroupCrossSectionPlotData', Tuple['MultigroupCrossSectionPlotData', 'UncertaintyBand']]:
    """Step-plot data and an optional absolute uncertainty band from raw arrays.

    The core of :meth:`SDFReactionData.to_plot_data`, usable on plain arrays, such as
    the profile vectors of
    :meth:`~kika.benchmarks.BenchmarksDatabase.get_profile_vector`. The step plot is on
    the boundaries *pert_energies*, for the :class:`~kika.plotting.PlotBuilder`. The
    band is ``y +/- sigma * std`` with ``std`` the absolute standard deviation, divided
    by the lethargy width too when *per_lethargy* is True.

    Parameters
    ----------
    pert_energies
        Group boundaries, ``n + 1`` values in ascending order, in MeV. A descending
        grid gives negative lethargy widths when *per_lethargy* is True.
    sensitivity
        The ``n`` group sensitivities, per group (not per unit lethargy).
    error : array_like, optional
        The ``n`` **absolute** one-sigma standard deviations, in the units of
        *sensitivity*. None gives no band.
    zaid, mt : int, optional
        Metadata of the plot, and the default label when *nuclide* or
        *reaction_name* is missing.
    nuclide : str, optional
        Symbol for the default label (for example ``"Fe-56"``).
    reaction_name : str, optional
        Reaction for the default label (for example ``"(n,el)"``).
    per_lethargy
        If True the values (and *error*) are divided by ``ln(E[i+1] / E[i])`` of each
        group.
    uncertainty
        If False no band is returned.
    sigma
        Number of standard deviations of the band.
    uncertainty_style : {"errorbar", "band"}
        How the band is drawn.
    label : str, optional
        Legend label. None joins the nuclide (or ``"ZAID <zaid>"``) and the reaction
        name (or ``"MT<mt>"``); None remains if none of them is known.
    **styling_kwargs
        Passed to :class:`~kika.plotting.MultigroupCrossSectionPlotData`.

    Returns
    -------
    ~kika.plotting.MultigroupCrossSectionPlotData or tuple
        The data alone when *uncertainty* is False or *error* is None; otherwise
        ``(data, band)`` with an :class:`~kika.plotting.UncertaintyBand`, whose own
        label is ``"<label> (<sigma>σ)"`` when *sigma* is not 1 and None otherwise.

    Raises
    ------
    ValueError
        If *uncertainty* is True and *error* is given with another shape than
        *sensitivity* (the check is skipped otherwise).
    """
    energies = np.asarray(pert_energies, dtype=float)
    sens = np.asarray(sensitivity, dtype=float)

    if per_lethargy:
        lethargy = np.log(energies[1:] / energies[:-1])
        y_vals = sens / lethargy
    else:
        y_vals = sens.copy()

    # Step plot: n+1 x-points, repeat last y value.
    x = energies
    y = np.append(y_vals, y_vals[-1])

    if label is None:
        parts = [
            nuclide or (f"ZAID {zaid}" if zaid is not None else ""),
            reaction_name or (f"MT{mt}" if mt is not None else ""),
        ]
        label = " ".join(p for p in parts if p).strip() or None

    plot_data = MultigroupCrossSectionPlotData(
        x=x,
        y=y,
        label=label,
        plot_type='step',
        step_where='post',
        zaid=zaid,
        mt=mt,
        energy_bins=energies,
        **styling_kwargs,
    )

    # No uncertainty band when not requested or when no per-group errors exist
    # (e.g. compact SDF dialect / benchmark reactions stored without errors).
    if not uncertainty or error is None:
        return plot_data

    std = np.asarray(error, dtype=float)
    if std.shape != sens.shape:
        raise ValueError("sensitivity and error must have the same length")
    if per_lethargy:
        std = std / lethargy
    std_extended = np.append(std, std[-1])

    band = UncertaintyBand(
        x=x,
        y_lower=y - sigma * std_extended,
        y_upper=y + sigma * std_extended,
        sigma=1.0,
        label=f"{label} ({sigma}σ)" if sigma != 1.0 else None,
        style=uncertainty_style,
    )
    return plot_data, band


@dataclass
class SDFReactionData:
    """The sensitivity profile of one nuclide and reaction on the SDF energy grid.

    Built by hand, or found in :attr:`SDFData.data` after
    :func:`~kika.sensitivities.sdf_parser.read_sdf` or
    :func:`~kika.sensitivities.sensitivity_processing.create_sdf_data`. The values are
    the groupwise
    sensitivities, as stored in an SDF: per group, not per unit lethargy. Their sum is
    the integrated coefficient that :meth:`SDFData.write_file` writes. Construction
    makes the two lists floats and checks them.

    Attributes
    ----------
    zaid : int
        ZAID of the nuclide.
    mt : int
        MT reaction number.
    sensitivity : list of float
        One finite value per energy group, in the order of ``SDFData.pert_energies``.
    error : list of float
        **Absolute** one-sigma standard deviation of each value, non-negative, with the
        units of *sensitivity*. All zeros, with ``error_known`` False, when the source
        gives none.
    nuclide : str
        Symbol of *zaid*, such as ``"Fe-56"``. Set on construction.
    reaction_name : str, optional
        Name of the reaction. None gives the name of *mt* in kika's reaction table,
        or ``"MT<mt>"`` when it has none. A reader keeps the name written in the file.
    unit, region : int, optional
        Unit and region indices of the profile's first metadata line in an SDF;
        ``(0, 0)`` is a region-integrated (system-total) profile. None when the
        source states none; they count as 0 in :meth:`SDFData.write_file` and in
        :meth:`SDFData.to_sensitivity_profile`, which keeps only the ``(0, 0)`` profiles.
    error_known : bool
        False when the source states no uncertainty for the profile (the compact
        TSUNAMI-1D dialect has no error block).

    Raises
    ------
    ValueError
        If *sensitivity* or *error* is not one-dimensional, they differ in length,
        either has a non-finite value, or *error* has a negative value.
    KeyError
        If the atomic number of *zaid* is not in kika's table of elements.
    """
    zaid: int
    mt: int
    sensitivity: List[float]
    error: List[float]
    nuclide: str = field(init=False)
    # reaction_name can be provided (e.g. for unknown MT numbers not in mapping);
    # if None it's inferred from MT_TO_REACTION in __post_init__.
    reaction_name: str | None = None
    # SCALE unit/region indices from the profile's first metadata line. (0, 0)
    # denotes a region-integrated (system-total) profile; other values are
    # per-region/per-mixture breakdowns. None for KIKA-written files without the
    # information. Used by the benchmarks subpackage to keep only system totals.
    unit: int | None = None
    region: int | None = None
    # False when the source states no uncertainty for these coefficients (the
    # compact TSUNAMI-1D dialect has no error block). ``error`` then holds
    # zeros only because the format needs a number; a profile built from this
    # reaction carries ``uncertainty=None`` ("unknown"), never "exact".
    error_known: bool = True

    def __post_init__(self):
        """Calculate and store nuclide symbol and reaction name after initialization.

        If the MT number is not present in ``MT_TO_REACTION`` and no
        ``reaction_name`` was supplied, we fall back to a generic label
        ``MT<mt>`` so that previously unknown / extended MT numbers do not
        cause parsing to fail. When a custom ``reaction_name`` is supplied
        (e.g. by the parser reading the legacy file), it is preserved as is.
        """
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        error = np.asarray(self.error, dtype=float)
        if sensitivity.ndim != 1 or error.ndim != 1:
            raise ValueError("sensitivity and error must be one-dimensional")
        if sensitivity.size != error.size:
            raise ValueError(
                "sensitivity and error must have the same length "
                f"({sensitivity.size} != {error.size})"
            )
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("sensitivity values must be finite")
        if not np.all(np.isfinite(error)):
            raise ValueError("error values must be finite")
        if np.any(error < 0):
            raise ValueError("error values must be non-negative absolute standard deviations")
        self.sensitivity = sensitivity.tolist()
        self.error = error.tolist()

        # Calculate nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000

        if z not in ATOMIC_NUMBER_TO_SYMBOL:
            raise KeyError(f"Atomic number {z} not found in ATOMIC_NUMBER_TO_SYMBOL dictionary")

        self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"

        # Calculate reaction name
        if self.reaction_name is None:
            if self.mt in MT_TO_REACTION:
                self.reaction_name = MT_TO_REACTION[self.mt]
            else:
                # Fallback: preserve unknown MT with generic name
                self.reaction_name = f"MT{self.mt}"

    @classmethod
    def from_relative_errors(
        cls,
        *,
        zaid: int,
        mt: int,
        sensitivity,
        relative_error,
        reaction_name: str | None = None,
        unit: int | None = None,
        region: int | None = None,
    ) -> "SDFReactionData":
        """Build a profile from relative one-sigma errors, stored as absolute ones.

        The stored error of each group is ``|sensitivity| * relative_error``, so a group
        with sensitivity 0 gets error 0 whatever its relative error.

        Parameters
        ----------
        zaid, mt
            Nuclide and reaction.
        sensitivity
            The groupwise sensitivities, per group.
        relative_error
            Relative one-sigma error of each value (a fraction of it), non-negative,
            of the shape of *sensitivity*.
        reaction_name, unit, region
            As the attributes of :class:`SDFReactionData`.

        Returns
        -------
        SDFReactionData
            With ``error_known`` True.

        Raises
        ------
        ValueError
            If the shapes differ, *relative_error* has a non-finite or negative value,
            or the checks of :class:`SDFReactionData` fail.
        """
        sensitivity_array = np.asarray(sensitivity, dtype=float)
        relative_array = np.asarray(relative_error, dtype=float)
        if sensitivity_array.shape != relative_array.shape:
            raise ValueError("sensitivity and relative_error must have the same shape")
        if not np.all(np.isfinite(relative_array)) or np.any(relative_array < 0):
            raise ValueError("relative_error values must be finite and non-negative")
        return cls(
            zaid=zaid,
            mt=mt,
            sensitivity=sensitivity_array.tolist(),
            error=(np.abs(sensitivity_array) * relative_array).tolist(),
            reaction_name=reaction_name,
            unit=unit,
            region=region,
        )

    @property
    def relative_error(self) -> List[float]:
        """Return the relative standard deviation for each sensitivity group."""
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        error = np.asarray(self.error, dtype=float)
        relative = np.zeros_like(error)
        nonzero = sensitivity != 0.0
        relative[nonzero] = error[nonzero] / np.abs(sensitivity[nonzero])
        relative[~nonzero & (error != 0.0)] = np.inf
        return relative.tolist()

    def __repr__(self) -> str:
        """Returns a formatted string representation of the reaction data.

        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.

        :return: Formatted string representation of the reaction data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'SDF Reaction Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"

        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels

        info_lines = []
        info_lines.append(f"{'Nuclide:':{label_width}} {self.nuclide} (ZAID {self.zaid})")
        info_lines.append(f"{'Reaction:':{label_width}} {self.reaction_name} (MT {self.mt})")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.sensitivity)}")

        stats = "\n".join(info_lines)

        # Data preview - show first few and last few sensitivity values
        data_preview = "\n\nSensitivity coefficients (preview):\n"
        data_preview += "  Group      Sensitivity        Abs. Std\n"
        data_preview += "  -----    --------------    ------------\n"

        # Show first 3 and last 3 groups, if available
        n_groups = len(self.sensitivity)
        preview_count = min(3, n_groups)

        for i in range(preview_count):
            data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"

        # Add ellipsis if there are more than 6 groups
        if n_groups > 6:
            data_preview += "  ...\n"

        # Show last 3 groups if there are more than 3 groups
        if n_groups > 3:
            for i in range(max(preview_count, n_groups-3), n_groups):
                data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"

        return header + stats + data_preview

    def to_plot_data(
        self,
        pert_energies,
        per_lethargy: bool = True,
        uncertainty: bool = True,
        sigma: float = 1.0,
        uncertainty_style: str = 'errorbar',
        label: str = None,
        **styling_kwargs
    ) -> Union['MultigroupCrossSectionPlotData', Tuple['MultigroupCrossSectionPlotData', 'UncertaintyBand']]:
        """Step-plot data of this profile, with an optional absolute uncertainty band.

        Calls :func:`~kika.sensitivities.sdf.sensitivity_to_plot_data` with the profile's
        ``sensitivity`` and ``error``. The band is built from ``error`` whatever
        ``error_known`` says, so a profile read from the compact SDF dialect gets a
        band of zero width.

        Parameters
        ----------
        pert_energies
            Group boundaries, ``len(sensitivity) + 1`` values in ascending order, in
            MeV (the SDF's own grid is ``SDFData.pert_energies``).
        per_lethargy
            If True the values and errors are divided by the lethargy width of each
            group.
        uncertainty
            If True the band is returned with the data.
        sigma
            Number of standard deviations of the band.
        uncertainty_style : {"errorbar", "band"}
            How the band is drawn.
        label : str, optional
            Legend label. None gives ``"<nuclide> <reaction_name>"``, for example
            ``"Fe-56 (n,el)"``.
        **styling_kwargs
            Passed to :class:`~kika.plotting.MultigroupCrossSectionPlotData`.

        Returns
        -------
        ~kika.plotting.MultigroupCrossSectionPlotData or tuple
            The data alone when *uncertainty* is False; otherwise ``(data, band)``
            with an :class:`~kika.plotting.UncertaintyBand`.
        """
        return sensitivity_to_plot_data(
            pert_energies,
            self.sensitivity,
            error=self.error,
            zaid=self.zaid,
            mt=self.mt,
            nuclide=self.nuclide,
            reaction_name=self.reaction_name,
            per_lethargy=per_lethargy,
            uncertainty=uncertainty,
            sigma=sigma,
            uncertainty_style=uncertainty_style,
            label=label,
            **styling_kwargs,
        )


@dataclass
class SDFData:
    """The sensitivity profiles of one response (an SDF), on one energy grid in MeV.

    A user gets one from
    :func:`~kika.sensitivities.sensitivity_processing.create_sdf_data`,
    :func:`~kika.sensitivities.sdf_parser.read_sdf` or :meth:`merge`, or builds one by
    hand. Sensitivities are per group and their uncertainties absolute one-sigma
    values, as in the SCALE convention. :meth:`write_file` writes the SCALE layout,
    with energies in eV and descending, and the response as ``r0 +/- e0`` in the line
    that SCALE labels k-eff; :func:`~kika.sensitivities.sdf_parser.read_sdf`
    reads it back. Construction checks the grid and the profiles.

    Attributes
    ----------
    title : str
        Title, the first line of the file.
    energy : str
        Label of the detector energy bin (``"lower_upper"`` in MeV or ``"integral"``
        when made by
        :func:`~kika.sensitivities.sensitivity_processing.create_sdf_data`). It is
        part of the file name :meth:`write_file` makes.
    pert_energies : list of float
        Group boundaries in MeV, ``n + 1`` values, finite, non-negative and strictly
        monotonic. :func:`~kika.sensitivities.sdf_parser.read_sdf` and
        :func:`~kika.sensitivities.sensitivity_processing.create_sdf_data` give them
        ascending.
    r0 : float, optional
        The response (for criticality, k-eff).
    e0 : float, optional
        **Absolute** one-sigma uncertainty of *r0*, finite and non-negative. None
        is written as 0.
    data : list of SDFReactionData
        The profiles. Each has ``n`` values.

    Raises
    ------
    ValueError
        If *pert_energies* is not a one-dimensional grid of at least two finite,
        non-negative, strictly monotonic values; a profile does not have ``n``
        values; *r0* or *e0* is not finite; or *e0* is negative.
    """
    title: str
    energy: str
    pert_energies: List[float]
    r0: float = None
    e0: float = None
    data: List[SDFReactionData] = field(default_factory=list)

    @property
    def relative_response_error(self) -> float | None:
        """Return the relative response uncertainty ``e0 / abs(r0)``."""
        if self.e0 is None:
            return None
        if self.r0 in (None, 0):
            return 0.0 if self.e0 == 0 else float("inf")
        return abs(self.e0 / self.r0)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        energies = np.asarray(self.pert_energies, dtype=float)
        if energies.ndim != 1 or energies.size < 2:
            raise ValueError("pert_energies must be a one-dimensional boundary grid")
        if not np.all(np.isfinite(energies)) or np.any(energies < 0):
            raise ValueError("pert_energies must contain finite non-negative values")
        differences = np.diff(energies)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("pert_energies must be strictly monotonic")
        ngroups = energies.size - 1
        for reaction in self.data:
            if len(reaction.sensitivity) != ngroups:
                raise ValueError(
                    f"ZAID={reaction.zaid} MT={reaction.mt} has "
                    f"{len(reaction.sensitivity)} groups; expected {ngroups}"
                )
        if self.r0 is not None and not np.isfinite(self.r0):
            raise ValueError("r0 must be finite")
        if self.e0 is not None and not np.isfinite(self.e0):
            raise ValueError("e0 must be finite")
        if self.e0 is not None and self.e0 < 0:
            raise ValueError("e0 must be a non-negative absolute standard deviation")

    def to_sensitivity_profile(self) -> "SensitivityProfile":
        """Return the system-total data as a format-neutral sensitivity profile.

        SDF entries with non-zero ``unit`` or ``region`` identify auxiliary
        mixture/region profiles and are deliberately excluded from UQ vectors.
        """
        from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction

        energies = np.asarray(self.pert_energies, dtype=float)
        reverse = bool(energies[0] > energies[-1])
        if reverse:
            energies = energies[::-1]

        reactions = []
        for reaction in self.data:
            if (reaction.unit or 0) != 0 or (reaction.region or 0) != 0:
                continue
            sensitivity = np.asarray(reaction.sensitivity, dtype=float)
            uncertainty = (np.asarray(reaction.error, dtype=float)
                           if reaction.error_known else None)
            if reverse:
                sensitivity = sensitivity[::-1]
                if uncertainty is not None:
                    uncertainty = uncertainty[::-1]
            reactions.append(
                SensitivityReaction(
                    zaid=reaction.zaid,
                    mt=reaction.mt,
                    sensitivity=sensitivity,
                    uncertainty=uncertainty,
                    label=reaction.reaction_name,
                )
            )

        return SensitivityProfile(
            energy_grid=energies,
            energy_unit="MeV",
            reactions=tuple(reactions),
            response=self.r0,
            response_uncertainty=self.e0,
            label=self.title,
            metadata={"source": "SDFData", "energy_label": self.energy},
        )

    def __repr__(self) -> str:
        """Returns a detailed formatted string representation of the SDF data.

        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.

        :return: Formatted string representation of the SDF data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'SDF Data: ' + self.title:^{header_width}}\n"
        header += f"{'Energy range: ' + self.energy:^{header_width}}\n"
        header += "=" * header_width + "\n\n"

        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels

        # Basic information section
        info_lines = []
        info_lines.append(f"{'Response value:':{label_width}} {self.r0:.6e} ± {self.e0:.6e} (abs. std)")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.pert_energies) - 1}")

        # Add energy grid structure identification
        grid_name = _identify_energy_grid(self.pert_energies)
        if grid_name:
            info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")

        info_lines.append(f"{'Sensitivity profiles:':{label_width}} {len(self.data)}")

        # Count unique nuclides
        nuclides = {react.nuclide for react in self.data}
        info_lines.append(f"{'Unique nuclides:':{label_width}} {len(nuclides)}")

        stats = "\n".join(info_lines)

        # Energy grid preview
        energy_preview = "\n\nEnergy grid (preview):"
        energy_grid = "  " + ", ".join(f"{e:.6e}" for e in self.pert_energies[:3])

        if len(self.pert_energies) > 6:
            energy_grid += ", ... , "
            energy_grid += ", ".join(f"{e:.6e}" for e in self.pert_energies[-3:])
        elif len(self.pert_energies) > 3:
            energy_grid += ", " + ", ".join(f"{e:.6e}" for e in self.pert_energies[3:])

        energy_preview += "\n  " + energy_grid

        # Data summary - most important nuclides and reactions with indices
        data_summary = "\n\nNuclides and reactions (with access indices):\n"

        # Group by nuclide
        nuclide_reactions = {}
        nuclide_indices = {}

        # Store all reaction data with their indices
        for idx, react in enumerate(self.data):
            if react.nuclide not in nuclide_reactions:
                nuclide_reactions[react.nuclide] = []
                nuclide_indices[react.nuclide] = []
            nuclide_reactions[react.nuclide].append((react.reaction_name, react.mt))
            nuclide_indices[react.nuclide].append(idx)

        # Determine width for consistent alignment
        reaction_width = 30  # Base width for reaction name + MT

        # Show data for each nuclide (limit to first 5 nuclides)
        for i, nuclide in enumerate(sorted(nuclide_reactions.keys())):
            if i >= 5:
                data_summary += f"\n  ... ({len(nuclides) - 5} more nuclides) ...\n"
                break

            data_summary += f"\n  {nuclide}:\n"
            reactions = nuclide_reactions[nuclide]
            indices = nuclide_indices[nuclide]

            # Sort by MT number but keep track of original indices
            sorted_data = sorted(zip(reactions, indices), key=lambda x: x[0][1])

            for j, ((name, mt), idx) in enumerate(sorted_data):
                if j >= 10:  # Limit to 10 reactions per nuclide
                    data_summary += f"    ... ({len(reactions) - 10} more reactions) ...\n"
                    break
                # Format the reaction info with consistent alignment for the "access with" part
                reaction_info = f"{name} (MT={mt})"
                data_summary += f"    {reaction_info:{reaction_width}} access with .data[{idx}]\n"

        # Footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .to_plot_data() - Get PlotData for use with PlotBuilder\n"
        footer += "- .write_file() - Write SDF data to a file\n"
        footer += "- .group_inelastic_reactions() - Group MT 51-91 into MT 4\n"
        footer += "- SDFData.merge() - Merge multiple SDF objects\n"
        footer += "- SDFData.can_merge() - Check if SDF objects can be merged\n"

        return header + stats + energy_preview + data_summary + footer

    def to_plot_data(
        self,
        index: int = None,
        zaid: int = None,
        mt: int = None,
        per_lethargy: bool = True,
        uncertainty: bool = True,
        sigma: float = 1.0,
        uncertainty_style: str = 'errorbar',
        label: str = None,
        **styling_kwargs
    ) -> Union['MultigroupCrossSectionPlotData', Tuple['MultigroupCrossSectionPlotData', 'UncertaintyBand']]:
        """Step-plot data of one profile, chosen by position or by ZAID and MT.

        Passes the profile and this object's ``pert_energies`` (MeV) to
        :meth:`SDFReactionData.to_plot_data`, which describes the data and the band.

        Parameters
        ----------
        index : int, optional
            Position in ``data``. When given, *zaid* and *mt* are ignored.
        zaid, mt : int, optional
            Both are needed to select the first profile of ``data`` with that ZAID and
            MT.
        per_lethargy, uncertainty, sigma, uncertainty_style, label
            As in :meth:`SDFReactionData.to_plot_data`.
        **styling_kwargs
            Passed to :class:`~kika.plotting.MultigroupCrossSectionPlotData`.

        Returns
        -------
        ~kika.plotting.MultigroupCrossSectionPlotData or tuple
            The data alone when *uncertainty* is False; otherwise ``(data, band)``.

        Raises
        ------
        ValueError
            If *index* is out of range, no profile has the given *zaid* and *mt*, or
            neither *index* nor both *zaid* and *mt* are given.
        """
        if index is not None:
            if index < 0 or index >= len(self.data):
                raise ValueError(
                    f"index {index} out of range (0..{len(self.data) - 1})"
                )
            reaction = self.data[index]
        elif zaid is not None and mt is not None:
            reaction = None
            for rd in self.data:
                if rd.zaid == zaid and rd.mt == mt:
                    reaction = rd
                    break
            if reaction is None:
                raise ValueError(
                    f"No reaction found with ZAID={zaid}, MT={mt}"
                )
        else:
            raise ValueError(
                "Provide either 'index' or both 'zaid' and 'mt'"
            )

        return reaction.to_plot_data(
            self.pert_energies,
            per_lethargy=per_lethargy,
            uncertainty=uncertainty,
            sigma=sigma,
            uncertainty_style=uncertainty_style,
            label=label,
            **styling_kwargs,
        )

    @classmethod
    def merge(cls, sdf_list: List['SDFData'],
              title: Optional[str] = None,
              energy: Optional[str] = None,
              r0: Optional[float] = None,
              e0: Optional[float] = None) -> 'SDFData':
        """Merge several SDFs on the same energy grid into one.

        The result holds every profile of the inputs (the same
        :class:`SDFReactionData` objects, not copies). The grids are compared as lists,
        so they must be equal value for value. ``r0`` and ``e0`` that the inputs state
        (not None) must be equal, unless given here.

        Parameters
        ----------
        sdf_list
            The SDFs to combine.
        title : str, optional
            Title of the result. None keeps the title the inputs share, or joins their
            distinct titles with ``" + "``.
        energy : str, optional
            Energy label of the result. None takes the label the inputs share.
        r0 : float, optional
            Response of the result. None takes the common value of the inputs that
            have one (None when none has).
        e0 : float, optional
            **Absolute** one-sigma error of the response, as *r0*.

        Returns
        -------
        SDFData
            A new object.

        Raises
        ------
        ValueError
            If *sdf_list* is empty; the grids differ; the inputs state different
            ``r0`` (or ``e0``) and none is given; the energy labels differ and none is
            given; or the same ``(zaid, mt)`` appears in two profiles.
        """
        if not sdf_list:
            raise ValueError("Cannot merge an empty list of SDFData objects")

        # --- energy grid ---
        ref_energies = sdf_list[0].pert_energies
        for i, sdf in enumerate(sdf_list[1:], 1):
            if sdf.pert_energies != ref_energies:
                raise ValueError(
                    f"Energy grids do not match: SDF 0 ('{sdf_list[0].title}') has "
                    f"{len(ref_energies)} boundaries, SDF {i} ('{sdf.title}') has "
                    f"{len(sdf.pert_energies)} boundaries"
                )

        # --- r0 / e0 ---
        if r0 is None:
            r0_values = {s.r0 for s in sdf_list if s.r0 is not None}
            if len(r0_values) > 1:
                raise ValueError(
                    f"SDFs have different r0 values {r0_values}. "
                    f"Provide the 'r0' parameter explicitly."
                )
            r0 = r0_values.pop() if r0_values else None

        if e0 is None:
            e0_values = {s.e0 for s in sdf_list if s.e0 is not None}
            if len(e0_values) > 1:
                raise ValueError(
                    f"SDFs have different e0 values {e0_values}. "
                    f"Provide the 'e0' parameter explicitly."
                )
            e0 = e0_values.pop() if e0_values else None

        # --- energy label ---
        if energy is None:
            labels = {s.energy for s in sdf_list}
            if len(labels) == 1:
                energy = labels.pop()
            else:
                raise ValueError(
                    f"SDFs have different energy labels {labels}. "
                    f"Provide the 'energy' parameter explicitly."
                )

        # --- title ---
        if title is None:
            titles = list(dict.fromkeys(s.title for s in sdf_list))  # unique, order-preserving
            title = " + ".join(titles)

        # --- duplicate check & collect data ---
        seen = {}
        combined_data = []
        for i, sdf in enumerate(sdf_list):
            for rd in sdf.data:
                key = (rd.zaid, rd.mt)
                if key in seen:
                    raise ValueError(
                        f"Duplicate reaction: {rd.nuclide} {rd.reaction_name} "
                        f"(ZAID={rd.zaid}, MT={rd.mt}) appears in SDF {seen[key]} "
                        f"and SDF {i} ('{sdf.title}')"
                    )
                seen[key] = i
                combined_data.append(rd)

        return cls(
            title=title,
            energy=energy,
            pert_energies=ref_energies,
            r0=r0,
            e0=e0,
            data=combined_data,
        )

    @classmethod
    def can_merge(cls, sdf_list: List['SDFData'],
                  energy: Optional[str] = None,
                  r0: Optional[float] = None,
                  e0: Optional[float] = None) -> Tuple[bool, str]:
        """Whether :meth:`merge` would succeed, with the reason when it would not.

        Runs :meth:`merge` and catches its ``ValueError``; the title does not matter
        to the outcome.

        Parameters
        ----------
        sdf_list
            The SDFs to check.
        energy, r0, e0
            The overrides you would give to :meth:`merge`.

        Returns
        -------
        tuple of (bool, str)
            ``(True, "")`` when the merge succeeds, otherwise ``(False, message)`` with
            the message of the ``ValueError``.
        """
        try:
            cls.merge(sdf_list, energy=energy, r0=r0, e0=e0)
            return True, ""
        except ValueError as exc:
            return False, str(exc)

    def __add__(self, other: 'SDFData') -> 'SDFData':
        """Merge two SDFData objects using the ``+`` operator.

        :param other: Another SDFData object.
        :type other: SDFData
        :returns: A new merged SDFData instance.
        :rtype: SDFData
        """
        if not isinstance(other, SDFData):
            return NotImplemented
        return SDFData.merge([self, other])

    def write_file(self, output_dir: Optional[str] = None):
        r"""Write ``<title>_<energy>.sdf`` in the SCALE layout and print its path.

        The name has spaces, ``/`` and ``\`` replaced by ``_``. An existing file of that
        name is overwritten. The file holds, in this order:

        - the title, the number of groups and of profiles (with the number of
          region-integrated ones, those with ``unit`` and ``region`` both 0);
        - the line ``r0 +/- e0`` (``e0`` absolute; None is written as 0), with no
          label after it;
        - the energy boundaries in **eV**, **descending**: the MeV grid times
          ``1e6``, reversed, so it assumes ``pert_energies`` is ascending;
        - for each profile, sorted by ZAID and then MT: a header line, the unit and
          region, a fixed line of zeros, five scalars, the group sensitivities and
          their absolute standard deviations, both in descending-energy order.

        The five scalars are the sum of the sensitivities; its standard deviation (the
        absolute errors added in quadrature); the sum of the absolute values; the sum of
        the sensitivities whose sign is opposite to the total (for a total of exactly
        0, the negative ones if any is negative, otherwise the positive ones); and the
        standard deviation of that subset in quadrature. Numbers are written with six
        decimals in scientific notation.

        Parameters
        ----------
        output_dir : str, optional
            Directory of the file, created if missing. None is the current directory.

        Raises
        ------
        ValueError
            If the object fails the checks of :class:`SDFData`.
        """
        self._validate()
        # Use current directory if output_dir is not provided
        if output_dir is None:
            output_dir = os.getcwd()

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a clean filename from title and energy
        filename = f"{self.title}_{self.energy}.sdf"
        # Ensure filename is valid by removing problematic characters
        filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')

        # Create full path to file
        filepath = os.path.join(output_dir, filename)

        ngroups = len(self.pert_energies) - 1
        nprofiles = len(self.data)
        n_integrated = sum((reaction.unit or 0, reaction.region or 0) == (0, 0) for reaction in self.data)

        # Sort the data by ZAID and then by MT number
        sorted_data = sorted(self.data, key=lambda x: (x.zaid, x.mt))

        with open(filepath, 'w', encoding='utf-8') as file:
            # Write header
            file.write(f"{self.title}\n")
            file.write(f"       {ngroups} number of neutron groups\n")
            file.write(f"       {nprofiles}  number of sensitivity profiles         {n_integrated} are region integrated\n")

            # Ensure r0 and e0 are properly formatted
            r0_value = 0.0 if self.r0 is None else self.r0
            e0_value = 0.0 if self.e0 is None else self.e0
            file.write(f"  {r0_value:.6E} +/-   {e0_value:.6E}\n")

            # Write energy grid data - reversed to be in descending order
            file.write("energy boundaries:\n")
            energy_lines = ""
            # Create reversed list of energies for writing in descending order
            # SCALE SDF energy boundaries are written in eV; KIKA stores MeV.
            descending_energies = [energy * 1.0e6 for energy in reversed(self.pert_energies)]
            for idx, energy in enumerate(descending_energies):
                if idx > 0 and idx % 5 == 0:
                    energy_lines += "\n"
                energy_lines += f"{energy: >14.6E}"
            energy_lines += "\n"
            file.write(energy_lines)

            # Write sensitivity coefficient and standard deviations data for each reaction
            # using the sorted data
            for reaction in sorted_data:
                file.write(self._format_reaction_data(reaction))

        # Add print message indicating where the file was saved
        print(f"SDF file saved successfully: {filepath}")

    def _format_reaction_data(self, reaction: SDFReactionData) -> str:
        """
        Format a single SDFReactionData block to match the legacy file structure.

        :param reaction: The reaction data to format
        :type reaction: SDFReactionData
        :returns: Formatted string for the reaction data block
        :rtype: str
        """
        # Use the properties to get the nuclide symbol and reaction name
        form = reaction.nuclide
        reac = reaction.reaction_name

        # Format the header line for this reaction
        block = f"{form:<13}{reac:<17}{reaction.zaid:>5}{reaction.mt:>7}\n"
        unit = int(reaction.unit or 0)
        region = int(reaction.region or 0)
        block += f"{unit:7d}{region:7d}\n"
        block += "  0.000000E+00  0.000000E+00      0      0\n"

        # Calculate the 5 scalar values according to SDF specification
        # 1. Energy-integrated sensitivity coefficient (sum of groupwise sensitivities)
        s_int = sum(reaction.sensitivity)

        # 2. Standard deviation of S_int (error propagation for sum).
        # reaction.error stores absolute per-group standard deviations.

        s_int_std = sum(err**2 for err in reaction.error) ** 0.5

        # 3. Sum of absolute values of groupwise sensitivities
        sum_abs = sum(abs(s) for s in reaction.sensitivity)

        # 4. "osc" = sum of sensitivities with sign opposite to S_int.
        # Strict comparisons keep zero-valued groups (and their uncertainty)
        # out of the oscillating subset.
        if s_int > 0.0:
            oscillates = [s < 0.0 for s in reaction.sensitivity]
        elif s_int < 0.0:
            oscillates = [s > 0.0 for s in reaction.sensitivity]
        else:
            # SCALE does not define an opposite sign for an exactly zero
            # integral. Select the negative side deterministically, matching
            # the signed OSC representation used by ordinary profiles.
            use_negative = any(s < 0.0 for s in reaction.sensitivity)
            oscillates = [
                s < 0.0 if use_negative else s > 0.0
                for s in reaction.sensitivity
            ]
        osc = sum(
            s for s, include in zip(reaction.sensitivity, oscillates)
            if include
        )

        # 5. Standard deviation of OSC (quadrature over the same subset).
        osc_std = (
            sum(
                err**2
                for err, include in zip(reaction.error, oscillates)
                if include
            )
        )**0.5

        block += f"{s_int:>14.6E}{s_int_std:>14.6E}{sum_abs:>14.6E}{osc:>14.6E}{osc_std:>14.6E}\n"

        # Reverse sensitivity and error arrays to match the descending energy order
        reversed_sensitivity = list(reversed(reaction.sensitivity))
        reversed_error = list(reversed(reaction.error))

        # Write sensitivity coefficients with 5 per line (in reversed order)
        for idx, sens in enumerate(reversed_sensitivity):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{sens:>14.6E}"
        block += "\n"

        # Write standard deviations with 5 per line (in reversed order)
        for idx, err in enumerate(reversed_error):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{err:>14.6E}"
        block += "\n"
        return block

    def group_inelastic_reactions(self, replace: bool = False, remove_originals: bool = True) -> None:
        """Sum the profiles of MT 51 to 91 of each nuclide into its MT 4, in place.

        The group sensitivities are added, and the absolute errors are added in
        quadrature, which assumes the profiles are independent. Profiles are matched by
        ZAID and MT only, so ``unit`` and ``region`` are ignored; a new MT 4 has none.
        A nuclide with no MT 51 to 91 is left alone. Messages are printed, and nothing is
        returned.

        Parameters
        ----------
        replace
            If True an existing MT 4 of the nuclide is overwritten with the sum (its
            own values are not added in). If False an existing MT 4 is an error.
        remove_originals
            If True the profiles of MT 51 to 91 of the nuclide are removed afterwards.

        Raises
        ------
        ValueError
            If a nuclide that has MT 51 to 91 already has an MT 4 and *replace* is
            False. Nuclides processed before it have been changed by then.
        """
        # Group data by ZAID
        nuclide_reactions = {}
        for react in self.data:
            if react.zaid not in nuclide_reactions:
                nuclide_reactions[react.zaid] = []
            nuclide_reactions[react.zaid].append(react)

        # Process each nuclide
        for zaid, reactions in nuclide_reactions.items():
            # Find MT 4 if it exists
            mt4_exists = False
            mt4_reaction = None
            for react in reactions:
                if react.mt == 4:
                    mt4_exists = True
                    mt4_reaction = react
                    break

            # Find inelastic reactions (MT 51-91)
            inelastic_reactions = [r for r in reactions if 51 <= r.mt <= 91]

            # Skip if no inelastic reactions found for this nuclide
            if not inelastic_reactions:
                continue

            # Handle existing MT 4 reaction
            if mt4_exists and not replace:
                # Calculate the nuclide symbol for more informative error message
                z = zaid // 1000
                a = zaid % 1000
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"unknown_{z}")
                nuclide = f"{symbol}-{a}"

                raise ValueError(
                    f"MT 4 already exists for nuclide {nuclide} (ZAID {zaid}). "
                    f"Set replace=True to overwrite."
                )

            # Sum sensitivities and combine absolute standard deviations.
            n_groups = len(inelastic_reactions[0].sensitivity)
            summed_sensitivity = [0.0] * n_groups
            summed_error_squared = [0.0] * n_groups

            for react in inelastic_reactions:
                for i in range(n_groups):
                    summed_sensitivity[i] += react.sensitivity[i]
                    summed_error_squared[i] += react.error[i] ** 2

            summed_error = [value ** 0.5 for value in summed_error_squared]

            # Create or update MT 4 reaction
            if mt4_exists:
                mt4_reaction.sensitivity = summed_sensitivity
                mt4_reaction.error = summed_error
                print(f"Updated MT 4 for {mt4_reaction.nuclide} (ZAID {zaid})")
            else:
                new_mt4 = SDFReactionData(
                    zaid=zaid,
                    mt=4,
                    sensitivity=summed_sensitivity,
                    error=summed_error
                )
                self.data.append(new_mt4)
                print(f"Created MT 4 for {new_mt4.nuclide} (ZAID {zaid})")

            # Remove original MT 51-91 reactions if requested
            if remove_originals:
                mt_values = [r.mt for r in inelastic_reactions]
                print(f"Removed MT {', '.join(map(str, mt_values))} for {inelastic_reactions[0].nuclide} (ZAID {zaid})")

                # Remove the reactions from self.data
                self.data = [r for r in self.data if not (r.zaid == zaid and 51 <= r.mt <= 91)]

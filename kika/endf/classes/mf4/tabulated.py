from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Sequence, Optional
import math
import numpy as np

from .base import MF4MT
from ....endf.utils import (
    auto_trim_legendre_tail, evaluate_tabulated_pdf, interpolate_1d_endf,
    project_tabulated_to_legendre
)


@dataclass
class MF4MTTabulated(MF4MT):
    """
    MT section in MF4 with tabulated probability distributions (LTT=2).
    
    Stores tabulated angular distributions for each energy.
    """
    _ltt: int = 2
    _ne: int = None  # Number of energy points
    _nr: int = None  # Number of interpolation regions for energy grid
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Energy interpolation scheme pairs
    
    # Tabulated data storage
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _cosines: List[List[float]] = field(default_factory=list)  # Cosine values for each energy
    _probabilities: List[List[float]] = field(default_factory=list)  # Probability values for each energy
    _angular_interpolation: List[List[Tuple[int, int]]] = field(default_factory=list)  # Angular interpolation schemes
    
    @property
    def num_energy_points(self) -> int:
        """Number of energy points in the grid"""
        return self._ne or len(self._energies)
    
    @property
    def num_interpolation_regions(self) -> int:
        """Number of interpolation regions for the energy grid"""
        return self._nr or 0
    
    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for energy grid (NBT, INT)"""
        return self._interpolation
    
    @property
    def energies(self) -> List[float]:
        """Energy grid for angular distribution data"""
        return self._energies
    
    @property
    def cosines(self) -> List[List[float]]:
        """
        Cosine values (μ) for each energy point.
        
        Returns a list of cosine lists, aligned with the energy grid.
        Each inner list contains the cosine values for one energy point.
        """
        return self._cosines
    
    @property
    def probabilities(self) -> List[List[float]]:
        """
        Probability values f(μ,E) for each energy point and cosine.
        
        Returns a list of probability lists, aligned with the energy grid and cosines.
        Each inner list contains the probability values for one energy point.
        """
        return self._probabilities
    
    @property
    def cosine_interpolation(self) -> List[List[Tuple[int, int]]]:
        """
        Interpolation scheme pairs for angular data at each energy.
        
        Returns a list of interpolation scheme lists, aligned with the energy grid.
        Each inner list contains (NBT, INT) pairs for one energy point.
        """
        return self._angular_interpolation
    

    # ------------------------- core helpers -------------------------
    def evaluate_angular_pdf(
        self,
        mu,
        energy,
        *,
        out_of_range: str = "zero",
    ) -> np.ndarray:
        r"""The stored :math:`f(\mu, E)`, read rather than reconstructed.

        Overrides the base class, which would project this table onto Legendre
        and sum the projection back.  A truncated expansion is a real loss on a
        forward-peaked distribution -- 26 % at the peak for JEFF-4.0 U-235
        elastic at 18 MeV with 10 orders -- and there is nothing to gain by
        paying it when the evaluator's own points are right here.
        """
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        e_arr = np.atleast_1d(np.asarray(energy, dtype=float))
        out = np.empty((e_arr.size, mu_arr.size), dtype=float)
        for k, e in enumerate(e_arr):
            out[k] = evaluate_tabulated_pdf(
                mu_arr,
                float(e),
                energies=self._energies,
                cosines=self._cosines,
                probabilities=self._probabilities,
                angular_interp=self._angular_interpolation,
                energy_interp=self._interpolation,
                out_of_range=out_of_range,
            )
        return out

    def native_cosine_grid(self, energy) -> Optional[np.ndarray]:
        """The file's own cosines at *energy*.

        At a grid energy that is the row itself.  Between two, it is the union
        of the two bracketing rows, so no point either of them resolves is
        lost on the way to the interpolated distribution.
        """
        energies = np.asarray(self._energies, dtype=float)
        if energies.size == 0:
            return None

        E = float(energy)
        if E <= energies[0]:
            rows = [0]
        elif E >= energies[-1]:
            rows = [energies.size - 1]
        else:
            hi = int(np.searchsorted(energies, E, side="right"))
            rows = [hi - 1, hi]

        grid = np.unique(
            np.concatenate([np.asarray(self._cosines[i], dtype=float) for i in rows])
        )
        return grid if grid.size else None

    # ------------------------- public API -------------------------
    def legendre_table(self, max_order: int):
        """``(E, A, laws, hold)`` -- see :meth:`MF4MT.legendre_table`.

        f(mu, E) is interpolated between incident energies at fixed mu, and the
        projection onto P_l is linear in f, so under a law that is linear in y
        (1, 2 or 3) a_l(E) interpolates between the projections at the file's
        energies under that same law. Under a log-y law (4, 5) it does not, and
        there is no table to give. Held outside the grid, as
        :meth:`extract_legendre_coefficients` holds it.
        """
        from ....algebra import interval_laws

        energies = np.asarray(self._energies, dtype=float)
        laws = (interval_laws(energies.size, self._interpolation or [(energies.size, 2)])
                if energies.size > 1 else np.zeros(0, dtype=np.int64))
        if np.any(np.isin(laws, (4, 5)) & (np.diff(energies) > 0)):
            raise ValueError(
                "a log-y energy law interpolates f(mu, E), not its Legendre "
                "projection, so a_l(E) is not a table under it")
        table = np.zeros((energies.size, max_order + 1))
        for i in range(energies.size):
            pairs = (self._angular_interpolation[i]
                     if i < len(self._angular_interpolation) and self._angular_interpolation[i]
                     else None)
            table[i] = project_tabulated_to_legendre(
                np.asarray(self._cosines[i], dtype=float),
                np.asarray(self._probabilities[i], dtype=float), max_order, pairs)
        return energies, table, laws, True

    def extract_legendre_coefficients(
        self,
        energy: Union[float, np.ndarray],
        max_legendre_order: int = 10,
        *,
        trim: bool = False,
        trim_tol: float = 1e-6,
        out_of_range: str = "zero"
    ) -> Dict[int, Union[float, np.ndarray]]:
        """
        a_l(E) = int P_l(mu) f(mu, E) dmu / int f(mu, E) dmu, from the file's
        tables under their angular laws and its energy law.

        Each table is projected exactly (:func:`project_tabulated_to_legendre`)
        and the projections are interpolated under the energy law, which is the
        projection of the interpolated f(mu, E) for any law linear in y
        (see :meth:`legendre_table`). Outside the energy grid the end values
        are held: *out_of_range* is accepted for the common signature and has
        no effect, as it never had on this projection.

        Parameters
        ----------
        energy : float or array
            Energy point(s) where to evaluate a_ℓ(E)
        max_legendre_order : int
            Maximum Legendre order to compute

        Returns
        -------
        Dict[int, Union[float, np.ndarray]]
            Dictionary mapping Legendre order ℓ to coefficient values a_ℓ(E)
        """
        from ....algebra import evaluate

        scalar_input = np.isscalar(energy)
        E_arr = np.atleast_1d(np.asarray(energy, dtype=float))
        if len(self._energies) == 0:
            out = {ell: np.zeros(E_arr.shape) for ell in range(max_legendre_order + 1)}
        else:
            E, A, laws, _hold = self.legendre_table(max_legendre_order)
            held = np.clip(E_arr, E[0], E[-1])
            out = {ell: (evaluate(E, A[:, ell], laws, held) if E.size > 1
                         else np.full(E_arr.shape, A[0, ell]))
                   for ell in range(max_legendre_order + 1)}

        if trim:
            out = auto_trim_legendre_tail(out, tol=trim_tol, min_order=0)

        # Return appropriate format
        if scalar_input:
            return {ell: float(vals[0]) for ell, vals in out.items()}
        return out

    def get_distribution_at_energy(self, energy: float) -> Tuple[List[float], List[float]]:
        """
        Return the stored tabulated (μ, f) at an exact grid energy, or ([],[]) if not found.
        """
        try:
            i = self._energies.index(energy)
            return (self._cosines[i], self._probabilities[i])
        except (ValueError, IndexError):
            return ([], [])

    def get_distribution_dict(self) -> Dict[float, Tuple[List[float], List[float]]]:
        """
        Map each grid energy to its (μ, f) table.
        """
        return {e: (c, p) for e, c, p in zip(self._energies, self._cosines, self._probabilities)}
    
    def to_plot_data(
        self,
        order: int,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for tabulated distribution projected to Legendre coefficients.
        
        For tabulated distributions (LTT=2), Legendre coefficients are computed by
        projecting the tabulated f(μ,E) distributions onto Legendre polynomials,
        exactly (:meth:`extract_legendre_coefficients`).
        
        Parameters
        ----------
        order : int
            Legendre polynomial order to extract
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        LegendreCoeffPlotData
            Plot data object ready to be added to a PlotBuilder
            
        Examples
        --------
        >>> # Project tabulated distribution to Legendre coefficients
        >>> data = mf4_tabulated.to_plot_data(order=1, color='blue')
        >>> builder = PlotBuilder().add_data(data).build()
        
        Notes
        -----
        Tabulated distributions (LTT=2) store f(μ,E) at discrete (μ, E) points.
        To obtain Legendre coefficients a_ℓ(E), we compute:
        
            a_ℓ(E) = (2ℓ+1)/2 ∫_{-1}^{1} P_ℓ(μ) f(μ,E) dμ
        
        exactly on each table's own panels.
        """
        from kika.plotting import LegendreCoeffPlotData
        
        # Get energy grid from tabulated data
        energies = np.array(self._energies, dtype=float)
        
        if len(energies) == 0:
            raise ValueError("No tabulated data available to create plot")
        
        # Extract Legendre coefficients at all energy points using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=order,
            out_of_range="zero"
        )
        
        # Get the coefficient values for the requested order
        coeff_values = coeffs_dict[order]
        
        # Get isotope information
        isotope = getattr(self, 'isotope', None)
        if isotope is None and hasattr(self, 'zaid'):
            isotope = str(self.zaid)
        
        mt = getattr(self, 'number', None)

        # Auto-generate label if not provided
        if label is None:
            from kika._constants import format_plot_label
            label = format_plot_label(isotope=isotope, mt=mt, order=order)

        return LegendreCoeffPlotData(
            x=energies,
            y=coeff_values,
            order=order,
            isotope=isotope,
            mt=mt,
            energy_range=(energies.min(), energies.max()),
            label=label,
            **styling_kwargs
        )

    def to_bulk_plot_data(
        self,
        max_order: int = 12,
    ) -> Dict[str, Union[List[float], Dict[int, List[float]], int, str]]:
        """
        Extract ALL Legendre orders at once for bulk loading.

        For tabulated distributions (LTT=2), coefficients are computed by projecting
        f(μ,E) onto Legendre polynomials, exactly.

        Parameters
        ----------
        max_order : int, optional
            Maximum Legendre order to compute (default: 12)

        Returns
        -------
        dict
            Dictionary containing:
            - 'energies': list of energy values (eV)
            - 'coefficients_by_order': dict mapping order (int) to list of coefficients
            - 'max_order': maximum order extracted
            - 'isotope': isotope identifier (if available)
            - 'mt': MT reaction number
        """
        energies = np.array(self._energies, dtype=float)

        if len(energies) == 0:
            return {
                'energies': [],
                'coefficients_by_order': {},
                'max_order': 0,
                'isotope': getattr(self, 'isotope', None),
                'mt': getattr(self, 'number', None),
            }

        # Extract all Legendre coefficients at once using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=max_order,
            out_of_range="zero"
        )

        # Convert to the expected format
        coefficients_by_order: Dict[int, List[float]] = {}
        for order in range(max_order + 1):
            if order in coeffs_dict:
                vals = coeffs_dict[order]
                if isinstance(vals, np.ndarray):
                    coefficients_by_order[order] = vals.tolist()
                else:
                    coefficients_by_order[order] = [float(vals)] * len(energies)
            else:
                coefficients_by_order[order] = [0.0] * len(energies)

        return {
            'energies': energies.tolist(),
            'coefficients_by_order': coefficients_by_order,
            'max_order': max_order,
            'isotope': getattr(self, 'isotope', None),
            'mt': getattr(self, 'number', None),
        }

    def __str__(self) -> str:
        """
        Convert the MF4MTTabulated object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        line_num = 1
        
        # Format first line - header - ZA, AWR as float, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        line_num += 1
        
        # Format second line - First value as float, rest as integers
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with number of interpolation regions and energy points
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, self._ne or 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Format energy interpolation scheme pairs - all as integers
        if self._interpolation and self._nr and self._nr > 0:
            # Process interpolation pairs in groups of 3 (6 values per line)
            remaining_pairs = self._interpolation.copy()
            while remaining_pairs:
                # Take up to 3 pairs for this line
                line_pairs = remaining_pairs[:3]
                remaining_pairs = remaining_pairs[3:]
                
                # Flatten pairs into a list of values
                values = []
                for nbt, interp in line_pairs:
                    values.append(nbt)
                    values.append(interp)
                
                # All interpolation values are integers
                # Format line - pad with blanks instead of zeros
                formats = [ENDF_FORMAT_INT] * len(values)
                if len(values) < 6:
                    values.extend([None] * (6 - len(values)))
                    formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                    
                interp_line = format_endf_data_line(
                    values, mat, mf, mt, line_num, formats=formats
                )
                lines.append(interp_line)
                line_num += 1
        
        # Format each tabulated distribution
        for i, energy in enumerate(self._energies):
            cosines = self._cosines[i]
            probabilities = self._probabilities[i]
            ang_interp = self._angular_interpolation[i] if i < len(self._angular_interpolation) else []
            
            np_val = len(cosines)  # Number of angular points
            nr_ang = len(ang_interp)  # Number of angular interpolation regions
            
            # Format header for this energy
            energy_header = format_endf_data_line(
                [0.0, energy, 0, 0, nr_ang, np_val],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(energy_header)
            line_num += 1
            
            # Format angular interpolation scheme pairs (if any)
            if ang_interp and nr_ang > 0:
                # Process interpolation pairs in groups of 3 (6 values per line)
                remaining_pairs = ang_interp.copy()
                while remaining_pairs:
                    # Take up to 3 pairs for this line
                    line_pairs = remaining_pairs[:3]
                    remaining_pairs = remaining_pairs[3:]
                    
                    # Flatten pairs into a list of values
                    values = []
                    for nbt, interp in line_pairs:
                        values.append(nbt)
                        values.append(interp)
                    
                    formats = [ENDF_FORMAT_INT] * len(values)
                    # Use None for blank fields instead of zeros
                    if len(values) < 6:
                        values.extend([None] * (6 - len(values)))
                        formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                    
                    # Format line
                    ang_interp_line = format_endf_data_line(
                        values, mat, mf, mt, line_num,
                        formats=formats
                    )
                    lines.append(ang_interp_line)
                    line_num += 1
            
            # Format cosine-probability pairs (3 pairs per line)
            pair_idx = 0
            while pair_idx < np_val:
                # Get up to 3 pairs for this line
                line_values = []
                for j in range(3):
                    if pair_idx + j < np_val:
                        line_values.append(cosines[pair_idx + j])
                        line_values.append(probabilities[pair_idx + j])
                pair_idx += 3
                
                # Use None for blank fields instead of zeros
                if len(line_values) < 6:
                    line_values.extend([None] * (6 - len(line_values)))
                
                # Format line
                pair_line = format_endf_data_line(
                    line_values, mat, mf, mt, line_num
                )
                lines.append(pair_line)
                line_num += 1
        
        # SEND record: C1=0.0, C2=0.0, L1=0, L2=0, N1=0, N2=0, MT=0
        end_line = format_endf_data_line(
            [0.0, 0.0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
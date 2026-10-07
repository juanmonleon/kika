"""
Classes for MT sections within MF4 (Angular Distributions) in ENDF files.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from ..mt import MT


@dataclass
class MF4MT(MT):
    """
    Base class for MT sections within MF4 (Angular Distributions).
    
    This class provides a common interface for all angular distribution formats.
    """
    _za: float = None    # ZA identifier
    _awr: float = None   # Atomic weight ratio
    _ltt: int = None     # Angular distribution format flag
    _li: int = None      # Flag to specify if angular distributions are isotropic (0=not all isotropic, 1=all isotropic)
    _lct: int = None     # Frame of reference (1=LAB, 2=CM)
    _mat: int = None     # Material identifier
    
    # Line count
    num_lines: int = 0  # Number of lines in this MT section
    
    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z+A)"""
        return self._za
    
    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio"""
        return self._awr
    
    @property
    def ltt(self) -> int:
        """LTT: 0 isotropic, 1 Legendre, 2 tabulated, 3 mixed.

        The numeric flag behind :attr:`type`. Exposed because a consumer often
        needs to branch on the representation rather than describe it — whether
        ``extract_legendre_coefficients`` interpolates per order or has to
        project tabulated distributions onto Legendre first, for instance.
        """
        return self._ltt

    @property
    def type(self) -> str:
        """Angular distribution format flag"""
        dist_type = ''
        if self._ltt == 0: dist_type = 'Isotropic'
        elif self._ltt == 1: dist_type = 'Legendre'
        elif self._ltt == 2: dist_type = 'Tabulated'
        elif self._ltt == 3: dist_type = 'Legendre and Tabulated'
        else:
            raise ValueError(f"Invalid LTT value: {self._ltt}. Expected 0, 1, 2, or 3.")
        return dist_type
    
    @property
    def is_isotropic(self) -> bool:
        """Flag for identical particles (0=not all isotropic, 1=all isotropic)"""
        if self._li == 0:
            return False
        elif self._li == 1:
            return True
        else:
            raise ValueError(f"Invalid value for LI: {self._li}. Expected 0 or 1.")
    
    @property
    def frame(self) -> str:
        """Frame of reference (1=LAB system, 2=CM system)"""
        if self._lct == 1: 
            return "LAB"
        elif self._lct == 2:
            return "CM"
        else:
            raise ValueError(f"Invalid value for LCT: {self._lct}. Expected 1 or 2.")
    
    def evaluate_angular_pdf(
        self,
        mu,
        energy,
        *,
        out_of_range: str = "zero",
    ) -> np.ndarray:
        r"""The angular distribution :math:`f(\mu, E)` itself.

        Returns an array of shape ``(n_energies, n_mu)``.

        The observable every MF4 view is really after.  Legendre coefficients
        are one *representation* of it, and only some evaluations use that one:
        this method is the single question all four classes can answer, each in
        the way its own representation makes exact.

        The default here rebuilds it from the coefficients the section carries,
        which *is* exact for LTT=0 and LTT=1 -- those files store the expansion,
        so summing it back is reading them.  The classes that store
        :math:`f(\mu)` as a table override this and read the table, instead of
        projecting it onto Legendre and summing the projection back.
        """
        from kika.endf.dcs import angular_pdf, max_legendre_order

        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        e_arr = np.atleast_1d(np.asarray(energy, dtype=float))

        order = max(1, max_legendre_order(self))
        coeffs = self.extract_legendre_coefficients(
            e_arr, max_legendre_order=order, out_of_range=out_of_range
        )
        # Not every class returns every order it was asked for: the mixed
        # class trims its tail by default, so the orders present are whatever
        # came back, and a missing one is a zero rather than a hole.
        present = [l for l in coeffs if isinstance(l, (int, np.integer)) and l >= 1]
        top = max(present) if present else 0

        out = np.empty((e_arr.size, mu_arr.size), dtype=float)
        for k in range(e_arr.size):
            # a_0 is the normalization and is not part of the a_1..a_L the
            # reconstruction takes.
            a = [
                float(np.atleast_1d(coeffs[l])[k]) if l in coeffs else 0.0
                for l in range(1, top + 1)
            ]
            out[k] = angular_pdf(mu_arr, a)
        return out

    def native_cosine_grid(self, energy) -> Optional[np.ndarray]:
        r"""The cosines this section stores at *energy*, or None if it stores none.

        A section that tabulates :math:`f(\mu)` has a preferred grid and it is
        not a uniform one: evaluators put their points where the distribution
        turns, which for a forward-peaked elastic means a cluster against
        :math:`\mu = 1`.  Resampling that onto a uniform grid throws the peak
        away -- 45 % low at 18 MeV for JEFF-4.0 U-235 elastic against a
        200-point uniform grid, which is worse than the truncated expansion it
        would be replacing.  So the grid travels with the values.

        None for a section stored as an expansion, which is defined at every
        cosine and has no grid of its own to prefer.
        """
        return None

    def legendre_table(self, max_order: int):
        r"""``(E, A, laws, hold)``: :math:`a_l(E)` as one table in incident energy.

        ``A`` has shape ``(len(E), max_order + 1)``, ``laws`` is one ENDF code
        per interval of ``E`` and ``hold`` says whether the coefficients keep
        their end values outside ``E`` (otherwise they are zero there). Read
        under ``laws`` this table **is** what :meth:`extract_legendre_coefficients`
        returns at any energy, so an integral or an extremum of :math:`a_l` over
        an energy cell is a property of the table and needs no sampling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not state its a_l(E) as a table")

    def _held_legendre_table(self, max_order: int, lo: float, hi: float):
        """:meth:`legendre_table` with a held table extended flat to ``[lo, hi]``."""
        E, A, laws, hold = self.legendre_table(max_order)
        E = np.asarray(E, dtype=float)
        A = np.asarray(A, dtype=float).reshape(E.size, max_order + 1)
        laws = np.asarray(laws, dtype=np.int64).reshape(max(E.size - 1, 0))
        if hold and E.size:
            if lo < E[0]:
                E, A, laws = np.r_[lo, E], np.vstack([A[:1], A]), np.r_[2, laws]
            if hi > E[-1]:
                E, A, laws = np.r_[E, hi], np.vstack([A, A[-1:]]), np.r_[laws, 2]
        return E, A, laws

    def legendre_cell_averages(self, edges, max_order: int):
        r"""``{l: <a_l>}`` over every cell of *edges*, exactly.

        The integral of the table :meth:`legendre_table` states, under its own
        laws (:func:`kika.algebra.group_averages`), divided by the cell width --
        not a quadrature of :meth:`extract_legendre_coefficients`. Five points
        per cell missed a resonance in a_1 by 0.13 on JEFF-4.0 Fe-56 elastic.
        A cell of zero width is ``nan``.
        """
        from ....algebra import group_averages

        edges = np.asarray(edges, dtype=float)
        E, A, laws = self._held_legendre_table(max_order, edges[0], edges[-1])
        if E.size < 2:
            return {l: np.zeros(edges.size - 1) for l in range(max_order + 1)}
        return {l: group_averages(E, A[:, l], laws, edges)
                for l in range(max_order + 1)}

    def legendre_cell_min_abs(self, edges, order: int) -> np.ndarray:
        r"""The smallest :math:`|a_l(E)|` inside every cell of *edges*.

        Each of the five laws is monotone on a panel, so the extremes of a_l on
        a cell are at its edges and at the table's own energies inside it -- both
        limits at a step -- and a sign change inside the cell makes the minimum
        zero.
        """
        from ....algebra import left_limit, right_limit

        edges = np.asarray(edges, dtype=float)
        E, A, laws = self._held_legendre_table(order, edges[0], edges[-1])
        out = np.zeros(edges.size - 1)
        if E.size < 2:
            return out
        y = A[:, order]
        q = np.union1d(edges, E[(E > edges[0]) & (E < edges[-1])])
        below, above = left_limit(E, y, laws, q), right_limit(E, y, laws, q)
        at = np.searchsorted(q, edges)
        for c in range(edges.size - 1):
            i, j = at[c], at[c + 1]
            seen = np.r_[above[i], below[i + 1:j], above[i + 1:j], below[j]]
            out[c] = 0.0 if seen.min() < 0.0 < seen.max() else np.abs(seen).min()
        return out

    def to_dense_plot_data(
        self,
        order: int,
        energy_range: Tuple[float, float],
        num_points: int = 1000,
        label: Optional[str] = None,
        **styling_kwargs,
    ):
        """
        Evaluate Legendre coefficients on a dense log-spaced grid and return plot data.

        This method works on any MF4MT subclass that implements
        ``extract_legendre_coefficients()``.

        Parameters
        ----------
        order : int
            Legendre polynomial order to extract.
        energy_range : tuple of float
            ``(e_min, e_max)`` in eV for the dense evaluation grid.
        num_points : int, default 1000
            Number of log-spaced evaluation points.
        label : str, optional
            Plot label.  Auto-generated if *None*.
        **styling_kwargs
            Passed through to ``LegendreCoeffPlotData`` (color, linestyle, …).

        Returns
        -------
        LegendreCoeffPlotData
        """
        from kika.plotting import LegendreCoeffPlotData

        e_min, e_max = energy_range
        dense_e = np.logspace(np.log10(e_min), np.log10(e_max), num_points)

        coeffs_dict = self.extract_legendre_coefficients(
            energy=dense_e,
            max_legendre_order=order,
            out_of_range="zero",
        )
        coeff_vals = coeffs_dict[order]

        isotope = getattr(self, "isotope", None)
        if isotope is None and hasattr(self, "zaid"):
            isotope = str(self.zaid)
        mt = getattr(self, "number", None)

        if label is None:
            label = f"ENDF L={order}"

        return LegendreCoeffPlotData(
            x=dense_e,
            y=coeff_vals,
            order=order,
            isotope=isotope,
            mt=mt,
            energy_range=energy_range,
            label=label,
            plot_type="line",
            **styling_kwargs,
        )

    def __str__(self) -> str:
        """
        Convert the MF4MT object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        
        # Format first line - header
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, 1,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        
        return "\n".join(lines)











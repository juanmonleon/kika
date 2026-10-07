"""MF35: covariances of energy distributions (ENDF-6 §35).

One HEAD record and NK subsections, each a LIST covering one **incident-energy
band** and holding an absolute covariance over an outgoing-energy grid.

**What the matrix is the covariance of, and why it matters here.** Measured on
ENDF/B-VIII.1 U-235, JEFF-4.0 U-235 and ENDF/B-VIII.1 Cf-252: the unweighted row
sums vanish (``max|Σ_j C_ij| / max|C|`` between 1.4e-7 and 3.1e-3) while the
dE-weighted row sums do not, and ``sqrt(diag C)/P`` is a few per cent while
``sqrt(diag C)/(P/dE)`` is order 1e+3. So the entries are the covariance of the
**group-integrated probabilities** ``P_i = ∫_{g_i}^{g_i+1} χ(E→E') dE'``, not of
the spectrum density. ``Σ_i P_i = 1`` is what forces ``C·1 ≈ 0``.

Everything downstream rests on that reading, so
:func:`row_sum_residual` is provided here and the sampler refuses a band that
fails it rather than quietly perturbing something else.

**Only LS=1, LB=7 exists.** All three tapes carry that and nothing else. MF33's
LB decoders are not reused: the record shape is close, but they are welded to
MF33's NC/NI subsection hierarchy, which MF35 does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    format_data_values,
    format_endf_data_line,
    format_endf_send_record,
)


@dataclass
class MF35SubSection:
    """One incident-energy band: ``E1, E2, LS, LB, NT, NE`` and the LIST body.

    The body is NE outgoing-energy boundaries followed by the upper triangle
    (diagonal included) of a symmetric ``(NE-1) × (NE-1)`` matrix, so
    ``NT = NE + NE(NE-1)/2``.
    """

    e1: float = 0.0
    e2: float = 0.0
    ls: int = 1
    lb: int = 7
    nt: int = 0
    ne: int = 0
    boundaries: List[float] = field(default_factory=list)
    upper_triangle: List[float] = field(default_factory=list)
    #: The LIST body exactly as read, kept so the section can be written back
    #: rather than rebuilt from the densified matrix. Mirrors MF33's
    #: ``NISubSubsectionRecord``: the values that reach the writer are the
    #: values that came off the tape, in that order.
    raw_list_values: List[float] = field(default_factory=list)

    @property
    def order(self) -> int:
        """``NE - 1`` — the number of groups, and the matrix dimension."""
        return max(self.ne - 1, 0)

    @staticmethod
    def expected_nt(ne: int) -> int:
        return ne + ne * (ne - 1) // 2

    def matrix(self) -> np.ndarray:
        """Densify the stored upper triangle into a symmetric matrix.

        Fifteen lines, and the one place a transposition or an off-by-one in
        the triangle ordering could hide, so it is unit-tested against a
        hand-built 3×3 rather than only against real tapes.
        """
        n = self.order
        out = np.zeros((n, n), dtype=float)
        values = np.asarray(self.upper_triangle, dtype=float)
        cursor = 0
        for row in range(n):
            width = n - row
            out[row, row:] = values[cursor:cursor + width]
            cursor += width
        return out + np.triu(out, 1).T

    def energy_grid(self) -> np.ndarray:
        return np.asarray(self.boundaries, dtype=float)

    def correlation(self) -> np.ndarray:
        """``C_ij / (σ_i σ_j)``, NaN where a variance is zero (or clipped to it)."""
        matrix = self.matrix()
        sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
        outer = np.outer(sigma, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(outer > 0.0, matrix / outer, np.nan)

    def to_heatmap_data(self, matrix_type: str = "corr", scale: str = "log",
                        relative_pct: Optional[np.ndarray] = None,
                        mt: Optional[int] = None, zaid: Optional[int] = None,
                        energy_range: Optional[Tuple[float, float]] = None,
                        label: Optional[str] = None):
        """This band as :class:`~kika.plotting.plot_data.CovarianceHeatmapData`.

        Laid out exactly as ``CrossSectionCovariance.to_heatmap_data`` lays out
        one MF33 reaction: edges in plot coordinates (``_log_edges`` on a log
        scale), a one-block ``block_info``, and cropping by *energy_range*. The
        builder then takes the same path for MF35 as for MF33, which is what
        gives it energy ticks and every option of the heatmap panel; a bare
        matrix with only an ``energy_grid`` drew neither ticks nor limits.

        One block, never several bands: their orders differ, so there is no
        common group count to lay blocks out on. The panel above the matrix
        takes *relative_pct* (from :meth:`MF35MT.relative_uncertainty`) and
        nothing else: ``sqrt(diag C)`` alone is in units of a group probability,
        and drawing it on a per-cent axis would mislabel it.

        ENDF/B-VIII U-235 bands 1-4 start at E' = 0. On a log scale that edge is
        drawn one decade below the next (``_log_edges``), and ``energy_grid``
        carries the same value so the energy ticks agree with the cells.
        """
        from kika.cov.cross_section_covariance import _log_edges
        from kika.plotting.plot_data import CovarianceHeatmapData

        if matrix_type not in ("corr", "cov"):
            raise ValueError("matrix_type must be 'corr' or 'cov'")
        if scale not in ("log", "linear"):
            raise ValueError("scale must be 'log' or 'linear'")
        matrix = self.correlation() if matrix_type == "corr" else self.matrix()
        edges = self.energy_grid().astype(float)

        keep = np.ones(len(edges) - 1, dtype=bool)
        if energy_range is not None:
            emin, emax = (float(v) for v in energy_range)
            if not emin < emax:
                raise ValueError("energy_range must be (emin, emax) with emin < emax")
            keep = (edges[1:] > emin) & (edges[:-1] < emax)
            if not keep.any():
                raise ValueError("energy_range removed all groups; nothing to plot")
        first, last = np.where(keep)[0][[0, -1]]
        edges = edges[first:last + 2]
        matrix = matrix[first:last + 1, first:last + 1]

        if matrix_type == "cov":
            # Zero-variance groups render grey, as they do for MF33.
            std = np.sqrt(np.abs(np.diag(matrix)))
            top = np.nanmax(std) if np.any(np.isfinite(std)) else 1.0
            dead = ~np.isfinite(std) | (std < (top * 1e-12 if top > 0 else 1e-30))
            if dead.any():
                matrix = matrix.copy()
                matrix[dead, :] = np.nan
                matrix[:, dead] = np.nan

        if scale == "log":
            plot_edges = _log_edges(edges)
            positive = edges[edges > 0]
            if positive.size and edges[0] <= 0.0:
                edges = edges.copy()
                edges[0] = positive.min() / 10.0
        else:
            plot_edges = edges.copy()
        x_edges = plot_edges - plot_edges[0]
        width = float(x_edges[-1])

        key = int(mt) if mt is not None else 0
        uncertainty = None
        if relative_pct is not None:
            pct = np.asarray(relative_pct, dtype=float)
            uncertainty = {key: pct[first:last + 1] if pct.size == len(keep) else pct}

        if label is None:
            kind = "Correlation" if matrix_type == "corr" else "Covariance"
            label = f"MF35 MT:{key} [{self.e1:.3g}, {self.e2:.3g}] eV {kind}"
        return CovarianceHeatmapData(
            matrix_data=matrix,
            matrix_type=matrix_type,
            zaid=zaid,
            block_info={
                "mts": [key],
                "G": int(len(edges) - 1),
                "ranges": [(0.0, width)],
                "energy_ranges": {key: (0.0, width)},
            },
            uncertainty_data=uncertainty,
            energy_grid=edges,
            mt_labels=[str(key)],
            is_diagonal=True,
            mask_value=0.0 if matrix_type == "corr" else None,
            scale=scale,
            x_edges=x_edges,
            y_edges=x_edges.copy(),
            extent=(0.0, width, 0.0, width),
            label=label,
            colorbar_label="Correlation" if matrix_type == "corr" else "Covariance",
        )

    # ------------------------------------------------------------------
    def row_sum_residual(self) -> float:
        """``max_i |Σ_j C_ij| / max|C|`` — how close ``C·1`` is to zero.

        The test of the group-probability reading. Measured between 1.4e-7 and
        3.1e-3 on the three reference tapes; JEFF-4.0 U-235's band 0 is the
        3.1e-3 outlier and sets the tolerance anyone downstream should use.
        """
        matrix = self.matrix()
        if matrix.size == 0:
            return 0.0
        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            return 0.0
        return float(np.max(np.abs(matrix.sum(axis=1))) / scale)

    def normalisation_drift(self) -> float:
        """``sqrt(1ᵀC1)`` — the standard deviation of a draw's sum-rule drift.

        The budget a linear-space draw has to stay inside before the projection
        closes it. Measured 9.5e-7 … 3.2e-5 on the reference tapes: physically
        negligible, ~75× the input tapes' own residual, and therefore something
        to close explicitly rather than assume away.

        ``1ᵀC1`` comes out slightly *negative* on about half the bands — it is
        a near-cancelling sum of ~4e5 terms, so the rounding noise straddles
        zero. The magnitude is reported rather than clamped to 0.0, because a
        band whose budget prints as exactly zero reads as "no drift is
        possible here", which is not what was measured.
        """
        return float(np.sqrt(abs(float(self.matrix().sum()))))

    def emit(self, mat: int, mf: int, mt: int, line_num: int):
        """The LIST record: header then NT values, six per line."""
        values = (self.raw_list_values
                  if self.raw_list_values
                  else list(self.boundaries) + list(self.upper_triangle))
        header = format_endf_data_line(
            [self.e1, self.e2, self.ls, self.lb, len(values), self.ne],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        line_num += 1
        body, line_num = format_data_values(list(values), mat, mf, mt, line_num)
        return [header] + body, line_num


@dataclass
class MF35RelativeUncertainty:
    """One band's diagonal, read against the spectrum it covaries.

    ``relative`` is a fraction (not per cent) and is ``None`` when the band
    could not be paired with an MF5 spectrum; ``warnings`` then says why. A
    band is still worth showing in absolute terms in that case, which is why
    this is a result and not an exception.
    """

    mt: int
    band_index: int
    e1: float
    e2: float
    boundaries: np.ndarray
    sigma: np.ndarray
    incident_energy: Optional[float] = None
    #: True when ``probabilities`` are the file's interpolant at exactly
    #: ``incident_energy``; False when it had to fall back to an MF5 node.
    exact: bool = False
    probabilities: Optional[np.ndarray] = None
    relative: Optional[np.ndarray] = None
    incident_nodes: List[float] = field(default_factory=list)
    #: ``(E1, E2)`` of every band of the section, so a caller can say which
    #: one an energy falls in without a second call.
    bands: List[Tuple[float, float]] = field(default_factory=list)
    #: Every MF5 node inside the section's bands, plus the band edges: the
    #: incident energies at which the answer changes, across all bands.
    incident_grid: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _tabulated_partial(mf5_section, mt: int):
    """The single LF=1 partial of *mf5_section*, or ``(None, reason)``.

    MF35 covaries group integrals of a *tabulated* spectrum, so the analytic
    laws have nothing to be relative to, and with two LF=1 partials the file
    does not say which one the band belongs to.
    """
    if mf5_section is None:
        return None, f"there is no MF5/MT{mt}, so MF35/MT{mt} has no spectrum to be relative to"
    tabulated = mf5_section.tabulated_partials()
    if not tabulated:
        laws = ",".join(str(p.lf) for p in mf5_section.partials)
        return None, (f"MF5/MT{mt} carries no LF=1 subsection (LF=[{laws}]), and "
                      f"the group probabilities MF35 is relative to are an "
                      f"integral of a tabulated spectrum")
    if len(tabulated) > 1:
        return None, (f"MF5/MT{mt} has {len(tabulated)} LF=1 subsections; which "
                      f"of them MF35's band is relative to is not stated")
    return tabulated[0][1], None


@dataclass
class MF35MT(MT):
    """One MT section of MF35: NK bands of one reaction's spectrum covariance."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _nk: Optional[int] = None
    _mat: Optional[int] = None
    _mf: int = 35
    subsections: List[MF35SubSection] = field(default_factory=list)

    @property
    def num_bands(self) -> int:
        return self._nk if self._nk is not None else len(self.subsections)

    def band_for_incident(self, energy: float) -> Optional[int]:
        """Index of the band containing *energy*, or None.

        Bands are half-open ``[E1, E2)`` so that a band edge — which, measured,
        is always also an MF5 incident node — belongs to exactly one band. The
        last band is closed at the top, otherwise the highest incident energy
        on the tape would fall outside every band.
        """
        last = len(self.subsections) - 1
        for index, band in enumerate(self.subsections):
            if band.e1 <= energy < band.e2:
                return index
            if index == last and energy == band.e2:
                return index
        return None

    def relative_uncertainty(
        self,
        mf5_section=None,
        incident_energy: Optional[float] = None,
        band_index: Optional[int] = None,
    ) -> MF35RelativeUncertainty:
        """``sqrt(C_ii) / P_i`` of one band, with ``P_i`` from *mf5_section*.

        The band is *band_index* if given, otherwise the one containing
        *incident_energy* (:meth:`band_for_incident`). ``P_i`` is the group
        integral of the MF5 spectrum at *incident_energy*, defaulting to the
        band's lower edge, which is measured to be an MF5 node on every tape.

        Comparing libraries is the reason for selecting by energy: their bands
        cover different ranges, so "band 2" means something different on each
        tape while "1 MeV" does not. The matrix is the same across the band,
        but ``P_i`` is not, which is why the energy is part of the answer.
        """
        if band_index is None:
            if incident_energy is None:
                raise ValueError("give a band_index or an incident_energy")
            band_index = self.band_for_incident(float(incident_energy))
            if band_index is None:
                spans = ", ".join(f"[{b.e1:.4e}, {b.e2:.4e}]" for b in self.subsections)
                raise ValueError(
                    f"incident energy {float(incident_energy):.6e} eV is outside "
                    f"every MF35/MT{self.number} band ({spans})"
                )
        if not 0 <= band_index < len(self.subsections):
            raise IndexError(
                f"MF35/MT{self.number} has {len(self.subsections)} band(s); "
                f"index {band_index} is out of range"
            )
        band = self.subsections[band_index]

        variance = np.diag(band.matrix())
        result = MF35RelativeUncertainty(
            mt=int(self.number), band_index=int(band_index),
            e1=float(band.e1), e2=float(band.e2),
            boundaries=band.energy_grid(),
            sigma=np.sqrt(np.clip(variance, 0.0, None)),
            bands=[(float(b.e1), float(b.e2)) for b in self.subsections],
        )
        edges = {e for span in result.bands for e in span}
        result.incident_grid = sorted(edges)
        if np.any(variance < 0.0):
            result.warnings.append(
                f"{int(np.sum(variance < 0.0))} of {variance.size} diagonal entries "
                f"are negative and were clipped to zero before the square root"
            )

        partial, reason = _tabulated_partial(mf5_section, int(self.number))
        if partial is None:
            result.warnings.append(f"no relative uncertainty: {reason}")
            return result

        result.incident_nodes = [float(e) for e in partial.incident_energies
                                 if band.e1 <= e <= band.e2]
        lo, hi = min(edges), max(edges)
        result.incident_grid = sorted(edges | {
            float(e) for e in partial.incident_energies if lo <= e <= hi})
        energy = float(incident_energy) if incident_energy is not None else float(band.e1)
        try:
            probabilities = partial.group_integrals_at(energy, result.boundaries)
            result.exact = True
        except NotImplementedError as exc:
            if not result.incident_nodes:
                result.warnings.append(f"no relative uncertainty: {exc}")
                return result
            nearest = min(result.incident_nodes, key=lambda e: abs(e - energy))
            result.warnings.append(
                f"{exc}; used the MF5 node {nearest:.6e} eV instead of {energy:.6e} eV")
            energy = nearest
            k = list(partial.incident_energies).index(nearest)
            probabilities = partial.group_integrals(k, result.boundaries)

        probabilities = np.asarray(probabilities, dtype=float)
        result.incident_energy = energy
        result.probabilities = probabilities
        with np.errstate(divide="ignore", invalid="ignore"):
            result.relative = np.where(
                probabilities > 0.0, result.sigma / probabilities, np.nan)
        return result

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        line_num = 1
        lines = [format_endf_data_line(
            [self._za, self._awr, 0, 0, self.num_bands, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )]
        line_num += 1

        for band in self.subsections:
            band_lines, line_num = band.emit(mat, mf, mt, line_num)
            lines.extend(band_lines)

        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        sizes = ",".join(str(b.ne) for b in self.subsections)
        return f"MF35MT({self.number}, NK={self.num_bands}, NE=[{sizes}])"

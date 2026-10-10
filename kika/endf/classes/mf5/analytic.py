"""The MF5 laws given by parameters rather than by a table (ENDF-6 §5.1.1).

LF=1 writes chi(E->E') out point by point; LF=5, 7, 9, 11 and 12 write a
handful of energy-dependent parameters and leave the shape to a formula. This
module reads those five. The formulae themselves are
:mod:`kika.nuclear_data.spectrum_laws`'s, shared with the model's §18.3 nodes, so the
reader and the model cannot evaluate two different spectra.

**Decoded for reading, emitted from bytes.** Every class here subclasses
:class:`~kika.endf.classes.mf5.partials.MF5PartialRaw` and so keeps
``raw_lines`` and inherits its :meth:`emit`. The decode is purely additive: the
records that go back onto the tape are the records that came off it, which is
what makes this change unable to disturb the byte gate in
``test_mf5_roundtrip.py``. Reconstructing the TAB1s on the way out would have
re-opened the ``format_interp_pairs`` padding defect that ENDF/B-VIII.1 already
pins as a strict xfail.

**One normalisation rule, not four.** ENDF-6 §5.1 requires
``int_0^(E-U) f(E->E') dE' = 1`` of every law, so each class here states only an
unnormalised *shape* and the closed form of ``I(E) = int_0^(E-U) shape dE'``,
and the division happens once in the base class. That is not tidiness: it is
what makes LF=5 unambiguous. The manual writes LF=5 as ``f = g(E'/theta)``, and
that reading cannot hold ``int f dE' = 1`` for all E when a single g is paired
with an energy-dependent theta -- ``f = g(x)/theta`` is the reading that can.
Normalising by the measured integral of the shape satisfies §5.1 under either,
and reduces to ``f = g`` exactly on the one witness available here (Cf-252
MT455, where theta == 1 and ``int g dx == 1``), so nothing has to be guessed.

**LF=12 (Madland-Nix) is here since 2026-10-08.** The reason it was kept out
-- "it needs a numerical double integral" -- was wrong: §5.1.1.6's density is
closed in E1 and the lower incomplete gamma function. What there still is not
is a tape on this machine to close it against (ENDF/B-VII.1 Am-241 and
JEFF-3.1.1 are on the cluster), so it is gated on its own two closed forms:
the density integrates to 1 and its first moment is ``(EFL+EFH)/2 + 4/3 T_M``,
to 1e-13. Its EFL and EFH are the C1/C2 of the T_M TAB1's header, which the
walker used to throw away. It has no ``U`` and is not truncated at ``E - U``.

**What is witnessed and what is not.** LF=5 is read off a committed fixture
(``micro_cf252_pfns.endf``, MT455, six subsections). LF=7, 9 and 11 have no tape
here: their record *layout* is the one the walker has always used, and their
formulae are gated against numerical quadrature of their own shape, which
catches a typo in either the shape or its normalisation constant but cannot
catch a misreading of the manual. See ``test_mf5_analytic.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ....algebra import evaluate, integral, interval_laws
from ....nuclear_data import spectrum_laws as spectra
from .partials import MF5PartialRaw

#: Points in the display grid a law builds for itself when the caller does not
#: supply one. Only ever a rendering choice -- an analytic law has no grid of
#: its own, so no result depends on this beyond how smooth the curve looks.
DEFAULT_GRID_POINTS = 400

#: How far below the upper bound the default log grid starts. Nine decades is
#: enough to show a Maxwellian's rise and cheap enough not to think about.
_DEFAULT_GRID_DECADES = 9.0


def tab1_at(x: Sequence[float], y: Sequence[float],
            interp: Sequence[Tuple[int, int]], at: float) -> float:
    """A TAB1 evaluated at one abscissa, under its own INT codes.

    Outside the table the end value is held. Every one of the five laws is
    evaluated, and integrated, in closed form by :mod:`kika.algebra`.
    """
    xs = np.asarray(x, dtype=float)
    if xs.size == 0:
        raise ValueError("cannot evaluate an empty TAB1")
    return float(evaluate(xs, y, interval_laws(xs.size, list(interp)), float(at),
                          outside="hold"))


@dataclass
class MF5PartialAnalytic(MF5PartialRaw):
    """Base of the laws stated by parameters: shape over its own integral.

    Subclasses supply :meth:`shape` and :meth:`normalisation_at`; everything a
    caller touches is here, and has the same signatures as
    :class:`~kika.endf.classes.mf5.partials.MF5PartialTabulated` so that code
    drawing a spectrum never has to ask which law it is holding.
    """

    @property
    def is_decoded(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def upper_bound(self, energy: float) -> float:
        """``E - U``: the largest outgoing energy this law admits.

        ``U`` is often negative -- Cf-252's delayed spectra write -30 MeV -- so
        this is not a truncation of the incident energy but a genuine bound of
        its own.
        """
        return float(energy) - float(self.u)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        """The law's unnormalised shape at outgoing energies *e_out*."""
        raise NotImplementedError

    def normalisation_at(self, energy: float) -> float:
        """``I(E) = int_0^(E-U) shape dE'`` -- in closed form where there is one."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def evaluate_on_grid(self, energy: float,
                         points: Sequence[float]) -> np.ndarray:
        """chi(E->E') at *energy*, on a grid the caller chose.

        Zero outside ``[0, E - U]``: the bound is part of the law, not a
        plotting range, and a Maxwellian continued past it would integrate to
        more than one.
        """
        points = np.asarray(points, dtype=float)
        hi = self.upper_bound(energy)
        out = np.zeros(points.shape, dtype=float)
        if hi <= 0.0:
            return out
        norm = self.normalisation_at(energy)
        if not np.isfinite(norm) or norm <= 0.0:
            return out
        inside = (points >= 0.0) & (points <= hi)
        if np.any(inside):
            out[inside] = self.shape(energy, points[inside]) / norm
        return out

    def default_grid(self, energy: float,
                     n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
        """A display grid over ``[0, E - U]``, log-spaced with 0 kept.

        Log because every consumer of a fission spectrum plots it log-log, and
        the interesting decade is the low-energy rise, which a linear grid of
        the same size renders as a single segment.
        """
        hi = self.upper_bound(energy)
        if hi <= 0.0:
            return np.zeros(0, dtype=float)
        lo = hi * 10.0 ** (-_DEFAULT_GRID_DECADES)
        return np.concatenate(([0.0], np.geomspace(lo, hi, max(n_points - 1, 2))))

    def evaluate_at_incident(
        self, energy: float, e_out: Optional[Sequence[float]] = None,
        n_points: int = DEFAULT_GRID_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(E' grid, chi)`` at *energy*.

        Same return shape as
        :meth:`~kika.endf.classes.mf5.partials.MF5PartialTabulated.evaluate_at_incident`.
        The extra arguments are optional because an analytic law has no grid of
        its own to hand back, so one has to be chosen; a caller with a grid in
        mind passes it.
        """
        grid = (self.default_grid(energy, n_points) if e_out is None
                else np.asarray(e_out, dtype=float))
        return grid, self.evaluate_on_grid(energy, grid)

    def normalisation(self, energy: float) -> float:
        """``int chi dE'`` over ``[0, E - U]`` -- 1 by construction.

        Kept, rather than dropped as a tautology, because it is the cheap check
        that the closed form in :meth:`normalisation_at` and the shape it is
        supposed to normalise have not drifted apart. Anything displaying it
        next to an LF=1 partial's own normalisation is comparing like with like.
        """
        hi = self.upper_bound(energy)
        if hi <= 0.0:
            return 0.0
        norm = self.normalisation_at(energy)
        return 1.0 if np.isfinite(norm) and norm > 0.0 else 0.0

    def normalisation_at_incident(self, energy: float) -> float:
        """The same number as :meth:`normalisation`, under the name LF=1 uses.

        A caller holding a subsection should not have to ask which law it is to
        ask what it integrates to. The tabulated partial needs two methods --
        one by node index for the sampler, one by energy -- and this is the
        second name, so the by-energy one is spelled the same on both.
        """
        return self.normalisation(energy)


@dataclass
class MF5GeneralEvaporation(MF5PartialAnalytic):
    """LF=5: a tabulated shape ``g(x)`` in the reduced variable ``x = E'/theta(E)``."""

    lf: int = 5
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)
    g_interp: List[Tuple[int, int]] = field(default_factory=list)
    g_x: List[float] = field(default_factory=list)
    g_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def _g_codes(self) -> np.ndarray:
        return interval_laws(len(self.g_x), self.g_interp)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        theta = self.theta(energy)
        if theta <= 0.0:
            return np.zeros(np.shape(e_out), dtype=float)
        x = np.asarray(e_out, dtype=float) / theta
        gx = np.asarray(self.g_x, dtype=float)
        gy = np.asarray(self.g_values, dtype=float)
        if gx.size < 2:
            return np.zeros(np.shape(e_out), dtype=float)
        return evaluate(gx, gy, self._g_codes(), x) / theta

    def normalisation_at(self, energy: float) -> float:
        """``int_0^(E-U) g(E'/theta)/theta dE'`` = ``int_0^{(E-U)/theta} g dx``.

        Integrated exactly over g's own panels rather than by quadrature: g is
        histogram-interpolated on the reference tape, and a trapezoid over a
        histogram is simply a different number.
        """
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return 0.0
        gx = np.asarray(self.g_x, dtype=float)
        gy = np.asarray(self.g_values, dtype=float)
        if gx.size < 2:
            return 0.0
        return integral(gx, gy, self._g_codes(), gx[0], hi / theta)

    def default_grid(self, energy: float,
                     n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
        """``theta * x`` over g's own abscissae -- the law's exact break points.

        Overridden because unlike the closed-form laws this one *does* have a
        natural grid, and sampling it anywhere else would round its corners.
        """
        theta = self.theta(energy)
        hi = self.upper_bound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return np.zeros(0, dtype=float)
        full = np.asarray(self.g_x, dtype=float) * theta
        grid = full[full <= hi]
        if full.size and full[-1] > hi:
            # The bound cuts the table: keep it, so the last panel is the
            # partial one the law actually has.
            grid = np.concatenate((grid, [hi]))
        elif grid.size == 0:
            grid = np.array([0.0, hi])
        # When g's support ends *before* the bound the grid stops there, and
        # deliberately so. Reaching on to ``hi`` would add one panel spanning
        # everything in between, and under the histogram convention that panel
        # holds the last tabulated value rather than zero -- on Cf-252 MT455
        # that single stretch is worth 3e-4 of a spectrum that integrates to 1.
        return grid

    def describe(self) -> str:
        return (f"LF=5, general evaporation, {len(self.g_x)}-point g(x), "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Maxwellian(MF5PartialAnalytic):
    """LF=7: ``sqrt(E') exp(-E'/theta(E))``, the simple Maxwellian fission spectrum."""

    lf: int = 7
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        return spectra.maxwellian(e_out, self.theta(energy))

    def normalisation_at(self, energy: float) -> float:
        """``theta^(3/2) [sqrt(pi)/2 erf(sqrt(y)) - sqrt(y) exp(-y)]``, ``y = (E-U)/theta``."""
        return spectra.maxwellian_integral(self.upper_bound(energy), self.theta(energy))

    def describe(self) -> str:
        return (f"LF=7, simple Maxwellian, "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Evaporation(MF5PartialAnalytic):
    """LF=9: ``E' exp(-E'/theta(E))``, the evaporation spectrum."""

    lf: int = 9
    theta_interp: List[Tuple[int, int]] = field(default_factory=list)
    theta_energies: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)

    def theta(self, energy: float) -> float:
        return tab1_at(self.theta_energies, self.theta_values,
                       self.theta_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        return spectra.evaporation(e_out, self.theta(energy))

    def normalisation_at(self, energy: float) -> float:
        """``theta^2 [1 - exp(-y)(1 + y)]``, ``y = (E-U)/theta``."""
        return spectra.evaporation_integral(self.upper_bound(energy), self.theta(energy))

    def describe(self) -> str:
        return (f"LF=9, evaporation, "
                f"{len(self.theta_energies)}-point theta(E)")


@dataclass
class MF5Watt(MF5PartialAnalytic):
    """LF=11: ``exp(-E'/a(E)) sinh(sqrt(b(E) E'))``, the energy-dependent Watt."""

    lf: int = 11
    a_interp: List[Tuple[int, int]] = field(default_factory=list)
    a_energies: List[float] = field(default_factory=list)
    a_values: List[float] = field(default_factory=list)
    b_interp: List[Tuple[int, int]] = field(default_factory=list)
    b_energies: List[float] = field(default_factory=list)
    b_values: List[float] = field(default_factory=list)

    def a(self, energy: float) -> float:
        return tab1_at(self.a_energies, self.a_values, self.a_interp, energy)

    def b(self, energy: float) -> float:
        return tab1_at(self.b_energies, self.b_values, self.b_interp, energy)

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        return spectra.watt(e_out, self.a(energy), self.b(energy))

    def normalisation_at(self, energy: float) -> float:
        """ENDF-6 §5.1.1.5's closed form for the Watt integral over ``[0, E-U]``."""
        return spectra.watt_integral(self.upper_bound(energy), self.a(energy), self.b(energy))

    def describe(self) -> str:
        return (f"LF=11, energy-dependent Watt, "
                f"{len(self.a_energies)}-point a(E), "
                f"{len(self.b_energies)}-point b(E)")


@dataclass
class MF5MadlandNix(MF5PartialAnalytic):
    """LF=12: the Madland-Nix spectrum, ``EFL``, ``EFH`` and a ``T_M(E)`` table.

    Normalised over ``[0, inf)`` by construction, and **not** bounded by
    ``E - U``: §5.1.1.6 gives the law no ``U`` (the subsection header's C1 is
    written 0), and the base class's bound would truncate a thermal-incident
    spectrum at 0.0253 eV. ``U`` stays in the dataclass because the header has
    the slot and the bytes go back as read.
    """

    lf: int = 12
    efl: float = 0.0
    efh: float = 0.0
    tm_interp: List[Tuple[int, int]] = field(default_factory=list)
    tm_energies: List[float] = field(default_factory=list)
    tm_values: List[float] = field(default_factory=list)

    def tm(self, energy: float) -> float:
        return tab1_at(self.tm_energies, self.tm_values, self.tm_interp, energy)

    def upper_bound(self, energy: float) -> float:
        return math.inf

    def shape(self, energy: float, e_out: np.ndarray) -> np.ndarray:
        return spectra.madland_nix(e_out, float(self.efl), float(self.efh), self.tm(energy))

    def normalisation_at(self, energy: float) -> float:
        return 1.0 if self.tm(energy) > 0.0 else 0.0

    def normalisation(self, energy: float) -> float:
        return self.normalisation_at(energy)

    def default_grid(self, energy: float,
                     n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
        hi = spectra.madland_nix_upper(float(self.efh), self.tm(energy))
        lo = hi * 10.0 ** (-_DEFAULT_GRID_DECADES)
        return np.concatenate(([0.0], np.geomspace(lo, hi, max(n_points - 1, 2))))

    def describe(self) -> str:
        return (f"LF=12, Madland-Nix, EFL={self.efl:g} eV, EFH={self.efh:g} eV, "
                f"{len(self.tm_energies)}-point T_M(E)")


#: ``LF`` -> the class that reads it. A law absent from here keeps its bytes and
#: is reported as a gap.
ANALYTIC_LAWS = {
    5: MF5GeneralEvaporation,
    7: MF5Maxwellian,
    9: MF5Evaporation,
    11: MF5Watt,
    12: MF5MadlandNix,
}

#: ``LF`` -> the ``(interp, abscissa, ordinate)`` field names of each TAB1 record
#: after the subsection header, **in tape order**.
#:
#: Separate from :data:`ANALYTIC_LAWS` because it says something the classes do
#: not: which record is which. LF=11 writes ``a(E)`` then ``b(E)`` and swapping
#: them produces a spectrum that is wrong and looks plausible, so the order is
#: written down once, here, and
#: ``test_analytic_record_names_match_the_walker`` holds it to the same counts
#: the walker in :data:`~kika.endf.classes.mf5.partials.TAB1_RECORDS_AFTER_HEADER`
#: uses.
ANALYTIC_RECORDS = {
    5: (("theta_interp", "theta_energies", "theta_values"),
        ("g_interp", "g_x", "g_values")),
    7: (("theta_interp", "theta_energies", "theta_values"),),
    9: (("theta_interp", "theta_energies", "theta_values"),),
    11: (("a_interp", "a_energies", "a_values"),
         ("b_interp", "b_energies", "b_values")),
    12: (("tm_interp", "tm_energies", "tm_values"),),
}

#: ``LF`` -> the field names of a TAB1 record's C1 and C2, for the laws that put
#: parameters there. Only LF=12 does: EFL and EFH are the header of its T_M
#: table (ENDF-6 §5.1.1.6), which is why the walker now keeps headers.
ANALYTIC_HEADER_FIELDS = {
    12: (("efl", "efh"),),
}

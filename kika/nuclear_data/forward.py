r"""What an elastic-scattering experiment sees, computed from a model ``ReactionSuite``.

The *forward operator* of a differential elastic measurement: from the evaluated
:math:`\sigma(E)` and :math:`f(\mu, E)` to the number the experiment reports in a
bin of incident energy at a detector angle,

.. math::
    \left\langle\frac{d\sigma}{d\Omega}\right\rangle(\mu_L; \text{bin})
    = \frac{J}{2\pi}\int K(E)\,\sigma(E)\,f(\mu_C, E)\,dE ,

with :math:`K` the bin seen through the energy resolution (a box convolved with a
Gaussian, :func:`kika.algebra.box_gaussian_fold_nodes`), :math:`\mu_C` the
centre-of-mass cosine of the laboratory one and :math:`J = d\Omega_C/d\Omega_L`.

**The product is folded, never the factors.**  Where :math:`\sigma` is resonant
the counts come from the energies where it is large, so the shape a detector sees
is :math:`\langle\sigma f\rangle/\langle\sigma\rangle`, not :math:`f` at the bin
centre nor :math:`\langle f\rangle`.  ``fold="sigma_only"`` gives the factor
reading :math:`\langle\sigma\rangle f(E_0)` only so the two can be compared.

Everything is read from the model, so a suite decoded from ENDF, from G4NDL or
modified in memory goes through the same code: :class:`ElasticView` extracts the
two factors once, and the readings are the plain-array primitives of
:mod:`kika.endf.dcs` and :mod:`kika.algebra`.

Kernel width.  :math:`\sigma_K^2 = \sigma_E^2 + \sigma_D^2`: the TOF resolution
(:class:`kika.endf.dcs.TofResolution`, whose ``min_sigma_e_kev`` floor must stay at
its default of zero here: a 1 keV floor is wider than ELISA below ~1 MeV) and the Doppler width of
the target's free-gas motion in its high-energy limit,
:math:`\sigma_D = \sqrt{2 E k_B T / A}`.  Adding the Doppler in quadrature is exact
for that limit (a Gaussian convolved with a Gaussian); it needs :math:`E \gg k_B T`,
which holds by six orders of magnitude at the MeV energies this is for.

Energies are in eV, cross sections in barns, differential ones in b/sr.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ElasticView",
    "ForwardSetup",
    "kernel_sigma_ev",
    "reading_nodes",
    "forward_sigma",
    "forward_dcs",
    "forward_measurement",
]

BOLTZMANN_EV_PER_K = 8.617333262e-5
ELASTIC_MT = 2

Fold = Literal["product", "sigma_only"]


# =============================================================================
# The two factors, out of the model
# =============================================================================

def _leaves(node) -> List[object]:
    """The ``XYs2d`` leaves of an angular form, in energy order."""
    from kika.nuclear_data.model.functions import Regions2d, XYs2d

    if isinstance(node, XYs2d):
        return [node]
    if isinstance(node, Regions2d):
        out: List[object] = []
        for child in node.function2ds:
            out.extend(_leaves(child))
        return out
    raise NotImplementedError(f"angular node of type {type(node).__name__}")


def _pdf_rows(angular, mu: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per leaf, ``(E_k, f(mu, E_k))`` at the leaf's own incident energies."""
    from kika.nuclear_data.model.enums import Interpolation

    rows = []
    for leaf in _leaves(angular):
        if leaf.interpolation != Interpolation.linlin:
            raise NotImplementedError(
                f"incident-energy interpolation {leaf.interpolation.value!r}: only lin-lin "
                "is read here (it is what every evaluated elastic MF4 uses)")
        if leaf.interpolationQualifier is not None:
            raise NotImplementedError(
                f"interpolationQualifier {leaf.interpolationQualifier.value!r} is not read here")
        e = np.array([float(f.outerDomainValue) for f in leaf.function1ds])
        v = np.array([float(np.atleast_1d(f.evaluate(np.array([mu])))[0])
                      for f in leaf.function1ds])
        rows.append((e, v))
    return rows


def _read_rows(rows, energies: np.ndarray) -> np.ndarray:
    """Lin-lin in energy inside each leaf; a leaf owns ``(E_previous_last, E_last]``.

    Where two leaves share an energy (the Legendre/table switch of ENDF LTT=3) the
    earlier one keeps it, as :meth:`AngularDistribution.evaluate_pdf` does, and the
    jump to the next is kept rather than averaged.  Outside the tabulated range the
    edge value is held.
    """
    out = np.empty(energies.shape, dtype=float)
    ends = [-np.inf] + [r[0][-1] for r in rows[:-1]] + [np.inf]
    for k, (e, v) in enumerate(rows):
        sel = (energies > ends[k]) & (energies <= ends[k + 1])
        if np.any(sel):
            out[sel] = np.interp(energies[sel], e, v)
    return out


@dataclass
class ElasticView:
    r"""The elastic channel of a suite, as the two factors a forward reading needs.

    An instance is built by :meth:`from_suite` (or by hand from the fields below).
    It is read through :func:`forward_sigma`, :func:`forward_dcs` and
    :func:`forward_measurement`.

    Attributes
    ----------
    xs_energies, xs_values : numpy.ndarray
        The pointwise :math:`\sigma(E)`, read lin-lin: energies in eV ascending,
        values in b.
    angular
        The neutron's ``AngularTwoBody.angular`` form (an ``XYs2d`` or a
        ``Regions2d`` of them, in incident energy and cosine), or ``None`` for an
        isotropic distribution, which reads as :math:`f = 1/2`. Reading it raises
        ``NotImplementedError`` unless its incident-energy interpolation is lin-lin
        with no qualifier.
    frame : {"lab", "cm"}
        The frame in which *angular* states the cosine.
    awr : float
        The target-to-neutron mass ratio.
    label : str
        The label of the cross section that was read.
    """

    xs_energies: np.ndarray
    xs_values: np.ndarray
    angular: object
    frame: str
    awr: float
    label: str = ""

    @classmethod
    def from_suite(cls, suite, *, cross_section_label: Optional[str] = None,
                   angular_label: str = "eval") -> "ElasticView":
        """Pull MT 2 out of ``suite``.

        Parameters
        ----------
        suite : ReactionSuite
            The evaluation. The target mass (AWR) is the first non-zero ``awr`` or
            ``targetMass`` attribute found on the MT 2 reaction's provenance, then
            on the suite's.
        cross_section_label : str, optional
            The label of the MT 2 cross section to read. None selects ``'recon'``
            (the pointwise 0 K reconstruction written by
            ``kika.endf.model_adapter.pendf.readReconstructed``) when the suite has
            it, else ``'eval'``. ``'eval'`` is accepted only when the suite has no
            resonance region, where it is the whole cross section and not just the
            background.
        angular_label : str
            The label of the neutron's angular distribution form on MT 2.

        Returns
        -------
        ElasticView
            With energies in eV and cross sections in b as the model holds them. The
            frame is ``"lab"`` when the distribution's ``productFrame`` is the
            laboratory and ``"cm"`` otherwise.

        Raises
        ------
        KeyError
            If the suite has no MT 2 reaction, or its cross section has no
            *cross_section_label*.
        ValueError
            If ``'eval'`` is read from a suite that has a resonance region, or no
            target mass is found.
        NotImplementedError
            If the cross section is not lin-lin throughout, or the distribution is
            neither an ``AngularTwoBody`` nor an isotropic one.
        """
        from kika.nuclear_data.model.distributions import AngularTwoBody, Isotropic2d
        from kika.nuclear_data.model.enums import Frame

        reaction = suite.reactions[ELASTIC_MT]
        container = reaction.crossSection
        labels = list(container.keys())
        label = cross_section_label or ("recon" if "recon" in labels else "eval")
        if label not in labels:
            raise KeyError(f"MT2 has no cross section {label!r} (it has {labels})")
        res = getattr(suite, "resonances", None)
        if label == "eval" and res is not None and (res.resolved or res.unresolved is not None):
            raise ValueError(
                "the suite has a resonance region, so its 'eval' MF3 is only the background; "
                "reconstruct it first (kika.endf.model_adapter.pendf.readReconstructed)")
        x, y, pairs = container[label].toEndfRegions()
        codes = {int(c) for _, c in pairs}
        if codes - {2}:
            raise NotImplementedError(f"cross section interpolation {sorted(codes)}: lin-lin only")

        dist = reaction.outputChannel.products.byPid("n")[0].distribution[angular_label]
        if isinstance(dist, AngularTwoBody) and isinstance(dist.angular, Isotropic2d):
            dist = Isotropic2d(productFrame=dist.productFrame)
        if not isinstance(dist, (AngularTwoBody, Isotropic2d)):
            raise NotImplementedError(f"elastic distribution of type {type(dist).__name__}")
        angular = None if isinstance(dist, Isotropic2d) else dist.angular
        frame = "lab" if dist.productFrame == Frame.lab else "cm"

        awr = None
        for p in (getattr(reaction, "provenance", None), getattr(suite, "provenance", None)):
            for attr in ("awr", "targetMass"):
                v = getattr(p, attr, None)
                if v:
                    awr = float(v)
                    break
            if awr:
                break
        if not awr:
            raise ValueError("no target mass (AWR) in the suite's provenance")
        return cls(np.asarray(x, float), np.asarray(y, float), angular, frame, awr, label)

    # -- readings -------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """The neutron-to-target mass ratio ``1 / awr``, the CM/LAB map parameter."""
        return 1.0 / self.awr

    def energy_grids(self) -> Tuple[np.ndarray, ...]:
        """The grids a fold must put nodes on: σ's and the angular incident energies."""
        cached = self.__dict__.get("_grids")
        if cached is None:
            grids = [self.xs_energies]
            if self.angular is not None:
                grids.append(np.unique(np.concatenate(
                    [r[0] for r in _pdf_rows(self.angular, 0.0)])))
            cached = self.__dict__["_grids"] = tuple(grids)
        return cached

    def sigma(self, energies) -> np.ndarray:
        """The cross section in b at *energies* (eV), interpolated lin-lin.

        Outside the tabulated range the nearest end value is held.

        Parameters
        ----------
        energies
            Incident energies in eV.

        Returns
        -------
        numpy.ndarray
            The cross section in b, one value per energy.
        """
        return np.interp(np.asarray(energies, float), self.xs_energies, self.xs_values)

    def pdf_native(self, mu: float) -> Callable[[np.ndarray], np.ndarray]:
        """``E -> f(mu, E)`` at one cosine in the distribution's own frame.

        Parameters
        ----------
        mu
            The cosine in the frame of the distribution (``frame``), not the
            laboratory one unless they coincide.

        Returns
        -------
        callable
            Maps incident energies in eV to the probability density in the cosine
            at *mu* (per unit cosine, normalised over [-1, 1]). It is lin-lin in
            energy inside each region, holds the end value outside the tabulated
            range, and is the constant 0.5 for an isotropic distribution. Where two
            regions share an incident energy the earlier one owns it.

        Raises
        ------
        NotImplementedError
            If the angular form is not lin-lin in incident energy, has an
            interpolation qualifier, or is not an ``XYs2d`` or ``Regions2d``.
        """
        if self.angular is None:
            return lambda e: np.full(np.shape(e), 0.5)
        rows = _pdf_rows(self.angular, float(mu))
        return lambda e: _read_rows(rows, np.asarray(e, float))

    def native_cosine(self, mu_lab: float) -> Tuple[float, float]:
        """``(mu_native, J)`` for a laboratory cosine: J = dΩ_native/dΩ_lab.

        Parameters
        ----------
        mu_lab
            The cosine of the scattering angle in the laboratory.

        Returns
        -------
        tuple of float
            The cosine in the frame of the distribution and the solid-angle
            Jacobian. For a laboratory distribution that is ``(mu_lab, 1.0)``; for a
            centre-of-mass one, the CM cosine of the same direction and
            dΩ_C/dΩ_L at it (single-valued for a target heavier than the neutron).
        """
        if self.frame == "lab":
            return float(mu_lab), 1.0
        from kika._legendre_frames import cos_cm_from_cos_lab, jacobian_cm_to_lab

        mu_cm = float(cos_cm_from_cos_lab(mu_lab, self.alpha))
        return mu_cm, float(jacobian_cm_to_lab(mu_cm, self.alpha))

    def dcs_lab(self, mu_lab: float, energies) -> np.ndarray:
        """Pointwise dσ/dΩ in the laboratory [b/sr], no resolution.

        The value is ``sigma * f(mu_native, E) * J / (2 pi)``, with no energy or
        angular averaging.

        Parameters
        ----------
        mu_lab
            The laboratory cosine of the scattering angle.
        energies
            Incident energies in eV.

        Returns
        -------
        numpy.ndarray
            The differential cross section in b/sr at each of *energies*.

        Raises
        ------
        NotImplementedError
            As :meth:`pdf_native`.
        """
        mu, jac = self.native_cosine(mu_lab)
        e = np.asarray(energies, float)
        return self.sigma(e) * self.pdf_native(mu)(e) * jac / (2 * np.pi)


# =============================================================================
# The reading of one bin
# =============================================================================

@dataclass(frozen=True)
class ForwardSetup:
    """What the experiment does to the evaluation before it is compared.

    Built by hand and passed to the ``forward_*`` functions. The defaults are an ideal
    experiment: no energy resolution, 0 K, a point detector.

    Attributes
    ----------
    tof : kika.endf.dcs.TofResolution, optional
        The time-of-flight resolution, whose ``sigma_e_mev`` gives the standard
        deviation of the energy resolution. None for none.
    temperature_k : float, optional
        Sample temperature in K for the Doppler width. None (or 0) for 0 K.
    angular_half_width_deg : float, optional
        Half-width in degrees of each detector's acceptance in the laboratory
        angle, averaged over solid angle. The window ``[theta - half, theta + half]``
        is clipped to ``[0, 180]`` degrees. None (or 0) for a point detector.
    n_angular : int
        Number of Gauss-Legendre points over the acceptance window (weighted by
        ``sin(theta)``). It does nothing for a point detector.
    """

    tof: Optional[object] = None
    temperature_k: Optional[float] = None
    angular_half_width_deg: Optional[float] = None
    n_angular: int = 5


def kernel_sigma_ev(setup: ForwardSetup, energy_ev: float, awr: float) -> float:
    """σ_K at ``energy_ev``: TOF resolution and free-gas Doppler, in quadrature.

    Parameters
    ----------
    setup
        Supplies the TOF resolution and the temperature.
    energy_ev
        The incident energy in eV.
    awr
        The target-to-neutron mass ratio, for the Doppler width
        ``sqrt(2 E k_B T / awr)``.

    Returns
    -------
    float
        The standard deviation (not the FWHM) of the Gaussian kernel, in eV. Each
        term is dropped when its setup field is None (or, for the temperature, 0),
        so with neither the result is 0.0.
    """
    s2 = 0.0
    if setup.tof is not None:
        s2 += (float(setup.tof.sigma_e_mev(energy_ev / 1e6)) * 1e6) ** 2
    if setup.temperature_k:
        s2 += 2.0 * energy_ev * BOLTZMANN_EV_PER_K * float(setup.temperature_k) / awr
    return float(np.sqrt(s2))


def reading_nodes(view: ElasticView, setup: ForwardSetup, energy_ev: float,
                  bin_ev: Optional[Tuple[float, float]] = None,
                  grids: Optional[Sequence[np.ndarray]] = None):
    """Nodes and weights for the bin ``bin_ev`` (or the point ``energy_ev``) seen through σ_K.

    Parameters
    ----------
    view
        Gives the target mass and, by default, the breakpoints of the integrand.
    setup
        Gives the kernel width through :func:`kernel_sigma_ev` at *energy_ev*.
    energy_ev
        The nominal energy in eV: the point read when *bin_ev* is None, and the
        energy at which the kernel width is evaluated in either case.
    bin_ev : tuple of float, optional
        The edges ``(lo, hi)`` of the bin in eV. None reads the point *energy_ev*.
    grids : sequence of numpy.ndarray, optional
        The breakpoints of the integrand in eV. None uses ``view.energy_grids()``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(nodes, weights)`` as :func:`kika.algebra.box_gaussian_fold_nodes` returns
        them: ascending energies in eV and one weight per node, summing to one.
    """
    from kika.algebra import box_gaussian_fold_nodes

    s = kernel_sigma_ev(setup, energy_ev, view.awr)
    lo, hi = (bin_ev if bin_ev is not None else (energy_ev, energy_ev))
    return box_gaussian_fold_nodes(lo, hi, s, grids if grids is not None else view.energy_grids())


def _all_nodes(view, setup, energies_ev, bins_ev):
    grids = view.energy_grids()
    parts, wparts = [], []
    for i, e0 in enumerate(energies_ev):
        b = None if bins_ev is None else bins_ev[i]
        n, w = reading_nodes(view, setup, float(e0), b, grids)
        parts.append(n)
        wparts.append(w)
    counts = np.array([p.size for p in parts])
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.concatenate(parts), np.concatenate(wparts), starts


def _detector_cosines(setup: ForwardSetup, mu_lab: float):
    """Solid-angle quadrature over the acceptance: ``(cosines, weights)``."""
    if not setup.angular_half_width_deg:
        return np.array([float(mu_lab)]), np.array([1.0])
    theta = np.arccos(np.clip(mu_lab, -1, 1))
    half = np.radians(float(setup.angular_half_width_deg))
    a, b = max(theta - half, 0.0), min(theta + half, np.pi)
    x, w = np.polynomial.legendre.leggauss(int(setup.n_angular))
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    w = w * np.sin(t)
    return np.cos(t), w / w.sum()


def forward_sigma(view: ElasticView, energies_ev, setup: ForwardSetup,
                  bins_ev: Optional[Sequence[Tuple[float, float]]] = None) -> np.ndarray:
    r""":math:`\langle\sigma\rangle` per bin [b].

    Parameters
    ----------
    view
        The elastic channel.
    energies_ev
        Nominal energies in eV, scalar or 1-D: the bin centres, or the points read
        when *bins_ev* is not given.
    setup
        The resolution and temperature of the experiment.
    bins_ev : sequence of tuple, optional
        One ``(lo, hi)`` pair in eV per entry of *energies_ev*; an entry may be None
        to read that energy as a point. None reads every energy as a point.

    Returns
    -------
    numpy.ndarray
        The folded cross section in b, one value per energy, shape ``(n_E,)``.
    """
    e = np.atleast_1d(np.asarray(energies_ev, float))
    nodes, w, starts = _all_nodes(view, setup, e, bins_ev)
    return np.add.reduceat(w * view.sigma(nodes), starts)


def forward_dcs(view: ElasticView, energies_ev, mu_lab, setup: ForwardSetup,
                bins_ev: Optional[Sequence[Tuple[float, float]]] = None,
                fold: Fold = "product") -> dict:
    r"""dσ/dΩ in the laboratory, per (energy or bin, cosine) [b/sr].

    Returns ``{"dcs": (n_E, n_mu), "sigma": (n_E,)}``, with ``sigma`` the folded
    :math:`\langle\sigma\rangle`.  ``fold="product"`` is the measurement;
    ``"sigma_only"`` is :math:`\langle\sigma\rangle f(\mu, E_0)`, kept to show the
    difference, never to compare with data.

    Parameters
    ----------
    view
        The elastic channel.
    energies_ev
        Nominal energies in eV, scalar or 1-D (bin centres, or the points read when
        *bins_ev* is not given).
    mu_lab
        Detector cosines in the laboratory, scalar or 1-D. With an angular
        acceptance in *setup* each is averaged over the window around its angle.
    setup
        The resolution, temperature and detector acceptance of the experiment.
    bins_ev : sequence of tuple, optional
        One ``(lo, hi)`` pair in eV per energy; an entry may be None to read that
        energy as a point. In ``"sigma_only"`` the factor ``f`` is taken at the bin
        midpoint (or at the energy for a point).
    fold : {"product", "sigma_only"}
        What is folded over the energy kernel: the product :math:`\sigma f`, or only
        :math:`\sigma`.

    Returns
    -------
    dict
        ``"dcs"``, a float array of shape ``(n_E, n_mu)`` in b/sr, and ``"sigma"``,
        the folded :math:`\langle\sigma\rangle` in b, shape ``(n_E,)``.

    Raises
    ------
    ValueError
        If *fold* is neither of the two strings.
    NotImplementedError
        As :meth:`ElasticView.pdf_native`.
    """
    e = np.atleast_1d(np.asarray(energies_ev, float))
    mus = np.atleast_1d(np.asarray(mu_lab, float))
    nodes, w, starts = _all_nodes(view, setup, e, bins_ev)
    sig_n = view.sigma(nodes)
    sigma_avg = np.add.reduceat(w * sig_n, starts)
    if fold == "sigma_only":
        centres = e if bins_ev is None else np.array(
            [e[i] if b is None else 0.5 * (b[0] + b[1]) for i, b in enumerate(bins_ev)])
    out = np.empty((e.size, mus.size))
    for j, mu in enumerate(mus):
        acc = np.zeros(e.size)
        for m, wm in zip(*_detector_cosines(setup, mu)):
            mu_n, jac = view.native_cosine(m)
            pdf = view.pdf_native(mu_n)
            if fold == "product":
                acc += wm * jac * np.add.reduceat(w * sig_n * pdf(nodes), starts)
            elif fold == "sigma_only":
                acc += wm * jac * sigma_avg * pdf(centres)
            else:
                raise ValueError(f"fold is 'product' or 'sigma_only', not {fold!r}")
        out[:, j] = acc / (2 * np.pi)
    return {"dcs": out, "sigma": sigma_avg}


def forward_measurement(view: ElasticView, measurement, setup: ForwardSetup,
                        fold: Fold = "product"):
    """The forward reading at every point of a ``kika.exfor.ExforAngularDistribution``.

    The measurement must be in the laboratory frame.  A block that carries
    ``bin = [lo, hi]`` (MeV) is read as that bin; one without it as a point.
    Returns a DataFrame: ``energy`` (MeV), ``angle`` (deg, lab), ``cos_lab``,
    ``value``, ``error``, ``model`` and ``sigma_model`` (the folded σ, b).

    Parameters
    ----------
    view
        The elastic channel.
    measurement
        A ``kika.exfor.ExforAngularDistribution``. Its cross-section unit must be
        ``b/sr``, ``mb/sr`` or ``ub/sr`` and its energy unit ``MeV``, ``keV`` or
        ``eV``; each point gives ``cos_lab`` (or ``angle`` in degrees) and
        ``cross_section``, with optional ``uncertainty_stat`` and ``uncertainty_sys``
        in the same unit as the cross section.
    setup
        The resolution, temperature and detector acceptance of the experiment.
    fold : {"product", "sigma_only"}
        As in :func:`forward_dcs`.

    Returns
    -------
    pandas.DataFrame
        One row per data point. ``energy`` in MeV; ``angle`` in degrees and
        ``cos_lab``; ``value`` and ``error`` in b/sr (``error`` is the statistical and
        systematic uncertainties added in quadrature, a missing one counting as 0);
        ``model`` in b/sr and ``sigma_model`` in b from the forward reading.

    Raises
    ------
    ValueError
        If the measurement's ``angle_frame`` is not the laboratory, or *fold* is
        invalid.
    KeyError
        If a unit of the measurement is not among those above.
    """
    import pandas as pd

    if str(getattr(measurement, "angle_frame", "LAB")).upper() != "LAB":
        raise ValueError("forward_measurement compares in the laboratory frame")
    xs_scale = {"b/sr": 1.0, "mb/sr": 1e-3, "ub/sr": 1e-6}[measurement.units["cross_section"]]
    e_scale = {"MeV": 1e6, "keV": 1e3, "eV": 1.0}[measurement.units["energy"]]
    rows = []
    for block in measurement._data_blocks:
        e0 = float(block.get("value") or block.get("E")) * e_scale
        b = block.get("bin")
        bin_ev = (float(b[0]) * e_scale, float(b[1]) * e_scale) if b else None
        for p in block.get("data", []):
            mu = p["cos_lab"] if "cos_lab" in p else float(np.cos(np.radians(p["angle"])))
            err = np.hypot(p.get("uncertainty_stat") or 0.0, p.get("uncertainty_sys") or 0.0)
            rows.append({"energy": e0 / 1e6, "angle": float(np.degrees(np.arccos(mu))),
                         "cos_lab": float(mu), "value": float(p["cross_section"]) * xs_scale,
                         "error": err * xs_scale, "_bin": bin_ev})
    df = pd.DataFrame(rows)
    df["model"] = np.nan
    df["sigma_model"] = np.nan
    # One call per detector cosine: every bin seen at that angle is read at once,
    # so a 400-bin energy series is one vectorised sweep and not 400 of them.
    for mu, idx in df.groupby("cos_lab").groups.items():
        sub = df.loc[idx]
        res = forward_dcs(view, sub["energy"].to_numpy() * 1e6, [mu], setup,
                          list(sub["_bin"]), fold)
        df.loc[idx, "model"] = res["dcs"][:, 0]
        df.loc[idx, "sigma_model"] = res["sigma"]
    return df.drop(columns="_bin")

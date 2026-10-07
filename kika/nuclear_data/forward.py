r"""What an elastic-scattering experiment sees, computed from a model ``ReactionSuite``.

The *forward operator* of a differential elastic measurement: from the evaluated
:math:`\sigma(E)` and :math:`f(\mu, E)` to the number the experiment reports in a
bin of incident energy at a detector angle,

.. math::
    \left\langle\frac{d\sigma}{d\Omega}\right\rangle(\mu_L; \text{bin})
    = \frac{J}{2\pi}\int K(E)\,\sigma(E)\,f(\mu_C, E)\,dE ,

with :math:`K` the bin seen through the energy resolution (a box convolved with a
Gaussian, :func:`kika.utils.numerics.box_gaussian_fold_nodes`), :math:`\mu_C` the
centre-of-mass cosine of the laboratory one and :math:`J = d\Omega_C/d\Omega_L`.

**The product is folded, never the factors.**  Where :math:`\sigma` is resonant
the counts come from the energies where it is large, so the shape a detector sees
is :math:`\langle\sigma f\rangle/\langle\sigma\rangle`, not :math:`f` at the bin
centre nor :math:`\langle f\rangle`.  ``fold="sigma_only"`` gives the factor
reading :math:`\langle\sigma\rangle f(E_0)` only so the two can be compared.

Everything is read from the model, so a suite decoded from ENDF, from G4NDL or
modified in memory goes through the same code: :class:`ElasticView` extracts the
two factors once, and the readings are the plain-array primitives of
:mod:`kika.endf.dcs` and :mod:`kika.utils.numerics`.

Kernel width.  :math:`\sigma_K^2 = \sigma_E^2 + \sigma_D^2`: the TOF resolution
(:class:`kika.endf.dcs.TofResolution`; mind its ``min_sigma_e_kev`` floor, 1 keV by
default, which is wider than a 27 m flight path at 1 MeV) and the Doppler width of
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

    ``xs_energies``/``xs_values`` are the pointwise :math:`\sigma(E)` (lin-lin),
    ``angular`` the neutron's ``AngularTwoBody`` form, ``frame`` its frame and
    ``awr`` the target-to-neutron mass ratio.
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
        """Pull MT2 out of ``suite``.

        ``cross_section_label`` defaults to ``'recon'`` (the pointwise 0 K
        reconstruction, :func:`kika.endf.model_adapter.pendf.readReconstructed`)
        when the suite has it.  ``'eval'`` is accepted only without a resonance
        region, where it is the whole cross section and not just the background.
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
        return np.interp(np.asarray(energies, float), self.xs_energies, self.xs_values)

    def pdf_native(self, mu: float) -> Callable[[np.ndarray], np.ndarray]:
        """``E -> f(mu, E)`` at one cosine in the distribution's own frame."""
        if self.angular is None:
            return lambda e: np.full(np.shape(e), 0.5)
        rows = _pdf_rows(self.angular, float(mu))
        return lambda e: _read_rows(rows, np.asarray(e, float))

    def native_cosine(self, mu_lab: float) -> Tuple[float, float]:
        """``(mu_native, J)`` for a laboratory cosine: J = dΩ_native/dΩ_lab."""
        if self.frame == "lab":
            return float(mu_lab), 1.0
        from kika.endf.dcs import cos_cm_from_cos_lab, jacobian_cm_to_lab

        mu_cm = float(cos_cm_from_cos_lab(mu_lab, self.alpha))
        return mu_cm, float(jacobian_cm_to_lab(mu_cm, self.alpha))

    def dcs_lab(self, mu_lab: float, energies) -> np.ndarray:
        """Pointwise dσ/dΩ in the laboratory [b/sr], no resolution."""
        mu, jac = self.native_cosine(mu_lab)
        e = np.asarray(energies, float)
        return self.sigma(e) * self.pdf_native(mu)(e) * jac / (2 * np.pi)


# =============================================================================
# The reading of one bin
# =============================================================================

@dataclass(frozen=True)
class ForwardSetup:
    """What the experiment does to the evaluation before it is compared.

    ``tof``: a :class:`kika.endf.dcs.TofResolution`, or ``None`` for none.
    ``temperature_k``: sample temperature for the Doppler width, ``None`` for 0 K.
    ``angular_half_width_deg``: half-width of each detector's acceptance in the
    laboratory, averaged over solid angle; ``None`` for a point detector.
    """

    tof: Optional[object] = None
    temperature_k: Optional[float] = None
    angular_half_width_deg: Optional[float] = None
    n_uniform: int = 101
    n_angular: int = 5


def kernel_sigma_ev(setup: ForwardSetup, energy_ev: float, awr: float) -> float:
    """σ_K at ``energy_ev``: TOF resolution and free-gas Doppler, in quadrature."""
    s2 = 0.0
    if setup.tof is not None:
        s2 += (float(setup.tof.sigma_e_mev(energy_ev / 1e6)) * 1e6) ** 2
    if setup.temperature_k:
        s2 += 2.0 * energy_ev * BOLTZMANN_EV_PER_K * float(setup.temperature_k) / awr
    return float(np.sqrt(s2))


def reading_nodes(view: ElasticView, setup: ForwardSetup, energy_ev: float,
                  bin_ev: Optional[Tuple[float, float]] = None,
                  grids: Optional[Sequence[np.ndarray]] = None):
    """Nodes and weights for the bin ``bin_ev`` (or the point ``energy_ev``) seen through σ_K."""
    from kika.utils.numerics import box_gaussian_fold_nodes

    s = kernel_sigma_ev(setup, energy_ev, view.awr)
    lo, hi = (bin_ev if bin_ev is not None else (energy_ev, energy_ev))
    return box_gaussian_fold_nodes(lo, hi, s, grids if grids is not None else view.energy_grids(),
                                   n_uniform=setup.n_uniform)


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
    r""":math:`\langle\sigma\rangle` per bin [b]."""
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

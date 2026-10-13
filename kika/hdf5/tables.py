"""A nuclide read from OpenMC's HDF5, as the plain arrays a viewer draws.

The kika-app explores an OpenMC library the way it explores a GNDS or G4NDL
one, and reads the same two shapes: every cross section pointwise (the
``/api/endf/mf3-bulk`` response) and each reaction's angular distribution as
MF4 (``/api/endf/bulk-legendre``). :mod:`kika.g4ndl.tables` produces those off
an evaluation; this module does it off the processed suite
:func:`kika.hdf5.read` returns, which differs in two ways:

* **σ is per temperature.** Every reaction carries one form per ``Heated``
  style (``heated_294K``, ...), each on that temperature's own energy grid,
  and the elastic adds the 0 K ``recon`` OpenMC keeps for resonance
  up-scattering. Nothing is labelled ``eval``. :func:`crossSections` takes a
  temperature in kelvin, or ``0`` for that 0 K elastic.
* **The responses are served beside σ.** Gas production (MT 203-207, in b),
  KERMA (301 and the local 901) and the damage energy (444, all three in eV·b)
  are not reactions but ``suite.productions``; :func:`crossSections` serves
  them per temperature with the reactions, and :func:`isotopeSummary` marks
  each row's ``unit``.
* **Angular tables are sampling tables.** Every angular distribution is a
  ``MonteCarlo`` ``XYs2d`` of :class:`Xs_pdf_cdf1d` (ACE law 61's cosine
  tables), lin-lin or histogram in μ, never Legendre. :func:`angularBulk`
  sends each one as a tabulated MF4 row (``representation`` 2) with its
  Legendre moments projected, the convention :mod:`kika.g4ndl.tables` uses
  above a G4NDL transition. A histogram table goes out as steps (each interior
  cosine twice), so a client interpolating linearly draws what OpenMC samples.

A reaction whose neutron law is a single-weight ``DistributionMixture``
(OpenMC's ``applicability`` with one entry) is unwrapped; one with several
weights, and the correlated laws (``EnergyAngularMC``, Kalbach-Mann), have no
angular distribution of their own and are not served, as in the G4NDL tables.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["temperatures", "referenceTemperature", "formLabel", "crossSections",
           "angularMTs", "angularBulk", "isotopeSummary", "ZERO_K_FORM",
           "PROJECTION_ORDER", "ENERGY_RESPONSE_MTS"]

#: The 0 K elastic OpenMC stores beside the heated ones.
ZERO_K_FORM = "recon"

#: Legendre order the tables are projected to when no ``maxOrder`` is asked.
PROJECTION_ORDER = 20

#: Boltzmann's constant in eV/K, the unit a ``Heated`` style's temperature is in.
_K_B = 8.617333262e-5

_HEATED = re.compile(r"^heated_(\d+(?:\.\d+)?)K$")

#: Responses whose values are an energy times a cross section (eV·b): KERMA,
#: its local part and the damage energy. Every other row is in barns.
ENERGY_RESPONSE_MTS = (301, 444, 901)


def temperatures(suite) -> List[Dict[str, Any]]:
    """Each ``Heated`` style as ``{label, kelvin}``, coldest first."""
    out = []
    for style in suite.styles:
        if type(style).__name__ != "Heated":
            continue
        label = str(style.label)
        match = _HEATED.match(label)
        if match:
            kelvin = float(match.group(1))
        else:
            value = getattr(getattr(style, "temperature", None), "value", None)
            kelvin = round(float(value) / _K_B, 1) if value is not None else float("nan")
        out.append(dict(label=label, kelvin=kelvin))
    return sorted(out, key=lambda t: t["kelvin"])


def referenceTemperature(suite) -> Optional[float]:
    """The temperature a viewer opens on: the one closest to 294 K."""
    temps = temperatures(suite)
    if not temps:
        return None
    return min(temps, key=lambda t: abs(t["kelvin"] - 294.0))["kelvin"]


def formLabel(suite, temperature: Optional[float]) -> str:
    """The form label of ``temperature`` (K): ``0`` → :data:`ZERO_K_FORM`,
    ``None`` → :func:`referenceTemperature`. Raises ``KeyError`` naming the
    temperatures the file has when it has not this one."""
    if temperature is not None and float(temperature) == 0.0:
        return ZERO_K_FORM
    temps = temperatures(suite)
    if temperature is None:
        temperature = referenceTemperature(suite)
    for t in temps:
        if abs(t["kelvin"] - float(temperature)) < 0.5:
            return t["label"]
    raise KeyError(f"{suite.target} has no data at {temperature} K; it has "
                   f"{[t['kelvin'] for t in temps]} K (and 0 K for the elastic)")


def _nodes(suite):
    return [n for n in list(suite.reactions) + list(suite.sums) + list(suite.productions)
            if n.ENDF_MT is not None]


def crossSections(suite, temperature: Optional[float] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """MT → ``(E, sigma)`` (eV, b) at ``temperature`` (K; see :func:`formLabel`)
    for every reaction, sum and response that has it; the responses in
    :data:`ENERGY_RESPONSE_MTS` are in eV·b. At 0 K only the elastic exists."""
    label = formLabel(suite, temperature)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for node in _nodes(suite):
        if label not in node.crossSection.keys():
            continue
        form = node.crossSection[label]
        out[int(node.ENDF_MT)] = (np.asarray(form.xs, dtype=float), np.asarray(form.ys, dtype=float))
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ angular

def _neutronLaw(suite, mt: int):
    reaction = suite.reactions.byENDF_MT(mt)
    products = reaction.outputChannel.products.byPid("n")
    if not products or products[0].distribution is None:
        return None
    forms = list(products[0].distribution.keys())
    if not forms:
        return None
    law = products[0].distribution[forms[0]]
    if type(law).__name__ == "DistributionMixture":
        weighted = list(law.weighted)
        if len(weighted) != 1:
            return None
        law = weighted[0].functional
    return law


def _angular(law):
    if law is None or type(law).__name__ not in ("AngularTwoBody", "Uncorrelated"):
        return None
    angular = getattr(law, "angular", None)
    return angular if hasattr(angular, "function1ds") else None


def _frame(law) -> Optional[str]:
    frame = getattr(law, "productFrame", None)
    if frame is None:
        return None
    return "LAB" if "lab" in str(getattr(frame, "value", frame)).lower() else "CM"


def angularMTs(suite) -> List[int]:
    """MTs whose outgoing neutron has a cosine table of its own (see the
    module docstring): what :func:`angularBulk` serves."""
    out = []
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        if mt is None:
            continue
        try:
            if _angular(_neutronLaw(suite, int(mt))) is not None:
                out.append(int(mt))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def _isHistogram(function) -> bool:
    return "flat" in str(getattr(function.interpolation, "value", function.interpolation))


def _moments(mu: np.ndarray, p: np.ndarray, histogram: bool, order: int) -> np.ndarray:
    """``int p P_l dmu``, exact on each segment for a polynomial ``P_l``:
    Gauss–Legendre with enough nodes for degree ``order + 1``."""
    if mu.size < 2:
        return np.zeros(order + 1)
    nodes, weights = np.polynomial.legendre.leggauss(order // 2 + 2)
    x1, x2 = mu[:-1, None], mu[1:, None]
    half = 0.5 * (x2 - x1)
    x = half * nodes + 0.5 * (x1 + x2)                       # (segments, nodes)
    if histogram:
        y = np.broadcast_to(p[:-1, None], x.shape)
    else:
        slope = np.divide(p[1:] - p[:-1], (x2 - x1)[:, 0],
                          out=np.zeros(mu.size - 1), where=x2[:, 0] != x1[:, 0])
        y = p[:-1, None] + slope[:, None] * (x - x1)
    w = (half * weights * y).reshape(-1)                     # zero on empty segments
    return np.polynomial.legendre.legvander(x.reshape(-1), order).T @ w


def _linear(mu: np.ndarray, p: np.ndarray, histogram: bool) -> Tuple[np.ndarray, np.ndarray]:
    """A table a client interpolates lin-lin: a histogram becomes its steps."""
    if not histogram or mu.size < 2:
        return mu, p
    x = np.repeat(mu, 2)[1:-1]
    y = np.repeat(p[:-1], 2)
    return x, y


def angularBulk(suite, maxOrder: Optional[int] = None, mt: int = 2) -> Dict[str, Any]:
    """Reaction ``mt``'s cosine tables in the shape of an MF4 bulk response
    (the keys :func:`kika.g4ndl.tables.angularBulk` returns), one tabulated row
    per incident energy with its moments up to ``maxOrder``."""
    law = _neutronLaw(suite, mt)
    angular = _angular(law)
    if angular is None:
        raise ValueError(f"MT{mt} of {suite.target} has no angular distribution of its own; "
                         f"angular MTs: {angularMTs(suite)}")
    order = PROJECTION_ORDER if maxOrder is None else int(maxOrder)
    energies: List[float] = []
    rows: List[np.ndarray] = []
    pdfMu: List[List[float]] = []
    pdf: List[List[float]] = []
    for f in angular.function1ds:
        mu = np.asarray(f.xs, dtype=float)
        p = np.asarray(f.pdf if hasattr(f, "pdf") else f.ys, dtype=float)
        histogram = _isHistogram(f)
        energies.append(float(f.outerDomainValue))
        rows.append(_moments(mu, p, histogram, order))
        lmu, lp = _linear(mu, p, histogram)
        pdfMu.append(lmu.tolist())
        pdf.append(lp.tolist())
    matrix = np.vstack(rows) if rows else np.zeros((0, order + 1))
    return dict(
        energies=energies,
        max_order=order,
        coefficients_by_order={str(l): matrix[:, l].tolist() for l in range(order + 1)},
        energy_interpolation=None,
        representation=2,
        frame=_frame(law),
        pdf_mu_by_energy=pdfMu,
        pdf_by_energy=pdf,
        pdf_angular_linlin=True,
        pdf_boundary_energy=None,
        pdf_boundary_mu=None,
        pdf_boundary=None,
    )


# ------------------------------------------------------------------ summary

def _range(e: np.ndarray, s: np.ndarray) -> Dict[str, Any]:
    return dict(count=int(e.size), energy_min=float(e[0]), energy_max=float(e[-1]),
                sigma_max=float(s.max()) if s.size else 0.0)


def isotopeSummary(suite, temperature: Optional[float] = None) -> Dict[str, Any]:
    """What a viewer shows above a nuclide's plots: its temperatures, one row
    per cross section at ``temperature`` (default :func:`referenceTemperature`)
    and what the decode lost."""
    reference = referenceTemperature(suite) if temperature is None else float(temperature)
    angular = set(angularMTs(suite))
    sums = {int(n.ENDF_MT) for n in suite.sums if n.ENDF_MT is not None}
    responses = {int(n.ENDF_MT) for n in suite.productions if n.ENDF_MT is not None}
    nodes = {int(n.ENDF_MT): n for n in _nodes(suite)}
    pointwise = crossSections(suite, reference) if reference is not None else {}

    rows: List[Dict[str, Any]] = []
    for mt, (e, s) in pointwise.items():
        node = nodes[mt]
        q = None
        if mt not in sums and mt not in responses:
            q = getattr(getattr(getattr(node, "outputChannel", None), "Q", None), "value", None)
        positive = np.nonzero(s > 0)[0]
        row = dict(mt=mt, label=str(node.label), q_value=None if q is None else float(q),
                   threshold=float(e[positive[0]]) if positive.size else None,
                   sum=mt in sums, angular=mt in angular, response=mt in responses,
                   unit="eV*b" if mt in ENERGY_RESPONSE_MTS else "b", **_range(e, s))
        rows.append(row)

    zeroK = None
    if 2 in nodes and ZERO_K_FORM in nodes[2].crossSection.keys():
        form = nodes[2].crossSection[ZERO_K_FORM]
        zeroK = _range(np.asarray(form.xs, dtype=float), np.asarray(form.ys, dtype=float))

    report = suite.report
    return dict(
        target=str(suite.target),
        projectile=str(suite.projectile),
        temperatures=temperatures(suite),
        reference_temperature=reference,
        elastic_0k=zeroK,
        reactions=rows,
        angular_mts=sorted(angular),
        counts=dict(reactions=len(suite.reactions), sums=len(suite.sums),
                    productions=len(suite.productions),
                    fission_components=len(suite.fissionComponents)),
        report=dict(warnings=list(report.warnings), losses=list(report.losses),
                    approximations=list(report.approximations),
                    unsupported=list(report.unsupported)) if report is not None else None,
    )

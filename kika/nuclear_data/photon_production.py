"""The photons a reaction suite emits: lines, continua and their production.

This is the physics behind KIKA's photon panel, moved into the library so that
any :class:`~kika.nuclear_data.model.ReactionSuite` answers it the same way,
whichever format it was read from (ENDF through
:func:`kika.endf.model_adapter.decodeReactionSuite`, GNDS, G4NDL). It asks a
physical question instead of listing a file's sections: *which gammas come out,
and how many?*

Every photon product of a reaction is one of four things, and each answers
differently:

* a **level cascade** (ENDF MF12 LO=2): the reaction leaves the residual in a
  level, and the level's decay lives in PoPs.
  :func:`~kika.nuclear_data.model.cascadeLines` follows it to the ground state;
  each line's photons per reaction are fixed, independent of the incident
  energy. A cascade is isotropic: the model adapter refuses an anisotropic
  MF14 on a cascade, since GNDS has no place for a per-transition angular
  distribution, so any cascade that reached the model is isotropic;
* a **discrete line** (MF12 LO=1 or MF13, ``DiscreteGamma``): its multiplicity
  is a function of the incident energy;
* a **primary photon** (``PrimaryGamma``): a line whose energy moves with the
  incident one, Eγ = binding + AWR/(AWR+1)·E;
* a **continuum** (MF15, or an MF6 photon): a multiplicity and a spectrum
  p(Eγ | E). The model deliberately has no interpolation between two incident
  energies of such a spectrum (``XYs2d.evaluate``), so a spectrum is always
  drawn at the *nearest tabulated* incident energy, and every answer says which.
  Its mean energy, a scalar, is interpolated linearly between nodes, which is
  safe in the way interpolating the spectrum itself is not.

Two quantities. ``multiplicity`` is photons per reaction, one reaction at a
time, since summing it over reactions means nothing. ``production`` is σ·Y in
barns, per reaction or summed over all of them.

**Cross sections a file only half states.** For a tape with resonance
parameters, the pointwise σ of the reactions the resonances complete is only
the background below the top of the resonance range. The library does not
reconstruct here; the caller says two things instead:

* ``cross_section``: σ to use in place of the model's for a reaction, either a
  callable ``(mt, energies) -> values | None`` or a mapping
  ``{mt: (energies, values)}`` (``.get`` is all that is called, so a lazy
  mapping that loads from a cache on demand works). ``None`` falls back to the
  model. A mapping is the better choice when there is one: its grid joins the
  curves' grid, so a reconstruction's resonances are drawn where they are. A
  callable is only evaluated on the grid the file and the multiplicities give;
* ``background_mts`` and ``resonance_top``: the MTs whose model σ is only a
  background below ``resonance_top`` (eV). Where one of them has no override
  and the energy is below the top, the reaction is *left out*: its
  contributions are NaN there, the totals are NaN, and ``excluded`` names it.
  An unknown contribution makes a sum unknown, not smaller.

:func:`resonance_background` derives the second pair from a flat ENDF object,
by the rule KIKA has always used.

Every method returns plain dicts, lists and floats, ready for JSON. Each line
carries a stable ``key`` that names it across calls and incident energies:

* ``cascade:<initial>:<final>``, e.g. ``cascade:Fe56_e1:Fe56``;
* ``line:<Eγ in eV, rounded>``, e.g. ``line:846778``. Two reactions' lines at
  the same energy to the eV are one line, so the key merges them;
* ``primary:<mt>:<binding energy in eV, rounded>``, e.g. ``primary:102:10829100``;
* ``continuum:<mt>:<k>``, the k-th continuum photon of a reaction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "PhotonProduction",
    "PhotonProductionError",
    "resonance_background",
    "NEUTRON_MASS_AMU",
    "MAX_CURVE_POINTS",
    "MAX_LINE_CURVES",
    "MAX_ENERGY_PARTS",
]

NEUTRON_MASS_AMU = 1.00866491595
#: A curve longer than this is decimated, keeping each bin's extremes.
MAX_CURVE_POINTS = 3000
#: Lines drawn as their own curve; the rest are summed into "N weaker lines".
MAX_LINE_CURVES = 60
#: Reactions given their own gamma-energy curve; the rest are "other reactions".
MAX_ENERGY_PARTS = 8
#: The union of resonance grids is ~10⁵ points; curves are evaluated on at most this.
MAX_CURVE_GRID = 20000
#: Cosines an angular distribution is sampled on.
ANGULAR_POINTS = 201
#: Outgoing-energy points of a continuum map.
CONTINUUM_POINTS = 300

#: Where the decoded photon items are cached on a suite: they do not depend on
#: the cross-section choices, so every ``PhotonProduction`` of a suite shares them.
_CONTRIB_ATTR = "_kika_photon_contributions"

QUANTITIES = ("multiplicity", "production")

CrossSectionOverride = Union[
    Callable[[Optional[int], np.ndarray], Optional[np.ndarray]],
    Mapping,
]


class PhotonProductionError(ValueError):
    """The request cannot be answered: an unknown quantity, a reaction with no
    photons, multiplicity summed over reactions, or a line that is not there."""


# ---------------------------------------------------------------------------
# Reading the model
# ---------------------------------------------------------------------------

#: Style labels in the order a form is preferred. ``eval`` is the evaluation;
#: ``recon`` its pointwise reconstruction (FUDGE's, in published GNDS);
#: a ``heated_<T>`` label (any label starting ``heated``) is what an
#: ACE-decoded suite holds, which has no ``eval`` at all.
#: Any other label is taken after these, in file order.
_LABEL_PREFERENCE = ("eval", "recon", "heated")


def _rank(label: str) -> int:
    for i, preferred in enumerate(_LABEL_PREFERENCE):
        if label == preferred or (preferred == "heated" and label.startswith("heated")):
            return i
    return len(_LABEL_PREFERENCE)


def _forms(component) -> List[Any]:
    """A component's forms, preferred label first; ``[]`` for nothing there.

    Style-label neutral on purpose: the same physics reads an ENDF-decoded
    suite (``eval``), a GNDS one (``eval`` beside ``recon``) and an ACE one
    (``heated_<T>`` and ``MonteCarlo`` only), and a hard-coded ``eval`` would
    find nothing in the last.
    """
    if component is None:
        return []
    try:
        labels = list(component.keys())
    except (AttributeError, TypeError):
        return []
    ordered = sorted(labels, key=_rank)  # stable: file order within a rank
    out = []
    for label in ordered:
        try:
            out.append(component[label])
        except (KeyError, TypeError):
            continue
    return out


def _evaluated(product):
    forms = _forms(getattr(product, "distribution", None))
    return forms[0] if forms else None


def _grid_of(form) -> np.ndarray:
    """The abscissae of a 1-d function: an XYs1d's, or a Regions1d's concatenated."""
    xs = getattr(form, "xs", None)
    if xs is not None:
        return np.asarray(xs, dtype=float)
    parts = [np.asarray(f.xs, dtype=float) for f in getattr(form, "function1ds", [])
             if getattr(f, "xs", None) is not None]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0)


def _xs_form(node):
    """The σ form to evaluate: the first preferred form that is a function.

    An ENDF-decoded suite's ``eval`` is MF3, a function. A GNDS evaluation with
    resonances writes ``eval`` as ``resonancesWithBackground``, an instruction
    and not a function, beside FUDGE's reconstructed ``recon``; that one is
    what can be evaluated. An ACE-decoded suite has only its heated, gridded
    form. With no evaluable form the first one is returned, so the caller
    reports it as not evaluable instead of as missing.
    """
    forms = _forms(getattr(node, "crossSection", None))
    for form in forms:
        if hasattr(form, "evaluate"):
            return form
    return forms[0] if forms else None


def _target_awr(suite) -> Optional[float]:
    target = suite.PoPs.particles.get(getattr(suite, "target", None))
    mass = getattr(target, "mass", None)
    if mass is None:
        return None
    value = mass.convertedTo("amu").value if getattr(mass, "unit", None) else mass.value
    return float(value) / NEUTRON_MASS_AMU


def _angular_kind(form) -> Optional[str]:
    """``'isotropic'``, ``'anisotropic'``, or ``None`` when the file states nothing."""
    if form is None:
        return None
    if type(form).__name__ == "Isotropic2d":
        return "isotropic"
    if getattr(form, "function1ds", None):
        return "anisotropic"
    return None


def _photon_items(channel, parent: Optional[str]) -> List[Dict[str, Any]]:
    """Every photon of ``channel`` and of its products' decays, classified."""
    items: List[Dict[str, Any]] = []
    for product in channel.products:
        if product.pid == "photon":
            form = _evaluated(product)
            energy = getattr(form, "energy", None)
            kind = type(energy).__name__ if energy is not None else None
            multiplicity = getattr(getattr(product, "multiplicity", None), "form", None)
            mkind = type(multiplicity).__name__ if multiplicity is not None else None
            angular = getattr(form, "angular", None)
            if mkind == "Branching1d" or kind == "Branching3d":
                if parent is not None:
                    items.append({"kind": "cascade", "level": parent})
            elif kind == "DiscreteGamma":
                items.append({"kind": "line", "energy": float(energy.value), "mult": multiplicity,
                              "angular": angular})
            elif kind == "PrimaryGamma":
                items.append({"kind": "primary", "binding": float(energy.value),
                              "mult": multiplicity, "angular": angular})
            elif kind == "DiscretePhotonEnergy":
                # A processed table's line (ACE law 2, OpenMC discrete_photon):
                # LP = 2 is a primary photon, any other LP a line at `energy`.
                if energy.primaryFlag == 2:
                    items.append({"kind": "primary", "binding": float(energy.energy),
                                  "mult": multiplicity, "angular": angular})
                else:
                    items.append({"kind": "line", "energy": float(energy.energy),
                                  "mult": multiplicity, "angular": angular})
            elif energy is not None and getattr(energy, "function1ds", None):
                spectrum, nd = _density_spectrum(energy)
                item = {"kind": "continuum", "spectrum": spectrum, "mult": multiplicity,
                        "angular": angular, "product": product.label}
                if any(nd):
                    item["nd"] = nd
                items.append(item)
            elif multiplicity is None and energy is None:
                # A placeholder: the photon of a level whose decay the file
                # states elsewhere (JENDL-5 gives Fe-56's level gammas in the
                # reaction's MF6, and the residual level still lists a
                # "photon" with nothing in it). There is nothing to draw and
                # nothing missing, so nothing is said.
                continue
            else:
                items.append({"kind": "unsupported", "what": kind or mkind or "nothing"})
        if product.outputChannel is not None:
            items += _photon_items(product.outputChannel, product.pid)
    return items


def _decode(suite) -> List[Dict[str, Any]]:
    """Per photon-producing reaction: its MT, cross section and photon items."""
    cached = getattr(suite, _CONTRIB_ATTR, None)
    if cached is not None:
        return cached
    from kika.nuclear_data.model import cascadeLines

    lines_of: Dict[str, list] = {}
    out = []
    for node, kind in ([(r, "reaction") for r in suite.reactions]
                       + [(r, "orphan") for r in getattr(suite, "orphanProducts", [])]):
        if getattr(node, "outputChannel", None) is None:
            continue
        items = _photon_items(node.outputChannel, None)
        if not items:
            continue
        discrete = _mf6_discrete_points(node)
        if discrete or any("nd" in item for item in items):
            items = [part for item in items for part in _split_discrete(item, discrete)]
        continuum_index = 0
        for item in items:
            if item["kind"] == "cascade":
                level = item["level"]
                if level not in lines_of:
                    lines_of[level] = cascadeLines(suite.PoPs, level)
                item["lines"] = lines_of[level]
            elif item["kind"] == "continuum":
                item["index"] = continuum_index
                continuum_index += 1
        xs = _xs_form(node)
        mt = int(node.ENDF_MT) if getattr(node, "ENDF_MT", None) is not None else None
        out.append({
            "mt": mt, "label": node.label, "kind": kind, "xs": xs,
            "xs_grid": _grid_of(xs) if xs is not None else np.empty(0),
            "items": items,
        })
    out.sort(key=lambda c: (c["mt"] is None, c["mt"] or 0))
    try:
        setattr(suite, _CONTRIB_ATTR, out)
    except AttributeError:  # pragma: no cover - a slotted suite just recomputes
        pass
    return out


# -- MF6 LAW=1 discrete points -------------------------------------------------

class _Tabulated:
    """A yield ``y(E)`` times a weight tabulated at the incident nodes.

    Just enough of a 1-d function for this module: ``xs`` for the axis and
    ``evaluate`` for everything else.
    """

    def __init__(self, yield_form, nodes, weights):
        self._y = yield_form
        self._nodes = np.asarray(nodes, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        grid = _grid_of(yield_form) if yield_form is not None else np.empty(0)
        self.xs = np.unique(np.concatenate([grid, self._nodes]))

    def evaluate(self, energies):
        energies = np.asarray(energies, dtype=float)
        weight = np.interp(energies, self._nodes, self._weights)
        y = _evaluate(self._y, energies) if self._y is not None else np.ones_like(energies)
        return y * weight


class _Node:
    """One incident energy of a spectrum: what ``function1ds`` holds."""

    def __init__(self, energy, xs, ys):
        self.outerDomainValue = float(energy)
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)

    def evaluate(self, energies):
        return np.interp(np.asarray(energies, dtype=float), self.xs, self.ys, left=0.0, right=0.0)


class _Spectrum:
    def __init__(self, nodes):
        self.function1ds = nodes


def _mf6_discrete_points(node) -> Dict[str, List[int]]:
    """``{product label: ND per incident node}`` for the MF6 LAW=1 photons that
    state discrete lines.

    ENDF-6 LAW=1 lets the first ``ND`` outgoing points of each node be discrete
    energies, each with the fraction of the yield it carries, ahead of the
    continuum density. The model has no node for that statement; the decoder
    keeps ``ND`` in the reaction's ENDF provenance, which is where it is read.
    A GNDS or ACE suite has no such provenance (FUDGE writes those lines as
    discrete gammas of their own), so for them this is empty.
    """
    provenance = getattr(node, "provenance", None)
    fields = getattr(provenance, "headerFields", None) or {}
    mf6 = fields.get("mf6") if isinstance(fields, dict) else None
    out: Dict[str, List[int]] = {}
    for record in (mf6 or {}).get("products", []) or []:
        law = record.get("law_fields") or {}
        if law.get("lines"):
            # Already split into products, one per line.
            continue
        nd = law.get("nd") or []
        if record.get("law") == 1 and record.get("pid") == "photon" and any(nd):
            out[record.get("label")] = [int(v) for v in nd]
    return out


def _density_spectrum(energy) -> Tuple[Any, List[int]]:
    """``(spectrum, ND per node)``: an energy table whose nodes are densities.

    A processed table (HDF5, ``Xs_pdf_cdf1d``) states each node as a density
    with its cumulative, opening with ``numberOfDiscrete`` discrete points (ACE
    law 4's ND) whose ``pdf`` is the fraction each carries. Those nodes become
    ``(Eγ, p)`` nodes, a histogram drawn as steps, and their ND is returned so
    the lines are split off as an MF6 LAW=1's are. Any other table is returned
    as it is, with no discrete points.
    """
    nodes = list(energy.function1ds)
    if not nodes or not all(hasattr(f, "pdf") for f in nodes):
        return energy, [0] * len(nodes)
    out, nd = [], []
    for f in nodes:
        n = int(getattr(f, "numberOfDiscrete", 0) or 0)
        xs, ps = np.asarray(f.xs, dtype=float), np.asarray(f.pdf, dtype=float)
        cx, cp = xs[n:], ps[n:]
        if getattr(getattr(f, "interpolation", None), "value", None) == "flat" and cx.size >= 2:
            # Each density holds up to the next point: a step, so the edges repeat.
            cx = np.repeat(cx, 2)[1:-1]
            cp = np.repeat(cp[:-1], 2)
        out.append(_Node(f.outerDomainValue, np.concatenate([xs[:n], cx]),
                         np.concatenate([ps[:n], cp])))
        nd.append(n)
    return _Spectrum(out), nd


def _split_discrete(item, discrete: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """A continuum item whose nodes open with discrete points → lines (one per
    distinct Eγ, weighted node by node; a negative one is a primary photon)
    plus what is left of the continuum. The counts are the item's own (a
    density table's) or its MF6 LAW=1's."""
    if item.get("kind") != "continuum":
        return [item]
    nd = item.pop("nd", None) or discrete.get(item.get("product"))
    if not nd:
        return [item]
    nodes = list(item["spectrum"].function1ds)
    if len(nd) != len(nodes):
        return [item]
    energies = [float(f.outerDomainValue) for f in nodes]
    weights: Dict[int, List[float]] = {}
    exact: Dict[int, float] = {}
    rest: List[_Node] = []
    for i, (f, n) in enumerate(zip(nodes, nd)):
        xs = np.asarray(f.xs, dtype=float)
        ys = np.asarray(f.ys, dtype=float)
        for x, y in zip(xs[:n], ys[:n]):
            key = int(round(x))
            exact.setdefault(key, float(x))
            weights.setdefault(key, [0.0] * len(nodes))[i] += float(y)
        if xs.size - n >= 2:
            rest.append(_Node(f.outerDomainValue, xs[n:], ys[n:]))
    # A negative discrete energy is a primary photon (ENDF-6 §6.2.3): it comes
    # out at |EP| + AWR/(AWR+1)·E, so it is a primary line and not a line at -|EP|.
    out: List[Dict[str, Any]] = [
        {"kind": "primary", "binding": -exact[key], "angular": item.get("angular"),
         "mult": _Tabulated(item["mult"], energies, w)}
        if exact[key] < 0 else
        {"kind": "line", "energy": exact[key], "angular": item.get("angular"),
         "mult": _Tabulated(item["mult"], energies, w)}
        for key, w in sorted(weights.items())
    ]
    if rest:
        out.append({**item, "spectrum": _Spectrum(rest)})
    return out


def _evaluate(form, energies) -> np.ndarray:
    """``form`` at ``energies``; zeros for a missing or unevaluable form."""
    energies = np.asarray(energies, dtype=float)
    if form is None:
        return np.zeros_like(energies)
    try:
        return np.asarray(form.evaluate(energies), dtype=float)
    except Exception:
        return np.zeros_like(energies)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def _mt_text(mt: Optional[int], label: Optional[str] = None) -> str:
    return str(mt) if mt is not None else (label or "none")


def _cascade_key(initial: str, final: str) -> str:
    return f"cascade:{initial}:{final}"


def _line_key(energy: float) -> str:
    """Two reactions' lines at the same energy (to the eV) are one line."""
    return f"line:{round(energy)}"


def _primary_key(contribution, binding: float) -> str:
    return f"primary:{_mt_text(contribution['mt'], contribution['label'])}:{round(binding)}"


def _continuum_key(contribution, index: int) -> str:
    return f"continuum:{_mt_text(contribution['mt'], contribution['label'])}:{index}"


# ---------------------------------------------------------------------------
# Cross sections: the model's, the caller's, or none where only a background
# ---------------------------------------------------------------------------

class _CrossSections:
    """σ_MT(E) for the production quantity, and what was left out of it. One per call."""

    def __init__(self, override, background_mts, resonance_top: float):
        self.override = override
        self.background_mts = background_mts
        self.resonance_top = resonance_top
        self._tables: Dict[Optional[int], Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self.used_override = False
        self.excluded: Dict[Optional[int], str] = {}

    def _table(self, mt):
        """The caller's ``(E, σ)`` for ``mt`` when it gave a mapping; else ``None``."""
        if not isinstance(self.override, Mapping) or mt is None:
            return None
        if mt not in self._tables:
            entry = None
            try:
                entry = self.override.get(mt)
            except Exception:  # a corrupt cache is a miss, not a failed plot
                entry = None
            if entry is not None:
                entry = (np.asarray(entry[0], dtype=float), np.asarray(entry[1], dtype=float))
            self._tables[mt] = entry
        return self._tables[mt]

    def _overridden(self, mt, energies) -> Optional[np.ndarray]:
        if self.override is None or mt is None:
            return None
        table = self._table(mt)
        if table is not None:
            return np.interp(energies, table[0], table[1], left=0.0, right=0.0)
        if callable(self.override) and not isinstance(self.override, Mapping):
            out = self.override(mt, energies)
            return None if out is None else np.asarray(out, dtype=float)
        return None

    def grid(self, contribution) -> np.ndarray:
        table = self._table(contribution["mt"])
        return table[0] if table is not None else contribution["xs_grid"]

    def __call__(self, contribution, energies) -> Optional[np.ndarray]:
        """σ at ``energies``, or ``None`` when the reaction cannot be counted."""
        energies = np.asarray(energies, dtype=float)
        mt = contribution["mt"]
        override = self._overridden(mt, energies)
        if override is not None:
            self.used_override = True
            return override
        if mt is not None and mt in self.background_mts and np.any(energies < self.resonance_top):
            self.excluded[mt] = "background"
            out = self._model(contribution, energies)
            return None if out is None else np.where(energies < self.resonance_top, np.nan, out)
        return self._model(contribution, energies)

    def _model(self, contribution, energies):
        xs = contribution["xs"]
        if xs is None:
            self.excluded[contribution["mt"]] = "no cross section"
            return None
        try:
            return np.asarray(xs.evaluate(energies), dtype=float)
        except Exception:
            self.excluded[contribution["mt"]] = "cross section not evaluable"
            return None

    def source(self) -> str:
        if self.used_override:
            return "reconstructed"
        if self.excluded:
            return "background_only"
        return "file"

    def excluded_rows(self) -> List[Dict[str, Any]]:
        return [{"mt": k, "reason": v}
                for k, v in sorted(self.excluded.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))]


# ---------------------------------------------------------------------------
# Tabulated functions of incident energy
# ---------------------------------------------------------------------------

def _nodes(form2d) -> List[float]:
    return [float(f.outerDomainValue) for f in form2d.function1ds]


def _nearest_node(form2d, energy: float):
    """The tabulated incident energy closest to ``energy`` on a log scale."""
    return min(form2d.function1ds,
               key=lambda f: abs(math.log(max(f.outerDomainValue, 1e-11))
                                 - math.log(max(energy, 1e-11))))


def _moments(function) -> Tuple[float, float]:
    """∫p dE and ∫E·p dE of a tabulated spectrum, trapezoidal."""
    x = np.asarray(function.xs, dtype=float)
    y = np.asarray(function.ys, dtype=float)
    if x.size < 2:
        return 0.0, 0.0
    return float(np.trapezoid(y, x)), float(np.trapezoid(x * y, x))


def _mean_energy(spectrum, energies: np.ndarray) -> np.ndarray:
    """<Eγ>(E) at each tabulated incident energy, linear between them."""
    nodes, means = [], []
    for f in spectrum.function1ds:
        area, first = _moments(f)
        if area > 0:
            nodes.append(float(f.outerDomainValue))
            means.append(first / area)
    if not nodes:
        return np.zeros_like(energies)
    return np.interp(energies, nodes, means)


def _domain(contribution) -> Tuple[float, float]:
    grid = contribution["xs_grid"]
    lo, hi = (float(grid[0]), float(grid[-1])) if grid.size else (math.inf, -math.inf)
    for item in contribution["items"]:
        mult = item.get("mult")
        g = _grid_of(mult) if mult is not None else np.empty(0)
        if g.size:
            lo, hi = min(lo, float(g[0])), max(hi, float(g[-1]))
    # Where σ starts being non-zero, for a threshold reaction.
    xs = contribution["xs"]
    if xs is not None and grid.size:
        nonzero = np.nonzero(_evaluate(xs, grid) > 0)[0]
        if nonzero.size:
            lo = max(lo, float(grid[nonzero[0]]))
    return lo, hi


def _thin(grid: np.ndarray, limit: int) -> np.ndarray:
    if grid.size <= limit:
        return grid
    idx = np.unique(np.round(np.linspace(0, grid.size - 1, limit)).astype(int))
    return grid[idx]


def _decimate(x: np.ndarray, y: np.ndarray, limit: int = MAX_CURVE_POINTS):
    """At most ``limit`` points, keeping each bin's first, min and max."""
    if x.size <= limit:
        return x, y
    bins = limit // 3
    edges = np.linspace(0, x.size, bins + 1).astype(int)
    keep = set()
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        finite = np.isfinite(seg)
        keep.update((a, a + int(np.argmax(np.where(finite, seg, -np.inf))),
                     a + int(np.argmin(np.where(finite, seg, np.inf)))))
    keep.add(x.size - 1)
    idx = np.array(sorted(keep))
    return x[idx], y[idx]


def _sig(value: float) -> float:
    """Six significant digits: a plot needs no more, and the payload halves."""
    return float(f"{value:.6g}")


def _primary_energy(binding: float, awr: Optional[float], energy):
    return binding + (awr / (awr + 1.0) if awr else 1.0) * energy


# ---------------------------------------------------------------------------
# The public object
# ---------------------------------------------------------------------------

class PhotonProduction:
    """The photons of a :class:`~kika.nuclear_data.model.ReactionSuite`.

    Parameters
    ----------
    suite
        Any model reaction suite.
    cross_section
        σ to use in place of the model's: a callable ``(mt, energies) -> values``
        or a mapping ``{mt: (energies, values)}``; ``None`` for a reaction falls
        back to the model. See the module docstring for why a mapping is better
        when there is one.
    background_mts, resonance_top
        The MTs whose model σ is only a background below ``resonance_top`` (eV).
        Without an override they are left out there, NaN and named.

    The photon items are decoded once per suite and cached on it. Cross-section
    bookkeeping (what was overridden, what was left out) is per call.
    """

    def __init__(self, suite, cross_section: Optional[CrossSectionOverride] = None,
                 background_mts: Iterable[int] = (), resonance_top: float = 0.0):
        self.suite = suite
        self.cross_section = cross_section
        self.background_mts = frozenset(int(mt) for mt in background_mts)
        self.resonance_top = float(resonance_top or 0.0)
        self._awr: Any = ...

    # -- what there is ------------------------------------------------------

    def contributions(self) -> List[Dict[str, Any]]:
        """The decoded photon items per reaction, **model objects included**.

        ``[{"mt", "label", "kind": "reaction"|"orphan", "xs", "xs_grid",
        "items": [...]}]``. For code that wants the forms themselves; the
        JSON-friendly summary is :meth:`reactions`.
        """
        return _decode(self.suite)

    def reactions(self) -> List[Dict[str, Any]]:
        """Each photon-producing reaction and what it emits, counted.

        ``[{"mt", "label", "kind", "lines", "primaries", "cascade_lines",
        "continua", "unsupported", "anisotropic"}]``: ``lines`` counts discrete
        lines, ``cascade_lines`` the lines of every cascade the reaction starts,
        ``anisotropic`` the lines and primaries with an anisotropic angular
        distribution.
        """
        out = []
        for c in self.contributions():
            counts = {"lines": 0, "primaries": 0, "cascade_lines": 0, "continua": 0,
                      "unsupported": 0, "anisotropic": 0}
            for item in c["items"]:
                kind = item["kind"]
                if kind == "cascade":
                    counts["cascade_lines"] += len(item["lines"])
                elif kind == "line":
                    counts["lines"] += 1
                elif kind == "primary":
                    counts["primaries"] += 1
                elif kind == "continuum":
                    counts["continua"] += 1
                else:
                    counts["unsupported"] += 1
                if _angular_kind(item.get("angular")) == "anisotropic":
                    counts["anisotropic"] += 1
            out.append({"mt": c["mt"], "label": c["label"], "kind": c["kind"], **counts})
        return out

    @property
    def awr(self) -> Optional[float]:
        """The target's mass in neutron masses, for a primary photon's recoil."""
        if self._awr is ...:
            self._awr = _target_awr(self.suite)
        return self._awr

    # -- selection ----------------------------------------------------------

    def _select(self, mt: Optional[int]):
        contributions = self.contributions()
        if mt is None:
            return contributions
        chosen = [c for c in contributions if c["mt"] == mt]
        if not chosen:
            raise PhotonProductionError(f"MT{mt} emits no photons in this evaluation")
        return chosen

    def _sigma(self) -> _CrossSections:
        return _CrossSections(self.cross_section, self.background_mts, self.resonance_top)

    @staticmethod
    def _check(quantity: str, mt: Optional[int]) -> None:
        if quantity not in QUANTITIES:
            raise PhotonProductionError(f"unknown quantity {quantity!r}; use one of {QUANTITIES}")
        if quantity == "multiplicity" and mt is None:
            raise PhotonProductionError("photons per reaction need one reaction (an mt)")

    def _empty_sigma_fields(self, sigma: Optional[_CrossSections], quantity: str) -> Dict[str, Any]:
        if sigma is None or quantity != "production":
            return {"xs_source": None, "excluded": [], "resonance_top_ev": None}
        return {"xs_source": sigma.source(), "excluded": sigma.excluded_rows(),
                "resonance_top_ev": sigma.resonance_top or None}

    # -- the incident-energy axis -------------------------------------------

    def incident_axis(self, mt: Optional[int] = None) -> Dict[str, Any]:
        """The incident energies worth stepping through, for a slider.

        Thresholds, multiplicity nodes and continuum nodes, never the cross
        sections' resonance grids, plus 121 log-spaced points so decades the
        tables leave empty are still walked smoothly. At most 1500 points.

        ``thresholds`` are the energies where a reaction starts emitting, the
        stops at which the spectrum gains lines (between two of them it only
        changes in scale), lowest first: ``[{"energy_ev", "mts": [...]}]``.

        ``{"grid": [eV...], "min": eV|None, "max": eV|None, "thresholds": [...]}``.
        """
        points = set()
        lo, hi = math.inf, -math.inf
        starts: Dict[float, List[Any]] = {}
        for c in self._select(mt):
            dlo, dhi = _domain(c)
            if dlo <= dhi:
                lo, hi = min(lo, dlo), max(hi, dhi)
                points.update((dlo, dhi))
                starts.setdefault(float(f"{dlo:.10g}"), []).append(
                    c["mt"] if c["mt"] is not None else c["label"])
            for item in c["items"]:
                mult = item.get("mult")
                if mult is not None:
                    points.update(_grid_of(mult).tolist())
                if item["kind"] == "continuum":
                    points.update(_nodes(item["spectrum"]))
        if not math.isfinite(lo):
            return {"grid": [], "min": None, "max": None, "thresholds": []}
        grid = np.array(sorted(p for p in points if lo <= p <= hi and p > 0))
        decades = np.logspace(math.log10(max(lo, 1e-5)), math.log10(hi), 121)
        # Ten digits: a decade point and a tabulated one that differ only in the
        # last bit of a float are one stop on the slider, not two.
        grid = np.unique(np.array([float(f"{v:.10g}") for v in np.concatenate([grid, decades])]))
        thresholds = [{"energy_ev": e, "mts": starts[e]} for e in sorted(starts)]
        return {"grid": _thin(grid, 1500).tolist(), "min": lo, "max": hi, "thresholds": thresholds}

    # -- the spectrum at one incident energy ---------------------------------

    def spectrum(self, incident_energy: float, mt: Optional[int] = None,
                 quantity: str = "production") -> Dict[str, Any]:
        """Lines and continua at ``incident_energy`` (eV), per reaction or summed.

        Returns ``{"incident_energy", "mt", "quantity", "lines", "continua",
        "totals", "xs_source", "excluded", "resonance_top_ev", "warnings"}``.

        Each line is ``{"key", "energy_ev", "value", "origin":
        "cascade"|"line"|"primary", "initial", "final", "by_mt": [{"mt",
        "value"}], "angular": "isotropic"|"anisotropic"|None}``, sorted by
        decreasing photon energy; ``value`` is photons per reaction or barns.
        A line several reactions emit is anisotropic when any of them says so.

        Each continuum is ``{"key", "mt", "label", "node_energy_ev",
        "energy_ev": [Eγ], "value": [per eV]}``, drawn at the nearest
        tabulated incident energy, and only inside the tabulated range: outside
        it the spectrum is not stated, and the nearest node would be a guess.

        ``totals`` holds ``line_photons``, ``line_energy_ev``,
        ``continuum_photons``, ``continuum_energy_ev``.
        """
        self._check(quantity, mt)
        contributions = self._select(mt)
        awr = self.awr
        sigma = self._sigma()
        E = float(incident_energy)

        lines: Dict[str, Dict[str, Any]] = {}
        continua: List[Dict[str, Any]] = []
        warnings: List[str] = []

        def add_line(key, energy, value, origin, contribution, angular, initial=None, final=None):
            if value <= 0 or not math.isfinite(value):
                return
            row = lines.get(key)
            if row is None:
                row = lines[key] = {"key": key, "energy_ev": energy, "value": 0.0, "origin": origin,
                                    "initial": initial, "final": final, "by_mt": [],
                                    "angular": angular}
            elif angular == "anisotropic" or row["angular"] is None:
                row["angular"] = angular if angular is not None else row["angular"]
            row["value"] += value
            row["by_mt"].append({"mt": contribution["mt"], "value": value})

        for c in contributions:
            if quantity == "production":
                s = sigma(c, [E])
                factor = None if s is None else float(s[0])
                if factor is None or not math.isfinite(factor):
                    continue
            else:
                factor = 1.0
            # A level reaction below its threshold emits nothing, even per reaction.
            if quantity == "multiplicity" and c["xs"] is not None and _evaluate(c["xs"], [E])[0] <= 0:
                factor = 0.0
            if factor <= 0:
                continue
            for item in c["items"]:
                kind = item["kind"]
                if kind == "cascade":
                    for line in item["lines"]:
                        add_line(_cascade_key(line.initial, line.final), line.energy,
                                 factor * line.photons, "cascade", c, "isotropic",
                                 line.initial, line.final)
                elif kind == "line":
                    y = float(_evaluate(item["mult"], [E])[0])
                    add_line(_line_key(item["energy"]), item["energy"], factor * y, "line", c,
                             _angular_kind(item.get("angular")))
                elif kind == "primary":
                    y = float(_evaluate(item["mult"], [E])[0])
                    add_line(_primary_key(c, item["binding"]),
                             float(_primary_energy(item["binding"], awr, E)), factor * y,
                             "primary", c, _angular_kind(item.get("angular")))
                elif kind == "continuum":
                    y = float(_evaluate(item["mult"], [E])[0])
                    nodes = _nodes(item["spectrum"])
                    if y <= 0 or not nodes or not nodes[0] <= E <= nodes[-1]:
                        continue
                    node = _nearest_node(item["spectrum"], E)
                    continua.append({
                        "key": _continuum_key(c, item["index"]),
                        "mt": c["mt"], "label": c["label"],
                        "node_energy_ev": float(node.outerDomainValue),
                        "energy_ev": [float(x) for x in node.xs],
                        "value": [float(v) * factor * y for v in node.ys],
                    })
                elif kind == "unsupported":
                    warnings.append(f"MT{c['mt']}: a photon whose energy is {item['what']} "
                                    f"is not drawn")

        rows = sorted(lines.values(), key=lambda r: -r["energy_ev"])
        line_photons = sum(r["value"] for r in rows)
        line_energy = sum(r["value"] * r["energy_ev"] for r in rows)
        cont_photons = cont_energy = 0.0
        for cont in continua:
            x = np.asarray(cont["energy_ev"])
            y = np.asarray(cont["value"])
            if x.size > 1:
                cont_photons += float(np.trapezoid(y, x))
                cont_energy += float(np.trapezoid(x * y, x))

        return {
            "incident_energy": E, "mt": mt, "quantity": quantity,
            "lines": rows, "continua": continua,
            "totals": {"line_photons": line_photons, "line_energy_ev": line_energy,
                       "continuum_photons": cont_photons, "continuum_energy_ev": cont_energy},
            **self._empty_sigma_fields(sigma, quantity),
            "warnings": sorted(set(warnings)),
        }

    # -- curves over incident energy ------------------------------------------

    def _curve_grid(self, contribution, sigma: Optional[_CrossSections]) -> np.ndarray:
        parts = [sigma.grid(contribution) if sigma is not None else contribution["xs_grid"]]
        for item in contribution["items"]:
            mult = item.get("mult")
            if mult is not None:
                parts.append(_grid_of(mult))
            if item["kind"] == "continuum":
                parts.append(np.asarray(_nodes(item["spectrum"])))
        grid = np.unique(np.concatenate([p for p in parts if p.size] or [np.empty(0)]))
        return grid[grid > 0]

    def curves(self, mt: Optional[int] = None, quantity: str = "production",
               max_lines: int = MAX_LINE_CURVES,
               max_points: int = MAX_CURVE_POINTS) -> Dict[str, Any]:
        """Each line's multiplicity or production cross section against E, plus totals.

        Returns ``{"mt", "quantity", "lines", "continua", "total", "energy",
        "energy_parts", "xs_source", "excluded", "resonance_top_ev",
        "warnings"}``. Every curve is ``{"label", "key", "energy_ev": [E],
        "value": [y or None]}`` (``None`` where a reaction is left out),
        decimated to ``max_points`` keeping each bin's extremes.

        * ``lines``: the ``max_lines`` strongest by peak, each with the same
          ``key`` :meth:`spectrum` gives it, plus ``gamma_energy_ev``,
          ``origin`` and ``peak``; the rest summed into one curve with key
          ``other`` and origin ``other``.
        * ``continua``: one per continuum photon, with its ``mt``.
        * ``total``: every photon, lines and continua (key ``total``).
        * ``energy``: Σ Y·Eγ, the gamma energy released, eV per reaction or
          b·eV, a continuum counted at its mean energy (key ``energy``).
        * ``energy_parts``: ``energy`` split. With ``mt`` None, per reaction
          (keys ``mt:<mt>``) for the :data:`MAX_ENERGY_PARTS` reactions with
          the highest peak, the rest summed as ``other``; with ``mt`` given,
          into ``lines`` (cascades, discrete and primary lines) and
          ``continuum``.

        A primary photon's ``gamma_energy_ev`` is its mean over the grid, since
        its energy moves with E; its energy curve uses the moving energy.
        """
        self._check(quantity, mt)
        contributions = self._select(mt)
        awr = self.awr
        sigma = self._sigma() if quantity == "production" else None

        grid = np.unique(np.concatenate(
            [self._curve_grid(c, sigma) for c in contributions] or [np.empty(0)]))
        if grid.size == 0:
            return {"mt": mt, "quantity": quantity, "lines": [], "continua": [],
                    "total": None, "energy": None, "energy_parts": [],
                    **self._empty_sigma_fields(None, quantity), "warnings": []}
        grid = _thin(grid, MAX_CURVE_GRID)

        lines: Dict[str, Dict[str, Any]] = {}
        continua: List[Dict[str, Any]] = []
        total = np.zeros_like(grid)
        energy = np.zeros_like(grid)
        # Per reaction, and per kind, the energy each carries; NaN kept, so a
        # reaction left out below the resonance top shows as a gap.
        per_mt: List[Tuple[Dict[str, Any], np.ndarray]] = []
        line_energy = np.zeros_like(grid)
        cont_energy = np.zeros_like(grid)

        def accumulate(key, y, eg, origin, label, energies=None):
            row = lines.get(key)
            if row is None:
                row = lines[key] = {"key": key, "y": np.zeros_like(grid),
                                    "eg": np.full_like(grid, eg), "origin": origin, "label": label}
            row["y"] = row["y"] + y
            if energies is not None:
                row["eg"] = np.asarray(energies, dtype=float)

        for c in contributions:
            if sigma is not None:
                factor = sigma(c, grid)
                if factor is None:
                    continue
            else:
                factor = np.ones_like(grid)
                if c["xs"] is not None:
                    factor = np.where(_evaluate(c["xs"], grid) > 0, 1.0, 0.0)
            mine = np.zeros_like(grid)
            for item in c["items"]:
                kind = item["kind"]
                if kind == "cascade":
                    for line in item["lines"]:
                        y = factor * line.photons
                        accumulate(_cascade_key(line.initial, line.final), y, line.energy,
                                   "cascade",
                                   f"{line.energy / 1e3:.1f} keV ({line.initial} → {line.final})")
                        mine = mine + y * line.energy
                        line_energy = line_energy + y * line.energy
                elif kind in ("line", "primary"):
                    y = factor * _evaluate(item["mult"], grid)
                    if kind == "line":
                        e = item["energy"]
                        accumulate(_line_key(e), y, e, "line", f"{e / 1e3:.1f} keV")
                        eg = e
                    else:
                        e = _primary_energy(item["binding"], awr, 0.0)
                        eg = _primary_energy(item["binding"], awr, grid)
                        accumulate(_primary_key(c, item["binding"]), y, e, "primary",
                                   f"primary, {e / 1e3:.1f} keV + E·A/(A+1)", energies=eg)
                    mine = mine + y * eg
                    line_energy = line_energy + y * eg
                elif kind == "continuum":
                    y = factor * _evaluate(item["mult"], grid)
                    mean = _mean_energy(item["spectrum"], grid)
                    continua.append({"key": _continuum_key(c, item["index"]), "mt": c["mt"],
                                     "label": f"continuum, MT{_mt_text(c['mt'], c['label'])}",
                                     "y": y})
                    total += np.nan_to_num(y)
                    energy += np.nan_to_num(y * mean)
                    mine = mine + y * mean
                    cont_energy = cont_energy + y * mean
            per_mt.append((c, mine))

        for row in lines.values():
            total += np.nan_to_num(row["y"])
            energy += np.nan_to_num(row["y"] * row["eg"])

        def peak(y) -> float:
            finite = y[np.isfinite(y)] if y.size else y
            return float(finite.max()) if finite.size else 0.0

        ranked = sorted(lines.values(), key=lambda r: -peak(r["y"]))
        shown, rest = ranked[:max_lines], ranked[max_lines:]

        def curve(label, key, y, **extra):
            x, yy = _decimate(grid, y, max_points)
            return {"label": label, "key": key, "energy_ev": [_sig(v) for v in x],
                    "value": [None if not math.isfinite(v) else _sig(v) for v in yy], **extra}

        out_lines = [curve(r["label"], r["key"], r["y"], gamma_energy_ev=float(np.mean(r["eg"])),
                           origin=r["origin"], peak=peak(r["y"]))
                     for r in shown]
        if rest:
            other = np.sum([np.nan_to_num(r["y"]) for r in rest], axis=0)
            out_lines.append(curve(f"{len(rest)} weaker lines", "other", other,
                                   gamma_energy_ev=None, origin="other", peak=peak(other)))

        background = np.zeros_like(grid, dtype=bool)
        if sigma is not None and sigma.excluded:
            background = grid < sigma.resonance_top
        total = np.where(background, np.nan, total)
        energy = np.where(background, np.nan, energy)

        if mt is None:
            ranked_mt = sorted(per_mt, key=lambda cm: -peak(cm[1]))
            parts = [curve(f"MT{_mt_text(c['mt'], c['label'])}", f"mt:{_mt_text(c['mt'], c['label'])}",
                           y, mt=c["mt"])
                     for c, y in ranked_mt[:MAX_ENERGY_PARTS]]
            others = ranked_mt[MAX_ENERGY_PARTS:]
            if others:
                other = np.sum([np.nan_to_num(y) for _, y in others], axis=0)
                parts.append(curve(f"{len(others)} other reactions", "other",
                                   np.where(background, np.nan, other), mt=None))
        else:
            parts = [curve("lines", "lines", line_energy), curve("continuum", "continuum", cont_energy)]
            parts = [p for p, y in zip(parts, (line_energy, cont_energy)) if np.any(np.nan_to_num(y))]

        return {
            "mt": mt, "quantity": quantity,
            "lines": out_lines,
            "continua": [curve(cn["label"], cn["key"], cn["y"], mt=cn["mt"]) for cn in continua],
            "total": curve("all photons", "total", total),
            "energy": curve("gamma energy", "energy", energy),
            "energy_parts": parts,
            **self._empty_sigma_fields(sigma, quantity),
            "warnings": [],
        }

    # -- one curve, for the plotter ---------------------------------------------

    def curve(self, line: str = "total", mt: Optional[int] = None,
              quantity: str = "production") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """One curve of :meth:`curves` as arrays, undecimated: ``(E, y, info)``.

        ``line`` is a line key, a continuum key, ``'total'`` or ``'energy'``.
        ``info`` carries ``label``, ``xs_source``, ``excluded`` and
        ``resonance_top_ev``. NaN marks where a reaction was left out.
        """
        full = self.curves(mt=mt, quantity=quantity, max_lines=10 ** 9, max_points=10 ** 9)
        if line in ("total", "energy"):
            found = full[line]
        else:
            found = next((c for c in full["lines"] + full["continua"] if c["key"] == line), None)
        if found is None:
            known = [c["key"] for c in full["lines"][:12]]
            raise PhotonProductionError(
                f"no photon {line!r}" + (f" in MT{mt}" if mt is not None else "")
                + (f"; some there are: {', '.join(known)}" if known else ""))
        x = np.asarray(found["energy_ev"], dtype=float)
        y = np.array([np.nan if v is None else v for v in found["value"]], dtype=float)
        info = {"label": found["label"], "xs_source": full["xs_source"],
                "excluded": full["excluded"], "resonance_top_ev": full["resonance_top_ev"]}
        return x, y, info

    # -- angular distribution of a line -----------------------------------------

    def angular(self, key: str, incident_energy: float, mt: Optional[int] = None) -> Dict[str, Any]:
        """The angular distribution of the line ``key`` at ``incident_energy`` (eV).

        Anisotropic photons (ENDF MF14 LTT=1/2: an ``XYs2d`` of ``Legendre`` or of
        ``XYs1d``) are drawn at the *nearest tabulated* incident energy, for the
        reason continua are. Returns ``{"key", "mt", "form":
        "legendre"|"tabulated"|"isotropic"|None, "frame": "lab",
        "node_energy_ev", "mu": [201 cosines], "pdf": [...], "coefficients":
        [a_0, a_1, ...] (legendre only)}``. ``form`` is ``None`` when the file
        states no angular distribution for the photon (no MF14), and then
        ``mu`` and ``pdf`` are ``None`` too.

        A ``line:`` key several reactions emit is taken from ``mt`` when given,
        and otherwise from the first reaction whose photon is anisotropic.
        """
        E = float(incident_energy)
        mu = np.linspace(-1.0, 1.0, ANGULAR_POINTS)
        base: Dict[str, Any] = {"key": key, "incident_energy": E, "frame": "lab",
                                "node_energy_ev": None, "coefficients": None}
        candidates = []
        for c in self._select(mt):
            for item in c["items"]:
                kind = item["kind"]
                if kind == "cascade":
                    if any(_cascade_key(l.initial, l.final) == key for l in item["lines"]):
                        candidates.append((c, "isotropic", None))
                elif kind == "line" and _line_key(item["energy"]) == key:
                    candidates.append((c, _angular_kind(item.get("angular")), item.get("angular")))
                elif kind == "primary" and _primary_key(c, item["binding"]) == key:
                    candidates.append((c, _angular_kind(item.get("angular")), item.get("angular")))
        if not candidates:
            raise PhotonProductionError(f"no line {key!r}" + (f" in MT{mt}" if mt is not None else ""))
        chosen = next((cand for cand in candidates if cand[1] == "anisotropic"),
                      next((cand for cand in candidates if cand[1] is not None), candidates[0]))
        c, kind, form = chosen
        base["mt"] = c["mt"]
        if kind is None:
            return {**base, "form": None, "mu": None, "pdf": None}
        if kind == "isotropic":
            return {**base, "form": "isotropic", "mu": mu.tolist(),
                    "pdf": np.full_like(mu, 0.5).tolist()}
        node = _nearest_node(form, E)
        pdf = np.asarray(node.evaluate(mu), dtype=float)
        out = {**base, "node_energy_ev": float(node.outerDomainValue), "mu": mu.tolist(),
               "pdf": pdf.tolist()}
        if type(node).__name__ == "Legendre":
            return {**out, "form": "legendre",
                    "coefficients": [float(a) for a in node.coefficients]}
        return {**out, "form": "tabulated"}

    # -- continuum spectra over incident energy -----------------------------------

    def continuum_map(self, mt: int, scaled: bool = False,
                      points: int = CONTINUUM_POINTS) -> Dict[str, Any]:
        """Every tabulated p(Eγ | E) of ``mt``'s continuum photons, on a common Eγ grid.

        Nothing is interpolated in E: each row is one tabulated incident
        energy, as the file states it. Along Eγ each node is evaluated under
        its own interpolation law on a grid of ``points`` energies, log-spaced
        when the range spans more than two decades from its smallest positive
        energy, linear otherwise; ``grid`` says which. With ``scaled`` each row
        is multiplied by the multiplicity Y(E) at its node, so it reads photons
        per eV per reaction instead of a probability density per eV.

        Returns ``{"mt", "scaled", "continua": [{"key", "label",
        "incident_energy_ev": [E_i], "photon_energy_ev": [Eγ_j], "grid":
        "log"|"linear", "multiplicity": [Y(E_i)], "value": [[row per E_i]]}]}``.
        """
        contributions = self._select(mt)
        out = []
        for c in contributions:
            for item in c["items"]:
                if item["kind"] != "continuum":
                    continue
                spectrum = item["spectrum"]
                nodes = list(spectrum.function1ds)
                xs_all = np.concatenate([np.asarray(f.xs, dtype=float) for f in nodes]) \
                    if nodes else np.empty(0)
                if xs_all.size == 0:
                    continue
                hi = float(xs_all.max())
                positive = xs_all[xs_all > 0]
                lo_pos = float(positive.min()) if positive.size else 0.0
                if lo_pos > 0 and hi / lo_pos > 100.0:
                    egrid, law = np.geomspace(lo_pos, hi, points), "log"
                else:
                    egrid, law = np.linspace(float(xs_all.min()), hi, points), "linear"
                incident = np.array([float(f.outerDomainValue) for f in nodes])
                mult = _evaluate(item["mult"], incident)
                rows = []
                for f, y in zip(nodes, mult):
                    try:
                        values = np.asarray(f.evaluate(egrid), dtype=float)
                    except Exception:
                        values = np.interp(egrid, np.asarray(f.xs, dtype=float),
                                           np.asarray(f.ys, dtype=float), left=0.0, right=0.0)
                    if scaled:
                        values = values * y
                    rows.append([_sig(v) for v in np.nan_to_num(values)])
                out.append({
                    "key": _continuum_key(c, item["index"]), "label": c["label"],
                    "incident_energy_ev": incident.tolist(),
                    "photon_energy_ev": [_sig(v) for v in egrid], "grid": law,
                    "multiplicity": [float(v) for v in mult], "value": rows,
                })
        if not out:
            raise PhotonProductionError(f"MT{mt} has no continuum photon")
        return {"mt": mt, "scaled": bool(scaled), "continua": out}


# ---------------------------------------------------------------------------
# An ENDF tape's background reactions
# ---------------------------------------------------------------------------

def _has_real_resonance_ranges(mf2) -> bool:
    """MF2 with at least one resolved or unresolved range that holds parameters.

    Some evaluations (H-1, H-2) carry an MF2 stub only for the LRP header; for
    those MF3 is the whole cross section, and nothing is a background.
    """
    try:
        mt151 = getattr(mf2, "mt", {}).get(151, None)
        for iso in getattr(mt151, "isotopes", []) or []:
            for rng in getattr(iso, "energy_ranges", []) or []:
                if getattr(rng, "lru", 0) not in (1, 2):
                    continue
                params = getattr(rng, "parameters", None)
                for lb in getattr(params, "l_values", []) or []:
                    if getattr(lb, "resonances", None) or getattr(lb, "j_states", None):
                        return True
    except Exception:
        return False
    return False


def _has_fission_widths(mf2) -> bool:
    try:
        for iso in getattr(mf2.mt.get(151), "isotopes", []) or []:
            for rng in getattr(iso, "energy_ranges", []) or []:
                params = getattr(rng, "parameters", None)
                for lb in getattr(params, "l_values", []) or []:
                    if any(getattr(res, "c6", 0.0) != 0.0 for res in getattr(lb, "resonances", []) or []):
                        return True
                    for js in getattr(lb, "j_states", []) or []:
                        gf = getattr(js, "gf", None)
                        if isinstance(gf, list) and any(v != 0.0 for v in gf):
                            return True
                        if any(getattr(ep, "gf", 0.0) != 0.0 for ep in getattr(js, "energy_points", []) or []):
                            return True
    except Exception:
        return False
    return False


def resonance_background(endf) -> Tuple[List[int], float]:
    """``(mts, top)``: the MTs whose MF3 is only a background, and up to where.

    ``endf`` is a flat ENDF object (:func:`kika.endf.read_endf`). The MTs are
    the ones a resonance reconstruction completes (elastic, capture, fission
    when MF2 states fission widths), together with the sums that hold them
    (total MT1, nonelastic MT3); ``top`` is the highest upper bound of a
    resolved or unresolved range, in eV. A tape without real resonance ranges
    returns ``([], 0.0)``. This is the rule KIKA's ENDF panels use, kept here
    so the photon quantity in :mod:`kika.plotting` and the app agree.
    """
    mf2 = getattr(endf, "mf", {}).get(2)
    if mf2 is None or not _has_real_resonance_ranges(mf2):
        return [], 0.0
    mts = {1, 2, 3, 102}
    if _has_fission_widths(mf2):
        mts.add(18)
    top = 0.0
    for iso in getattr(mf2.mt.get(151), "isotopes", []) or []:
        for rng in getattr(iso, "energy_ranges", []) or []:
            if getattr(rng, "lru", 0) in (1, 2):
                top = max(top, float(getattr(rng, "eh", 0.0) or 0.0))
    return sorted(mts), top

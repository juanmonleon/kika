"""MF6 LAW=1 photons whose first ``ND`` outgoing points are discrete lines (D40).

ENDF-6 §6.2.3 lets a LAW=1 LANG=1 record open each incident node with ``ND``
discrete energies, each with the fraction of the yield it carries, ahead of the
continuum density. 6 271 photon subsections of ENDF/B-VIII.1 do it (194 tapes:
the MT 6xx-8xx level gammas of Cl-35, Au-197 and most of the structural and
light nuclides, capture in Pt and Ta-180m). The model used to keep those lines
as points of the spectrum, which is a table whose x *decreases*; no GNDS reader
takes one, so the GNDS writer left 137 distributions of Cl-35 empty.

**The split is FUDGE's** (``ENDF_ITYPE_0_Misc.readMF6``, «Divide gammas into
primary and discrete»), measured product by product against
``ENDF-B-VIII.1-GNDS``:

* the continuum, if any node has one, is one ``photon`` product whose spectrum
  is the continuum part of each node **normalised**, and whose multiplicity is
  ``y(E)`` times the integral the normalisation took out;
* each line is a product of its own, ``Uncorrelated(isotropic2d,
  discreteGamma)``, multiplicity ``P(E)·y(E)`` at the nodes that list it, the
  lines sorted by energy and keyed ``(E_γ, n)`` so a line a node lists twice is
  two products, as FUDGE's ``duplicateCounter`` makes it;
* a negative energy is a **primary** photon, ``primaryGamma`` with the binding
  energy as its value: the file writes ``-(E_B + A/(A+1)·E)`` at each incident
  energy (Pt-190 MT102: 5 812 615 eV at 2 keV and 25 706 980 eV at 20 MeV) and
  FUDGE takes the shift back out above 0.01 eV;
* labels follow ``uniqueLabel``: ``photon``, ``photon__a`` … in that order.

Deviation, declared: a line a node does not list *between* its first and last
node gets a zero there, where FUDGE's table simply skips the node and so
interpolates through it. None of the library's lines does that today.

**The way back is byte for byte.** What the split cannot keep in the model --
which line sits where in each node, the probabilities of a node whose yield is
zero (412 nodes in ENDF/B-VIII.1), a continuum whose integral is zero (211) --
stays in the record, with the node values as read and a SHA-256 of the pieces.
While the pieces are what the decoder made, the encoder writes the node values
back (the precedent is MF13, ``photons.py``); once the model changes, the nodes
are rebuilt from it and the report says so.

The yield itself goes in the suite as FUDGE puts it, a ``multiplicitySum``
"total gamma multiplicity" over the pieces (``decode._totalLineMultiplicity``),
so a suite with no ENDF provenance still knows it: :func:`deriveLineSpectrum`
regroups the pieces into one subsection the way FUDGE's ``gammasToENDF6_MF6``
does, and on Pt-190's capture every number it writes is the file's.

Consumers that read the section as ENDF states it -- the HDF5 and G4NDL writers,
which put the lines in the table as ACER does -- get the single product back
from :func:`mergedLineSpectra`.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["isLineSpectrum", "decodeLineSpectrum", "restoreLineSpectrum",
           "lineSpectrumLabels", "mergedLineSpectra", "isLinePiece",
           "deriveLineSpectrum", "LINES_KEY"]

#: The key of a split record's bookkeeping in its ``law_fields``.
LINES_KEY = "lines"

#: FUDGE's tolerance on the binding energy a primary photon gives at each node.
_PRIMARY_TOLERANCE = 1e-4

#: FUDGE takes the projectile term out of a primary only above this energy.
_PRIMARY_SHIFT_FLOOR = 1e-2


def isLineSpectrum(product) -> bool:
    """A photon LAW=1 LANG=1 subsection that states discrete lines and no angle."""
    from kika.endf.classes.mf6.laws import MF6LawContinuum

    body = getattr(product, "law_data", None)
    return (int(product.law) == 1 and isinstance(body, MF6LawContinuum)
            and int(body.lang) == 1 and int(round(float(product.zap))) == 0
            and any(body.nd) and not any(body.na))


def lineSpectrumLabels(record: dict) -> List[str]:
    """Every model product a split record became, in the order it made them."""
    meta = (record.get("law_fields") or {}).get(LINES_KEY)
    if not meta:
        return [record["label"]] if record.get("label") is not None else []
    return [piece["label"] for piece in meta["pieces"]]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _yieldOf(record_or_product):
    """``y(E)`` of the subsection, as the decoder builds it."""
    from kika.nuclear_data.model import multiplicityAxes

    from .energy_angle import _multiplicityOf

    return _multiplicityOf(record_or_product, multiplicityAxes()).form


def _yieldAtNodes(y, energies: List[float]) -> List[float]:
    """``y`` at each incident node; the first of a repeated energy takes the left limit.

    A TAB2 may state one incident energy twice -- a step, 8 nodes in JEFF-4.0
    and 9 in JENDL-5 (Fe-58 and the Hf isotopes' capture) -- and evaluating at
    the energy itself gives both nodes the right-hand value.
    """
    out = []
    for k, e in enumerate(energies):
        if k + 1 < len(energies) and energies[k + 1] == e:
            e = float(np.nextafter(e, -np.inf))
        out.append(float(y.evaluate(e)))
    return out


def _pointsByNode(meta) -> Dict[str, List[int]]:
    """The incident nodes each piece's multiplicity has a point at, in order.

    A line has one at every node from the first that lists it to the last (a
    zero where a node in between does not); a primary at the nodes that list it.
    """
    present: Dict[str, List[int]] = {}
    for k, names in enumerate(meta["nodes"]):
        for name in names:
            present.setdefault(name, []).append(k)
    out = {}
    for piece in meta["pieces"]:
        nodes = present.get(piece["label"], [])
        if piece["kind"] == "discrete" and nodes:
            nodes = list(range(nodes[0], nodes[-1] + 1))
        out[piece["label"]] = nodes
    return out


def _valueAt(function, energy: float, k: int, nodes: List[int]) -> float:
    """The piece's multiplicity at node *k*: its own point when the table still has
    one per node (a repeated incident energy is two points), else evaluated."""
    xs = getattr(function, "xs", None)
    if xs is not None and len(xs) == len(nodes) and k in nodes:
        return float(np.asarray(function.ys, dtype=float)[nodes.index(k)])
    return float(function.evaluate(energy))


def _lineKeys(energies: np.ndarray) -> List[Tuple[float, int]]:
    """``(E_γ, n)`` per line of one node: ``n`` counts a repeat of the energy before it."""
    keys, prior, count = [], None, 0
    for eg in energies:
        count = count + 1 if eg == prior else 1
        keys.append((float(eg), count))
        prior = eg
    return keys


def _binding(eg: float, energy: float, ratio: float, shift: bool) -> float:
    value = -float(eg)
    if shift and energy > _PRIMARY_SHIFT_FLOOR:
        value -= ratio * energy
    return value


def _primaryValues(nodes, ratio: float):
    """``(values, shift)`` for the primaries, or ``None`` where no reading holds.

    FUDGE's reading first (the projectile term in the file), then the JENDL one
    (the binding energy itself); a primary whose binding energy moves by more
    than 1e-4 under both is not one this split can name.
    """
    counts = {len(p) for _, p in nodes if p is not None}
    if len(counts) > 1:
        return None
    for shift in (True, False):
        values = None
        ok = True
        for energy, primaries in nodes:
            if not primaries:
                continue
            here = [_binding(eg, energy, ratio, shift) for eg in primaries]
            if values is None:
                values = here
                continue
            if any(abs(a - b) > _PRIMARY_TOLERANCE * abs(a) for a, b in zip(values, here)):
                ok = False
                break
        if ok:
            return values, shift
    return None


def decodeLineSpectrum(product, index: int, mt: int, frame, awr: Optional[float],
                       taken: set, report):
    """One line-spectrum subsection → ``(pieces, meta)``, or ``None`` to keep it whole.

    ``pieces`` is a list of ``(label, multiplicity, distribution)``, continuum
    first; ``meta`` goes in the record's ``law_fields`` under :data:`LINES_KEY`.
    ``taken`` holds the labels already given in this section and is extended.
    """
    from kika.nuclear_data.model import (ENDF_INT_TO_INTERPOLATION, DiscreteGamma,
                                         Isotropic2d, Multiplicity, PrimaryGamma,
                                         Uncorrelated, XYs1d, energyAxes, fromEndfTab2,
                                         multiplicityAxes)
    from kika.nuclear_data.model.functions.integration import integrateFunction1d

    from .residuals import _uniqueLabel

    body = product.law_data
    energies = [float(e) for e in body.incident_energies]
    if len(body.tab2_interp) > 1:
        report.lost(f"MF6/MT{mt} product {index}: discrete lines on a TAB2 of "
                    f"{len(body.tab2_interp)} interpolation regions; kept as one "
                    f"spectrum", unaffectedScopes=("cross-sections",))
        return None
    inner = ENDF_INT_TO_INTERPOLATION.get(int(body.lep), "lin-lin")
    outer = int(body.tab2_interp[0][1]) if body.tab2_interp else 2
    y = _yieldOf(product)
    yAt = _yieldAtNodes(y, energies)
    ratio = float(awr) / (1.0 + float(awr)) if awr else 0.0

    discrete: Dict[Tuple[float, int], List[Tuple[int, float]]] = {}
    primaries, continuum, nodeKeys = [], [], []
    for k, energy in enumerate(energies):
        table = body.block(k)
        nd = int(body.nd[k])
        keys, pAt = [], []
        for (eg, count), p in zip(_lineKeys(table[:nd, 0]), table[:nd, 1]):
            if eg < 0:
                pAt.append(eg)
                keys.append(("primary", len(pAt) - 1))
            else:
                discrete.setdefault((eg, count), []).append((k, float(p)))
                keys.append(("discrete", eg, count))
        primaries.append((energy, pAt if pAt else None))
        rest = table[nd:]
        if len(rest) == 1:
            report.lost(f"MF6/MT{mt} product {index} node {k}: a continuum of one "
                        f"point after the lines; kept as one spectrum",
                        unaffectedScopes=("cross-sections",))
            return None
        if len(rest):
            continuum.append((k, rest))
        nodeKeys.append(keys)

    primaryValues, shift = [], True
    if any(p for _, p in primaries):
        found = _primaryValues(primaries, ratio)
        if found is None:
            report.lost(f"MF6/MT{mt} product {index}: its primary photons do not keep "
                        f"one binding energy across the incident grid; kept as one "
                        f"spectrum", unaffectedScopes=("cross-sections",))
            return None
        primaryValues, shift = found

    eAxes, mAxes = energyAxes(), multiplicityAxes()
    pieces, meta = [], []

    def label():
        name = _uniqueLabel("photon", taken)
        taken.add(name)
        return name

    def multiplicity(xs, ys):
        return Multiplicity(form=XYs1d(xs=np.asarray(xs, dtype=float),
                                       ys=np.asarray(ys, dtype=float), axes=mAxes))

    def isotropic():
        return Isotropic2d(productFrame=frame)

    if continuum:
        functions, scale = [], []
        for j, (k, rest) in enumerate(continuum):
            x, f = rest[:, 0], rest[:, 1]
            integral = float(integrateFunction1d(XYs1d(xs=x, ys=f, interpolation=inner)))
            ys = f / integral if integral > 0 else f
            functions.append(XYs1d(xs=x, ys=ys, interpolation=inner, axes=eAxes,
                                   outerDomainValue=energies[k], index=j))
            scale.append(yAt[k] * integral)
        energy = fromEndfTab2(functions, [(len(functions), outer)], axes=eAxes)
        name = label()
        pieces.append((name, multiplicity([energies[k] for k, _ in continuum], scale),
                       Uncorrelated(angular=isotropic(), energy=energy, productFrame=frame)))
        meta.append({"label": name, "kind": "continuum"})

    keyLabel = {}
    for key in sorted(discrete):
        points = discrete[key]
        first, last = points[0][0], points[-1][0]
        listed = dict(points)
        xs, ys = [], []
        for k in range(first, last + 1):
            xs.append(energies[k])
            ys.append(listed[k] * yAt[k] if k in listed else 0.0)
        name = label()
        line = DiscreteGamma(value=key[0], domainMin=energies[first],
                             domainMax=energies[last], axes=energyAxes())
        pieces.append((name, multiplicity(xs, ys),
                       Uncorrelated(angular=isotropic(), energy=line, productFrame=frame)))
        meta.append({"label": name, "kind": "discrete", "eg": key[0], "n": key[1]})
        keyLabel[("discrete",) + key] = name

    for j, value in enumerate(primaryValues):
        nodes = [k for k, (_, p) in enumerate(primaries) if p]
        xs = [energies[k] for k in nodes]
        ys = [_probabilityAt(body, k, j) * yAt[k] for k in nodes]
        name = label()
        line = PrimaryGamma(value=float(value), domainMin=xs[0], domainMax=xs[-1],
                            axes=energyAxes())
        pieces.append((name, multiplicity(xs, ys),
                       Uncorrelated(angular=isotropic(), energy=line, productFrame=frame)))
        meta.append({"label": name, "kind": "primary", "position": j})
        keyLabel[("primary", j)] = name

    fields = {
        "pieces": meta,
        "continuum": meta[0]["label"] if continuum else None,
        "nodes": [[keyLabel[key] for key in keys] for keys in nodeKeys],
        "energies": energies,
        "tab2": [[int(a), int(b)] for a, b in body.tab2_interp],
        "values": [[float(v) for v in row] for row in body.values],
        "shift": bool(shift),
        "ratio": ratio,
        "digest": _digest(meta, {n: m.form for n, m, _ in pieces},
                          {n: d for n, _, d in pieces}),
    }
    report.approximated(
        f"MF6/MT{mt} product {index}: LAW=1 with {sum(1 for m in meta if m['kind'] != 'continuum')} "
        f"discrete line(s) is {len(pieces)} photon products, FUDGE's split: "
        f"{'a normalised continuum, ' if continuum else ''}each line a discreteGamma "
        f"or primaryGamma with P·y as its multiplicity. The ENDF section comes back "
        f"as read while the products are untouched",
        unaffectedScopes=("cross-sections",))
    return pieces, fields


def _probabilityAt(body, k: int, position: int) -> float:
    """The probability of the ``position``-th primary of node *k*."""
    table = body.block(k)[: int(body.nd[k])]
    return float(table[table[:, 0] < 0][position, 1])


# ---------------------------------------------------------------------------
# What the decoder made, as a hash
# ---------------------------------------------------------------------------

def _tableBytes(digest, function) -> None:
    xs = getattr(function, "xs", None)
    ys = getattr(function, "ys", None)
    if xs is None or ys is None:
        digest.update(repr(function).encode())
        return
    digest.update(np.asarray(xs, dtype=float).tobytes())
    digest.update(np.asarray(ys, dtype=float).tobytes())


def _digest(meta: List[dict], multiplicities: Dict[str, object],
            forms: Dict[str, object]) -> str:
    """SHA-256 of the pieces' multiplicities and distributions, in record order."""
    from kika.nuclear_data.model import DiscreteGamma, PrimaryGamma

    digest = hashlib.sha256()
    for piece in meta:
        name = piece["label"]
        digest.update(name.encode())
        _tableBytes(digest, multiplicities.get(name))
        form = forms.get(name)
        energy = getattr(form, "energy", None)
        digest.update(type(form).__name__.encode())
        digest.update(type(energy).__name__.encode())
        digest.update(type(getattr(form, "angular", None)).__name__.encode())
        if isinstance(energy, (DiscreteGamma, PrimaryGamma)):
            digest.update(np.asarray([energy.value, energy.domainMin, energy.domainMax],
                                     dtype=float).tobytes())
        elif energy is not None and hasattr(energy, "function1ds"):
            for function in energy.function1ds:
                digest.update(np.asarray([float(function.outerDomainValue)]).tobytes())
                _tableBytes(digest, function)
        else:
            digest.update(repr(energy).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def restoreLineSpectrum(record: dict, forms: Dict[str, object],
                        multiplicities: Optional[Dict[str, object]], mt: int, report=None):
    """A split record and its pieces → the LAW=1 body it was read from.

    The node values as read while the pieces hash to what the decoder made;
    otherwise each node rebuilt from the model -- ``P = m/y`` per line, the
    continuum scaled back by ``m/y`` -- with the values as read standing in only
    where ``y = 0`` makes the model silent.
    """
    from kika.endf.classes.mf6.laws import MF6LawContinuum

    fields = record["law_fields"]
    meta = fields[LINES_KEY]
    index = record["index"]
    labels = [piece["label"] for piece in meta["pieces"]]
    multiplicities = multiplicities or {}
    missing = [n for n in labels if n not in forms or n not in multiplicities]
    nd = [int(v) for v in fields["nd"]]
    na = [int(v) for v in fields["na"]]
    energies = [float(e) for e in meta["energies"]]
    pairs = [(int(a), int(b)) for a, b in meta["tab2"]]
    raw = [list(row) for row in meta["values"]]

    if not missing and _digest(meta["pieces"], multiplicities, forms) == meta["digest"]:
        return MF6LawContinuum(law=1, lang=int(fields["lang"]), lep=int(fields["lep"]),
                               tab2_interp=pairs, incident_energies=energies,
                               nd=nd, na=na, values=raw)
    if missing:
        raise ValueError(
            f"MF6/MT{mt} product {index} was split into the photon products "
            f"{labels} (D40) and the model does not hand back "
            f"{missing}: each line's multiplicity is its share of the yield, so "
            f"the section cannot be written without every piece")

    if report is not None:
        report.approximated(
            f"MF6/MT{mt} product {index}: the photon lines changed in the model, so "
            f"each node is rebuilt from it (P = m/y per line, the continuum times m/y)",
            unaffectedScopes=("cross-sections",))
    y = _yieldOf(SimpleNamespace(
        y_interp=record["y_interp"], y_energies=record["y_energies"],
        y_values=record["y_values"]))
    byLabel = {piece["label"]: piece for piece in meta["pieces"]}
    cont = meta.get("continuum")
    contNodes = list(forms[cont].energy.function1ds) if cont else []
    rawCont = [k for k in range(len(energies)) if len(raw[k]) // 2 > nd[k]]
    if len(contNodes) != len(rawCont):
        raise ValueError(
            f"MF6/MT{mt} product {index}: the continuum {cont!r} has "
            f"{len(contNodes)} incident energies and the file {len(rawCont)}; a node "
            f"added or removed in the model cannot be written back")
    yAt = _yieldAtNodes(y, energies)
    points = _pointsByNode(meta)
    values = []
    for k, energy in enumerate(energies):
        yk = yAt[k]
        table = np.asarray(raw[k], dtype=float).reshape(-1, 2)
        row = []
        for j, name in enumerate(meta["nodes"][k]):
            piece = byLabel[name]
            line = forms[name].energy
            eg = table[j, 0]
            if piece["kind"] == "discrete" and float(line.value) != piece["eg"]:
                eg = float(line.value)
            elif piece["kind"] == "primary":
                if float(line.value) != _decodedPrimary(meta, piece):
                    shifted = meta["ratio"] * energy if (
                        meta["shift"] and energy > _PRIMARY_SHIFT_FLOOR) else 0.0
                    eg = -(float(line.value) + shifted)
            m = _valueAt(multiplicities[name], energy, k, points[name])
            p = m / yk if yk else table[j, 1]
            row.extend([eg, p])
        if k in rawCont:
            node = contNodes[rawCont.index(k)]
            m = _valueAt(multiplicities[cont], energy, k, rawCont)
            scale = m / yk if yk else None
            if scale == 0.0 and _rawIntegral(table[nd[k]:], fields["lep"]) == 0.0:
                # A continuum the file states with zero weight (a histogram
                # whose only nonzero value is its last point: 1e-19 in JEFF-4.0
                # Pr-143 MT91): no photons either way, and its own values back.
                scale = None
            if scale is None:
                row.extend(float(v) for v in table[nd[k]:].reshape(-1))
            else:
                for x, f in zip(node.xs, node.ys):
                    row.extend([float(x), float(f) * scale])
        values.append(row)
    nd = [len(names) for names in meta["nodes"]]
    return MF6LawContinuum(law=1, lang=int(fields["lang"]), lep=int(fields["lep"]),
                           tab2_interp=pairs, incident_energies=energies,
                           nd=nd, na=na, values=values)


def _rawIntegral(table, lep) -> float:
    from kika.nuclear_data.model import ENDF_INT_TO_INTERPOLATION, XYs1d
    from kika.nuclear_data.model.functions.integration import integrateFunction1d

    if len(table) < 2:
        return 0.0
    return float(integrateFunction1d(XYs1d(
        xs=table[:, 0], ys=table[:, 1],
        interpolation=ENDF_INT_TO_INTERPOLATION.get(int(lep), "lin-lin"))))


def _decodedPrimary(meta, piece) -> float:
    """The binding energy the decoder gave this primary (from the first node that lists it)."""
    for k, names in enumerate(meta["nodes"]):
        if piece["label"] in names:
            first = np.asarray(meta["values"][k], dtype=float).reshape(-1, 2)
            position = names.index(piece["label"])
            return _binding(first[position, 0], meta["energies"][k], meta["ratio"],
                            meta["shift"])
    return float("nan")


# ---------------------------------------------------------------------------
# For the writers that put the lines back in the table (HDF5, G4NDL)
# ---------------------------------------------------------------------------

def mergedLineSpectra(reaction):
    """``(merged, hidden)`` for *reaction*'s split MF6 photons.

    ``merged`` maps the first piece's label to a product with the section's own
    yield and one ``uncorrelated`` spectrum that opens each node with its lines
    (the shape the decoder gave before D40, which is ACER's: lines first in the
    table, counted by ND in the record); ``hidden`` is the other pieces' labels.
    Both empty for a reaction with none.
    """
    from kika.nuclear_data.model import (EVAL_LABEL, ConversionReport, Distribution,
                                         Product)

    from .energy_angle import _continuumLegendre, _restoreProduct, frameForProduct

    provenance = getattr(reaction, "provenance", None)
    fields = getattr(provenance, "headerFields", None) or {}
    block = fields.get("mf6") if isinstance(fields, dict) else None
    records = [r for r in ((block or {}).get("products") or [])
               if isinstance(r, dict) and (r.get("law_fields") or {}).get(LINES_KEY)]
    if not records:
        return {}, set()
    channel = getattr(reaction, "outputChannel", None)
    products = list(getattr(channel, "products", None) or ())
    products += [d for p in products
                 for d in (getattr(getattr(p, "outputChannel", None), "products", None) or ())]
    # The reaction's own products first: a residual's decay can hold a
    # placeholder photon of the same label (residuals.py), and it is not MF6's.
    byLabel = {}
    for p in products:
        byLabel.setdefault(p.label or p.pid, p)
    merged, hidden = {}, set()
    for record in records:
        labels = lineSpectrumLabels(record)
        if any(n not in byLabel for n in labels):
            continue
        forms = {n: _evaluated(byLabel[n]) for n in labels}
        multiplicities = {n: getattr(byLabel[n].multiplicity, "form", None) for n in labels}
        restored = _restoreProduct(record, forms, int(reaction.ENDF_MT or 0),
                                   multiplicities=multiplicities)
        frame = frameForProduct(int(block.get("lct", 2)), 0)
        quiet = ConversionReport()
        distribution, _, _ = _continuumLegendre(restored.law_data, record["index"],
                                                int(reaction.ENDF_MT or 0), frame, quiet)
        from .energy_angle import _multiplicityOf
        from kika.nuclear_data.model import multiplicityAxes

        product = Product(pid="photon", label=labels[0],
                          multiplicity=_multiplicityOf(restored, multiplicityAxes()))
        product.distribution = Distribution()
        product.distribution[EVAL_LABEL] = distribution
        merged[labels[0]] = product
        hidden.update(labels[1:])
    return merged, hidden


def _evaluated(product):
    from kika.nuclear_data.model import EVAL_LABEL

    distribution = getattr(product, "distribution", None)
    if distribution is None or EVAL_LABEL not in distribution:
        return None
    return distribution[EVAL_LABEL]


# ---------------------------------------------------------------------------
# From a model with no ENDF provenance (G4c): FUDGE's toENDF6 regrouping
# ---------------------------------------------------------------------------

def isLinePiece(product) -> bool:
    """A photon an MF6 line spectrum is made of: a line, a primary or an isotropic continuum."""
    from kika.nuclear_data.model import (DiscreteGamma, Isotropic2d, PrimaryGamma,
                                         Uncorrelated)

    form = _evaluated(product)
    if product.pid != "photon" or not isinstance(form, Uncorrelated):
        return False
    if isinstance(form.energy, (DiscreteGamma, PrimaryGamma)):
        return True
    return isinstance(form.angular, Isotropic2d) and hasattr(form.energy, "function1ds")


def deriveLineSpectrum(pieces, total, awr: Optional[float], index: int, mt: int, report):
    """Photon products → one LAW=1 record whose nodes open with their lines.

    FUDGE's ``gammasToENDF6_MF6``: the yield is the reaction's total photon
    multiplicity (``total``, or the pieces' sum), each line's probability is its
    multiplicity over that total at the incident energies its own table lists,
    a primary's energy is written ``-(E_B + A/(A+1)·E)``, lines go in decreasing
    energy ahead of the continuum, and the continuum is scaled by its
    multiplicity over the total. ``None`` when the pieces cannot be one record.
    """
    from kika.nuclear_data.model import DiscreteGamma, PrimaryGamma, Regions1d

    if isinstance(total, Regions1d) and len(total.function1ds) > 1:
        report.lost(f"MF6/MT{mt}: the total photon multiplicity has "
                    f"{len(total.function1ds)} regions; one TAB2 region is all this writes")
        return None
    ratio = float(awr) / (1.0 + float(awr)) if awr else 0.0
    forms = {p.label: _evaluated(p) for p in pieces}
    multiplicities = {p.label: p.multiplicity.form for p in pieces}
    continuum = [p for p in pieces if not isinstance(forms[p.label].energy,
                                                     (DiscreteGamma, PrimaryGamma))]
    if len(continuum) > 1:
        report.lost(f"MF6/MT{mt}: {len(continuum)} photon continua; one LAW=1 record "
                    f"holds one")
        return None
    cont = continuum[0] if continuum else None
    lines = [p for p in pieces if p is not cont]

    nodes: Dict[float, List] = {}
    for p in lines:
        for e in np.asarray(multiplicities[p.label].xs, dtype=float):
            nodes.setdefault(float(e), [])
    contNodes = {}
    if cont is not None:
        for node in forms[cont.label].energy.function1ds:
            contNodes[float(node.outerDomainValue)] = node
            nodes.setdefault(float(node.outerDomainValue), [])
    energies = sorted(nodes)
    tAt = {e: float(total.evaluate(e)) for e in energies}

    meta = ([{"label": cont.label, "kind": "continuum"}] if cont is not None else [])
    for p in lines:
        energy = forms[p.label].energy
        kind = "primary" if isinstance(energy, PrimaryGamma) else "discrete"
        meta.append({"label": p.label, "kind": kind, "eg": float(energy.value), "n": 1}
                    if kind == "discrete" else {"label": p.label, "kind": kind, "position": 0})

    values, nd, nodeLabels = [], [], []
    for e in energies:
        entries = []
        if cont is None and e == energies[0] and not tAt[e]:
            # FUDGE's fix for the lowest node of a spectrum with no continuum
            # whose total is zero: the lines share the node equally.
            here = [p for p in lines
                    if np.any(np.asarray(multiplicities[p.label].xs, dtype=float) == e)]
            share = 1.0 / len(here) if here else 0.0
        else:
            share = None
        for p in lines:
            xs = np.asarray(multiplicities[p.label].xs, dtype=float)
            if not np.any(xs == e):
                continue
            m = float(multiplicities[p.label].evaluate(e))
            probability = m / tAt[e] if tAt[e] else (share if share is not None else m)
            energy = forms[p.label].energy
            real = (-(float(energy.value) + ratio * e) if isinstance(energy, PrimaryGamma)
                    else float(energy.value))
            entries.append((real, p.label, probability))
        counts: Dict[float, int] = {}
        keyed = []
        for real, label, probability in entries:
            counts[real] = counts.get(real, 0) + 1
            keyed.append((real, -counts[real], label, probability))
        keyed.sort(reverse=True)
        row = []
        for real, _, _, probability in keyed:
            row.extend([real, probability])
        nodeLabels.append([label for _, _, label, _ in keyed])
        nd.append(len(keyed))
        if e in contNodes:
            node = contNodes[e]
            scale = float(multiplicities[cont.label].evaluate(e)) / tAt[e] if tAt[e] else 1.0
            for x, f in zip(node.xs, node.ys):
                row.extend([float(x), float(f) * scale])
        values.append(row)

    outer = 2
    lep = 2
    if cont is not None:
        energy = forms[cont.label].energy
        outer = int(getattr(energy, "endfInterpolationCode", 2) or 2)
        lep = int(getattr(energy.function1ds[0], "endfInterpolationCode", 2) or 2)
    fields = {
        "pieces": meta,
        "continuum": cont.label if cont is not None else None,
        "nodes": nodeLabels,
        "energies": energies,
        "tab2": [[len(energies), outer]],
        "values": values,
        "shift": True,
        "ratio": ratio,
        "digest": _digest(meta, multiplicities, forms),
    }
    return {"lang": 1, "lep": lep, "nd": nd, "na": [0] * len(energies), LINES_KEY: fields}

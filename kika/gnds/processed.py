"""A processed suite in GNDS: native nodes for the physics, ``KIKA::processed`` for the rest.

ACE and OpenMC's HDF5 sample forms GNDS has no node for:

- ACE law 3 / OpenMC ``level`` (:class:`LevelEnergy`) is two-body kinematics,
  so it is written as an ``angularTwoBody`` with the Monte Carlo angle -- what
  FUDGE's ``processProtare`` writes for a level -- and ACER's two constants
  (threshold, mass ratio), which the kinematics gives only to the last bits,
  go to the block.
- ACE law 2 / OpenMC ``discrete_photon`` (:class:`DiscretePhotonEnergy`) is a
  ``discreteGamma`` (LP 0, 1) or ``primaryGamma`` (LP 2) over the incident
  range of its multiplicity; LP, which GNDS cannot spell, and the AWR of a
  primary go to the block.
- A :class:`DistributionMixture` of one law (OpenMC writes an
  ``applicability`` even then) is its law; the applicability goes to the block.
- An ``xs_pdf_cdf1d`` that opens with ND discrete lines (ACE law 4) is written
  whole; ND goes to the block, since FUDGE refuses any attribute it does not
  know on the node.
- A child repeating the one before it (a step in E' with the same cosine
  table on both sides) is written once, as FUDGE's reader needs, and its
  place is recorded.
- An isotropy stated in another frame than its ``uncorrelated`` (OpenMC's
  fission neutrons, CM inside lab): ``<isotropic2d/>`` has no attributes.

And what is not a distribution:

- photons on a sum (ACE and OpenMC hang them there, a ``crossSectionSum``
  has no products) go on an ``orphanProduct`` that references the sum;
- the URR probability tables, whole: FUDGE's ``URR_probabilityTables1d`` is a
  P(sigma|E) per reaction and loses the bands' joint structure;
- ACE's GPD as stated, a yield's TAB1 with its steps, the elastic recoil the
  writer adds (H-1's would come back as a proton), and ACE's and OpenMC's
  bookkeeping (``AceProvenance``/``Hdf5Provenance``: header, MTR order, TY,
  MFTYPE, q_prompt...).

A reader that ignores the institution still has every distribution, as
FUDGE's processed GNDS states it. kika reads the block back and the suite is
the one that was written, so ACE -> GNDS -> ACE comes back word for word.

Forms are swapped in place for the write (:func:`lower`) and put back after
(:func:`restore`): a deep copy of a processed U-238 is the whole library.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import fields, is_dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

LABEL = "KIKA::processed"
VERSION = "1"

#: The containers a product can sit in, in the order the GNDS writer walks them.
_CONTAINERS = ("reactions", "orphanProducts", "sums", "productions", "fissionComponents",
               "incompleteReactions")


def _m():
    from kika.nuclear_data import model

    return model


# ---------------------------------------------------------------------------
# Keys: what names one product's distribution in the file and in the model
# ---------------------------------------------------------------------------

def _products(suite) -> Iterator[Tuple[Tuple[str, ...], object]]:
    """``(key, product)`` for every product of the suite, the same walk on both sides.

    A key is the container, the reaction's MT and its rank among that MT, then
    the product's pid and rank among that pid (and ``delayed`` with the
    family's rank for a delayed neutron). Labels are not used: a suite read
    from ACE has none the writer would keep.
    """
    for name in _CONTAINERS:
        container = getattr(suite, name, None)
        if container is None:
            continue
        seen: Dict[int, int] = {}
        for reaction in container:
            channel = getattr(reaction, "outputChannel", None)
            if channel is None:
                continue
            mt = int(getattr(reaction, "ENDF_MT", None) or 0)
            rank = seen.get(mt, 0)
            seen[mt] = rank + 1
            base = (name, str(mt), str(rank))
            yield from _channelProducts(channel, base)
            data = getattr(channel, "fissionFragmentData", None)
            for k, family in enumerate(getattr(data, "delayedNeutrons", None) or ()):
                if family.product is not None:
                    yield base + ("delayed", str(k), family.product.pid, "0"), family.product


def _channelProducts(channel, base):
    ranks: Dict[str, int] = {}
    for product in channel.products:
        rank = ranks.get(product.pid, 0)
        ranks[product.pid] = rank + 1
        key = base + (product.pid, str(rank))
        yield key, product
        inner = getattr(product, "outputChannel", None)
        if inner is not None:
            yield from _channelProducts(inner, key + ("decay",))


def _multiplicityOwners(suite) -> Iterator[Tuple[Tuple[str, ...], object]]:
    """``(key, owner)`` of every multiplicity: the products', then the
    multiplicity sums' (total nu-bar), keyed by ENDF MT and rank."""
    yield from _products(suite)
    seen: Dict[int, int] = {}
    for summed in getattr(getattr(suite, "sums", None), "multiplicitySums", None) or ():
        mt = int(getattr(summed, "ENDF_MT", None) or 0)
        rank = seen.get(mt, 0)
        seen[mt] = rank + 1
        yield ("multiplicitySums", str(mt), str(rank)), summed


def _tables(form) -> Iterator[object]:
    """Every ``Xs_pdf_cdf1d`` under ``form``, in field and list order."""
    m = _m()
    if isinstance(form, m.Xs_pdf_cdf1d):
        yield form
        return
    if isinstance(form, (list, tuple)):
        for item in form:
            yield from _tables(item)
        return
    if is_dataclass(form) and not isinstance(form, type):
        for f in fields(form):
            value = getattr(form, f.name)
            if is_dataclass(value) or isinstance(value, (list, tuple)):
                yield from _tables(value)


# ---------------------------------------------------------------------------
# Lowering: the processed forms -> GNDS nodes, before the write
# ---------------------------------------------------------------------------

class Lowered:
    """What :func:`lower` swapped (to put back) and the block it owes the file."""

    def __init__(self) -> None:
        self.swapped: List[Tuple[object, str, object]] = []
        self.entries: List[Tuple[Tuple[str, ...], str, dict]] = []
        self.unsupported: List[str] = []
        #: ``(key, AceProvenance)``: the suite's (key ``()``), every reaction's
        #: own, every product's.
        self.ace: List[Tuple[Tuple[str, ...], object]] = []
        #: ``(sum, its channel, the orphan that carries it)`` (:func:`_foldSums`).
        self.folded: List[Tuple[object, object, object]] = []
        #: ``(forms dict, its content before)`` and ``(table, [(reaction key, label)])``.
        self.urrSwapped: List[Tuple[dict, dict]] = []
        self.urr: List[Tuple[object, List[Tuple[str, str]]]] = []
        #: ``(product key, label, (x, y, NBT/INT pairs))`` of a multiplicity with steps.
        self.multiplicities: List[Tuple[Tuple[str, ...], str, tuple]] = []

    def __bool__(self) -> bool:
        return bool(self.entries or self.ace or self.folded or self.urr or self.multiplicities or self.recoils
                    or getattr(self, "photonTotal", None) is not None)

    recoils: List[str] = []

    def foldKeys(self, suite) -> List[Tuple[str, str]]:
        """``(orphan key, sum key)`` of each fold, as the reader will find them."""
        keys = {id(r): k for k, r in _reactions(suite)}
        return [(keys[id(orphan)], keys[id(summed)]) for summed, _, orphan in self.folded]


def _reactions(suite) -> Iterator[Tuple[str, object]]:
    """``(key, reaction)`` of every reaction of every container, ranked as in :func:`_products`."""
    for name in _CONTAINERS:
        seen: Dict[int, int] = {}
        for reaction in getattr(suite, name, None) or ():
            mt = int(getattr(reaction, "ENDF_MT", None) or 0)
            rank = seen.get(mt, 0)
            seen[mt] = rank + 1
            yield "/".join((name, str(mt), str(rank))), reaction


#: The scalar fields of each bookkeeping kind, as written.
_FIELDS = {"ace": ("zaid", "extension", "awr", "comment", "date", "matid", "formatVersion",
                   "temperatureMeV"),
           "hdf5": ("path", "node")}


def _kind(provenance) -> Optional[str]:
    m = _m()
    if isinstance(provenance, m.AceProvenance):
        return "ace"
    if isinstance(provenance, m.Hdf5Provenance):
        return "hdf5"
    return None


def _jsonable(value):
    """JSON for what bookkeeping holds: arrays, numpy scalars, and the two
    functions OpenMC states beside a fission energy release (q_prompt,
    q_recoverable as XYs1d; a polynomial)."""
    m = _m()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, m.XYs1d):
        return {"__XYs1d__": True, "xs": value.xs, "ys": value.ys,
                "interpolation": str(value.interpolation.value)}
    if isinstance(value, m.Polynomial1d):
        return {"__Polynomial1d__": True, "coefficients": list(value.coefficients),
                "domainMin": value.domainMin, "domainMax": value.domainMax}
    raise TypeError(f"{type(value).__name__} is not JSON")


def _fromJson(content: dict):
    m = _m()
    if content.get("__XYs1d__"):
        return m.XYs1d(xs=np.asarray(content["xs"], dtype=float),
                       ys=np.asarray(content["ys"], dtype=float),
                       interpolation=content["interpolation"])
    if content.get("__Polynomial1d__"):
        return m.Polynomial1d(coefficients=content["coefficients"],
                              domainMin_=content["domainMin"], domainMax_=content["domainMax"])
    return content


def _releases(reaction) -> list:
    """A fission reaction's energy releases (MF1/458, OpenMC's fission_energy_release)."""
    data = getattr(getattr(reaction, "outputChannel", None), "fissionFragmentData", None)
    return list(getattr(data, "fissionEnergyReleases", None) or ())


def _aceProvenances(suite) -> Iterator[Tuple[Tuple[str, ...], object]]:
    """``(key, provenance)`` -- ACE's or OpenMC's bookkeeping -- of the suite, of
    reactions with their own, of products."""
    table = getattr(suite, "provenance", None)
    if _kind(table):
        yield (), table
    for name in _CONTAINERS:
        seen: Dict[int, int] = {}
        for reaction in getattr(suite, name, None) or ():
            mt = int(getattr(reaction, "ENDF_MT", None) or 0)
            rank = seen.get(mt, 0)
            seen[mt] = rank + 1
            own = getattr(reaction, "provenance", None)
            if _kind(own) and own is not table:
                yield (name, str(mt), str(rank)), own
            for k, release in enumerate(_releases(reaction)):
                if _kind(getattr(release, "provenance", None)):
                    yield (name, str(mt), str(rank), "fissionEnergyRelease", str(k)), release.provenance
    for key, product in _products(suite):
        own = getattr(product, "provenance", None)
        if _kind(own):
            yield key, own


def _domain(product, label) -> Tuple[float, float]:
    """The incident range a line is emitted over: its multiplicity's."""
    multiplicity = getattr(product, "multiplicity", None)
    forms = dict(getattr(multiplicity, "forms", None) or {})
    if not forms and getattr(multiplicity, "form", None) is not None:
        forms = {label: multiplicity.form}
    for form in [forms.get(label)] + list(forms.values()):
        lo, hi = getattr(form, "domainMin", None), getattr(form, "domainMax", None)
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            return float(lo), float(hi)
    return 1.0e-5, 2.0e7


def _lowerForm(form, product, label, info: dict):
    """One form as GNDS can state it, filling ``info`` with what it cannot."""
    m = _m()
    if (isinstance(form, m.Uncorrelated) and isinstance(form.angular, m.Isotropic2d)
            and form.angular.productFrame != form.productFrame):
        # <isotropic2d/> has no attributes, so an isotropy stated in another
        # frame than its <uncorrelated>'s -- OpenMC's fission neutrons, CM
        # inside lab -- would come back in the outer frame.
        info["isotropicFrame"] = str(form.angular.productFrame)
    if isinstance(form, m.DistributionMixture):
        if len(form.weighted) != 1:
            return None
        term = form.weighted[0]
        info["mixture"] = term.weight
        return _lowerForm(term.functional, product, label, info)
    if isinstance(form, m.Uncorrelated) and isinstance(form.energy, m.LevelEnergy):
        info["level"] = (float(form.energy.threshold), float(form.energy.massRatio))
        info["frame"] = str(form.productFrame)
        return m.AngularTwoBody(label=label, productFrame=form.productFrame, angular=form.angular)
    if isinstance(form, m.Uncorrelated) and isinstance(form.energy, m.DiscretePhotonEnergy):
        energy = form.energy
        lo, hi = _domain(product, label)
        cls = m.PrimaryGamma if int(energy.primaryFlag) == 2 else m.DiscreteGamma
        info["photon"] = (int(energy.primaryFlag), float(energy.atomicWeightRatio))
        return m.Uncorrelated(angular=form.angular, label=form.label, productFrame=form.productFrame,
                              energy=cls(value=float(energy.energy), domainMin=lo, domainMax=hi,
                                         axes=m.energyAxes()))
    return form


def _foldSums(suite, out: "Lowered") -> None:
    """A sum's products onto an ``orphanProduct`` that references it, for the write.

    ACE and OpenMC's HDF5 hang photons on a sum (O-16 MT103-107, 108 photons);
    a GNDS ``crossSectionSum`` has no output channel, and §14's place for them
    is an orphan product whose cross section is a reference to the sum -- what
    the ENDF adapter builds for MF13 on a sum. The sum keeps its Q.
    """
    from kika.nuclear_data.model.cross_section_forms import Reference

    m = _m()
    for summed in list(getattr(suite, "sums", None) or ()):
        channel = getattr(summed, "outputChannel", None)
        if channel is None or not len(channel.products):
            continue
        orphan = m.Reaction(id=m.ReactionId(label=f"{summed.label} products",
                                            ENDF_MT=summed.ENDF_MT),
                            outputChannel=channel)
        for label, form in summed.crossSection.forms.items():
            tag = {"XYs1d": "XYs1d", "Regions1d": "regions1d"}.get(type(form).__name__, "XYs1d")
            orphan.crossSection[label] = Reference(
                href=(f"/reactionSuite/sums/crossSectionSums/crossSectionSum"
                      f"[@label='{summed.label}']/crossSection/{tag}[@label='{label}']"),
                label=label)
        summed.outputChannel = m.OutputChannel(Q=channel.Q, genre=channel.genre)
        suite.orphanProducts.append(orphan)
        out.folded.append((summed, channel, orphan))


def _liftProbabilityTables(suite, out: "Lowered") -> None:
    """Take the URR probability tables out of the cross sections, one entry per table.

    ACE's and OpenMC's tables are **joint**: one band index for the total,
    elastic, fission, capture and heating at once, with INT/ILF/IOA/IFF.
    FUDGE's ``URR_probabilityTables1d`` is a P(sigma|E) per reaction, which
    loses which band of one reaction goes with which of another,
    so the joint table goes to the block, once, with the reactions that use it.
    """
    m = _m()
    byTable: Dict[int, Tuple[object, List[Tuple[str, str]]]] = {}
    for key, reaction in _reactions(suite):
        cs = getattr(reaction, "crossSection", None)
        forms = getattr(cs, "forms", None)
        if not forms:
            continue
        found = [(label, form) for label, form in forms.items()
                 if isinstance(form, m.URR_probabilityTables1d) and form.table is not None]
        if not found:
            continue
        out.urrSwapped.append((forms, dict(forms)))
        for label, form in found:
            byTable.setdefault(id(form.table), (form.table, []))[1].append((key, label))
            del forms[label]
    out.urr = list(byTable.values())


def _same(a, b) -> bool:
    """Two model nodes that say the same thing (arrays compared by value)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if is_dataclass(a) and not isinstance(a, type):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    return a == b


def _children(node) -> Optional[list]:
    m = _m()
    if isinstance(node, m.XYs2d):
        return node.function1ds
    if isinstance(node, m.XYs3d):
        return node.function2ds
    return None


def _visit(form, act, counter=None) -> None:
    """Depth first over the containers of ``form``; ``act(ordinal, children, recurse)``."""
    counter = counter if counter is not None else [0]
    children = _children(form)
    if children is not None:
        ordinal = counter[0]
        counter[0] += 1
        act(ordinal, children, lambda: [_visit(c, act, counter) for c in list(children)])
        return
    if isinstance(form, (list, tuple)):
        for item in form:
            _visit(item, act, counter)
        return
    if is_dataclass(form) and not isinstance(form, type):
        for f in fields(form):
            value = getattr(form, f.name)
            if is_dataclass(value) or isinstance(value, (list, tuple)):
                _visit(value, act, counter)


def _dropRepeats(form) -> List[Tuple[int, List[int]]]:
    """Remove a child that repeats the one before (same outer value, same content).

    ACE states a step in E' with the two cosine tables at the repeated point
    identical (Mn-55 MT5, Pu-242 MT91); the GNDS writer drops such a repeat,
    which FUDGE's reader refuses (encode.py _dropRepeatedOuter). Done here
    instead, recorded, and put back by the reader.
    """
    dropped = []

    def act(ordinal, children, recurse):
        repeats = [j for j in range(1, len(children))
                   if children[j].outerDomainValue == children[j - 1].outerDomainValue
                   and _same(children[j], children[j - 1])]
        for j in reversed(repeats):
            del children[j]
        if repeats:
            dropped.append((ordinal, repeats))
        recurse()

    _visit(form, act)
    return dropped


def _hasRepeats(form) -> bool:
    found = []

    def act(ordinal, children, recurse):
        if any(children[j].outerDomainValue == children[j - 1].outerDomainValue
               and _same(children[j], children[j - 1]) for j in range(1, len(children))):
            found.append(ordinal)
        else:
            recurse()

    _visit(form, act)
    return bool(found)


def _putRepeatsBack(form, dropped) -> None:
    import copy

    where = dict(dropped)

    def act(ordinal, children, recurse):
        recurse()
        for j in where.get(ordinal, ()):
            children.insert(j, copy.deepcopy(children[j - 1]))

    _visit(form, act)


def _oneTwoBodyNeutron(reaction) -> bool:
    """The condition under which the GNDS writer adds the elastic recoil."""
    m = _m()
    channel = getattr(reaction, "outputChannel", None)
    bodies = list(channel.products) if channel is not None else []
    if len(bodies) != 1 or bodies[0].pid != "n" or bodies[0].distribution is None:
        return False
    return any(isinstance(f, m.AngularTwoBody) for f in bodies[0].distribution.forms.values())


def lower(suite) -> Lowered:
    """Swap every processed form GNDS has no node for, in place; :func:`restore` undoes it."""
    m = _m()
    out = Lowered()
    # Only a processed suite (a heated or MonteCarlo_cdf style) gets the
    # rules below that an evaluation would also meet (a yield with a step, the
    # elastic recoil); an evaluation's GNDS is written as before.
    processed = any(isinstance(style, (m.Heated, m.MonteCarlo_cdf))
                    for style in getattr(suite, "styles", None) or ())
    _foldSums(suite, out)
    _liftProbabilityTables(suite, out)
    # A yield with a step, as ACE states it: one TAB1 with the energy
    # repeated. GNDS splits it into regions1d (FUDGE drops a repeat with
    # equal values), so the table comes back with more NBT/INT; the block
    # keeps it as it was (Pu-240's MT4 photons, U-238's total nu-bar).
    for key, owner in _multiplicityOwners(suite):
        for label, form in (getattr(getattr(owner, "multiplicity", None), "forms", None) or {}).items():
            if processed and isinstance(form, (m.XYs1d, m.Regions1d)):
                x, y, pairs = form.toEndfRegions()
                if len(pairs) > 1 or np.any(np.diff(np.asarray(x, dtype=float)) <= 0):
                    out.multiplicities.append((key, label, (x, y, pairs)))
    for key, product in _products(suite):
        distribution = getattr(product, "distribution", None)
        if distribution is None:
            continue
        for label in list(distribution.forms):
            form = distribution.forms[label]
            info: dict = {}
            lowered = _lowerForm(form, product, label, info)
            if lowered is None:
                out.unsupported.append(f"{'/'.join(key)} {label!r}: a mixture of "
                                       f"{len(form.weighted)} laws")
                continue
            if _hasRepeats(lowered):
                import copy

                lowered = copy.deepcopy(lowered)
                info["repeats"] = _dropRepeats(lowered)
            discrete = [int(t.numberOfDiscrete) for t in _tables(lowered)]
            if any(discrete):
                info["discrete"] = discrete
            if lowered is not form:
                distribution.forms[label] = lowered
                out.swapped.append((distribution, label, form))
            if info:
                out.entries.append((key, label, info))
    out.ace = list(_aceProvenances(suite))
    out.folds = out.foldKeys(suite)
    # The GNDS writer adds the target's recoil to an MT2 of one two-body
    # neutron, as FUDGE writes it (encode.py elasticRecoil). The suite had
    # none; H-1's would come back as a proton the ACE writer must produce.
    out.recoils = [key for key, reaction in _reactions(suite)
                   if processed and getattr(reaction, "ENDF_MT", None) == 2
                   and _oneTwoBodyNeutron(reaction)]
    out.photonTotal = getattr(suite, "photonProductionTotal", None)
    return out


def restore(lowered: Lowered, suite=None) -> None:
    for distribution, label, form in reversed(lowered.swapped):
        distribution.forms[label] = form
    for forms, before in lowered.urrSwapped:
        forms.clear()
        forms.update(before)
    for summed, channel, orphan in reversed(lowered.folded):
        summed.outputChannel = channel
        if suite is not None:
            suite.orphanProducts.remove(orphan)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

def _float(value: float) -> str:
    return repr(float(value))


def _floats(values) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def write(root: ET.Element, lowered: Lowered, report, function) -> None:
    """``<applicationData><institution label="KIKA::processed">``, one ``<form>`` per entry.

    ``function`` is the encoder's functional writer, for an applicability.
    """
    if lowered.unsupported:
        for line in lowered.unsupported:
            report.unsupportedNode(f"{line}: GNDS has no node for a mixture of several laws "
                                   f"of one product, and kika writes none")
    if not lowered:
        return
    import json

    application = root.find("applicationData")
    if application is None:
        application = ET.SubElement(root, "applicationData")
    institution = ET.SubElement(application, "institution", label=LABEL)
    if lowered.ace:
        # ACE's bookkeeping: the header, MTR/MTRP/NXS/YP, and per
        # product TY, MFTYPE, MTRP, the laws ACE chained -- what encodeAce
        # needs to give the table back word for word. JSON keeps every float
        # exact (repr).
        holder = ET.SubElement(institution, "aceProvenance", version=VERSION)
        for key, provenance in lowered.ace:
            kind = _kind(provenance)
            node = ET.SubElement(holder, "provenance", key="/".join(key), kind=kind)
            content = {name: getattr(provenance, name) for name in _FIELDS[kind]}
            content["attributes"] = provenance.attributes
            node.text = json.dumps(content, default=_jsonable, separators=(",", ":"))
    total = getattr(lowered, "photonTotal", None)
    if total is not None:
        # ACE's GPD as the table stated it; rebuilt from the productions it
        # differs in the last digits (7e-7 on O-16).
        holder = ET.SubElement(institution, "photonProductionTotal", version=VERSION)
        function(holder, total, report, f"{LABEL} photonProductionTotal")
    if lowered.urr:
        holder = ET.SubElement(institution, "probabilityTables", version=VERSION)
        for table, uses in lowered.urr:
            values = np.asarray(table.table, dtype=float)
            node = ET.SubElement(holder, "table", interpolation=str(int(table.interpolation)),
                                 inelastic=str(int(table.inelastic)),
                                 absorption=str(int(table.absorption)),
                                 multiplySmooth=str(bool(table.multiplySmooth)).lower(),
                                 shape=" ".join(str(n) for n in values.shape))
            for key, label in uses:
                ET.SubElement(node, "use", key=key, label=label)
            ET.SubElement(node, "energies").text = _floats(table.energies)
            ET.SubElement(node, "values").text = _floats(values.ravel())
        report.warn(f"{len(lowered.urr)} URR probability table(s) written whole in "
                    f"applicationData {LABEL} (joint bands, factors, INT/ILF/IOA/IFF); a reader "
                    f"other than kika does not see them")
    if lowered.multiplicities:
        holder = ET.SubElement(institution, "multiplicityTables", version=VERSION)
        for key, label, (x, y, pairs) in lowered.multiplicities:
            node = ET.SubElement(holder, "table", key="/".join(key), label=label,
                                 regions=" ".join(f"{int(n)}:{int(i)}" for n, i in pairs))
            ET.SubElement(node, "x").text = _floats(x)
            ET.SubElement(node, "y").text = _floats(y)
    if lowered.recoils:
        holder = ET.SubElement(institution, "writerRecoils", version=VERSION)
        for key in lowered.recoils:
            ET.SubElement(holder, "recoil", key=key)
    if lowered.folds:
        holder = ET.SubElement(institution, "sumProducts", version=VERSION)
        for orphanKey, sumKey in lowered.folds:
            ET.SubElement(holder, "fold", orphan=orphanKey, sum=sumKey)
    block = ET.SubElement(institution, "processedForms", version=VERSION)
    for key, label, info in lowered.entries:
        node = ET.SubElement(block, "form", key="/".join(key), label=label)
        if "level" in info:
            threshold, ratio = info["level"]
            node.attrib.update(threshold=_float(threshold), massRatio=_float(ratio),
                               productFrame=info["frame"])
        if "photon" in info:
            flag, awr = info["photon"]
            node.attrib.update(primaryFlag=str(flag), atomicWeightRatio=_float(awr))
        if "isotropicFrame" in info:
            node.attrib["isotropicFrame"] = info["isotropicFrame"]
        if "discrete" in info:
            node.attrib["numberOfDiscrete"] = " ".join(str(n) for n in info["discrete"])
        if info.get("repeats"):
            # "ordinal:j,j ordinal:j": the containers (depth first, as written)
            # whose child j repeated child j - 1 and was not written.
            node.attrib["repeats"] = " ".join(f"{o}:{','.join(map(str, js))}"
                                              for o, js in info["repeats"])
        if "mixture" in info:
            holder = ET.SubElement(node, "applicability")
            if info["mixture"] is not None:
                function(holder, info["mixture"], report, f"{LABEL} {node.attrib['key']}")
    report.warn(f"{len(lowered.entries)} processed form(s) carry data GNDS has no spelling for "
                f"(ACE's LP, law 3's constants, a table's discrete lines, an applicability); "
                f"they are in applicationData {LABEL}, which a reader other than kika ignores "
                f"and still has every distribution")


# ---------------------------------------------------------------------------
# Raising: the block -> the processed forms, after the read
# ---------------------------------------------------------------------------

def read(application: ET.Element, suite, report, readFunction) -> ET.Element:
    """Apply ``KIKA::processed`` to ``suite``; return ``application`` without it."""
    m = _m()
    rest = ET.Element(application.tag, application.attrib)
    blocks = {"processedForms": [], "aceProvenance": [], "sumProducts": [],
              "photonProductionTotal": [], "probabilityTables": [], "writerRecoils": [],
              "multiplicityTables": []}
    for institution in application:
        if institution.get("label") != LABEL:
            rest.append(institution)
            continue
        for block in institution:
            if block.tag not in blocks or block.get("version") != VERSION:
                report.lost(f"applicationData {LABEL}/{block.tag} version "
                            f"{block.get('version')!r} is not one kika reads")
                continue
            blocks[block.tag].extend(block)
    entries = blocks["processedForms"]
    if blocks["aceProvenance"]:
        _readAce(blocks["aceProvenance"], suite, report)
    _applyForms(entries, suite, report, readFunction)
    if blocks["multiplicityTables"]:
        products = {"/".join(key): owner for key, owner in _multiplicityOwners(suite)}
        for node in blocks["multiplicityTables"]:
            product = products.get(node.get("key"))
            forms = getattr(getattr(product, "multiplicity", None), "forms", None)
            label = node.get("label")
            if forms is None or label not in forms:
                report.lost(f"{LABEL}: a multiplicity table for {node.get('key')} {label!r}, "
                            f"which the file does not have")
                continue
            pairs = [tuple(int(v) for v in item.split(":")) for item in node.get("regions").split()]
            x = np.array(node.find("x").text.split(), dtype=float)
            y = np.array(node.find("y").text.split(), dtype=float)
            old = forms[label]
            if len(pairs) == 1:
                from kika.nuclear_data.model.enums import ENDF_INT_TO_INTERPOLATION

                new = m.XYs1d(xs=x, ys=y, interpolation=ENDF_INT_TO_INTERPOLATION[pairs[0][1]],
                              axes=old.axes, label=label)
            else:
                new = m.Regions1d.fromEndfRegions(x, y, pairs, axes=old.axes, label=label)
            forms[label] = new
    if blocks["probabilityTables"]:
        _readProbabilityTables(blocks["probabilityTables"], suite, report)
    if blocks["writerRecoils"]:
        reactions = dict(_reactions(suite))
        for node in blocks["writerRecoils"]:
            reaction = reactions.get(node.get("key"))
            if reaction is None:
                continue
            products = reaction.outputChannel.products.products
            products[:] = [p for p in products
                           if not (p.pid == suite.target and p.distribution is not None
                                   and any(getattr(f, "isRecoil", False)
                                           for f in p.distribution.forms.values()))]
    if blocks["sumProducts"]:
        _unfold(blocks["sumProducts"], suite, report)
    for node in blocks["photonProductionTotal"]:
        suite.photonProductionTotal = readFunction(node)
    return rest


def _unfold(nodes, suite, report) -> None:
    """Each orphan the writer made for a sum's products back onto the sum."""
    reactions = dict(_reactions(suite))
    for node in nodes:
        orphan, summed = reactions.get(node.get("orphan")), reactions.get(node.get("sum"))
        if orphan is None or summed is None:
            report.lost(f"{LABEL}: the products of {node.get('sum')} were written on "
                        f"{node.get('orphan')}, and one of the two is not in the file")
            continue
        q = getattr(summed.outputChannel, "Q", None)
        summed.outputChannel = orphan.outputChannel
        if q is not None:
            summed.outputChannel.Q = q
        suite.orphanProducts.remove(orphan)


def _applyForms(entries, suite, report, readFunction) -> None:
    m = _m()
    if not entries:
        return
    byKey = {"/".join(key): product for key, product in _products(suite)}
    missed = 0
    for node in entries:
        product = byKey.get(node.get("key"))
        label = node.get("label")
        distribution = getattr(product, "distribution", None)
        if distribution is None or label not in distribution.forms:
            missed += 1
            continue
        form = distribution.forms[label]
        if "threshold" in node.attrib and isinstance(form, m.AngularTwoBody):
            form = m.Uncorrelated(
                angular=form.angular, label=label,
                productFrame=m.Frame(node.get("productFrame", str(form.productFrame))),
                energy=m.LevelEnergy(threshold=float(node.get("threshold")),
                                     massRatio=float(node.get("massRatio"))))
        if "primaryFlag" in node.attrib and isinstance(form, m.Uncorrelated):
            form = m.Uncorrelated(
                angular=form.angular, label=form.label, productFrame=form.productFrame,
                energy=m.DiscretePhotonEnergy(primaryFlag=int(node.get("primaryFlag")),
                                              energy=float(form.energy.value),
                                              atomicWeightRatio=float(node.get("atomicWeightRatio"))))
        if "isotropicFrame" in node.attrib:
            inner = form.weighted[0].functional if isinstance(form, m.DistributionMixture) else form
            if isinstance(getattr(inner, "angular", None), m.Isotropic2d):
                inner.angular.productFrame = m.Frame(node.get("isotropicFrame"))
        if "numberOfDiscrete" in node.attrib:
            counts = [int(v) for v in node.get("numberOfDiscrete").split()]
            tables = list(_tables(form))
            if len(tables) != len(counts):
                report.lost(f"{LABEL} {node.get('key')} {label!r}: {len(counts)} discrete-line "
                            f"counts for {len(tables)} tables; not applied")
            else:
                for table, count in zip(tables, counts):
                    table.numberOfDiscrete = count
        if node.get("repeats"):
            dropped = []
            for item in node.get("repeats").split():
                ordinal, js = item.split(":")
                dropped.append((int(ordinal), [int(j) for j in js.split(",")]))
            _putRepeatsBack(form, dropped)
        holder = node.find("applicability")
        if holder is not None:
            child = next(iter(holder), None)
            weight = readFunction(child) if child is not None else None
            form = m.DistributionMixture(weighted=[m.Weighted(weight=weight, functional=form)],
                                         label=label)
        distribution.forms[label] = form
    if missed:
        report.lost(f"{missed} {LABEL} entr(ies) name a product or form the file does not "
                    f"have; not applied")


def _readAce(nodes, suite, report) -> None:
    """Put each ``AceProvenance`` back on its owner; reactions that shared the suite's, again."""
    import json

    m = _m()
    owners = {"": suite}
    for name in _CONTAINERS:
        seen: Dict[int, int] = {}
        for reaction in getattr(suite, name, None) or ():
            mt = int(getattr(reaction, "ENDF_MT", None) or 0)
            rank = seen.get(mt, 0)
            seen[mt] = rank + 1
            owners["/".join((name, str(mt), str(rank)))] = reaction
            for k, release in enumerate(_releases(reaction)):
                owners["/".join((name, str(mt), str(rank), "fissionEnergyRelease", str(k)))] = release
    owners.update({"/".join(key): product for key, product in _products(suite)})
    stated = {node.get("key"): (node.get("kind", "ace"),
                                json.loads(node.text or "{}", object_hook=_fromJson))
              for node in nodes}
    table = None
    missed = 0
    for key, (kind, content) in stated.items():
        owner = owners.get(key)
        if owner is None:
            missed += 1
            continue
        attributes = content.pop("attributes", {}) or {}
        cls = m.AceProvenance if kind == "ace" else m.Hdf5Provenance
        provenance = cls(attributes=attributes, **content)
        if key == "":
            table = provenance
        owner.provenance = provenance
    if table is not None:
        # The reactions without their own shared the table's, as decodeAce leaves them.
        for name in _CONTAINERS:
            seen = {}
            for reaction in getattr(suite, name, None) or ():
                mt = int(getattr(reaction, "ENDF_MT", None) or 0)
                rank = seen.get(mt, 0)
                seen[mt] = rank + 1
                if "/".join((name, str(mt), str(rank))) not in stated:
                    reaction.provenance = table
    if missed:
        report.lost(f"{missed} {LABEL} ACE provenance entr(ies) name a node the file does not "
                    f"have; not applied")


def _readProbabilityTables(nodes, suite, report) -> None:
    """Each joint table back, the same object on every reaction that used it."""
    m = _m()
    reactions = dict(_reactions(suite))
    for node in nodes:
        shape = tuple(int(n) for n in node.get("shape").split())
        values = np.array(node.find("values").text.split(), dtype=float).reshape(shape)
        table = m.ProbabilityTables(
            energies=np.array(node.find("energies").text.split(), dtype=float), table=values,
            interpolation=int(node.get("interpolation")), inelastic=int(node.get("inelastic")),
            absorption=int(node.get("absorption")),
            multiplySmooth=node.get("multiplySmooth") == "true")
        for use in node.findall("use"):
            reaction = reactions.get(use.get("key"))
            if reaction is None:
                report.lost(f"{LABEL}: a URR table is used by {use.get('key')}, which the file "
                            f"does not have")
                continue
            label = use.get("label")
            reaction.crossSection[label] = m.URR_probabilityTables1d(label=label, table=table)

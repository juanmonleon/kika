"""MF5 ↔ the ``energy`` half of :class:`~kika.nuclear_data.model.distributions.Uncorrelated`.

MF5 is a section header and NK subsections, each an independent law weighted by
``p_k(E)``. **Every law has a model node since 2026-10-08 (roadmap E2).** LF=1
is a TAB2 over incident energy whose nodes are TAB1s in E′, which is GNDS
§18.3's ``energy`` child spelled in ENDF; LF=5/7/9/11/12 are §18.3's
parametrised spectra (:mod:`kika.nuclear_data.model.energy_spectra`) -- carried
as their parameters, never tabulated into numbers the evaluator did not write.

**NK > 1 is ``weightedFunctionals``**, ``sum_k p_k(E) f_k``, and is modelled as
such -- a single partial out of the sum would be a *piece* of the distribution
hung on the product as though it were the whole. **Except MT455**, where the NK
subsections are one spectrum per delayed-neutron precursor family and not a
sum: :func:`decodeMF5Families` returns them one by one for §18.4, which is how
FUDGE reads it too.

**Written back from the bytes, and refused if the model was edited.** Only an
NK=1 LF=1 table is re-encoded from the model. A parametrised or weighted
section goes back as the records it came in as -- re-emitting their TAB1s from
the model would re-open the ``format_interp_pairs`` padding defect that
ENDF/B-VIII.1 pins -- and :func:`encodeMF5MT` first re-decodes those records
and compares: a parameter changed since decoding is refused by name rather
than silently replaced by the original.

That case is not hypothetical and it is not rare where it occurs. ENDF/B-VIII.1
has **zero** NK>1 sections containing an LF=1 (595 sections measured
2026-08-24: 487 are NK=1/LF=1 and the other 108 are homogeneous LF=5/7/9), so a
reader tested only against it would never meet one. **JEFF-4.0's U-235 states
MF5/MT455 as NK=8 with all eight LF=1** — the per-precursor delayed spectra —
and that is the library the thesis track reads.

**What is kept and where.** The tables — the incident grid, every outgoing grid,
every chi, and the interpolation of each — are the model's. NK, LF, U and
``p_k(E)`` are ENDF bookkeeping with no GNDS counterpart and go to
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`'s ``headerFields``
under an ``"mf5"`` key of their own, beside MF4's flat ``ltt``/``li``/``lct``:
one product carries one provenance, and the two files must not tread on each
other. The verbatim records of a law kika does not model go there too, **as
text**. Keeping the parsed ``MF5PartialRaw`` object instead would put an ENDF
class inside the format-neutral model, which is the accidental format surface
the model exists in order not to have; a list of strings is plain data, and it
is enough to write the section back byte for byte in its original order.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport,
    EndfProvenance,
    Evaporation,
    Function1d,
    GeneralEvaporation,
    MadlandNix,
    PhysicalQuantity,
    Regions1d,
    SimpleMaxwellianFission,
    Watt,
    Weighted,
    WeightedFunctionals,
    energyAxes,
    fromEndfTab2,
    toEndfTab2,
)
from kika.nuclear_data.model.axes import Axes

__all__ = ["decodeMF5MT", "encodeMF5MT", "decodeMF5Families"]

#: The MT whose NK subsections are per-family spectra, not a weighted sum.
DELAYED_MT = 455


def _za(section) -> Optional[int]:
    """ZA, **rounded**. See :func:`kika.nuclear_data.model.provenance._asEndfInt`."""
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return int(round(float(value))) if value is not None else None


def _verbatimBody(partial, mat, mf, mt) -> List[str]:
    """Columns 1-66 of every record of *partial* after its subsection header.

    Asked of the partial itself rather than read off it, because only
    ``MF5PartialRaw`` stores such a list — a ``MF5PartialTabulated`` holds
    parsed tables. Emitting and slicing gets the same bytes from either, which
    is what makes the provenance uniform over laws kika models and laws it does
    not. The line numbers here are throwaway: ``MF5PartialRaw.emit`` re-stamps
    MAT/MF/MT and the running sequence on the way out, exactly as
    ``MF5MT.__str__`` always did.
    """
    header, _ = partial.emit_header(mat, mf, mt, 1)
    whole, _ = partial.emit(mat, mf, mt, 1)
    return [line[:66] for line in whole[len(header):]]


def _partialRecord(partial, index: int, modelled: bool, mat, mf, mt) -> dict:
    """One subsection's fields, as primitives.

    The tables of the **modelled** partial are not here — they are the model's,
    and the encoder rebuilds the records from them. Every other partial keeps
    its records as the bytes the evaluator wrote, whatever its law: that is
    already what ``MF5PartialRaw`` does for the analytic spectra, and an LF=1
    inside an NK>1 needs exactly the same treatment. It is not a hypothetical —
    JEFF-4.0's U-235 states MF5/MT455 as **NK=8 and all eight LF=1**, the
    per-precursor delayed spectra, and none of them can be the section's
    ``energy`` on its own.
    """
    record = {
        "index": index,
        "lf": int(partial.lf),
        "u": float(partial.u),
        "p_interp": [(int(nbt), int(code)) for nbt, code in partial.p_interp],
        "p_energies": [float(v) for v in partial.p_energies],
        "p_values": [float(v) for v in partial.p_values],
    }
    if not modelled:
        record["raw_lines"] = _verbatimBody(partial, mat, mf, mt)
    return record


def _headerProvenance(mf5mt, modelled: Optional[int]) -> EndfProvenance:
    return EndfProvenance(
        mat=getattr(mf5mt, "_mat", None),
        awr=getattr(mf5mt, "_awr", None),
        za=_za(mf5mt),
        headerFields={
            "mf5": {
                # **MF5's own MAT, ZA and AWR, not the product's.** One
                # `EndfProvenance` per product carries MF4's at the top level,
                # and merging MF5 into it would let one file's header decide
                # the other's. That is not hypothetical: ENDF/B-VIII.1's
                # Ce-140 writes AWR=1.387036+2 in MF5/MT91 and 1.387030+2 in
                # MF4, and Am-243 disagrees with itself the same way in MT18 —
                # two sections out of 595, found by writing whole tapes back
                # and comparing (2026-08-24). Six significant figures apart is
                # still a different byte.
                "mat": getattr(mf5mt, "_mat", None),
                "za": _za(mf5mt),
                "awr": getattr(mf5mt, "_awr", None),
                "nk": mf5mt.num_partials,
                # Which subsection the model form came from, so the encoder puts
                # it back where it was rather than assuming subsection 0.
                "modelled": modelled,
                "partials": [
                    _partialRecord(partial, index, index == modelled,
                                   getattr(mf5mt, "_mat", 0) or 0, 5,
                                   mf5mt.number)
                    for index, partial in enumerate(mf5mt.partials)
                ],
            },
        },
    )


# ---------------------------------------------------------------------------
# The per-incident-energy 1-d functions
# ---------------------------------------------------------------------------

def _spectrumAt(energy: float, outgoing, chi, regions, index: int) -> Function1d:
    """One TAB1 node of an LF=1 → a function over E′.

    A :class:`Regions1d` when the record really has more than one interpolation
    region, and the single :class:`XYs1d` inside it when it does not — the same
    rule, and for the same schema reason, as
    :func:`kika.endf.model_adapter.angular._tabulatedAt`: ``gnds.xsd`` puts
    ``minOccurs="2"`` on the children of ``regions1d``, so a one-region
    ``regions1d`` is a node the schema rejects.

    Both classes answer ``toEndfRegions()``, so the encoder re-emits either
    without asking which one it got.
    """
    pairs = [(int(nbt), int(code)) for nbt, code in regions]
    x = np.asarray(outgoing, dtype=float)
    if not pairs and x.size:
        pairs = [(int(x.size), 2)]
    function = Regions1d.fromEndfRegions(x, np.asarray(chi, dtype=float), pairs)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
    function.outerDomainValue = float(energy)
    function.index = index
    return function


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _parameter(partial, prefix: str, label: str, unit: str, xLabel: str = "energy_in",
               xUnit: str = "eV") -> Function1d:
    """One of a law's parameter TAB1s → an ``XYs1d`` (a ``regions1d`` if NR > 1)."""
    x = np.asarray(getattr(partial, f"{prefix}_energies" if prefix != "g" else "g_x"), dtype=float)
    y = np.asarray(getattr(partial, f"{prefix}_values"), dtype=float)
    pairs = [(int(a), int(b)) for a, b in getattr(partial, f"{prefix}_interp")] or [(len(x), 2)]
    axes = Axes.forFunction1d(label, unit, xLabel, xUnit)
    table = Regions1d.fromEndfRegions(x, y, pairs, axes=axes)
    return table.function1ds[0] if len(table.function1ds) == 1 else table


def _functional(partial):
    """A decoded analytic partial → its §18.3 node, or ``None`` for a raw one.

    Units and axis labels are FUDGE's (``toEnergyFunctionalData``): theta, a and
    T_M in eV, b in 1/eV, and g against the dimensionless E'/theta.
    """
    from kika.endf.classes.mf5.analytic import (MF5Evaporation,
                                                MF5GeneralEvaporation,
                                                MF5MadlandNix, MF5Maxwellian,
                                                MF5Watt)

    u = PhysicalQuantity(float(partial.u), "eV")
    if isinstance(partial, MF5GeneralEvaporation):
        return GeneralEvaporation(
            U=u, theta=_parameter(partial, "theta", "theta", "eV"),
            g=_parameter(partial, "g", "g", "", "energy_out / theta(energy_in)", ""))
    if isinstance(partial, MF5Maxwellian):
        return SimpleMaxwellianFission(U=u, theta=_parameter(partial, "theta", "theta", "eV"))
    if isinstance(partial, MF5Evaporation):
        return Evaporation(U=u, theta=_parameter(partial, "theta", "theta", "eV"))
    if isinstance(partial, MF5Watt):
        return Watt(U=u, a=_parameter(partial, "a", "a", "eV"),
                    b=_parameter(partial, "b", "b", "1/eV"))
    if isinstance(partial, MF5MadlandNix):
        return MadlandNix(EFL=PhysicalQuantity(float(partial.efl), "eV"),
                          EFH=PhysicalQuantity(float(partial.efh), "eV"),
                          T_M=_parameter(partial, "tm", "T_M", "eV"))
    return None


def _tabulated(partial):
    """An LF=1 partial → its ``XYs2d``/``Regions2d``."""
    axes = energyAxes()
    functions = [
        _spectrumAt(energy, partial.outgoing_grids[k], partial.chi[k],
                    partial.outgoing_interp[k], k)
        for k, energy in enumerate(partial.incident_energies)
    ]
    return fromEndfTab2(functions, partial.tab2_interp, axes=axes)


def _weight(partial) -> Function1d:
    """``p_k(E)`` → a function with FUDGE's axes (``weight`` against ``energy_in``)."""
    x = np.asarray(partial.p_energies, dtype=float)
    y = np.asarray(partial.p_values, dtype=float)
    pairs = [(int(a), int(b)) for a, b in partial.p_interp] or [(len(x), 2)]
    table = Regions1d.fromEndfRegions(x, y, pairs,
                                      axes=Axes.forFunction1d("weight", "", "energy_in", "eV"))
    return table.function1ds[0] if len(table.function1ds) == 1 else table


def _member(partial):
    """A partial → its model form: a table for LF=1, a §18.3 node otherwise."""
    from kika.endf.classes.mf5.partials import MF5PartialTabulated

    if isinstance(partial, MF5PartialTabulated):
        return _tabulated(partial)
    return _functional(partial)


def decodeMF5Families(mf5mt, report: Optional[ConversionReport] = None):
    """MF5/MT455 → ``([(p_k, form_k), ...], provenance, report)``, one per family.

    The NK subsections of MT455 are the precursor families' spectra, in the
    order MF1/455 lists their decay constants (ENDF-6 §5.2). ``p_k(E)`` is the
    family's share of the delayed nu-bar. An empty list when a subsection's law
    is one the reader kept as bytes; the provenance is returned regardless.
    """
    report = report if report is not None else ConversionReport()
    provenance = _headerProvenance(mf5mt, None)
    families = []
    for index, partial in enumerate(mf5mt.partials):
        form = _member(partial)
        if form is None:
            report.unsupportedNode(
                f"MF5/MT{mf5mt.number} family {index + 1} is LF={partial.lf}, "
                f"which the reader keeps as bytes; no family gets a spectrum"
            )
            return [], provenance, report
        families.append((_weight(partial), form))
    provenance.headerFields["mf5"]["parametrised"] = True
    return families, provenance, report


def decodeMF5MT(mf5mt, report: Optional[ConversionReport] = None):
    """One MF5/MT section → ``(energyForm, provenance, report)``.

    ``energyForm`` is an :class:`XYs2d` (or :class:`Regions2d`) for NK=1 LF=1,
    a §18.3 parametrised node for NK=1 with LF=5/7/9/11/12, a
    :class:`WeightedFunctionals` for NK > 1, and ``None`` when a subsection's
    law is one the reader kept as bytes. MT455 is not a weighted sum -- call
    :func:`decodeMF5Families` for it. The provenance is returned in every case,
    and when the form is not an LF=1 table it holds the *whole* section as
    bytes, which is what the encoder writes back.
    """
    from kika.endf.classes.mf5.partials import MF5PartialTabulated

    report = report if report is not None else ConversionReport()
    partials = list(mf5mt.partials)
    mt = mf5mt.number

    modelled = None
    if len(partials) == 1 and isinstance(partials[0], MF5PartialTabulated):
        modelled = 0

    provenance = _headerProvenance(mf5mt, modelled)

    if modelled is None:
        members = [_member(partial) for partial in partials]
        missing = [p.lf for p, m in zip(partials, members) if m is None]
        if missing:
            report.unsupportedNode(
                f"MF5/MT{mt} has a subsection of LF={missing[0]}, a law the "
                f"reader keeps as bytes, so the energy distribution is absent "
                f"from this reactionSuite. The section's own bytes are kept, so "
                f"the tape still comes back with it"
            )
            return None, provenance, report
        provenance.headerFields["mf5"]["parametrised"] = True
        if len(partials) == 1:
            return members[0], provenance, report
        if mt == DELAYED_MT:
            # Reached only when there are no MF1/455 families to place them on
            # (attachDelayedSpectra tried first): a cut tape, or MF1 without 455.
            laws = ",".join(str(p.lf) for p in partials)
            report.unsupportedNode(
                f"MF5/MT{mt} has NK={len(partials)} subsections (LF=[{laws}]): "
                f"one spectrum per delayed-neutron precursor family, not a "
                f"weighted sum. Their home is §18.4's delayedNeutrons, and this "
                f"evaluation has no MF1/455 families to put them on, so they are "
                f"absent from this reactionSuite"
            )
            return None, provenance, report
        weighted = WeightedFunctionals(weighted=[
            Weighted(weight=_weight(partial), functional=member)
            for partial, member in zip(partials, members)
        ])
        return weighted, provenance, report

    partial = partials[0]
    if partial.p_values and not all(value == 1.0 for value in partial.p_values):
        report.warn(
            f"MF5/MT{mt} has NK=1 and its p(E) is not identically 1, which "
            f"ENDF-6 §5 requires of a single subsection; the model carries the "
            f"law and the weight stays in the provenance, so the round trip is "
            f"unaffected, but the two do not multiply to what the file states"
        )

    # One object, shared by the container and everything under it. Calling
    # `energyAxes()` per region would give distinct objects, and
    # `kika/gnds/encode.py:_axesUnlessNested` tests inheritance by *identity* --
    # so a second object would be read as "this child carries axes of its own".
    axes = energyAxes()

    functions = [
        _spectrumAt(energy, partial.outgoing_grids[k], partial.chi[k],
                    partial.outgoing_interp[k], k)
        for k, energy in enumerate(partial.incident_energies)
    ]
    energy = fromEndfTab2(functions, partial.tab2_interp, axes=axes)
    return energy, provenance, report


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _spectrumRecord(function: Function1d):
    x, y, pairs = function.toEndfRegions()
    return ([float(v) for v in x], [float(v) for v in y],
            [(int(a), int(b)) for a, b in pairs])


def _signature(node):
    """A nested, comparable view of a model form: every field, arrays as floats."""
    import dataclasses

    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return (type(node).__name__,
                tuple((f.name, _signature(getattr(node, f.name)))
                      for f in dataclasses.fields(node)
                      if f.name not in ("axes", "label", "index")))
    if isinstance(node, np.ndarray):
        return tuple(np.asarray(node, dtype=float).ravel().tolist())
    if isinstance(node, (list, tuple)):
        return tuple(_signature(v) for v in node)
    if hasattr(node, "value") and hasattr(node, "unit"):
        return (float(node.value), str(node.unit))
    if hasattr(node, "__iter__") and not isinstance(node, (str, bytes, dict)):
        return tuple(_signature(v) for v in node)
    return node


def _formFromProvenance(provenance: EndfProvenance, mt: int):
    """Re-decode the section the provenance holds -- what the bytes say today."""
    from kika.endf.parsers.parse_mf5 import parse_mf5_mt

    fields = provenance.headerFields["mf5"]
    section = _rebuildSection(fields, mt, None)
    lines = [line for line in str(section).split("\n") if line[72:75].strip() == str(mt)]
    parsed = parse_mf5_mt(lines, mt)
    if mt == DELAYED_MT and fields.get("nk", 1) > 1:
        families, _, _ = decodeMF5Families(parsed)
        return [form for _weight_, form in families]
    form, _, _ = decodeMF5MT(parsed)
    return form


def _refuseEdited(energyForm, provenance: EndfProvenance, mt: int) -> None:
    """Raise if *energyForm* is no longer what the kept bytes decode to.

    The parametrised laws are written back from their bytes. That is right
    only while the model still says what the bytes say; a parameter changed
    after decoding would otherwise be dropped without a word. Writing it from
    the model instead is roadmap E2's follow-up, and until then this refuses.
    """
    original = _formFromProvenance(provenance, mt)
    if _signature(original) != _signature(energyForm):
        raise ValueError(
            f"MF5/MT{mt}: the {type(energyForm).__name__} differs from the "
            f"section it was decoded from. A parametrised MF5 is written back "
            f"from its bytes, so an edit would be lost; writing one from the "
            f"model is not implemented, and refusing is the honest answer"
        )


def _rebuildSection(fields: dict, mt: int, energyForm):
    """The provenance's MF5 section, every partial from its record."""
    from kika.endf.classes.mf5.base import MF5MT

    section = MF5MT(number=mt)
    za = fields.get("za")
    section._za = float(za) if za is not None else None
    section._awr = fields.get("awr")
    section._mat = fields.get("mat")
    section._nk = fields["nk"]
    modelled = fields.get("modelled")
    section.partials = [
        _restorePartial(record, energyForm if index == modelled else None)
        for index, record in enumerate(fields["partials"])
    ]
    return section


def _restorePartial(record: dict, energyForm):
    """One provenance record → the ENDF partial it was read from."""
    from kika.endf.classes.mf5.partials import (MF5PartialRaw,
                                                MF5PartialTabulated)

    common = dict(
        u=record["u"],
        lf=record["lf"],
        p_interp=[(int(a), int(b)) for a, b in record["p_interp"]],
        p_energies=list(record["p_energies"]),
        p_values=list(record["p_values"]),
    )

    if energyForm is None:
        return MF5PartialRaw(raw_lines=list(record.get("raw_lines", [])), **common)

    functions, pairs = toEndfTab2(energyForm)
    records = [_spectrumRecord(function) for function in functions]
    return MF5PartialTabulated(
        tab2_interp=pairs,
        incident_energies=[float(f.outerDomainValue) for f in functions],
        outgoing_grids=[r[0] for r in records],
        chi=[r[1] for r in records],
        outgoing_interp=[r[2] for r in records],
        **common,
    )


def encodeMF5MT(energyForm, provenance: Optional[EndfProvenance], mt: int,
                report: Optional[ConversionReport] = None):
    """The inverse of :func:`decodeMF5MT`, byte-identical to the source section.

    Requires the ENDF provenance. NK, LF, U and ``p_k(E)`` have no GNDS
    counterpart, and the laws kika does not model live there and nowhere else —
    so without it this is not a lossy encode, it is an impossible one.
    """
    from kika.endf.classes.mf5.base import MF5MT

    report = report if report is not None else ConversionReport()
    fields = (provenance.headerFields.get("mf5")
              if provenance is not None and provenance.sourceFormat == "endf"
              else None)
    if fields is None:
        raise ValueError(
            "encodeMF5MT needs the EndfProvenance decodeMF5MT produced: NK, LF, "
            "U and p(E) are not recoverable from the model alone, and neither "
            "are the laws it does not model"
        )

    modelled = fields.get("modelled")
    parametrised = bool(fields.get("parametrised"))
    if parametrised:
        if energyForm is not None:
            _refuseEdited(energyForm, provenance, mt)
        energyForm = None
    elif (modelled is None) != (energyForm is None):
        raise ValueError(
            f"MT{mt}: the provenance says subsection {modelled!r} carried the "
            f"model form and it was handed "
            f"{'nothing' if energyForm is None else type(energyForm).__name__}"
        )

    # MAT, ZA and AWR from the MF5 block and not from the provenance's top
    # level, which is MF4's — see `_headerProvenance`. Older provenances have
    # no such keys, so the top level is the fallback rather than the source.
    fields = dict(fields, za=fields.get("za", provenance.za),
                  awr=fields.get("awr", provenance.awr),
                  mat=fields.get("mat", provenance.mat))
    return _rebuildSection(fields, mt, energyForm), report

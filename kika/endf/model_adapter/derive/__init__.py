"""GNDS → ENDF: derive the ENDF bookkeeping a suite read from GNDS does not carry.

**The problem.** Every ENDF encoder writes a section's bookkeeping — MAT, AWR,
QM, LR, the interpolation regions, MF4's LTT/LCT, MF6's laws — from the node's
``provenance``, and only an ENDF read fills that. A suite read from GNDS, or
built by hand, has none, so it could not be written as a tape at all.

**The answer, FUDGE's.** That bookkeeping is not stored anywhere in GNDS; it is
*derived* from the physics when the tape is written
(``brownies/legacy/toENDF6``). This package does the same in one pass:
:func:`deriveEndfProvenance` walks the suite and gives every node that has no
ENDF provenance one derived from the model. The encoders do not change — they
write ``provenance`` as they always have, whether a read or this pass put it
there — and a suite that came from ENDF is never touched, so the ENDF → ENDF
path stays byte-identical.

**Three kinds of field** (decided 2026-10-08, ``docs/library/gnds_to_endf_plan.md``):
derived from the model; taken from the file's own FUDGE ``ENDFconversionFlags``
when it has them; or a documented neutral default, named in the report. A field
with no neutral default is **refused**, never invented.

**The gate** is :mod:`.oracle`: decode real tapes, strip every provenance,
derive, and compare field by field with what the file said.

Deriver modules register themselves in :data:`DERIVERS`, in the order they run;
the suite's runs first because every other node takes MAT, ZA and AWR from it.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

__all__ = ["DERIVED", "DerivationContext", "DERIVERS", "hasEndfProvenance", "needsDerivation",
           "provenanceNodes", "deriveEndfProvenance"]

#: ``sourceFormat`` of a provenance this package wrote, so a reader of the
#: model can tell bookkeeping the file stated from bookkeeping kika derived.
DERIVED = "derived"


@dataclass
class DerivationContext:
    """What every deriver needs and only the suite knows."""

    suite: object
    mat: Optional[int] = None
    za: Optional[int] = None
    awr: Optional[float] = None
    #: FUDGE's ``ENDFconversionFlags``, ``href → flags``, when the file had them.
    conversionFlags: Dict[str, str] = field(default_factory=dict)


#: ``(name, applies, derive)``: ``applies(node)`` says whether the deriver is for
#: this kind of node, ``derive(node, path, context, report)`` returns its
#: provenance or ``None`` when the node has nothing to write.
DERIVERS: List[Tuple[str, Callable[[object], bool],
                     Callable[[object, str, DerivationContext, object], Optional[object]]]] = []


#: The order the derivers run in, by name. **Explicit, not import order**: the
#: suite deriver sets the MAT, ZA and AWR every other one stamps, and a test
#: that imported ``derive.reactions`` first used to register it ahead of
#: ``derive.suite``, so every reaction came out with ZA and AWR of ``None``.
#: A deriver not named here runs after these, in registration order.
RUN_ORDER = ("suite", "reactions", "products", "multiplicities",
             "fissionEnergyRelease", "delayedNeutrons", "covariances")


def register(name: str, applies, derive) -> None:
    if any(entry[0] == name for entry in DERIVERS):
        return
    DERIVERS.append((name, applies, derive))


def _inRunOrder():
    rank = {name: k for k, name in enumerate(RUN_ORDER)}
    return sorted(DERIVERS, key=lambda entry: rank.get(entry[0], len(rank)))


def hasEndfProvenance(node) -> bool:
    """Whether *node* carries ENDF bookkeeping, read or derived.

    Not "has any provenance": a suite read from GNDS carries a
    ``GndsProvenance``, which says where it came from and nothing an ENDF
    encoder can use.
    """
    provenance = getattr(node, "provenance", None)
    return getattr(provenance, "sourceFormat", None) in ("endf", DERIVED)


def needsDerivation(suite) -> bool:
    """Whether *suite* has no ENDF header to write back: read from GNDS, or by hand.

    A suite decoded from ENDF is never derived, even where a node of it lacks
    provenance: there the absence is a statement (a distribution kika inferred,
    which the tape did not carry), and deriving would add sections the source
    never had.
    """
    provenance = getattr(suite, "provenance", None)
    return provenance is None or getattr(provenance, "sourceFormat", None) in ("gnds", DERIVED)


_LEAVES = (str, bytes, int, float, complex, bool, type(None))


def provenanceNodes(suite) -> Iterator[Tuple[str, object]]:
    """Every node under *suite* that has a ``provenance`` slot, with a stable path.

    The path names nodes by what they are (``reactions[MT51]``, ``products[n]``)
    rather than by position, so the oracle can pair the nodes of two suites
    built differently. ``provenance`` and ``report`` are not descended into, and
    arrays are leaves.
    """
    seen = set()

    def name(obj, fallback):
        for attr in ("label", "pid", "id"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                return value
            label = getattr(value, "label", None)
            if isinstance(label, str) and label:
                return label
        return fallback

    def walk(obj, path):
        if isinstance(obj, _LEAVES) or id(obj) in seen:
            return
        if type(obj).__module__.startswith("numpy"):
            return
        seen.add(id(obj))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            if any(f.name == "provenance" for f in dataclasses.fields(obj)):
                yield path, obj
            for f in dataclasses.fields(obj):
                if f.name in ("provenance", "report"):
                    continue
                yield from walk(getattr(obj, f.name), f"{path}.{f.name}")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                yield from walk(value, f"{path}[{key}]")
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                yield from walk(value, f"{path}[{name(value, index)}]")
        elif hasattr(obj, "__iter__") and hasattr(obj, "__len__"):
            try:
                items = list(obj)
            except Exception:
                return
            for index, value in enumerate(items):
                yield from walk(value, f"{path}[{name(value, index)}]")
            # `Sums` iterates its crossSectionSums and holds §21.1's other
            # child beside them: without this the nu-bars of MT452/455, which
            # are multiplicitySums, were never visited.
            extra = getattr(obj, "multiplicitySums", None)
            if extra is not None:
                yield from walk(extra, f"{path}.multiplicitySums")

    yield from walk(suite, "suite")


def deriveEndfProvenance(suite, report=None, *, mat: Optional[int] = None):
    """Give every node of *suite* that lacks ENDF provenance a derived one. In place.

    Returns the report. Nodes that already carry provenance keep it. Callers that
    must not change their suite — the tape writer — pass a copy.
    """
    from kika.nuclear_data.model import ConversionReport

    from . import suite as _suite  # noqa: F401  registers, in run order
    from . import reactions as _reactions  # noqa: F401
    from . import products as _products  # noqa: F401
    from . import fission as _fission  # noqa: F401
    from . import covariances as _covariances  # noqa: F401

    report = report if report is not None else ConversionReport()
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    context = DerivationContext(suite=suite, mat=mat)
    flags = EndfConversionFlags.of(suite)
    if flags:
        for href, text in flags.conversions:
            context.conversionFlags[href] = text
    if hasEndfProvenance(suite):
        # Nodes added to a suite that did come from a tape take its numbers.
        context.mat = mat if mat is not None else suite.provenance.mat
        context.za, context.awr = suite.provenance.za, suite.provenance.awr
    for name, applies, derive in _inRunOrder():
        for path, node in provenanceNodes(suite):
            if hasEndfProvenance(node) or not applies(node):
                continue
            provenance = derive(node, path, context, report)
            if provenance is not None:
                node.provenance = provenance

    # What is still without ENDF bookkeeping is not written, and a tape that is
    # missing it has to say so: the writer only emits a section a provenance
    # asks for, so without this the gap would be silent.
    from ..multiplicity import nubarNode

    nubars = {id(nubarNode(suite, mt)) for mt in (452, 455, 456)} - {id(None)}
    underived = Counter()
    for path, node in provenanceNodes(suite):
        if hasEndfProvenance(node) or not _carriesData(node, nubars):
            continue
        underived[type(node).__name__] += 1
    for kind, count in sorted(underived.items()):
        report.lost(
            f"{count} {kind} node(s) have no ENDF bookkeeping that kika can derive "
            f"yet (docs/library/gnds_to_endf_plan.md), so their sections are not "
            f"in the written tape"
        )
    return report


def _carriesData(node, nubars=frozenset()) -> bool:
    """Whether *node* is something a tape would have a section for.

    A product's multiplicity is written inside its MF6 section, so only the
    nu-bars, which are sections of their own (MF1/452-456), count here.
    """
    from kika.nuclear_data.model import Product, Reaction
    from kika.nuclear_data.model.output_channel import Multiplicity

    if isinstance(node, Multiplicity):
        return id(node) in nubars

    if isinstance(node, Reaction):
        return node.ENDF_MT is not None
    if isinstance(node, Product):
        from kika.nuclear_data.model import Unspecified

        distribution = getattr(node, "distribution", None)
        forms = list(distribution.values()) if distribution is not None else []
        return any(not isinstance(form, Unspecified) for form in forms)
    return True

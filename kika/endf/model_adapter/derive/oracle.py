"""G0: the gate — strip a real tape's provenance, derive it back, compare.

A suite decoded from ENDF carries the bookkeeping the file stated. Hide it,
run :func:`~kika.endf.model_adapter.derive.deriveEndfProvenance`, and every
field that comes back different is either a deriver's mistake or a field the
model does not determine. The table says which and how many, before anyone has
to argue about it. Run as a module for the table::

    python -m kika.endf.model_adapter.derive.oracle tape.endf [tape.endf ...]
"""
from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import deriveEndfProvenance, hasEndfProvenance, provenanceNodes

__all__ = ["Mismatch", "COMPARED", "IGNORED_HEADER_FIELDS", "strip", "compare", "oracle"]

#: The provenance fields an encoder writes. ``covarianceFindings``,
#: ``descriptiveText``, ``evaluationInfo`` and ``directory`` are not compared:
#: the first is a checker's note, the next two are MF1/451's text (gated in
#: ``test_header_synthesis.py``) and the directory is rebuilt from the written
#: tape.
COMPARED = ("mat", "za", "awr", "qm", "lr", "interpolationRegions")

#: ``headerFields`` keys the derivation adds for the encoder's sake and the
#: decoder never writes.
IGNORED_HEADER_FIELDS = {"qi"}


@dataclass(frozen=True)
class Mismatch:
    path: str
    kind: str
    field: str
    stated: object
    derived: object


def strip(suite):
    """A deep copy of *suite* with every ENDF provenance removed."""
    bare = copy.deepcopy(suite)
    for _, node in provenanceNodes(bare):
        if hasEndfProvenance(node):
            node.provenance = None
    return bare


def _fields(record: str):
    """An ENDF record's six 11-column fields, a blank one read as the 0 it means."""
    text = record[:66].ljust(66)
    return [text[k:k + 11].strip() or "0" for k in range(0, 66, 11)]


def _same(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str) and a != b and max(len(a), len(b)) >= 60:
        # Kept records (MF5's raw lines): an evaluator who pads an interpolation
        # record with zeros and a writer that pads it with blanks state the same
        # numbers. ENDF reads a blank integer field as 0, so this compares what
        # the record says, not its spelling.
        return _fields(a) == _fields(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        # "pad" is how a section's writer spelled its short records (blank or
        # zero), not a number the section states; the derived side has none.
        a = {k: v for k, v in a.items() if k != "pad"}
        b = {k: v for k, v in b.items() if k != "pad"}
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def compare(original, derived) -> List[Mismatch]:
    """Every compared field of every node where *derived* differs from *original*."""
    stated: Dict[str, Tuple[str, object]] = {
        path: (type(node).__name__, node.provenance)
        for path, node in provenanceNodes(original) if hasEndfProvenance(node)}
    got = {path: node.provenance for path, node in provenanceNodes(derived)
           if hasEndfProvenance(node)}

    out: List[Mismatch] = []
    for path, (kind, provenance) in stated.items():
        mine = got.get(path)
        if mine is None:
            out.append(Mismatch(path, kind, "<node>", "provenance", None))
            continue
        for name in COMPARED:
            a, b = getattr(provenance, name, None), getattr(mine, name, None)
            if not _same(a, b):
                out.append(Mismatch(path, kind, name, a, b))
        fields = dict(getattr(provenance, "headerFields", None) or {})
        derivedFields = {k: v for k, v in (getattr(mine, "headerFields", None) or {}).items()
                         if k not in IGNORED_HEADER_FIELDS}
        for key in sorted(set(fields) | set(derivedFields), key=str):
            if not _same(fields.get(key), derivedFields.get(key)):
                out.append(Mismatch(path, kind, f"headerFields.{key}",
                                    fields.get(key), derivedFields.get(key)))
    return out


def oracle(suite, report=None):
    """``(mismatches, report)`` for one ENDF-decoded *suite*.

    A deriver that refuses is a finding, not a crash: the refusal is returned as
    one mismatch on the node it stopped at, so a table over many tapes still
    comes out.
    """
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()
    bare = strip(suite)
    try:
        deriveEndfProvenance(bare, report, mat=getattr(suite.provenance, "mat", None))
    except ValueError as refused:
        return [Mismatch("suite", "refused", "<derive>", None, str(refused))], report
    return compare(suite, bare), report


def _main(paths) -> None:  # pragma: no cover - a reporting tool
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf

    totals: Counter = Counter()
    wrong: Counter = Counter()
    examples: Dict[Tuple[str, str], Mismatch] = {}
    for path in paths:
        suite, _ = decodeReactionSuite(read_endf(path))
        mismatches, report = oracle(suite)
        counted = defaultdict(int)
        for _, node in provenanceNodes(suite):
            if hasEndfProvenance(node):
                counted[type(node).__name__] += 1
        for kind, n in counted.items():
            totals[kind] += n
        for m in mismatches:
            wrong[(m.kind, m.field)] += 1
            examples.setdefault((m.kind, m.field), m)
        print(f"{path}: {len(mismatches)} mismatch(es); "
              f"{len(report.approximations)} approximation(s)")
    print()
    print(f"{'node':<22}{'field':<34}{'wrong':>7}{'nodes':>7}  example")
    for (kind, field), n in sorted(wrong.items()):
        m = examples[(kind, field)]
        print(f"{kind:<22}{field:<34}{n:>7}{totals.get(kind, 0):>7}  "
              f"{m.path}: {m.stated!r} -> {m.derived!r}"[:200])


if __name__ == "__main__":  # pragma: no cover
    import sys

    _main(sys.argv[1:])

"""Every node FUDGE's ``gnds.xsd``, ``covariances.xsd`` and ``xData.xsd``
declare, written to a file this repository carries.

**Why the list is committed rather than read at test time.** The schemas ship
with FUDGE (``$KIKA_FUDGE_DIR``, a checkout of its latest release), which not
every machine has. A gate that parses them directly is not a gate there — it is
a ``pytest.skip``. So the extraction runs where FUDGE is, by hand, and its
output is the fixed point that ``test_capabilities.py`` measures the capability
table against. A second test re-runs this extraction where the schemas exist
and fails on drift, which is how a new FUDGE release arrives:
re-run this script, read the diff and give every new node a capability row.

**Why it parses XML instead of grepping.** ``xs:element`` declarations do not
put ``name`` first. A regex ``xs:element name="…"`` misses twelve of the three
hundred of FUDGE 6.10 — ``resolved``, ``unresolved``, ``spinGroup``, ``isotope``,
``isotopes``, ``atomic``, ``J``, ``L``, ``configuration``,
``averageProductEnergy``, ``resonanceReaction``, ``weighted`` — every one of
them because ``minOccurs`` precedes ``name`` on that line. Four of those twelve
are load-bearing §19 nodes, so the cheap route would have under-declared
exactly the region kika reads most carefully.

**Why the line numbers come from a second pass.** ``xml.etree.ElementTree``
does not record them and ``lxml`` is not a dependency of this library (nor is
it installed in the poetry venv). So pass one is ElementTree, which gets the
name set right, and pass two is a text scan for the line each name is first
declared on — a citation, not a parse. Every one of the three hundred gets one.

Usage::

    python kika/gnds/tests/data/build_schema_census.py [--check]

``--check`` re-extracts and diffs against the committed file without writing,
which is what the ``tape``-marked drift test calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent

#: Where the extraction writes, and where the test reads. Under ``tests/``
#: rather than beside the module because only the test needs it and
#: ``pyproject.toml``'s ``exclude`` keeps ``kika/**/tests`` out of the wheel —
#: so this costs the published package nothing.
CENSUS = HERE / "gnds_schema_nodes.json"

#: ``covarianceSuite`` has no global declaration in ``gnds.xsd`` (§25.1.1 makes
#: it a root in its own right), and since FUDGE 6.11 both schemas take their
#: shared types from ``xData.xsd``. Paths are relative to the checkout, so the
#: committed file names no machine.
FUDGE = Path(os.environ.get("KIKA_FUDGE_DIR", "missing"))
SCHEMAS = (
    ("gnds.xsd", Path("fudge/gnds.xsd")),
    ("covariances.xsd", Path("fudge/covariances/covariances.xsd")),
    ("xData.xsd", Path("xData/xData.xsd")),
)

#: The GNDS version these schemas describe. ``gnds.xsd``'s own header lists 2.0
#: as its newest entry; ``kika/gnds/version.py`` reads 2.0 and 2.1 through one
#: path because, for everything kika models, they are the same format.
GNDS_VERSION = "2.0"

XS = "{http://www.w3.org/2001/XMLSchema}"


def declaredNodes(path: Path) -> Dict[str, int]:
    """``{node name: the line it is first declared on}`` for one schema.

    Pass one is the parse, and it is what decides membership. Pass two only
    attaches a line to a name already known, so a scan that fails to find one
    is a bug here rather than a node quietly missing from the census — hence
    the raise instead of a ``None``.
    """
    names = {element.get("name")
             for element in ET.parse(str(path)).getroot().iter(XS + "element")
             if element.get("name")}

    lines = path.read_text(encoding="utf-8").splitlines()
    located = {}
    for name in names:
        pattern = re.compile(r"<xs:element\b[^>]*\bname=\"%s\"" % re.escape(name))
        hit = next((n for n, line in enumerate(lines, 1) if pattern.search(line)),
                   None)
        if hit is None:
            raise AssertionError(
                f"{path.name} declares <{name}> and the line scan did not find "
                f"it. The scan is a citation aid for a name the parse already "
                f"produced, so this means the two passes disagree"
            )
        located[name] = hit
    return located


def fudgeVersion() -> str:
    text = (FUDGE / "fudge" / "fudgeVersion.py").read_text(encoding="utf-8")
    major, minor = (re.search(rf"^FUDGE_{part}VERSION\s*=\s*(\d+)", text, re.M).group(1)
                    for part in ("MAJOR", "MINOR"))
    return f"{major}.{minor}"


def extract() -> dict:
    """The whole census, as it is written to disk. Sorted, so a diff is a diff."""
    missing = [str(FUDGE / path) for _, path in SCHEMAS
               if not (FUDGE / path).exists()]
    if missing:
        raise FileNotFoundError(
            "the FUDGE schemas are not on this machine: " + ", ".join(missing) +
            ". This script only runs where they are; the committed census is "
            "what every other machine reads."
        )

    nodes: Dict[str, Tuple[str, int]] = {}
    schemas = {}
    for label, relative in SCHEMAS:
        path = FUDGE / relative
        raw = path.read_bytes()
        schemas[label] = {
            "path": relative.as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, line in declaredNodes(path).items():
            # A name declared in two files keeps the first in SCHEMAS order.
            nodes.setdefault(name, (label, line))

    return {
        "gndsVersion": GNDS_VERSION,
        "fudgeVersion": fudgeVersion(),
        "schemas": schemas,
        "nodeCount": len(nodes),
        "nodes": {name: list(where) for name, where in sorted(nodes.items())},
    }


def write() -> None:
    census = extract()
    CENSUS.write_text(json.dumps(census, indent=2, sort_keys=False) + "\n",
                      encoding="utf-8")
    print(f"[write] {CENSUS} — {census['nodeCount']} nodes")


def check() -> int:
    """Non-zero if the committed census no longer matches the live schemas."""
    live = extract()
    committed = json.loads(CENSUS.read_text(encoding="utf-8"))

    problems = []
    for label in live["schemas"]:
        was = committed["schemas"].get(label, {}).get("sha256")
        now = live["schemas"][label]["sha256"]
        if was != now:
            problems.append(f"{label} changed: sha256 {was} -> {now}")

    added = sorted(set(live["nodes"]) - set(committed["nodes"]))
    removed = sorted(set(committed["nodes"]) - set(live["nodes"]))
    if added:
        problems.append(f"the schemas now declare {len(added)}: {', '.join(added)}")
    if removed:
        problems.append(f"the schemas no longer declare {len(removed)}: "
                        f"{', '.join(removed)}")

    for problem in problems:
        print(f"[drift] {problem}")
    if not problems:
        print(f"[ok] {live['nodeCount']} nodes, FUDGE {live['fudgeVersion']}, "
              f"every schema byte-identical")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv[1:] else (write() or 0))

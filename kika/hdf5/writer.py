"""``writeSuite``: a suite → ``<dir>/<name>.h5`` plus ``<dir>/cross_sections.xml``.

Two kinds of suite come in:

- **already processed** -- read from an OpenMC file, or built by
  :func:`processSuite`: ``heated_<T>`` styles on a common grid per temperature
  and ``MonteCarlo`` distributions. It is serialised as it is.
- **an evaluation** -- decoded from ENDF or GNDS. It is processed first
  (:func:`processSuite`, roadmap H4): reconstructed at 0 K if it has no
  ``recon`` style, broadened to each temperature, put on a common grid per
  temperature, its distributions tabulated with their CDF, and the pieces
  kika does not compute yet taken from NJOY (D7).

Like the G4NDL writer, the file is written to a temporary name in the target
directory, **read back** (records, then the neutron decoder) and compared with
what was meant, and only then moved into place. ``cross_sections.xml`` in the
directory is created or updated to index it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import records as R
from .exceptions import HDF5FormatError
from .library import CrossSectionsXml, LibraryEntry

__all__ = ["writeSuite", "isProcessed", "indexFile"]


def _m():
    from kika.nuclear_data import model

    return model


def isProcessed(suite) -> bool:
    """The suite carries ``heated_<T>`` cross sections, the shape the encoder takes."""
    m = _m()
    return any(isinstance(s, m.Heated) and s.label.startswith("heated_") for s in suite.styles)


def indexFile(directory: Path, file: Path, material: str, kind: str = "neutron") -> Path:
    """Add (or replace) the entry for ``file`` in ``directory/cross_sections.xml``."""
    xml = directory / "cross_sections.xml"
    index = CrossSectionsXml.read(xml) if xml.is_file() else CrossSectionsXml(source=xml)
    relative = os.path.relpath(file, directory).replace(os.sep, "/")
    index.entries = [e for e in index.entries
                     if not (e.type == kind and material in e.materials)]
    index.entries.append(LibraryEntry((material,), relative, kind))
    return index.write(xml)


def writeSuite(suite, directory, temperatures=None, *, report=None, index: bool = True,
               fileName: Optional[str] = None, **options):
    """Write ``suite`` into ``directory`` as OpenMC HDF5. Returns the report.

    ``temperatures`` (kelvin, or group names like ``"294K"``) chooses which
    temperatures go in; for a processed suite, by default all it has. The
    remaining ``options`` reach :func:`processSuite` when the suite is an
    evaluation.
    """
    m = _m()
    from .neutron_decode import decodeNeutron
    from .neutron_encode import encodeNeutron

    report = report if report is not None else m.ConversionReport()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if suite.interaction == "thermalNeutronScatteringLaw":
        from .thermal_encode import writeThermal

        return writeThermal(suite, directory, temperatures=temperatures, report=report,
                            index=index, fileName=fileName)
    if not isProcessed(suite):
        from .processing import processSuite

        suite = processSuite(suite, temperatures=temperatures, report=report, **options)
        temperatures = None
    elif options:
        raise ValueError(f"processing options {sorted(options)} were given for a suite that "
                         f"is already processed")

    tree = encodeNeutron(suite, temperatures, report=report,
                         awr=options.get("awr") if options else None)
    name = next(iter(tree.children))
    target = directory / (fileName or f"{name}.h5")
    temporary = target.with_name(f".{target.name}.tmp")
    R.writeFile(tree, temporary)
    try:
        written = R.readFile(temporary)
        differences = R.compareTrees(tree, written, limit=10)
        if differences:
            raise HDF5FormatError(f"the file read back differs from what was written: "
                                  f"{differences}", path=temporary)
        decodeNeutron(written, path=temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
    if index:
        indexFile(directory, target, name)
    return report

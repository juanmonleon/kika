"""What writing a collection of materials for one code leaves out, and whether the
library the code will run with has every table the cards ask for."""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

__all__ = ["EmissionReport", "read_library_tables"]


def read_library_tables(path) -> Set[str]:
    """The table names (``92235.80c``, ``lwtr.20t``, ``U235``) a library index lists, in lower case.

    Parameters
    ----------
    path
        An MCNP ``xsdir`` (recognised by its ``directory`` line), an OpenMC
        ``cross_sections.xml`` (recognised by its ``<cross_sections>`` root) or a
        Serpent ``xsdata`` file (alias and name in the first two columns).

    Returns
    -------
    set of str
    """
    from ..ace.xsdir import _entries

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "<cross_sections" in text:
        from ..hdf5.library import CrossSectionsXml

        return {name.lower() for entry in CrossSectionsXml.read(path).entries for name in entry.materials}
    if any(line.strip().lower() == "directory" for line in text.splitlines()):
        _, entries = _entries(text)
        return {name.lower() for name, _ in entries}
    names = set()
    for line in text.splitlines():
        tokens = line.split()
        names.update(t.lower() for t in tokens[:2])
    return names


@dataclass
class EmissionReport:
    """The report of :meth:`~kika.materials.MaterialCollection.emission_report`.

    Each material is **Failed** when a table its card asks for is not in the
    library or its name clashes with another, **Warning** when its card leaves
    something out, **OK** otherwise.

    Attributes
    ----------
    code : str
        ``'mcnp'`` or ``'serpent'``.
    library : str or None
        The library file the tables were checked against, or None.
    materials : list of dict
        Per material: ``id``, ``name``, ``status``, ``losses`` (what the card
        leaves out), ``missing`` (``[symbol, table]`` not in the library),
        ``no_library`` (symbols whose card gives no suffix and no default) and
        ``read`` (what reading the material reported).
    clashes : list of str
        Card names given to more than one material.
    """

    code: str
    library: Optional[str]
    materials: List[Dict[str, Any]] = field(default_factory=list)
    clashes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``'Failed'``, ``'Warning'`` or ``'OK'``: the worst over the materials."""
        statuses = {m["status"] for m in self.materials}
        if self.clashes or "Failed" in statuses:
            return "Failed"
        return "Warning" if "Warning" in statuses else "OK"

    def to_dict(self) -> Dict[str, Any]:
        """The report as JSON-ready values."""
        return json.loads(json.dumps({"code": self.code, "library": self.library, "status": self.status,
                                      "materials": self.materials, "clashes": self.clashes}))

    def to_html(self) -> str:
        """The report as one HTML page."""
        rows = []
        for m in self.materials:
            notes = [f"not in the library: {t} ({s})" for s, t in m["missing"]]
            notes += [f"{s}: no library suffix and no default; {self.code.upper()} picks one" for s in m["no_library"]]
            notes += m["losses"] + [f"on reading: {r}" for r in m["read"]]
            items = "".join(f"<li>{html.escape(n)}</li>" for n in notes) or "<li>everything is written</li>"
            name = html.escape(m["name"] or "")
            rows.append(f'<tr class="{m["status"].lower()}"><td>{m["id"]}</td><td>{name}</td>'
                        f'<td>{m["status"]}</td><td><ul>{items}</ul></td></tr>')
        clashes = "".join(f"<li>{html.escape(c)}</li>" for c in self.clashes)
        library = html.escape(self.library) if self.library else "none given: tables not checked"
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Materials for "
            f"{self.code.upper()}</title><style>{_CSS}</style></head><body>"
            f"<h1>Materials written for {self.code.upper()}: {self.status}</h1>"
            f"<p>Library: {library}</p>"
            + (f"<h2>Names given to more than one material</h2><ul>{clashes}</ul>" if clashes else "")
            + "<table><tr><th>Id</th><th>Name</th><th>Status</th><th>What to know</th></tr>"
            + "".join(rows) + "</table></body></html>"
        )

    def save(self, path) -> Path:
        """Write :meth:`to_html` to *path* and return it."""
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return f"<EmissionReport {self.code}: {self.status}, {len(self.materials)} materials>"

    def _repr_html_(self) -> str:
        return self.to_html()


_CSS = (
    "body{font-family:system-ui,sans-serif;margin:1.5rem;color:#1d1d1f;background:#fff}"
    "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.3rem .5rem;"
    "text-align:left;vertical-align:top}ul{margin:0;padding-left:1.1rem}"
    "tr.failed td:nth-child(3){color:#b00020;font-weight:600}"
    "tr.warning td:nth-child(3){color:#8a5a00;font-weight:600}"
)

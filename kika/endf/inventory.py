"""What a tape holds, from its MF1/MT451 directory alone.

:func:`tape_inventory` reads the tape identification line, the four CONT
records of MT451, the NWD lines of text and the NXC lines of the directory,
and stops: a folder of hundreds of tapes is listed in the time it takes to
read a few lines of each, where parsing them would read gigabytes.

A directory is only what the evaluator wrote. When it is missing or does not
add up, the inventory says so (``sections=None`` and ``problem``) rather than
hand back an empty or partial list as if it were the tape's content;
``scan=True`` reads the MAT/MF/MT columns of every line instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .utils import parse_line

__all__ = ["TapeInventory", "tape_inventory"]


@dataclass(frozen=True)
class TapeInventory:
    """The identity of a tape's first material and the (MF, MT) it holds.

    ``sections`` is ``None`` when they could not be established; ``problem``
    then says why. ``source`` is ``"directory"`` (MT451) or ``"scan"`` (every
    line's MF/MT columns).
    """

    path: Path
    mat: Optional[int] = None
    za: Optional[int] = None
    awr: Optional[float] = None
    nlib: Optional[int] = None
    nmod: Optional[int] = None
    nsub: Optional[int] = None
    library_tag: Optional[str] = None
    zsymam: Optional[str] = None
    sections: Optional[Tuple[Tuple[int, int], ...]] = None
    source: Optional[str] = None
    problem: Optional[str] = None

    @property
    def mf(self) -> Optional[Tuple[int, ...]]:
        """The files present, sorted; ``None`` when the sections are unknown."""
        if self.sections is None:
            return None
        return tuple(sorted({mf for mf, _ in self.sections}))

    def mt(self, mf: int) -> Optional[Tuple[int, ...]]:
        """The sections of one file; ``None`` when the sections are unknown."""
        if self.sections is None:
            return None
        return tuple(mt for f, mt in self.sections if f == mf)

    def to_dict(self) -> Dict[str, Any]:
        """Plain Python types only, for JSON and msgpack."""
        return {
            "path": str(self.path), "name": self.path.name, "mat": self.mat, "za": self.za,
            "awr": self.awr, "nlib": self.nlib, "nmod": self.nmod, "nsub": self.nsub,
            "library_tag": self.library_tag, "zsymam": self.zsymam,
            "sections": None if self.sections is None else [list(s) for s in self.sections],
            "mf": None if self.mf is None else list(self.mf),
            "source": self.source, "problem": self.problem,
        }


def _ids(line: str) -> Optional[Tuple[int, int, int]]:
    """MAT, MF, MT from columns 67-75, or ``None`` if they are not integers."""
    try:
        return int(line[66:70]), int(line[70:72]), int(line[72:75])
    except ValueError:
        return None


def _int(value) -> Optional[int]:
    return None if value is None else int(round(float(value)))


def _lines(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\r\n")


def _scan(path: Path, mat: Optional[int]) -> Tuple[Tuple[int, int], ...]:
    """Every (MF, MT) of material *mat* (the first one if ``None``), in file order."""
    seen: List[Tuple[int, int]] = []
    last = None
    for line in _lines(path):
        ids = _ids(line)
        if ids is None or ids[0] <= 0 or ids[1] == 0 or ids[2] == 0:
            continue
        if mat is None:
            mat = ids[0]
        elif ids[0] != mat:
            if seen:
                break  # the next material of a multi-material tape
            continue
        key = (ids[1], ids[2])
        if key != last and key not in seen:
            seen.append(key)
        last = key
    return tuple(seen)


def tape_inventory(path: Union[str, Path], *, scan: bool = False) -> TapeInventory:
    """Identify an ENDF tape and list its sections from the MT451 directory.

    Parameters
    ----------
    path : str or Path
        An ENDF-6 tape. On a tape with several materials, the first one.
    scan : bool
        Read the MF/MT columns of every line instead of trusting the
        directory. Slower (the whole tape is read, though nothing is parsed),
        and what to fall back to when ``problem`` is set.

    Returns
    -------
    TapeInventory
        ``sections`` is ``None``, with ``problem`` saying why, when the tape
        has no MT451 or its directory is empty, cut short, or does not list
        MT451 itself. Nothing is raised for a tape that is not ENDF; a file
        that cannot be opened raises ``OSError``.
    """
    path = Path(path)
    library_tag = None
    head: List[str] = []
    nwd = nxc = None
    body: List[str] = []
    first = True
    for line in _lines(path):
        ids = _ids(line)
        if first:
            first = False
            if ids is not None and ids[1] == 0 and ids[2] == 0:
                library_tag = line[:66].strip() or None  # the TPID record
                continue
        if ids is None or ids[1] != 1 or ids[2] != 451:
            if head:
                break  # past MT451 (or a SEND)
            continue
        if len(head) < 4:
            head.append(line)
            if len(head) == 4:
                c = parse_line(head[3])
                nwd, nxc = _int(c.get("C5")) or 0, _int(c.get("C6")) or 0
            continue
        body.append(line)
        if len(body) >= nwd + nxc:
            break

    if len(head) < 4:
        problem = "no MF1/MT451 found" if not head else "MF1/MT451 cut short"
        inv = TapeInventory(path, library_tag=library_tag, problem=problem)
        return _rescan(inv, None) if scan else inv

    c1, c3 = parse_line(head[0]), parse_line(head[2])
    mat = _ids(head[0])[0]
    zsymam = (body[0][:11].strip() or None) if nwd and body else None
    fields = dict(path=path, mat=mat, za=_int(c1.get("C1")),
                  awr=None if c1.get("C2") is None else float(c1.get("C2")),
                  nlib=_int(c1.get("C5")), nmod=_int(c1.get("C6")), nsub=_int(c3.get("C5")),
                  library_tag=library_tag, zsymam=zsymam)
    if scan:
        return TapeInventory(**fields, sections=_scan(path, mat), source="scan")

    entries: List[Tuple[int, int]] = []
    for line in body[nwd:nwd + nxc]:
        c = parse_line(line)
        mf, mt = _int(c.get("C3")), _int(c.get("C4"))
        if mf is None or mt is None or mt == 0:
            break
        entries.append((mf, mt))
    problem = None
    if nxc == 0:
        problem = "the MT451 directory is empty (NXC = 0)"
    elif len(body) < nwd + nxc:
        problem = (f"the MT451 directory is cut short: NXC = {nxc}, "
                   f"{max(0, len(body) - nwd)} lines in the section")
    elif len(entries) < nxc:
        problem = f"the MT451 directory has {len(entries)} usable entries of NXC = {nxc}"
    elif (1, 451) not in entries:
        problem = "the MT451 directory does not list MF1/MT451 itself"
    if problem is not None:
        return TapeInventory(**fields, problem=problem)
    return TapeInventory(**fields, sections=tuple(entries), source="directory")


def _rescan(inv: TapeInventory, mat: Optional[int]) -> TapeInventory:
    sections = _scan(inv.path, mat)
    if not sections:
        return inv
    return TapeInventory(inv.path, library_tag=inv.library_tag, sections=sections,
                         source="scan", problem=inv.problem)

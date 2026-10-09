"""A directory of ENDF tapes or ACE tables, as a library: index it from headers only.

A viewer that lists a library before reading any of it needs, per file, the
nuclide and little else; a full ``read_endf`` costs seconds per tape and a
``read_ace`` far more, and a library is hundreds of files, often on a network
share. :func:`index_endf` and :func:`index_ace` read **the head of each file**
-- the first records of MF1/MT451 for ENDF, the first line of an ACE table --
and nothing else. The files themselves are read later, one at a time, by
whoever opens them.

What each header gives:

* **ENDF** -- ``ZA``, ``AWR`` and ``MAT`` from the HEAD of MF1/MT451, ``LISO``
  (the isomeric state) from its second record, ``NSUB`` (the sublibrary) from
  the third and ``TEMP`` from the fourth. A file whose first material is not
  MF1/MT451-led (a GENDF, a covariance-only tape) is reported as unreadable.
* **ACE** -- the ZAID, ``AWR`` and the temperature (``kT`` in MeV, converted to
  kelvin) from the first line, legacy or 2.0. **The mass number is taken from
  the AWR**, not the ZAID: libraries encode an isomer differently in the ZAID
  (JEFF-4.0 writes Am-242m as 95342, LANL Lib81 as 95642), and ``round(AWR x
  m_n)`` is unambiguous for every nuclide. The isomeric state is what the ZAID
  adds on top of it (``+100 m``, or ``+300 +100 m``).

Both return :class:`IndexedFile` records keyed by a target name in the G4NDL
spelling (``Fe56``, ``Am242m1``, ``Cnat``), so an app can treat every library
kind alike.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

__all__ = ["IndexedFile", "LibraryIndex", "index_endf", "index_ace", "target_name"]

_HEAD_BYTES = 2048
_NEUTRON_MASS_AMU = 1.00866491595
_K_PER_MEV = 1.160451812e10


@dataclass(frozen=True)
class IndexedFile:
    target: str
    Z: int
    A: Optional[int]
    M: int
    path: str
    file_name: str
    size: int
    awr: Optional[float] = None
    temperature: Optional[float] = None      # kelvin
    mat: Optional[int] = None                # ENDF
    nsub: Optional[int] = None               # ENDF
    zaid: Optional[str] = None               # ACE, as written ("26056.10c")


@dataclass
class LibraryIndex:
    root: str
    format: str
    files: List[IndexedFile]
    unreadable: List[str]
    duplicates: List[str]

    @property
    def name(self) -> str:
        return Path(self.root).name

    def temperatures(self) -> List[float]:
        return sorted({round(f.temperature, 1) for f in self.files if f.temperature is not None})

    def describe(self) -> dict:
        """The library as a viewer lists it: one entry per file, sorted by nuclide."""
        from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

        entries = []
        for f in sorted(self.files, key=lambda f: (f.Z, f.A or 0, f.M, f.temperature or 0, f.file_name)):
            entry = asdict(f)
            entry["element"] = ATOMIC_NUMBER_TO_SYMBOL.get(f.Z, f"Z{f.Z}")
            entry["id"] = f.target
            entries.append(entry)
        return dict(root=self.root, name=self.name, format=self.format, isotopes=entries,
                    temperatures=self.temperatures(),
                    unreadable=len(self.unreadable), duplicates=len(self.duplicates))


def target_name(Z: int, A: Optional[int], M: int = 0) -> str:
    """``Fe56``, ``Am242m1``, ``Cnat``: the G4NDL spelling of a nuclide."""
    from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

    symbol = ATOMIC_NUMBER_TO_SYMBOL.get(Z, f"Z{Z}")
    if not A:
        return f"{symbol}nat"
    return f"{symbol}{A}" + (f"m{M}" if M else "")


def _head(path: Path) -> str:
    with path.open("rb") as handle:
        return handle.read(_HEAD_BYTES).decode("ascii", errors="replace")


def _candidates(root: Path, recursive: bool = False) -> List[Path]:
    """The files of *root*, sorted; with *recursive*, of every subdirectory too.

    Hidden files and directories (``.git``, ``.DS_Store``) are skipped.
    """
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Neither a file nor a directory: {root}")
    if not recursive:
        return sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith("."))
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def _dedupe(files: List[IndexedFile], key) -> Tuple[List[IndexedFile], List[str]]:
    seen: Dict = {}
    duplicates: List[str] = []
    for f in files:
        k = key(f)
        if k in seen:
            duplicates.append(f.path)
        else:
            seen[k] = f
    return list(seen.values()), duplicates


# ------------------------------------------------------------------- ENDF

def _endf_float(field: str) -> float:
    s = field.strip().replace("D", "E").replace("d", "e")
    if not s:
        return 0.0
    # 2.605600+4 -> 2.605600E+4 (an exponent sign not preceded by E).
    s = re.sub(r"(?<=[0-9.])([+-])(\d+)$", r"E\1\2", s)
    return float(s)


def _endf_int(field: str) -> int:
    s = field.strip()
    return int(s) if s else 0


def _endf_header(text: str):
    """``(ZA, AWR, MAT, LISO, NSUB, TEMP)`` from the first records of MF1/MT451."""
    records = []
    for line in text.splitlines():
        if len(line) < 75:
            continue
        try:
            mf, mt = int(line[70:72]), int(line[72:75])
        except ValueError:
            continue
        if mf == 1 and mt == 451:
            records.append(line)
            if len(records) == 4:
                break
        elif records:
            break
    if len(records) < 3:
        return None
    head = records[0]
    za = int(round(_endf_float(head[0:11])))
    awr = _endf_float(head[11:22])
    mat = _endf_int(head[66:70])
    liso = _endf_int(records[1][33:44])
    nsub = _endf_int(records[2][44:55])
    temp = _endf_float(records[3][0:11]) if len(records) > 3 else None
    return za, awr, mat, liso, nsub, temp


def index_endf(root: Union[str, os.PathLike], recursive: bool = False) -> LibraryIndex:
    """Index a directory of ENDF tapes (or one tape) from their MF1/MT451 headers.

    With *recursive*, the subdirectories too: JEFF and JENDL are sometimes
    unpacked one directory per element or sublibrary.
    """
    root = Path(root)
    files: List[IndexedFile] = []
    unreadable: List[str] = []
    for path in _candidates(root, recursive):
        try:
            header = _endf_header(_head(path))
        except (OSError, ValueError):
            header = None
        if header is None:
            unreadable.append(str(path))
            continue
        za, awr, mat, liso, nsub, temp = header
        Z, A = divmod(za, 1000)
        files.append(IndexedFile(
            target=target_name(Z, A or None, liso), Z=Z, A=A or None, M=liso,
            path=str(path), file_name=path.name, size=path.stat().st_size,
            awr=awr, temperature=temp, mat=mat, nsub=nsub))
    files, duplicates = _dedupe(files, lambda f: (f.target, f.nsub, f.temperature))
    return LibraryIndex(str(root), "endf", files, unreadable, duplicates)


# -------------------------------------------------------------------- ACE

_ZAID_RE = re.compile(r"^(\d+)\.(\d+[a-z]+)$")


def _ace_header(text: str):
    """``(zaid, AWR, kT MeV)`` from the first line(s) of a legacy or 2.0 table."""
    lines = text.splitlines()
    if not lines:
        return None
    tokens = lines[0].split()
    if tokens and tokens[0].startswith("2.0") and len(tokens) > 1 and len(lines) > 1:
        zaid = tokens[1]
        second = lines[1].split()
        awr, kt = float(second[0]), float(second[1])
    elif len(tokens) >= 3:
        zaid, awr, kt = tokens[0], float(tokens[1]), float(tokens[2])
    else:
        return None
    return zaid, awr, kt


def _zaid_nuclide(zaid: str, awr: float) -> Optional[Tuple[int, Optional[int], int]]:
    m = _ZAID_RE.match(zaid)
    if not m:
        return None
    Z, raw = divmod(int(m.group(1)), 1000)
    if raw == 0:
        return Z, None, 0
    A = int(round(awr * _NEUTRON_MASS_AMU))
    if raw == A:
        return Z, A, 0
    # JEFF-4.0 (NEA ACE): an isomer is 300 + (A mod 100), the hundreds of A
    # dropped -- Co-58m 27358, Ag-106m 47306, Am-242m 95342. One state only.
    if raw == 300 + A % 100:
        return Z, A, 1
    extra = raw - A
    if extra <= 0 or extra % 100:
        return Z, raw, 0          # not an encoding we know: keep the ZAID's A
    # LANL (MCNP): A + 300 + 100 m -- Am-242m1 95642. Otherwise A + 100 m.
    isomer = (extra - 300) // 100 if extra > 300 else extra // 100
    return Z, A, int(isomer)


def index_ace(root: Union[str, os.PathLike], recursive: bool = False) -> LibraryIndex:
    """Index a directory of ACE tables (or one table) from their first line.

    Only continuous-energy neutron tables (``c``/``nc`` suffix) are indexed;
    thermal (``t``), photon and dosimetry tables are reported as unreadable.
    *recursive* walks the subdirectories too.
    """
    root = Path(root)
    files: List[IndexedFile] = []
    unreadable: List[str] = []
    for path in _candidates(root, recursive):
        if path.name.lower() in ("xsdir", "xsdir.txt") or path.suffix.lower() in (".pdf", ".md", ".txt", ".json"):
            continue
        try:
            header = _ace_header(_head(path))
            nuclide = _zaid_nuclide(header[0], header[1]) if header else None
        except (OSError, ValueError, IndexError):
            header, nuclide = None, None
        if header is None or nuclide is None or not header[0].endswith("c"):
            unreadable.append(str(path))
            continue
        zaid, awr, kt = header
        Z, A, M = nuclide
        files.append(IndexedFile(
            target=target_name(Z, A, M), Z=Z, A=A, M=M,
            path=str(path), file_name=path.name, size=path.stat().st_size,
            awr=awr, temperature=round(kt * _K_PER_MEV, 1), zaid=zaid))
    files, duplicates = _dedupe(files, lambda f: (f.target, f.temperature))
    return LibraryIndex(str(root), "ace", files, unreadable, duplicates)

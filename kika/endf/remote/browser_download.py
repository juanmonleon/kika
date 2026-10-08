"""Files the IAEA serves to a web browser but not to kika.

Since October 2026 ``nds.iaea.org`` answers every automated client with a
Cloudflare challenge (:class:`~kika.endf.remote.exceptions.AccessBlockedError`)
while a browser still gets the file. This module closes that loop: a file the
user fetched in a browser is found where the browser put it and stored in the
cache exactly as :func:`~kika.endf.remote.download_entry` would have stored it,
so nothing downstream can tell the two routes apart.

Examples
--------
>>> from kika.endf.remote import get_catalog
>>> from kika.endf.remote.browser_download import find_browser_download, import_entry
>>> entry = get_catalog().find("jeff4.0", "Fe56")
>>> # ...the user opens entry.url in a browser...
>>> path = find_browser_download(entry)
>>> if path is not None:
...     cached = import_entry(entry, path)
"""

from __future__ import annotations

import io
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .cache import get_cache

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import CatalogEntry


def default_download_dirs() -> list[Path]:
    """Where a browser saves files on this machine: the user's Downloads folder.

    On Windows the folder is asked of the shell, because it is often moved
    (OneDrive, another drive); elsewhere ``XDG_DOWNLOAD_DIR`` is honoured.
    ``~/Downloads`` is always the last resort.
    """
    dirs: list[Path] = []
    if sys.platform == "win32":
        known = _windows_downloads_folder()
        if known is not None:
            dirs.append(known)
    else:
        xdg = _xdg_download_dir()
        if xdg is not None:
            dirs.append(xdg)
    fallback = Path.home() / "Downloads"
    if fallback not in dirs:
        dirs.append(fallback)
    return [d for d in dirs if d.is_dir()]


def find_browser_download(
    entry: "CatalogEntry",
    directories: Iterable[str | Path] | None = None,
) -> Path | None:
    """The newest complete copy of *entry* in the download folders, or None.

    Matches the server's filename and the copies a browser makes when the name
    is taken (``n_026-Fe-56_2631 (1).zip``, ``n_026-Fe-56_2631(1).zip``). An
    archive still being written does not open as a zip and is skipped, so a
    caller polling while the browser works never imports half a file.
    """
    dirs = [Path(d) for d in directories] if directories else default_download_dirs()
    stem, suffix = os.path.splitext(entry.filename)
    pattern = re.compile(
        rf"^{re.escape(stem)}(?: ?\(\d+\))?{re.escape(suffix)}$", re.IGNORECASE
    )
    found: list[Path] = []
    for d in dirs:
        try:
            found.extend(p for p in d.iterdir() if pattern.match(p.name) and p.is_file())
        except OSError:
            continue
    for path in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_archive or zipfile.is_zipfile(path):
            return path
    return None


def import_entry(entry: "CatalogEntry", source: str | Path) -> Path:
    """Store a file fetched outside kika in the cache as *entry*; return the cached path.

    *source* may be the archive the server serves or the ENDF file inside it.
    The content is checked before it is stored: a saved HTML page, or a file
    for another material, raises ``ValueError`` and leaves the cache alone.
    """
    source = Path(source)
    raw = source.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        content = extract_endf_from_zip(raw)
    else:
        content = raw
    check_endf_content(content, expected_mat=entry.mat, name=source.name)
    return get_cache().put(entry.cache_key, entry.library, content, entry.sublib)


def extract_endf_from_zip(zip_content: bytes) -> bytes:
    """The ENDF file inside an IAEA archive: its one member that is not itself an archive."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if not name.endswith((".zip", ".gz", ".tar")):
                return zf.read(name)
    raise ValueError("No ENDF file found in ZIP archive")


# Columns 67-75 of an ENDF-6 record: MAT (4), MF (2), MT (3).
_CONTROL = re.compile(r"^.{66}([ \d-]{4})([ \d]{2})([ \d]{3})")


def check_endf_content(content: bytes, expected_mat: int | None = None, name: str = "file") -> None:
    """Raise ``ValueError`` unless *content* reads as an ENDF-6 tape for *expected_mat*.

    Only the head is read. The MAT is taken from the first record after the
    tape identification, which is where every IAEA file starts its material.
    """
    head = content[:4096].decode("latin-1", errors="replace").splitlines()
    if head and head[0].lstrip().lower().startswith(("<!doctype", "<html")):
        raise ValueError(f"{name} is a web page, not an ENDF file")
    records = [m for m in (_CONTROL.match(line) for line in head[:6]) if m]
    if len(records) < 2:
        raise ValueError(f"{name} does not read as an ENDF-6 file")
    if expected_mat is None:
        return
    try:
        mat = int(records[1].group(1))
    except ValueError:
        raise ValueError(f"{name} does not read as an ENDF-6 file") from None
    if mat != expected_mat:
        raise ValueError(f"{name} holds MAT {mat}, not MAT {expected_mat}")


def _windows_downloads_folder() -> Path | None:
    try:
        import ctypes
        from ctypes import wintypes
        from uuid import UUID

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        # FOLDERID_Downloads
        guid = UUID("374DE290-123F-4565-9164-39C4925E467B")
        folder = _GUID(guid.fields[0], guid.fields[1], guid.fields[2], (ctypes.c_ubyte * 8)(*guid.bytes[8:]))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)
    except Exception:
        return None


def _xdg_download_dir() -> Path | None:
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "user-dirs.dirs"
    try:
        text = config.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^XDG_DOWNLOAD_DIR="(.+)"', text, re.MULTILINE)
    if not match:
        return None
    return Path(match.group(1).replace("$HOME", str(Path.home())))

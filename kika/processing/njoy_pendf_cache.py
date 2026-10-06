"""
Cache-backed NJOY PENDF generation for MF33 NC LTY=0 resolution.

Some libraries (e.g. JENDL Fe-56 MT=2) store MF33 covariance with NC LTY=0
sub-subsections (sum-rule across other MTs). Resolving them against
*relative* contributing covariances requires reconstructed pointwise σ(E),
i.e. a PENDF — the smooth-only MF3 in the original ENDF file is not enough.

This module provides:

  - ``mf33_needs_pendf``        — sentinel: True iff a section's MF33 needs σ(E).
  - ``get_or_create_pendf``     — returns the cached tape22 path, running NJOY
                                  only on cache miss. SHA256(ENDF) keyed.
  - ``read_pendf_mf3_sections`` — parse the cached tape22 into MT → MF3MT.
  - ``find_njoy_executable``    — the NJOY binary to run, or an error that says
                                  how to point kika at one.
  - ``attach_pendf``            — the above in one step for a parsed tape: sets
                                  ``endf.pendf`` from a cached RECONR run.

The default cache directory resolves via ``tempfile.gettempdir()`` so the
module works on any machine without machine-specific paths baked in.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

# LB types in MF33 NI sub-subsections that store *relative* covariances.
# Anything outside this set is absolute and needs σ(E) for diagonal use.
# LB=8 is NOT one of them: ENDF-6 §33.2 gives its F_k "the dimension of
# squared cross sections". It sat in this set until 2026-10-06, and LB=1/2
# (relative) were missing from it.
_RELATIVE_LB_TYPES = frozenset({1, 2, 3, 4, 5, 6})

#: Environment variable read by :func:`find_njoy_executable`.
NJOY_ENV_VAR = "NJOY_EXECUTABLE"


class NjoyNotFoundError(RuntimeError):
    """σ(E) has to be reconstructed with NJOY and no NJOY executable was found."""


# Default cache directory — portable across Linux / macOS / Windows.
# Linux:   /tmp/kika_pendf_cache
# macOS:   /var/folders/.../T/kika_pendf_cache
# Windows: C:\Users\<user>\AppData\Local\Temp\kika_pendf_cache
DEFAULT_PENDF_CACHE_DIR = Path(tempfile.gettempdir()) / "kika_pendf_cache"


def mf33_needs_pendf(mf33_file, mt: int) -> bool:
    """True if MF33 for ``mt`` needs σ(E) (NC sum-rule or absolute NI LB)."""
    if mf33_file is None or mt not in mf33_file.sections:
        return False
    sec = mf33_file.sections[mt]
    for sub in sec.subsections:
        if getattr(sub, "nc", False) and (sub.nc_records or []):
            return True
        for ni in (sub.ni_records or []):
            lb = getattr(ni, "lb", None) or getattr(ni, "LB", None)
            if lb is None or lb not in _RELATIVE_LB_TYPES:
                return True
    return False


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA256 over the raw bytes — any byte edit invalidates the cache."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _pendf_cache_path(endf_path: Path, tolerance: float, cache_dir: Path) -> Path:
    """``cache_dir / {sha256}_{tolerance}.pendf`` — SHA256 over the ENDF bytes."""
    return cache_dir / f"{_sha256_of_file(endf_path)}_{tolerance}.pendf"


def get_or_create_pendf(
    endf_path: str | Path,
    *,
    tolerance: float,
    njoy_exe: str | Path,
    cache_dir: str | Path | None = None,
    timeout_s: float = 600.0,
    keep_njoy_io_dir: str | Path | None = None,
) -> Path:
    """Return a path to a cached PENDF (tape22) for ``endf_path``.

    On cache miss runs ``njoy_reconstruct_stream`` and atomically copies the
    PENDF bytes into the cache before the generator's TempDir is cleaned up.

    ``cache_dir=None`` uses ``DEFAULT_PENDF_CACHE_DIR`` (portable tempdir).

    If ``keep_njoy_io_dir`` is set, the RECONR input deck and stdout are
    copied there as ``<base>_recon.input`` / ``<base>_recon.output``. On
    cache hit, NJOY didn't run, so nothing is copied (look in
    ``cache_dir`` for the original I/O if needed). The same files are
    always saved next to the cached PENDF as ``<sha>_<tol>.input`` /
    ``.output`` so they are recoverable even after a hit.
    """
    endf_path = Path(endf_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_PENDF_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _pendf_cache_path(endf_path, tolerance, cache_dir)

    if target.is_file() and target.stat().st_size > 0:
        print(f"  PENDF cache hit: {target.name}")
        if keep_njoy_io_dir is not None:
            _copy_cached_recon_io(
                target, Path(keep_njoy_io_dir).expanduser(), endf_path,
            )
        return target

    print(f"  PENDF cache miss for {endf_path.name} — running NJOY reconr "
          f"(tolerance={tolerance})")

    # Local import — keeps module light when caller doesn't need NJOY.
    from kika.processing.njoy_reconstruct import njoy_reconstruct_stream

    src_pendf: Optional[Path] = None
    for kind, payload in njoy_reconstruct_stream(
        str(endf_path), str(njoy_exe),
        tolerance=tolerance, timeout_s=timeout_s,
    ):
        if kind == "pendf_path":
            # Copy synchronously while the generator's TempDir is still alive.
            src = Path(str(payload))
            tmp = target.with_suffix(".pendf.tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(target)
            src_pendf = target
            # Persist the deck + stdout next to the cached PENDF and, if
            # requested, also into the per-run keep_njoy_io_dir.
            workdir = src.parent
            _save_recon_io(workdir, target)
            if keep_njoy_io_dir is not None:
                _copy_cached_recon_io(
                    target, Path(keep_njoy_io_dir).expanduser(), endf_path,
                )
        elif kind == "warning":
            print(f"  NJOY: {payload}")

    if src_pendf is None or not target.is_file():
        raise RuntimeError(
            f"NJOY reconr did not emit a PENDF for {endf_path.name}"
        )
    print(f"  PENDF cached: {target.name} "
          f"({target.stat().st_size / 1024:.0f} KB)")
    return target


def _save_recon_io(workdir: Path, cached_pendf: Path) -> None:
    """Copy ``njoy.inp`` / ``njoy.out`` from the RECONR workdir next to the cache."""
    inp_src = workdir / "njoy.inp"
    out_src = workdir / "njoy.out"
    inp_dst = cached_pendf.with_suffix(".input")
    out_dst = cached_pendf.with_suffix(".output")
    try:
        if inp_src.is_file():
            shutil.copyfile(inp_src, inp_dst)
        if out_src.is_file():
            shutil.copyfile(out_src, out_dst)
    except OSError:
        pass


def _copy_cached_recon_io(
    cached_pendf: Path, dst_dir: Path, endf_path: Path,
) -> None:
    """Copy the cache's stored RECONR I/O into ``dst_dir`` for visibility."""
    inp_src = cached_pendf.with_suffix(".input")
    out_src = cached_pendf.with_suffix(".output")
    if not (inp_src.is_file() or out_src.is_file()):
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = endf_path.stem
    try:
        if inp_src.is_file():
            shutil.copyfile(inp_src, dst_dir / f"{base}_recon.input")
        if out_src.is_file():
            shutil.copyfile(out_src, dst_dir / f"{base}_recon.output")
    except OSError:
        pass


def read_pendf_mf3_sections(pendf_path: str | Path) -> Dict[int, object]:
    """Parse a cached tape22 and return the MT → MF3MT map."""
    from kika.endf import read_endf
    pendf = read_endf(str(pendf_path), mf_numbers=[3])
    mf3 = pendf.get_file(3)
    if mf3 is None or not getattr(mf3, "sections", None):
        raise RuntimeError(f"PENDF {pendf_path} has no MF3 sections")
    return {int(mt): sec for mt, sec in mf3.sections.items()}


def find_njoy_executable(njoy_executable: str | Path | None = None, *,
                         why: str = "this operation") -> Path:
    """The NJOY binary to run: the argument, else ``$NJOY_EXECUTABLE``, else ``njoy`` on PATH.

    kika never bundles NJOY, so when none of the three gives an existing file
    this raises :class:`NjoyNotFoundError` with the three ways to fix it —
    the message is the documentation a user hits at the moment they need it.
    ``why`` says what the reconstruction is for, and goes into the message.
    """
    tried = []
    if njoy_executable is not None:
        path = Path(njoy_executable).expanduser()
        if path.is_file():
            return path
        found = shutil.which(str(njoy_executable))
        if found:
            return Path(found)
        tried.append(f"njoy_executable={str(njoy_executable)!r} (no such file)")
    env = os.environ.get(NJOY_ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path
        tried.append(f"{NJOY_ENV_VAR}={env!r} (no such file)")
    found = shutil.which("njoy")
    if found:
        return Path(found)
    tried.append("`njoy` on PATH (not found)")
    raise NjoyNotFoundError(
        f"{why} needs σ(E) reconstructed by NJOY RECONR, and no NJOY executable "
        f"was found. Tried: {'; '.join(tried)}.\n"
        f"Fix it in one of three ways:\n"
        f"  1. set the environment variable {NJOY_ENV_VAR} to the NJOY binary, e.g.\n"
        f"       Windows:  setx {NJOY_ENV_VAR} C:\\path\\to\\NJOY2016\\build\\njoy.exe\n"
        f"       Linux:    export {NJOY_ENV_VAR}=/path/to/NJOY2016/build/njoy\n"
        f"     (on Windows use a statically linked build: a MinGW build linked\n"
        f"     against libgfortran.dll misreads numbers under a comma-decimal locale);\n"
        f"  2. pass njoy_executable=... to the call that raised this;\n"
        f"  3. reconstruct σ(E) yourself and attach it before decoding:\n"
        f"       endf.pendf = kika.processing.njoy_reconstruct(path, njoy_executable=...)"
    )


def attach_pendf(endf, *, endf_path: str | Path | None = None,
                 njoy_executable: str | Path | None = None,
                 tolerance: float = 0.001,
                 cache_dir: str | Path | None = None,
                 why: str = "this operation") -> Dict[int, object]:
    """Make sure ``endf.pendf`` holds reconstructed σ(E), and return it.

    Kept as is when already set — the caller chose the source. Otherwise NJOY
    RECONR runs on the tape the object was read from (``endf.source_path``,
    or ``endf_path``), through the SHA256-keyed cache of
    :func:`get_or_create_pendf`, so a second call on the same tape costs a
    file read.
    """
    if getattr(endf, "pendf", None):
        return endf.pendf
    path = endf_path if endf_path is not None else getattr(endf, "source_path", None)
    if path is None:
        raise NjoyNotFoundError(
            f"{why} needs σ(E) reconstructed by NJOY RECONR, and this ENDF "
            f"object does not know which file it was read from, so there is "
            f"nothing to run NJOY on. Read it with kika.endf.read_endf(path), "
            f"pass endf_path=..., or set endf.pendf yourself."
        )
    exe = find_njoy_executable(njoy_executable, why=why)
    pendf_path = get_or_create_pendf(path, tolerance=tolerance, njoy_exe=exe,
                                     cache_dir=cache_dir)
    endf.pendf = read_pendf_mf3_sections(pendf_path)
    return endf.pendf

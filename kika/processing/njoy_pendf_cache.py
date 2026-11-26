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
  - ``pointwise_cross_sections`` — σ(E) for a parsed tape without NJOY: its own
                                  MF3, or kika's reconstruction when it has resonances.

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


def _readTape(path, mfNumbers):
    from kika.endf import read_endf
    return read_endf(str(path), mf_numbers=list(mfNumbers))


def mf33_needs_pendf(mf33_file, mt: int) -> bool:
    """True if MF33 for ``mt`` needs σ(E) (NC sum-rule or absolute NI LB).

    Parameters
    ----------
    mf33_file
        The MF33 file of a parsed tape (it has a ``sections`` mapping by MT), or
        None.
    mt
        The reaction whose covariance section is inspected.

    Returns
    -------
    bool
        True if any subsection of the MT has NC records, or any NI record has an
        LB that is missing or not one of 1 to 6 (the relative types). False
        otherwise, and False when *mf33_file* is None or has no section for *mt*.
    """
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


def _is_complete_tape(path: Path) -> bool:
    """A tape ends with its TEND record (MAT = -1) and holds no NUL byte at
    its end: what a crash leaves after the rename but before the data reached
    the disk is the right size and filled with NULs."""
    size = path.stat().st_size
    if size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read()
    if b"\0" in tail:
        return False
    lines = tail.rstrip().splitlines()
    return bool(lines) and lines[-1][66:70].strip() == b"-1"


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

    The cache file is named ``<sha256 of the ENDF bytes>_<tolerance>.pendf``, so
    any edit of the tape or any other *tolerance* is a miss. A cached file that is
    truncated is deleted and rebuilt. Progress lines go to standard output.

    Parameters
    ----------
    endf_path
        The ENDF tape to reconstruct.
    tolerance
        RECONR's relative tolerance, as a fraction (part of the cache key).
    njoy_exe
        The NJOY binary, passed as given to
        :func:`~kika.processing.njoy_reconstruct.njoy_reconstruct_stream`; no
        search is made.
    cache_dir
        Directory of the cache, created if missing. None is
        ``DEFAULT_PENDF_CACHE_DIR``.
    timeout_s
        Seconds allowed to one NJOY run.
    keep_njoy_io_dir
        Directory to copy the RECONR input deck and listing to; None copies
        nothing.

    Returns
    -------
    pathlib.Path
        The cached PENDF file (ASCII, as NJOY writes it as ``tape22``).

    Raises
    ------
    RuntimeError
        If NJOY finishes without emitting a PENDF. A failure of NJOY itself
        raises :class:`~kika.processing.njoy_reconstruct.NjoyReconstructError`,
        which is a
        ``RuntimeError`` too.
    """
    endf_path = Path(endf_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_PENDF_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _pendf_cache_path(endf_path, tolerance, cache_dir)

    if target.is_file() and not _is_complete_tape(target):
        print(f"  PENDF cache entry {target.name} is truncated; reprocessing")
        target.unlink()
    if target.is_file():
        print(f"  PENDF cache hit: {target.name}")
        if keep_njoy_io_dir is not None:
            _copy_cached_recon_io(
                target, Path(keep_njoy_io_dir).expanduser(), endf_path,
            )
        return target

    print(f"  PENDF cache miss for {endf_path.name} - running NJOY reconr "
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
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
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
    """Parse a cached tape22 and return the MT → MF3MT map.

    Parameters
    ----------
    pendf_path
        A PENDF file; only its MF3 is read.

    Returns
    -------
    dict of int to object
        The MF3 sections of the tape (the parser's section objects, not
        ``CrossSection``), keyed by MT.

    Raises
    ------
    RuntimeError
        If the tape has no MF3 sections.
    """
    pendf = _readTape(pendf_path, [3])
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

    Parameters
    ----------
    njoy_executable
        A path to the binary, or a command name looked up on ``PATH``. If it
        names neither, the search goes on to the environment variable, then to
        ``PATH``.
    why
        What the reconstruction is for, quoted in the error message.

    Returns
    -------
    pathlib.Path
        The first of the three that exists. The environment variable is
        ``NJOY_EXECUTABLE`` and must name a file.

    Raises
    ------
    NjoyNotFoundError
        If none of the three finds an NJOY executable.
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

    Parameters
    ----------
    endf
        A parsed tape. Its ``pendf`` attribute is set by this call.
    endf_path
        The tape to run NJOY on. None uses ``endf.source_path``.
    njoy_executable
        As in :func:`find_njoy_executable`.
    tolerance
        RECONR's relative tolerance, as a fraction.
    cache_dir
        As in :func:`get_or_create_pendf`.
    why
        What the reconstruction is for, quoted in the error messages.

    Returns
    -------
    dict of int to object
        ``endf.pendf``: the MF3 sections of the PENDF keyed by MT.

    Raises
    ------
    NjoyNotFoundError
        If ``endf.pendf`` is not set and no tape path is known (neither
        *endf_path* nor ``endf.source_path``), or no NJOY executable is found.
    RuntimeError
        From :func:`get_or_create_pendf` and :func:`read_pendf_mf3_sections`
        (including NJOY failures).
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


def pointwise_cross_sections(endf, *, tolerance: float = 0.001,
                             cache_dir: str | Path | None = None,
                             why: str = "this operation") -> tuple[Dict[int, object], str]:
    """σ(E) of every MT of a parsed tape, without NJOY, and where it came from.

    ``endf.pendf`` is used as it is when set: that is how a caller chooses NJOY
    (:func:`~kika.processing.njoy_pendf_cache.attach_pendf`) or any other source. Otherwise a tape whose MF3 is
    complete (LRP≠1, or no resolved or unresolved range in MF2) gives its own
    MF3. A tape with resonance ranges is reconstructed by kika at *tolerance*,
    through the cache of ``kika.sampling.resonance_region.reconstructPendf``,
    and the result is set as ``endf.pendf``. MF1, MF2 or MF3 the object was read
    without are read from ``endf.source_path``.

    Parameters
    ----------
    endf
        A parsed tape.
    tolerance
        Relative tolerance of the reconstruction, as a fraction.
    cache_dir
        Directory of the reconstruction cache; None is ``DEFAULT_PENDF_CACHE_DIR``.
    why
        What σ(E) is for, quoted in the error messages.

    Returns
    -------
    tuple of (dict of int to object, str)
        The MF3-like sections keyed by MT, and a phrase naming their source:
        ``endf.pendf``, the tape's MF3, or kika's reconstruction.

    Raises
    ------
    ValueError
        If something has to be read from the tape or reconstructed and
        ``endf.source_path`` is not set, or the tape lacks the MF1/451 or MF2
        that say whether its MF3 is complete.
    ~kika.processing.resonances.UnsupportedResonanceError
        If the tape uses resonance data kika does not reconstruct; and
        ``ReconstructionConvergenceError`` if a budget runs out or a check fails.
        Setting ``endf.pendf`` first (for instance with
        :func:`~kika.processing.njoy_pendf_cache.attach_pendf`)
        avoids the reconstruction.
    """
    if getattr(endf, "pendf", None):
        return endf.pendf, "`endf.pendf`"
    from kika.sampling.resonance_region import _lrp, needsReconstruction, reconstructPendf

    path = getattr(endf, "source_path", None)
    mf = getattr(endf, "mf", None) or {}

    def fromTape(mfNumbers):
        if path is None:
            raise ValueError(
                f"{why} needs σ(E), and this ENDF object was read without MF"
                f"{'/'.join(map(str, mfNumbers))} and does not know which file it was "
                f"read from. Read it with kika.endf.read_endf(path), or set endf.pendf.")
        return _readTape(path, mfNumbers)

    header = endf
    if _lrp(endf) is None or (_lrp(endf) == 1 and 2 not in mf):
        header = fromTape((1, 2))
    lrp = _lrp(header)
    if lrp is None or (lrp == 1 and 2 not in (getattr(header, "mf", None) or {})):
        raise ValueError(
            f"{why} needs σ(E), and the tape {'has no MF1/451' if lrp is None else 'states LRP=1 but has no MF2'}, "
            f"so whether its MF3 is the whole cross section cannot be told. Set endf.pendf.")
    if not needsReconstruction(header):
        mf3 = mf.get(3) if 3 in mf else fromTape((3,)).mf.get(3)
        return ({int(mt): section for mt, section in getattr(mf3, "mt", {}).items()},
                "the tape's MF3 (no resonance ranges)")
    if path is None:
        raise ValueError(
            f"{why} needs σ(E) reconstructed from the resonance parameters, and this "
            f"ENDF object does not know which file it was read from. Read it with "
            f"kika.endf.read_endf(path), or set endf.pendf.")
    try:
        pendf = reconstructPendf(path, engine="kika", tolerance=tolerance, cacheDir=cache_dir)
    except Exception as failure:
        failure.add_note(f"{why}: kika could not reconstruct σ(E). Set endf.pendf first, for "
                         f"instance with kika.processing.attach_pendf(endf, njoy_executable=...).")
        raise
    endf.pendf = read_pendf_mf3_sections(pendf)
    return endf.pendf, f"kika's resonance reconstruction (tolerance {tolerance:g})"


def pointwise_cross_sections_from_suite(suite, *, tolerance: float = 0.001
                                        ) -> tuple[Dict[int, object], str]:
    """σ(E) of every MT of a model ``ReactionSuite``, with its interpolation laws, and where it came from.

    The model's counterpart of :func:`pointwise_cross_sections`, for a suite read
    from GNDS: a suite with no resolved or unresolved range (at most a scattering
    radius, like MF2 LRU=0) gives its ``eval`` tables, and one with them is
    reconstructed by kika at *tolerance*
    (:func:`~kika.processing.resonances.suite.reconstruct_suite` on a copy), the
    reconstruction the tape path uses. A ``crossSectionReconstructed`` style the
    file already states is another code's processing and is not read. A reaction
    whose form is not a table is left out. Each table is ``energies``/``values``/
    ``energy_interpolation`` (ENDF ``(NBT, INT)`` pairs), as an MF3 section reads.

    Parameters
    ----------
    suite
        A model ``ReactionSuite``, read from any format; it is not modified.
    tolerance
        Relative tolerance of the reconstruction, as a fraction.

    Returns
    -------
    tuple of (dict of int to object, str)
        The tables keyed by MT (eV and b), and a phrase naming their source.

    Raises
    ------
    ~kika.processing.resonances.UnsupportedResonanceError
        If the suite uses resonance data kika does not reconstruct; and
        ``ReconstructionConvergenceError`` if a budget runs out or a check fails.
    """
    import copy
    from types import SimpleNamespace

    import numpy as np

    label, source = "eval", "the suite's evaluated cross sections (no resonance ranges)"
    resonances = getattr(suite, "resonances", None)
    if resonances is not None and (resonances.resolved or resonances.unresolved is not None):
        from kika.processing.resonances import ReconstructionOptions
        from kika.processing.resonances.suite import attach_reconstruction, reconstruct_suite

        taken = set(suite.styles.labels)
        label = next(c for c in ("recon", "kika-recon", *(f"kika-recon-{i}" for i in range(1, 100)))
                     if c not in taken)
        result = reconstruct_suite(suite, label=label, options=ReconstructionOptions(rtol=tolerance))
        suite = copy.deepcopy(suite)
        attach_reconstruction(suite, result)
        label, source = result.label, f"kika's resonance reconstruction (tolerance {tolerance:g})"

    tables: Dict[int, object] = {}
    for container in (suite.reactions, suite.sums, suite.fissionComponents):
        for reaction in container:
            mt = getattr(reaction, "ENDF_MT", None)
            if mt is None or int(mt) in tables:
                continue
            for form in dict.fromkeys((label, "eval")):
                try:
                    suite.cross_section(int(mt), form)
                except (KeyError, TypeError, ValueError):
                    continue
                energies, values, laws = reaction.crossSection[form].toEndfRegions()
                tables[int(mt)] = SimpleNamespace(energies=np.asarray(energies, dtype=float),
                                                  values=np.asarray(values, dtype=float),
                                                  energy_interpolation=list(laws))
                break
    return tables, source

"""Replace one isotope's data in a copy of a whole G4NDL library.

Phase 6 of the G4NDL roadmap, for the case it was asked for: an iterative loop
(insert measured σ or a measured p(μ), run Geant4, compare, repeat) needs a
complete library ``G4NEUTRONHPDATA`` can point at, in which exactly one
isotope's files differ from a reference library. Phase 10 extends it from the
elastic to the inelastic channels, and the capture work extends it to MT102.

:func:`patch_isotope` is "copy the base library, then
:func:`~kika.g4ndl.encode.writeSuite` one isotope into the copy", for the
processes asked (``elastic``, ``inelastic`` or both; default: what the suite
holds). :func:`patch_elastic` is the elastic-only case. Both come with the
guarantees the roadmap asks for:

* **The base library is never modified.** Everything is written into a
  temporary sibling of ``output_library``, verified, and only then renamed
  into place. With ``share="hardlink"`` the unchanged files are hard links to
  the base. The isotope's own files are never copied or linked, only written
  fresh: the distributed libraries are read-only, Windows will not replace a
  read-only file, and the read-only bit of a hard link is the base file's.
* **Only that isotope's files of those processes change**: two for the
  elastic; for the inelastic ``Inelastic/CrossSection`` and one file per
  channel the suite has, and the isotope's file in any other ``Fxx`` is
  *removed* (Geant4 would otherwise read a channel the suite does not have);
  for the capture ``Capture/CrossSection`` and the one final state the suite
  has, ``FSMF6`` or ``FS``, with the other removed for the same reason.
  Its other variant (``.z`` or plain) is left out so it cannot shadow the new
  file; every other file is the base's, which the verification checks by
  relative path and size. Level schemes of residual nuclei
  (``Inelastic/Gammas``) are not the isotope's: they change only when passed
  as ``gammas=``.
* **What was done is written down** in ``kika_manifest.json`` at the library
  root, outside every numeric stream: the base, the files replaced or added
  and their SHA-256 before and after, the labels and mass written, the
  conversion report. Geant4 opens files by constructed name only, so the
  manifest is invisible to it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from kika.g4ndl.exceptions import IsotopeNotFoundError
from kika.g4ndl.library import CHANNELS, GAMMAS_DIR, PROCESSES, G4NDLLibrary

__all__ = ["MANIFEST_NAME", "PatchResult", "patch_elastic", "patch_isotope"]

MANIFEST_NAME = "kika_manifest.json"
_SHARE_MODES = ("copy", "hardlink")


@dataclass
class PatchResult:
    """What :func:`patch_isotope` (or :func:`patch_elastic`) produced.

    ``replaced`` maps each written file (relative to the library root) to the
    base file it replaces, or ``None`` when the base had no file for the
    isotope; ``removed`` lists the twins deleted from the copy.
    """

    output: Path
    target: str
    replaced: Dict[str, Optional[str]] = field(default_factory=dict)
    removed: List[str] = field(default_factory=list)
    manifest: Optional[Path] = None
    report: object = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(a: Path, b: Path) -> bool:
    try:
        a.relative_to(b)
        return True
    except ValueError:
        return False


def _fileSet(root: Path) -> Dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


def _removeTree(root: Path, base: Path) -> None:
    """``rmtree`` that copes with read-only files without touching ``base``.

    Windows will not delete a read-only file, and its read-only bit lives on
    the file, not on the link: clearing it on a hard link clears it on the
    base library's file too. So the bit is cleared to delete the link and,
    when the link was one of the base's files, put back on the base file.
    """
    def handler(func, path, _exc):
        p = Path(path)
        mode = os.stat(p).st_mode
        mirror = base / p.relative_to(root)
        linked = mirror.is_file() and os.path.samefile(p, mirror)
        os.chmod(p, mode | stat.S_IWRITE)
        func(path)
        if linked:
            os.chmod(mirror, mode)

    if sys.version_info >= (3, 12):
        shutil.rmtree(root, onexc=handler)
    else:  # pragma: no cover
        shutil.rmtree(root, onerror=handler)


def patch_elastic(base_library, suite, output_library, *, compressed: Optional[bool] = None,
                  share: str = "copy", overwrite: bool = False,
                  crossSectionLabel: Optional[str] = None, angularLabel: str = "eval",
                  targetMass: Optional[float] = None, header="keep") -> PatchResult:
    """:func:`patch_isotope` with ``processes=("elastic",)``: only MT2 is replaced."""
    return patch_isotope(base_library, suite, output_library, processes=("elastic",),
                         compressed=compressed, share=share, overwrite=overwrite,
                         crossSectionLabel=crossSectionLabel, angularLabel=angularLabel,
                         targetMass=targetMass, header=header)


def _subdirsOf(processes) -> List[str]:
    out = []
    for p in processes:
        out.extend(PROCESSES[p])
        out.extend(CHANNELS.get(p, ()))
    return out


def patch_isotope(base_library, suite, output_library, *,
                  processes: Optional[Sequence[str]] = None, gammas=(),
                  compressed: Optional[bool] = None, share: str = "copy",
                  overwrite: bool = False, crossSectionLabel: Optional[str] = None,
                  angularLabel: str = "eval", targetMass: Optional[float] = None,
                  header="keep") -> PatchResult:
    """Copy ``base_library`` to ``output_library`` with ``suite``'s data in it.

    Parameters
    ----------
    base_library
        The G4NDL root to start from (what ``G4NEUTRONHPDATA`` points at).
        Read only.
    suite
        The isotope to write; its target names the files.
    output_library
        Where the patched library goes. Must not exist, unless ``overwrite``
        and it is a library this function wrote (it has a ``kika_manifest.json``):
        kika never deletes a directory it did not make.
    processes
        Any of ``"elastic"``, ``"inelastic"``, ``"capture"``. Default: what the suite
        holds (:func:`kika.g4ndl.encode.suiteProcesses`). The isotope's files
        of the other processes are the base's, untouched.
    gammas
        :class:`~kika.g4ndl.inelastic_records.GammasRecord` level schemes to
        write into ``Inelastic/Gammas`` in place of the base's (plain text, as
        Geant4 reads them).
    compressed
        Write ``.z`` (zlib) or plain text. Default: whatever the base has for
        that isotope, ``.z`` when the base has neither.
    share
        ``"copy"`` (default) copies every file; ``"hardlink"`` links the
        unchanged ones to the base, which takes no space and seconds instead
        of a 1 GB copy per iteration. Hard links need both trees on one
        volume. Editing the output by hand afterwards would then edit the
        base too; kika itself never does.
    overwrite, crossSectionLabel, angularLabel, targetMass, header
        ``overwrite`` replaces a previous output of this function; the rest go
        to :func:`kika.g4ndl.encode.writeSuite`. The inelastic sums follow the
        suite's partials there: edit MT51 and MT4 and the total are rebuilt.

    Returns
    -------
    PatchResult
        With the :class:`~kika.nuclear_data.model.conversion.ConversionReport`
        in ``.report``. Read it.
    """
    from kika.g4ndl.encode import (encodeElastic, recordDifferences, suiteProcesses,
                                   targetKey, writeSuite)
    from kika.g4ndl.capture import CaptureMF6Record, encodeCapture, finalStateDifferences
    from kika.g4ndl.inelastic_encode import encodeInelastic
    from kika.g4ndl.inelastic_format import formatGammas, inelasticDifferences
    from kika.g4ndl.names import file_name

    if share not in _SHARE_MODES:
        raise ValueError(f"share must be one of {_SHARE_MODES}, got {share!r}")
    wanted = suiteProcesses(suite) if processes is None else list(processes)
    if not wanted or set(wanted) - set(PROCESSES):
        raise ValueError(f"processes must be a non-empty subset of {tuple(PROCESSES)}, "
                         f"got {wanted!r}")
    base = Path(base_library).resolve()
    output = Path(output_library).resolve()
    baseLib = G4NDLLibrary(base)  # refuses a directory that is not a library
    if output == base or _inside(output, base) or _inside(base, output):
        raise ValueError(f"output {output} and base {base} must be separate trees: "
                         f"the base library is never written to")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True to replace a "
                                  f"library patch_isotope wrote before")
        if not (output / MANIFEST_NAME).is_file():
            raise FileExistsError(f"{output} exists and has no {MANIFEST_NAME}: kika "
                                  f"only replaces a library it wrote itself")

    # Encode and check before copying a gigabyte: a refusal costs nothing.
    encoded = {}
    if "elastic" in wanted:
        cs, fs, _ = encodeElastic(suite, crossSectionLabel=crossSectionLabel,
                                  angularLabel=angularLabel, targetMass=targetMass,
                                  header=header)
        encoded["Elastic/CrossSection"] = cs
        encoded["Elastic/FS"] = fs
    if "inelastic" in wanted:
        total, files, _ = encodeInelastic(suite, header=header, targetMass=targetMass)
        if total is not None:
            encoded["Inelastic/CrossSection"] = total
        for ch, record in files.items():
            encoded[f"Inelastic/{ch}"] = record
    if "capture" in wanted:
        cs, fs, _ = encodeCapture(suite, header=header, targetMass=targetMass)
        encoded["Capture/CrossSection"] = cs
        if fs is not None:
            encoded["Capture/FSMF6" if isinstance(fs, CaptureMF6Record)
                    else "Capture/FS"] = fs
    key = targetKey(suite)
    located = {}
    for sub in _subdirsOf(wanted):
        try:
            located[sub] = baseLib.locate(key, sub)
        except IsotopeNotFoundError:
            located[sub] = None
    found = [f for f in located.values() if f is not None]
    if compressed is None:
        compressed = found[0].compressed if found else True
    elementName = found[0].elementName if found else None
    stem = file_name(key)
    if elementName is not None:
        stem = stem.rsplit("_", 1)[0] + "_" + elementName

    # The isotope's files (both variants) are neither copied nor linked: they are
    # written fresh. Overwriting a copy would fail on a read-only base file, and
    # making a hard link writable would make the base writable with it.
    skip = {f"{sub}/{stem}{suffix}" for sub in located for suffix in ("", ".z")}
    skip |= {f"{sub}/{f.path.name}" for sub, f in located.items() if f is not None}
    written = {f"{sub}/{stem}{'.z' if compressed else ''}":
               (located[sub].path.relative_to(base).as_posix() if located[sub] else None)
               for sub in encoded}
    gammaFiles = {}
    for g in gammas or ():
        rel = f"{GAMMAS_DIR}/z{int(g.Z)}.a{int(g.A)}"
        gammaFiles[rel] = g
        skip.add(rel)
        written[rel] = rel if (base / rel).is_file() else None

    def ignore(directory, names):
        rel = Path(directory).relative_to(base).as_posix()
        return {n for n in names if f"{rel}/{n}" in skip}

    tmp = output.parent / f".{output.name}.kika-tmp-{uuid.uuid4().hex[:8]}"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(base, tmp, ignore=ignore,
                        copy_function=os.link if share == "hardlink" else shutil.copy2)
        report = writeSuite(suite, tmp, processes=wanted, compressed=compressed,
                            crossSectionLabel=crossSectionLabel, angularLabel=angularLabel,
                            targetMass=targetMass, header=header, elementName=elementName)
        for rel, g in gammaFiles.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text(formatGammas(g), encoding="ascii")

        # The copy must be the base minus the isotope's files plus the written ones.
        baseFiles = _fileSet(base)
        after = _fileSet(tmp)
        removed = sorted(p for p in skip if p in baseFiles and p not in written)
        expected = (set(baseFiles) - skip) | set(written)
        unexpected = sorted(set(after) ^ expected)
        unexpected += sorted(p for p in set(baseFiles) - skip if after.get(p) != baseFiles[p])
        if unexpected:
            raise RuntimeError(f"patch_isotope changed files outside {key}: {unexpected[:5]}")

        # The written isotope must read back to what was encoded.
        check = G4NDLLibrary(tmp)
        diffs = []
        for sub, record in encoded.items():
            if sub == "Elastic/CrossSection":
                diffs += recordDifferences(record, check.crossSection(key))
            elif sub == "Elastic/FS":
                diffs += recordDifferences(record, check.elasticFinalState(key))
            elif sub == "Inelastic/CrossSection":
                diffs += recordDifferences(record, check.inelasticCrossSection(key))
            elif sub == "Capture/CrossSection":
                diffs += recordDifferences(record, check.captureCrossSection(key))
            elif sub.startswith("Capture/"):
                diffs += finalStateDifferences(record, check.captureFinalState(key))
            else:
                diffs += inelasticDifferences(record, check.inelasticFinalState(
                    key, sub.split("/")[1]))
        for rel, g in gammaFiles.items():
            diffs += inelasticDifferences(g, check.gammas(g.Z, g.A))
        if diffs:
            raise RuntimeError(f"the patched {key} does not read back: {diffs[:5]}")

        fs = encoded.get("Elastic/FS")
        manifest = {
            "tool": "kika.g4ndl.patch_isotope",
            "kika": _kikaVersion(),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "base_library": str(base),
            "target": str(key),
            "processes": wanted,
            "share": share,
            "compressed": bool(compressed),
            "crossSectionLabel": crossSectionLabel or "recon",
            "angularLabel": angularLabel,
            "repFlag": fs.repFlag if fs is not None else None,
            "targetMass": fs.targetMass if fs is not None else targetMass,
            "frameFlag": fs.frameFlag if fs is not None else None,
            "channels": sorted(s.split("/")[1] for s in encoded
                               if s.startswith("Inelastic/F")),
            "source": _sourceOf(suite),
            "files": [
                {"path": new, "sha256": _sha256(tmp / new),
                 "replaces": old,
                 "replaces_sha256": _sha256(base / old) if old else None}
                for new, old in written.items()
            ],
            "removed": removed,
            "report": {"warnings": report.warnings, "losses": report.losses,
                       "approximations": report.approximations,
                       "unsupported": report.unsupported},
        }
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if output.exists():
            trash = output.parent / f".{output.name}.kika-old-{uuid.uuid4().hex[:8]}"
            os.replace(output, trash)
            os.replace(tmp, output)
            _removeTree(trash, base)
        else:
            os.replace(tmp, output)
    except BaseException:
        if tmp.exists():
            _removeTree(tmp, base)
        raise

    return PatchResult(output=output, target=str(key), replaced=written, removed=removed,
                       manifest=output / MANIFEST_NAME, report=report)


def _kikaVersion() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("kika-nd")
    except Exception:
        return None


def _sourceOf(suite) -> dict:
    p = getattr(suite, "provenance", None)
    if p is None:
        return {}
    out = {"format": getattr(p, "sourceFormat", None)}
    for name in ("libraryName", "crossSectionPath", "finalStatePath",
                 "crossSectionSha256", "finalStateSha256", "mat", "awr"):
        v = getattr(p, name, None)
        if v is not None:
            out[name] = v
    return out

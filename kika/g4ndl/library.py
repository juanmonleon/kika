"""A G4NDL library: a directory, indexed by isotope, read one file at a time.

A G4NDL library is not one file. It is the tree ``G4NEUTRONHPDATA`` points at
— ``Elastic/``, ``Capture/``, ``Inelastic/``, ``Fission/`` and more, each with
per-isotope files — and Geant4 opens files in it by building a name from
``Z``, ``A`` and ``M``. :class:`G4NDLLibrary` does the same, with three
deliberate differences, each because the consumer's behaviour hides an error
from the user:

* **Exact lookup.** An isotope the library lacks raises
  :class:`~kika.g4ndl.exceptions.IsotopeNotFoundError`; Geant4 would quietly
  substitute the natural element or a neighbouring ``A``.
* **Duplicates are reported.** When ``<name>`` and ``<name>.z`` both exist,
  the ``.z`` wins, as in ``GetDataStream``, and the library lists the pair in
  :attr:`G4NDLLibrary.duplicates`.
* **Only the processes kika reads are indexed.** Today that is the elastic
  (``Elastic/CrossSection``, ``Elastic/FS``), the inelastic
  (``Inelastic/CrossSection``, the channel directories ``Inelastic/F01`` …
  ``F36`` and the level schemes ``Inelastic/Gammas``) and the capture
  (``Capture/CrossSection`` and its final state, ``Capture/FSMF6`` or
  ``Capture/FS``) and the fission (``Fission/CrossSection``, ``Fission/FS``,
  the chances ``Fission/FC`` … ``LC`` and the yields ``Fission/FF``), at the
  library root and nowhere else: ``JENDL_HE/neutron/Elastic`` is a different, high-energy data
  set that ships in the same tarball, and walking every directory named
  ``Elastic`` would mix the two.

Nothing is read until asked for; building the index lists the directories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from kika.g4ndl.exceptions import G4NDLError, IsotopeNotFoundError
from kika.g4ndl.names import IsotopeKey, TargetLike, parse_file_name, parse_target
from kika.g4ndl.inelastic_records import CHANNEL_MT, GammasRecord, InelasticFSRecord
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.records import CrossSectionRecord, ElasticFSRecord
from kika.g4ndl.stream import DEFAULT_MAX_BYTES, read_text
from kika.g4ndl.tokens import TokenStream

__all__ = ["G4NDLLibrary", "IndexedFile", "open"]

#: Process name -> the subdirectories every isotope of the process has.
PROCESSES: Dict[str, Tuple[str, ...]] = {
    "elastic": ("Elastic/CrossSection", "Elastic/FS"),
    "inelastic": ("Inelastic/CrossSection",),
    "capture": ("Capture/CrossSection",),
    # ``Fission/FS`` is required with the σ: without it Geant4 emits no neutron.
    "fission": ("Fission/CrossSection", "Fission/FS"),
}

#: Process name -> the per-channel subdirectories an isotope has some of.
CHANNELS: Dict[str, Tuple[str, ...]] = {
    "inelastic": tuple(f"Inelastic/{ch}" for ch in sorted(CHANNEL_MT)),
    # The final state: Geant4 reads FSMF6 and, only when there is none, FS.
    "capture": ("Capture/FSMF6", "Capture/FS"),
    # The chances (MT19, 20, 21, 38) and the fragment yields, each optional.
    "fission": ("Fission/FC", "Fission/SC", "Fission/TC", "Fission/LC", "Fission/FF"),
}

#: The residual nuclei's level schemes, ``z<Z>.a<A>``, plain text only.
GAMMAS_DIR = "Inelastic/Gammas"

_GAMMAS_RE = re.compile(r"^z(\d+)\.a(\d+)$")


def _indexedSubdirs() -> Tuple[str, ...]:
    return (tuple(s for subs in PROCESSES.values() for s in subs)
            + tuple(s for subs in CHANNELS.values() for s in subs))

@dataclass(frozen=True)
class IndexedFile:
    """One isotope's file in one subdirectory.

    ``path`` is what gets read; ``shadowed`` is the plain-text twin a ``.z``
    took precedence over, when there is one.
    """

    key: IsotopeKey
    subdir: str
    path: Path
    elementName: str
    shadowed: Optional[Path] = None

    @property
    def compressed(self) -> bool:
        return self.path.suffix == ".z"


class G4NDLLibrary:
    """Index of a G4NDL library directory.

    Parameters
    ----------
    root
        The directory ``G4NEUTRONHPDATA`` would point at, e.g. ``.../G4NDL4.7.1``.
    max_bytes
        Ceiling on the decompressed size of any one file.

    Raises
    ------
    G4NDLError
        If ``root`` is not a directory or holds none of the indexed
        subdirectories — most often a path one level too high, since the
        IAEA tarballs nest ``JEFF-4.0/`` inside ``JEFF-4.0/``.
    """

    def __init__(self, root, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root)
        self.max_bytes = max_bytes
        if not self.root.is_dir():
            raise G4NDLError(f"not a directory: {self.root}")
        self._index: Dict[Tuple[str, IsotopeKey], IndexedFile] = {}
        #: Files in an indexed subdirectory whose name is not ``Z_A[mM]_Element``.
        self.unindexed: List[Path] = []
        #: ``(text, .z)`` pairs where the ``.z`` won.
        self.duplicates: List[Tuple[Path, Path]] = []
        #: ``(Z, A)`` -> the ``Inelastic/Gammas`` file of that nucleus.
        self._gammas: Dict[Tuple[int, int], Path] = {}
        found = False
        for sub in _indexedSubdirs():
            d = self.root / sub
            if d.is_dir():
                found = True
                self._indexDirectory(sub, d)
        gd = self.root / GAMMAS_DIR
        if gd.is_dir():
            for p in gd.iterdir():
                m = _GAMMAS_RE.match(p.name)
                if p.is_file() and m:
                    self._gammas[(int(m.group(1)), int(m.group(2)))] = p
                elif p.is_file() and p.name != "README":
                    self.unindexed.append(p)
        if not found:
            nested = [p for p in self.root.iterdir()
                      if p.is_dir() and (p / "Elastic").is_dir()]
            hint = f"; did you mean {nested[0]}?" if nested else ""
            raise G4NDLError(f"{self.root} holds none of the directories kika reads "
                             f"(Elastic/, Inelastic/, Capture/, Fission/){hint}")

    def _indexDirectory(self, sub: str, d: Path) -> None:
        plain: Dict[str, Path] = {}
        compressed: Dict[str, Path] = {}
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix == ".z":
                compressed[p.name[:-2]] = p
            else:
                plain[p.name] = p
        for stem in sorted(set(plain) | set(compressed)):
            parsed = parse_file_name(stem)
            if parsed is None:
                self.unindexed.extend(x for x in (plain.get(stem), compressed.get(stem)) if x)
                continue
            key, element = parsed
            path = compressed.get(stem) or plain[stem]
            shadowed = plain.get(stem) if stem in compressed else None
            if shadowed is not None:
                self.duplicates.append((shadowed, path))
            self._index[(sub, key)] = IndexedFile(key, sub, path, element, shadowed)

    # ------------------------------------------------------------- queries
    def isotopes(self, process: str = "elastic") -> List[IsotopeKey]:
        """Isotopes with **every** file ``process`` needs, sorted by (Z, A, M).

        ``0_0_Zero``, which both libraries ship in ``Elastic/CrossSection``
        only, is therefore not an elastic isotope. An inelastic isotope is one
        with an ``Inelastic/CrossSection``; its channels are
        :meth:`inelasticChannels`.

        Parameters
        ----------
        process : {"elastic", "inelastic", "capture", "fission"}
            The files needed are ``Elastic/CrossSection`` and ``Elastic/FS``;
            ``Inelastic/CrossSection``; ``Capture/CrossSection``;
            ``Fission/CrossSection`` and ``Fission/FS``.

        Returns
        -------
        list of IsotopeKey
            Ordered by ``Z``, then ``A`` (a natural element, ``A=None``, before
            any ``A``), then ``M``.

        Raises
        ------
        ValueError
            If ``process`` is not one of the four above.
        """
        subdirs = self._subdirs(process)
        keys = {k for (s, k) in self._index if s == subdirs[0]}
        for s in subdirs[1:]:
            keys &= {k for (s2, k) in self._index if s2 == s}
        return sorted(keys, key=lambda k: (k.Z, -1 if k.A is None else k.A, k.M))

    def has(self, target: TargetLike, process: str) -> bool:
        """Whether ``target`` has every file ``process`` needs.

        Parameters
        ----------
        target
            Anything :class:`G4NDLLibrary` accepts as a target: ``"Fe56"``,
            ``"Am242m1"``, ``"Cnat"``, a ZA integer such as ``26056``, a tuple
            ``(Z, A)`` or ``(Z, A, M)``, or an :class:`~kika.g4ndl.IsotopeKey`.
        process : {"elastic", "inelastic", "capture", "fission"}
            The same files as in :meth:`isotopes`.

        Returns
        -------
        bool
            True when the library holds all of them for exactly that isotope.

        Raises
        ------
        ValueError
            If ``process`` is not one of the four above, or ``target`` cannot be
            parsed.
        """
        key = parse_target(target)
        return all((s, key) in self._index for s in self._subdirs(process))

    def inelasticChannels(self, target: TargetLike) -> List[str]:
        """The ``Fxx`` directories holding a file for ``target``, in order.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        list of str
            Names such as ``"F01"``, in the order ``F01`` to ``F36``; empty when
            the isotope has no inelastic channel file.

        Raises
        ------
        ValueError
            If ``target`` cannot be parsed.
        """
        key = parse_target(target)
        return [s.split("/")[1] for s in CHANNELS["inelastic"] if (s, key) in self._index]

    def gammaNuclei(self) -> List[Tuple[int, int]]:
        """``(Z, A)`` of every level scheme in ``Inelastic/Gammas``."""
        return sorted(self._gammas)

    def __contains__(self, target) -> bool:
        try:
            key = parse_target(target)
        except ValueError:
            return False
        return all((s, key) in self._index for s in PROCESSES["elastic"])

    def __len__(self) -> int:
        return len(self.isotopes())

    def __iter__(self) -> Iterator[IsotopeKey]:
        return iter(self.isotopes())

    def presentTopLevel(self) -> List[str]:
        """Top-level directories present, indexed or not — for partial-read reports."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def describe(self) -> dict:
        """The library as a viewer lists it, from the index alone: no file is
        opened, so it stays cheap on a 560-isotope library on a network share.

        ``isotopes`` holds one entry per isotope :meth:`isotopes` returns, with
        the ``target`` name :meth:`read` takes (``Fe56``, ``Cnat``, ``Co58m1``),
        its GNDS id (``Fe56``, ``C``, ``Co58_m1``), ``Z``, ``A`` (``None`` for
        a natural element), ``M``, the element name of its file, whether
        its files are ``.z``, the ``inelastic`` channels it has, whether it has
        ``capture`` and its ``fission`` chances (``None`` without fission,
        ``[]`` for MT18 alone). ``unread``
        names the top-level directories kika does not read yet (``Capture``,
        ``Fission``, ...).
        """
        from kika.g4ndl.decode import targetId

        entries = []
        for key in self.isotopes():
            files = [self._index[(s, key)] for s in PROCESSES["elastic"]]
            entries.append(dict(target=str(key), id=targetId(key), Z=key.Z, A=key.A, M=key.M,
                                element=files[0].elementName,
                                compressed=all(f.compressed for f in files),
                                inelastic=self.inelasticChannels(key),
                                capture=self.has(key, "capture"),
                                fission=(self.fissionChances(key)
                                         if self.has(key, "fission") else None)))
        read = {s.split("/")[0] for s in _indexedSubdirs()}
        return dict(root=str(self.root), name=self.root.name, isotopes=entries,
                    processes=sorted(p for p in PROCESSES if self.isotopes(p)),
                    unread=[d for d in self.presentTopLevel() if d not in read],
                    unindexed=len(self.unindexed), duplicates=len(self.duplicates))

    def locate(self, target: TargetLike, subdir: str) -> IndexedFile:
        """The file holding ``target`` in ``subdir``, e.g. ``"Elastic/FS"``.

        Parameters
        ----------
        target
            As in :meth:`has`.
        subdir
            One of the subdirectories the library indexes: ``Elastic/CrossSection``,
            ``Elastic/FS``, ``Inelastic/CrossSection``, ``Inelastic/F01`` to
            ``F36``, ``Capture/CrossSection``, ``Capture/FSMF6``, ``Capture/FS``,
            ``Fission/CrossSection``, ``Fission/FS``, ``Fission/FC``, ``SC``,
            ``TC``, ``LC`` and ``Fission/FF``.

        Returns
        -------
        IndexedFile
            An entry of the index, with ``key`` (the :class:`~kika.g4ndl.IsotopeKey`),
            ``subdir``, ``path`` (the file that is read: the ``.z`` when both
            variants exist), ``elementName`` (the element name in the file name),
            ``shadowed`` (the plain-text twin the ``.z`` took precedence over, or
            ``None``) and the property ``compressed`` (whether ``path`` is a
            ``.z``).

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no file for exactly that isotope in ``subdir``.
        ValueError
            If ``subdir`` is not indexed, or ``target`` cannot be parsed.
        """
        if subdir not in _indexedSubdirs():
            raise ValueError(f"subdir {subdir!r} is not indexed; "
                             f"indexed: {sorted(_indexedSubdirs())}")
        key = parse_target(target)
        try:
            return self._index[(subdir, key)]
        except KeyError:
            raise IsotopeNotFoundError(
                f"{key} ({key.Z}_{'nat' if key.A is None else key.A}"
                f"{f'm{key.M}' if key.M else ''}) has no file in "
                f"{self.root / subdir}; kika does not substitute the natural "
                "element or a neighbouring isotope") from None

    def tokens(self, target: TargetLike, subdir: str) -> TokenStream:
        """Open ``target``'s file in ``subdir`` as a token stream.

        Parameters
        ----------
        target, subdir
            As in :meth:`locate`.

        Returns
        -------
        TokenStream
            A cursor over the whitespace-separated tokens of the file, after
            zlib decompression for a ``.z``. It consumes the optional
            ``G4NDL <label>`` header (kept in its ``header`` attribute) and reads
            integers, floats and arrays of floats token by token; a malformed or
            missing token raises :class:`~kika.g4ndl.G4NDLFormatError`.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no file for exactly that isotope in ``subdir``.
        ValueError
            If ``subdir`` is not indexed, or ``target`` cannot be parsed.
        ~kika.g4ndl.G4NDLFormatError
            If the file is not valid ASCII or zlib, or decompresses to more than
            ``max_bytes`` of the library.
        """
        f = self.locate(target, subdir)
        return TokenStream(read_text(f.path, self.max_bytes), path=f.path)

    def crossSection(self, target: TargetLike) -> CrossSectionRecord:
        """``target``'s ``Elastic/CrossSection``, parsed to its last token.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        CrossSectionRecord
            ``energy`` and ``sigma`` (arrays of equal length), ``bookkeeping``
            (the two integers that open the file), ``header`` (``("G4NDL",
            label)`` or ``None``) and ``path``. G4NDL stores energies in eV and
            cross-sections in barn; kika does not convert. The pairs keep the
            file order, repeated energies included. The file has no
            interpolation record.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the cross-section grammar, has an energy
            that decreases or a negative cross-section.
        ValueError
            If ``target`` cannot be parsed.
        """
        return parse_cross_section(self.tokens(target, "Elastic/CrossSection"))

    def elasticFinalState(self, target: TargetLike) -> ElasticFSRecord:
        """``target``'s ``Elastic/FS``, parsed to its last token.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        ElasticFSRecord
            The angular distribution of the elastic neutron: ``repFlag`` (0
            isotropic, 1 Legendre, 2 tabulated, 3 Legendre then tabulated),
            ``targetMass`` (the mass ratio AWR, not the mass number),
            ``frameFlag`` (1 laboratory, 2 centre of mass), ``frameFlag2`` (the
            second frame flag, present for ``repFlag=0`` only; Geant4 keeps it
            over the first), ``legendre`` and ``tabulated`` (blocks of records
            per incident energy, ``None`` when ``repFlag`` does not select
            them), ``header`` and ``path``. A Legendre record holds
            ``temperature``, ``energy``, ``tempdep`` and the coefficients
            ``a_1 ... a_NL`` (``a_0 = 1`` is implicit and not stored). A
            tabulated record holds the same three scalars, an interpolation
            record in ``mu``, and ``mu`` and ``probability`` arrays. The record
            has the properties ``transitionEnergy`` (the last Legendre energy
            for ``repFlag=3``, else ``None``), ``temperatures`` and ``tempdeps``.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar.
        ValueError
            If ``target`` cannot be parsed.
        """
        return parse_elastic_fs(self.tokens(target, "Elastic/FS"))

    def inelasticCrossSection(self, target: TargetLike) -> CrossSectionRecord:
        """``target``'s ``Inelastic/CrossSection``, the process cross section.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        CrossSectionRecord
            The same record as :meth:`crossSection` returns, with the energies
            in eV and the cross-sections in barn as G4NDL stores them (kika does
            not convert).

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the cross-section grammar.
        ValueError
            If ``target`` cannot be parsed.
        """
        return parse_cross_section(self.tokens(target, "Inelastic/CrossSection"))

    def inelasticFinalState(self, target: TargetLike, channel: str) -> InelasticFSRecord:
        """``target``'s ``Inelastic/<channel>`` file (``"F01"`` …), parsed to its last token.

        Parameters
        ----------
        target
            As in :meth:`has`.
        channel
            The directory name, ``"F01"`` to ``"F36"``. It decides how the file
            is read: ``F01``, ``F23`` to ``F27`` are read section by section
            with an ``sfType`` per section (one file holds several reactions);
            every other channel is one reaction, with one ``Qvalue`` after the
            first section's header.

        Returns
        -------
        InelasticFSRecord
            ``channel``, ``composite`` (whether the channel is one of the
            section-by-section ones), ``sections`` (in file order; each holds
            ``infoType``, ``dataType`` (the ENDF file number: 3 cross-section, 4
            angular, 5 energy, 6 energy-angle, 12 to 15 photons), ``sfType`` and
            ``dummy`` (both ``None`` outside the composite channels) and the
            parsed ``body``), ``Qvalue`` and ``Qdummy`` (only outside the
            composite channels, else ``None``), ``header`` and ``path``. The
            property ``reactionMTs`` lists the MTs the file states, in order of
            first appearance (the channel's single MT outside the composite
            channels).

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no file for exactly that isotope in the channel.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar of its channel.
        ValueError
            If ``channel`` is not an indexed directory, or ``target`` cannot be
            parsed.
        """
        from kika.g4ndl.inelastic_parse import parse_inelastic_fs

        return parse_inelastic_fs(self.tokens(target, f"Inelastic/{channel}"), channel)

    def captureCrossSection(self, target: TargetLike) -> CrossSectionRecord:
        """``target``'s ``Capture/CrossSection``, MT102's σ.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        CrossSectionRecord
            As :meth:`crossSection`: energies in eV and cross-sections in barn
            as G4NDL stores them (kika does not convert).

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the cross-section grammar.
        ValueError
            If ``target`` cannot be parsed.
        """
        return parse_cross_section(self.tokens(target, "Capture/CrossSection"))

    def captureFinalStateDirectory(self, target: TargetLike) -> Optional[str]:
        """``"Capture/FSMF6"``, ``"Capture/FS"`` or ``None``: what Geant4 reads for ``target``.

        Geant4 would read ``FSMF6`` and never look at ``FS``; no real library
        has an isotope with both, so it is a library someone assembled by hand,
        and which of the two they meant is not kika's to guess.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        str or None
            The directory that holds the file for ``target``; ``None`` when
            neither has one.

        Raises
        ------
        ~kika.g4ndl.G4NDLError
            If both ``Capture/FSMF6`` and ``Capture/FS`` have a file for
            ``target``.
        ValueError
            If ``target`` cannot be parsed.
        """
        key = parse_target(target)
        present = [s for s in CHANNELS["capture"] if (s, key) in self._index]
        if len(present) > 1:
            raise G4NDLError(f"{key} has both {present[0]} and {present[1]}: Geant4 reads "
                             f"the first and ignores the second. Remove the one you did "
                             f"not mean")
        return present[0] if present else None

    def captureFinalState(self, target: TargetLike):
        """``target``'s capture final state (``FSMF6`` or ``FS``), or ``None`` without one.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        object or None
            ``None`` when the library has neither file for ``target``. For a
            ``Capture/FSMF6`` file, a record with ``body`` (one ENDF MF6 body:
            ``targetMass``, ``frameFlag``, ``nProducts`` and the products), ``header``
            and ``path``. For a ``Capture/FS`` file, a record with ``mean`` (the
            mean photon multiplicity, ``repFlag`` 1, or the photon cascade,
            ``repFlag`` 2; its ``targetMass`` is the mass Geant4 uses for the
            capture kinematics), ``angular`` (the photon angular data),
            ``energies`` (the photon spectra; ``energies.needed`` says whether
            the consumer reads any), ``header`` and ``path``.

        Raises
        ------
        ~kika.g4ndl.G4NDLError
            If both ``Capture/FSMF6`` and ``Capture/FS`` have a file for
            ``target``.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar, or is empty (Geant4 would
            then sample the photons from its evaporation model).
        ValueError
            If ``target`` cannot be parsed.
        """
        from kika.g4ndl.capture import parse_capture_mf6, parse_capture_photons

        sub = self.captureFinalStateDirectory(target)
        if sub is None:
            return None
        parser = parse_capture_mf6 if sub == "Capture/FSMF6" else parse_capture_photons
        return parser(self.tokens(target, sub))

    def fissionCrossSection(self, target: TargetLike) -> CrossSectionRecord:
        """``target``'s ``Fission/CrossSection``, MT18's σ.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        CrossSectionRecord
            As :meth:`crossSection`: energies in eV and cross-sections in barn
            as G4NDL stores them (kika does not convert).

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the cross-section grammar.
        ValueError
            If ``target`` cannot be parsed.
        """
        return parse_cross_section(self.tokens(target, "Fission/CrossSection"))

    def fissionFinalState(self, target: TargetLike):
        """``target``'s ``Fission/FS``: ν̄, spectra, energy release, photons.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        object
            A record with ``sections`` (a tuple in file order; each section holds
            ``infoType``, ``dataType`` and the parsed ``body``), ``header`` and
            ``path``. The section bodies are ν̄ total (``infoType`` 2), the
            delayed-neutron data (3), ν̄ prompt (4), the energy release (5, nine
            terms in eV) and the angular, energy and photon data. The method
            ``section(infoType, dataType)`` returns the last section of that
            type, which is the one the consumer keeps when one repeats, or
            ``None``.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar, or holds no section.
        ValueError
            If ``target`` cannot be parsed.
        """
        from kika.g4ndl.fission import parse_fission_fs

        return parse_fission_fs(self.tokens(target, "Fission/FS"))

    def fissionChances(self, target: TargetLike) -> List[str]:
        """The chance directories (``"FC"`` … ``"LC"``) holding a file for ``target``.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        list of str
            A subset of ``["FC", "SC", "TC", "LC"]`` (the chances MT19, 20, 21
            and 38), in that order; empty when the isotope has no chance file.

        Raises
        ------
        ValueError
            If ``target`` cannot be parsed.
        """
        key = parse_target(target)
        return [s.split("/")[1] for s in CHANNELS["fission"][:4] if (s, key) in self._index]

    def chanceFission(self, target: TargetLike, chance: str):
        """``target``'s ``Fission/<chance>`` file (``"FC"`` … ``"LC"``).

        Parameters
        ----------
        target
            As in :meth:`has`.
        chance : {"FC", "SC", "TC", "LC"}
            The chance directory.

        Returns
        -------
        object
            A record with ``chance``, ``crossSection`` (the record
            :meth:`crossSection` returns, in eV and barn as G4NDL stores them;
            its ``bookkeeping[0]`` is the chance's Q in eV, and a negative
            cross-section is accepted), ``angularHead`` and ``energyHead`` (the
            two integer pairs the consumer reads and discards), ``angular`` and
            ``energy`` (the neutron's angular and energy data), and ``path``.
            The last four are ``None`` when the file holds only the cross-section.
            The property ``hasFinalState`` is true when ``angular`` is present.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no such file for exactly that isotope.
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar.
        ValueError
            If ``chance`` is not one of the four, or ``target`` cannot be parsed.
        """
        from kika.g4ndl.fission import parse_chance_fission

        return parse_chance_fission(self.tokens(target, f"Fission/{chance}"), chance)

    def fragmentYields(self, target: TargetLike):
        """``target``'s ``Fission/FF``, or ``None`` when the library has none for it.

        Parameters
        ----------
        target
            As in :meth:`has`.

        Returns
        -------
        object or None
            ``None`` without a file. Otherwise a record with ``blocks`` (in file
            order; each holds ``MT``, ``MF``, ``dummy``, ``imax`` and ``energies``,
            the ``imax + 1`` incident energies) plus ``header`` and ``path``. An
            energy holds ``energy``, ``interpolation`` (the code to the next
            energy) and three arrays with one entry per fragment: ``fsp``
            (``Z*1000 + A``), ``isomer`` and ``yields`` (non-negative).

        Raises
        ------
        ~kika.g4ndl.G4NDLFormatError
            If the file does not follow the grammar, holds no block, or has a
            negative yield.
        ValueError
            If ``target`` cannot be parsed.
        """
        from kika.g4ndl.fission import parse_fragment_yields

        key = parse_target(target)
        if ("Fission/FF", key) not in self._index:
            return None
        return parse_fragment_yields(self.tokens(key, "Fission/FF"))

    def gammas(self, Z: int, A: int) -> GammasRecord:
        """The ``Inelastic/Gammas/z<Z>.a<A>`` level scheme of nucleus ``(Z, A)``.

        Parameters
        ----------
        Z, A
            Charge and mass number of the residual nucleus, as integers.

        Returns
        -------
        GammasRecord
            ``Z``, ``A``, ``path`` and three arrays of equal length, one entry
            per row of the file and in file order: ``levels`` (the level
            energy), ``gammas`` (the photon energy) and ``probabilities``.
            The format stores ``levels`` and ``gammas`` in keV (G4NDL's
            convention); kika does not convert. The file is plain text with no
            header.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If the library has no level scheme for ``(Z, A)``.
        ~kika.g4ndl.G4NDLFormatError
            If the file has a header, or a token count that is not a multiple of
            three.
        """
        from kika.g4ndl.inelastic_parse import parse_gammas

        try:
            path = self._gammas[(int(Z), int(A))]
        except KeyError:
            raise IsotopeNotFoundError(f"no level scheme z{Z}.a{A} in "
                                       f"{self.root / GAMMAS_DIR}") from None
        return parse_gammas(TokenStream(read_text(path, self.max_bytes), path=path),
                            int(Z), int(A))

    def read(self, target: TargetLike, processes: Optional[Sequence[str]] = None):
        """``target`` as a :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

        The elastic is MT2; the inelastic is one reaction per channel and
        partial MT (``kika.g4ndl.inelastic_decode``) plus the ``inelastic``
        total in ``suite.sums``; the capture is MT102
        (``kika.g4ndl.capture``); the fission is MT18 and its chances
        (``kika.g4ndl.fission_model``). What the library holds and kika does
        not read (``ThermalScattering``, ...) is listed in ``suite.report``. This is
        what ``kika.read(root, format="g4ndl", target=...)`` calls.

        Parameters
        ----------
        target
            As in :meth:`has`.
        processes
            A subset of ``("elastic", "inelastic", "capture", "fission")``.
            ``None`` means every one of them that the library has for
            ``target``. An inelastic target counts as present when it has an
            ``Inelastic/CrossSection`` or at least one channel file.

        Returns
        -------
        ~kika.nuclear_data.model.suite.ReactionSuite
            The isotope's reactions and sums, and the conversion report in
            ``suite.report``.

        Raises
        ------
        ~kika.g4ndl.IsotopeNotFoundError
            If a requested process is missing for exactly that isotope, or the
            library holds nothing kika reads for it.
        ValueError
            If a name in ``processes`` is not one of the four, or ``target``
            cannot be parsed.
        ~kika.g4ndl.G4NDLFormatError
            If a file of the requested processes does not follow its grammar.
        ~kika.g4ndl.G4NDLUnsupportedError
            If a file uses what kika will not decide on, such as interpolation
            code 1.
        """
        from kika.g4ndl.decode import decodeElastic, newSuite

        key = parse_target(target)
        present = [p for p in PROCESSES if self.has(key, p)
                   or (p == "inelastic" and self.inelasticChannels(key))]
        wanted = present if processes is None else list(processes)
        for p in wanted:
            self._subdirs(p)
            if p not in present:
                raise IsotopeNotFoundError(f"{key} has no {p} data in {self.root}")
        if not wanted:
            raise IsotopeNotFoundError(f"{key} has no data kika reads in {self.root}")
        readDirs = tuple(sorted({PROCESSES[p][0].split("/")[0] for p in wanted}))
        absent = tuple(sorted({PROCESSES[p][0].split("/")[0] for p in PROCESSES
                               if p not in present}))
        if "elastic" in wanted:
            suite, report = decodeElastic(self.crossSection(key), self.elasticFinalState(key),
                                          key, library=self, readDirectories=readDirs,
                                          absentDirectories=absent)
        else:
            suite, report = newSuite(key, library=self, readDirectories=readDirs,
                                     absentDirectories=absent)
        if "inelastic" in wanted:
            from kika.g4ndl.inelastic_decode import decodeInelastic

            total = (self.inelasticCrossSection(key)
                     if (PROCESSES["inelastic"][0], key) in self._index else None)
            files = [self.inelasticFinalState(key, ch) for ch in self.inelasticChannels(key)]
            decodeInelastic(total, files, suite, library=self, report=report)
        if "capture" in wanted:
            from kika.g4ndl.capture import decodeCapture

            decodeCapture(self.captureCrossSection(key), self.captureFinalState(key), suite,
                          library=self, report=report)
        if "fission" in wanted:
            from kika.g4ndl.fission_model import decodeFission

            chances = {c: self.chanceFission(key, c) for c in self.fissionChances(key)}
            decodeFission(self.fissionCrossSection(key), self.fissionFinalState(key), suite,
                          chances=chances, fragmentYields=self.fragmentYields(key),
                          library=self, report=report)
        return suite

    def _subdirs(self, process: str) -> Tuple[str, ...]:
        try:
            return PROCESSES[process]
        except KeyError:
            raise ValueError(f"process {process!r} is not read by kika yet; "
                             f"supported: {sorted(PROCESSES)}") from None

    def __repr__(self) -> str:
        return f"G4NDLLibrary({str(self.root)!r}, {len(self)} elastic isotopes)"


def open(root, max_bytes: int = DEFAULT_MAX_BYTES) -> G4NDLLibrary:  # noqa: A001
    """Index the G4NDL library at ``root``. See :class:`G4NDLLibrary`.

    Parameters
    ----------
    root
        The directory ``G4NEUTRONHPDATA`` would point at.
    max_bytes
        Ceiling on the decompressed size of any one file.

    Returns
    -------
    G4NDLLibrary
        The index of ``root``; no file is read yet.

    Raises
    ------
    ~kika.g4ndl.G4NDLError
        If ``root`` is not a directory or holds none of the indexed
        subdirectories.
    """
    return G4NDLLibrary(root, max_bytes=max_bytes)

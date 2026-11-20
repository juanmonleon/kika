"""A directory of thermal-scattering (TSL) tapes, as a library of materials.

:func:`kika.library_index.index_endf` lists a library by nuclide, and a TSL
tape has none: its ZA is a pseudo-ZA (H in H₂O is 1001, Al metal 13027 in one
library and 53 in another), so decoding it as a nuclide gives nonsense and
deduplicating on that nonsense drops real materials. :func:`index_tsl` lists a
TSL library the way it is organised: one record per tape, keyed by **MAT and
path**, grouped by compound (:mod:`kika.tsl.naming`), with the files that came
with it (the LEAPR input, the FLASSH inputs, the readme).

It reads, per tape, only what a listing needs:

* MF1/MT451 -- MAT, ZA, AWR, NSUB, ZSYMAM, the lab and author, the description
  text and the directory;
* MF7/MT2 in full (an elastic section is small) -- LTHR and its temperatures;
* MF7/MT4 **up to the end of its first β block** -- LAT, LASYM, LLN, the B
  array (principal and secondary scatterers), the number of α and β, and the
  temperatures every β shares;
* MF7/MT451 from **the tail of the file**, where it sits after MT4.

So a 90 MB tape (``tsl-HinH2O.endf``, 94 temperatures) costs a few hundred kB
of reading, and the whole of ENDF/B-VIII.1's 1.5 GB in a few seconds.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from kika.endf.classes.mf7.scatterer import THERMAL_SCATTERING_NSUB

from .naming import tsl_naming, tsl_stem

__all__ = ["TSLTape", "TSLCompound", "TSLLibraryIndex", "read_tsl_header",
           "index_tsl", "companion_kind"]

_TAIL_BYTES = 65536

#: Elastic treatment by LTHR (ENDF-6 §7.2).
_ELASTIC = {1: "coherent", 2: "incoherent", 3: "both"}

#: B(6i+1) of a secondary scatterer (ENDF-6 §7.4): the law it follows.
_SECONDARY_LAW = {0: "sct", 1: "free-gas", 2: "diffusive"}


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

def _float(field_text: str) -> float:
    s = field_text.strip().replace("D", "E").replace("d", "e")
    if not s:
        return 0.0
    s = re.sub(r"(?<=[0-9.])([+-])(\d+)$", r"E\1\2", s)
    return float(s)


def _int(field_text: str) -> int:
    s = field_text.strip()
    return int(s) if s else 0


def _fields(line: str) -> List[str]:
    return [line[i:i + 11] for i in range(0, 66, 11)]


def _mf_mt(line: str):
    try:
        return int(line[70:72]), int(line[72:75])
    except ValueError:
        return None, None


def _values(lines: List[str], start: int, count: int):
    """*count* numbers from the six-per-line body starting at line *start*."""
    out: List[float] = []
    index = start
    while len(out) < count:
        for f in _fields(lines[index]):
            if len(out) == count:
                break
            out.append(_float(f))
        index += 1
    return out, index


def _lines_for(count: int) -> int:
    return math.ceil(count / 6) if count else 0


# --------------------------------------------------------------------------
# what a tape is
# --------------------------------------------------------------------------

@dataclass
class TSLTape:
    """One TSL tape, as a library lists it.

    Returned by :func:`read_tsl_header` for an ENDF-6 tape and listed in
    :attr:`TSLLibraryIndex.tapes` for every format. Fields a format does not hold
    keep their default (None, empty, or 0 for ``mat``, ``za`` and ``awr``).
    Cross sections are in barns. Temperatures are in K for ENDF-6 (by its
    definition), ACE (from the header's kT) and OpenMC (from kT, rounded to
    0.1 K). GNDS and G4NDL give the numbers as the file states them, with no unit
    conversion.

    Attributes
    ----------
    path
        The file, as a string. For ACE, the file of the lowest temperature; for
        G4NDL, the ``ThermalScattering`` folder.
    file_name
        The file's name. For ACE tables of one material in several files,
        ``<stem>.*t``; for G4NDL, the material name.
    size
        Bytes on disk (the sum over the material's files for ACE and G4NDL).
    mat
        MAT. For ACE, the digits of the header's MAT; for GNDS, the number after
        ``MATERIAL`` in its ENDF-compatible text. 0 for OpenMC and G4NDL.
    za, awr
        ZA and AWR of MF1/MT451. A TSL tape's ZA is a pseudo-ZA, not a nuclide's.
        ACE and OpenMC give the AWR of their header and attributes and leave ZA at
        0; GNDS and G4NDL leave both at 0.
    nsub
        NSUB of the tape. 12 for every non-ENDF format.
    zsymam
        The first 11 characters of the first description line of an ENDF-6 tape
        (or of a GNDS file's ENDF-compatible text). None for the other formats.
    name, principal, compound, variant, name_source
        The labels of :class:`TSLNaming` for this tape, from the rule named by
        ``name_source``. ENDF-6 and GNDS: ``"filename"``, ``"zsymam"`` or
        ``"verbatim"``. ACE: ``"title"`` (``"verbatim"`` when the title matches
        no pattern), and ``name`` is the ZAID stem. OpenMC: ``"openmc"`` (or
        ``"verbatim"``). G4NDL: ``"g4ndl"``.
    lab, author, evaluation_date
        From the first description line of MF1/MT451 (columns 12-22, 34-66 and
        23-32); None when blank. GNDS gives its author and the date of its
        evaluated style, ACE the date of its header.
    description
        The description lines of MF1/MT451 after the first. GNDS gives all the lines
        of its ENDF-compatible text, ACE the table's title.
    directory
        The MF1/MT451 index, one ``[mf, mt, nc]`` row per section. ENDF-6 only.
    elastic
        ``"coherent"``, ``"incoherent"``, ``"both"``, or None when there is no
        elastic part.
    elastic_temperatures
        The temperatures of the elastic part.
    bragg_edges
        The number of Bragg edges. None without a coherent elastic part.
    temperatures
        The temperatures at which S(α,β) (a processed file: the inelastic data)
        is tabulated. G4NDL gives the elastic ones when it has no inelastic part.
    n_alpha, n_beta
        The number of α and of β values of the S(α,β) table (counts of the first
        β block for ENDF-6). None when the tape has no tabulated S(α,β).
    lat
        1 when α and β are scaled with 0.0253 eV instead of kT (the ENDF-6 flag; for
        GNDS, ``calculatedAtThermal``).
    lasym
        1 when S(α,β) is asymmetric (negative β tabulated).
    lln
        1 when the table stores ln S. ENDF-6 only.
    free_xs
        B(1) of MF7/MT4: the free-atom σ of the principal scatterer, which
        already includes the number of principal atoms. ENDF-6 only.
    principal_awr
        AWR of the principal scatterer.
    principal_atoms
        Number of principal atoms in the compound (B(6)).
    secondaries
        One dict per other scatterer, with ``law`` (ENDF-6: ``"sct"``,
        ``"free-gas"`` or ``"diffusive"``; None for GNDS), ``free_xs``, ``awr``
        and ``atoms``.
    composition
        The entries of MF7/MT451 when the tape has one, as dicts with ``zai``,
        ``fraction``, ``awr`` and ``free_xs``.
    companions
        Paths of the files that came with the tape, by kind (see
        :func:`companion_kind`). Filled by :func:`index_tsl`.
    problems
        One string per part of the header that could not be read, with the reason.
    format
        One of ``"endf"``, ``"gnds"``, ``"ace"``, ``"openmc"``, ``"g4ndl"``.
    library
        The library the file names (``ENDF/B-VIII.1``): GNDS ``evaluation``, the
        ACE header's library, the G4NDL source tag. None for the other formats.
    title
        The format's own name for the material: GNDS ``target``, the ACE title,
        the OpenMC group, the G4NDL material.
    paths
        ACE only: every file of the material, one per temperature, by increasing
        temperature.
    member
        The material inside *path* when it holds several: the ACE ZAID stem, the
        OpenMC group, the G4NDL material.
    details
        Format-specific fields: ACE ``zaids``, ``nuclides``, ``outgoing`` and
        ``elastic_mode``; OpenMC ``nuclides``; G4NDL ``processes``.
    """

    path: str
    file_name: str
    size: int
    mat: int
    za: float
    awr: float
    nsub: int
    zsymam: Optional[str]
    name: str
    principal: Optional[str]
    compound: str
    variant: Optional[str]
    name_source: str
    lab: Optional[str] = None
    author: Optional[str] = None
    evaluation_date: Optional[str] = None
    description: List[str] = field(default_factory=list)
    directory: List[List[int]] = field(default_factory=list)   # [mf, mt, nc]
    #: ``'coherent'``, ``'incoherent'``, ``'both'`` or ``None`` (no MT2).
    elastic: Optional[str] = None
    elastic_temperatures: List[float] = field(default_factory=list)
    bragg_edges: Optional[int] = None
    #: MT4: the temperatures every β is tabulated at.
    temperatures: List[float] = field(default_factory=list)
    n_alpha: Optional[int] = None
    n_beta: Optional[int] = None
    lat: Optional[int] = None
    lasym: Optional[int] = None
    lln: Optional[int] = None
    #: Principal scatterer from the B array: free-atom σ, AWR, atoms per molecule.
    free_xs: Optional[float] = None
    principal_awr: Optional[float] = None
    principal_atoms: Optional[float] = None
    #: Non-principal scatterers: ``{law, free_xs, awr, atoms}``.
    secondaries: List[dict] = field(default_factory=list)
    #: MF7/MT451, when the tape has one: ``{zai, fraction, awr, free_xs}``.
    composition: List[dict] = field(default_factory=list)
    #: Files that came with the tape, by kind (:func:`companion_kind`).
    companions: Dict[str, List[str]] = field(default_factory=dict)
    #: Header parts that could not be read, with the reason.
    problems: List[str] = field(default_factory=list)
    #: ``endf``, ``gnds``, ``ace``, ``openmc`` or ``g4ndl``.
    format: str = "endf"
    #: The library the file names (``ENDF/B-VIII.1``), for the formats that say.
    library: Optional[str] = None
    #: The format's own name for the material (an ACE title, an OpenMC group).
    title: Optional[str] = None
    #: Every file of the material when it is more than one (ACE: one per temperature).
    paths: List[str] = field(default_factory=list)
    #: The material inside *path* when *path* holds several (a G4NDL folder).
    member: Optional[str] = None
    #: Format-specific fields worth showing.
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def has_phonon_spectrum(self) -> bool:
        """True when the tape came with a LEAPR input or a FLASSH DOS file."""
        return bool(self.companions.get("leapr") or self.companions.get("flassh-dos"))

    @property
    def key(self) -> str:
        """Unique in a library, stable across re-indexing: ``<MAT>:<file>`` for ENDF-6."""
        if self.format == "endf":
            return f"{self.mat}:{self.file_name}"
        return f"{self.format}:{self.member or self.file_name}"

    def to_dict(self) -> dict:
        """The fields of the tape as a dict, with two more entries.

        ``id`` is :attr:`key` and ``has_phonon_spectrum`` is the property of that name.
        """
        out = asdict(self)
        out["id"] = self.key
        out["has_phonon_spectrum"] = self.has_phonon_spectrum
        return out


@dataclass
class TSLCompound:
    """The tapes of one compound: H₂O's H, UO₂'s O and U in every enrichment.

    Returned by :meth:`TSLLibraryIndex.compounds`, which orders the tapes by
    principal atom, variant and name.

    Attributes
    ----------
    name
        The compound, as :attr:`TSLNaming.compound` gives it.
    tapes
        Its tapes.
    aliases
        Spellings of the compound in the tapes' ZSYMAM that differ from its name
        (``Lucite`` for C5O2H8, ``teflon`` for CF2).
    companions
        Compound-level files by kind (``tsl-H2O.readme``): paths as strings.
    """

    name: str
    tapes: List[TSLTape]
    #: ZSYMAM spellings of the compound that differ from its name (``Lucite``
    #: for C5O2H8, ``teflon`` for CF2).
    aliases: List[str] = field(default_factory=list)
    #: Compound-level files (``tsl-H2O.readme``).
    companions: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def atoms(self) -> List[str]:
        """The principal atoms of the tapes (``H``, ``O``), each once, in tape order."""
        seen: List[str] = []
        for tape in self.tapes:
            if tape.principal and tape.principal not in seen:
                seen.append(tape.principal)
        return seen

    @property
    def variants(self) -> List[str]:
        """The variants of the tapes (``5P``, ``HALEU``), each once, in tape order."""
        seen: List[str] = []
        for tape in self.tapes:
            if tape.variant and tape.variant not in seen:
                seen.append(tape.variant)
        return seen


@dataclass
class TSLLibraryIndex:
    """A TSL library listed by :func:`index_tsl`.

    Attributes
    ----------
    root
        The folder indexed (the thermal folder an OpenMC or G4NDL root was followed
        to), or the file when one tape was given.
    tapes
        One entry per tape or material. Sorted by compound, principal atom,
        variant and name, except when the root is one ACE, OpenMC or GNDS file.
    other_endf
        Paths of ENDF-6 tapes of another sublibrary (NSUB other than 12).
    unreadable
        Paths of the files that are neither a tape nor a companion of one, other
        than an ``xsdir``, ``cross_sections.xml`` or ``readme``. Empty when the root
        is a file.
    compound_companions
        ``{compound: {kind: [paths]}}`` for the files that belong to a compound and
        not to one tape.
    single_file
        True when *root* was a file: the "library" is one material.
    """

    root: str
    tapes: List[TSLTape]
    #: ENDF tapes of another sublibrary found in the folder (not TSL).
    other_endf: List[str]
    #: Files that are neither a tape nor a companion of one.
    unreadable: List[str]
    #: Compound-level companions, by compound name.
    compound_companions: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    #: Indexed from one tape, not a folder: a "library" of one material.
    single_file: bool = False

    @property
    def name(self) -> str:
        """The name of the indexed folder or file (the last part of ``root``)."""
        return Path(self.root).name

    def compounds(self) -> List[TSLCompound]:
        """The tapes grouped by compound, sorted by compound name without regard to case."""
        groups: Dict[str, List[TSLTape]] = defaultdict(list)
        for tape in self.tapes:
            groups[tape.compound].append(tape)
        out = []
        for compound, tapes in groups.items():
            tapes.sort(key=lambda t: (t.principal or "", _variant_order(t.variant), t.name))
            aliases = []
            for tape in tapes:
                alias = _zsymam_compound(tape.zsymam)
                if alias and _is_alias(alias, compound) and alias not in aliases:
                    aliases.append(alias)
            out.append(TSLCompound(compound, tapes, aliases,
                                   self.compound_companions.get(compound, {})))
        out.sort(key=lambda c: c.name.lower())
        return out

    def describe(self) -> dict:
        """The library as a viewer lists it: compounds, each with its tapes."""
        compounds = []
        for compound in self.compounds():
            compounds.append(dict(
                name=compound.name, atoms=compound.atoms, variants=compound.variants,
                aliases=compound.aliases, companions=compound.companions,
                tapes=[tape.to_dict() for tape in compound.tapes]))
        temperatures = sorted({round(t, 1) for tape in self.tapes for t in tape.temperatures})
        formats: Dict[str, int] = {}
        for tape in self.tapes:
            formats[tape.format] = formats.get(tape.format, 0) + 1
        return dict(root=self.root, name=self.name, format="tsl", compounds=compounds,
                    single_file=self.single_file, formats=formats,
                    tapes=len(self.tapes), temperatures=temperatures,
                    other_endf=len(self.other_endf), unreadable=len(self.unreadable))


_VARIANT_ORDER = ["nat", "5P", "10P", "20P", "30P", "HALEU", "HEU", "100P"]


def _variant_order(variant: Optional[str]):
    if variant is None:
        return (0, "")
    if variant in _VARIANT_ORDER:
        return (1, f"{_VARIANT_ORDER.index(variant):02d}")
    return (2, variant)


def _is_alias(alias: str, compound: str) -> bool:
    """A ZSYMAM compound that names the compound differently (``Lucite``, ``teflon``).

    Spellings of the same name (``ice-Ih`` for IceIh, ``SiO2a``, ``UO2-5P``) and
    variant tags read as a compound (``CinUC_5P`` → ``5P``) are not aliases.
    """
    plain = re.sub(r"[^a-z0-9]", "", alias.lower())
    name = re.sub(r"[^a-z0-9]", "", compound.lower())
    if not plain or plain in name or name in plain:
        return False
    return _VARIANT_TOKEN.fullmatch(alias) is None


_VARIANT_TOKEN = re.compile(r"nat|\d+P|HALEU|HEU|alpha|beta|mixed", re.IGNORECASE)


def _zsymam_compound(zsymam: Optional[str]) -> Optional[str]:
    from kika.endf.classes.mf7.scatterer import split_tsl_name
    parsed = split_tsl_name(zsymam.strip()) if zsymam else None
    return parsed[2] if parsed else None


# --------------------------------------------------------------------------
# reading one tape
# --------------------------------------------------------------------------

def _section_lines(path: Path):
    """MF1/451, MF7/2 and the head of MF7/4 of *path*, read front to back.

    MT4 is collected only up to the end of its first β block: its length is
    known from the records' own counts as they arrive, and nothing after it
    is needed for a listing.
    """
    mf1: List[str] = []
    mt2: List[str] = []
    mt4: List[str] = []
    mt4_needed: Optional[int] = None
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            mf, mt = _mf_mt(line)
            if mf == 1 and mt == 451:
                mf1.append(line)
            elif mf == 7 and mt == 2:
                mt2.append(line)
            elif mf == 7 and mt == 4:
                mt4.append(line)
                if mt4_needed is None:
                    mt4_needed = _mt4_head_length(mt4)
                if mt4_needed is not None and len(mt4) >= mt4_needed:
                    break
            elif mf == 7 and mt == 451:
                break
            elif mf not in (None, 0, 1, 7) and mf1:
                break
    return mf1, mt2, mt4


def _mt4_head_length(lines: List[str]) -> Optional[int]:
    """Lines from MT4's HEAD to the end of its first β block, once knowable.

    ``None`` while the records that say so have not arrived yet.
    """
    try:
        if len(lines) < 2:
            return None
        ni = _int(_fields(lines[1])[4])
        idx = 2 + _lines_for(ni)
        if len(lines) < idx:
            return None
        b1 = _float(_fields(lines[2])[0]) if ni else 0.0
        if b1 == 0.0:
            return idx                                   # no tabulated S(α,β)
        if len(lines) <= idx:
            return None
        nr = _int(_fields(lines[idx])[4])
        idx += 1 + _lines_for(2 * nr)                    # TAB2 over β
        if len(lines) <= idx:
            return None
        head = _fields(lines[idx])
        lt, nr, np_ = _int(head[2]), _int(head[4]), _int(head[5])
        idx += 1 + _lines_for(2 * nr) + _lines_for(2 * np_)   # TAB1 at T0
        return idx + lt * (1 + _lines_for(np_))         # LT LISTs at the other T
    except (ValueError, IndexError):
        return None


def _read_mf1(lines: List[str], tape: dict) -> None:
    head = _fields(lines[0])
    tape["za"], tape["awr"], tape["mat"] = _float(head[0]), _float(head[1]), _int(lines[0][66:70])
    tape["nsub"] = _int(_fields(lines[2])[4])
    nwd, nxc = _int(_fields(lines[3])[4]), _int(_fields(lines[3])[5])
    text = [line[:66] for line in lines[4:4 + nwd]]
    if text:
        first = text[0]
        tape["zsymam"] = first[0:11].strip() or None
        tape["lab"] = first[11:22].strip() or None
        tape["evaluation_date"] = first[22:32].strip() or None
        tape["author"] = first[33:66].strip() or None
    tape["description"] = [line.rstrip() for line in text[1:]]
    directory = []
    for line in lines[4 + nwd:4 + nwd + nxc]:
        f = _fields(line)
        directory.append([_int(f[2]), _int(f[3]), _int(f[4])])
    tape["directory"] = directory


def _read_mt2(lines: List[str], tape: dict) -> None:
    lthr = _int(_fields(lines[0])[2])
    tape["elastic"] = _ELASTIC.get(lthr)
    idx = 1
    temperatures: List[float] = []
    if lthr in (1, 3):
        head = _fields(lines[idx])
        t0, lt, nr, np_ = _float(head[0]), _int(head[2]), _int(head[4]), _int(head[5])
        tape["bragg_edges"] = np_
        temperatures.append(t0)
        idx += 1 + _lines_for(2 * nr) + _lines_for(2 * np_)
        for _ in range(lt):
            list_head = _fields(lines[idx])
            temperatures.append(_float(list_head[0]))
            idx += 1 + _lines_for(_int(list_head[4]))
    if lthr in (2, 3):
        head = _fields(lines[idx])
        nr, np_ = _int(head[4]), _int(head[5])
        pairs, _ = _values(lines, idx + 1 + _lines_for(2 * nr), 2 * np_)
        for t in pairs[0::2]:
            if t not in temperatures:
                temperatures.append(t)
    tape["elastic_temperatures"] = temperatures


def _read_mt4(lines: List[str], tape: dict) -> None:
    head = _fields(lines[0])
    tape["lat"], tape["lasym"] = _int(head[3]), _int(head[4])
    b_head = _fields(lines[1])
    tape["lln"] = _int(b_head[2])
    ni, ns = _int(b_head[4]), _int(b_head[5])
    b, idx = _values(lines, 2, ni)
    if len(b) >= 6:
        tape["free_xs"], tape["principal_awr"], tape["principal_atoms"] = b[0], b[2], b[5]
    secondaries = []
    for k in range(1, ns + 1):
        base = 6 * k
        if len(b) < base + 6:
            break
        secondaries.append(dict(law=_SECONDARY_LAW.get(int(b[base]), str(b[base])),
                                free_xs=b[base + 1], awr=b[base + 2], atoms=b[base + 5]))
    tape["secondaries"] = secondaries
    if not b or b[0] == 0.0:
        return
    tab2 = _fields(lines[idx])
    nr, nb = _int(tab2[4]), _int(tab2[5])
    tape["n_beta"] = nb
    idx += 1 + _lines_for(2 * nr)
    tab1 = _fields(lines[idx])
    t0, lt, nr, np_ = _float(tab1[0]), _int(tab1[2]), _int(tab1[4]), _int(tab1[5])
    tape["n_alpha"] = np_
    temperatures = [t0]
    idx += 1 + _lines_for(2 * nr) + _lines_for(2 * np_)
    for _ in range(lt):
        temperatures.append(_float(_fields(lines[idx])[0]))
        idx += 1 + _lines_for(np_)
    tape["temperatures"] = temperatures


def _read_mf7_mt451(path: Path, mat: int, tape: dict) -> None:
    """MF7/MT451 from the tail of the file, where it follows MT4."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - _TAIL_BYTES))
        text = handle.read().decode("latin-1")
    lines = [line for line in text.splitlines()
             if _mf_mt(line) == (7, 451) and _int(line[66:70]) == mat]
    if not lines:
        return
    na = _int(_fields(lines[0])[2])
    idx = 1
    composition = []
    for _ in range(na):
        nw = _int(_fields(lines[idx])[4])
        values, idx = _values(lines, idx + 1, nw)
        for start in range(0, nw, 6):
            zai, _lis, fraction, awr, free_xs, _ = values[start:start + 6]
            composition.append(dict(zai=int(round(zai)), fraction=fraction, awr=awr,
                                    free_xs=free_xs))
    tape["composition"] = composition


def read_tsl_header(path: Union[str, os.PathLike]) -> TSLTape:
    """What a listing needs from one TSL tape, read from its head and its tail.

    Raises ``ValueError`` when *path* is not an ENDF tape at all. A tape of
    another sublibrary comes back with its ``nsub`` and no MF7 data; parts of
    a TSL tape that cannot be read are named in :attr:`TSLTape.problems`.

    Parameters
    ----------
    path
        An ENDF-6 tape. The labels of the result come from its file name, with
        ZSYMAM as the fallback (see :func:`tsl_naming`).

    Returns
    -------
    TSLTape
        With ``format="endf"``. Its temperatures are in K. ``companions`` is empty:
        :func:`index_tsl` fills it.

    Raises
    ------
    ValueError
        If the file has fewer than four MF1/MT451 lines.
    """
    path = Path(path)
    mf1, mt2, mt4 = _section_lines(path)
    if len(mf1) < 4:
        raise ValueError(f"{path.name}: no MF1/MT451 header")
    tape: dict = {}
    _read_mf1(mf1, tape)
    problems: List[str] = []
    if tape["nsub"] == THERMAL_SCATTERING_NSUB:
        for label, lines, reader in (("MF7/MT2", mt2, _read_mt2), ("MF7/MT4", mt4, _read_mt4)):
            if not lines:
                continue
            try:
                reader(lines, tape)
            except (ValueError, IndexError) as exc:
                problems.append(f"{label}: {exc}")
        if any(mf == 7 and mt == 451 for mf, mt, _ in tape["directory"]):
            try:
                _read_mf7_mt451(path, tape["mat"], tape)
            except (ValueError, IndexError) as exc:
                problems.append(f"MF7/MT451: {exc}")
    naming = tsl_naming(path.name, tape.get("zsymam"))
    return TSLTape(path=str(path), file_name=path.name, size=path.stat().st_size,
                   name=naming.name, principal=naming.principal, compound=naming.compound,
                   variant=naming.variant, name_source=naming.source,
                   zsymam=tape.pop("zsymam", None), problems=problems, **tape)


# --------------------------------------------------------------------------
# companions and the index
# --------------------------------------------------------------------------

_COMPANION_SUFFIXES = (
    ("flassh-dos", ("-DOS.txt", "_flassh-DOS.txt")),
    ("flassh-control", ("-Control.txt", "_flassh-Control.txt")),
    ("flassh-crystal", ("-Crystal.txt", "_flassh-Crystal.txt")),
    ("flassh-header", ("-Header_Information.txt", "_flassh-header.txt")),
)


def companion_kind(file_name: str) -> str:
    """The kind of a file that comes with a TSL tape, from its name.

    Parameters
    ----------
    file_name
        The name, with its extension.

    Returns
    -------
    str
        ``"flassh-dos"``, ``"flassh-control"``, ``"flassh-crystal"`` or
        ``"flassh-header"`` for the FLASSH inputs (by their ``-DOS.txt``,
        ``-Control.txt``, ``-Crystal.txt`` and ``-Header_Information.txt`` endings,
        or the ``_flassh-`` spellings); else by extension (not case-sensitive)
        ``"leapr"`` (``.leapr``), ``"readme"``, ``"njoy"`` or ``"script"``
        (``.py``); ``"other"`` for the rest.
    """
    for kind, suffixes in _COMPANION_SUFFIXES:
        if file_name.endswith(suffixes):
            return kind
    suffix = Path(file_name).suffix.lower()
    return {".leapr": "leapr", ".readme": "readme", ".njoy": "njoy",
            ".py": "script"}.get(suffix, "other")


_SEPARATORS = (".", "_", "-")

#: What follows a tape's stem when the file belongs to a sibling variant
#: (``tsl-NinUN`` + ``-5P.readme``), whether or not that sibling is present.
_SIBLING_VARIANT = re.compile(r"^[-_](nat|\d+P|HALEU|HEU|alpha|beta|mixed)(?=[._-]|$)")


def _owner(file_name: str, stems: List[str]) -> Optional[str]:
    """The longest tape stem *file_name* extends (``tsl-UinUO2-5P`` over ``tsl-UinUO2``)."""
    best = None
    for stem in stems:
        if file_name.startswith(stem) and len(file_name) > len(stem) \
                and file_name[len(stem)] in _SEPARATORS \
                and not _SIBLING_VARIANT.match(file_name[len(stem):]):
            if best is None or len(stem) > len(best):
                best = stem
    return best


def _looks_endf(path: Path) -> bool:
    try:
        with path.open("r", encoding="latin-1") as handle:
            for _ in range(3):
                line = handle.readline()
                if _mf_mt(line) == (1, 451):
                    return True
    except OSError:
        return False
    return False


def _nsub_of(path: Union[str, os.PathLike]) -> Optional[int]:
    """NSUB of an ENDF tape from its MF1/MT451 head, or None."""
    try:
        with Path(path).open("r", encoding="latin-1") as handle:
            lines = [handle.readline() for _ in range(4)]
        for line in lines:
            if _mf_mt(line) == (1, 451):
                head = [ln for ln in lines if _mf_mt(ln) == (1, 451)]
                return _int(_fields(head[2])[4]) if len(head) > 2 else None
    except (OSError, ValueError, IndexError):
        return None
    return None


def _thermal_root(root: Path) -> Path:
    """Where a library's thermal data is: an OpenMC root keeps it in ``thermal/``, G4NDL in
    ``ThermalScattering/``. Any other folder is itself."""
    if root.is_dir():
        from .g4ndl import g4ndl_thermal_root
        g4 = g4ndl_thermal_root(root)
        if g4 is not None:
            return g4
        thermal = root / "thermal"
        if thermal.is_dir() and (root / "cross_sections.xml").is_file():
            return thermal
    return root


def index_tsl(root: Union[str, os.PathLike], recursive: bool = False) -> TSLLibraryIndex:
    """Index a directory of TSL materials (or one file), with the files that came with them.

    ENDF-6 tapes (NSUB=12, whatever their name), GNDS TNSL files, ACE thermal
    tables (one material over several files, one per temperature), OpenMC
    thermal HDF5 files and a G4NDL ``ThermalScattering`` folder are all
    listed, each entry carrying its ``format``. An OpenMC or G4NDL library
    root is followed to its thermal folder. Every other file is matched to an
    ENDF-6 tape by name -- the LEAPR input, the FLASSH inputs, the readme --
    or, failing that, to a compound (``tsl-H2O.readme``).

    Parameters
    ----------
    root
        A folder, or one file. For a folder, an OpenMC root (with
        ``cross_sections.xml`` and a ``thermal/`` folder) and a G4NDL root are
        followed to their thermal folder. For one ENDF-6 tape, the files beside
        it that belong to it are gathered too; for one ACE table, its siblings of
        other temperatures.
    recursive
        Descend into sub-folders. It has no effect on a G4NDL
        ``ThermalScattering`` folder, which is read as a whole.

    Returns
    -------
    TSLLibraryIndex
        The tapes (headers only, no S(α,β) is read), the files that belong to
        them, and what was not recognised.
    """
    from kika.library_index import _candidates
    from .ace import ACE_THERMAL_NAME
    from .headers import (ace_tsl_tapes, g4ndl_tsl_tapes, gnds_tsl_header, is_gnds_tsl,
                          openmc_tsl_header)

    root = _thermal_root(Path(root))
    tapes: List[TSLTape] = []
    other_endf: List[str] = []
    rest: List[Path] = []
    ace: List[Path] = []
    from .g4ndl import g4ndl_thermal_root
    if root.is_dir() and g4ndl_thermal_root(root) == root:
        tapes.extend(g4ndl_tsl_tapes(root))
        candidates: List[Path] = []
    else:
        candidates = _candidates(root, recursive)
    for path in candidates:
        lower = path.name.lower()
        if ACE_THERMAL_NAME.match(path.name):
            from .ace import is_ace_thermal
            if is_ace_thermal(path):
                ace.append(path)
                continue
        if lower.endswith((".h5", ".hdf5")):
            from .openmc import is_openmc_thermal
            if is_openmc_thermal(path):
                try:
                    tapes.append(openmc_tsl_header(path))
                    continue
                except (OSError, KeyError, ValueError, StopIteration):
                    pass
            rest.append(path)
            continue
        if lower.endswith(".xml"):
            if is_gnds_tsl(path):
                try:
                    tapes.append(gnds_tsl_header(path))
                    continue
                except (OSError, ValueError):
                    pass
            rest.append(path)
            continue
        # A FLASSH ``_flassh-header.txt`` *is* an MF1/MT451 block with NSUB=12,
        # written for the tape it belongs to; it is that tape's input, not a tape.
        if companion_kind(path.name) != "other" or not _looks_endf(path):
            rest.append(path)
            continue
        try:
            tape = read_tsl_header(path)
        except (ValueError, OSError):
            rest.append(path)
            continue
        if tape.nsub != THERMAL_SCATTERING_NSUB:
            other_endf.append(str(path))
            continue
        tapes.append(tape)
    if ace:
        try:
            tapes.extend(ace_tsl_tapes(ace))
        except (OSError, ValueError, IndexError):
            rest.extend(ace)

    single = root.is_file()
    if single and tapes and tapes[0].format != "endf":
        # A processed file carries no inputs beside it. An ACE table on its own
        # still gathers its siblings of other temperatures.
        if tapes[0].format == "ace":
            siblings = [p for p in root.parent.iterdir() if p.is_file()
                        and ACE_THERMAL_NAME.match(p.name)
                        and p.name.rsplit(".", 1)[0] == root.name.rsplit(".", 1)[0]]
            if len(siblings) > 1:
                tapes = ace_tsl_tapes(sorted(siblings))
        return TSLLibraryIndex(str(root), tapes, other_endf, [], {}, single_file=True)
    if single:
        # One tape on its own still has its inputs beside it: look at the
        # folder it sits in, and keep only what belongs to it.
        rest = sorted(p for p in root.parent.iterdir()
                      if p.is_file() and p != root and not p.name.startswith("."))

    by_dir: Dict[Path, Dict[str, TSLTape]] = defaultdict(dict)
    for tape in tapes:
        if tape.format == "endf":
            by_dir[Path(tape.path).parent][Path(tape.file_name).stem] = tape
    compounds = {tape.compound for tape in tapes}

    compound_companions: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    unreadable: List[str] = []
    for path in rest:
        stems = by_dir.get(path.parent, {})
        owner = _owner(path.name, list(stems))
        kind = companion_kind(path.name)
        if owner is not None:
            stems[owner].companions.setdefault(kind, []).append(str(path))
            continue
        compound = tsl_stem(path.name)
        if compound in compounds and kind != "other":
            compound_companions[compound][kind].append(str(path))
            continue
        if not single and path.name.lower() not in ("xsdir", "cross_sections.xml", "readme"):
            unreadable.append(str(path))

    tapes.sort(key=lambda t: (t.compound.lower(), t.principal or "", _variant_order(t.variant), t.name))
    return TSLLibraryIndex(str(root), tapes, other_endf, unreadable,
                           {c: dict(k) for c, k in compound_companions.items()}, single_file=single)

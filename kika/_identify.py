"""``kika.identify()`` — what kind of file this is, for every format kika reads.

**Why this is not** :func:`kika.sniff_format`. ``sniff_format`` answers a
narrower question: which door of :func:`kika.read` opens this file, and that door
leads only to the GNDS model, so it knows four formats (ENDF, ACE, GNDS, G4NDL).
A program that is handed an arbitrary file — the app's file picker is the case
that asked for this — needs the whole catalogue: covariance libraries, SDF,
MCNP and Serpent input and output, NJOY decks. ``identify`` is that catalogue.
It reuses ``sniff_format``'s directory and GNDS/ACE checks and the MCTAL
detector rather than writing second copies of them.

**Content, never extension**, for the reason ``sniff_format`` gives: in this
field extensions mislead (``.dat``, ``.txt``, no extension, ACE files named
after their ZAID). The one exception is that nothing here *contradicts* a
file's name either; it just never reads it.

**Cheap by construction.** Every check reads the head of the file — a few
kilobytes, at most the first ~64 KiB — except one: telling an NJOY COVFIL from
a groupr GENDF tape means finding the first covariance MF, which streams the
tape and stops there. No parser runs. The app calls this once per picked file,
on files that may live on a network share, and a full parse there costs
seconds per file.

**What comes back** is a short string from :data:`KINDS`. ``identify`` raises
:class:`~kika._read.UnknownFormatError` rather than guess, naming what was
tried; ``'csv'`` is the last check and the weakest, and anything that falls
past it is unknown.
"""
from __future__ import annotations

import os
import re

__all__ = ["identify", "KINDS"]

#: Every answer :func:`identify` can give.
#:
#: The three covariance kinds are kept apart on purpose. ``read_covfil``,
#: ``read_coverx`` and ``read_boxer`` are three readers, and a caller that knows
#: which one to call avoids ``load_covariance``'s try-each-in-turn.
KINDS = (
    "endf", "ace", "gnds", "g4ndl",
    "covfil", "coverx", "boxer",
    "sdf",
    "mcnp-input", "mcnp-mctal",
    "serpent-input", "serpent-sens", "serpent-det", "serpent-res",
    "serpent-his", "serpent-dep",
    "njoy-input",
    "csv",
)

_HEAD_BYTES = 64 * 1024

#: MF numbers that only a covariance file opens with after MF1 (and MF3).
_COVARIANCE_MF = {31, 32, 33, 34, 35, 40}


def identify(path) -> str:
    """Which of :data:`KINDS` the file (or directory) at ``path`` is.

    Raises
    ------
    FileNotFoundError
        Nothing at ``path``.
    UnknownFormatError
        It matched none of them. The message names what was tried.
    """
    from ._read import UnknownFormatError, sniff_format

    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    if os.path.isdir(path):
        # The only directory format; sniff_format owns that test and its message.
        return sniff_format(path)

    with open(path, "rb") as handle:
        raw = handle.read(_HEAD_BYTES)
    if not raw.strip():
        raise UnknownFormatError(f"{path} is empty")

    if _isBinary(raw):
        if _looksLikeCoverxBinary(raw):
            return "coverx"
        raise UnknownFormatError(
            f"{path} is binary and does not open with a Fortran record marker, "
            f"so it is not a binary COVERX file, the only binary format kika reads"
        )

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    from ._read import _looksLikeAce
    if _gndsRoot(text) in ("reactionSuite", "covarianceSuite"):
        return "gnds"
    if _looksLikeAce(lines[:6]):
        return "ace"
    if _THERMAL_ACE.match(lines[0]):
        raise UnknownFormatError(
            f"{path} is a thermal-scattering ACE table ({lines[0].split()[0]}); "
            f"kika's ACE reader reads continuous-energy tables only")

    from .mcnp.detect_file_type import _is_mctal
    if _is_mctal([ln for ln in lines[:20] if ln.strip()][:4]):
        return "mcnp-mctal"

    serpent = _serpentOutputKind(lines)
    if serpent:
        return serpent

    if _looksLikeSdf(lines):
        return "sdf"

    if _looksLikeEndfTape(lines):
        return _endfOrCovfil(path, lines)

    if _looksLikeBoxer(lines):
        return "boxer"

    if _looksLikeCoverxText(lines):
        return "coverx"

    if _looksLikeNjoyDeck(lines):
        return "njoy-input"

    listing = _outputListing(lines)
    if listing:
        raise UnknownFormatError(
            f"{path} is a {listing} output listing (the run's text log); kika "
            f"reads that code's structured outputs, not the listing")

    deck = _inputDeckKind(lines)
    if deck:
        return deck

    if _looksLikeCsv(lines):
        return "csv"

    raise UnknownFormatError(
        f"{path} matched none of the formats kika reads ({', '.join(KINDS)}). "
        f"Checked: GNDS XML, ACE header, MCTAL header, Serpent MATLAB output, SDF "
        f"header, ENDF MAT/MF/MT columns, BOXER and COVERX headers, NJOY module "
        f"names, MCNP and Serpent input cards, CSV."
    )


# ----------------------------------------------------------------------
# Binary
# ----------------------------------------------------------------------

def _isBinary(raw: bytes) -> bool:
    """A NUL byte, or a head that is mostly not text.

    The same four-byte test ``parse_covmat._detect_coverx_format`` uses is too
    narrow to stand alone here: it was written for a file already known to be
    COVERX, and a UTF-8 file with an accented title in its first four bytes
    would fail it.
    """
    if b"\x00" in raw[:4096]:
        return True
    sample = raw[:4096]
    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127 or b >= 128)
    return printable < 0.9 * len(sample)


def _looksLikeCoverxBinary(raw: bytes) -> bool:
    """Sequential unformatted Fortran, and record 2 is COVERX's control record.

    A record marker alone is not enough: NJOY's own binary tapes (``moder`` to
    a negative unit) are sequential unformatted Fortran too, and the first scan
    of a real workspace called every one of them COVERX. What they do not have
    is ``read_coverx``'s second record — exactly seven integers, NGROUP and
    NNGRUP among them, both positive.
    """
    import struct
    for endian in (">", "<"):
        records, offset = [], 0
        while len(records) < 2 and offset + 4 <= len(raw):
            length = struct.unpack(f"{endian}i", raw[offset:offset + 4])[0]
            end = offset + 4 + length
            if not 0 <= length <= 100_000 or end + 4 > len(raw):
                break
            if struct.unpack(f"{endian}i", raw[end:end + 4])[0] != length:
                break
            records.append(raw[offset + 4:end])
            offset = end + 4
        if len(records) == 2 and len(records[1]) == 28:
            ngroup, nngrup = struct.unpack(f"{endian}7i", records[1])[:2]
            if 0 < ngroup <= 10_000 and 0 < nngrup <= 10_000:
                return True
    return False


#: A thermal ACE header: ``lwtr.20t 0.999167 2.5300E-08 07/19/20``. The table
#: name is a material, not a ZA, so ``_looksLikeAce`` does not match it and the
#: deck heuristics below would (the first scan called ``hh2o.90t`` MCNP input).
_THERMAL_ACE = re.compile(
    r"^\s*[A-Za-z][\w.-]*\.\d{2,3}t\s+[\d.]+\s+[\d.]+E[+-]\d+\s+\d\d/\d\d/\d\d")

_XML_ROOT = re.compile(r"<(?![?!])([A-Za-z_][\w.-]*)")


def _gndsRoot(text):
    """The root element's name, if the file is XML; ``None`` otherwise.

    ``sniff_format`` accepts any ``<?xml`` prolog, which is enough for a door
    that is told it holds nuclear data. Here it is not: a workspace also holds
    XSD schemas and fragments of GNDS (a bare ``<XYs1d>``), and only the two
    GNDS roots are files kika can read.
    """
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped.startswith("<"):
        return None
    match = _XML_ROOT.search(stripped[:8192])
    return match.group(1) if match else None


# ----------------------------------------------------------------------
# Serpent's MATLAB-style output
# ----------------------------------------------------------------------

#: Checked against each line in turn; the first line that matches any of them
#: decides. Order inside a line does not matter, the patterns are disjoint.
#: Each one is the variable the corresponding ``kika.serpent.parse_*`` reader
#: is built around, so a file that matches is one that reader can open.
_SERPENT_OUTPUT = (
    ("serpent-res", re.compile(r"^\s*\w+\s*\(\s*idx\s*,\s*(\[|1\s*\))")),
    ("serpent-sens", re.compile(r"^\s*(SENS_\w+|ADJ_PERT_\w+_SENS\w*)\s*=")),
    ("serpent-his", re.compile(r"^\s*HIS_\w+\s*=\s*\[")),
    ("serpent-dep", re.compile(r"^\s*(ZAI|NAMES|BU|DAYS|MAT_\w+_(ADENS|MDENS|VOLUME))\s*=\s*\[")),
    ("serpent-det", re.compile(r"^\s*DET\w+\s*=\s*\[")),
)


def _serpentOutputKind(lines):
    for line in lines[:2000]:
        if not line or line.lstrip().startswith("%"):
            continue
        for kind, pattern in _SERPENT_OUTPUT:
            if pattern.match(line):
                return kind
    return None


# ----------------------------------------------------------------------
# SDF
# ----------------------------------------------------------------------

def _looksLikeSdf(lines) -> bool:
    """Lines 2 and 3 are fixed phrases in SCALE's SDF, and the reader requires both."""
    from .sensitivities.sdf_parser import NGROUP_LINE_RE, NPROF_LINE_RE
    return (len(lines) >= 3 and bool(NGROUP_LINE_RE.match(lines[1]))
            and bool(NPROF_LINE_RE.match(lines[2])))


# ----------------------------------------------------------------------
# ENDF-formatted tapes: evaluations and NJOY's COVFIL
# ----------------------------------------------------------------------

def _endfId(line):
    # A comma never appears in an ENDF record's numeric fields; a CSV row with
    # long numbers can put digits in columns 67-75 by accident.
    if len(line) < 75 or "," in line:
        return None
    try:
        mat, mf, mt = int(line[66:70]), int(line[70:72]), int(line[72:75])
    except ValueError:
        return None
    if mat < -1 or not 0 <= mf <= 99 or not 0 <= mt <= 999:
        return None
    return mat, mf, mt


def _looksLikeEndfTape(lines) -> bool:
    """Three or more of the opening records carry the same positive MAT.

    Stricter than ``sniff_format``'s two-matching-lines rule, because here the
    competition is wider: BOXER header cards are 80 columns of integers too,
    and two of them can pass a column test by accident. What they cannot do is
    repeat one material number in columns 67-70 the way every record of an
    ENDF section does.
    """
    mats = [ident[0] for ident in map(_endfId, lines[:12]) if ident and ident[0] > 0]
    return any(mats.count(m) >= 3 for m in set(mats))


def _endfOrCovfil(path, lines) -> str:
    """An evaluation, or an NJOY-processed group tape carrying covariances.

    The MF1/MT451 HEAD record tells them apart: an evaluation stores NLIB in
    its N1 field (zero or more) and NJOY's grouped tapes store a negative
    number there (GENDF's ``-1``, ERRORR's ``-NK``; ``write_covfil`` follows
    ERRORR). A grouped tape is then streamed to its first MF33 or MF34, which
    is what ``read_covfil`` reads. ERRORR also writes MF31 (with MF3) and MF35
    (with MF5), and groupr writes MF3 and MF6 with no covariance at all; those
    are refused by name, since there is no reader to hand them to.

    The order of sections alone is **not** a test. A COVFIL goes MF1, MF3,
    MF33 and never has MF2, which looked like enough until the committed
    synthetic tapes (``micro_fe56_cov.endf``: MF3 and MF33, no MF1 at all)
    turned out to be evaluations of exactly that shape.
    """
    from ._read import UnknownFormatError

    head = next((ln for ln in lines[:4] if (_endfId(ln) or (0, 0, 0))[1:] == (1, 451)), None)
    try:
        grouped = head is not None and int(head[44:55]) < 0
    except ValueError:
        grouped = False
    if not grouped:
        return "endf"

    seen = set()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            ident = _endfId(line.rstrip("\r\n"))
            if not ident:
                continue
            if ident[1] in (33, 34):
                return "covfil"
            seen.add(ident[1])
    others = sorted(seen & _COVARIANCE_MF)
    if others:
        raise UnknownFormatError(
            f"{path} is NJOY ERRORR output carrying "
            f"{', '.join(f'MF{mf}' for mf in others)}; kika's COVFIL reader "
            f"reads MF33 and MF34 only")
    raise UnknownFormatError(
        f"{path} is an NJOY group tape (GENDF: negative N1 in MF1/MT451) with "
        f"no covariance section, so it is not a COVFIL; kika has no GENDF reader")


# ----------------------------------------------------------------------
# BOXER and COVERX text
# ----------------------------------------------------------------------

def _looksLikeBoxer(lines) -> bool:
    """NJOY COVR's card image: header cards ``(I1,A3,8A4,2(I5,I4),2(I4,I3),3I4)``.

    At least two header cards with a positive MAT and a value-format code NVF
    of 0 or 7-14, which is what ``read_boxer`` will accept.
    """
    headers = 0
    for line in lines[:200]:
        padded = line.rstrip("\r").ljust(80)
        if padded[0] not in "01234" or len(line.rstrip()) < 60 or "," in line:
            continue
        try:
            mat, mt = int(padded[36:41]), int(padded[41:45])
            nvf = int(padded[58:61] or 0)
        except ValueError:
            continue
        if mat > 0 and mt >= 0 and (nvf == 0 or 7 <= nvf <= 14):
            headers += 1
            if headers >= 2:
                return True
    return False


def _looksLikeCoverxText(lines) -> bool:
    """SCALE's text COVERX: a title, ``<NGROUP> ...``, then NGROUP+1 boundaries.

    The reader takes the group count from line 2 and the energy boundaries that
    follow it, so that is what is checked — and that the boundaries are
    monotonic, which no title-then-integer text file is by accident.
    """
    if len(lines) < 4:
        return False
    try:
        # The control line is integers only (the writer puts four there).
        control = [int(tok) for tok in lines[1].split()]
    except ValueError:
        return False
    if len(control) < 2:
        return False
    ngroups = control[0]
    if not 2 <= ngroups <= 2000:
        return False
    values = []
    for line in lines[2:2 + ngroups + 2]:
        try:
            values.extend(float(tok.replace("D", "E")) for tok in line.split())
        except ValueError:
            break
        if len(values) >= ngroups + 1:
            break
    values = values[:ngroups + 1]
    # The whole grid, all of it positive: an MCNP deck's second line is a cell
    # number too, and its next card can be a run of numbers, but not a run of
    # ngroups+1 positive ones in order.
    if len(values) < ngroups + 1 or min(values) <= 0:
        return False
    rising = all(a < b for a, b in zip(values, values[1:]))
    falling = all(a > b for a, b in zip(values, values[1:]))
    return rising or falling


# ----------------------------------------------------------------------
# Text input decks
# ----------------------------------------------------------------------

def _looksLikeNjoyDeck(lines) -> bool:
    """The first card that is not a ``--`` comment is an NJOY module name."""
    from .njoy.reader import NJOY_MODULE_NAMES
    for line in lines[:50]:
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        return stripped.split()[0].lower() in NJOY_MODULE_NAMES
    return False


#: Serpent's card names are lowercase and the parser matches them so; matching
#: them case-insensitively let a README's "Include ..." count as a card.
_SERPENT_CARD = re.compile(
    r"^\s*(set|surf|cell|mat|mix|det|pin|lat|therm|include|src|ene|sens|plot|"
    r"mesh|dep|div|fun|trans|particle|nest|solid|utrans|ftrans)\s")
#: The cards a Serpent geometry or material cannot do without.
_SERPENT_STRUCTURE = re.compile(r"^\s*(surf|cell|mat|pin|lat)\s")
_MCNP_CARD = re.compile(
    r"^\s{0,4}(c(\s|$)|m\d+\s|mt\d+\s|f\d+:|\*?f\d+\s|e\d+\s|kcode\s|ksrc\s|nps\s|"
    r"mode\s|sdef\s|imp:|print(\s|$)|tr\d+\s|fm\d+\s|pert\d*:|phys:|cut:)",
    re.IGNORECASE)
#: An MCNP cell card: number, material, [density], then surfaces or keywords.
_MCNP_CELL = re.compile(r"^\s{0,4}\d+\s+\d+\s+(-?[\d.]+(e[+-]?\d+)?\s+)?[-+(:#\d]", re.IGNORECASE)
#: An MCNP surface card: number, optional transform, mnemonic.
_MCNP_SURFACE = re.compile(
    r"^\s{0,4}\*?\d+\s+(\d+\s+)?(p[xyz]?|s[oxyz]?|c/?[xyz]|k/?[xyz]|t[xyz]|sq|gq|rpp|rcc|box|sph|rhp|hex)\s",
    re.IGNORECASE)

_MCNP_COMMENT = re.compile(r"c(\s|$)", re.IGNORECASE)

#: The text logs both codes write. Their numeric tables look like MCNP cell
#: cards: run over the 59 NEA Serpent examples, every ``.out`` was called MCNP
#: input until this check existed.
_LISTING_SIGNATURES = (
    ("Serpent", re.compile(r"^\s*--- Table\s+\d+:")),
    ("Serpent", re.compile(r"Serpent \d+(\.\d+)* -- ")),
    ("MCNP", re.compile(r"Code Name & Version\s*=\s*MCNP", re.IGNORECASE)),
    ("MCNP", re.compile(r"^1mcnp", re.IGNORECASE)),
)


def _outputListing(lines):
    for line in lines[:200]:
        for code, pattern in _LISTING_SIGNATURES:
            if pattern.search(line):
                return code
    return None


def _inputDeckKind(lines):
    """MCNP or Serpent input, by which language more of the cards are in.

    The two share no keywords that matter: Serpent names its cards (``surf``,
    ``cell``, ``mat``) and comments with ``%``; MCNP numbers its cells and
    surfaces, comments with ``c``, and spells materials and tallies ``m1``,
    ``f4:n``. A deck is called for the side that has at least two cards, more
    of them than the other, and at least one card of geometry or material —
    the scan of a real workspace found Makefiles (``include``, ``%.o:``) and
    shell scripts scoring on keywords alone.
    """
    serpent = mcnp = mcnpData = slashed = cards = 0
    serpentGeometry = mcnpGeometry = False
    for line in lines[:1500]:
        if not line.strip():
            continue
        stripped = line.lstrip()
        cards += 1
        slashed += line.rstrip().endswith("/")
        if stripped == "%" or stripped.startswith("% "):
            serpent += 1
        elif _SERPENT_CARD.match(line):
            serpent += 1
            serpentGeometry = serpentGeometry or bool(_SERPENT_STRUCTURE.match(line))
        elif _MCNP_CELL.match(line) or _MCNP_SURFACE.match(line):
            mcnp += 1
            mcnpGeometry = True
        elif _MCNP_CARD.match(line):
            mcnp += 1
            mcnpData += not _MCNP_COMMENT.match(stripped)
    # NJOY's free-format cards end in "/"; a deck of them (plotr's input tapes,
    # an NJOY deck missing its module names) is neither language, but its
    # numeric cards look like MCNP cells.
    if slashed > 0.3 * cards:
        return None
    if serpentGeometry and serpent >= 2 and serpent > mcnp:
        return "serpent-input"
    # A file of PERT cards or of materials (what kika's own perturbation and
    # material generators write) has no geometry but is still MCNP input; two
    # data cards and nothing of Serpent's is the bar for it.
    if (mcnpGeometry or (mcnpData >= 2 and serpent == 0)) and mcnp >= 2 and mcnp > serpent:
        return "mcnp-input"
    return None


def _looksLikeCsv(lines) -> bool:
    """A header and rows with the same number of commas, at least one."""
    rows = [ln for ln in lines[:20] if ln.strip()]
    if len(rows) < 2:
        return False
    if rows[0].lstrip().startswith(("{", "[")):
        return False                    # JSON lines carry commas too
    count = rows[0].count(",")
    return count >= 1 and all(r.count(",") == count for r in rows[1:])

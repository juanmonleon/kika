"""Which material a TSL tape belongs to, for listing a library by compound.

A thermal-scattering library is one tape per *atom in a compound*: H in H₂O
and O in D₂O are separate evaluations, and UO₂ comes as O and U in six
enrichments each. A viewer lists such a library by compound, so each tape needs
three labels: the **principal** atom it describes, the **compound** it is bound
in, and the **variant** that tells it apart from its siblings (``5P``,
``HALEU``, ``ortho``, ``alpha``, ``crystalline``).

**The labels come from the file name first, and say so.** ZSYMAM, the
material's own label, is free text that the ENDF/B-VIII.1 tapes do not use
consistently: ``tsl-UinUO2.endf``, ``-5P``, ``-10P`` and ``-100P`` all write
``U(UO2)``, the polystyrene tapes write ``H(polystyr`` (eleven characters, cut),
the UoS ones ``Be in BeF2``. NNDC's file names follow one convention,
``tsl-<principal>in<compound>[-<variant>]``, so they are read first and ZSYMAM
is the fallback (JEFF-4.0 names its files ``tsl_Be_BeO.txt``, which the same
rules read). :attr:`TSLNaming.source` records which rule produced the labels.

This is for *grouping* a library, not for identifying an evaluation: a tape's
identity is its MAT, and :func:`kika.endf.thermal_scatterer` keeps its own
ZSYMAM-first naming for that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from kika.endf.classes.mf7.scatterer import split_tsl_name

__all__ = ["TSLNaming", "tsl_naming", "tsl_stem", "tsl_naming_from_title", "tsl_naming_from_openmc"]

_PREFIX = re.compile(r"^tsl[-_]", re.IGNORECASE)

#: A suffix that makes a tape one of several of the same atom in the same
#: compound: an enrichment ladder (``-5P`` … ``-100P``, ``-HALEU``, ``-HEU``,
#: ``_nat``), a crystal phase (``-alpha``, ``-beta``) or NNL's ``-mixed``
#: elastic treatment.
_VARIANT = re.compile(r"^(?P<base>.+?)[-_](?P<variant>nat|\d+P|HALEU|HEU|alpha|beta|mixed)$")

#: ``HinH2O``, ``H1inCaH2``, ``7LiinFLiBe``: the principal is a symbol with an
#: optional mass number before it or a site index after it, then ``in``.
_JOINED = re.compile(r"^(?P<principal>\d*[A-Z][a-z]?\d*?)in(?P<compound>[A-Z0-9].*)$")

#: ``013_Al_027``: LANL's metals, named the neutron-library way.
_NUMBERED_METAL = re.compile(r"^\d{3}_(?P<symbol>[A-Z][a-z]?)_(?P<a>\d{3})$")

#: ``Be-metal``, ``U-metal-HEU``, ``Be-metal+Sd``.
_METAL = re.compile(r"^(?P<symbol>[A-Z][a-z]?)-metal(?:[-+](?P<variant>.+))?$")

#: ``crystalline-graphite``, ``reactor-graphite-10P``, ``graphiteSd``.
_GRAPHITE = re.compile(r"^(?:(?P<form>crystalline|reactor)-)?graphite(?:-(?P<porosity>\d+P)|(?P<sd>Sd))?$",
                       re.IGNORECASE)

#: ``ortho-H``, ``para-D``: molecular hydrogen and deuterium by spin state.
_SPIN = re.compile(r"^(?P<spin>ortho|para)-(?P<symbol>[HD])$", re.IGNORECASE)

#: ``l-CH4``, ``s-CH4``: methane by phase.
_PHASE = re.compile(r"^(?P<phase>[ls])-(?P<compound>.+)$")

#: JEFF-4.0's ``4-Be``: Z, then the element.
_Z_SYMBOL = re.compile(r"^\d+-(?P<symbol>[A-Z][a-z]?)$")


@dataclass(frozen=True)
class TSLNaming:
    """The labels a library listing groups a TSL tape by."""

    #: The tape's own name in the library: the file stem without ``tsl-``.
    name: str
    #: The atom the tape describes (``H``, ``7Li``, ``H1``); ``None`` when the
    #: tape treats a whole molecule (benzene, SiO2-beta).
    principal: Optional[str]
    #: The compound the atom is bound in, the key tapes are grouped by
    #: (``H2O``, ``UO2``, ``Graphite``, ``Be metal``).
    compound: str
    #: What sets the tape apart within its compound and atom (``5P``,
    #: ``ortho``, ``alpha``, ``reactor 10P``), ``None`` for the plain one.
    variant: Optional[str]
    #: ``'filename'``, ``'zsymam'`` or ``'verbatim'`` (neither rule applied:
    #: the name is its own compound).
    source: str


def tsl_stem(file_name: Union[str, Path]) -> str:
    """``tsl-HinH2O.endf`` → ``HinH2O``; ``tsl_Be_BeO.txt`` → ``Be_BeO``."""
    return _PREFIX.sub("", Path(file_name).stem)


def _from_name(name: str) -> Optional[TSLNaming]:
    base, variant = name, None
    match = _VARIANT.match(name)
    if match:
        base, variant = match.group("base"), match.group("variant")

    match = _JOINED.match(base)
    if match:
        return TSLNaming(name, match.group("principal"), match.group("compound"), variant, "filename")

    match = _NUMBERED_METAL.match(name)
    if match:
        symbol = match.group("symbol")
        return TSLNaming(name, symbol, f"{symbol} metal", None, "filename")

    match = _METAL.match(name)
    if match:
        symbol = match.group("symbol")
        return TSLNaming(name, symbol, f"{symbol} metal", match.group("variant"), "filename")

    match = _GRAPHITE.match(name)
    if match:
        if match.group("sd"):
            graphite_variant = "Sd"
        elif match.group("form") and match.group("form").lower() == "reactor":
            graphite_variant = f"reactor {match.group('porosity')}" if match.group("porosity") else "reactor"
        else:
            graphite_variant = match.group("form")
        return TSLNaming(name, "C", "Graphite", graphite_variant, "filename")

    match = _SPIN.match(name)
    if match:
        symbol = match.group("symbol").upper()
        return TSLNaming(name, symbol, f"{symbol}2", match.group("spin").lower(), "filename")

    match = _PHASE.match(name)
    if match:
        phase = {"l": "liquid", "s": "solid"}[match.group("phase")]
        return TSLNaming(name, "H", match.group("compound"), phase, "filename")

    match = _Z_SYMBOL.match(name)
    if match:
        symbol = match.group("symbol")
        return TSLNaming(name, symbol, f"{symbol} metal", None, "filename")

    parsed = split_tsl_name(base)
    if parsed is not None:
        _, principal, compound = parsed
        return TSLNaming(name, principal, compound, variant, "filename")

    if variant is not None:
        # ``SiO2-beta``: a whole molecule in one tape, told apart by its phase.
        return TSLNaming(name, None, base, variant, "filename")
    return None


def tsl_naming(file_name: Union[str, Path], zsymam: Optional[str] = None) -> TSLNaming:
    """The principal, compound and variant of a TSL tape, from its name or its ZSYMAM.

    >>> tsl_naming("tsl-OinUO2-HALEU.endf")
    TSLNaming(name='OinUO2-HALEU', principal='O', compound='UO2', variant='HALEU', source='filename')
    """
    name = tsl_stem(file_name)
    naming = _from_name(name)
    if naming is not None:
        return naming
    parsed = split_tsl_name(zsymam.strip()) if zsymam else None
    if parsed is not None:
        _, principal, compound = parsed
        return TSLNaming(name, principal, compound, None, "zsymam")
    return TSLNaming(name, None, name, None, "verbatim")


# --------------------------------------------------------------------------
# processed formats: ACE titles and OpenMC group names
# --------------------------------------------------------------------------

#: ``(5% U235)``, ``(19.75% U235, HALEU)``, ``(natural U)``: an enrichment, as
#: the ENDF file names write it.
_ENRICHMENT = re.compile(r"^(?:(?P<pct>[\d.]+)% U235(?:, (?P<tag>HALEU|HEU))?|natural U)$")


def _enrichment_variant(text: str) -> Optional[str]:
    match = _ENRICHMENT.match(text.strip())
    if not match:
        return None
    if match.group("tag"):
        return match.group("tag")
    if match.group("pct"):
        return f"{float(match.group('pct')):g}P"
    return "nat"


#: ``H in H2O``, ``U in UO2 (5% U235)``, ``H in CH2 (polyethylene)``.
_TITLE_IN = re.compile(r"^(?P<principal>\S+) in (?P<compound>\S+(?: oil)?)(?: \((?P<note>.+)\))?$")
#: ``Al metal``, ``Fe56 metal``, ``U metal (5% U235)``, ``Be metal with Sd effects``.
_TITLE_METAL = re.compile(r"^(?P<symbol>[A-Z][a-z]?)\d*\s+metal(?P<sd> with Sd effects)?(?: \((?P<note>.+)\))?$")
_TITLE_SPIN = re.compile(r"^(?P<spin>ortho|para) (?P<symbol>[HD])$", re.IGNORECASE)
_TITLE_GRAPHITE = re.compile(r"^C in (?P<form>reactor )?graphite(?P<sd> with Sd effects)?"
                             r"(?: - (?P<porosity>\d+)% porosity)?$", re.IGNORECASE)


def tsl_naming_from_title(title: str, fallback: str) -> TSLNaming:
    """The labels of a processed table from its title (ACE: ``H in H2O - ENDF/B-VIII.1 - …``).

    *title* is the material part only (the library and temperature removed).
    """
    text = title.strip()
    match = _TITLE_GRAPHITE.match(text)
    if match:
        if match.group("porosity"):
            variant = f"reactor {match.group('porosity')}P"
        elif match.group("sd"):
            variant = "Sd"
        else:
            variant = "reactor" if match.group("form") else None
        return TSLNaming(text, "C", "Graphite", variant, "title")
    match = _TITLE_METAL.match(text)
    if match:
        symbol = match.group("symbol")
        variant = "Sd" if match.group("sd") else _enrichment_variant(match.group("note") or "")
        return TSLNaming(text, symbol, f"{symbol} metal", variant, "title")
    match = _TITLE_SPIN.match(text)
    if match:
        symbol = match.group("symbol").upper()
        return TSLNaming(text, symbol, f"{symbol}2", match.group("spin").lower(), "title")
    match = _TITLE_IN.match(text)
    if match:
        note = match.group("note") or ""
        variant = _enrichment_variant(note)
        if variant is None and note.startswith("alpha"):
            variant = "alpha"
        return TSLNaming(text, match.group("principal"), match.group("compound"), variant, "title")
    return TSLNaming(text or fallback, None, text or fallback, None, "verbatim")


_OPENMC_VARIANT = re.compile(r"^(?P<base>.+?)_(?P<variant>\d+p|HALEU|HEU|nat|solid|liquid|alpha|beta|distinct)$",
                             re.IGNORECASE)


def tsl_naming_from_openmc(group: str) -> TSLNaming:
    """The labels of an OpenMC thermal table from its group name (``c_O_in_UO2_5p``)."""
    name = group[2:] if group.startswith("c_") else group
    base, variant = name, None
    match = _OPENMC_VARIANT.match(name)
    if match:
        base, variant = match.group("base"), match.group("variant")
        variant = variant.upper() if variant[-1] in "pP" and variant[:-1].isdigit() else variant
    if "_in_" in base:
        principal, compound = base.split("_in_", 1)
        return TSLNaming(name, principal, compound.replace("_", " "), variant, "openmc")
    if base.lower().startswith("graphite"):
        return TSLNaming(name, "C", "Graphite", variant, "openmc")
    match = re.match(r"^(ortho|para)_([HD])$", base)
    if match:
        return TSLNaming(name, match.group(2), f"{match.group(2)}2", match.group(1), "openmc")
    match = re.match(r"^([A-Z][a-z]?)\d*$", base)
    if match:
        return TSLNaming(name, match.group(1), f"{match.group(1)} metal", variant, "openmc")
    return TSLNaming(name, None, base, variant, "verbatim")

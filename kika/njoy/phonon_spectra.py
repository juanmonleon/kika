"""The phonon spectra a thermal-scattering evaluation was made from.

A TSL evaluation's MF7 is LEAPR's *output*; the physics it was computed from is
the phonon density of states (DOS) of each scattering atom, which ENDF/B-VIII.1
distributes beside the tapes: 101 LEAPR inputs (``tsl-*.leapr``, cards 11-12,
"delta ni / rho") and the FLASSH ones (``*-DOS.txt``). It is read here **as
LEAPR's input**, which is what it is (decision of 2026-10-09): a spectrum per
temperature and per atom, an ``XYs1d`` of rho against energy in eV, and it is
**not** hung on the reactionSuite -- it is not an alternative to S(alpha,beta)
but what produced it, and nothing consumes it there yet. GNDS has a node for
one (``GaussianApproximation.phononSpectrum``) and kika reads and writes it
when a file carries it.

The LEAPR input is read the way NJOY reads it: list-directed, so a card starts
on a new record, takes up to its count of values across as many records as it
needs, stops early at ``/``, and the rest of its last record is ignored. The
card order is LEAPR's (``leapr.f90``, cards 1-19); the principal scatterer's
temperatures, then the secondary scatterer's when ``nss > 0`` and ``b7 = 0``.
A negative temperature reuses the previous spectrum, as in LEAPR.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

__all__ = ["PhononSpectrum", "leaprPhononSpectra", "leaprSummary", "flasshPhononSpectra",
           "phononSpectra"]


@dataclass
class PhononSpectrum:
    """One atom's density of states at one temperature, as the evaluation's input states it."""

    atom: str
    spectrum: object                      # an XYs1d: rho(E), E in eV
    temperature: Optional[float] = None   # K; None when the source states none (FLASSH)
    source: Optional[str] = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# list-directed reading
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|/|[^\s,/]+")


def _records(lines):
    """Each line as its tokens; a ``/`` is a token of its own, quotes kept whole."""
    out = []
    for line in lines:
        out.append(_TOKEN.findall(line))
    return out


class _Cards:
    """NJOY's list-directed READ over the records of one module."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.records = _records(self.lines)
        self.at = 0

    def text(self) -> str:
        """The next record as text, up to its ``/``: a title card, read whole."""
        line = self.lines[self.at] if self.at < len(self.lines) else ""
        self.at += 1
        return line.split("/")[0].strip().strip("'\"").strip()

    def read(self, count: Optional[int]):
        """Up to *count* values (all of a ``/``-ended card when ``None``)."""
        values = []
        while self.at < len(self.records):
            tokens = self.records[self.at]
            self.at += 1
            for token in tokens:
                if token == "/":
                    return values
                # r*c is c repeated r times (list-directed input; ENDF/B-VIII.1's
                # tsl-HinUH3.leapr writes 58*0.00000E+00).
                head, star, tail = token.partition("*")
                if star and head.isdigit() and not token.startswith(("'", '"')):
                    values.extend([tail] * int(head))
                else:
                    values.append(token)
                if count is not None and len(values) >= count:
                    return values[:count]
        return values


def _float(token) -> float:
    text = token.strip()
    # Fortran's 1.0-5 (an exponent without E) is valid list-directed input.
    match = re.fullmatch(r"([+-]?\d*\.?\d*)([+-]\d+)", text)
    if match and match.group(1) not in ("", "+", "-"):
        return float(f"{match.group(1)}e{match.group(2)}")
    return float(text)


def _int(token) -> int:
    return int(_float(token))


def _leaprCards(text: str):
    """The card lines of the first ``leapr`` module of an NJOY deck."""
    lines = text.splitlines()
    def name(line):
        word = line.strip().lower().split()[0] if line.strip() else ""
        return word.strip("'\"")

    start = next((i for i, line in enumerate(lines) if name(line) == "leapr"), None)
    if start is None:
        raise ValueError("no leapr module in this input")
    names = {"moder", "reconr", "broadr", "unresr", "heatr", "thermr", "groupr", "gaminr",
             "errorr", "covr", "acer", "powr", "wimsr", "plotr", "viewr", "mixr", "purr",
             "gaspr", "stop", "leapr", "matxsr", "resxsr", "dtfr", "ccccr"}
    body = []
    for line in lines[start + 1:]:
        if name(line) in names:
            break
        body.append(line)
    return body


def _spectrum(delta: float, rho: List[float]):
    from kika.nuclear_data.model import XYs1d
    from kika.nuclear_data.model.axes import Axes

    energies = delta * np.arange(len(rho), dtype=float)
    return XYs1d(xs=energies, ys=np.asarray(rho, dtype=float),
                 axes=Axes.forFunction1d("rho", "1/eV", "energy", "eV"))


def leaprPhononSpectra(path) -> List[PhononSpectrum]:
    """The phonon spectra of a LEAPR input (``tsl-*.leapr``), one per atom and temperature."""
    spectra, _ = _readLeapr(path)
    return spectra


def leaprSummary(path) -> dict:
    """How a LEAPR input made its tape: the deck's parameters and, per atom and
    temperature, the spectrum's grid, the diffusion and the discrete oscillators.

    The numbers a viewer shows beside the spectra; the spectra themselves are
    :func:`leaprPhononSpectra`'s.
    """
    _, summary = _readLeapr(path)
    return summary


_NEUTRON_MASS_AMU = 1.00866491595


def _readLeapr(path):
    path = Path(path)
    cards = _Cards(_leaprCards(path.read_text(encoding="latin-1")))
    cards.read(1)                                    # 1: nout
    title = cards.text()                             # 2: title
    c3 = cards.read(3)                               # 3: ntempr iprint nphon
    ntempr = _int(c3[0])
    c4 = cards.read(5)                               # 4: mat za isabt ilog smin
    c5 = cards.read(6)                               # 5: awr spr npr iel ncold nsk
    nsk = _int(c5[5]) if len(c5) > 5 else 0
    if len(c5) > 3 and _int(c5[3]) == 99:
        # NJOY+NCrystal's LEAPR (ENDF/B-VIII.1 CF2: "slim is only for
        # NJOY+NCrystal"): iel=99 takes the material, its density of states
        # included, from an NCrystal .ncmat file that is not distributed beside
        # the tape. Refused by name rather than read as a deck it is not.
        raise ValueError(f"{path.name}: an NJOY+NCrystal LEAPR input (iel=99); its "
                         f"phonon spectrum is in the NCrystal .ncmat file it names, "
                         f"not in the deck")
    c6 = cards.read(5)                               # 6: nss b7 aws sps mss
    nss = _int(c6[0])
    b7 = _int(c6[1]) if len(c6) > 1 else 0
    c7 = cards.read(3)                               # 7: nalpha nbeta lat
    alphas = cards.read(_int(c7[0]))                 # 8: alphas
    betas = cards.read(_int(c7[1]))                  # 9: betas
    mat, za = _int(c4[0]), _int(c4[1])

    def number(values, index, default=None, cast=_float):
        return cast(values[index]) if len(values) > index else default

    summary = dict(
        source=str(path),
        title=title or None,
        ntempr=ntempr, nphon=number(c3, 2, cast=_int), mat=mat, za=za,
        isabt=number(c4, 2, 0, _int), ilog=number(c4, 3, 0, _int),
        awr=number(c5, 0), spr=number(c5, 1), npr=number(c5, 2, cast=_int),
        iel=number(c5, 3, 0, _int), ncold=number(c5, 4, 0, _int), nsk=nsk,
        nss=nss, b7=b7, aws=number(c6, 2), sps=number(c6, 3), mss=number(c6, 4, cast=_int),
        nalpha=_int(c7[0]), nbeta=_int(c7[1]), lat=number(c7, 2, 0, _int),
        alpha_range=[_float(alphas[0]), _float(alphas[-1])] if alphas else None,
        beta_range=[_float(betas[0]), _float(betas[-1])] if betas else None,
        temperatures=[],
    )

    out: List[PhononSpectrum] = []

    def temperatures(atom: str, principal: bool):
        previous = None
        for _ in range(ntempr):
            temperature = _float(cards.read(1)[0])   # 10
            reused = temperature < 0 and previous is not None
            if reused:
                delta, rho, physics = previous
            else:
                c11 = cards.read(2)                  # 11: delta ni
                delta, ni = _float(c11[0]), _int(c11[1])
                rho = [_float(v) for v in cards.read(ni)]     # 12
                c13 = [_float(v) for v in cards.read(3)]      # 13: twt c tbeta
                nd = _int(cards.read(1)[0])          # 14
                energies, weights = [], []
                if nd > 0:
                    energies = [_float(v) for v in cards.read(nd)]   # 15
                    weights = [_float(v) for v in cards.read(nd)]    # 16
                if principal and nsk != 0:
                    c17 = cards.read(2)              # 17: nka dka
                    nka = _int(c17[0])
                    if nka > 0:
                        cards.read(nka)              # 18
                    if nsk == 2:
                        cards.read(1)                # 19
                physics = dict(
                    twt=c13[0] if c13 else None,
                    c=c13[1] if len(c13) > 1 else None,
                    tbeta=c13[2] if len(c13) > 2 else None,
                    oscillators=[dict(energy=e, weight=w) for e, w in zip(energies, weights)])
                previous = (delta, rho, physics)
            out.append(PhononSpectrum(atom=atom, spectrum=_spectrum(delta, rho),
                                      temperature=abs(temperature), source=str(path),
                                      extra={"mat": mat, "za": za, "reused": reused,
                                             "symbol": symbols.get(atom), **physics}))
            summary["temperatures"].append(dict(
                atom=atom, temperature=abs(temperature), reused=reused,
                delta=delta, points=len(rho), **physics))

    # Which atom each block is: the masses (AWR x m_n) matched against the
    # elements the file name gives, as for FLASSH sites.
    elements = _elementsNamedBy(path)
    symbols = {}
    for atom, awr in (("principal", number(c5, 0)), ("secondary", number(c6, 2))):
        if awr:
            symbols[atom] = _symbolForMass(awr * _NEUTRON_MASS_AMU, elements or None)
    summary["symbols"] = symbols

    temperatures("principal", True)
    if nss > 0 and b7 == 0:
        temperatures("secondary", False)
    return out, summary


def flasshPhononSpectra(path) -> List[PhononSpectrum]:
    """The densities of states of a FLASSH ``*-DOS.txt``: one per atom, no temperature.

    The file states the energy interval, the number of points, then one comma
    separated row per atom (``... /DOS k``).
    """
    path = Path(path)
    lines = [l for l in path.read_text(encoding="latin-1").splitlines() if l.strip()]
    delta = _float(lines[0].split("/")[0].split(",")[0])
    count = _int(lines[1].split("/")[0].split(",")[0])
    # The rows follow, comma separated, one per scattering atom, labelled after
    # a "/" in some files ("/DOS 1", "/O DOS", "/2Zr1") and not in others, and
    # a row may wrap over several lines. The numbers are read in order and cut
    # every `count` points. A shorter "/"-terminated card where a row would
    # start is FLASSH's energy-grid specification ("1, 200, 381, 181 /") and
    # ends the DOS. Values are kept as written: graphite+Sd's correction rows
    # have small negative entries, and they are the evaluator's.
    rows, buffer, labels = [], [], []
    for line in lines[2:]:
        data, slash, label = line.partition("/")
        numbers = [_float(v) for v in data.replace(",", " ").split()]
        if not buffer and slash and len(numbers) < count:
            break
        buffer.extend(numbers)
        while len(buffer) >= count:
            rows.append(buffer[:count])
            buffer = buffer[count:]
            labels.append(label.strip() or f"DOS {len(rows)}")
    if not rows or buffer:
        raise ValueError(f"{path.name}: the DOS values are not whole rows of the "
                         f"{count} points the file states")
    sites = _flasshSites(path, _elementsNamedBy(path))
    out = []
    for k, row in enumerate(rows):
        label, extra = labels[k], {"row_label": labels[k]}
        if k < len(sites):
            extra.update(sites[k])
            # "DOS 1" says nothing; the crystal file's site mass names the atom.
            if re.fullmatch(r"DOS\s*\d+", label, re.IGNORECASE) and sites[k].get("symbol"):
                label = sites[k]["symbol"]
        # "DOS C", "O DOS", "H1 DOS": the word adds nothing beside a symbol.
        bare = re.sub(r"\bDOS\b", "", label, flags=re.IGNORECASE).strip()
        if bare and not bare.isdigit():
            label = bare
        out.append(PhononSpectrum(atom=label, spectrum=_spectrum(delta, row), source=str(path), extra=extra))
    return out


def _elementsNamedBy(path: Path) -> List[str]:
    """The element symbols a TSL file name names: ``tsl-OinUO2-5P`` → O, U.

    Heavy masses do not name an element on their own (enriched uranium's 237.98
    is nearer Np-238 than any U), so the site masses are matched only against
    the elements of the material.
    """
    from kika._constants import SYMBOL_TO_ATOMIC_NUMBER
    from kika.tsl.naming import tsl_naming

    stem = path.name.split("_flassh-")[0]
    stem = re.sub(r"-(DOS|Control|Crystal|Header_Information)\.txt$", "", stem)
    naming = tsl_naming(stem + ".endf")
    found: List[str] = []
    for text in (naming.principal or "", naming.compound):
        for symbol in re.findall(r"[A-Z][a-z]?", text):
            if (symbol in SYMBOL_TO_ATOMIC_NUMBER or symbol == "D") and symbol not in found:
                found.append(symbol)
    return found


def _flasshSites(dos_path: Path, elements: Optional[List[str]] = None) -> List[dict]:
    """The atom sites of the FLASSH ``-Crystal.txt`` beside a ``-DOS.txt``.

    The crystal file states each non-equivalent site's mass and free-atom cross
    section, in the order the DOS rows follow; the element is the one whose
    mass is nearest (``7.016`` is Li-7, ``2.014`` is D, ``91.22`` natural Zr).
    Empty when there is no crystal file or it does not state masses.
    """
    crystal = dos_path.with_name(dos_path.name[: -len("DOS.txt")] + "Crystal.txt")
    if not crystal.is_file():
        return []
    masses: List[float] = []
    free_xs: List[float] = []
    for line in crystal.read_text(encoding="latin-1").splitlines():
        data, _, comment = line.partition("/")
        values = [v for v in data.replace(",", " ").split()]
        comment = comment.lower()
        try:
            if "mass" in comment:
                masses = [_float(v) for v in values]
            elif "free atom cross section of each" in comment:
                free_xs = [_float(v) for v in values]
        except ValueError:
            continue
    sites = []
    for k, mass in enumerate(masses):
        site = {"mass": mass, "symbol": _symbolForMass(mass, elements)}
        if k < len(free_xs):
            site["free_xs"] = free_xs[k]
        sites.append(site)
    return sites


def _symbolForMass(mass: float, elements: Optional[List[str]] = None) -> Optional[str]:
    """The element (``D`` for hydrogen-2) whose isotope or natural mass is nearest *mass*.

    With *elements*, only those are candidates; without, any element within
    half a mass unit, which is reliable for light atoms only.
    """
    from kika._constants import (ATOMIC_MASS, ATOMIC_NUMBER_TO_SYMBOL, NATURAL_ABUNDANCE,
                                 SYMBOL_TO_ATOMIC_NUMBER)

    candidates = dict(ATOMIC_MASS)
    if elements:
        allowed = {SYMBOL_TO_ATOMIC_NUMBER.get("H" if e == "D" else e) for e in elements}
        candidates = {za: m for za, m in candidates.items() if za // 1000 in allowed}
        if "D" in elements and "H" not in elements:
            candidates = {za: m for za, m in candidates.items() if za != 1001}
    for za, abundances in NATURAL_ABUNDANCE.items():
        if all(z in ATOMIC_MASS for z in abundances) and (not elements or za // 1000 in allowed):
            candidates[za] = sum(f * ATOMIC_MASS[z] for z, f in abundances.items())
    if not candidates:
        return None
    za, best = min(candidates.items(), key=lambda item: abs(item[1] - mass))
    # Enriched uranium (235-238) still lands on U; an unrelated element does not.
    if abs(best - mass) > (5.0 if elements else 0.5):
        return None
    if za == 1002:
        return "D"
    return ATOMIC_NUMBER_TO_SYMBOL.get(za // 1000)


def phononSpectra(path) -> List[PhononSpectrum]:
    """Whichever input *path* is: a LEAPR deck, or a FLASSH density of states."""
    name = Path(path).name
    if name.endswith("-DOS.txt"):
        return flasshPhononSpectra(path)
    return leaprPhononSpectra(path)

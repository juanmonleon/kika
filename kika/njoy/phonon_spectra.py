"""The phonon spectra a thermal-scattering evaluation was made from (roadmap E4-r2).

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

__all__ = ["PhononSpectrum", "leaprPhononSpectra", "flasshPhononSpectra",
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
        self.records = _records(lines)
        self.at = 0

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
    path = Path(path)
    cards = _Cards(_leaprCards(path.read_text(encoding="latin-1")))
    cards.read(1)                                    # 1: nout
    cards.read(1)                                    # 2: title (one quoted string)
    ntempr = _int(cards.read(3)[0])                  # 3: ntempr iprint nphon
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
    cards.read(_int(c7[0]))                          # 8: alphas
    cards.read(_int(c7[1]))                          # 9: betas
    mat, za = _int(c4[0]), _int(c4[1])

    out: List[PhononSpectrum] = []

    def temperatures(atom: str, principal: bool):
        previous = None
        for _ in range(ntempr):
            temperature = _float(cards.read(1)[0])   # 10
            if temperature < 0 and previous is not None:
                delta, rho = previous
            else:
                c11 = cards.read(2)                  # 11: delta ni
                delta, ni = _float(c11[0]), _int(c11[1])
                rho = [_float(v) for v in cards.read(ni)]     # 12
                cards.read(3)                        # 13: twt c tbeta
                nd = _int(cards.read(1)[0])          # 14
                if nd > 0:
                    cards.read(nd)                   # 15
                    cards.read(nd)                   # 16
                if principal and nsk != 0:
                    c17 = cards.read(2)              # 17: nka dka
                    nka = _int(c17[0])
                    if nka > 0:
                        cards.read(nka)              # 18
                    if nsk == 2:
                        cards.read(1)                # 19
                previous = (delta, rho)
            out.append(PhononSpectrum(atom=atom, spectrum=_spectrum(delta, rho),
                                      temperature=abs(temperature), source=str(path),
                                      extra={"mat": mat, "za": za}))

    temperatures("principal", True)
    if nss > 0 and b7 == 0:
        temperatures("secondary", False)
    return out


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
    return [PhononSpectrum(atom=labels[k], spectrum=_spectrum(delta, row), source=str(path))
            for k, row in enumerate(rows)]


def phononSpectra(path) -> List[PhononSpectrum]:
    """Whichever input *path* is: a LEAPR deck, or a FLASSH density of states."""
    name = Path(path).name
    if name.endswith("-DOS.txt"):
        return flasshPhononSpectra(path)
    return leaprPhononSpectra(path)

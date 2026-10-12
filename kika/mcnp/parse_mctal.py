"""Reader for MCNP MCTAL files.

The structure is described in the MCTAL section of the MCNP manuals and follows
the reader in LANL's ``mcnptools`` (BSD-3-Clause, Copyright 2022 Triad National
Security, LLC). Two properties of the format make the parse simple and strict:

* keyword lines (``tally``, ``f``, ``d`` ... ``vals``, ``tfc``, ``kcode``)
  start in column 1, and every data line under them starts with a blank;
* the eight bin lines ``f d u s m c e t`` fix the shape of ``vals``, so the
  number of values read is checked against it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from ._fortran import numbers, numbers_array
from .mctal import AXIS_KEYS, BinAxis, Kcode, Mctal, Perturbation, Tally, Tfc, particles_from_header

_AXIS_LINE = re.compile(r"^([fdusmcret])([ct])?\s+(\d+)((?:\s+-?\d+)*)\s*$")


def read_mctal(filename: Union[str, Path]) -> Mctal:
    """Read an MCNP MCTAL file.

    Every tally, its perturbations (``vals pert``), its fluctuation chart and
    the KCODE block are read. Any tally type, particle and bin structure is
    accepted.

    :param filename: Path to the MCTAL file.
    :returns: The parsed file.
    :raises ValueError: If the file is not a MCTAL file or is inconsistent
        (e.g. the number of values does not match the bin structure).
    """
    with open(filename, "r", errors="replace") as handle:
        lines = handle.read().splitlines()
    return _MctalParser(lines, str(filename)).parse()


class _MctalParser:
    def __init__(self, lines: List[str], name: str):
        self.lines = lines
        self.name = name
        self.i = 0

    # -- helpers ----------------------------------------------------------

    def _fail(self, message: str) -> ValueError:
        return ValueError(f"{self.name}, line {self.i + 1}: {message}")

    def _line(self) -> str:
        return self.lines[self.i] if self.i < len(self.lines) else ""

    def _at_end(self) -> bool:
        return self.i >= len(self.lines)

    def _data_lines(self) -> List[str]:
        """Consume the data lines (blank-started, non-empty) under a keyword."""
        out = []
        while not self._at_end():
            line = self.lines[self.i]
            if line and not line[0].isspace():
                break
            if line.strip():
                out.append(line)
            self.i += 1
        return out

    # -- file -------------------------------------------------------------

    def parse(self) -> Mctal:
        mctal = Mctal()
        self._header(mctal)
        while not self._at_end():
            word = self._line().split()[0].lower() if self._line().strip() else ""
            if word == "tally":
                tally = self._tally(mctal.nps)
                mctal.tallies[tally.id] = tally
            elif word == "kcode":
                mctal.kcode = self._kcode()
            else:
                self.i += 1
        missing = [n for n in mctal.tally_numbers if n not in mctal.tallies]
        if missing:
            raise ValueError(f"{self.name}: tallies listed in ntal but not found: {missing}")
        return mctal

    def _header(self, mctal: Mctal) -> None:
        tokens = self._line().split()
        try:
            if len(tokens) == 3:  # header without code name/version/probid
                mctal.dump, mctal.nps, mctal.random_numbers = (int(t) for t in tokens)
            else:
                mctal.code, mctal.version = tokens[0], tokens[1]
                mctal.probid = " ".join(tokens[2:-3])
                mctal.dump, mctal.nps, mctal.random_numbers = (int(t) for t in tokens[-3:])
        except (ValueError, IndexError):
            raise self._fail(f"not a MCTAL header: {self._line()!r}") from None
        self.i += 1
        mctal.title = self._line().strip()
        self.i += 1

        tokens = self._line().split()
        if len(tokens) < 2 or tokens[0].lower() != "ntal":
            raise self._fail(f"expected 'ntal', got {self._line()!r}")
        ntal = int(tokens[1])
        if len(tokens) >= 4 and tokens[2].lower() == "npert":
            mctal.npert = int(tokens[3])
        self.i += 1
        numbers_listed = [int(v) for v in numbers_array(self._data_lines())]
        if len(numbers_listed) != ntal:
            raise self._fail(f"ntal {ntal} but {len(numbers_listed)} tally numbers listed")
        mctal.tally_numbers = numbers_listed

    # -- tally ------------------------------------------------------------

    def _tally(self, nps: int) -> Tally:
        head = [int(v) for v in self._line().split()[1:]]
        if len(head) < 2:
            raise self._fail(f"malformed tally line {self._line()!r}")
        self.i += 1
        flags: List[int] = []
        if head[1] < 0:
            flags = [int(v) for v in self._line().split()]
            self.i += 1
        tally = Tally(
            id=head[0],
            particles=particles_from_header(head[1], flags),
            detector_type=head[2] if len(head) > 2 else 0,
            modifier=head[3] if len(head) > 3 else 0,
            header=tuple(head[1:]),
        )

        comment = []
        while not self._at_end() and not _AXIS_LINE.match(self._line()):
            if self._line().strip():
                comment.append(self._line().strip())
            self.i += 1
        tally.comment = "\n".join(comment)

        tally.axes = self._axes(tally.id)
        shape = tally.shape

        while not self._at_end():
            words = self._line().split()
            if not words:
                self.i += 1
                continue
            word = words[0].lower()
            if word in ("tally", "kcode"):
                break
            if word != "vals":
                raise self._fail(f"tally {tally.id}: unexpected line {self._line()!r}")
            pert = int(words[2]) if len(words) >= 3 and words[1].lower() == "pert" else None
            self.i += 1
            result, error = self._values(tally.id, shape)
            tfc = self._tfc(tally.id) if self._line().lower().startswith("tfc") else None
            if pert is None:
                tally.result, tally.error, tally.tfc = result, error, tfc
            else:
                tally.perturbations[pert] = Perturbation(pert, result, error, tfc)
        return tally

    def _axes(self, tally_id: int) -> Dict[str, BinAxis]:
        found: Dict[str, BinAxis] = {}
        while not self._at_end():
            match = _AXIS_LINE.match(self._line())
            if not match:
                break
            key, suffix, n, rest = match.groups()
            self.i += 1
            values = numbers_array(self._data_lines())
            slot = "c" if key == "r" else key
            if slot in found:
                raise self._fail(f"tally {tally_id}: bin line {key!r} repeated")
            found[slot] = BinAxis(
                key=key,
                n=int(n),
                values=values if values.size else None,
                has_total=suffix == "t",
                cumulative=suffix == "c",
                flags=tuple(int(v) for v in rest.split()),
            )
        if "f" not in found:
            raise self._fail(f"tally {tally_id}: no 'f' bin line")
        axes = {}
        for key in AXIS_KEYS:
            axes[found[key].key if key in found else key] = found.get(key, BinAxis(key))
        return axes

    def _values(self, tally_id: int, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flat = numbers_array(self._data_lines())
        expected = 2 * int(np.prod(shape))
        if flat.size != expected:
            raise self._fail(
                f"tally {tally_id}: {flat.size // 2} result/error pairs, the bin "
                f"structure {shape} needs {expected // 2}"
            )
        pairs = flat.reshape(*shape, 2)
        return pairs[..., 0].copy(), pairs[..., 1].copy()

    def _tfc(self, tally_id: int) -> Tfc:
        words = self._line().split()
        try:
            n = int(words[1])
            bins = {k: int(v) - 1 for k, v in zip(AXIS_KEYS, words[2:10])}
        except (ValueError, IndexError):
            raise self._fail(f"tally {tally_id}: malformed tfc line {self._line()!r}") from None
        self.i += 1
        rows = [numbers(line) for line in self._data_lines()]
        if len(rows) != n:
            raise self._fail(f"tally {tally_id}: tfc announces {n} entries, found {len(rows)}")
        cols = np.array([row + [0.0] * (4 - len(row)) for row in rows], dtype=float).reshape(n, 4)
        return Tfc(bins=bins, nps=cols[:, 0].astype(np.int64), result=cols[:, 1],
                   error=cols[:, 2], fom=cols[:, 3])

    # -- kcode ------------------------------------------------------------

    def _kcode(self) -> Kcode:
        words = self._line().split()
        try:
            cycles, settle, ndat = (int(v) for v in words[1:4])
        except ValueError:
            raise self._fail(f"malformed kcode line {self._line()!r}") from None
        self.i += 1
        flat = numbers_array(self._data_lines())
        if flat.size != cycles * ndat:
            raise self._fail(f"kcode announces {cycles}x{ndat} values, found {flat.size}")
        return Kcode(cycles=cycles, settle=settle, data=flat.reshape(cycles, ndat))

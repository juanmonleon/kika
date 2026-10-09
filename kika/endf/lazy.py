"""An ENDF tape whose MF files are parsed the first time they are asked for.

``read_endf(path)`` parses every MF that has a registered parser, and that list
only grows: on JENDL-5 U-238 (25 MB) a full read takes 4.1 s where MF3 alone
takes 0.2 s. A caller that wants one file pays for all of them.
:func:`open_endf` reads the tape once, notes where each (MF, MT) section lies
in it, and parses nothing::

    endf = open_endf("n_092-U-238.dat")
    endf.mt451.temperature      # MF1/MT451 alone, not the nubar sections
    endf.files.sections         # every (MF, MT) on the tape, in file order
    3 in endf.files             # True, and still nothing parsed
    endf[3].sections[102]       # parses MF3, once; later accesses reuse it

The object it returns is an ordinary :class:`~kika.endf.classes.endf.ENDF`;
only its ``files`` is a :class:`LazyFiles`, a mapping that parses on
``__getitem__``. Everything written against ``ENDF`` (``endf.mf[4]``,
``endf.files.get(33)``, ``to_plot_data``, the writers) works unchanged, and an
iteration over ``files.values()`` simply parses everything, as a full read would.

**A lazily parsed MF is the eager one.** Each MF is handed to its parser exactly
as :func:`~kika.endf.parsers.parse_endf.parse_endf_file` hands it: the same
lines, grouped by the same function, with the same ``num_lines``. It is *not*
what ``read_endf(path, mf_numbers=[n])`` produces, which feeds the parser the
SEND records as well and has its own history; the full read is the reference.

The index is built by **scanning the MF/MT columns of every line**, not from
the MT451 directory, because a directory is only what the evaluator wrote (see
:func:`kika.endf.tape_inventory`, and its test of a tape whose directory leaves
out the MF34 it carries).

The tape stays on disk and is read again for each MF. If it changes after
:func:`open_endf` -- size or modification time -- the next parse raises rather
than slice the new content with the old offsets.

Thread safety: each MF has its own lock, so two threads asking for the same MF
parse it once, and two asking for different MFs do not wait on each other
(beyond the GIL).
"""
from __future__ import annotations

import io
import os
import threading
from collections.abc import MutableMapping
from typing import Dict, Iterator, List, Optional, Tuple

from .classes.endf import ENDF
from .classes.mf import MF
from .parsers.parse_endf import (
    MF_PARSERS,
    _group_lines_by_mf_mt_with_positions,
    scan_mat_number,
    scan_tape_id,
)

__all__ = ["open_endf", "LazyFiles", "TapeChangedError"]

_Run = Tuple[int, int]  # byte offsets [start, end)


class TapeChangedError(RuntimeError):
    """The tape behind a :class:`LazyFiles` changed after it was indexed."""


def _ids(field: bytes) -> Tuple[Optional[int], Optional[int]]:
    """MF and MT from columns 71-75, ``None`` for a field that is not an integer."""
    try:
        mf = int(field[:2])
    except ValueError:
        mf = None
    try:
        mt = int(field[2:5])
    except ValueError:
        mt = None
    return mf, mt


def _decode(chunk: bytes) -> List[str]:
    """Lines as ``open(path, 'r').readlines()`` would give them.

    Same default encoding and the same universal-newline translation, so the
    parsers see byte-for-byte what the eager path gives them.
    """
    return io.TextIOWrapper(io.BytesIO(chunk)).readlines()


class LazyFiles(MutableMapping):
    """``{MF: MF}`` for one tape, parsing each file on first access.

    The keys are the MF numbers present on the tape *that have a parser* -- the
    files a full ``read_endf`` would hold -- so ``in``, ``len`` and iteration
    cost nothing. Getting a value parses it; :attr:`sections` lists every
    (MF, MT), parser or not.

    Assigning a key stores the object as given (``ENDF.add_file`` does this);
    deleting one forgets it.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        st = os.stat(self.path)
        self._stamp = (st.st_size, st.st_mtime_ns)
        with open(self.path, "rb") as fh:
            data = fh.read()

        # Consecutive lines sharing columns 71-75 form a run; a run is kept
        # under its (MF, MT), SEND (MT = 0) and FEND (MF = 0) included, so a
        # file's lines are its runs in tape order.
        runs: Dict[Tuple[Optional[int], Optional[int]], List[_Run]] = {}
        order: List[Tuple[int, int]] = []
        seen = set()
        pos = 0
        start = 0
        key_field = None
        key = None
        for line in data.splitlines(keepends=True):
            field = line[70:75]
            if field != key_field:
                if key is not None:
                    runs.setdefault(key, []).append((start, pos))
                key_field = field
                key = _ids(field)
                start = pos
                mf, mt = key
                if mf and mt and mf > 0 and mt > 0 and key not in seen:
                    seen.add(key)
                    order.append(key)
            pos += len(line)
        if key is not None:
            runs.setdefault(key, []).append((start, pos))

        self._file_runs: Dict[int, List[_Run]] = {}
        for (mf, _), spans in runs.items():
            if mf is not None and mf > 0:
                self._file_runs.setdefault(mf, []).extend(spans)
        for spans in self._file_runs.values():
            spans.sort()
        self._mt451_runs = sorted(runs.get((1, 451), []))
        self._sections = tuple(order)

        head = _decode(data[:16384])
        self.tape_id = scan_tape_id(head)
        self.mat = scan_mat_number(head)
        if self.mat is None:  # nothing in the first 16 KB carries a MAT
            self.mat = scan_mat_number(_decode(data))

        self._keys = [mf for mf in sorted(self._file_runs) if mf in MF_PARSERS]
        self._parsed: Dict[int, MF] = {}
        self._mt451 = None
        self._locks: Dict[int, threading.Lock] = {}
        self._guard = threading.Lock()

    # -- what is on the tape, without parsing ---------------------------------

    @property
    def sections(self) -> Tuple[Tuple[int, int], ...]:
        """Every (MF, MT) on the tape, in file order -- including MFs kika does not parse."""
        return self._sections

    def mts(self, mf: int) -> Tuple[int, ...]:
        """The MT numbers of one file, from the scan; ``()`` if it is absent."""
        return tuple(mt for f, mt in self._sections if f == mf)

    @property
    def loaded(self) -> Tuple[int, ...]:
        """The MF numbers parsed (or assigned) so far."""
        return tuple(sorted(self._parsed))

    def is_loaded(self, mf: int) -> bool:
        return mf in self._parsed

    def mt451(self):
        """MF1/MT451 parsed on its own, without the nubar sections of MF1.

        The same object a full parse holds in ``files[1].sections[451]``, and
        that one is returned once MF1 has been parsed.
        """
        if 1 in self._parsed:
            return self._parsed[1].sections.get(451)
        with self._lock_for(1):
            if self._mt451 is None and self._mt451_runs:
                from .parsers.parse_mf1 import parse_mf1
                mf1 = parse_mf1(self._read(self._mt451_runs))
                self._mt451 = mf1.sections.get(451)
            return self._mt451

    # -- parsing ---------------------------------------------------------------

    def load(self, *mfs: int) -> None:
        """Parse the given files now (all of them if none are given)."""
        for mf in (mfs or tuple(self._keys)):
            self[mf]

    def _lock_for(self, mf: int) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(mf)
            if lock is None:
                lock = self._locks[mf] = threading.Lock()
            return lock

    def _read(self, spans: List[_Run]) -> List[str]:
        st = os.stat(self.path)
        if (st.st_size, st.st_mtime_ns) != self._stamp:
            raise TapeChangedError(
                f"{self.path} changed on disk after it was opened; open it again"
            )
        lines: List[str] = []
        with open(self.path, "rb") as fh:
            for start, end in spans:
                fh.seek(start)
                lines.extend(_decode(fh.read(end - start)))
        return lines

    def _parse(self, mf: int) -> MF:
        # What parse_endf_file does with the whole tape, done with this file's
        # lines only: the grouping keys on (MF, MT), so the lines of other
        # files never change how this one is grouped.
        groups, counts = _group_lines_by_mf_mt_with_positions(self._read(self._file_runs[mf]))
        mf_lines: List[str] = []
        for mt_lines in groups.get(mf, {}).values():
            mf_lines.extend(mt_lines)
        result = MF_PARSERS[mf](mf_lines)
        if mf in counts:
            result.num_lines = counts[mf]
        return result

    # -- the mapping -----------------------------------------------------------

    def __getitem__(self, mf: int) -> MF:
        try:
            return self._parsed[mf]
        except KeyError:
            pass
        if mf not in self._keys:
            raise KeyError(mf)
        with self._lock_for(mf):
            if mf not in self._parsed:
                self._parsed[mf] = self._parse(mf)
            return self._parsed[mf]

    def __setitem__(self, mf: int, value: MF) -> None:
        self._parsed[mf] = value
        if mf not in self._keys:
            self._keys.append(mf)

    def __delitem__(self, mf: int) -> None:
        if mf not in self._keys:
            raise KeyError(mf)
        self._keys.remove(mf)
        self._parsed.pop(mf, None)

    def __contains__(self, mf) -> bool:
        return mf in self._keys

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"LazyFiles({os.path.basename(self.path)!r}, mf={self._keys}, loaded={list(self.loaded)})"

    # Locks do not pickle or deepcopy; a copy gets fresh ones.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_locks"] = {}
        state["_guard"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._guard = threading.Lock()


def open_endf(filepath: str) -> ENDF:
    """Open an ENDF tape without parsing it; each MF is parsed when first used.

    Parameters
    ----------
    filepath : str
        An ENDF-6 tape.

    Returns
    -------
    ENDF
        With ``files`` a :class:`LazyFiles`. ``mat``, ``tape_id`` and
        ``source_path`` are set as :func:`~kika.endf.read_endf` sets them;
        ``endf.mt451`` and ``endf.files.sections`` answer the identity and
        content questions without parsing any data file.

    Notes
    -----
    Opening costs one read of the tape and a pass over its lines (about 0.1 s
    per 10 MB); see the module docstring for what is and is not guaranteed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENDF file not found: {filepath}")
    files = LazyFiles(filepath)
    endf = ENDF(files=files)
    endf.mat = files.mat
    endf.tape_id = files.tape_id
    endf.source_path = files.path
    return endf

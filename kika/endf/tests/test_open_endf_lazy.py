"""``open_endf`` parses each MF on first access, and gives what ``read_endf`` gives.

The lazy path exists for speed (the app opens a tape to show its MF3 and was
paying for MF1-MF35), and speed is only worth having if nothing changes: every
MF parsed lazily must be the very object a full ``read_endf`` builds. That is
checked here on every committed micro-tape, field by field, arrays included.
Measured on real tapes as well (JENDL-5 U-238: full read 4.35 s, open 0.08 s,
MF3 0.015 s) -- that measurement is not a test, the identity is.
"""
from __future__ import annotations

import copy
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from kika.endf import LazyFiles, TapeChangedError, open_endf, read_endf

DATA = Path(__file__).resolve().parent / "data"
TAPES = sorted(DATA.glob("micro_*.endf"))


def _diff(a, b, path="", seen=None):
    """The first place two parsed objects differ, or None."""
    seen = set() if seen is None else seen
    if id(a) in seen:
        return None
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, np.ndarray):
        ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        return None if ok else f"{path}: arrays differ"
    if isinstance(a, float):
        return None if a == b or (a != a and b != b) else f"{path}: {a} vs {b}"
    if isinstance(a, (int, str, bytes, bool, type(None), complex)):
        return None if a == b else f"{path}: {a!r} vs {b!r}"
    seen.add(id(a))
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys differ"
        return next((d for k in a if (d := _diff(a[k], b[k], f"{path}[{k!r}]", seen))), None)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        return next((d for i, (x, y) in enumerate(zip(a, b))
                     if (d := _diff(x, y, f"{path}[{i}]", seen))), None)
    if hasattr(a, "__dict__"):
        return _diff(vars(a), vars(b), path, seen)
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_the_comparison_can_fail(micro_tape):
    """Guard on the guard: two different MFs must not compare equal."""
    endf = _quiet(read_endf, str(micro_tape))
    assert _diff(endf.files[3], endf.files[4]) is not None


@pytest.mark.parametrize("tape", TAPES, ids=lambda p: p.stem)
def test_every_lazily_parsed_mf_is_the_eager_one(tape):
    eager = _quiet(read_endf, str(tape))
    lazy = open_endf(str(tape))

    assert isinstance(lazy.files, LazyFiles)
    assert lazy.files.loaded == ()
    assert list(lazy.files) == list(eager.files)
    assert (lazy.mat, lazy.zaid, lazy.tape_id) == (eager.mat, eager.zaid, eager.tape_id)
    assert lazy.source_path == eager.source_path

    for mf in eager.files:
        assert _diff(_quiet(lazy.files.__getitem__, mf), eager.files[mf], f"MF{mf}") is None


def test_opening_parses_nothing_and_membership_is_free(micro_tape):
    endf = open_endf(str(micro_tape))
    assert 3 in endf.files and 4 in endf.mf
    assert 99 not in endf.files
    assert len(endf.files) == 5 and repr(endf) == "ENDF(5 files)"
    assert endf.files.loaded == ()


def test_each_mf_is_parsed_once(micro_tape):
    endf = open_endf(str(micro_tape))
    first = endf[3]
    assert endf.files.loaded == (3,)
    assert endf.mf[3] is first and endf.files.get(3) is first and endf.get_file(3) is first


def test_an_absent_mf_is_a_key_error(micro_tape):
    endf = open_endf(str(micro_tape))
    with pytest.raises(KeyError):
        endf.files[33]
    assert endf.files.get(33) is None
    assert endf.files.loaded == ()


def test_mt451_comes_without_parsing_mf1(micro_tape):
    endf = open_endf(str(micro_tape))
    mt451 = endf.mt451
    assert endf.files.loaded == ()
    eager = _quiet(read_endf, str(micro_tape))
    assert _diff(mt451, eager.files[1].sections[451]) is None
    assert endf.is_thermal_scattering is False
    assert endf.files.loaded == ()
    # once MF1 is parsed, mt451 is the section it holds
    mf1 = endf[1]
    assert endf.mt451 is mf1.sections[451]


def test_mt451_on_an_eager_tape_reads_mf1(micro_tape):
    eager = _quiet(read_endf, str(micro_tape))
    assert eager.mt451 is eager.files[1].sections[451]
    assert _quiet(read_endf, str(micro_tape), mf_numbers=[3]).mt451 is None


def test_sections_come_from_the_lines_not_the_directory(micro_tape):
    """This tape's MT451 directory leaves out the MF34 it carries."""
    from kika.endf import tape_inventory

    endf = open_endf(str(micro_tape))
    assert 34 not in tape_inventory(micro_tape).mf
    assert endf.files.sections == tape_inventory(micro_tape, scan=True).sections
    assert endf.files.sections[-1] == (34, 2) and 34 in endf.files
    assert endf.files.mts(3) == (1, 2, 102)
    assert endf.files.mts(33) == ()


def test_tsl_identity_without_parsing_mf7():
    tape = DATA / "micro_tsl_bemetal_elastic.endf"
    endf = open_endf(str(tape))
    assert endf.is_thermal_scattering is True
    assert 7 not in endf.files.loaded


def test_threads_asking_for_one_mf_get_one_object(micro_tape):
    endf = open_endf(str(micro_tape))
    got = []
    barrier = threading.Barrier(6)

    def worker(mf):
        barrier.wait()
        got.append((mf, endf.files[mf]))

    threads = [threading.Thread(target=worker, args=(mf,)) for mf in (3, 3, 3, 4, 4, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for mf in (2, 3, 4):
        assert len({id(obj) for m, obj in got if m == mf}) == 1


def test_a_tape_changed_after_opening_is_refused(micro_tape, tmp_path):
    copy_path = tmp_path / "tape.endf"
    copy_path.write_bytes(micro_tape.read_bytes())
    endf = open_endf(str(copy_path))
    endf[2]
    copy_path.write_bytes(micro_tape.read_bytes() + b"\n")
    st = os.stat(copy_path)
    os.utime(copy_path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(TapeChangedError):
        endf[3]
    assert endf[2] is endf.files[2]  # what was parsed stays usable


def test_assigning_and_deleting_files(micro_tape):
    endf = open_endf(str(micro_tape))
    mf4 = endf[4]
    del endf.files[4]
    assert 4 not in endf.files
    endf.add_file(mf4)
    assert endf.files[4] is mf4


def test_deepcopy_keeps_what_was_parsed_and_stays_lazy(micro_tape):
    endf = open_endf(str(micro_tape))
    endf[3]
    clone = copy.deepcopy(endf)
    assert clone.files.loaded == (3,)
    assert _diff(clone[4], endf[4]) is None


def test_crlf_tapes_parse_the_same(micro_tape, tmp_path):
    crlf = tmp_path / "crlf.endf"
    crlf.write_bytes(micro_tape.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    eager = _quiet(read_endf, str(crlf))
    lazy = open_endf(str(crlf))
    for mf in eager.files:
        assert _diff(lazy[mf], eager.files[mf], f"MF{mf}") is None


def test_open_is_cheaper_than_a_full_read(micro_tape):
    """Loose, so a busy box does not fail it: opening must not parse."""
    t = time.perf_counter()
    _quiet(read_endf, str(micro_tape))
    eager = time.perf_counter() - t
    t = time.perf_counter()
    open_endf(str(micro_tape))
    lazy = time.perf_counter() - t
    assert lazy < eager / 3

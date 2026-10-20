"""The PENDF cache serves only whole tapes.

A crash after the rename of a new entry but before its data reached the disk
leaves a file of the right size whose end is NUL bytes; served as a hit, NJOY
then dies on it far from the cause. The entry is reprocessed instead.
"""
import importlib
from pathlib import Path

from kika.processing import njoy_pendf_cache as cache

# the package exports a function of the same name, which hides the module
reconstruct = importlib.import_module("kika.processing.njoy_reconstruct")

TEND = " " * 66 + "  -1 0  0    0\n"


def _tape(path: Path, end: str = TEND) -> Path:
    path.write_text(" 2.605600+4 5.545443+1          0          0          0          02631 1451    1\n" + end)
    return path


def test_a_tape_is_whole_when_it_ends_with_its_tend_record(tmp_path):
    assert cache._is_complete_tape(_tape(tmp_path / "whole.pendf"))
    assert not cache._is_complete_tape(_tape(tmp_path / "cut.pendf", end=""))
    padded = _tape(tmp_path / "nul.pendf")
    with open(padded, "ab") as f:
        f.write(b"\0" * 8192)
    assert not cache._is_complete_tape(padded)


def test_a_truncated_cache_entry_is_reprocessed(tmp_path, monkeypatch):
    endf = _tape(tmp_path / "fe56.endf")
    store = tmp_path / "cache"
    store.mkdir()
    entry = cache._pendf_cache_path(endf.resolve(), 1e-3, store)
    entry.write_bytes(b"\0" * 4096)
    (tmp_path / "work").mkdir()
    fresh = _tape(tmp_path / "work" / "tape22")

    def stream(*args, **kwargs):
        yield "pendf_path", str(fresh)

    monkeypatch.setattr(reconstruct, "njoy_reconstruct_stream", stream)
    served = cache.get_or_create_pendf(endf, tolerance=1e-3, njoy_exe="njoy", cache_dir=store)
    assert served == entry
    assert served.read_bytes() == fresh.read_bytes()

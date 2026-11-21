"""SHA-256 of the remote cache: recorded on entry, checked on every read, and pinnable."""
from __future__ import annotations

import hashlib

import pytest

import kika.endf.remote as remote
from kika.endf.remote.cache import ENDFCache
from kika.endf.remote.exceptions import ChecksumMismatchError
from kika.endf.remote.iaea_client import IAEAClient

TAPE = b" 2.605600+4 5.545440+1          0          0          0          02631 1451    1\n"
TAPE_SHA = hashlib.sha256(TAPE).hexdigest()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    cache = ENDFCache(tmp_path)
    monkeypatch.setattr(remote, "get_cache", lambda: cache)
    monkeypatch.setattr("kika.endf.remote.iaea_client.get_cache", lambda: cache)
    monkeypatch.setattr("kika.endf.remote.iaea_client._catalog_entry", lambda *a: None)
    monkeypatch.setattr(IAEAClient, "_fetch", lambda self, url, is_archive, **kw: TAPE)
    return cache


def test_the_hash_is_recorded_and_a_changed_file_is_refused(cache):
    path = cache.put(26056, "endfb8.1", TAPE)
    assert cache.sha256(26056, "endfb8.1") == TAPE_SHA
    assert cache.get(26056, "endfb8.1") == path
    path.write_bytes(TAPE.replace(b"5.545440", b"5.545441"))
    with pytest.raises(ChecksumMismatchError) as caught:
        cache.get(26056, "endfb8.1")
    assert caught.value.expected == TAPE_SHA


def test_a_file_cached_before_hashes_were_recorded_is_returned_unchecked(cache):
    path = cache.put(26056, "endfb8.1", TAPE)
    metadata = cache._load_metadata()
    del metadata["endfb8.1/n/26056"]["sha256"]
    cache._save_metadata(metadata)
    path.write_bytes(b"anything")
    assert cache.get(26056, "endfb8.1") == path


def test_a_pinned_hash_passes_downloaded_and_cached(cache, tmp_path):
    first = remote.download_endf("Fe56", tmp_path / "a.endf", sha256=TAPE_SHA.upper())
    second = remote.download_endf("Fe56", tmp_path / "b.endf", sha256=TAPE_SHA)
    assert first.read_bytes() == second.read_bytes() == TAPE


def test_a_wrong_pinned_hash_writes_nothing(cache, tmp_path):
    with pytest.raises(ChecksumMismatchError):
        remote.download_endf("Fe56", tmp_path / "a.endf", sha256="0" * 64)
    assert not (tmp_path / "a.endf").exists()
    with pytest.raises(ChecksumMismatchError):
        remote.download_endf("Fe56", tmp_path / "b.endf", sha256="0" * 64)
    assert not (tmp_path / "b.endf").exists()

"""Files fetched in a browser, put in the cache as if kika had downloaded them.

The IAEA serves its files to browsers only (a Cloudflare challenge since
October 2026), so the app opens the URL and collects the archive from the
Downloads folder. What it collects has to be the right file, complete, and
land in the same cache slot a download would have used.
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import pytest

from kika.endf.remote.browser_download import (
    check_endf_content,
    find_browser_download,
    import_entry,
)
from kika.endf.remote.cache import ENDFCache
from kika.endf.remote.catalog import CatalogEntry

FE56 = CatalogEntry(
    library="jeff4.0",
    library_dir="JEFF-4.0",
    sublib="n",
    filename="n_026-Fe-56_2631.zip",
    size_bytes=0,
    modified="2025-01-01",
    mat=2631,
    z=26,
    a=56,
    isomer=0,
    label="Fe-56",
)


def endf_text(mat: int) -> bytes:
    def record(body: str, mf: int, mt: int) -> str:
        return f"{body:<66}{mat:4d}{mf:2d}{mt:3d}\n"

    tpid = f"{'tape':<66}{1:4d}{0:2d}{0:3d}\n"
    return (tpid + record(" 2.605600+4 5.545400+1", 1, 451) + record("", 1, 451)).encode()


def zipped(content: bytes, member: str = "n_026-Fe-56_2631.endf") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, content)
    return buf.getvalue()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from kika.endf.remote import browser_download

    c = ENDFCache(tmp_path / "cache")
    monkeypatch.setattr(browser_download, "get_cache", lambda: c)
    return c


def test_the_newest_complete_copy_is_found(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    older = downloads / "n_026-Fe-56_2631.zip"
    older.write_bytes(zipped(endf_text(2631)))
    os.utime(older, (time.time() - 100, time.time() - 100))
    newer = downloads / "n_026-Fe-56_2631 (1).zip"
    newer.write_bytes(zipped(endf_text(2631)))
    # a half-written copy is newer still, and is not a zip yet
    (downloads / "n_026-Fe-56_2631 (2).zip").write_bytes(b"PK\x03\x04 partial")
    (downloads / "n_026-Fe-56_2631.zip.crdownload").write_bytes(b"")
    (downloads / "n_092-U-235_9228.zip").write_bytes(zipped(endf_text(9228)))

    assert find_browser_download(FE56, [downloads]) == newer


def test_nothing_downloaded_yet(tmp_path):
    assert find_browser_download(FE56, [tmp_path]) is None


def test_an_archive_lands_in_the_download_slot(tmp_path, cache):
    archive = tmp_path / "n_026-Fe-56_2631.zip"
    archive.write_bytes(zipped(endf_text(2631)))

    cached = import_entry(FE56, archive)

    assert cached == cache.get("26056", "jeff4.0", "n")
    assert cached.read_bytes() == endf_text(2631)


def test_the_bare_endf_file_is_accepted_too(tmp_path, cache):
    endf = tmp_path / "n_026-Fe-56_2631.endf"
    endf.write_bytes(endf_text(2631))
    assert import_entry(FE56, endf).read_bytes() == endf_text(2631)


@pytest.mark.parametrize(
    "content, message",
    [
        (b"<!DOCTYPE html><html><title>Just a moment...</title>", "web page"),
        (b"not an endf file\n" * 5, "does not read as an ENDF-6"),
        (endf_text(9228), "MAT 9228, not MAT 2631"),
    ],
)
def test_the_wrong_file_is_refused_and_the_cache_left_alone(tmp_path, cache, content, message):
    source = tmp_path / "picked.zip"
    source.write_bytes(zipped(content))
    with pytest.raises(ValueError, match=message):
        import_entry(FE56, source)
    assert cache.get("26056", "jeff4.0", "n") is None


def test_an_entry_without_a_mat_is_only_checked_for_shape():
    check_endf_content(endf_text(1), expected_mat=None)

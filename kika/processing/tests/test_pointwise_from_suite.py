"""σ(E) of a model suite keeps each reaction's interpolation laws, as an MF3 section does."""
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.processing import pointwise_cross_sections_from_suite

GNDS = Path(__file__).resolve().parents[2] / "gnds" / "tests" / "data"


def test_a_suite_without_resonance_ranges_gives_its_evaluated_tables_with_their_laws():
    suite = kika.read(str(GNDS / "n-001_H_001.endf.gnds.xml"), covariances=False)
    tables, source = pointwise_cross_sections_from_suite(suite)
    assert "evaluated" in source
    assert {1, 2, 102} <= set(tables)
    for mt in (1, 2, 102):
        form = suite.reactionByENDF_MT(mt).crossSection["eval"]
        energies, values, laws = form.toEndfRegions()
        np.testing.assert_array_equal(tables[mt].energies, energies)
        np.testing.assert_array_equal(tables[mt].values, values)
        assert tables[mt].energy_interpolation == list(laws)


def test_a_suite_with_resonance_ranges_is_reconstructed_under_the_tape_paths_point_budget(monkeypatch):
    import pytest

    import kika.processing.resonances.suite as suite_module
    from kika.endf.reconstruction import DEFAULT_MAX_POINTS

    seen = {}

    def capture(suite, label, options):
        seen["options"] = options
        raise RuntimeError("captured")

    monkeypatch.setattr(suite_module, "reconstruct_suite", capture)
    suite = kika.read(str(GNDS / "micro_fe56.gnds.xml"), covariances=False)
    with pytest.raises(RuntimeError, match="captured"):
        pointwise_cross_sections_from_suite(suite, tolerance=0.002)
    assert seen["options"].max_points == DEFAULT_MAX_POINTS
    assert seen["options"].rtol == 0.002


TAPE = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data" / "micro_fe56_structural.endf"


class Reconstructed(Exception):
    pass


@pytest.fixture(scope="module")
def cached(tmp_path_factory):
    """A GNDS file with resolved resonances, reconstructed once into a fresh cache."""
    root = tmp_path_factory.mktemp("suite_cache")
    gnds = root / "fe56.gnds.xml"
    kika.write(kika.read(str(TAPE)), str(gnds))
    cache = root / "cache"
    first = pointwise_cross_sections_from_suite(kika.read(str(gnds), covariances=False),
                                                tolerance=0.05, cache_dir=cache)
    return gnds, cache, first


@pytest.fixture
def no_reconstruction(monkeypatch):
    import kika.processing.resonances.suite as suite_module

    def refuse(*args, **kwargs):
        raise Reconstructed

    monkeypatch.setattr(suite_module, "reconstruct_suite", refuse)


def test_a_reconstruction_is_read_back_from_disk_bit_for_bit(cached, no_reconstruction):
    gnds, cache, (tables, source) = cached
    assert [p.suffix for p in cache.iterdir()] == [".npz"]
    again, again_source = pointwise_cross_sections_from_suite(
        kika.read(str(gnds), covariances=False), tolerance=0.05, cache_dir=cache)
    assert "reconstruction" in source and again_source == source
    assert list(again) == list(tables)
    for mt, table in tables.items():
        for name in ("energies", "values"):
            mine, theirs = getattr(again[mt], name), getattr(table, name)
            assert mine.dtype == theirs.dtype and mine.tobytes() == theirs.tobytes()
        assert again[mt].energy_interpolation == table.energy_interpolation
        assert all(type(n) is int and type(law) is int for n, law in again[mt].energy_interpolation)


def test_another_tolerance_or_other_file_bytes_is_another_entry(cached, no_reconstruction, tmp_path):
    gnds, cache, _ = cached
    with pytest.raises(Reconstructed):
        pointwise_cross_sections_from_suite(kika.read(str(gnds), covariances=False),
                                            tolerance=0.02, cache_dir=cache)
    edited = tmp_path / gnds.name
    edited.write_bytes(gnds.read_bytes() + b"\n")
    with pytest.raises(Reconstructed):
        pointwise_cross_sections_from_suite(kika.read(str(edited), covariances=False),
                                            tolerance=0.05, cache_dir=cache)


def test_without_a_cache_dir_the_cache_is_not_consulted(cached, no_reconstruction, monkeypatch):
    import kika.processing.njoy_pendf_cache as module

    def untouched(*args, **kwargs):
        raise AssertionError("the cache was consulted")

    monkeypatch.setattr(module, "_sha256_of_file", untouched)
    monkeypatch.setattr(module, "_read_suite_tables", untouched)
    with pytest.raises(Reconstructed):
        pointwise_cross_sections_from_suite(kika.read(str(cached[0]), covariances=False), tolerance=0.05)


def test_a_cache_folder_removed_during_the_reconstruction_is_made_again(cached, monkeypatch, tmp_path):
    import shutil

    import kika.processing.resonances.suite as suite_module

    gnds, _, (tables, _) = cached
    cache = tmp_path / "cache"
    original = suite_module.reconstruct_suite

    def reconstruct_while_the_folder_goes(*args, **kwargs):
        shutil.rmtree(cache, ignore_errors=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(suite_module, "reconstruct_suite", reconstruct_while_the_folder_goes)
    again, _ = pointwise_cross_sections_from_suite(kika.read(str(gnds), covariances=False),
                                                   tolerance=0.05, cache_dir=cache)
    assert list(again) == list(tables)
    assert [p.suffix for p in cache.iterdir()] == [".npz"]


def test_a_cache_that_cannot_be_written_only_warns(cached, tmp_path):
    gnds, _, (tables, _) = cached
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("")
    with pytest.warns(UserWarning, match="not cached"):
        again, _ = pointwise_cross_sections_from_suite(kika.read(str(gnds), covariances=False),
                                                       tolerance=0.05, cache_dir=blocked)
    assert list(again) == list(tables)


def test_whether_a_reconstruction_is_cached_is_told_without_reconstructing(cached, no_reconstruction, tmp_path):
    from kika.processing import suite_reconstruction_cached

    gnds, cache, _ = cached
    suite = kika.read(str(gnds), covariances=False)
    assert suite_reconstruction_cached(suite, cache_dir=cache, tolerance=0.05)
    assert not suite_reconstruction_cached(suite, cache_dir=cache, tolerance=0.02)
    assert not suite_reconstruction_cached(suite, cache_dir=tmp_path / "empty", tolerance=0.05)
    hydrogen = kika.read(str(GNDS / "n-001_H_001.endf.gnds.xml"), covariances=False)
    assert not suite_reconstruction_cached(hydrogen, cache_dir=cache)

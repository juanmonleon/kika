"""The curated TOF metadata file shipped with kika.exfor and its reader.

The package file is the one the Fe-56 evaluation scripts read on the cluster
(``scripts/tof_parameters.py``), with its ``tof``/``energy_resolution``
schema. Until 2026-10 the package carried an older 10-entry copy in a nested
``energy_resolution_input`` schema; the reader still accepts that one for
files passed through ``configure``.
"""

import json

import pytest

from kika.exfor import config, database
from kika.exfor.database import _get_tof_params_for_experiment, _load_tof_metadata


@pytest.fixture
def metadata_file(tmp_path):
    """Point the reader at a throwaway file, and restore the package one after."""

    def _use(payload):
        path = tmp_path / "tof.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        config.configure(tof_metadata_path=str(path))
        _load_tof_metadata(force_reload=True)

    yield _use
    config.reset_config()
    _load_tof_metadata(force_reload=True)


def test_package_file_is_the_unified_schema():
    meta = _load_tof_metadata(force_reload=True)
    assert "_meta" in meta and "schema" in meta["_meta"]
    entries = {k: v for k, v in meta.items() if not k.startswith("_")}
    assert len(entries) >= 99
    assert not any("energy_resolution_input" in v for v in entries.values())


@pytest.mark.parametrize(
    "dataset_id, flight_path_m, time_resolution_ns",
    [
        ("10037024", 2.75, 1.5),     # Boschung 1971
        ("10571002", 40.0, 8.0),     # Kinney, ORELA
        ("10886002", 5.25, 5.0),     # Smith 1980, the 2026-08-26 fix
        ("20743002", 57.0, 5.47),    # Cierjacks, Karlsruhe
    ],
)
def test_package_file_curated_pairs(dataset_id, flight_path_m, time_resolution_ns):
    _load_tof_metadata(force_reload=True)
    params = _get_tof_params_for_experiment(dataset_id)
    assert params == {
        "flight_path_m": flight_path_m,
        "time_resolution_ns": time_resolution_ns,
        "source": "file",
    }


def test_entry_without_a_pair_falls_back_to_default():
    # 30463020 states a timing resolution but no flight path.
    _load_tof_metadata(force_reload=True)
    assert _get_tof_params_for_experiment("30463020")["source"] == "default"
    assert _get_tof_params_for_experiment("not-a-dataset")["source"] == "default"


def test_legacy_nested_schema_is_still_read(metadata_file):
    metadata_file({
        "1": {"energy_resolution_input": {
            "distance": {"value": 8.3}, "time_resolution": {"value": 1.0}}},
        "2": {"energy_resolution_inputs": {
            "distance": {"value": 27.037}, "time_resolution": {"value": 10.0}}},
    })
    assert _get_tof_params_for_experiment("1")["flight_path_m"] == 8.3
    assert _get_tof_params_for_experiment("2")["time_resolution_ns"] == 10.0


def test_tof_block_with_a_null_falls_back(metadata_file):
    metadata_file({"1": {"tof": {"flight_path_m": None, "time_resolution_ns": 3.0}}})
    assert _get_tof_params_for_experiment("1")["source"] == "default"

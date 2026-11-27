"""σ(E) of a model suite keeps each reaction's interpolation laws, as an MF3 section does."""
from pathlib import Path

import numpy as np

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

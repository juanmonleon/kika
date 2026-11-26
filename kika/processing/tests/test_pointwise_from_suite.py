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

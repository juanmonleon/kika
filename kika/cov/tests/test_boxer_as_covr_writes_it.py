"""read_boxer on BOXER files written by NJOY's COVR (``data/njoy2016``, see its NOTICE).

The same U-238 matrices, MT2 and MT64, written once as covariances (matype 3) and once
as correlations (matype 4). COVR writes the correlation MT2 x MT64 before the standard
deviations of MT64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kika.cov import read_boxer

NJOY = Path(__file__).parent / "data" / "njoy2016"
COV = NJOY / "covr_u238_mt2_mt64_cov.boxer"
CORR = NJOY / "covr_u238_mt2_mt64_corr.boxer"
STRUCTURE = {"count_mismatch", "count_defaulted", "record_longer", "index_out_of_range",
             "row_order", "repeated_block", "skipped_line"}


def test_covr_files_have_no_structure_event():
    for path in (COV, CORR):
        events = []
        read_boxer(path, diagnostics=events)
        assert not STRUCTURE & {e.kind for e in events}


def test_correlations_give_the_covariances_within_the_rounding_of_the_two_files():
    events = []
    corr = read_boxer(CORR, diagnostics=events)
    cov = read_boxer(COV)
    sd = {e.where["mt"]: e.evidence["stddev"] for e in events if e.kind == "raw_stddev"}
    pairs = list(zip(cov.reaction_rows, cov.reaction_cols))
    assert list(zip(corr.reaction_rows, corr.reaction_cols)) == pairs == [(2, 2), (2, 64), (64, 64)]
    for (r, c), a, b in zip(pairs, cov.matrices, corr.matrices):
        # rho to 1e-4 (F7.4); C and each sigma to four figures (1P8E10.3), 5e-4 each at most.
        bound = 5.0e-5 * np.outer(sd[r], sd[c]) + 1.5e-3 * np.abs(a)
        assert np.all(np.abs(b - a) <= bound + 1e-30), (r, c)


def test_a_written_correlation_has_a_unit_diagonal_in_its_self_blocks():
    events = []
    read_boxer(CORR, diagnostics=events)
    for e in events:
        if e.kind == "raw_correlation" and e.where["mt"] == e.where["mt1"]:
            live = np.diag(e.evidence["correlation"]) != 0
            np.testing.assert_array_equal(np.diag(e.evidence["correlation"])[live], 1.0)

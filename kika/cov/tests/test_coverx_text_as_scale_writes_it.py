"""read_coverx on text files laid out as SCALE's own covariance libraries are.

SCALE's text libraries (scale.rev05.44groupcov, scale.rev08.56groupcov7.1) give the
NGROUP upper bounds of the groups and no lowest boundary, then the matrix headers.
The real libraries cannot be versioned (RSICC licence); KIKA_TEST_SCALE_TEXT_DIR
points at a folder that holds them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.cov import read_coverx
from kika.cov.parse_covmat import InvalidDataFormatError
from kika.energy_grids import SCALE44

UPPER_EV = [e * 1.0e6 for e in sorted(SCALE44, reverse=True)[:-1]]


def _scale_text(path: Path, upper_ev, matrix: np.ndarray) -> Path:
    n = len(upper_ev)
    lines = [" v7rec ornl-10/2008", f"{n:12d}{2:12d}{1:12d}{1:12d}"]
    lines += ["".join(f"{v:15.7E}" for v in upper_ev[i:i + 5]) for i in range(0, n, 5)]
    lines.append(f"{8016:12d}{2:12d}{8016:12d}{2:12d}{1:12d}")
    flat = matrix.ravel()
    lines += ["".join(f"{v:15.7E}" for v in flat[i:i + 5]) for i in range(0, flat.size, 5)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_grid_of_upper_bounds_gets_its_lowest_boundary_from_the_scale_structure(tmp_path):
    matrix = np.diag(np.linspace(0.01, 0.44, 44))
    events = []
    cov = read_coverx(_scale_text(tmp_path / "s44.txt", UPPER_EV, matrix), diagnostics=events)
    np.testing.assert_allclose(cov.energy_grid, np.array(sorted(SCALE44)) * 1.0e6, rtol=1e-7)
    np.testing.assert_allclose(cov.matrices[0], matrix[::-1, ::-1], rtol=1e-7)
    (event,) = [e for e in events if e.kind == "count_defaulted"]
    assert event.evidence["structure"] == "SCALE44"
    assert event.evidence["emin_ev"] == pytest.approx(1.0e-5, rel=1e-12)


def test_upper_bounds_of_no_known_structure_are_refused(tmp_path):
    odd = [u * 1.01 for u in UPPER_EV]
    with pytest.raises(InvalidDataFormatError, match="no lowest boundary"):
        read_coverx(_scale_text(tmp_path / "odd.txt", odd, np.eye(44)))


SCALE_DIR = Path(os.environ.get("KIKA_TEST_SCALE_TEXT_DIR", "__missing__"))
REAL = [("scale.rev05.44groupcov.txt", 44, 2558), ("scale.rev08.56groupcov7.1.txt", 56, 4563)]


@pytest.mark.parametrize("name, groups, matrices", REAL, ids=[r[0] for r in REAL])
def test_the_scale_text_libraries_are_read_whole(name, groups, matrices):
    path = SCALE_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not in KIKA_TEST_SCALE_TEXT_DIR")
    events = []
    cov = read_coverx(path, diagnostics=events)
    assert (cov.num_groups, cov.num_matrices, len(cov.energy_grid)) == (groups, matrices, groups + 1)
    assert not [e for e in events if e.kind in ("dropped_block", "skipped_line")]

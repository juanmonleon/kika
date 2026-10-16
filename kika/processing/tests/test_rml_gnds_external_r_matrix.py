"""An RML range with an external R-matrix, read from GNDS, goes to the RML
preparation, as it does read from ENDF.

The preparation is chosen by the pairs: with every pair's kinematics, RML;
without, Reich-Moore, which refuses an external R-matrix ("RM external
R-matrix is deferred to RML"). Read from ENDF every pair has them; read from
GNDS they are rebuilt only for a range that needs RML, and an external
R-matrix (KBK) was not among the reasons, so NNDC's Sr-88 (ENDF/B-VIII.1)
could not be processed while its ENDF could. Measured 2026-10-16 once routed:
the PENDF of NNDC's GNDS has the ENDF route's MTs and agrees to 1e-6 (median
8e-8, the GNDS file's digits).

``KIKA_GNDS_LIB`` points at the folder of NNDC's ``n-*.endf.gnds.xml`` files;
by default ``NuclearData/gnds/endfb81`` beside ``KIKA_LIB_TAPES``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.tape


def _library() -> Path:
    stated = os.environ.get("KIKA_GNDS_LIB")
    if stated:
        return Path(stated)
    tapes = os.environ.get("KIKA_LIB_TAPES")
    return Path(tapes).parent / "gnds" / "endfb81" if tapes else Path("missing")


def test_nndc_sr88_external_r_matrix_takes_the_rml_route():
    path = _library() / "n-038_Sr_088.endf.gnds.xml"
    if not path.is_file():
        pytest.skip(f"{path} not found (set KIKA_GNDS_LIB)")
    from kika.endf.process import _context_from_model
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs

    suite, _ = readReactionSuite(Document.parse(path))
    external = [r for r in suite.resonances.resolved
                if any(ch.externalRMatrix is not None or ch.tabulatedBackground is not None
                       for g in r.formalism.spinGroups for ch in g.channels)]
    assert external, "Sr-88 states an external R-matrix"
    resolved = normalize_suite_pairs(suite, _context_from_model(suite)[0])
    for region in resolved.resolved:
        if any(r.domainMin == region.domainMin for r in external):
            assert all(rr.kinematics is not None for rr in region.formalism.resonanceReactions)

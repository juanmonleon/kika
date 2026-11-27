"""An RML charged pair read from GNDS carries the nuclear charge, not the atom's.

GNDS keeps the atom's net charge on the ``nuclide`` (0 in every distributed
file) and the nuclear charge on its ``nucleus``; kika reads the first as
``Nuclide.charge`` and the second as ``Nuclide.Z``. The RML pair normalization
took ``charge``, so NNDC's Cl-35 (ENDF/B-VIII.1) gave its ``H1 + S35`` channel
two neutral particles and ``process_gnds`` refused it ("charged exit MT requires
a repulsive charged pair"), while the same evaluation read from ENDF processed.

With ``Z``, measured 2026-10-16: the PENDF of NNDC's GNDS has the grid of the
ENDF route's in every MT, and its values agree to 1.7e-7 in MT600/103 and
6e-10 in MT1/2 (3.5e-6 at worst, MT720 at 15 MeV: the GNDS file's digits).

NNDC's ``n-*.endf.gnds.xml`` files are read from ``$KIKA_NUCLEAR_DATA/gnds/endfb81``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika._nuclear_data import nuclear_data

pytestmark = pytest.mark.tape


def _library() -> Path:
    return nuclear_data("gnds", "endfb81") or Path("missing")


def test_nndc_cl35_proton_channel_is_coulomb_charged():
    path = _library() / "n-017_Cl_035.endf.gnds.xml"
    if not path.is_file():
        pytest.skip(f"{path} not found (set KIKA_NUCLEAR_DATA)")
    from kika.endf.process import _context_from_model
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs

    suite, _ = readReactionSuite(Document.parse(path))
    assert suite.PoPs.particles["S35"].charge == 0  # the atom's, as NNDC writes it
    resolved = normalize_suite_pairs(suite, _context_from_model(suite)[0])
    (pair,) = [rr for region in resolved.resolved
               for rr in region.formalism.resonanceReactions if rr.label == "H1 + S35"]
    assert (pair.kinematics.particleA.charge, pair.kinematics.particleB.charge) == (1, 16)

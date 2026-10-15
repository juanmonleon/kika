"""MF13 on an MT with no MF3 reaches the processed suite (N-14 MT28 and MT32).

FUDGE drops these photons and they were kept as text, so no ACE of kika had
them: ENDF/B-VIII.1 N-14 lost 8 % of its photon production at 5-20 MeV and
Pu-241, whose fission photons above 100 keV are MF13 on MT3, all of it above
0.1 MeV. They hang on an orphan whose σ is the section's own σ_γ; with no
partials to sum, the processed suite keeps that orphan with σ_γ on the grid,
which the ACE writes as NJOY does (MFTYPE 13) and an OpenMC file cannot hold.
"""
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeReactionSuite
from kika.hdf5.processing import processSuite

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_n14_photons.endf"


@pytest.fixture(scope="module")
def processed():
    suite, _ = decodeReactionSuite(read_endf(str(TAPE)))
    return suite, processSuite(suite, [293.6], endf=TAPE, omit=("urr", "kerma", "damage"),
                               thinning=None, workers=1)


def test_the_orphans_are_kept_with_their_photons(processed):
    _, out = processed
    kept = {o.ENDF_MT: o for o in out.orphanProducts}
    assert sorted(kept) == [28, 32]
    assert [sum(p.pid == "photon" for p in kept[mt].outputChannel.products) for mt in (28, 32)] == [6, 1]


def test_yield_times_sigma_is_the_evaluated_production(processed):
    suite, out = processed
    source = next(o for o in suite.orphanProducts if o.ENDF_MT == 32)
    sigma = source.crossSection.forms["eval"]
    kept = next(o for o in out.orphanProducts if o.ENDF_MT == 32)
    heated = next(iter(kept.crossSection.forms.values()))
    x = np.asarray(sigma.xs)[1:-1]
    assert np.allclose(np.interp(x, heated.xs, heated.ys), np.asarray(sigma.ys)[1:-1], rtol=1e-12)


def test_an_openmc_file_declares_them_lost(processed, tmp_path):
    from kika.hdf5.writer import writeSuite

    _, out = processed
    report = writeSuite(out, tmp_path / "h5")
    assert any("orphan" in m and "28" in m and "32" in m for m in report.losses)

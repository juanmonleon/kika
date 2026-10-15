"""Radionuclide productions (ENDF MF8-10, roadmap E6) through the resonance reconstruction.

A production shares its MT with the reaction that makes it, and the
reconstruction used to project every node of ``suite.productions`` by MT: from
E6 on, every tape with MF9, MF10 or MF8 ``LMF=3`` failed with "ambiguous MT
projection" (D41), and with it ``process_endf`` and the HDF5 processing. And a
production read at ``eval`` through its reference got the MF3 background: Am-241
capture to Am-242m was 0 at thermal (D42).

The witness is the committed Fe-56 slice (MF2, MF3/MT1, 2, 102) with an MF9 and
an MF10 for MT102 spliced in, so the reactions are the ones every other
processing test reconstructs and the productions are the only difference. The
reconstruction is done once, in a module fixture (about 10 s).
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.classes.mf9.base import MF9MT, MF10MT, ProductionState
from kika.endf.utils import format_endf_fend_record
from kika.nuclear_data.model import Interpolation
from kika.nuclear_data.model.cross_section_forms import Reference
from kika.processing.resonances.suite import (_entries, _fingerprint, attach_reconstruction,
                                              reconstruct_suite)

TAPE = Path(__file__).resolve().parents[2] / "endf/tests/data/micro_fe56_structural.endf"
#: Fe-56(n,γ): the ground state of Fe-57 by MF9 (a multiplicity of the MT102 σ),
#: its 14.4 keV level by MF10 (a cross section of its own, log-log).
Q_CAPTURE = 7646150.0
MULTIPLICITY = 0.9


def _spliced(path: Path) -> Path:
    lines = TAPE.read_text().splitlines()
    mat = int(lines[1][66:70])
    mf9 = MF9MT(number=102, _za=26056.0, _awr=55.4545, _mat=mat, states=[
        ProductionState(qm=Q_CAPTURE, qi=Q_CAPTURE, izap=26057, lfs=0, interp=[(2, 2)],
                        x=[1.0e-5, 2.0e7], y=[MULTIPLICITY, MULTIPLICITY])])
    mf10 = MF10MT(number=102, _za=26056.0, _awr=55.4545, _mat=mat, states=[
        ProductionState(qm=Q_CAPTURE, qi=Q_CAPTURE - 14412.95, izap=26057, lfs=1,
                        interp=[(3, 5)], x=[1.0e-5, 1.0e3, 2.0e7], y=[2.0, 0.01, 1.0e-4])])
    block = (str(mf9).splitlines() + [format_endf_fend_record(mat)]
             + str(mf10).splitlines() + [format_endf_fend_record(mat)])
    # Before the first file past MF8 (MF34 here), where ENDF-6 puts MF9-10.
    at = next(i for i, line in enumerate(lines)
              if len(line) >= 75 and int(line[70:72].strip() or 0) > 10)
    path.write_text("\n".join(lines[:at] + block + lines[at:]) + "\n")
    return path


def _suite(path):
    read = kika.read(str(path), covariances=False)
    return read.suite if hasattr(read, "suite") else read


@pytest.fixture(scope="module")
def spliced(tmp_path_factory):
    return _spliced(tmp_path_factory.mktemp("e6") / "fe56_with_mf9_mf10.endf")


@pytest.fixture(scope="module")
def reconstructed(spliced):
    suite = _suite(spliced)
    attach_reconstruction(suite, reconstruct_suite(suite))
    return suite


def test_the_splice_decodes_into_two_productions(spliced):
    suite = _suite(spliced)
    assert sorted(p.label for p in suite.productions) == ["MT102 Fe57", "MT102 Fe57_e1"]


def test_the_reconstruction_sees_the_same_material_with_or_without_productions(spliced):
    """D41. The reconstruction is deterministic in what it is given, so it is
    enough to show that it is given the same thing and the reactions come out
    bit for bit -- without paying for two Reich-Moore reconstructions.

    The fingerprint hashes every input, PoPs included, and PoPs legitimately
    gains the two nuclides made. So: PoPs is the plain one plus exactly those,
    nothing it had is touched, and with PoPs, header and report set equal the
    fingerprint matches -- reactions, resonances and styles are identical.
    """
    from copy import deepcopy

    plain, withProductions = _suite(TAPE), _suite(spliced)
    assert set(_entries(withProductions)) == set(_entries(plain))
    before, after = plain.PoPs.particles, withProductions.PoPs.particles
    assert set(after) - set(before) == {"Fe57", "Fe57_e1"}
    assert all(after[pid] == before[pid] for pid in before)

    normalised = deepcopy(withProductions)
    normalised.PoPs, normalised.report = plain.PoPs, plain.report
    normalised.provenance.headerFields = plain.provenance.headerFields
    assert _fingerprint(normalised, "eval") == _fingerprint(plain, "eval")


def test_a_reference_follows_its_reaction_to_recon(reconstructed):
    suite = reconstructed
    production = suite.productions.byLabel("MT102 Fe57")
    form = production.crossSection["recon"]
    assert isinstance(form, Reference) and "[@label='MT102']" in form.href
    e, sigma = suite.production_cross_section(production, form="recon")
    capture = suite.reactionByENDF_MT(102).crossSection["recon"]
    np.testing.assert_allclose(sigma, MULTIPLICITY * capture.evaluate(e), rtol=1e-12)
    # Thermal capture reaches the product: the resonance part, which MF3 lacks.
    assert float(np.interp(0.0253, e, sigma)) > 1.0


def test_a_reference_at_eval_is_refused_where_resonances_reach(reconstructed):
    """D42: the evaluated MT102 is MF3's background below the resonance range."""
    with pytest.raises(ValueError, match="form='recon'"):
        reconstructed.production_cross_section("MT102 Fe57")


def test_an_mf10_table_is_linearised_as_reconr_does(reconstructed):
    production = reconstructed.productions.byLabel("MT102 Fe57_e1")
    evaluated, processed = production.crossSection["eval"], production.crossSection["recon"]
    assert not isinstance(processed, Reference)
    regions = getattr(processed, "function1ds", [processed])
    assert all(r.interpolation == Interpolation.linlin for r in regions)
    # A table needs no resonances, so it is read at eval as the file states it.
    reconstructed.production_cross_section(production)
    probe = np.geomspace(2.0e-5, 1.0e7, 400)
    np.testing.assert_allclose(processed.evaluate(probe), evaluated.evaluate(probe), rtol=1.1e-3)


def test_process_endf_reconstructs_a_tape_with_productions(spliced):
    """The app's entry, through the stage D41 broke. Broadening runs on the same
    ``_entries`` the fingerprint test above shows unchanged, and costs minutes on
    this slice, so it is not repeated here."""
    from kika.endf import ProcessingPlan, process_endf

    result = process_endf(spliced, ProcessingPlan(temperatures=(), workers=1))
    status = {s.stage: s.status for s in result.stages}
    assert status["reconstruct"] == "done", result.stages

"""Phase 4: the elastic records into the common model.

Acceptance (roadmap §5 Fase 4): σ and the angular distribution are reached
the way they are for ENDF/ACE/GNDS; the partial read is reported; the model
does not import the format. The gate against the evaluation itself — G4NDL
JEFF-4.0 against the JEFF-4.0 tape's MF4 — is in ``test_full_libraries.py``.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import kika.g4ndl as g4ndl
from kika.g4ndl import G4NDLUnsupportedError, IsotopeKey
from kika.g4ndl.decode import decodeElastic, targetId
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.tokens import TokenStream
from kika.nuclear_data.model import (
    AngularTwoBody, Frame, Isotropic2d, Legendre, Regions2d, XYs1d, XYs2d,
)
from kika.nuclear_data.model.enums import INTERPOLATION_TO_ENDF_INT

DATA = Path(__file__).parent / "data"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
G4 = g4ndl.open(DATA / "G4NDL-4.7.1")
SYNTH = g4ndl.open(DATA / "synthetic")


def _angular(suite):
    return suite.reactions[2].outputChannel.products.byPid("n")[0].distribution["eval"]


def _code(function):
    return INTERPOLATION_TO_ENDF_INT[function.interpolation]


# ------------------------------------------------------------------- suite

def test_the_suite_is_an_elastic_channel_with_its_styles():
    suite = JEFF.read("H1")
    assert (suite.projectile, suite.target, suite.evaluation) == ("n", "H1", "JEFF-4.0")
    assert suite.reactions.ENDF_MTs == [2]
    assert suite.styleLabels() == ["eval", "recon"]
    assert suite.styles.chain("recon")[-1].label == "eval"
    channel = suite.reactions[2].outputChannel
    assert channel.Q.value == 0.0 and channel.genre == "twoBody"
    assert [p.pid for p in channel.products] == ["n"]


@pytest.mark.parametrize("lib,target", [(JEFF, "H1"), (JEFF, "C12"), (JEFF, "Co58m1"),
                                        (G4, "Cnat")])
def test_cross_section_is_the_file_under_recon(lib, target):
    record = lib.crossSection(target)
    suite = lib.read(target)
    E, sigma = suite.cross_section(2, form="recon")
    assert E.tolist() == record.energy.tolist() and sigma.tolist() == record.sigma.tolist()
    form = suite.reactions[2].crossSection["recon"]
    assert isinstance(form, XYs1d) and _code(form) == 2


def test_there_is_no_evaluated_cross_section_and_asking_for_one_says_so():
    suite = JEFF.read("H1")
    with pytest.raises(KeyError, match="recon"):
        suite.cross_section(2)
    assert any("does not record which evaluation" in m for m in suite.report.losses)


def test_repeated_energies_stay_and_are_reported():
    suite = JEFF.read("Co58m1")
    E, _ = suite.cross_section(2, form="recon")
    assert int(np.sum(np.diff(E) == 0)) == 1
    assert any("1 repeated energies" in w for w in suite.report.warnings)


@pytest.mark.parametrize("key,pid", [
    (IsotopeKey(26, 56), "Fe56"), (IsotopeKey(6, None), "C"),
    (IsotopeKey(27, 58, 1), "Co58_m1"), (IsotopeKey(1, 1), "H1"),
])
def test_target_ids(key, pid):
    assert targetId(key) == pid


def test_a_metastable_target_is_named_and_its_level_is_reported_unknown():
    suite = JEFF.read("Co58m1")
    assert suite.target == "Co58_m1" and "Co58_m1" in suite.PoPs.particles
    assert any("M=1" in w for w in suite.report.warnings)


# ----------------------------------------------------------------- angular

def test_legendre_gets_a0_and_keeps_the_file_coefficients():
    record = JEFF.elasticFinalState("H1")
    angular = _angular(JEFF.read("H1"))
    assert isinstance(angular, AngularTwoBody) and angular.productFrame == Frame.centerOfMass
    xys2d = angular.angular
    assert isinstance(xys2d, XYs2d) and _code(xys2d) == record.legendre.interpolation.codes[0]
    assert len(xys2d.function1ds) == len(record.legendre)
    for f, r in zip(xys2d.function1ds, record.legendre.records):
        assert isinstance(f, Legendre) and f.outerDomainValue == r.energy
        assert f.coefficients[0] == 1.0
        assert f.coefficients[1:].tolist() == r.coefficients.tolist()


def test_repeated_incident_energies_survive_into_the_model():
    energies = [f.outerDomainValue for f in _angular(JEFF.read("H1")).angular.function1ds]
    assert sum(a == b for a, b in zip(energies, energies[1:])) == 1


def test_tabulated_records_are_xys1d_with_their_mu_interpolation():
    record = JEFF.elasticFinalState("He3")
    xys2d = _angular(JEFF.read("He3")).angular
    for f, r in zip(xys2d.function1ds, record.tabulated.records):
        assert isinstance(f, XYs1d) and f.outerDomainValue == r.energy
        assert f.xs.tolist() == r.mu.tolist() and f.ys.tolist() == r.probability.tolist()
        assert _code(f) == r.interpolation.codes[0]


def test_two_energy_regions_become_regions2d():
    angular = _angular(JEFF.read("N14")).angular
    assert isinstance(angular, Regions2d)
    # N14 is repFlag=3: [Legendre block with two regions, table].
    legendre = angular.function2ds[0]
    assert isinstance(legendre, Regions2d)
    assert [_code(r) for r in legendre.function2ds] == [3, 2]
    assert sum(len(r.function1ds) for r in legendre.function2ds) == 614


def test_mixed_is_legendre_then_table_and_keeps_loglin_in_mu():
    record = JEFF.elasticFinalState("C12")
    angular = _angular(JEFF.read("C12")).angular
    leg, tab = angular.function2ds
    assert all(isinstance(f, Legendre) for f in leg.function1ds)
    assert all(isinstance(f, XYs1d) and _code(f) == 4 for f in tab.function1ds)
    assert leg.function1ds[-1].outerDomainValue == tab.function1ds[0].outerDomainValue \
        == record.transitionEnergy


def test_isotropic_uses_the_frame_geant4_uses():
    d = _angular(SYNTH.read("H1"))
    assert isinstance(d, Isotropic2d) and d.productFrame == Frame.centerOfMass


def test_laboratory_frame_and_tempdep_are_kept_and_reported():
    suite = SYNTH.read("Li6")
    assert _angular(suite).productFrame == Frame.lab
    assert suite.provenance.legendreTempdeps == [0, 1]
    assert suite.provenance.legendreTemperatures == [293.6, 293.6]
    assert any("tempdep" in w for w in suite.report.warnings)


def test_histogram_interpolation_is_refused_with_the_reason():
    cs = parse_cross_section(TokenStream("0 0\n2\n1.0 1.0 2.0 1.0\n"))
    fs = parse_elastic_fs(TokenStream("1 1.0 2\n2\n1 2 1\n0.0 1.0e6 0 1 0.1\n"
                                      "0.0 2.0e6 0 1 0.2\n"))
    with pytest.raises(G4NDLUnsupportedError, match="evaluates it as lin-lin"):
        decodeElastic(cs, fs, IsotopeKey(1, 1))


def test_histogram_in_mu_is_read_lin_lin_and_written_back():
    """D10-3: Geant4 evaluates a code 1 lin-lin, so the model holds lin-lin.

    The report calls it an approximation (ENDF reads a histogram), and the
    declared code is what the encoder writes back.
    """
    from kika.g4ndl.encode import encodeElastic, recordDifferences

    cs = parse_cross_section(TokenStream("0 0\n2\n1.0 1.0 2.0 1.0\n"))
    fs = parse_elastic_fs(TokenStream("2 1.0 2\n1\n1 1 2\n"
                                      "0.0 1.0e6 0 2 1 2 1 -1.0 0.5 1.0 0.5\n"))
    suite, report = decodeElastic(cs, fs, IsotopeKey(1, 1))
    assert any("code 1 in mu read lin-lin" in m for m in report.approximations)
    assert suite.provenance.tabulatedCode1 == {"0": [[2, 1]]}
    table = _angular(suite).angular.function1ds[0]
    assert table.interpolation.value == "lin-lin"
    _, fs2, _ = encodeElastic(suite)
    assert recordDifferences(fs, fs2) == []


# -------------------------------------------------------------- provenance

def test_provenance_names_the_files_and_keeps_the_tokens():
    suite = JEFF.read("C12")
    p = suite.provenance
    assert p.sourceFormat == "g4ndl" and p.libraryName == "JEFF-4.0"
    assert p.bookkeeping == (0, 0) and p.repFlag == 3 and p.frameFlag == 2
    assert p.targetMass == JEFF.elasticFinalState("C12").targetMass
    for path, digest in ((p.crossSectionPath, p.crossSectionSha256),
                         (p.finalStatePath, p.finalStateSha256)):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert suite.reactions[2].provenance is p


def test_a_library_with_other_processes_reports_a_partial_read(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(DATA / "JEFF-4.0", root)
    (root / "ThermalScattering").mkdir()
    (root / "Fission").mkdir()
    (root / "Inelastic").mkdir()
    (root / "Capture").mkdir()
    report = g4ndl.open(root).read("H1").report
    # Inelastic/, Capture/ and Fission/ are read, and hold nothing for H1;
    # ThermalScattering/ is not read.
    assert len(report.unsupported) == 1
    assert "ThermalScattering/" in report.unsupported[0]
    assert "elastic channel (MT2) only" in report.unsupported[0]


def test_reading_does_not_wake_the_model_until_asked():
    code = textwrap.dedent("""
        import sys
        import kika.g4ndl as g
        g.open(sys.argv[1]).crossSection("H1")
        assert not any(m.startswith("kika.nuclear_data.model") for m in sys.modules), "woke"
        g.open(sys.argv[1]).read("H1")
        assert "kika.nuclear_data.model" in sys.modules
    """)
    subprocess.run([sys.executable, "-c", code, str(DATA / "JEFF-4.0")], check=True)

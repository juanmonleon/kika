"""Roadmap E7b: the radioactive decay sublibrary, ENDF ↔ a standalone §12 PoPs ↔ GNDS.

The fixtures are whole ENDF/B-VIII.1 decay files (``kika/endf/tests/data``):
Co-60 (one beta- mode; gamma, beta, e- and x-ray lines), Fe-56 (stable),
Cs-137 (two beta- modes that differ only in RFS, one ending on Ba-137m) and
Rb-93 (beta- and beta-delayed n, with a neutron continuum).

Gates:

- ENDF → model → ENDF byte for byte (here, and ``-m tape`` over all 3 821
  files of ``KIKA_DECAY_TAPES``);
- model → GNDS → model identical;
- ``-m fudge``: the model kika reads from FUDGE's GNDS is the one it decodes
  from the tape (masses aside: FUDGE takes them from the AME table, kika from
  the tape's AWR), and FUDGE's ``toENDF6`` writes the same tape from kika's
  GNDS as from its own.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

import kika
from kika.nuclear_data.model import PoPs
from kika.nuclear_data.model.compare import modelDifferences

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
FIXTURES = ["dec-027_Co_060", "dec-026_Fe_056", "dec-055_Cs_137", "dec-037_Rb_093"]


def _roundTrip(path: Path, folder: Path) -> PoPs:
    pops = kika.read(path)
    assert isinstance(pops, PoPs)
    out = folder / path.name
    kika.write(pops, out, format="endf")
    assert out.read_bytes() == path.read_bytes(), path.name
    kika.write(pops, folder / "decay.xml")
    back = kika.read(folder / "decay.xml")
    assert modelDifferences(pops.particles, back.particles) == []
    assert modelDifferences(pops.aliases, back.aliases) == []
    return pops


@pytest.mark.parametrize("name", FIXTURES)
def test_a_decay_tape_comes_back_byte_for_byte_and_through_gnds(name, tmp_path):
    _roundTrip(DATA / f"{name}.endf", tmp_path)


def test_the_model_names_the_decay():
    co60 = kika.read(DATA / "dec-027_Co_060.endf").particles["Co60"]
    assert co60.halflife.value == pytest.approx(1.663442e8)
    assert co60.halflife.uncertainty.value == pytest.approx(12096.0)
    assert co60.spin.value == 5 and co60.parity == 1
    (mode,) = co60.decayData.decayModes
    assert (mode.mode, mode.probability, mode.Q.value) == ("beta-", 1.0, pytest.approx(2.8239e6))
    assert [p.pid for p in mode.decayPath.decays[0].products] == ["e-", "nu_e-_anti", "Ni60"]
    assert [s.label for s in mode.spectra] == ["gamma", "beta-", "discrete electron", "x-ray"]
    line = next(d for d in mode.spectra[0].discretes if d.energy.value == 1173228.0)
    assert line.intensity.value == pytest.approx(0.9985)
    assert [s.label for s in line.internalConversionCoefficients] == ["total", "K", "L"]
    assert [a.label for a in co60.decayData.averageEnergies] == [
        "lightParticles", "electroMagneticRadiation", "heavyParticles"]

    fe56 = kika.read(DATA / "dec-026_Fe_056.endf").particles["Fe56"]
    assert fe56.halflife == "stable" and fe56.decayData is None

    cs137 = kika.read(DATA / "dec-055_Cs_137.endf")
    assert cs137.aliases["Ba137_m1"].pid == "Ba137_e1"
    modes = list(cs137.particles["Cs137"].decayData.decayModes)
    assert modes[1].decayPath.decays[0].products[-1].pid == "Ba137_m1"
    assert modes[0].spectra == [] and modes[1].spectra   # RTYP=1 twice: FUDGE files by the last

    rb93 = kika.read(DATA / "dec-037_Rb_093.endf").particles["Rb93"]
    betaN = rb93.decayData.decayModes.decayModes[1]
    assert betaN.mode == "beta-,n"
    assert [[p.pid for p in d.products] for d in betaN.decayPath] == [
        ["e-", "nu_e-_anti", "Sr93"], ["n", "Sr92"]]
    neutrons = next(s for m in rb93.decayData.decayModes for s in m.spectra if s.label == "neutron")
    assert neutrons.pid == "n" and len(neutrons.continua) == 1


def test_a_decay_read_from_gnds_writes_a_tape_that_reads_back_the_same(tmp_path):
    """No provenance: the section is built FUDGE's way (one spectrum per
    radiation, FD = 1, ERAV = sum ER*RI) and still carries the same physics."""
    for name in FIXTURES:
        pops = kika.read(DATA / f"{name}.endf")
        kika.write(pops, tmp_path / "d.xml")
        fromGnds = kika.read(tmp_path / "d.xml")
        report = kika.write(fromGnds, tmp_path / "d.endf", format="endf")
        assert any("FUDGE's rule" in w for w in report.warnings)
        again = kika.read(tmp_path / "d.endf")
        assert modelDifferences(fromGnds.particles, again.particles) == [], name


def test_an_edit_to_the_model_is_not_undone_by_the_kept_fields(tmp_path):
    pops = kika.read(DATA / "dec-027_Co_060.endf")
    mode = pops.particles["Co60"].decayData.decayModes.decayModes[0]
    line = mode.spectra[0].discretes[2]
    from kika.nuclear_data.model import PhysicalQuantity
    line.intensity = PhysicalQuantity(value=0.5, unit="")
    kika.write(pops, tmp_path / "e.endf", format="endf")
    back = kika.read(tmp_path / "e.endf").particles["Co60"]
    assert back.decayData.decayModes.decayModes[0].spectra[0].discretes[2].intensity.value == 0.5


@pytest.mark.tape
@pytest.mark.slow
def test_every_decay_evaluation_of_the_library_comes_back(tmp_path):
    root = os.environ.get("KIKA_DECAY_TAPES")
    if not root:
        pytest.skip("KIKA_DECAY_TAPES not set")
    paths = sorted(Path(root).glob("*.endf"))
    if not paths:
        pytest.fail(f"KIKA_DECAY_TAPES={root!r} holds no *.endf file")
    for path in paths:
        _roundTrip(path, tmp_path)


# ----------------------------------------------------------------------
# FUDGE in the loop
# ----------------------------------------------------------------------

_RUNNER = ("import sys;s=sys.stdin.read();exec(s,{'__name__':'oracle'})")
_MARKER = "KIKA-ORACLE-JSON:"


def _fudge(endfText: str, kikaGnds: str = "") -> dict:
    command = os.environ.get("KIKA_FUDGE_PYTHON", "").split()
    if not command:
        pytest.skip("KIKA_FUDGE_PYTHON is not set: no interpreter with FUDGE")
    source = (Path(__file__).parent / "fudge_decay_oracle.py").read_text()
    payload = json.dumps({"endf": endfText, "kika": kikaGnds})
    done = subprocess.run(command + ["-c", _RUNNER], input=f"PAYLOAD = {payload!r}\n" + source,
                          capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    lines = [l for l in done.stdout.splitlines() if l.startswith(_MARKER)]
    if done.returncode != 0 or not lines:
        pytest.fail(f"FUDGE did not answer (exit {done.returncode}):\n{done.stderr[-2000:]}")
    return json.loads(lines[-1][len(_MARKER):])


@pytest.mark.fudge
@pytest.mark.parametrize("name", FIXTURES)
def test_fudge_and_kika_agree_on_a_decay_evaluation(name, tmp_path):
    path = DATA / f"{name}.endf"
    pops = kika.read(path)
    kika.write(pops, tmp_path / "kika.xml")
    oracle = _fudge(path.read_text(), (tmp_path / "kika.xml").read_text())
    (tmp_path / "fudge.xml").write_text(oracle["gnds"])
    theirs = kika.read(tmp_path / "fudge.xml")
    differences = [d for d in modelDifferences(pops.particles, theirs.particles)
                   if not d.endswith("mass.value") and ".mass.value:" not in d]
    assert differences == []
    assert "kikaError" not in oracle, oracle.get("kikaError")
    ours = [l[:66] for l in oracle["endfFromKika"].splitlines()]
    fudge = [l[:66] for l in oracle["endfFromFudge"].splitlines()]
    assert ours == fudge

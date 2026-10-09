"""Roadmap E7c: fission product yields (MF8/MT454, MT459) ↔ §18.4 ``productYields``.

Fixtures (``kika/endf/tests/data``), whole ENDF/B-VIII.1 files: the Cm-242
neutron-induced yields (NSUB=11, one energy, 1 193 products), the Fm-254
spontaneous-fission yields (NSUB=5), and the Pu-244 decay tape, which also
carries spontaneous-fission neutron data (MF1/452) that a PoPs has no node for.

Gates: ENDF → model → ENDF byte for byte and model → GNDS → model identical
(here, and ``-m tape`` over ``KIKA_NFY_TAPES``/``KIKA_SFY_TAPES``: 31 + 9 files);
``-m fudge``: kika reads FUDGE's GNDS as the yields it decodes from the tape,
FUDGE reads kika's spontaneous-fission file and writes the same tape from it as
from its own, and reads the neutron-induced yields from kika's
``fissionFragmentData`` file.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.nuclear_data.model import CUMULATIVE, INDEPENDENT, FissionFragmentData, PoPs
from kika.nuclear_data.model.compare import modelDifferences

from .test_decay_sublibrary import _fudge, _roundTrip

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
FIXTURES = ["nfy-096_Cm_242", "sfy-100_Fm_254", "dec-094_Pu_244"]


@pytest.mark.parametrize("name", FIXTURES)
def test_a_yield_tape_comes_back_byte_for_byte_and_through_gnds(name, tmp_path):
    _roundTrip(DATA / f"{name}.endf", tmp_path)


def _yieldOf(pops):
    (nuclide,) = [p for p in pops.particles.values() if getattr(p, "fissionFragmentData", None)]
    (productYield,) = nuclide.fissionFragmentData.productYields
    return nuclide, productYield


def test_neutron_induced_yields_are_listed_per_incident_energy():
    nuclide, productYield = _yieldOf(kika.read(DATA / "nfy-096_Cm_242.endf"))
    assert nuclide.id == "Cm242"
    assert [e.label for e in productYield.elapsedTimes] == [INDEPENDENT, CUMULATIVE]
    assert len(productYield.nuclides) == 1193 and "Cu68_m1" in productYield.nuclides
    independent = productYield.elapsedTime(INDEPENDENT)
    assert independent.time.value == 0.0 and independent.yields is None
    (energy,) = independent.incidentEnergies
    assert energy.energy.value == 5e5 and energy.yields.nuclides is None
    assert sum(energy.yields.values) == pytest.approx(2.0, rel=0.02)
    assert productYield.elapsedTime(CUMULATIVE).time == "unspecified"


def test_spontaneous_yields_sit_under_the_elapsed_time():
    nuclide, productYield = _yieldOf(kika.read(DATA / "sfy-100_Fm_254.endf"))
    assert nuclide.id == "Fm254" and nuclide.halflife == "unstable"
    for elapsed in productYield.elapsedTimes:
        assert elapsed.incidentEnergies == [] and elapsed.yields is not None
        assert len(elapsed.yields.values) == len(productYield.nuclides)


def test_a_decay_tape_keeps_what_a_pops_cannot_hold(tmp_path):
    pops = kika.read(DATA / "dec-094_Pu_244.endf")
    assert any("MF1/MT452" in entry for entry in pops.report.unsupported)
    kika.write(pops, tmp_path / "d.xml")
    assert "<reactionSuite" not in (tmp_path / "d.xml").read_text()
    kika.write(pops, tmp_path / "d.endf", format="endf")
    assert " 1452" in (tmp_path / "d.endf").read_text()


def test_the_yields_alone_are_fudges_fissionFragmentData_file(tmp_path):
    nuclide, productYield = _yieldOf(kika.read(DATA / "nfy-096_Cm_242.endf"))
    kika.write(nuclide.fissionFragmentData, tmp_path / "y.xml")
    assert (tmp_path / "y.xml").read_text().splitlines()[1].startswith("<fissionFragmentData")
    back = kika.read(tmp_path / "y.xml")
    assert isinstance(back, FissionFragmentData)
    assert modelDifferences(back.productYields, [productYield]) == []


def _library(env):
    root = os.environ.get(env)
    if not root:
        pytest.skip(f"{env} not set")
    paths = sorted(Path(root).glob("*.endf"))
    if not paths:
        pytest.fail(f"{env}={root!r} holds no *.endf file")
    return paths


@pytest.mark.tape
@pytest.mark.slow
@pytest.mark.parametrize("env", ["KIKA_NFY_TAPES", "KIKA_SFY_TAPES"])
def test_every_yield_evaluation_of_the_library_comes_back(env, tmp_path):
    for path in _library(env):
        _roundTrip(path, tmp_path)


@pytest.mark.tape
def test_u235_lists_its_products_per_energy():
    root = os.environ.get("KIKA_NFY_TAPES")
    if not root or not (Path(root) / "nfy-092_U_235.endf").exists():
        pytest.skip("KIKA_NFY_TAPES with nfy-092_U_235.endf not set")
    _, productYield = _yieldOf(kika.read(Path(root) / "nfy-092_U_235.endf"))
    assert productYield.nuclides is None
    energies = productYield.elapsedTime(INDEPENDENT).incidentEnergies
    assert [len(e.yields.nuclides) for e in energies] == [1247, 1260, 1237]


# ----------------------------------------------------------------------
# FUDGE in the loop
# ----------------------------------------------------------------------

def _sameYields(ours: dict, theirs: dict) -> None:
    """Two oracle dumps: names and values exact, variances to FUDGE's 12 digits."""
    assert set(ours) == set(theirs)
    for label in ours:
        assert len(ours[label]) == len(theirs[label])
        for (e1, a), (e2, b) in zip(ours[label], theirs[label]):
            assert e1 == e2 and a["nuclides"] == b["nuclides"] and a["values"] == b["values"]
            np.testing.assert_allclose(a["variances"], b["variances"], rtol=1e-11, atol=0)


def _kikaDump(productYield) -> dict:
    out = {}
    for elapsed in productYield.elapsedTimes:
        rows = []
        blocks = ([(ie.energy.value, ie.yields) for ie in elapsed.incidentEnergies]
                  if elapsed.incidentEnergies else [(None, elapsed.yields)])
        for energy, yields in blocks:
            rows.append([energy, {"nuclides": yields.nuclides or productYield.nuclides,
                                  "values": list(yields.values),
                                  "variances": list(yields.uncertainty or [])}])
        out[elapsed.label] = rows
    return out


@pytest.mark.fudge
def test_fudge_and_kika_agree_on_spontaneous_yields(tmp_path):
    path = DATA / "sfy-100_Fm_254.endf"
    pops = kika.read(path)
    kika.write(pops, tmp_path / "kika.xml")
    oracle = _fudge(path.read_text(), (tmp_path / "kika.xml").read_text())
    _sameYields(_kikaDump(_yieldOf(pops)[1]), oracle["yieldsFromFudge"])
    _sameYields(oracle["yieldsFromKika"], oracle["yieldsFromFudge"])
    ours = [l[:66] for l in oracle["endfFromKika"].splitlines()]
    assert ours == [l[:66] for l in oracle["endfFromFudge"].splitlines()]


@pytest.mark.fudge
def test_fudge_and_kika_agree_on_neutron_induced_yields(tmp_path):
    path = DATA / "nfy-096_Cm_242.endf"
    pops = kika.read(path)
    nuclide, productYield = _yieldOf(pops)
    kika.write(nuclide.fissionFragmentData, tmp_path / "kika.xml")
    oracle = _fudge(path.read_text(), (tmp_path / "kika.xml").read_text())
    _sameYields(_kikaDump(productYield), oracle["yieldsFromFudge"])
    _sameYields(oracle["yieldsFromKika"], oracle["yieldsFromFudge"])
    (tmp_path / "fudge.xml").write_text(oracle["gnds"])
    _sameYields(_kikaDump(kika.read(tmp_path / "fudge.xml").productYields[0]),
                oracle["yieldsFromFudge"])


@pytest.mark.fudge
def test_fudge_does_not_read_induced_yields_under_a_pops_nuclide(tmp_path):
    """The named limit: FUDGE 6.14's PoPs reader refuses ``incidentEnergies``
    in an ``elapsedTime`` that ``gnds.xsd`` admits. When it stops refusing,
    this fails, and kika's PoPs file becomes FUDGE-readable as it stands."""
    path = DATA / "nfy-096_Cm_242.endf"
    kika.write(kika.read(path), tmp_path / "kika.xml")
    oracle = _fudge(path.read_text(), (tmp_path / "kika.xml").read_text())
    assert "incidentEnergies" in oracle.get("KikaError", "")

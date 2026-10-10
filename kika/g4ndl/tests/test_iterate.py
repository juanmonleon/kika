"""The correction loop (roadmap G4NDL §0bis, Fase 14) on the JEFF-4.0 fixtures.

The "experiment" is C-12's own elastic σ read through a TOF setup and scaled by
a smooth 5 % bump; the transport hole hands the same measurement back, so the
loop must stop at its second turn with nothing left to move.
"""
from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

import kika.g4ndl as g4ndl
from kika.endf.dcs import TofResolution
from kika.g4ndl.iterate import ITERATION_MANIFEST, LoopMeasurement, iterate_once, run_loop
from kika.nuclear_data.forward import ElasticView, ForwardSetup, forward_sigma

DATA = Path(__file__).parent / "data"
SETUP = ForwardSetup(tof=TofResolution(flight_path_m=27.037, delta_t_ns=3.5,
                                       delta_t_is_fwhm=True, min_sigma_e_kev=0.0))
E = np.linspace(3.0e6, 4.0e6, 40)
BINS = [(x - 5e3, x + 5e3) for x in E]


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    shutil.copytree(DATA / "JEFF-4.0", root)
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)
    return root


@pytest.fixture
def evaluation(base):
    return g4ndl.open(base).read("C12", processes=["elastic"])


@pytest.fixture
def measurement(evaluation):
    sig = forward_sigma(ElasticView.from_suite(evaluation), E, SETUP, BINS)
    bump = 1.0 + 0.05 * np.exp(-0.5 * ((E - 3.5e6) / 2.0e5) ** 2)
    return LoopMeasurement(E, sig * bump, 0.01 * sig * bump, BINS, version="raw")


def _tree(root: Path):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def test_one_turn_writes_a_library_and_its_manifest(tmp_path, base, evaluation, measurement):
    before = _tree(base)
    rec = iterate_once(evaluation, measurement, SETUP, base, tmp_path / "loop", 0,
                       evaluation_label="JEFF-4.0 fixture C12", notes={"dt": "assumed"})
    assert _tree(base) == before
    assert rec.directory.name == "iter_00" and rec.change is None and not rec.converged
    man = json.loads((rec.directory / ITERATION_MANIFEST).read_text(encoding="utf-8"))
    assert man["measurement"]["digest"] == measurement.digest()
    assert man["notes"] == {"dt": "assumed"} and man["angular"] is None
    assert sorted(man["files"]["replaced"]) == ["Elastic/CrossSection/6_12_Carbon.z",
                                                "Elastic/FS/6_12_Carbon.z"]
    assert man["read_back"]["sigma_max_rel"] < 1e-12
    # the library Geant4 would read carries the correction
    back = ElasticView.from_suite(g4ndl.open(rec.library).read("C12", processes=["elastic"]))
    ratio = back.sigma([3.5e6])[0] / ElasticView.from_suite(evaluation).sigma([3.5e6])[0]
    assert ratio == pytest.approx(1.05, abs=0.01)
    with pytest.raises(FileExistsError, match="never overwritten"):
        iterate_once(evaluation, measurement, SETUP, base, tmp_path / "loop", 0)


def test_the_loop_stops_when_the_measurement_stops_moving(tmp_path, base, evaluation,
                                                          measurement):
    seen = []

    def transport(library, record):
        seen.append(library)
        return LoopMeasurement(measurement.energies_ev, measurement.sigma,
                               measurement.sigma_err, measurement.bins_ev,
                               version=f"corrected with {record.directory.name}")

    recs = run_loop(evaluation, measurement, SETUP, base, tmp_path / "loop", transport,
                    max_iterations=4)
    assert [r.directory.name for r in recs] == ["iter_00", "iter_01"]
    assert recs[-1].converged and recs[-1].change == pytest.approx(0.0, abs=1e-12)
    assert seen == [recs[0].library]
    man = json.loads((recs[1].directory / ITERATION_MANIFEST).read_text(encoding="utf-8"))
    assert man["measurement"]["version"] == "corrected with iter_00" and man["converged"]

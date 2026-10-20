"""Photo-atomic (NSUB=3) and atomic-relaxation (NSUB=6) tapes ↔ the model.

The committed micro-tapes of iron are read into the model and must say what
the tape says (each MF23 section a reaction or a sum, MF27 as the coherent and
incoherent forms, MF28 as the element's configurations and transitions), and
written back byte for byte, spelling included: the B-VIII.1 photon
sublibraries write ``26000.0000`` and a left-aligned ``0.0``, which the
provenance keeps.

With the ENDF/B-VIII.1 sublibraries at ``$KIKA_PHOTON_ENDF`` (or
``../NuclearData`` beside the repo), every one of the 100 photoat and 100
atomic_relax tapes must come back byte for byte as well.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.nuclear_data.model import (ATOMIC_INTERACTION, ChemicalElement, CoherentPhotonScattering,
                                     CrossSectionSum, IncoherentPhotonScattering, PoPs, ReactionSuite)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
PHOTOAT = DATA / "micro_fe_photoatomic.endf"
RELAX = DATA / "micro_fe_relaxation.endf"


def _same_bytes(source: Path, written: Path) -> bool:
    return source.read_bytes().replace(b"\r\n", b"\n") == written.read_bytes()


@pytest.fixture(scope="module")
def photoat():
    return kika.read(str(PHOTOAT))


@pytest.fixture(scope="module")
def relax():
    return kika.read(str(RELAX))


def test_photoatomic_tape_is_an_atomic_reaction_suite(photoat):
    from kika.endf.read_endf import read_endf

    assert isinstance(photoat, ReactionSuite)
    assert (photoat.projectile, photoat.target, photoat.interaction) == ("photon", "Fe", ATOMIC_INTERACTION)
    tape = read_endf(str(PHOTOAT))
    mts = sorted(tape.mf[23].mt)
    modelled = sorted([r.ENDF_MT for r in photoat.reactions] + [r.ENDF_MT for r in photoat.sums])
    assert modelled == mts
    for mt in (501, 516, 522):
        if mt in mts:
            assert isinstance(photoat.sums.byENDF_MT(mt) if hasattr(photoat.sums, "byENDF_MT") else
                              next(r for r in photoat.sums if r.ENDF_MT == mt), CrossSectionSum)
    for r in list(photoat.reactions) + list(photoat.sums):
        x, y, _ = r.crossSection["eval"].toEndfRegions()
        np.testing.assert_array_equal(x, tape.mf[23].mt[r.ENDF_MT].x)
        np.testing.assert_array_equal(y, tape.mf[23].mt[r.ENDF_MT].y)


def test_form_factors_are_the_scattering_forms(photoat):
    from kika.endf.read_endf import read_endf

    tape = read_endf(str(PHOTOAT))
    coherent = next(r for r in photoat.reactions if r.ENDF_MT == 502)
    incoherent = next(r for r in photoat.reactions if r.ENDF_MT == 504)
    c = coherent.doubleDifferentialCrossSection["eval"]
    i = incoherent.doubleDifferentialCrossSection["eval"]
    assert isinstance(c, CoherentPhotonScattering) and isinstance(i, IncoherentPhotonScattering)
    np.testing.assert_array_equal(c.formFactor.toEndfRegions()[1], tape.mf[27].mt[502].y)
    np.testing.assert_array_equal(i.scatteringFactor.toEndfRegions()[1], tape.mf[27].mt[504].y)
    # ENDF-6 §27: MT505 is the imaginary anomalous factor f'' and MT506 the real one f'
    if 505 in tape.mf[27].mt:
        np.testing.assert_array_equal(c.imaginaryAnomalousFactor.toEndfRegions()[1], tape.mf[27].mt[505].y)
    if 506 in tape.mf[27].mt:
        np.testing.assert_array_equal(c.realAnomalousFactor.toEndfRegions()[1], tape.mf[27].mt[506].y)
    assert "1/Ang" in [a.unit for a in c.formFactor.axes.axes]


def test_relaxation_tape_is_an_element_in_pops(relax):
    from kika.endf.read_endf import read_endf

    assert isinstance(relax, PoPs)
    element = relax.chemicalElements["Fe"]
    assert isinstance(element, ChemicalElement) and element.Z == 26
    tape = read_endf(str(RELAX)).mf[28].mt[533]
    assert [c.designator for c in element.atomic.configurations] == [s.subi for s in tape.subshells]
    k = element.atomic["1s1/2"]
    shell = tape.subshell(1)
    assert k.bindingEnergy.value == shell.binding_energy and k.electronNumber == shell.electrons
    modes = k.decayData.decayModes.decayModes
    assert [m.probability for m in modes] == [t.probability for t in shell.transitions]
    first = modes[0].decayPath.decays[0].products
    assert [p.pid for p in first][0] in ("photon", "e-")


@pytest.mark.parametrize("source", [PHOTOAT, RELAX], ids=["photoat", "relax"])
def test_micro_tapes_write_back_byte_for_byte(source, tmp_path):
    out = tmp_path / source.name
    kika.write(kika.read(str(source)), str(out), format="endf")
    assert _same_bytes(source, out)


def test_an_edit_reaches_the_tape(photoat, tmp_path):
    import copy

    from kika.endf.read_endf import read_endf

    edited = copy.deepcopy(photoat)
    coherent = next(r for r in edited.reactions if r.ENDF_MT == 502)
    form = coherent.crossSection["eval"]
    form.ys = np.asarray(form.ys) * 2.0
    out = tmp_path / "edited.endf"
    kika.write(edited, str(out), format="endf")
    np.testing.assert_allclose(read_endf(str(out)).mf[23].mt[502].y,
                               2.0 * read_endf(str(PHOTOAT)).mf[23].mt[502].y, rtol=1e-6)


def _library():
    root = os.environ.get("KIKA_PHOTON_ENDF")
    root = Path(root) if root else Path(__file__).resolve().parents[6] / "NuclearData"
    tapes = sorted((root / "photoat" / "endfb81").glob("*.endf")) + \
        sorted((root / "atomic_relax" / "endfb81").glob("*.endf"))
    return tapes


@pytest.mark.parametrize("source", _library() or [None], ids=lambda p: p.name if p else "none")
def test_b81_photon_sublibraries_write_back(source, tmp_path):
    if source is None:
        pytest.skip("no ENDF/B-VIII.1 photoat/atomic_relax ($KIKA_PHOTON_ENDF or ../NuclearData)")
    out = tmp_path / source.name
    kika.write(kika.read(str(source)), str(out), format="endf")
    assert _same_bytes(source, out)

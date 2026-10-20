"""Electro-atomic tapes (NSUB=113: MF23 + MF26) ↔ the model.

MF26's records are MF6's plus LAW=8, the energy transfer. The committed micro
tape (iron, cut to one ionised subshell) is read into the model as FUDGE writes
the electro-atomic GNDS (the large-angle elastic an incomplete reaction with an
``angularTwoBody`` electron; bremsstrahlung and ionisation ``uncorrelated``
spectra; the energy the electron keeps after excitation and bremsstrahlung as
its ``averageProductEnergy``, E − ET) and written back byte for byte.

With the ENDF/B-VIII.1 electrons sublibrary at ``$KIKA_PHOTON_ENDF`` (or
``../NuclearData`` beside the repo), each of the 100 tapes must come back byte
for byte, and its reactions, channels and products must be the ones FUDGE's
GNDS translation has, when that is at ``../NuclearData/gnds/endfb81/electrons``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.read_endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
MICRO = DATA / "micro_fe_electrons.endf"


@pytest.fixture(scope="module")
def suite():
    return kika.read(str(MICRO))


def test_mf26_reads_every_record():
    tape = read_endf(str(MICRO))
    laws = {mt: [p.law for p in s.products] for mt, s in tape.mf[26].mt.items()}
    assert laws == {525: [2], 527: [1, 8], 528: [8], 534: [1]}


def test_the_model_is_fudges(suite):
    assert (suite.projectile, suite.target, suite.interaction) == ("e-", "Fe", "atomic")
    assert sorted(suite.PoPs.particles) == ["e-", "photon"]
    assert [r.ENDF_MT for r in suite.incompleteReactions] == [525]
    assert sorted(r.ENDF_MT for r in suite.sums) == [501, 522]
    elastic = list(suite.incompleteReactions)[0].outputChannel.products[0]
    assert type(elastic.distribution["eval"]).__name__ == "AngularTwoBody"
    brem = next(r for r in suite.reactions if r.ENDF_MT == 527).outputChannel.products
    assert [(p.pid, p.label) for p in brem] == [("photon", "photon"), ("e-", "e-"), ("Fe", "Fe")]
    assert type(brem[0].distribution["eval"]).__name__ == "Uncorrelated"
    ion = next(r for r in suite.reactions if r.ENDF_MT == 534).outputChannel
    assert [(p.pid, p.label) for p in ion.products] == [("e-", "e-"), ("e-", "e-__a"), ("Fe{1s1/2}", "Fe{1s1/2}")]
    assert ion.Q.value == -next(r for r in suite.reactions if r.ENDF_MT == 534).crossSection["eval"].toEndfRegions()[0][0]


def test_energy_transfer_is_the_average_the_electron_keeps(suite):
    tape = read_endf(str(MICRO)).mf[26].mt[528].products[0].law_data
    ape = next(r for r in suite.reactions if r.ENDF_MT == 528).outputChannel.products[0].averageProductEnergy["eval"]
    np.testing.assert_array_equal(ape.xs, tape.energies)
    # E - (E - ET) gives ET back to the rounding of E, not of ET
    e = np.asarray(ape.xs)
    assert np.all(np.abs((e - np.asarray(ape.ys)) - np.asarray(tape.transfer)) <= 4 * np.finfo(float).eps * e)


def test_micro_tape_writes_back_byte_for_byte(suite, tmp_path):
    out = tmp_path / MICRO.name
    kika.write(suite, str(out), format="endf")
    assert out.read_bytes() == MICRO.read_bytes().replace(b"\r\n", b"\n")


def _library():
    root = os.environ.get("KIKA_PHOTON_ENDF")
    root = Path(root) if root else Path(__file__).resolve().parents[6] / "NuclearData"
    return sorted((root / "electrons" / "endfb81").glob("e-*.endf"))


@pytest.mark.parametrize("source", _library() or [None], ids=lambda p: p.name if p else "none")
def test_b81_electrons_write_back_and_match_fudge(source, tmp_path):
    if source is None:
        pytest.skip("no ENDF/B-VIII.1 electrons sublibrary ($KIKA_PHOTON_ENDF or ../NuclearData)")
    model = kika.read(str(source))
    out = tmp_path / source.name
    kika.write(model, str(out), format="endf")
    assert out.read_bytes() == source.read_bytes().replace(b"\r\n", b"\n")
    gnds = source.parents[2] / "gnds" / "endfb81" / "electrons" / f"{source.name}.gnds.xml"
    if not gnds.exists():
        return
    fudge = kika.read(str(gnds))
    for container in ("reactions", "sums", "incompleteReactions"):
        mine = {r.ENDF_MT: r for r in getattr(model, container)}
        theirs = {r.ENDF_MT: r for r in getattr(fudge, container)}
        assert set(mine) == set(theirs)
        for mt, r in mine.items():
            assert r.label == theirs[mt].label
            if container == "sums":
                continue
            a, b = r.outputChannel, theirs[mt].outputChannel
            assert (a.genre, a.process, a.Q.value) == (b.genre, b.process, b.Q.value)
            assert [(p.pid, p.label) for p in a.products] == [(p.pid, p.label) for p in b.products]
            assert [type(p.distribution["eval"]).__name__ for p in a.products] == \
                   [type(p.distribution["eval"]).__name__ for p in b.products]

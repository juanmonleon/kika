"""A continuum's (and MT103-107's) Q is its QM, its QI a ``QI=`` flag (D44/D45).

FUDGE writes these reactions that way in every one of NNDC's 558 files: their
residual is in no particular level, so the channel's Q is the ground state's.
The ENDF decoder used to keep QI as the Q, so the model of a tape and the model
of NNDC's GNDS of it meant different things by the same Q (Fe-56 MT91: -2.334
MeV against 0), kika's own GNDS lost an evaluator's QM that disagrees with its
levels (Fe-56 MT649, 400 eV: D44), and the heating of any GNDS evaluation, which
had no MF3 header to read QM from, balanced every level with its QI (Fe-56 by
GNDS: KERMA x21 above MT51's threshold: D45).

No micro-tape states a flagged MT with QI != QM, so the first tests edit
B-11's MT103 QI by 100 keV. The tape tests (``KIKA_LIB_TAPES``) compare the
model read from ENDF with the model read from NNDC's GNDS.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import kika
from kika.endf.model_adapter.derive.reactions import reactionQValues
from kika.nuclear_data.model.endf_conversion import QI_FLAGGED_MTS, EndfConversionFlags, reactionQI

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
QM, QI = -1.07237e7, -1.08237e7


def _mt103Lines(path: Path):
    return [line for line in path.read_text().splitlines()
            if line[70:72] == " 3" and line[72:75] == "103"]


@pytest.fixture
def tape(tmp_path) -> Path:
    """B-11 with MT103's QI 100 keV below its QM."""
    lines = (DATA / "micro_b11_photons.endf").read_text().splitlines(keepends=True)
    head = next(i for i, line in enumerate(lines) if line[70:72] == " 3" and line[72:75] == "103")
    record = lines[head + 1]
    assert record[:22] == "-1.072370+7-1.072370+7", record
    lines[head + 1] = record[:11] + "-1.082370+7" + record[22:]
    path = tmp_path / "b11_qi.endf"
    path.write_text("".join(lines))
    return path


def _mt103(suite):
    return suite.reactions.byENDF_MT(103)


def test_the_model_q_is_qm_and_the_qi_is_flagged(tape):
    suite = kika.read(str(tape), format="endf", covariances=False)
    reaction = _mt103(suite)
    assert reaction.outputChannel.Q.value == QM
    assert reactionQI(reaction, suite) == QI
    assert reaction.provenance.headerFields["qi"] == QI
    assert reactionQValues(suite)[103] == (QI, QM)


def test_the_tape_writes_back_its_qm_and_qi(tape, tmp_path):
    """MF3's header comes back with both Q values, and nothing else in MT103
    moves against what kika writes of the unedited tape."""
    written = {}
    for name, source in (("edited", tape), ("original", DATA / "micro_b11_photons.endf")):
        out = tmp_path / f"{name}.endf"
        kika.write(kika.read(str(source), format="endf", covariances=False), out, format="endf")
        written[name] = _mt103Lines(out)
    edited, original = written["edited"], written["original"]
    assert edited[1][:22] == "-1.072370+7-1.082370+7"
    assert edited[1][22:] == original[1][22:]
    assert edited[:1] + edited[2:] == original[:1] + original[2:]


def test_kikas_gnds_keeps_the_evaluators_qm(tape, tmp_path):
    suite = kika.read(str(tape), format="endf", covariances=False)
    path = tmp_path / "b11.gnds.xml"
    kika.write(suite, path, format="gnds")
    back = kika.read(str(path), format="gnds", covariances=False)
    reaction = _mt103(back)
    assert reaction.outputChannel.Q.value == QM
    assert reactionQI(reaction, back) == QI
    assert reactionQValues(back)[103] == (QI, QM)


def test_every_flagged_mt_is_flagged_even_when_qi_is_qm():
    """FUDGE's rule, 2 041 of 2 041 in NNDC's files: N-14's MT103-107 state QI = QM."""
    suite = kika.read(str(DATA / "micro_n14_photons.endf"), format="endf", covariances=False)
    flags = EndfConversionFlags.of(suite)
    flagged = [r for r in suite.reactions if r.ENDF_MT is not None and int(r.ENDF_MT) in QI_FLAGGED_MTS]
    assert flagged
    for reaction in flagged:
        from kika.nuclear_data.model.endf_conversion import reactionHref
        assert flags.qiFor(reactionHref(reaction)) == reaction.outputChannel.Q.value
        assert "qi" not in reaction.provenance.headerFields


def _library(kind: str) -> Path:
    tapes = os.environ.get("KIKA_LIB_TAPES")
    if kind == "endf":
        return Path(tapes) / "endfb81" if tapes else Path("missing")
    stated = os.environ.get("KIKA_GNDS_LIB")
    return Path(stated) if stated else (Path(tapes).parent / "gnds" / "endfb81" if tapes else Path("missing"))


@pytest.mark.tape
@pytest.mark.parametrize("name", ["n-026_Fe_056", "n-092_U_235", "n-017_Cl_035"])
def test_endf_and_nndcs_gnds_give_the_same_q_values(name):
    """What the heating and every Monte Carlo writer read, from either source."""
    endf = _library("endf") / f"{name}.endf"
    gnds = _library("gnds") / f"{name}.endf.gnds.xml"
    if not (endf.is_file() and gnds.is_file()):
        pytest.skip(f"{endf} or {gnds} not found (set KIKA_LIB_TAPES)")
    a = kika.read(str(endf), format="endf", covariances=False)
    b = kika.read(str(gnds), format="gnds", covariances=False)
    qa, qb = reactionQValues(a), reactionQValues(b)
    common = sorted(set(qa) & set(qb))
    assert {91, 649} <= set(common) or name == "n-017_Cl_035"
    for mt in common:
        assert qa[mt] == pytest.approx(qb[mt], rel=1e-12, abs=1e-6), mt
    for reaction in a.reactions:
        other = b.reactions.byENDF_MT(reaction.ENDF_MT) if reaction.ENDF_MT is not None else None
        # Fission's GNDS Q is not MF3's (U-235: 181.24 against 194.08 MeV); both
        # tables above give MF1/458's release, as FUDGE writes it back.
        if other is not None and int(reaction.ENDF_MT) not in (18, 19, 20, 21, 38):
            assert reaction.outputChannel.Q.value == other.outputChannel.Q.value, reaction.ENDF_MT

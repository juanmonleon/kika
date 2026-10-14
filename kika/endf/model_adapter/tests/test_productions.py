"""MF8/MF9/MF10 ↔ ``reactionSuite/productions`` (roadmap E6).

What is pinned, on the two committed activation cuts (JENDL-5 Co-59: LMF=3, 9
and 10 on one tape; ENDF/B-VIII.1 Am-241: MF9 with MF8's ``NO=0`` decay LISTs):

* the decode: one production per final state, with a ``reference`` to the
  reaction's σ for MF9 and LMF=3 and the MF10 table otherwise, the level in
  PoPs and the isomers aliased;
* the tape fixed point, and more than it: MF8, MF9 and MF10 come back as the
  source's text, decay LISTs included;
* that an edit reaches the file (MF9 rewritten, MF8 rebuilt only when the set
  of states changes) and what is not modelled (LMF=6, MF9 without MF3) still
  travels as text;
* GNDS: through kika's GNDS and back to ENDF, MF9 and MF10 are the source's
  and MF8 is derived with NO=1;
* :meth:`ReactionSuite.production_cross_section`, the format-agnostic σ.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.productions import (ACTIVATION_MF8_KEY,
                                                 ACTIVATION_VERBATIM_KEY,
                                                 productionLabel)
from kika.endf.read_endf import read_endf
from kika.endf.utils import parse_endf_id
from kika.endf.writers import remove_sections
from kika.endf.writers.assemble import writeEndfTape
from kika.nuclear_data.model import EVAL_LABEL
from kika.nuclear_data.model.cross_section_forms import Reference
from kika.nuclear_data.model.functions.simple import Constant1d

from .test_endf_tape_fixed_point import _walk

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
CO59 = DATA / "micro_co59_activation.endf"
AM241 = DATA / "micro_am241_activation.endf"


def _sections(path, files=(8, 9, 10)):
    """``{(MF, MT): [the 66 data columns of each record]}``; IDs and sequence aside."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf in files and mt:
            out[(mf, mt)].append(line[:66])
    return dict(out)


def _decode(path):
    suite, report = decodeReactionSuite(read_endf(str(path)))
    return suite, report


def _write(suite, tmp_path, name="written.endf"):
    out = tmp_path / name
    report = writeEndfTape(suite, out)
    return out, report


def _states(lines):
    """``{(ZAP, LFS, LMF)}`` of an MF8 section's state records (decay LISTs skipped)."""
    from kika.endf.utils import parse_line

    head = parse_line(lines[0])
    no, out, idx = int(head["C6"] or 0), set(), 1
    for _ in range(int(head["C5"] or 0)):
        record = parse_line(lines[idx])
        out.add((int(round(record["C1"])), int(record["C4"] or 0), int(record["C3"] or 0)))
        idx += 1 + ((int(record["C5"] or 0) + 5) // 6 if no == 0 else 0)
    return out


def _messages(report) -> str:
    return "\n".join(report.warnings + report.losses + report.approximations
                     + report.unsupported)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def test_every_final_state_is_one_production():
    suite, _ = _decode(CO59)
    labels = [p.label for p in suite.productions]
    assert labels == [productionLabel(p.ENDF_MT, p.id.residual) for p in suite.productions]
    assert len(set(labels)) == len(labels)
    assert {"MT16 Co58", "MT16 Co58_e1", "MT17 Co57",
            "MT102 Co60", "MT102 Co60_e1"} <= set(labels)
    assert {p.ENDF_MT for p in suite.productions} == {16, 17, 102, 111}
    for production in suite.productions:
        assert production.outputChannel.genre == "production"
        [product] = list(production.outputChannel.products)
        assert product.pid == production.id.residual


def test_mf9_and_lmf3_refer_to_the_reaction_and_mf10_carries_its_table():
    suite, _ = _decode(CO59)
    byLabel = {p.label: p for p in suite.productions}
    capture = byLabel["MT102 Co60_e1"]
    assert isinstance(capture.crossSection[EVAL_LABEL], Reference)
    assert "[@label='MT102']" in capture.crossSection[EVAL_LABEL].href
    assert not isinstance(capture.outputChannel.products[0].multiplicity.form, Constant1d)

    threeBody = [p for p in suite.productions if p.ENDF_MT == 17]
    assert len(threeBody) == 1                                   # LMF=3: MF8 alone
    assert isinstance(threeBody[0].crossSection[EVAL_LABEL], Reference)
    one = threeBody[0].outputChannel.products[0].multiplicity.form
    assert isinstance(one, Constant1d) and one.constant == 1.0

    n2n = byLabel["MT16 Co58_e1"]
    assert not isinstance(n2n.crossSection[EVAL_LABEL], Reference)
    assert n2n.outputChannel.products[0].multiplicity.form.constant == 1.0


def test_the_level_is_in_pops_and_the_isomers_are_aliased():
    suite, _ = _decode(CO59)
    level = suite.PoPs["Co58_e1"]
    assert level.nuclearLevel == 1 and level.energy.value == pytest.approx(24950.0)
    aliases = {a.id: a.pid for a in suite.PoPs.aliases.values()}
    assert aliases["Co58_m1"] == "Co58_e1" and aliases["Co60_m1"] == "Co60_e1"


def test_q_is_qi_and_qm_stays_in_the_provenance():
    suite, _ = _decode(AM241)
    isomer = suite.productions.byLabel("MT102 Am242_e2")
    assert isomer.outputChannel.Q.value == pytest.approx(5489125.0)
    assert isomer.provenance.qm == pytest.approx(5537755.0)
    assert suite.PoPs["Am242_e2"].energy.value == pytest.approx(48630.0)


def test_mf8_is_no_longer_declared_unsupported_wholesale():
    _, report = _decode(CO59)
    text = _messages(report)
    assert "activation data and are not in this reactionSuite" not in text
    assert "MF9 is present" not in text and "MF10 is present" not in text


# ---------------------------------------------------------------------------
# ENDF → model → ENDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [CO59, AM241], ids=["co59", "am241"])
def test_the_tape_comes_back_with_mf8_mf9_mf10_as_read(path, tmp_path):
    suite, _ = _decode(path)
    out, _ = _write(suite, tmp_path)
    assert _sections(out) == _sections(path)
    after, _ = _decode(out)
    assert _walk(after.productions) == _walk(suite.productions)
    assert _walk(after.PoPs) == _walk(suite.PoPs)


def test_an_edited_multiplicity_reaches_mf9_and_mf8_stays_as_read(tmp_path):
    suite, _ = _decode(AM241)
    form = suite.productions.byLabel("MT102 Am242_e2").outputChannel.products[0].multiplicity.form
    form.ys[:] = np.asarray(form.ys) * 1.5
    out, _ = _write(suite, tmp_path)
    before, after = _sections(AM241), _sections(out)
    assert after[(8, 102)] == before[(8, 102)]                    # decay LISTs kept
    assert after[(9, 102)] != before[(9, 102)]
    edited = read_endf(str(out)).mf[9].mt[102].get_state(95242, 2)
    assert edited.y[0] == pytest.approx(0.15)


def test_dropping_a_state_rebuilds_mf8_and_keeps_what_it_cannot_model(tmp_path):
    suite, _ = _decode(CO59)
    suite.productions.reactions = [p for p in suite.productions
                                   if p.label != "MT16 Co58_e1"]
    out, report = _write(suite, tmp_path)
    mf8 = read_endf(str(out)).mf[8].mt[16]
    assert len(mf8.raw_lines) == 2                               # HEAD + one state
    assert "MF8/MT16: the productions changed" in _messages(report)
    assert [s.lfs for s in read_endf(str(out)).mf[10].mt[16].states] == [0]


def test_an_lmf6_state_travels_as_text(tmp_path):
    """LMF=6 (multiplicity in MF6) has no node; the MF8 record must still come back."""
    text = CO59.read_text().splitlines()
    for k, line in enumerate(text):
        _, mf, mt = parse_endf_id(line) if len(line) >= 75 else (0, 0, 0)
        if (mf, mt) == (8, 17) and line[22:33].strip() == "3":
            text[k] = line[:22] + f"{6:>11}" + line[33:]
    edited = tmp_path / "lmf6.endf"
    edited.write_text("\n".join(text) + "\n")
    suite, report = _decode(edited)
    assert not [p for p in suite.productions if p.ENDF_MT == 17]
    assert "LMF=6" in _messages(report)
    out, _ = _write(suite, tmp_path)
    assert _sections(out)[(8, 17)] == _sections(edited)[(8, 17)]


def test_mf9_with_no_mf3_is_kept_as_text(tmp_path):
    trimmed, removed = remove_sections(CO59.read_text(), [(3, 102)])
    assert removed
    edited = tmp_path / "no_mf3.endf"
    edited.write_text(trimmed)
    suite, report = _decode(edited)
    assert not [p for p in suite.productions if p.ENDF_MT == 102]
    assert "9/102" in suite.provenance.headerFields[ACTIVATION_VERBATIM_KEY]
    assert 102 in suite.provenance.headerFields[ACTIVATION_MF8_KEY]
    out, _ = _write(suite, tmp_path)
    assert _sections(out)[(9, 102)] == _sections(edited)[(9, 102)]
    assert _sections(out)[(8, 102)] == _sections(edited)[(8, 102)]


# ---------------------------------------------------------------------------
# Format agnostic: GNDS, and the production cross section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [CO59, AM241], ids=["co59", "am241"])
def test_through_gnds_mf9_mf10_return_and_mf8_is_derived(path, tmp_path):
    suite = kika.read(str(path), covariances=False)
    xml = tmp_path / "suite.xml"
    kika.write(suite, xml)
    back = kika.read(str(xml), covariances=False)
    assert sorted(p.label for p in back.productions) == sorted(p.label for p in suite.productions)

    out = tmp_path / "from_gnds.endf"
    report = kika.write(back, out, format="endf")
    source, written = _sections(path), _sections(out)
    for key in source:
        if key[0] in (9, 10):
            assert written[key] == source[key], key
    for mt in {mt for mf, mt in source if mf == 8}:
        stated = read_endf(str(path)).mf[8].mt[mt].raw_lines
        derived = read_endf(str(out)).mf[8].mt[mt].raw_lines
        assert _states(derived) == _states(stated), mt
        assert derived[0][55:66].strip() == "1"                  # NO=1
    assert "derived from the productions with NO=1" in _messages(report)


def test_the_production_cross_section_is_sigma_times_multiplicity():
    suite = kika.read(str(AM241), covariances=False)
    ground, gx = suite.production_cross_section("MT102 Am242")
    isomer, ix = suite.production_cross_section("MT102 Am242_e2")
    np.testing.assert_array_equal(ground, isomer)
    sigma = suite.reactions[102].crossSection[EVAL_LABEL].evaluate(ground)
    np.testing.assert_allclose(gx + ix, sigma, rtol=1e-12)

    co59 = kika.read(str(CO59), covariances=False)
    n2n = co59.productions.byLabel("MT16 Co58_e1")
    energies, values = co59.production_cross_section(n2n)
    table = n2n.crossSection[EVAL_LABEL]
    np.testing.assert_allclose(values, table.evaluate(energies))


def test_an_mf8_section_that_is_not_activation_is_not_passed_over(tmp_path):
    """MF8/MT457 on an incident-neutron tape: unreadable here, and said so."""
    lines = CO59.read_text().splitlines(keepends=True)
    mat = int(lines[1][66:70])
    injected = (f"{' 0.000000+0 0.000000+0          0          0          0          0':<66}"
                f"{mat:>4}{8:>2}{457:>3}{1:>5}\n")
    doctored = tmp_path / "with_mf8_457.endf"
    doctored.write_text("".join(lines[:-1] + [injected] + lines[-1:]))
    _, report = _decode(doctored)
    assert any("MF8/MT457" in m for m in report.losses + report.unsupported)

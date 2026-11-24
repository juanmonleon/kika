"""MF3's QM and QI come back through GNDS where the derivation's rule would move them.

Back to ENDF, kika applies FUDGE's rules: a fission reaction's QM = QI is MF1/458's
ER at thermal, and a level reaction's QM is QI plus the level's PoPs energy. An
evaluation that does not follow them (ENDF/B-VIII.1 U-238 MT18: QM 197.786 MeV,
ER 196.535 MeV; Zr-90 MT651: QM - QI 908970 eV, MF8's level 909000 eV) keeps its
numbers through a flag: FUDGE's ``QI=`` on a fission Q, which then states QM, and
kika's ``decayQ`` on a level's decay Q. A tape that follows the rules gets none.
"""
from __future__ import annotations

import re
from pathlib import Path

import kika

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _edited(name, tmp_path, mfmt, line, fields):
    """*name* with the leading fields of MF/MT *mfmt*'s *line*-th line replaced."""
    out = []
    count = 0
    for raw in (DATA / name).read_bytes().decode("ascii").splitlines(keepends=True):
        if raw[70:75] == mfmt:
            count += 1
            if count == line:
                raw = fields + raw[len(fields):]
        out.append(raw)
    path = tmp_path / name
    path.write_bytes("".join(out).encode("ascii"))
    return path


def _roundTrip(tape, tmp_path):
    gnds = tmp_path / "rt.gnds.xml"
    kika.write(kika.read(str(tape)), str(gnds), format="gnds")
    endf = tmp_path / "rt.endf"
    kika.write(kika.read(str(gnds)), str(endf), format="endf")
    return gnds.read_text(encoding="utf-8"), endf


def _header(path, mfmt):
    return [l[:22] for l in Path(path).read_bytes().decode("ascii").splitlines()
            if l[70:75] == mfmt][1]


def test_a_fission_q_that_is_mf1_458s_er_needs_no_flag(tmp_path):
    text, endf = _roundTrip(DATA / "micro_th232_fission_energy.endf", tmp_path)
    assert "QI=" not in text
    assert _header(endf, " 3 18") == " 1.884700+8 1.884700+8"


def test_a_fission_q_off_mf1_458s_er_comes_back_and_fudges_rule_without_the_flag(tmp_path):
    tape = _edited("micro_th232_fission_energy.endf", tmp_path, " 3 18", 2, " 1.884800+8 1.884800+8")
    text, endf = _roundTrip(tape, tmp_path)
    assert re.search(r'flags="QI=188480000\.0" href="[^"]*MT18\']/outputChannel/Q"', text)
    assert _header(endf, " 3 18") == " 1.884800+8 1.884800+8"
    bare = tmp_path / "bare.gnds.xml"
    bare.write_text(re.sub(r'\s*<conversion flags="QI=[^>]*/>', "", text), encoding="utf-8")
    fudge = tmp_path / "fudge.endf"
    kika.write(kika.read(str(bare)), str(fudge), format="endf")
    assert _header(fudge, " 3 18") == " 1.884700+8 1.884700+8"


def test_zr90_mt651_comes_back_with_the_tapes_qm(zr90_b81_tape, tmp_path):
    """MF3 states QM - QI = 908970 eV, MF8 the level at 909000 eV."""
    text, endf = _roundTrip(zr90_b81_tape, tmp_path)
    assert text.count('flags="decayQ"') == 1
    assert "product[@label='Y89_e1']/outputChannel/Q" in text
    assert _header(endf, " 3651") == _header(zr90_b81_tape, " 3651") == "-6.128816+6-7.037786+6"

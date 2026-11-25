"""A fission MF3's QM/QI survive GNDS where MF1/458's ER would move them (B-VIII.1 U-238 MT18):
a ``QI=`` flag on the fission Q, none on a tape that follows the rule.
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

"""Files from other tools and editors: BOMs, stray cp1252 bytes, UTF-16, other CSV dialects."""

import pytest

from kika._identify import identify
from kika._read import UnknownFormatError
from kika._text import sniff_csv


def test_a_utf16_file_is_refused_with_how_to_fix_it(tmp_path):
    path = tmp_path / "deck.njoy"
    path.write_text("moder\n 20 -21/\nstop\n", encoding="utf-16")
    with pytest.raises(UnknownFormatError, match="UTF-16.*Save it as UTF-8"):
        identify(path)


@pytest.mark.parametrize("text, delimiter", [
    ("a,b,c\n1,2,3\n4,5,6\n", ","),
    ("a;b;c\n1,5;2,0;3\n4,5;5,0;6\n", ";"),
    ("file\tmat\tstatus\nfe56.endf\t2631\tok\nu235.endf\t9228\twarning\n", "\t"),
    ('name,value\n"Fe-56, MT102",1.0\n"U-235, MT18",2.0\n', ","),
])
def test_delimited_tables_are_csv(tmp_path, text, delimiter):
    path = tmp_path / "table.csv"
    path.write_text("# exported by a script\n" + text, encoding="utf-8")
    assert identify(path) == "csv"
    assert sniff_csv(path.read_text()) == (delimiter, 1)


def test_a_coverx_text_file_with_an_accented_title_reads_as_text(tmp_path):
    from kika.cov.parse_covmat import _detect_coverx_format

    path = tmp_path / "coverx.txt"
    path.write_bytes("Évaluation 3d\n".encode("utf-8-sig"))
    assert _detect_coverx_format(str(path)) == "text"

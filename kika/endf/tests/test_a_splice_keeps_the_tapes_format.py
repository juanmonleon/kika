"""A splice changes the section it replaces, and leaves the tape's format alone.

Every writer that edits an existing tape -- ``ENDFWriter.replace_mt_section``
and ``replace_mf_section``, ``write_mf_section_to_file``,
``remove_mf34_from_file`` -- read in text mode and wrote in text mode. On
Windows that turned every line of an LF tape into CRLF, so a one-section edit
touched every line of the file at the byte level and ``cmp`` against the source
could no longer show what moved (B1a of the PFNS roadmap). Each one now writes
the line ending it read.

Both endings are built here from one committed tape rather than trusted to the
checkout, because git's ``autocrlf`` decides what a committed file ends its lines
with on a given machine.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.writers.endf_writer import ENDFWriter
from kika.endf.writers.mf34_writer import remove_mf34_from_file

DATA = Path(__file__).resolve().parent / "data"
PFNS = DATA / "micro_cf252_pfns.endf"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"

ENDINGS = {"LF": b"\n", "CRLF": b"\r\n"}


def _copy(source, tmp_path, ending):
    """*source* rewritten with *ending* on every line."""
    lines = Path(source).read_bytes().splitlines()
    out = tmp_path / f"source_{ending.hex()}.endf"
    out.write_bytes(b"".join(line + ending for line in lines))
    return out


def _assertEndings(path, ending):
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")[:-1]
    assert raw.endswith(b"\n")
    if ending == b"\r\n":
        assert all(line.endswith(b"\r") for line in lines), "a line lost its CR"
    else:
        assert not any(line.endswith(b"\r") for line in lines), "a CR appeared"


@pytest.mark.parametrize("name", ENDINGS)
def test_replacing_a_section_keeps_the_line_ending(name, tmp_path):
    ending = ENDINGS[name]
    source = _copy(PFNS, tmp_path, ending)
    section = read_endf(str(source), mf_numbers=[5]).mf[5].mt[18]
    out = tmp_path / "out.endf"

    assert ENDFWriter(str(source)).replace_mt_section(
        section, mf_number=5, output_filepath=str(out))
    _assertEndings(out, ending)


@pytest.mark.parametrize("name", ENDINGS)
def test_replacing_a_whole_file_keeps_the_line_ending(name, tmp_path):
    ending = ENDINGS[name]
    source = _copy(PFNS, tmp_path, ending)
    mf5 = read_endf(str(source), mf_numbers=[5]).mf[5]
    out = tmp_path / "out.endf"

    assert ENDFWriter(str(source)).replace_mf_section(mf5, str(out))
    _assertEndings(out, ending)


@pytest.mark.parametrize("name", ENDINGS)
def test_removing_mf34_keeps_the_line_ending(name, tmp_path):
    ending = ENDINGS[name]
    source = _copy(FE56, tmp_path, ending)

    assert remove_mf34_from_file(str(source))
    _assertEndings(source, ending)


@pytest.mark.parametrize("name", ENDINGS)
def test_a_splice_of_an_unchanged_section_leaves_the_rest_byte_identical(
        name, tmp_path):
    """Outside MF5/MT18 and the directory, the bytes are the source's bytes."""
    ending = ENDINGS[name]
    source = _copy(PFNS, tmp_path, ending)
    section = read_endf(str(source), mf_numbers=[5]).mf[5].mt[18]
    out = tmp_path / "out.endf"
    ENDFWriter(str(source)).replace_mt_section(
        section, mf_number=5, output_filepath=str(out))

    def outside(path):
        """Drop MF5/MT18 with its SEND, and MF1/451 with its SEND."""
        kept, inReplaced = [], False
        for line in Path(path).read_bytes().splitlines(keepends=True):
            if line[70:75] in (b" 5 18", b" 1451"):
                inReplaced = True
                continue
            if inReplaced and line[72:75] == b"  0":
                inReplaced = False
                continue
            inReplaced = False
            kept.append(line)
        return kept

    assert outside(out) == outside(source)


@pytest.mark.parametrize("operation", ["mt", "mf"])
def test_a_splice_into_a_75_column_tape_stays_75_columns(operation, tmp_path):
    """The encoders write a sequence number; a tape without one must not gain it.

    Built by cutting columns 76-80 off the micro-tape, because the committed
    one is not a clean witness: its MF1/451 is 80 wide (a leftover of the
    directory rebuild before it learned the width), so it measures as 80.
    """
    lines = PFNS.read_bytes().splitlines()
    source = tmp_path / "source_75.endf"
    source.write_bytes(b"".join(line[:75].rstrip(b"\r") + b"\n" for line in lines))
    endf = read_endf(str(source), mf_numbers=[5])
    out = tmp_path / "out.endf"

    writer = ENDFWriter(str(source))
    if operation == "mt":
        assert writer.replace_mt_section(endf.mf[5].mt[18], mf_number=5,
                                         output_filepath=str(out))
    else:
        assert writer.replace_mf_section(endf.mf[5], str(out))

    widths = {len(line) for line in out.read_bytes().splitlines()}
    assert max(widths) <= 75, f"record widths in the output: {sorted(widths)}"

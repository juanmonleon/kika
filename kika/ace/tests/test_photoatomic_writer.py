"""Writing EPRDATA photo-atomic tables: the encoder is the decoder's inverse.

The synthetic EPRDATA table of ``test_photoatomic_ace`` is read, laid out again
from its decoded blocks by :func:`kika.ace.encoders.photoatomic.encode_photoatomic`
and written: the XSS must be the same words, the file must read back to the
same blocks, and every word is spelled ``%20.12E`` as the EPRDATA libraries
spell it.

LANL's ``eprdata25`` (looked for in ``$KIKA_EPRDATA25`` or
``../NuclearData/ace/eprdata25`` beside the repo) must come back byte for byte,
read and written, for every table; re-encoded from the blocks, byte for byte
for every table whose Compton blocks have the lengths NXS(5) states. The ~25
that do not (LNEPS/LBEPS/LPIPS laid out over more words than NXS(5)) are
written in the consistent layout, and their decoded blocks must survive.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.encoders.photoatomic import encode_photoatomic
from kika.ace.tests.test_photoatomic_ace import build_photoatomic, write_photoatomic
from kika.ace.writers.write_ace import write_ace


def _synthetic(tmp_path):
    return read_ace(str(write_photoatomic(tmp_path / "fe.14p", "eprdata", zaid="26000.14p")))


def test_encode_gives_back_the_words(tmp_path):
    t = _synthetic(tmp_path)
    e = encode_photoatomic(t)
    # NXS(13)-(14), the JINC and JCOH lengths, are written as eprdata14/25 write them;
    # the synthetic table leaves them 0, as eprdata12 did
    assert list(e.header.nxs_array[:13]) == list(t.header.nxs_array[:13])
    assert (e.header.nxs_array[13], e.header.nxs_array[14]) == (t.incoherent_x.size, t.coherent_x.size)
    assert list(e.header.jxs_array) == list(t.header.jxs_array)
    np.testing.assert_allclose(e.xss_data, t.xss_data, rtol=1e-14, atol=0.0)


def test_every_word_is_a_twelve_digit_real(tmp_path):
    t = _synthetic(tmp_path)
    out = tmp_path / "again.14p"
    write_ace(encode_photoatomic(t), str(out))
    data = out.read_text().splitlines()[12:]
    fields = [line[k:k + 20] for line in data for k in range(0, len(line), 20)]
    assert fields and all(len(f) == 20 and "E" in f and len(f.split("E")[0].split(".")[1]) == 12
                          for f in fields)
    back = read_ace(str(out))
    np.testing.assert_allclose(back.energies, t.energies, rtol=1e-12)
    np.testing.assert_allclose(back.heating, t.heating, rtol=1e-12)
    assert back.relaxation.keys() == t.relaxation.keys()


def test_mismatched_blocks_are_refused(tmp_path):
    t = _synthetic(tmp_path)
    t.heating = t.heating[:-1]
    with pytest.raises(ValueError, match="heating"):
        encode_photoatomic(t)


def _eprdata25():
    root = os.environ.get("KIKA_EPRDATA25")
    root = Path(root) if root else Path(__file__).resolve().parents[3] / ".." / ".." / "NuclearData" / "ace" / "eprdata25"
    return sorted((root / "eprdata25").glob("*.25p")) if (root / "eprdata25").is_dir() else []


@pytest.mark.parametrize("path", _eprdata25() or [None], ids=lambda p: p.name if p else "none")
def test_eprdata25_writes_back(path, tmp_path):
    if path is None:
        pytest.skip("no eprdata25 ($KIKA_EPRDATA25 or ../NuclearData/ace/eprdata25)")
    reference = path.read_bytes().replace(b"\r\n", b"\n")
    t = read_ace(str(path))
    out = tmp_path / "rt"
    write_ace(t, str(out))
    assert out.read_bytes() == reference
    nxs, jxs = t.header.nxs_array, t.header.jxs_array
    consistent = all(jxs[k + 1] - jxs[k] == nxs[5] for k in (6, 7, 8, 9))
    again = tmp_path / "enc"
    write_ace(encode_photoatomic(t), str(again))
    if consistent:
        assert again.read_bytes() == reference
    else:
        back = read_ace(str(again))
        np.testing.assert_array_equal(back.energies, t.energies)
        assert [p.pz.size for p in back.compton_profiles] == [p.pz.size for p in t.compton_profiles]

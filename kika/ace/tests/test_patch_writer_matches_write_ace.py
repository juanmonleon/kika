"""A patched ACE sample is the file ``write_ace`` would have written.

:class:`~kika.ace.writers.patch_writer.AcePatchWriter` writes a perturbed table
as its unperturbed template with only the changed XSS lines re-formatted. The
whole point is that nobody downstream can tell: the bytes must equal
``write_ace``'s, for a table whose last XSS line is full and for one that ends
in a short line, and every layout change ``write_ace`` would make (another
header, another length, the all-integer format) must go through ``write_ace``.

Synthetic tables only, so these run in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace.classes.ace import Ace
from kika.ace.classes.header import Header
from kika.ace.writers.patch_writer import AcePatchWriter
from kika.ace.writers.write_ace import write_ace


def _ace(n: int, seed: int = 0) -> Ace:
    rng = np.random.default_rng(seed)
    values = rng.lognormal(0.0, 4.0, n) * rng.choice([-1.0, 1.0], n)
    values[::17] = 0.0
    values[5] = 1.0e-310          # subnormal
    xss = np.concatenate([[0.0], values])
    header = Header(
        format_version="legacy", zaid=26056, extension=".02c",
        atomic_weight_ratio=55.454, temperature=2.5301e-08, date="01/01/26",
        comment="synthetic fixture", matid=2631,
        izaw_array=[(0, 0.0)] * 16, nxs_array=[n] + [0] * 15,
        jxs_array=[1] + [0] * 31,
    )
    return Ace(filename=None, header=header, xss_data=xss)


def _perturb(ace: Ace, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = ace.xss_data.size - 1
    for start, stop in ((1, 40), (n // 2, n // 2 + 333), (n - 7, n + 1)):
        ace.xss_data[start:stop] *= rng.normal(1.0, 0.1, stop - start)


@pytest.mark.parametrize("n", [4000, 4003, 4001])
def test_patched_samples_are_byte_identical(tmp_path, n):
    ace = _ace(n)
    writer = AcePatchWriter(ace, str(tmp_path / "template.02c"))
    assert writer.usable
    for sample in range(3):
        _perturb(ace, sample)
        patched, reference = tmp_path / f"p{sample}.02c", tmp_path / f"r{sample}.02c"
        assert writer.write(ace, str(patched))
        write_ace(ace, str(reference), overwrite=True)
        assert patched.read_bytes() == reference.read_bytes()
        writer.restore(ace)
    np.testing.assert_array_equal(ace.xss_data, writer.base_xss)


def test_an_unchanged_table_is_the_template(tmp_path):
    ace = _ace(4002)
    writer = AcePatchWriter(ace, str(tmp_path / "template.02c"))
    writer.write(ace, str(tmp_path / "same.02c"))
    assert (tmp_path / "same.02c").read_bytes() == (tmp_path / "template.02c").read_bytes()


def test_restore_keeps_the_array_the_blocks_view(tmp_path):
    ace = _ace(4000)
    xss = ace.xss_data
    writer = AcePatchWriter(ace, str(tmp_path / "template.02c"))
    _perturb(ace, 1)
    writer.restore(ace)
    assert ace.xss_data is xss


@pytest.mark.parametrize("change", ["header", "length", "integers"])
def test_a_layout_change_falls_back_to_write_ace(tmp_path, change):
    ace = _ace(4003)
    writer = AcePatchWriter(ace, str(tmp_path / "template.02c"))
    if change == "header":
        ace.header.comment = "another comment"
    elif change == "length":
        ace.xss_data = np.concatenate([ace.xss_data, [1.5, 2.5]])
        ace.header.nxs_array[0] += 2
    else:
        ace.xss_data[1:] = np.rint(ace.xss_data[1:])
    patched, reference = tmp_path / "p.02c", tmp_path / "r.02c"
    assert not writer.write(ace, str(patched))
    write_ace(ace, str(reference), overwrite=True)
    assert patched.read_bytes() == reference.read_bytes()


def test_close_removes_the_template(tmp_path):
    writer = AcePatchWriter(_ace(400), str(tmp_path / "template.02c"))
    writer.close()
    assert not (tmp_path / "template.02c").exists()

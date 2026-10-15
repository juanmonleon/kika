"""The light charged particles of an ACE table reach the model and come back (roadmap A11).

PTYPE/NTRO/IXS become one product per (particle, reaction in MTRH): yield
from SIGH, Monte Carlo form from DLWH and ANDH, frame from TYRH. HPD is kept
as the table stated it and given back while the production the suite
rebuilds still matches. Checked on the O-16 table kika writes (protons,
deuterons, tritons, alphas), as test_ace_particles.py does; the committed
micro-tapes give no table with light particles.
"""
import numpy as np
import pytest

import kika
from kika.ace import read_ace
from kika.ace.encoders.layout import encode_xss
from kika.ace.model_adapter.encode import encodeAce
from kika.endf import ProcessingPlan, process_endf

pytestmark = pytest.mark.tape


@pytest.fixture(scope="module")
def table(o16_b81_tape, tmp_path_factory):
    out = tmp_path_factory.mktemp("o16")
    (path,) = process_endf(o16_b81_tape, ProcessingPlan(temperatures=(293.6,), workers=1)).write_ace(out)
    return path


def test_the_particles_are_products(table):
    ace = read_ace(str(table))
    suite = kika.read(str(table))
    assert list(ace.secondary_particle_types.particle_ids)
    pids = {p.pid for r in suite.reactions for p in r.outputChannel.products}
    assert pids & {"H1", "H2", "H3", "He3", "He4"}
    assert not [m for m in suite.report.unsupported if "IXS" in m]


def test_the_table_comes_back_with_its_particles(table):
    original = read_ace(str(table))
    again, _ = encodeAce(kika.read(str(table)))
    a, b = encode_xss(original), encode_xss(again)
    assert list(a.nxs) == list(b.nxs) and list(a.jxs) == list(b.jxs)
    np.testing.assert_allclose(b.xss[1:], a.xss[1:], rtol=1e-12, atol=0)


def test_an_edited_yield_rebuilds_the_hpd(table):
    """The table's HPD is given back only while what it is made of is unchanged."""
    suite = kika.read(str(table))
    product = next(p for r in suite.reactions for p in r.outputChannel.products if p.pid == "He4")
    form = next(iter(product.multiplicity.forms.values()))
    form.ys = np.asarray(form.ys, dtype=float) * 2.0
    _, report = encodeAce(suite)
    assert any("HPD He4" in line and "changed" in line for line in report.approximations)

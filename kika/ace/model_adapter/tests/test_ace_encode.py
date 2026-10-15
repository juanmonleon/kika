"""ACE roadmap A4: a processed suite → ACE, and the A4a gate on a real table.

The gate: ACE → model (A3) → ACE (A4) is the original word for word, once the
two things A4 does not write are taken out of the original (the
charged-particle blocks, A7, and NJOY's unused room after some law tables).
Both sides are laid out by the A2 encoders, whose own gate makes the
original's re-encoding its XSS; so a difference here is the model losing or
changing something. Integer words must be equal and real words within a few
ulp: the model works in eV and ACE in MeV.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.encoders.layout import encode_xss
from kika.ace.model_adapter import decodeAce
from kika.ace.model_adapter.encode import AceEncodeError, encodeAce

pytestmark = pytest.mark.tape


def _strip(ace):
    """The original without the charged-particle blocks and NJOY's padding."""
    a = copy.deepcopy(ace)
    if a.secondary_particle_types is not None:
        a.secondary_particle_types.particle_ids = []
    a.reaction_mt_data.particle_production = []
    ed = a.energy_distributions
    for chain in list(ed.incident_neutron.values()) + list(ed.photon_production.values()) + \
            list(ed.delayed_neutron):
        for law in chain:
            if getattr(law, "padding", None):
                law.padding = None
            if getattr(law, "padding_after", 0):
                law.padding_after = 0
    return a


def _assertSameWords(original, written, rtol=1e-14):
    a, b = encode_xss(_strip(original)), encode_xss(written)
    assert a.nxs[1:] == b.nxs[1:]
    assert a.jxs[1:] == b.jxs[1:]
    np.testing.assert_array_equal(a.is_int, b.is_int)
    ints = a.is_int
    np.testing.assert_array_equal(a.xss[1:][ints], b.xss[1:][ints])
    np.testing.assert_allclose(b.xss[1:][~ints], a.xss[1:][~ints], rtol=rtol, atol=0.0)


@pytest.fixture(scope="module")
def original(fe56_ace):
    return read_ace(str(fe56_ace))


@pytest.fixture(scope="module")
def decoded(original):
    return decodeAce(original)


def test_ace_model_ace_is_the_original_word_for_word(original, decoded):
    suite, _ = decoded
    written, report = encodeAce(suite)
    _assertSameWords(original, written)
    assert written.header.zaid == original.header.zaid
    assert written.header.temperature == pytest.approx(original.header.temperature, rel=1e-15)


def test_the_written_table_is_a_file_kika_reads_back_strictly(original, decoded, tmp_path):
    from kika.ace.writers.write_ace import write_ace

    suite, _ = decoded
    written, _ = encodeAce(suite)
    path = str(tmp_path / "fe56.ace")
    write_ace(written, path, overwrite=True)
    again = read_ace(path, strict=True)
    _assertSameWords(original, again)


def _withoutAceProvenance(suite):
    """The suite as kika's own processing would hand it over: no ACE bookkeeping."""
    s = copy.deepcopy(suite)
    for key in ("MTR", "YP", "MTRP", "NXS", "ZA"):
        s.provenance.attributes.pop(key, None)
    return s


def test_without_ace_provenance_mtr_follows_njoys_rule(original, decoded):
    """Secondary-neutron reactions first, the rest ascending, the yield hosts last."""
    suite, _ = decoded
    written, _ = encodeAce(_withoutAceProvenance(suite))
    mtr = [int(m) for m in written.reaction_mt_data.incident_neutron]
    secondary = [int(m) for m in written.reaction_mt_data.secondary_neutron_mt]
    assert mtr[:len(secondary)] == sorted(secondary)
    assert set(mtr) == {int(m) for m in original.reaction_mt_data.incident_neutron}
    for mt in mtr:
        np.testing.assert_allclose(written.cross_section.reaction[mt]._xs_entries,
                                   original.cross_section.reaction[mt]._xs_entries, rtol=1e-14)


def test_a_suite_with_no_ace_suffix_takes_its_temperatures(decoded):
    """The suffix kika's NJOY runs give: ``.02c`` at 293.6 K; ``extension=`` overrides it."""
    suite, _ = decoded
    s = copy.deepcopy(suite)
    s.provenance.extension = None
    written, _ = encodeAce(s)
    assert written.header.extension == ".02c"
    written, _ = encodeAce(s, extension=".12c")
    assert written.header.extension == ".12c"


def test_an_ace_table_holds_one_temperature(decoded):
    from kika.nuclear_data.model import Heated, PhysicalQuantity

    suite, _ = decoded
    hot = copy.deepcopy(suite)
    hot.styles.add(Heated(label="heated_600K", derivedFrom="eval",
                          temperature=PhysicalQuantity(value=600 * 8.617333262e-05, unit="eV/k")))
    with pytest.raises(AceEncodeError, match="temperature="):
        encodeAce(hot)


def test_the_charged_particle_blocks_are_reported_not_written(decoded):
    suite, _ = decoded
    written, report = encodeAce(suite)
    assert not list(getattr(written.secondary_particle_types, "particle_ids", []) or [])
    assert any("charged-particle production" in line for line in report.unsupported)

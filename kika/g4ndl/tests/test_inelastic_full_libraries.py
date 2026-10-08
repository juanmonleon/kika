"""Phase 10 against the two whole libraries. ``tape``-marked: needs ``KIKA_G4NDL``.

Every ``Inelastic/`` file is read to its last token, written back as text and
read again (the text fixed point), and every inelastic isotope is read into
the model, written back from it and compared record for record (the model
fixed point). The counts are the census of
``kika-workspace/myworkspace/G4NDL/phase10/`` (2026-10-08). Several minutes
per library: the model fixed point decodes ~27 000 files.
"""
from __future__ import annotations

import collections

import pytest

import kika.g4ndl as g4ndl
from kika.g4ndl.encode import recordDifferences
from kika.g4ndl.inelastic_encode import SUM_RTOL, encodeInelastic, partialSumCheck
from kika.g4ndl.inelastic_format import formatGammas, formatInelasticFS, inelasticDifferences
from kika.g4ndl.inelastic_parse import parse_gammas, parse_inelastic_fs
from kika.g4ndl.library import CHANNELS
from kika.g4ndl.tokens import TokenStream

pytestmark = pytest.mark.tape

#: library -> (channel files, files per dataType, isotopes with inelastic data).
CENSUS = {
    "g4ndl_jeff40_library": (13855, {3: 31640, 4: 15055, 5: 21, 6: 15843, 12: 15100,
                                     13: 33, 14: 15133, 15: 13}, 592),
    "g4ndl_g4ndl471_library": (10668, {3: 26868, 4: 13641, 5: 313, 6: 12049,
                                       12: 12278, 13: 67, 14: 12345, 15: 133}, 559),
}


def _keys(lib):
    keys = set(lib.isotopes("inelastic"))
    for sub in CHANNELS["inelastic"]:
        keys |= {k for (s, k) in lib._index if s == sub}
    return sorted(keys, key=lambda k: (k.Z, -1 if k.A is None else k.A, k.M))


@pytest.mark.parametrize("fixture", sorted(CENSUS))
def test_every_inelastic_file_reads_and_is_a_text_fixed_point(request, fixture):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    n_files, per_type, _ = CENSUS[fixture]
    files, types = 0, collections.Counter()
    for key in _keys(lib):
        for ch in lib.inelasticChannels(key):
            record = lib.inelasticFinalState(key, ch)
            files += 1
            types.update(s.dataType for s in record.sections)
            back = parse_inelastic_fs(TokenStream(formatInelasticFS(record)), ch)
            assert inelasticDifferences(record, back) == [], (key, ch)
    assert files == n_files and dict(types) == per_type
    for z, a in lib.gammaNuclei():
        g = lib.gammas(z, a)
        assert inelasticDifferences(g, parse_gammas(TokenStream(formatGammas(g)), z, a)) == []


@pytest.mark.parametrize("fixture", sorted(CENSUS))
def test_every_inelastic_isotope_is_a_fixed_point_through_the_model(request, fixture):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    keys = _keys(lib)
    assert len(keys) == CENSUS[fixture][2]
    for key in keys:
        suite = lib.read(key, processes=["inelastic"])
        total, files, _ = encodeInelastic(suite)
        if lib.has(key, "inelastic"):
            assert recordDifferences(lib.inelasticCrossSection(key), total) == [], key
        assert sorted(files) == lib.inelasticChannels(key), key
        for ch, record in files.items():
            assert inelasticDifferences(lib.inelasticFinalState(key, ch), record) == [], (key, ch)
        if key.Z > 0:   # 0_0_Zero has a total and no channel
            for row in partialSumCheck(suite):
                assert row["maxRel"] < SUM_RTOL, (key, row)

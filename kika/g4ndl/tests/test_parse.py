"""Phases 2 and 3: the cross-section and elastic final-state parsers.

Acceptance (roadmap §5 Fases 2-3): every real case of repFlag 1/2/3 is read
to its last token with nothing changed; repeated energies survive in order;
``repFlag=0``, the laboratory frame and ``tempdep`` come from synthetic files;
each malformed input stops with the file, the record and the token.

The whole-library counts are in ``test_full_libraries.py`` (``tape``).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

import kika.g4ndl as g4ndl
from kika.g4ndl import G4NDLFormatError
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs, parse_interpolation
from kika.g4ndl.records import REP_ISOTROPIC, REP_LEGENDRE, REP_MIXED, REP_TABULATED
from kika.g4ndl.tokens import TokenStream

DATA = Path(__file__).parent / "data"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
G4 = g4ndl.open(DATA / "G4NDL-4.7.1")
SYNTH = g4ndl.open(DATA / "synthetic")


def _raw_tokens(lib, target, subdir):
    """The decompressed tokens, read without kika.g4ndl."""
    path = lib.locate(target, subdir).path
    raw = path.read_bytes()
    return (zlib.decompress(raw) if path.suffix == ".z" else raw).decode().split()


def _fs(text):
    return parse_elastic_fs(TokenStream(text))


# ------------------------------------------------------------ cross section

@pytest.mark.parametrize("lib,target", [
    (JEFF, "H1"), (JEFF, "He3"), (JEFF, "C12"), (JEFF, "N14"), (JEFF, "Co58m1"),
    (G4, "Cnat"),
])
def test_cross_section_values_are_the_tokens(lib, target):
    cs = lib.crossSection(target)
    tok = _raw_tokens(lib, target, "Elastic/CrossSection")
    assert cs.bookkeeping == (0, 0) and cs.header is None
    assert len(cs) == int(tok[2])
    # No numerical change: each value is float() of its own token.
    assert cs.energy.tolist() == [float(x) for x in tok[3::2]]
    assert cs.sigma.tolist() == [float(x) for x in tok[4::2]]


def test_repeated_cross_section_energies_are_kept_in_order():
    cs = JEFF.crossSection("Co58m1")
    tok = _raw_tokens(JEFF, "Co58m1", "Elastic/CrossSection")
    repeats = int(np.sum(np.diff(cs.energy) == 0))
    assert repeats > 0
    assert repeats == sum(a == b for a, b in zip(tok[3::2], tok[5::2]))


def test_header_is_kept_on_the_cross_section():
    cs = SYNTH.crossSection("H1")
    assert cs.header == ("G4NDL", "synthetic")
    assert cs.energy.tolist() == [1.0e-5, 1.0e6, 2.0e7]


def test_negative_cross_section_is_refused():
    with pytest.raises(G4NDLFormatError, match="negative cross section -3.0") as e:
        SYNTH.crossSection("C13")
    assert e.value.path.name == "6_13_Carbon" and e.value.token == 6


def test_short_cross_section_is_refused():
    with pytest.raises(G4NDLFormatError, match="8 values needed, 6 left"):
        SYNTH.crossSection("Ne20")


@pytest.mark.parametrize("text,match", [
    ("0 0\n0\n", "N=0"),
    ("0 0\n2\n2.0 1.0 1.0 1.0\n", "energy decreases at point 1"),
    ("0 0\n2\n0.0 1.0 1.0 1.0\n", "first energy 0.0"),
    ("0 0\n1\n1.0 1.0 7\n", "1 unread tokens"),
    ("0.0 0\n1\n1.0 1.0\n", "expected an integer"),
])
def test_malformed_cross_sections(text, match):
    with pytest.raises(G4NDLFormatError, match=match):
        parse_cross_section(TokenStream(text))


# ------------------------------------------------------------ interpolation

@pytest.mark.parametrize("text,n,nbt,codes", [
    ("1 4 2", 4, (4,), (2,)),
    ("2 14 3 614 2", 614, (14, 614), (3, 2)),
])
def test_interpolation_records(text, n, nbt, codes):
    r = parse_interpolation(TokenStream(text), n, "t")
    assert (r.nbt, r.codes) == (nbt, codes)


@pytest.mark.parametrize("text,n,match", [
    ("0", 4, "NR=0"),
    ("1 3 2", 4, "last NBT is 3 but the record has 4"),
    ("2 3 2 3 2", 3, "does not increase strictly"),
    ("1 4 6", 4, "code 6 is not supported: not a code Geant4 knows"),
    ("1 4 12", 4, "code 12 is not supported: a C/U variant"),
    ("1 4 22", 4, "code 22 is not supported: a C/U variant"),
])
def test_malformed_interpolation(text, n, match):
    with pytest.raises(G4NDLFormatError, match=match):
        parse_interpolation(TokenStream(text), n, "t")


# -------------------------------------------------------------- final state

def test_legendre_only_with_a_repeated_incident_energy():
    fs = JEFF.elasticFinalState("H1")
    assert fs.repFlag == REP_LEGENDRE and fs.tabulated is None
    assert fs.frameFlag == 2 and fs.header is None and fs.transitionEnergy is None
    e = fs.legendre.energies
    assert int(np.sum(np.diff(e) == 0)) == 1
    # Coefficients start at a_1: a_0 = 1 is not in the file and not added.
    tok = _raw_tokens(JEFF, "H1", "Elastic/FS")
    r0 = fs.legendre.records[0]
    i = 3 + 1 + 1 + 2 * fs.legendre.interpolation.nRegions  # repFlag.., NE, NR, pairs
    assert [float(tok[i]), float(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] == \
        [r0.temperature, r0.energy, r0.tempdep, len(r0.coefficients)]
    assert r0.coefficients.tolist() == [float(x) for x in tok[i + 4:i + 4 + len(r0.coefficients)]]


def test_tabulated_only():
    fs = JEFF.elasticFinalState("He3")
    assert fs.repFlag == REP_TABULATED and fs.legendre is None
    for r in fs.tabulated.records:
        assert len(r.mu) == len(r.probability) == r.interpolation.nbt[-1]
        assert r.mu[0] >= -1 and r.mu[-1] <= 1


def test_mixed_with_loglin_in_mu():
    fs = JEFF.elasticFinalState("C12")
    assert fs.repFlag == REP_MIXED
    assert fs.tabulated.interpolation.nbt == (17,)
    assert {c for r in fs.tabulated.records for c in r.interpolation.codes} == {4}
    assert all(np.all(r.probability > 0) for r in fs.tabulated.records)
    assert fs.transitionEnergy == fs.legendre.records[-1].energy == fs.tabulated.records[0].energy


def test_two_energy_regions_are_kept_apart():
    fs = JEFF.elasticFinalState("N14")
    assert fs.legendre.interpolation.nbt == (14, 614)
    assert fs.legendre.interpolation.codes == (3, 2)
    assert len(fs.legendre) == 614


@pytest.mark.parametrize("lib,target", [(JEFF, "Co58m1"), (G4, "Cnat")])
def test_targetMass_is_the_file_value_not_A(lib, target):
    fs = lib.elasticFinalState(target)
    assert fs.targetMass == float(_raw_tokens(lib, target, "Elastic/FS")[1])


def test_repflag0_reads_the_second_frame_flag():
    fs = SYNTH.elasticFinalState("H1")
    assert fs.repFlag == REP_ISOTROPIC and fs.header == ("G4NDL", "synthetic")
    assert (fs.frameFlag, fs.frameFlag2) == (2, 2)
    assert fs.legendre is None and fs.tabulated is None


def test_lab_frame_and_tempdep_are_kept():
    fs = SYNTH.elasticFinalState("Li6")
    assert fs.frameFlag == 1
    assert fs.temperatures.tolist() == [293.6, 293.6]
    assert fs.tempdeps.tolist() == [0, 1]


def test_the_compressed_twin_is_what_gets_parsed():
    assert SYNTH.elasticFinalState("Be9").repFlag == REP_ISOTROPIC


@pytest.mark.parametrize("target,match,record", [
    ("B10", "interpolation code 6", "energy interpolation: INT[0]"),
    ("B11", "unexpected end of data", "energy 2 of 2"),
    ("N15", "expected an integer, found '2.0'", "NE"),
    ("O17", "3 unread tokens", "end of file"),
    ("F19", "table starts at 21000000.0 eV", "transition"),
])
def test_malformed_final_states_name_file_and_record(target, match, record):
    with pytest.raises(G4NDLFormatError, match=match) as e:
        SYNTH.elasticFinalState(target)
    assert e.value.path is not None and record in e.value.record


_HEAD = "2 3.0 2\n1\n1 1 2\n0.0 1.0e6 0 "


@pytest.mark.parametrize("text,match", [
    ("4 1.0 2\n", "repFlag=4"),
    ("1 -1.0 2\n", "targetMass=-1.0"),
    ("1 1.0 3\n", "frameFlag=3"),
    ("0 1.0 2\n5\n", "second frameFlag"),
    ("1 1.0 2\n0\n", "NE=0"),
    ("3 1.0 2\n0\n", "NE=0"),
    ("1 1.0 2\n2\n1 2 2\n0.0 2.0e6 0 0\n0.0 1.0e6 0 0\n", "incident energy decreases"),
    ("1 1.0 2\n1\n1 1 2\n0.0 1.0e6 0 -1\n", "NL=-1"),
    (_HEAD + "2 1 2 2 -1.5 0.5 1.0 0.5\n", "mu=-1.5 outside"),
    (_HEAD + "2 1 2 2 0.5 0.5 -0.5 0.5\n", "mu decreases"),
    (_HEAD + "2 1 2 2 -1.0 -0.1 1.0 0.5\n", "negative probability"),
    (_HEAD + "2 1 2 4 -1.0 0.0 1.0 1.0\n", "code 4 takes ln y"),
    (_HEAD + "2 1 2 3 -1.0 0.5 1.0 0.5\n", "code 3 takes ln x"),
])
def test_malformed_final_states(text, match):
    with pytest.raises(G4NDLFormatError, match=match):
        _fs(text)


def test_transition_within_the_consumer_tolerance_is_accepted():
    e = 2.0e7
    text = ("3 1.0 2\n1\n1 1 2\n"
            f"0.0 {e!r} 0 1 0.1\n"
            "1\n1 1 2\n"
            f"0.0 {e * (1 + 5e-16)!r} 0 2 1 2 2 -1.0 0.5 1.0 0.5\n")
    assert _fs(text).transitionEnergy == e


def test_log_axes_only_bite_inside_their_region():
    # LINLIN on the first two mu points, LOGLIN on the last two: p=0 at mu=-1 is fine.
    fs = _fs("2 3.0 2\n1\n1 1 2\n0.0 1.0e6 0 4 2 2 2 4 4 "
             "-1.0 0.0 0.0 0.5 0.5 0.5 1.0 0.5\n")
    assert fs.tabulated.records[0].probability[0] == 0.0


def test_a_region_boundary_node_belongs_to_both_regions():
    # Same regions, but p=0 at mu=0: the last LINLIN node is the first LOGLIN one.
    with pytest.raises(G4NDLFormatError, match="code 4 takes ln y but y=0.0 at point 1"):
        _fs("2 3.0 2\n1\n1 1 2\n0.0 1.0e6 0 4 2 2 2 4 4 "
            "-1.0 0.5 0.0 0.0 0.5 0.5 1.0 0.5\n")

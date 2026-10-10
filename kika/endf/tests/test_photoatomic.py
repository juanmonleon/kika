"""MF23, MF27 and MF28: the photo-atomic and atomic-relaxation files.

**Two committed fixtures, both openly generated.** ``micro_fe_photoatomic.endf``
is ENDF/B-VIII.0's Fe photo-atomic tape (EPICS2017, MAT 2600) cut to MF1/451,
nine MF23 sections and all four MF27 sections, with every TAB1 thinned to a
handful of points (the first duplicated abscissa of a table, a photoelectric
edge, is kept so the step survives). ``micro_fe_relaxation.endf`` is the
matching EADL tape (NSUB=6) with MF28 cut to two subshells, K's transition
list shortened to seven. Thinning rewrites records, so these are kika's
emitter's bytes and not EPICS's: a verbatim cut would be 1 600 lines for the
same coverage, and the full tapes are reached by the ``tape`` tests below.
The MF1/451 records are the evaluator's, and their directory is rebuilt.

Regenerate after an intentional change, with the downloads in place::

    REGEN_MICRO_TAPES=1 pytest kika/endf/tests/test_photoatomic.py -k regen

**The full tapes** (``photo_2600_26-Fe-0.dat`` and ``ard_2600_26-Fe-0.dat``,
from the IAEA's ENDF-B-VIII.0 ``photo/`` and ``ard/`` directories) are looked
for in ``$KIKA_PHOTOAT`` or ``../kika-files/photoat`` next to this repo.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.endf import form_factors, photoatomic_cross_sections, photoatomic_summary, read_endf
from kika.endf.classes.mf12.base import PhotonTable
from kika.endf.classes.mf23.base import MF23MT, MF27MT, photoatomic_mt_name
from kika.endf.classes.mf28.base import MF28MT
from kika.endf.parsers.parse_endf import MF_PARSERS
from kika.endf.parsers.parse_photoatomic import (
    parse_mf23,
    parse_mf23_mt,
    parse_mf27_mt,
    parse_mf28_mt,
)
from kika.endf.utils import parse_endf_id
from kika.endf.writers import remove_sections, update_mf1_directory

DATA = Path(__file__).resolve().parent / "data"
PHOTOAT_FIXTURE = DATA / "micro_fe_photoatomic.endf"
RELAX_FIXTURE = DATA / "micro_fe_relaxation.endf"

REPO = Path(__file__).resolve().parents[3]
PHOTOAT_DIR = Path(os.environ.get("KIKA_PHOTOAT", REPO.parent / "kika-files" / "photoat"))
FULL_PHOTOAT = PHOTOAT_DIR / "photo_2600_26-Fe-0.dat"
FULL_RELAX = PHOTOAT_DIR / "ard_2600_26-Fe-0.dat"

KEEP_MF23 = {501, 502, 504, 515, 516, 517, 522, 534, 535}
KEEP_MF27 = {502, 504, 505, 506}
POINTS_PER_TABLE = 8


def section_lines(text: str):
    """``{(MF, MT): [lines]}``, SEND records excluded."""
    out = {}
    for line in text.splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf and mt:
            out.setdefault((mf, mt), []).append(line)
    return out


def values_of(lines):
    """Every numeric field of a block, as parsed, so two spellings of one value compare equal."""
    from kika._records import parse_number
    return [parse_number(line[i:i + 11]) for line in lines for i in range(0, 66, 11)]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _thin(table: PhotonTable, n: int) -> PhotonTable:
    x, y = np.asarray(table.x), np.asarray(table.y)
    keep = set(np.unique(np.linspace(0, x.size - 1, min(n, x.size)).round().astype(int)))
    steps = np.flatnonzero(np.diff(x) == 0.0)
    if steps.size:
        keep |= {int(steps[0]), int(steps[0]) + 1}
    idx = sorted(keep)
    return PhotonTable(c1=table.c1, c2=table.c2, l1=table.l1, l2=table.l2,
                       interp=[(len(idx), 2)], x=[table.x[i] for i in idx],
                       y=[table.y[i] for i in idx])


def _splice(text: str, replacements: dict) -> str:
    """Replace each (MF, MT)'s records with new ones, SEND and the rest untouched."""
    out, done = [], set()
    for line in text.splitlines():
        key = None
        if len(line) >= 75:
            _, mf, mt = parse_endf_id(line)
            key = (mf, mt)
        if key in replacements:
            if key not in done:
                out.extend(replacements[key])
                done.add(key)
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def build_photoatomic_fixture(source: Path, dest: Path) -> None:
    content = source.read_text()
    inv = section_lines(content)
    drop = [(mf, mt) for (mf, mt) in inv
            if (mf == 23 and mt not in KEEP_MF23) or (mf == 27 and mt not in KEEP_MF27)]
    trimmed, _ = remove_sections(content, drop)
    sections = section_lines(trimmed)
    replacements = {}
    for (mf, mt), lines in sections.items():
        if mf not in (23, 27):
            continue
        section = (parse_mf23_mt if mf == 23 else parse_mf27_mt)(lines, mt)
        section.table = _thin(section.table, POINTS_PER_TABLE)
        replacements[(mf, mt)] = str(section).splitlines()[:-1]
    dest.write_text(_splice(trimmed, replacements))
    assert update_mf1_directory(str(dest))


def build_relaxation_fixture(source: Path, dest: Path) -> None:
    content = source.read_text()
    lines = section_lines(content)[(28, 533)]
    section = parse_mf28_mt(lines, 533)
    k, n1 = section.subshell(1), section.subshell(10)
    k.values = k.values[:6 * 8]
    k.ntr_as_written = 7
    section.subshells = [k, n1]
    dest.write_text(_splice(content, {(28, 533): str(section).splitlines()[:-1]}))
    assert update_mf1_directory(str(dest))


@pytest.mark.skipif(not os.environ.get("REGEN_MICRO_TAPES"), reason="set REGEN_MICRO_TAPES=1 to rebuild")
def test_regen_photoatomic_fixtures():
    if not FULL_PHOTOAT.is_file() or not FULL_RELAX.is_file():
        pytest.skip(f"full photo-atomic tapes not found in {PHOTOAT_DIR}")
    build_photoatomic_fixture(FULL_PHOTOAT, PHOTOAT_FIXTURE)
    build_relaxation_fixture(FULL_RELAX, RELAX_FIXTURE)


# ---------------------------------------------------------------------------
# Registry and parse of the committed fixtures
# ---------------------------------------------------------------------------

def test_mf23_mf27_mf28_are_registered():
    assert {23, 27, 28} <= set(MF_PARSERS)


@pytest.fixture(scope="module")
def photo():
    return read_endf(str(PHOTOAT_FIXTURE))


@pytest.fixture(scope="module")
def relax():
    return read_endf(str(RELAX_FIXTURE))


def test_fixture_parses_every_section_without_error(photo):
    assert set(photo.files[23].sections) == KEEP_MF23
    assert set(photo.files[27].sections) == KEEP_MF27
    assert photo.files[23].parse_errors == {}
    assert photo.files[27].parse_errors == {}
    assert all(isinstance(s, MF23MT) for s in photo.files[23].sections.values())
    assert all(isinstance(s, MF27MT) for s in photo.files[27].sections.values())


def test_subshell_sections_carry_binding_energy_and_fluorescence_yield(photo):
    k = photo.files[23].sections[534]
    assert k.is_subshell and k.subshell == "K"
    assert k.epe == pytest.approx(7117.0)       # Fe K edge, EPICS2017
    # EPICS2017 leaves EFL at zero on every subshell; it is kept as written.
    assert k.efl == 0.0
    # The cross section starts at the edge: no photoelectric K below the K binding energy.
    assert k.energies[0] == pytest.approx(k.epe)
    total = photo.files[23].sections[501]
    assert not total.is_subshell and total.epe == 0.0 and total.efl == 0.0


def test_mf27_carries_z_and_its_abscissa_unit(photo):
    for mt, s in photo.files[27].sections.items():
        assert s.table_z == 26
        assert s.x_unit == ("eV" if mt in (505, 506) else "1/angstrom")
    F = photo.files[27].sections[502]
    S = photo.files[27].sections[504]
    # F(0) = Z (all electrons scatter in phase) and S(x→∞) = Z.
    assert F.values[0] == pytest.approx(26.0)
    assert S.values[-1] == pytest.approx(26.0)


def test_pair_production_thresholds_name_the_field(photo):
    """515 is the electron field (triplet, 4 m_e c²), 517 the nucleus (2 m_e c²)."""
    s = photo.files[23].sections
    assert s[515].energies[0] == pytest.approx(2.044e6, rel=1e-3)
    assert s[517].energies[0] == pytest.approx(1.022e6, rel=1e-3)
    assert photoatomic_mt_name(515) == "pair production, electron field"
    assert photoatomic_mt_name(517) == "pair production, nuclear field"
    assert photoatomic_mt_name(516) == "pair production, total"


def test_a_photoelectric_edge_survives_as_a_repeated_energy(photo):
    E = photo.files[23].sections[522].energies
    assert np.any(np.diff(E) == 0.0)


@pytest.mark.parametrize("fixture_path", [PHOTOAT_FIXTURE, RELAX_FIXTURE], ids=["photoat", "relax"])
def test_fixture_sections_re_emit_byte_for_byte(fixture_path):
    """The fixtures are kika's own output, so the emitter must reproduce them exactly."""
    endf = read_endf(str(fixture_path))
    source = section_lines(fixture_path.read_text())
    for number in (23, 27, 28):
        mf = endf.files.get(number)
        if mf is None:
            continue
        for mt, section in mf.sections.items():
            have = [line[:75] for line in str(section).splitlines()[:-1]]
            want = [line[:75] for line in source[(number, mt)]]
            assert have == want, f"MF{number}/MT{mt}"


# ---------------------------------------------------------------------------
# MF28
# ---------------------------------------------------------------------------

def test_mf28_subshells_and_transitions(relax):
    mf28 = relax.files[28]
    assert mf28.parse_errors == {}
    section = mf28.sections[533]
    assert isinstance(section, MF28MT)
    assert section.z == 26
    assert [s.name for s in section.subshells] == ["K", "N1"]
    k = section.subshell(1)
    assert k.binding_energy == pytest.approx(7117.0)
    assert k.electrons == pytest.approx(2.0)
    transitions = k.transitions
    assert len(transitions) == 7
    # K-L2 and K-L3 are the Kα lines, radiative (no second subshell).
    assert (transitions[0].subj, transitions[0].subk) == (3, 0) and transitions[0].radiative
    assert transitions[0].energy == pytest.approx(6349.85)
    # The fixture keeps a value the source wrote with a Fortran D exponent.
    assert transitions[5].probability == pytest.approx(8.77226e-6)
    assert any(not t.radiative for t in transitions)
    assert section.subshell(10).transitions == []


def test_fortran_d_exponents_are_read_not_dropped():
    """``6.00794D-6`` would otherwise be dropped from the LIST and shift the rest."""
    head = " 26000.0000 55.3672319          0          0          1          0260028533    1"
    lst = [
        " 1.00000000 0.0                 0          0         12          1260028533    2",
        " 7117.00000 2.00000000 0.0        0.0        0.0        0.0       260028533    3",
        " 8.00000000 0.0        7070.49000 6.00794D-6 0.0        0.0       260028533    4",
    ]
    section = parse_mf28_mt([head] + lst, 533)
    t = section.subshells[0].transitions[0]
    assert t.probability == pytest.approx(6.00794e-6)
    assert t.energy == pytest.approx(7070.49)


def test_mf28_list_length_must_match_ntr():
    head = " 26000.0000 55.3672319          0          0          1          0260028533    1"
    lst = [
        " 1.00000000 0.0                 0          0         12          2260028533    2",
        " 7117.00000 2.00000000 0.0        0.0        0.0        0.0       260028533    3",
        " 8.00000000 0.0        7070.49000 6.00794E-6 0.0        0.0       260028533    4",
    ]
    with pytest.raises(ValueError, match="NTR"):
        parse_mf28_mt([head] + lst, 533)


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------

def _mf23_lines():
    return section_lines(PHOTOAT_FIXTURE.read_text())[(23, 502)]


def test_left_over_records_raise():
    lines = _mf23_lines()
    extra = lines[-1][:75] + f"{len(lines) + 1:5d}"
    with pytest.raises(ValueError, match="left over"):
        parse_mf23_mt(lines + [extra], 502)


def test_a_bad_section_is_recorded_and_the_rest_survive():
    text = PHOTOAT_FIXTURE.read_text()
    lines = []
    for (mf, mt), block in section_lines(text).items():
        if mf != 23:
            continue
        if mt == 504:
            block = block + [block[-1]]     # one record too many
        lines.extend(block)
    mf = parse_mf23(lines)
    assert 504 in mf.parse_errors and "left over" in mf.parse_errors[504]
    assert 504 not in mf.sections
    assert set(mf.sections) == KEEP_MF23 - {504}


# ---------------------------------------------------------------------------
# The accessor module
# ---------------------------------------------------------------------------

def test_summary(photo):
    summary = photoatomic_summary(photo)
    assert summary["z"] == 26
    assert summary["nsub"] == 3
    assert [e["mt"] for e in summary["cross_sections"]] == sorted(KEEP_MF23)
    k = next(e for e in summary["cross_sections"] if e["mt"] == 534)
    assert k["subshell"] == "K" and k["epe"] == pytest.approx(7117.0)
    assert {e["mt"]: e["x_unit"] for e in summary["form_factors"]} == {
        502: "1/angstrom", 504: "1/angstrom", 505: "eV", 506: "eV"}
    assert summary["relaxation"] == [] and summary["parse_errors"] == {}


def test_summary_of_the_relaxation_tape(relax):
    summary = photoatomic_summary(relax)
    assert summary["z"] == 26 and summary["nsub"] == 6
    assert [r["subshell"] for r in summary["relaxation"]] == ["K", "N1"]
    assert summary["relaxation"][0]["n_transitions"] == 7


def test_cross_sections_and_form_factors_are_arrays(photo):
    xs = photoatomic_cross_sections(photo)
    assert set(xs) == KEEP_MF23
    E, sigma = xs[502]
    assert isinstance(E, np.ndarray) and E.shape == sigma.shape
    assert np.all(np.diff(E) >= 0.0)
    assert set(photoatomic_cross_sections(photo, mts=[501, 999])) == {501}
    ff = form_factors(photo)
    assert set(ff) == KEEP_MF27
    x, F = ff[502]
    assert x[0] == 0.0 and F[0] == pytest.approx(26.0)


@pytest.mark.tape
def test_total_is_the_sum_of_the_partials_on_the_full_tape():
    """MT501 = 502 + 504 + 516 + 522, to the precision EPICS writes (tape test)."""
    if not FULL_PHOTOAT.is_file():
        pytest.skip(f"{FULL_PHOTOAT} not found")
    endf = read_endf(str(FULL_PHOTOAT))
    xs = photoatomic_cross_sections(endf)
    E = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    parts = sum(np.interp(E, *xs[mt]) for mt in (502, 504, 516, 522))
    np.testing.assert_allclose(np.interp(E, *xs[501]), parts, rtol=2e-3)


@pytest.mark.tape
def test_full_tapes_parse_every_section_and_keep_every_value():
    """Every MF23/27/28 section of the real Fe tapes, values compared field by field.

    Text is not compared: EPICS writes ``1.00000000`` where kika writes
    ``1.000000+0``. One known value loss on re-emission is pinned here: nine
    significant digits of a negative number in fixed point (``-.572542602``,
    MF27/MT506) do not fit kika's precise formatter, which writes
    ``-5.725426-1``; the parsed value is exact, the re-emitted one is not.
    """
    if not FULL_PHOTOAT.is_file() or not FULL_RELAX.is_file():
        pytest.skip(f"full photo-atomic tapes not found in {PHOTOAT_DIR}")
    lossy = {(27, 506)}
    for path, mfs in ((FULL_PHOTOAT, (23, 27)), (FULL_RELAX, (28,))):
        endf = read_endf(str(path))
        source = section_lines(path.read_text().replace("D-", "E-").replace("D+", "E+"))
        for number in mfs:
            mf = endf.files[number]
            assert mf.parse_errors == {}
            for mt, section in mf.sections.items():
                have = str(section).splitlines()[:-1]
                assert len(have) == len(source[(number, mt)]), f"MF{number}/MT{mt}"
                if (number, mt) in lossy:
                    continue
                assert values_of(have) == values_of(source[(number, mt)]), f"MF{number}/MT{mt}"
    endf = read_endf(str(FULL_PHOTOAT))
    assert set(endf.files[23].sections) == {501, 502, 504, 515, 516, 517, 522, *range(534, 544)}
    assert set(endf.files[27].sections) == {502, 504, 505, 506}

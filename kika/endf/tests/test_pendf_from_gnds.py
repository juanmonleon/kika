"""A PENDF from a GNDS evaluation (``process_gnds(...).write_pendf``).

A GNDS file does not carry the ENDF bookkeeping of its resonance ranges (QX,
LRX, LAD, the particle pairs), so MF2/151 cannot be written back as the
evaluation stated it. A PENDF does not need it: NJOY RECONR never copies MF2
into its PENDF, it writes a default MF2/151 with the scattering radius alone
(one LRU=0 range up to the top of the resolved range, ``reconr.f90`` ``recout``),
and states with LRP=2 that the resonances are in MF3. kika writes the same
sections RECONR does: MF1/451, that MF2/151, and MF3.
"""
from pathlib import Path

import numpy as np
import pytest

import kika
import kika.gnds
from kika.endf import ProcessingPlan, process_endf
from kika.endf.model_adapter.pendf import pendfResonances
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import PENDF_SECTIONS, encodeTapeSections
from kika.nuclear_data.model.resonances import radiusToEndf

ROOT = Path(__file__).resolve().parents[2]
TAPE = ROOT / "endf/tests/data/micro_fe56_structural.endf"
H2_GNDS = ROOT / "gnds/tests/data/n-001_H_002.endf.gnds.xml"
TA182_GNDS = ROOT / "gnds/tests/data/micro_ta182.gnds.xml"
FE56_GNDS = ROOT / "gnds/tests/data/micro_fe56.gnds.xml"


def _mf2Range(path):
    """MF2/151 of a one-material tape, read off the records: (NIS, NER, EL, EH, LRU, NRO, SPI, AP, NLS)."""
    from kika.endf.model_adapter.pendf import _endfFloat
    lines = [l for l in Path(path).read_text().splitlines() if l[70:75] == " 2151"]
    head, isotope, rng, spin = lines[:4]
    field = lambda line, i: line[11 * i:11 * (i + 1)]
    return dict(nis=int(field(head, 4)), ner=int(field(isotope, 4)),
                el=_endfFloat(field(rng, 0)), eh=_endfFloat(field(rng, 1)),
                lru=int(field(rng, 2)), nro=int(field(rng, 4)),
                spi=_endfFloat(field(spin, 0)), ap=_endfFloat(field(spin, 1)),
                nls=int(field(spin, 4)), records=len(lines))


def test_the_pendf_mf2_is_reconrs_radius_only_range():
    """Resolved then unresolved: EH is the top of the resolved range, AP the last stated radius."""
    suite = kika.read(str(TA182_GNDS), format="gnds", covariances=False)
    radius = pendfResonances(suite).scatteringRadius
    assert (radius.domainMin, radius.domainMax) == (1e-5, 35.0)
    assert radius.constant == pytest.approx(8.1271) and radius.unit == "fm"
    suite = kika.read(str(FE56_GNDS), format="gnds", covariances=False)
    radius = pendfResonances(suite).scatteringRadius
    assert radius.domainMax == 8.5e5 and radius.constant == pytest.approx(5.444)


def test_a_pendf_request_writes_reconrs_sections_and_reports_nothing_lost():
    """``files=PENDF_SECTIONS``: what is not asked for (the products' MF4/5/6/12-15,
    the resolved MF2) is neither written nor reported lost."""
    suite = kika.read(str(TA182_GNDS), format="gnds", covariances=False)
    full, report_full, _ = encodeTapeSections(suite)
    assert any("Product" in m for m in report_full.losses)       # the whole tape would lose them
    suite.resonances = pendfResonances(suite)
    sections, report, _ = encodeTapeSections(suite, files=PENDF_SECTIONS)
    assert {(mf, mt) for mf, mt, _ in sections} ==         {(1, 451), (2, 151)} | {(mf, mt) for mf, mt, _ in full if mf == 3}
    assert report.isClean, (report.losses, report.approximations)


def test_a_section_selection_takes_whole_files_and_single_sections():
    from kika.endf.model_adapter.derive import sectionSelected
    assert sectionSelected(1, 451, PENDF_SECTIONS) and not sectionSelected(1, 452, PENDF_SECTIONS)
    assert sectionSelected(3, 102, PENDF_SECTIONS) and sectionSelected(2, None, PENDF_SECTIONS)
    assert not sectionSelected(4, 2, PENDF_SECTIONS) and sectionSelected(1, None, PENDF_SECTIONS)


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    """The micro Fe-56 tape processed from ENDF, and the GNDS kika writes of it processed back."""
    folder = tmp_path_factory.mktemp("m6")
    plan = ProcessingPlan(temperatures=(293.6,), workers=2)
    from_endf = process_endf(TAPE, plan)
    from_gnds = kika.gnds.process_gnds(from_endf.write_gnds(folder / "fe56.gnds.xml"), plan)
    return folder, from_endf, from_gnds


def test_process_gnds_writes_a_pendf(routes):
    folder, from_endf, from_gnds = routes
    path = from_gnds.write_pendf(folder / "gnds.pendf")
    stage = [s for s in from_gnds.stages if s.stage == "write"][-1]
    assert stage.status == "done" and stage.details["temperatures"] == [0.0, 293.6]
    tape = read_endf(str(path))
    assert set(tape.mf) == {1, 2, 3} and set(tape.mf[1].mt) == {451}
    header = tape.mf[1].mt[451]
    assert int(header._lrp) == 2


def test_the_mf2_states_the_evaluations_radius_and_spin(routes):
    folder, from_endf, from_gnds = routes
    one = from_gnds.write_pendf(folder / "gnds_0K.pendf", temperatures=[0.0])
    mf2 = _mf2Range(one)
    # micro_fe56_structural: one Reich-Moore range 1e-5 - 850 keV, SPI 0, AP 0.5444.
    assert mf2 == dict(nis=1, ner=1, el=1e-5, eh=8.5e5, lru=0, nro=0, spi=0.0,
                       ap=pytest.approx(0.5444), nls=0, records=mf2["records"])


@pytest.mark.parametrize("temperature", [0.0, 293.6])
def test_both_doors_write_the_same_mf3(routes, temperature):
    """Same evaluation, same tables: the GNDS PENDF's MF3 is the ENDF PENDF's.

    The two reconstructions agree to rounding (``test_process_gnds_reprocesses_kikas_own_gnds``),
    so the tapes agree to the 7 significant digits ENDF writes."""
    folder, from_endf, from_gnds = routes
    a = read_endf(str(from_endf.write_pendf(folder / f"endf_{temperature}.pendf", temperatures=[temperature])),
                  mf_numbers=[3])
    b = read_endf(str(from_gnds.write_pendf(folder / f"gnds_{temperature}.pendf", temperatures=[temperature])),
                  mf_numbers=[3])
    assert set(a.mf[3].mt) == set(b.mf[3].mt)
    for mt in a.mf[3].mt:
        assert str(b.mf[3].mt[mt]).splitlines()[1:] == str(a.mf[3].mt[mt]).splitlines()[1:], mt


def test_a_radius_only_gnds_evaluation_writes_its_own_range(tmp_path):
    """H-2 has no resonances, only a radius: the range is the radius' own domain."""
    run = kika.gnds.process_gnds(H2_GNDS, ProcessingPlan(temperatures=(600.0,), workers=2))
    path = run.write_pendf(tmp_path / "h2.pendf")
    mf2 = _mf2Range(run.write_pendf(tmp_path / "h2_0K.pendf", temperatures=[0.0]))
    suite = kika.read(str(H2_GNDS), format="gnds", covariances=False)
    radius = suite.resonances.scatteringRadius
    assert (mf2["lru"], mf2["el"], mf2["eh"]) == (0, radius.domainMin, radius.domainMax)
    assert mf2["ap"] == pytest.approx(radiusToEndf(radius.constant))
    back = read_endf(str(path), mf_numbers=[3])
    for mt, (energies, values) in run.cross_sections(600.0).items():
        if mt not in back.mf[3].mt:
            continue
        from kika.endf.model_adapter import decodeMF3MT
        form = decodeMF3MT(back.mf[3].mt[mt])[0].crossSection["eval"]
        xs, ys, _ = form.toEndfRegions()
        # ENDF's 7 significant digits; a threshold's 1e-13 b first point is noise
        # at that scale, so the floor is the table's own.
        np.testing.assert_allclose(np.interp(energies, xs, ys), values, rtol=1e-6,
                                   atol=1e-9 * np.abs(values).max())

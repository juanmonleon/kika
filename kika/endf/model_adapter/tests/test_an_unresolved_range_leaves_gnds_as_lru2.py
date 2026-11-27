"""An unresolved range read from GNDS is written as the LRU=2 range the tape had.

GNDS holds the averages per (L, J) with no ENDF bookkeeping. The case comes
from FUDGE's ``LRF1``/``LFW<n>`` flags, else LRF=2 and LFW from fission in the
suite; widths go to ENDF's columns by their reaction's MT, a column the region
lacks is written 0, INT is the level spacing's interpolation and AWRI the
target's mass over the neutron's in PoPs. On NNDC's ENDF/B-VIII.1 GNDS (FUDGE's
translation) every record comes back as the tape has it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._nuclear_data import nuclear_data

import kika
from kika.endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _urr(path):
    (isotope,) = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151].isotopes
    (energyRange,) = [r for r in isotope.energy_ranges if r.lru == 2]
    p = energyRange.parameters
    head = (isotope.za, isotope.abn, isotope.lfw, energyRange.el, energyRange.eh,
            energyRange.lrf, energyRange.nro, energyRange.naps, type(p).__name__,
            p.spi, p.ap, p.lssf, p.nls, list(getattr(p, "energies", None) or []))
    body = []
    for block in p.l_values:
        body.append((block.l, block.awri, len(block.j_states)))
        for state in block.j_states:
            fields = dict(vars(state))
            points = fields.pop("energy_points", None) or []
            body.append(sorted(fields.items(), key=str))
            body.extend(sorted(vars(point).items()) for point in points)
    return head, body


def _nndc(name):
    tapes = nuclear_data("endf") or Path("missing")
    gnds = tapes.parent / "gnds" / "endfb81"
    endf, xml = tapes / "endfb81" / f"{name}.endf", gnds / f"{name}.endf.gnds.xml"
    if not (endf.is_file() and xml.is_file()):
        pytest.skip(f"needs {endf} and {xml}")
    return endf, xml


def _written(suite, tmp_path, stem):
    out = tmp_path / f"{stem}.endf"
    report = kika.write(suite, str(out), format="endf")
    return out, report


def test_mn55_comes_back_record_for_record_through_kikas_gnds(tmp_path):
    source = DATA / "micro_mn55_mf2.endf"
    gnds = tmp_path / "mn55.gnds.xml"
    kika.write(kika.read(str(source)), str(gnds), format="gnds")
    out, report = _written(kika.read(str(gnds)), tmp_path, "mn55")
    assert _urr(out) == _urr(source)
    assert not any("unresolved" in message for message in report.losses)


@pytest.mark.parametrize("name", [
    "n-026_Fe_058",  # case A (LRF1 flag)
    "n-025_Mn_055",  # case C, LFW=0, a competitive width and no fission
    "n-092_U_235",   # case C, LFW=1, INT=2, no competitive width
    "n-095_Am_241",  # case C, LFW=1, INT=5
    "n-040_Zr_091",  # case C, LFW=1 on a non-fissile target (LFW1 flag)
])
def test_nndc_gnds_gives_the_tapes_urr(name, tmp_path):
    endf, xml = _nndc(name)
    out, report = _written(kika.read(str(xml)), tmp_path, name)
    assert _urr(out) == _urr(endf)
    assert not any("unresolved" in message for message in report.losses)


def test_its_average_cross_sections_are_the_tapes(tmp_path):
    from kika.endf.reconstruction import context_from_endf
    from kika.processing.resonances.unresolved import evaluate_unresolved, prepare_unresolved

    endf, xml = _nndc("n-038_Sr_090")  # LSSF=0, INT=5
    out, _ = _written(kika.read(str(xml)), tmp_path, "sr90")
    sigma = []
    for path in (endf, out):
        suite = kika.read(str(path))
        context, _ = context_from_endf(read_endf(str(path)), suite)
        region = suite.resonances.unresolved
        energies = np.geomspace(region.domainMin, region.domainMax, 2001)[1:-1]
        sigma.append(evaluate_unresolved(energies, prepare_unresolved(region, context, [])))
    assert sigma[0].keys() == sigma[1].keys()
    assert any(np.any(values) for values in sigma[0].values())
    for mt in sigma[0]:
        assert np.array_equal(sigma[0][mt], sigma[1][mt]), mt


def test_a_case_a_region_with_tabulated_widths_is_refused(tmp_path):
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    gnds = tmp_path / "mn55.gnds.xml"
    kika.write(kika.read(str(DATA / "micro_mn55_mf2.endf")), str(gnds), format="gnds")
    suite = kika.read(str(gnds))
    href = "/reactionSuite/resonances/unresolved/tabulatedWidths[@label='eval']"
    EndfConversionFlags.of(suite).conversions.append((href, "LRF1"))
    out, report = _written(suite, tmp_path, "mn55")
    (isotope,) = read_endf(str(out), mf_numbers=[2]).mf[2].mt[151].isotopes
    assert [r.lru for r in isotope.energy_ranges] == [1]
    assert any("case A (LRF=1) states one D, GN0 and GG per J" in message
               for message in report.losses)

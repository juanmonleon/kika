"""An R-Matrix Limited range read from GNDS is written as MF2/151 LRF=7.

The pairs and channels come from the model as FUDGE's ``toENDF6`` writes them.
On NNDC's ENDF/B-VIII.1 GNDS (FUDGE's translation) Si-28, Fe-54 and Rh-103 come
back record for record. Cl-35 differs in two fields no GNDS node holds, NAPS and
the capture residual's Z, both unused by the reconstruction; its 0 K resonance
cross sections are the tape's bit for bit. kika's own GNDS of Si-28 carries the
capture residual and the boundary condition, and comes back the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._nuclear_data import nuclear_data

import kika
from kika.endf import read_endf


def _nndc(name):
    tapes = nuclear_data("endf") or Path("missing")
    gnds = tapes.parent / "gnds" / "endfb81"
    endf, xml = tapes / "endfb81" / f"{name}.endf", gnds / f"{name}.endf.gnds.xml"
    if not (endf.is_file() and xml.is_file()):
        pytest.skip(f"needs {endf} and {xml}")
    return endf, xml


def _written(xml, tmp_path):
    out = tmp_path / f"{xml.stem}.endf"
    report = kika.write(kika.read(str(xml), covariances=False), str(out), format="endf")
    return out, report


def _records(path):
    (isotope,) = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151].isotopes
    out = [("isotope", isotope.za, isotope.abn, isotope.lfw)]
    for r in isotope.energy_ranges:
        out.append(("range", r.el, r.eh, r.lru, r.lrf, r.nro, r.naps))
        p = r.parameters
        if r.lrf != 7:
            continue
        out.append(("head", p.spi, p.ap, p.ifg, p.krm, p.krl, len(p.spin_groups)))
        out += [("pair", sorted(vars(pair).items())) for pair in p.particle_pairs]
        for group in p.spin_groups:
            out.append(("group", group.aj, group.pj, group.kbk, group.kps))
            out += [("channel", sorted(vars(c).items())) for c in group.channels]
            out += [("resonance", x.er, list(x.widths)) for x in group.resonances]
    return out


@pytest.mark.parametrize("name", ["n-014_Si_028", "n-026_Fe_054", "n-045_Rh_103"])
def test_the_range_comes_back_record_for_record(name, tmp_path):
    endf, xml = _nndc(name)
    out, report = _written(xml, tmp_path)
    assert _records(out) == _records(endf)
    assert not any("MF2" in message for message in report.losses)


def test_cl35_differs_only_where_gnds_has_no_node(tmp_path):
    endf, xml = _nndc("n-017_Cl_035")
    out, _ = _written(xml, tmp_path)
    differ = [(a, b) for a, b in zip(_records(endf), _records(out)) if a != b]
    assert [a[0] for a, _ in differ] == ["range", "pair"]
    (tape, kika_), (tapePair, kikaPair) = differ
    assert (tape[-1], kika_[-1]) == (0, 1)
    assert (dict(tapePair[1])["zb"], dict(kikaPair[1])["zb"]) == (0, 17)


def test_kikas_own_gnds_of_si28_comes_back_record_for_record(tmp_path):
    endf, _ = _nndc("n-014_Si_028")
    gnds = tmp_path / "si28.gnds.xml"
    kika.write(kika.read(str(endf), covariances=False), str(gnds), format="gnds")
    out, report = _written(gnds, tmp_path)
    assert _records(out) == _records(endf)
    assert not any("MF2" in message for message in report.losses)


@pytest.mark.parametrize("name", ["n-017_Cl_035", "n-014_Si_028"])
def test_its_resonance_cross_sections_are_the_tapes(name, tmp_path):
    from kika.endf.reconstruction import context_from_endf
    from kika.processing.resonances.prepare import prepare_resonances

    endf, xml = _nndc(name)
    out, _ = _written(xml, tmp_path)
    sigma = []
    for path in (endf, out):
        suite = kika.read(str(path), covariances=False)
        context, _ = context_from_endf(read_endf(str(path), mf_numbers=[1, 2, 3]), suite)
        region = suite.resonances.resolved[0]
        energies = np.geomspace(max(region.domainMin, 1e-3), region.domainMax, 4001)[:-1]
        sigma.append(prepare_resonances(suite.resonances, context).evaluate(energies))
    assert sigma[0].keys() == sigma[1].keys()
    for mt in sigma[0]:
        assert np.array_equal(sigma[0][mt], sigma[1][mt]), mt

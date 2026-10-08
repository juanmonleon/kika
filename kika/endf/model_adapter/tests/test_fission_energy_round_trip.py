"""MF1/458 ↔ ``fissionEnergyRelease``, and MF1/460 kept verbatim.

The gate is the one :mod:`test_nubar_round_trip` uses for the nu-bars: the
section written from the model is the section the flat class writes from the
file, character for character. Swept on 2026-10-08 over all 161 MF1/458 of
ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 on this machine, 161 of 161; the three
committed fixtures are one per shape the libraries use.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.classes.mf1.mf1mt460 import MF1MT460
from kika.endf.model_adapter import (decodeReactionSuite, encodeMF1MT458,
                                     encodeMF1MT460)
from kika.endf.model_adapter.fission_energy import (MF1MT460_KEY,
                                                    attachFissionEnergyRelease,
                                                    fissionEnergyReleaseNode)
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import encodeTapeSections, writeEndfTape
from kika.nuclear_data.model import (ConversionReport, EndfProvenance,
                                     FissionEnergyRelease, Polynomial1d,
                                     Regions1d)


def _decode(path):
    endf = read_endf(str(path))
    suite, report = decodeReactionSuite(endf)
    return endf, suite, report


def test_the_section_written_from_the_model_is_the_section_read(micro_fission_energy_tape):
    endf, suite, _report = _decode(micro_fission_energy_tape)
    section, _ = encodeMF1MT458(suite)
    assert str(section) == str(endf.mf[1].mt[458])


def test_the_node_sits_on_the_fission_channel_with_nine_terms(micro_fission_energy_tape):
    _endf, suite, _report = _decode(micro_fission_energy_tape)
    data = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData
    assert len(data.fissionEnergyReleases) == 1
    node = data.fissionEnergyReleases[0]
    assert node is fissionEnergyReleaseNode(suite)
    assert [name for name, _ in node.terms()] == list(FissionEnergyRelease.TERMS)


def test_a_polynomial_carries_the_file_coefficients_and_their_uncertainties(
        micro_fission_energy_tape):
    endf, suite, _report = _decode(micro_fission_energy_tape)
    section = endf.mf[1].mt[458]
    node = fissionEnergyReleaseNode(suite)
    values = section._coefficients if section.lfc == 0 else section._thermal_values
    orders = len(values) // 18
    for index, name in enumerate(FissionEnergyRelease.TERMS):
        term = getattr(node, name)
        if not isinstance(term, Polynomial1d):
            continue
        expected = [values[o * 18 + 2 * index] for o in range(orders)]
        sigma = [values[o * 18 + 2 * index + 1] for o in range(orders)]
        assert list(term.coefficients) == expected, name
        assert list(term.uncertainty.standard.coefficients) == sigma, name
        assert term.axes.axes[0].unit == "eV" and term.axes.axes[1].unit == "eV"


def test_the_written_tape_carries_mf1_458_back(micro_fission_energy_tape, tmp_path):
    endf, suite, report = _decode(micro_fission_energy_tape)
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out, report=report)
    written = read_endf(str(out))
    assert sorted(written.mf[1].mt) == sorted(endf.mf[1].mt)
    assert str(written.mf[1].mt[458]) == str(endf.mf[1].mt[458])


def test_lfc1_keeps_what_ldrv_and_the_thermal_list_said():
    """U-235 B-VIII.1: EFR is a primary evaluation (LDRV=2), ENP/END/EGP derived."""
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2] / "tests" / "data"
            / "micro_u235_fission_energy.endf")
    endf, suite, report = _decode(path)
    node = fissionEnergyReleaseNode(suite)
    tabulated = [name for name, f in node.terms() if not isinstance(f, Polynomial1d)]
    assert tabulated == ["promptProductKE", "promptNeutronKE",
                         "delayedNeutronKE", "promptGammaEnergy"]
    header = node.provenance.headerFields
    assert header["ldrv"] == {"promptProductKE": 2, "promptNeutronKE": 1,
                              "delayedNeutronKE": 1, "promptGammaEnergy": 1}
    assert header["replaced"]["promptProductKE"][0] == endf.mf[1].mt[458]._thermal_values[0]
    assert any("LFC=1" in line for line in report.unsupported)

    # MF1/455 made the fissionFragmentData first; MF1/458 added to it.
    data = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData
    assert len(data.delayedNeutrons) > 0


def test_an_edited_term_is_written_from_the_model(micro_fission_energy_tape):
    _endf, suite, _report = _decode(micro_fission_energy_tape)
    node = fissionEnergyReleaseNode(suite)
    node.neutrinoEnergy.coefficients = node.neutrinoEnergy.coefficients * 1.5
    section, _ = encodeMF1MT458(suite)
    values = section._coefficients if section.lfc == 0 else section._thermal_values
    assert values[12] == pytest.approx(float(node.neutrinoEnergy.coefficients[0]))


def test_a_table_with_no_provenance_is_written_and_says_what_it_assumed(
        micro_fission_energy_tape):
    _endf, suite, _report = _decode(micro_fission_energy_tape)
    node = fissionEnergyReleaseNode(suite)
    if any(isinstance(f, Polynomial1d) and len(f.coefficients) > 1 for _, f in node.terms()):
        pytest.skip("a polynomial of degree > 0 cannot sit next to a table")
    node.provenance = None
    node.promptProductKE = Regions1d.fromEndfRegions([1e-5, 2e7], [1.7e8, 1.65e8], [(2, 2)])
    report = ConversionReport()
    section, report = encodeMF1MT458(suite, report=report)
    assert section.lfc == 1
    assert section._tab_components[0]["ldrv"] == 1
    said = "\n".join(report.approximations)
    assert "no LDRV was kept" in said and "no thermal value was kept" in said


def test_a_polynomial_of_higher_degree_next_to_a_table_is_refused():
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2] / "tests" / "data"
            / "micro_ac227_fission_energy.endf")
    _endf, suite, _report = _decode(path)
    node = fissionEnergyReleaseNode(suite)
    node.promptProductKE = Regions1d.fromEndfRegions([1e-5, 2e7], [1.7e8, 1.65e8], [(2, 2)])
    with pytest.raises(ValueError, match="no format for the pair"):
        encodeMF1MT458(suite)


def test_mf1_460_has_no_node_and_comes_back_verbatim():
    """No tape on this machine carries one, so the section is built by hand."""
    section = MF1MT460(_za=94239.0, _awr=236.9986, _mat=9437, _lo=2,
                       _nnf=3, _decay_constants=[0.0127, 0.0317, 0.115])
    mf1 = type("MF1", (), {"mt": {460: section}})()
    suite = type("Suite", (), {"provenance": EndfProvenance(mat=9437)})()
    report = attachFissionEnergyRelease(suite, mf1, ConversionReport())

    assert MF1MT460_KEY in suite.provenance.headerFields
    assert any("MF1/460" in line for line in report.unsupported)
    back, _ = encodeMF1MT460(suite, mat=9437)
    assert str(back) == str(section)


def test_a_tape_without_458_writes_no_458(micro_nubar_tape):
    _endf, suite, _report = _decode(micro_nubar_tape)
    assert fissionEnergyReleaseNode(suite) is None
    sections, _report, _mat = encodeTapeSections(suite)
    assert (1, 458) not in {(mf, mt) for mf, mt, _ in sections}


# ----------------------------------------------------------------------
# E3b: the same node through GNDS (gnds.xsd:1296-1335)
# ----------------------------------------------------------------------

def _through_gnds(path, tmp_path):
    import kika

    suite = kika.read(str(path))
    out = tmp_path / "written.gnds.xml"
    kika.write(suite, str(out), format="gnds")
    return suite, kika.read(str(out)), out


def test_the_fission_fragment_data_survives_gnds(micro_fission_energy_tape, tmp_path):
    before, after, _out = _through_gnds(micro_fission_energy_tape, tmp_path)
    one = before.findReactionByENDF_MT(18).outputChannel.fissionFragmentData
    two = after.findReactionByENDF_MT(18).outputChannel.fissionFragmentData

    assert [(f.label, f.rate.value, f.rate.unit) for f in one.delayedNeutrons] == \
        [(f.label, f.rate.value, f.rate.unit) for f in two.delayedNeutrons]

    first, second = one.fissionEnergyReleases[0], two.fissionEnergyReleases[0]
    assert second.label == first.label
    for name in FissionEnergyRelease.TERMS:
        a, b = getattr(first, name), getattr(second, name)
        assert type(a) is type(b), name
        if isinstance(a, Polynomial1d):
            assert np.array_equal(a.coefficients, b.coefficients), name
            assert np.array_equal(a.uncertainty.standard.coefficients,
                                  b.uncertainty.standard.coefficients), name
            assert (a.domainMin, a.domainMax) == (b.domainMin, b.domainMax), name
        else:
            assert np.array_equal(a.xs, b.xs) and np.array_equal(a.ys, b.ys), name
            assert a.interpolation == b.interpolation, name


def test_the_written_fission_fragment_data_is_schema_valid(micro_fission_energy_tape,
                                                          tmp_path):
    """No schema error inside fissionEnergyReleases or a delayedNeutron's rate.

    The delayedNeutron's product does not validate -- its multiplicity and
    distribution are MF5/455's per-family spectra, which the model does not
    carry yet -- and neither do several things outside §18.4 on a cut tape
    (an MT18 with no MF5 has an empty distribution). Those are counted
    elsewhere; this asserts on the nodes E3b writes.
    """
    from kika.gnds.tests.test_encode import _schemaErrors

    _before, _after, out = _through_gnds(micro_fission_energy_tape, tmp_path)
    errors = _schemaErrors(out)
    # By the element the error is *about* -- "Element 'multiplicity': ...
    # Expected is one of (..., polynomial1d, ...)" names polynomial1d and is not
    # about one.
    tags = {"fissionFragmentData", "delayedNeutrons", "delayedNeutron", "rate",
            "double", "fissionEnergyReleases", "fissionEnergyRelease",
            "polynomial1d", "XYs1d", "uncertainty", "axes", "axis", "values",
            *FissionEnergyRelease.TERMS}
    ours = [e for e in errors if e.split("'")[1] in tags]
    assert ours == []


def test_an_lnu1_polynomial_now_carries_its_required_domain(tmp_path):
    """xData_polynomial_1d_primary makes domainMin/domainMax required."""
    import xml.etree.ElementTree as ET

    from kika.gnds.encode import _function

    parent = ET.Element("x")
    form = Polynomial1d(coefficients=np.array([2.4, 1e-7]), domainMin_=1e-5,
                        domainMax_=2e7)
    element = _function(parent, form, ConversionReport(), "test")
    assert element.attrib["domainMin"] and element.attrib["domainMax"]

"""URR semantics the model has to carry before R6 computes anything.

Each test pins a defect the corpus round trip found (workspace
``resonance_oracle/tools/urr_roundtrip.py`` over B-VIII.1, JEFF-4.0, JENDL-5):
a value that read back unchanged into ENDF but meant something else in the
model, or was dropped on the way to GNDS.
"""
import xml.etree.ElementTree as ET

import numpy as np

from kika.endf.classes.mf2.mf2mt151 import (
    MF2MT151, Isotope, EnergyRange, UnresolvedCaseA, URR_LValue_CaseA, URR_JState_CaseA,
    UnresolvedCaseB, URR_LValue_CaseB, URR_JState_CaseB)
from kika.endf.model_adapter import decodeMF2MT151, encodeMF2MT151
from kika.endf.utils import parse_data_values
from kika.gnds.encode_resonances import writeResonances
from kika.gnds.resonances import readResonances
from kika.nuclear_data.model.conversion import ConversionReport


def section(parameters, lrf=1, lfw=0):
    s = MF2MT151(number=151)
    s._za, s._awr, s._mat, s._nis = 26058, 57.44, 2637, 1
    s._isotopes = [Isotope(26058, 1., lfw, 1, [EnergyRange(4e5, 1e6, 2, lrf, 0, 0, parameters)])]
    return s


def caseA():
    return UnresolvedCaseA(0., .6, 1, 2, [
        URR_LValue_CaseA(57.44, 0, [URR_JState_CaseA(d=2.e4, aj=.5, amun=1., gn0=3.e0, gg=.9)]),
        URR_LValue_CaseA(57.44, 1, [URR_JState_CaseA(d=2.e4, aj=.5, amun=2., gn0=1.e0, gg=.4),
                                    URR_JState_CaseA(d=1.e4, aj=1.5, amun=2., gn0=1.e0, gg=.4)])])


def channels(group):
    return {c.label: c for c in group.channels}


def gnds_round_trip(model):
    root = ET.Element("reactionSuite")
    written = ConversionReport()
    writeResonances(root, model, written, ("1e-5", "2e7"))
    read = ConversionReport()
    return readResonances(root.find("resonances"), "/reactionSuite", None, read,
                          lambda element: None), written, read


def test_case_a_capture_does_not_fluctuate():
    """ENDF case A has no AMUG: the capture width is fixed, ν = 0, not 1."""
    model, provenance, report = decodeMF2MT151(section(caseA()))
    assert report.isClean
    groups = model.unresolved.tabulatedWidths.spinGroups
    assert [channels(g)["capture"].degreesOfFreedom for g in groups] == [0., 0., 0.]
    assert [channels(g)["neutron"].degreesOfFreedom for g in groups] == [1., 2., 2.]
    assert str(encodeMF2MT151(model, provenance)) == str(section(caseA()))


def test_case_b_capture_is_fixed_and_fission_keeps_muf():
    parameters = UnresolvedCaseB(0., .9, 0, 3, 1, [1e3, 5e3, 1e4], [
        URR_LValue_CaseB(235., 0, [URR_JState_CaseB(d=.5, aj=3., amun=1., gn0=1e-4, gg=.04,
                                                    muf=2, gf=[.1, .2, .3])])])
    model, provenance, report = decodeMF2MT151(section(parameters, lrf=1, lfw=1))
    assert report.isClean
    group = model.unresolved.tabulatedWidths.spinGroups[0]
    assert channels(group)["capture"].degreesOfFreedom == 0.
    assert channels(group)["fission"].degreesOfFreedom == 2.
    assert str(encodeMF2MT151(model, provenance)) == str(section(parameters, lrf=1, lfw=1))


def test_case_a_level_spacing_reaches_gnds_as_a_constant():
    """It was a one-element array with no grid, and the writer dropped it."""
    model, _, _ = decodeMF2MT151(section(caseA()))
    reread, written, read = gnds_round_trip(model)
    assert not any("levelSpacing" in line for line in written.losses)
    groups = reread.unresolved.tabulatedWidths.spinGroups
    assert [float(np.asarray(g.levelSpacing)[0]) for g in groups] == [2.e4, 2.e4, 1.e4]
    assert [channels(g)["capture"].degreesOfFreedom for g in groups] == [0., 0., 0.]


def test_case_b_constant_spacing_is_not_put_against_the_fission_grid():
    parameters = UnresolvedCaseB(0., .9, 0, 2, 1, [1e3, 1e4], [
        URR_LValue_CaseB(235., 0, [URR_JState_CaseB(d=.5, aj=3., amun=1., gn0=1e-4, gg=.04,
                                                    muf=1, gf=[.1, .2])])])
    model, _, _ = decodeMF2MT151(section(parameters, lrf=1, lfw=1))
    reread, written, _ = gnds_round_trip(model)
    assert not any("levelSpacing" in line for line in written.losses)
    group = reread.unresolved.tabulatedWidths.spinGroups[0]
    assert float(np.asarray(group.levelSpacing)[0]) == .5


def test_an_absent_gnds_degrees_of_freedom_is_a_fixed_width():
    model, _, _ = decodeMF2MT151(section(caseA()))
    root = ET.Element("reactionSuite")
    writeResonances(root, model, ConversionReport(), ("1e-5", "2e7"))
    for width in root.iter("width"):
        width.attrib.pop("degreesOfFreedom", None)
    reread = readResonances(root.find("resonances"), "/reactionSuite", None,
                            ConversionReport(), lambda element: None)
    group = reread.unresolved.tabulatedWidths.spinGroups[0]
    assert {c.degreesOfFreedom for c in group.channels} == {0.}


def test_a_blank_field_inside_a_list_body_is_zero():
    """JEFF-4.0 Gd-155/157 leave RML channel columns blank; skipping them
    shifted the rest of the list and lost the whole MF2."""
    lines = [" 1.000000+0 0.000000+0 0.000000+0 0.000000+0" + " " * 22 + "6434 2151    1",
             " 2.000000+0 0.000000+0-1.000000+0 0.000000+0 7.900000-1 7.900000-16434 2151    2"]
    values, index = parse_data_values(lines, 0, 12)
    assert index == 2
    assert values == [1., 0., 0., 0., 0., 0., 2., 0., -1., 0., .79, .79]


def test_trailing_blanks_after_the_last_value_are_not_read():
    lines = [" 1.000000+0 2.000000+0 3.000000+0" + " " * 33 + "6434 2151    1"]
    values, _ = parse_data_values(lines, 0, 3)
    assert values == [1., 2., 3.]


# ---------------------------------------------------------------------------
# P2: the URR's own radius policy (NAPS, NRO=1)
# ---------------------------------------------------------------------------

from dataclasses import replace

from kika.endf.classes.mf2.mf2mt151 import (
    EnergyDependentScatteringRadius, ScatteringRadiusOnly,
    UnresolvedCaseC, URR_LValue_CaseC, URR_JState_CaseC, URR_EnergyPoint)
from kika.nuclear_data.model.resonances import ScatteringRadius


def caseC():
    points = [URR_EnergyPoint(e, 20., 0., 1e-3, .1, 0.) for e in (2e3, 1e4, 1e5)]
    return UnresolvedCaseC(1.5, .9, 0, 1, [
        URR_LValue_CaseC(195.27, 0, [URR_JState_CaseC(1., 2, 0., 1., 0., 0., points)])])


def au197_like(nro=1, naps=0):
    """A resolved-side range with its own AP, then a URR with NRO/NAPS: Au-197's shape."""
    s = MF2MT151(number=151)
    s._za, s._awr, s._mat, s._nis = 79197, 195.27, 7925, 1
    apE = (EnergyDependentScatteringRadius([(3, 2)], [2e3, 1e4, 1e5], [.8835121, .9, .9136729])
           if nro else None)
    s._isotopes = [Isotope(79197, 1., 0, 2, [
        EnergyRange(1e-5, 2e3, 0, 0, 0, 0, ScatteringRadiusOnly(1.5, .76)),
        EnergyRange(2e3, 1e5, 2, 2, nro, naps, caseC(), apE)])]
    return s


def test_the_urr_table_is_the_urrs_and_not_the_evaluations():
    model, provenance, report = decodeMF2MT151(au197_like())
    assert report.isClean
    policy = model.unresolved.tabulatedWidths.radiusPolicy
    assert policy.channelMode == "mass"
    np.testing.assert_allclose(policy.phaseRadius.values, [8.835121, 9., 9.136729])
    assert policy.phaseRadius.interpolation == [(3, 2)]
    # The global radius is the first range's constant, as for any evaluation
    # whose resolved side states no table.
    assert not model.scatteringRadius.isEnergyDependent
    assert model.scatteringRadius.constant == 7.6
    assert str(encodeMF2MT151(model, provenance)) == str(au197_like())


def test_naps_one_without_a_table_is_phase_mode():
    model, provenance, _ = decodeMF2MT151(au197_like(nro=0, naps=1))
    policy = model.unresolved.tabulatedWidths.radiusPolicy
    assert policy.channelMode == "phase" and policy.phaseRadius is None
    assert str(encodeMF2MT151(model, provenance)) == str(au197_like(nro=0, naps=1))


def test_an_edited_urr_policy_reaches_the_file():
    model, provenance, _ = decodeMF2MT151(au197_like())
    widths = model.unresolved.tabulatedWidths
    table = widths.radiusPolicy.phaseRadius
    widths.radiusPolicy = replace(widths.radiusPolicy, channelMode="phase",
                                  phaseRadius=replace(table, values=np.asarray(table.values) * 1.1))
    rng = encodeMF2MT151(model, provenance).isotopes[0].energy_ranges[1]
    assert (rng.nro, rng.naps) == (1, 1)
    np.testing.assert_allclose(rng.ap_e.ap_values, [.97186331, .99, 1.00504019])
    widths.radiusPolicy = replace(widths.radiusPolicy, phaseRadius=None)
    rng = encodeMF2MT151(model, provenance).isotopes[0].energy_ranges[1]
    assert (rng.nro, rng.ap_e) == (0, None)


def test_the_urr_policy_survives_gnds_as_fudge_writes_it():
    model, _, _ = decodeMF2MT151(au197_like())
    root = ET.Element("reactionSuite")
    written = ConversionReport()
    writeResonances(root, model, written, ("1e-5", "2e7"))
    node = root.find("resonances/unresolved/tabulatedWidths")
    assert node.attrib["calculateChannelRadius"] == "true"
    assert node.find("hardSphereRadius/XYs1d") is not None
    assert node.find("scatteringRadius/constant1d") is not None
    read = ConversionReport()
    reread = readResonances(root.find("resonances"), "/reactionSuite", None, read, lambda e: None)
    policy = reread.unresolved.tabulatedWidths.radiusPolicy
    assert policy.channelMode == "mass"
    np.testing.assert_allclose(policy.phaseRadius.values, [8.835121, 9., 9.136729])
    assert reread.unresolved.tabulatedWidths.scatteringRadius == 9.


def test_a_phase_mode_table_is_the_gnds_scattering_radius():
    model, _, _ = decodeMF2MT151(au197_like(nro=1, naps=1))
    root = ET.Element("reactionSuite")
    writeResonances(root, model, ConversionReport(), ("1e-5", "2e7"))
    node = root.find("resonances/unresolved/tabulatedWidths")
    assert node.attrib["calculateChannelRadius"] == "false"
    assert node.find("scatteringRadius/XYs1d") is not None and node.find("hardSphereRadius") is None
    reread = readResonances(root.find("resonances"), "/reactionSuite", None,
                            ConversionReport(), lambda e: None)
    policy = reread.unresolved.tabulatedWidths.radiusPolicy
    assert policy.channelMode == "phase" and policy.phaseRadius.isEnergyDependent

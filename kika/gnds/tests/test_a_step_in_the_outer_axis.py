"""A step in an XYs2d's outer axis is written as FUDGE writes it, and the PoPs note once."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import kika
from kika.gnds.encode import _function
from kika.nuclear_data.model import ConversionReport, XYs1d, XYs2d, energyAxes

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"


def _spectrum(outer, height, axes):
    return XYs1d(xs=np.array([0.0, 1.0e6]), ys=np.array([height, height]),
                 outerDomainValue=outer, axes=axes)


def _stepped(axes):
    # U-238 B-VIII.1 MF5/MT18 has this shape: the grid repeats 20 MeV with two
    # different spectra and goes on to 30 MeV.
    return XYs2d(axes=axes, label="eval", function1ds=[
        _spectrum(1.0e-5, 1.0, axes), _spectrum(2.0e7, 1.0, axes),
        _spectrum(2.0e7, 2.0, axes), _spectrum(3.0e7, 2.0, axes)])


def test_a_step_becomes_two_regions_that_share_the_outer_value():
    axes = energyAxes()
    root, report = ET.Element("root"), ConversionReport()
    _function(root, _stepped(axes), report, "test")

    assert root.find("XYs2d") is None
    regions = root.find("regions2d")
    assert regions.attrib == {"label": "eval"}
    assert regions[0].tag == "axes"
    children = list(regions.find("function2ds"))
    assert [child.attrib["index"] for child in children] == ["0", "1"]
    assert [[float(node.attrib["outerDomainValue"]) for node in child.find("function1ds")]
            for child in children] == [[1.0e-5, 2.0e7], [2.0e7, 3.0e7]]
    assert all(child.find("axes") is None for child in children)
    assert not report.warnings


def test_an_identical_repeat_is_still_dropped_and_not_split():
    axes = energyAxes()
    form = XYs2d(axes=axes, function1ds=[
        _spectrum(1.0, 1.0, axes), _spectrum(2.0, 1.0, axes), _spectrum(2.0, 1.0, axes)])
    root = ET.Element("root")
    _function(root, form, ConversionReport(), "test")
    assert len(root.find("XYs2d").find("function1ds")) == 2


def test_inside_a_regions_container_the_step_is_kept_and_reported():
    axes = energyAxes()
    root, report = ET.Element("root"), ConversionReport()
    _function(root, _stepped(axes), report, "test", nested=True, parentAxes=axes, index=0)
    assert root.find("XYs2d") is not None
    assert any("two different distributions" in message for message in report.warnings)


def test_the_pops_note_is_said_once_per_file(tmp_path):
    suite = kika.read(DATA / "micro_fe56_xs_and_angular.endf")
    report = kika.write(suite, tmp_path / "out.xml", format="gnds")
    assert sum("PoPs serialization" in message for message in report.warnings) == 1

"""A style's documentation: what the model carries comes back, the rest is named.

``documentation.py`` arrived without a test of its own, and ``capabilities()``
kept saying the element was written back empty. This file is what the
``documentationModelled`` group of ``_capabilities.py`` stands on.
"""
import xml.etree.ElementTree as ET

from kika.gnds.documentation import readDocumentation, writeDocumentation
from kika.nuclear_data.model.conversion import ConversionReport

SOURCE = """<documentation doi="10.1/x" version="8.1">
  <authors><author name="A. Evaluator" orcid="0000-0001"/></authors>
  <dates><date value="2024-02-01" dateType="issued"/></dates>
  <title>Fe-56</title>
  <body encoding="ascii" markup="none">free text</body>
  <endfCompatible> 2.605600+4 5.545443+1 ...</endfCompatible>
  <keywords><keyword>iron</keyword></keywords>
</documentation>"""


def test_the_modelled_children_round_trip_and_the_rest_is_reported():
    report = ConversionReport()
    document = readDocumentation(ET.fromstring(SOURCE), report)

    assert document.doi == "10.1/x" and document.version == "8.1"
    assert [a.name for a in document.authors] == ["A. Evaluator"]
    assert document.dates[0].dateType == "issued"
    assert document.endfCompatible.text.startswith(" 2.605600+4")
    assert any("keywords" in line for line in report.losses)

    parent = ET.Element("evaluated")
    again = readDocumentation(writeDocumentation(parent, document),
                              ConversionReport())
    assert again == document

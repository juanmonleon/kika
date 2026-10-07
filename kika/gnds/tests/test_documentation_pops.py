"""Preserve represented GNDS provenance and independent particle scopes."""
import xml.etree.ElementTree as ET
from kika.gnds.styles import readStyles,writeStyles
from kika.gnds.encode_pops import writePoPs
from kika.nuclear_data.model import ConversionReport,PoPs,Nuclide,Particle,PhysicalQuantity
from kika.gnds.encode import _number


def test_text_author_and_date_provenance_survive_style_roundtrip():
    xml=ET.fromstring("""<styles><evaluated label="eval"><documentation doi="10.test/example" version="2">
      <authors><author name="Evaluator" orcid="0000-test" email="a@example.org"/></authors>
      <dates><date value="2026-10-07" dateType="created"/></dates>
      <title encoding="utf8">Title &amp; data</title><body markup="none">Line 1
Line 2 &lt;x&gt;</body>
      <endfCompatible> original ENDF header
 second line</endfCompatible>
      </documentation></evaluated></styles>""")
    report=ConversionReport();styles=readStyles(xml,'/styles',report)
    assert report.isClean
    root=ET.Element('reactionSuite');writeStyles(root,styles,_number)
    reread=readStyles(root.find('styles'),'/styles',report)
    assert report.isClean
    assert reread['eval'].documentation==styles['eval'].documentation


def test_empty_documentation_does_not_invent_loss_or_data():
    report=ConversionReport()
    styles=readStyles(ET.fromstring('<styles><evaluated label="eval"><documentation/></evaluated></styles>'),'/styles',report)
    assert report.isClean
    assert styles['eval'].documentation is None


def test_unmodeled_documentation_is_still_reported():
    report=ConversionReport()
    readStyles(ET.fromstring('<styles><evaluated label="eval"><documentation><bibliography/></documentation></evaluated></styles>'),'/styles',report)
    assert not report.isClean
    assert any('bibliography' in entry for entry in report.losses)


def test_local_pops_has_independent_masses_and_correct_isotope_labels():
    # Particle fragments are checked through the same reader used for a suite.
    pops=PoPs(name='local',version='1')
    pops.add(Particle('n',mass=PhysicalQuantity(1.,'amu'),spin=PhysicalQuantity(.5,'hbar'),
                      parity=1,charge=0,halflife='stable'))
    pops.add(Nuclide('Co58_e1',mass=PhysicalQuantity(58.,'amu'),spin=PhysicalQuantity(5.,'hbar'),
                      parity=1,charge=0,Z=27,A=58,nuclearLevel=1))
    root=ET.Element('holder');report=ConversionReport();node=writePoPs(root,pops,report)
    assert report.isClean
    chemical=node.find('chemicalElements/chemicalElement')
    assert chemical.get('symbol')=='Co'
    assert chemical.find('isotopes/isotope').get('symbol')=='Co58'
    assert node.find('.//nuclide').get('id')=='Co58_e1'


def test_unidentified_nuclide_cannot_disappear_without_a_loss_report():
    pops=PoPs();pops.add(Nuclide('missing',Z=None,A=None))
    report=ConversionReport();writePoPs(ET.Element('holder'),pops,report)
    assert not report.isClean
    assert any('Z/A is missing' in loss for loss in report.losses)

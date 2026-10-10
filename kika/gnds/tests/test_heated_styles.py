"""Processed GNDS styles retain temperature, provenance and labelled forms."""
import copy
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.encode import _number
from kika.gnds.styles import readStyles, writeStyles
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (
    ConversionReport, Heated, PhysicalQuantity, Styles,
)


def style_xml(value='293.6', unit='K', source='eval'):
    return ET.fromstring(f'''<styles><heated label="hot" derivedFrom="{source}"
      date="2026-10-10"><temperature value="{value}" unit="{unit}"/>
      <documentation><body markup="none">Cross sections only &amp; coverage.
Source distributions retained.</body></documentation></heated></styles>''')


@pytest.mark.parametrize('value', ['0', '293.6', '600', '1200', '1e-12'])
def test_heated_style_roundtrip(value):
    report = ConversionReport()
    before = readStyles(style_xml(value), '/styles', report)
    root = ET.Element('reactionSuite')
    writeStyles(root, before, _number)
    after = readStyles(root.find('styles'), '/styles', report)
    assert report.isClean
    assert isinstance(after['hot'], Heated)
    assert after['hot'] == before['hot']
    assert after['hot'].temperature == PhysicalQuantity(float(value), 'K')
    assert after['hot'].derivedFrom == 'eval'
    assert after['hot'].date == '2026-10-10'
    assert after['hot'].documentation.body.text.startswith('Cross sections only')


@pytest.mark.parametrize('value,unit', [
    ('-1', 'K'), ('nan', 'K'), ('inf', 'K'), ('-inf', 'K'),
    ('1', ''), ('1', 'eV'), ('invalid', 'K'),
])
def test_invalid_xml_temperature_rejected(value, unit):
    with pytest.raises(ValueError):
        readStyles(style_xml(value, unit), '/styles', ConversionReport())


@pytest.mark.parametrize('missing', ['temperature', 'value', 'unit', 'derivedFrom'])
def test_required_heated_metadata(missing):
    xml = style_xml()
    hot = xml.find('heated')
    if missing == 'temperature':
        hot.remove(hot.find('temperature'))
    elif missing == 'derivedFrom':
        hot.attrib.pop(missing)
    else:
        hot.find('temperature').attrib.pop(missing)
    with pytest.raises(ValueError, match='heated style'):
        readStyles(xml, '/styles', ConversionReport())


@pytest.mark.parametrize('temperature', [
    None, 600., PhysicalQuantity(-1., 'K'), PhysicalQuantity(float('nan'), 'K'),
    PhysicalQuantity(float('inf'), 'K'), PhysicalQuantity(1., 'eV'),
    PhysicalQuantity(True, 'K'), PhysicalQuantity('600', 'K'),
])
def test_invalid_mutable_model_rejected_before_tree_changes(temperature):
    styles = Styles()
    styles.add(Heated(label='hot', derivedFrom='eval', temperature=temperature))
    root = ET.Element('reactionSuite')
    with pytest.raises(ValueError, match='heated style'):
        writeStyles(root, styles, _number)
    assert len(root) == 0


def test_missing_source_rejected_on_write():
    styles = Styles()
    styles.add(Heated(label='hot', temperature=PhysicalQuantity(600., 'K')))
    with pytest.raises(ValueError, match='derivedFrom'):
        writeStyles(ET.Element('reactionSuite'), styles, _number)


def test_complete_suite_keeps_heated_forms_and_originals(h2_gnds, tmp_path):
    # This exercises publication, not the numerical broadening engine.
    before = kika.read(h2_gnds, covariances=False)
    hot = readStyles(style_xml(), '/styles', ConversionReport())['hot']
    before.styles.add(hot)
    for collection in (before.reactions, before.sums):
        for reaction in collection:
            form = copy.deepcopy(reaction.crossSection['eval'])
            form.label = 'hot'
            reaction.crossSection['hot'] = form
    path = tmp_path / 'heated.xml'
    kika.write(before, path)
    after, read_report = readReactionSuite(Document.parse(path))
    assert not any('/styles/heated:' in item for item in read_report.unsupported)
    assert after.styles['hot'] == hot
    for left_collection, right_collection in (
        (before.reactions, after.reactions), (before.sums, after.sums),
    ):
        for left, right in zip(left_collection, right_collection):
            for label in ('eval', 'hot'):
                a, b = left.crossSection[label], right.crossSection[label]
                assert type(a) is type(b)
                curves_a = getattr(a, 'function1ds', [a])
                curves_b = getattr(b, 'function1ds', [b])
                for x, y in zip(curves_a, curves_b):
                    if hasattr(x, 'xs'):
                        np.testing.assert_array_equal(x.xs, y.xs)
                        np.testing.assert_array_equal(x.ys, y.ys)
                    elif hasattr(x, 'link'):
                        assert x.link == y.link
    second = tmp_path / 'heated_again.xml'
    kika.write(after, second)
    assert ET.canonicalize(from_file=str(path), strip_text=True) == ET.canonicalize(
        from_file=str(second), strip_text=True)


def test_heated_publication_validates_against_fudge_schema(h2_gnds, tmp_path):
    from kika.gnds.tests.test_encode import SCHEMA
    if not SCHEMA.exists() or shutil.which('xmllint') is None:
        pytest.skip('FUDGE schema or xmllint unavailable')
    suite = kika.read(h2_gnds, covariances=False)
    suite.styles.add(readStyles(style_xml(), '/styles', ConversionReport())['hot'])
    path = tmp_path / 'schema_heated.xml'
    kika.write(suite, path)
    validation = subprocess.run(
        ['xmllint', '--noout', '--huge', '--schema', str(SCHEMA), str(path)],
        capture_output=True, text=True,
    )
    assert validation.returncode == 0, validation.stderr


def test_multiple_temperatures_keep_source_chain_and_optional_documentation():
    xml = ET.fromstring('''<styles><evaluated label="eval"/>
      <crossSectionReconstructed label="recon" derivedFrom="eval"/>
      <heated label="hot600" derivedFrom="recon" date="2026-10-10">
        <temperature value="600" unit="K"/></heated>
      <heated label="hot1200" derivedFrom="hot600" date="2026-10-10">
        <temperature value="1200" unit="K"/></heated></styles>''')
    report = ConversionReport()
    before = readStyles(xml, '/styles', report)
    root = ET.Element('reactionSuite')
    writeStyles(root, before, _number)
    after = readStyles(root.find('styles'), '/styles', report)
    assert report.isClean
    assert after.labels == ['eval', 'recon', 'hot600', 'hot1200']
    for label in ('hot600', 'hot1200'):
        assert after[label] == before[label]
        assert after[label].documentation is None

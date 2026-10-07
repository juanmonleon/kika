"""Numerical GNDS equivalence and preservation of interpolation semantics."""
import xml.etree.ElementTree as ET
import numpy as np
from kika.gnds.resonances import readResonances
from kika.gnds.encode_resonances import writeResonances
from kika.nuclear_data.model import ConversionReport
from kika.processing.resonances import prepare_resonances, NeutronContext


def read(xml):
    report = ConversionReport()
    return readResonances(xml,'/reactionSuite',None,report,lambda element:None),report


def test_gnds_bw_units_and_radius_law_survive_roundtrip():
    xml = ET.fromstring('''<resonances><resolved domainMin="0.001" domainMax="1" domainUnit="keV">
      <BreitWigner approximation="MultiLevel"><scatteringRadius><XYs1d interpolation="log-log">
      <axes><axis index="1" label="energy_in" unit="keV"/><axis index="0" label="radius" unit="fm"/></axes>
      <values>0.001 3 1 7</values></XYs1d></scatteringRadius><resonanceParameters>
      <table rows="1" columns="6"><columnHeaders>
      <column index="0" name="energy" unit="keV"/><column index="1" name="L" unit=""/>
      <column index="2" name="J" unit=""/><column index="3" name="totalWidth" unit="keV"/>
      <column index="4" name="neutronWidth" unit="keV"/><column index="5" name="captureWidth" unit="keV"/>
      </columnHeaders><data>0.1 1 1.5 0.0003 0.0001 0.0002</data></table>
      </resonanceParameters></BreitWigner></resolved></resonances>''')
    m,report = read(xml)
    assert report.isClean
    p = prepare_resonances(m,NeutronContext(56.,0.),conversion_report=report)
    expected = 3.*100.**(np.log(7/3)/np.log(1000.))
    assert np.isclose(p.regions[0].groups[0].phase_radius.evaluate(100.),expected)
    root = ET.Element('reactionSuite')
    writer = ConversionReport()
    writeResonances(root,m,writer,('1','1000'))
    assert writer.isClean
    reloaded,report = read(root.find('resonances'))
    q = prepare_resonances(reloaded,NeutronContext(56.,0.),conversion_report=report)
    energies = [1.,25.,99.9,100.,100.1,400.,1000.]
    for mt,v in p.evaluate(energies).items():
        np.testing.assert_allclose(q.evaluate(energies)[mt],v,rtol=2e-14)


def test_urr_region_functions_are_preserved():
    xml = ET.fromstring('''<resonances><unresolved domainMin="1" domainMax="10" domainUnit="eV">
      <tabulatedWidths><Ls><L value="0"><Js><J value="0.5"><levelSpacing><regions1d>
      <axes><axis index="1" label="energy_in" unit="eV"/><axis index="0" label="spacing" unit="eV"/></axes>
      <function1ds><XYs1d index="0" interpolation="flat"><values>1 2 3 4</values></XYs1d>
      <XYs1d index="1" interpolation="log-log"><values>3 4 10 8</values></XYs1d></function1ds>
      </regions1d></levelSpacing><widths/></J></Js></L></Ls></tabulatedWidths></unresolved></resonances>''')
    m,report = read(xml)
    assert report.isClean
    function = m.unresolved.tabulatedWidths.spinGroups[0].levelSpacingFunction
    assert function is not None
    assert len(function.function1ds) == 2
    root = ET.Element('reactionSuite')
    writeResonances(root,m,ConversionReport(),('1','10'))
    reloaded,_ = read(root.find('resonances'))
    np.testing.assert_allclose(function.evaluate([2.,3.,5.,9.]),
        reloaded.unresolved.tabulatedWidths.spinGroups[0].levelSpacingFunction.evaluate([2.,3.,5.,9.]))


def test_urr_constant_units_are_normalized_and_roundtrip():
    xml=ET.fromstring('''<resonances><unresolved domainMin="0.001" domainMax="1" domainUnit="keV">
      <tabulatedWidths><resonanceReactions><resonanceReaction label="n"><link href="/reactionSuite/reactions/reaction[@label='elastic']"/></resonanceReaction></resonanceReactions><Ls><L value="0"><Js><J value="0.5"><levelSpacing>
      <constant1d value="0.002" domainMin="0.001" domainMax="1">
      <axes><axis index="1" label="energy_in" unit="keV"/><axis index="0" label="spacing" unit="keV"/></axes>
      </constant1d></levelSpacing><widths/></J></Js></L></Ls></tabulatedWidths></unresolved></resonances>''')
    model,report=read(xml)
    assert report.isClean
    assert model.unresolved.domainMin == 1.
    assert model.unresolved.domainMax == 1000.
    assert model.unresolved.domainUnit == 'eV'
    group=model.unresolved.tabulatedWidths.spinGroups[0]
    assert group.levelSpacingFunction.constant == 2.
    assert group.levelSpacingFunction.domainMin == 1.
    root=ET.Element('reactionSuite')
    writer=ConversionReport()
    writeResonances(root,model,writer,('1','1000'))
    assert writer.isClean
    reread,report=read(root.find('resonances'))
    assert report.isClean
    assert reread.unresolved.tabulatedWidths.spinGroups[0].levelSpacingFunction.constant == 2.


def test_reduced_amplitude_units_are_normalized_and_written_as_amplitudes():
    xml = ET.fromstring('''<resonances><resolved domainMin="1" domainMax="1000" domainUnit="eV">
      <RMatrix approximation="ReichMoore" reducedWidthAmplitudes="true"><resonanceReactions>
      <resonanceReaction label="n"><link href="/reactionSuite/reactions/reaction[@label='elastic']"/></resonanceReaction>
      </resonanceReactions><spinGroups><spinGroup label="s" spin="0.5" parity="1"><channels>
      <channel label="n" resonanceReaction="n" columnIndex="1" L="0" channelSpin="0.5"/></channels>
      <resonanceParameters><table rows="1" columns="2"><columnHeaders>
      <column index="0" name="energy" unit="keV"/><column index="1" name="n width" unit="keV**(1/2)"/>
      </columnHeaders><data>0.1 -0.2</data></table></resonanceParameters></spinGroup></spinGroups></RMatrix>
      </resolved></resonances>''')
    m,report = read(xml)
    assert report.isClean
    f = m.resolved[0].formalism
    assert f.spinGroups[0].energies == [100.]
    np.testing.assert_allclose(f.spinGroups[0].widths,[[-.2*np.sqrt(1000.)]])
    root=ET.Element('reactionSuite')
    writer=ConversionReport()
    writeResonances(root,m,writer,('1','1000'))
    assert writer.isClean
    reloaded,report=read(root.find('resonances'))
    assert report.isClean
    np.testing.assert_allclose(reloaded.resolved[0].formalism.spinGroups[0].widths,f.spinGroups[0].widths)

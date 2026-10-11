"""Real Doppler results published through GNDS and derived ENDF tapes."""
from pathlib import Path
import copy
import numpy as np
import pytest
import kika
from kika.nuclear_data.model import XYs1d, Regions1d, Axis, crossSectionAxes, RangeQuantity, PhysicalQuantity
from kika.processing.doppler import DopplerContext, SupportPolicy, ReactionKey, broaden_suite, attach_broadening
from kika.endf.model_adapter.pendf import pendfHeader, attachReconstruction
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import writeHeatedEndfTape

DATA = Path(__file__).resolve().parents[2] / 'tests/data/micro_fe56_xs_and_angular.endf'


def _processed():
    suite = kika.read(DATA, format='endf')
    suite.resonances = None
    suite.covarianceSuite = None  # Synthetic scalar fixture, no covariance claim.
    suite.styles['eval'].projectileEnergyDomain = RangeQuantity(.001, 10., 'eV')
    for r in (*suite.reactions, *suite.sums):
        y = {2: 2., 102: 1., 1: 3.}[r.ENDF_MT]
        r.crossSection['eval'] = XYs1d([.001, .0253, 1., 10.], [y]*4,
                                     axes=crossSectionAxes(), label='eval')
    support = {ReactionKey('reactions', r.label): SupportPolicy('constant', 'constant')
               for r in suite.reactions}
    result = broaden_suite(suite, DopplerContext(55.45443), source_style='eval',
                           label='hot', temperature=600., support=support)
    attach_broadening(suite, result)
    from datetime import date
    assert suite.styles['hot'].date == date.today().isoformat()
    return suite, result


@pytest.fixture(scope='module')
def baseline():
    return _processed()


@pytest.fixture
def processed(baseline):
    return copy.deepcopy(baseline[0]), baseline[1]


def test_heated_endf_header_and_selected_curves(tmp_path, processed):
    suite, result = processed
    original = copy.deepcopy(suite.provenance)
    path = tmp_path/'hot.pendf'
    kika.write(suite, path, format='endf', label='hot')
    header = pendfHeader(path)
    assert header['temp'] == 600. and header['awr'] == 55.45443
    tape = read_endf(str(path), mf_numbers=[1,3])
    assert tape.mf[1].mt[451]._lrp == 2 and tape.mf[1].mt[451]._ldrv == 1
    assert tape.mf[1].mt[451]._emax == 10.
    q = np.geomspace(.001, 10., 61)
    reread = {}
    for mt, form in result.forms_by_mt.items():
        section = tape.mf[3].mt[mt]
        values = np.interp(q, section.energies, section.cross_sections)
        source = np.interp(q, form.xs, form.ys)
        ratio = abs(values-source)/(result.options.atol+result.options.rtol*abs(source))
        assert max(ratio) < result.options.serialization_share
        reread[mt] = values
    np.testing.assert_allclose(reread[1], reread[2]+reread[102], rtol=1e-6)
    assert suite.provenance == original and suite.styles['eval'].temperature.value == 0
    with pytest.raises(ValueError, match='count the Doppler effect twice'):
        attachReconstruction(suite, path)


def test_real_heated_gnds_roundtrip(tmp_path, processed):
    suite, result = processed
    path = tmp_path/'hot.xml'
    kika.write(suite, path)
    after = kika.read(path, covariances=False)
    assert after.styles['hot'].temperature == PhysicalQuantity(600., 'K')
    assert after.styles['hot'].documentation.body.text == suite.styles['hot'].documentation.body.text
    forms = {}
    for container in ('reactions', 'sums'):
        for left, right in zip(getattr(suite, container), getattr(after, container)):
            np.testing.assert_array_equal(left.crossSection['eval'].xs, right.crossSection['eval'].xs)
            np.testing.assert_array_equal(left.crossSection['eval'].ys, right.crossSection['eval'].ys)
            forms[ReactionKey(container, right.label)] = right.crossSection['hot']
    # The in-memory stale-source guard includes ENDF provenance, which GNDS
    # intentionally replaces. Verify represented source arrays and hot tables.
    result.verify_table(forms=forms)


def test_verified_heated_publication(tmp_path, processed):
    suite, result = processed
    path = tmp_path/'verified.pendf'
    report = writeHeatedEndfTape(suite, result, path)
    assert report.isClean and pendfHeader(path)['temp'] == 600.


def test_verified_heated_publication_coalesces_redundant_ulp_knots(tmp_path, processed):
    suite, result = processed
    before = {}
    for reaction in (*suite.reactions, *suite.sums):
        form = reaction.crossSection['hot']
        index = len(form.xs)//2
        energy = float(form.xs[index])
        neighbour = np.nextafter(energy, np.inf)
        value = np.interp(neighbour, form.xs, form.ys)
        form.xs = np.insert(form.xs, index+1, neighbour)
        form.ys = np.insert(form.ys, index+1, value)
        before[reaction.ENDF_MT] = (form.xs.copy(), form.ys.copy())
    path = tmp_path/'coalesced.pendf'
    assert writeHeatedEndfTape(suite, result, path).isClean
    tape = read_endf(str(path), mf_numbers=[3])
    for reaction in (*suite.reactions, *suite.sums):
        section = tape.mf[3].mt[reaction.ENDF_MT]
        assert np.all(np.diff(section.energies)>0)
        x,y = before[reaction.ENDF_MT]
        np.testing.assert_array_equal(reaction.crossSection['hot'].xs,x)
        np.testing.assert_array_equal(reaction.crossSection['hot'].ys,y)
        reread = np.interp(x,section.energies,section.cross_sections)
        assert np.max(abs(reread-y)/(result.options.atol+result.options.rtol*abs(y))) <= result.options.serialization_share


@pytest.mark.parametrize('date', ['2026-10-11','20261011','2026-W41-7'])
def test_explicit_processing_date_and_invalid_date(processed,date):
    from kika.nuclear_data.model import Styles
    suite, result = processed
    suite.styles = Styles([s for s in suite.styles if s.label != result.label])
    for r in (*suite.reactions, *suite.sums):r.crossSection.forms.pop(result.label)
    with pytest.raises(ValueError, match='ISO calendar date'):
        attach_broadening(suite, result, date='not-a-date')
    assert result.label not in suite.styles.labels
    attach_broadening(suite, result, date=date)
    assert suite.styles[result.label].date == '2026-10-11'


def test_equal_region_axes_inherit_and_differing_units_are_reported(tmp_path, processed):
    suite, result = processed
    regions=Regions1d([
        XYs1d([.001,1.],[1.,2.],axes=crossSectionAxes()),
        XYs1d([1.,10.],[3.,4.],axes=crossSectionAxes())],axes=crossSectionAxes(),label='hot')
    suite.reactions[102].crossSection['hot']=regions
    path=tmp_path/'regions.xml'
    report=kika.write(suite,path)
    # This micro-tape intentionally lacks a neutron multiplicity. Its unrelated
    # conversion loss remains reported; equal region axes must not add a loss.
    assert not any('axes of its own' in entry for entry in report.losses)
    after=kika.read(path,covariances=False).reactions[102].crossSection['hot']
    for before,reread in zip(regions.function1ds,after.function1ds):
        np.testing.assert_array_equal(before.xs,reread.xs)
        np.testing.assert_array_equal(before.ys,reread.ys)
        assert reread.axes is after.axes
    regions.function1ds[0].axes.axes[0]=Axis(0,'crossSection','mb')
    differing=kika.write(suite,tmp_path/'different.xml')
    assert any('axes of its own' in entry for entry in differing.losses)


def test_verified_publication_rejects_corrupted_serialization(tmp_path, processed, monkeypatch):
    import kika.endf.writers.assemble as writer
    suite, result = processed
    original = writer._resample_reconstructed_endf
    def corrupted(suite, label):
        candidate = original(suite, label)
        candidate.reactions[102].crossSection[label].ys *= 1.1
        return candidate
    monkeypatch.setattr(writer, '_resample_reconstructed_endf', corrupted)
    path = tmp_path/'keep.pendf'
    path.write_text('keep')
    with pytest.raises(ValueError, match='serialization budget'):
        writeHeatedEndfTape(suite, result, path)
    assert path.read_text() == 'keep'


def test_verified_publication_refuses_lossy_covariance_conversion(tmp_path, processed):
    suite, result = processed
    suite.covarianceSuite = kika.read(DATA, format='endf').covarianceSuite
    path = tmp_path/'keep.pendf'
    path.write_text('keep')
    with pytest.raises(ValueError, match='conversion loses represented data'):
        writeHeatedEndfTape(suite, result, path)
    assert path.read_text() == 'keep'


@pytest.mark.parametrize('failure', ['missing-form', 'temperature', 'coverage'])
def test_invalid_heated_export_preserves_destination(tmp_path, failure, processed):
    suite, result = processed
    if failure == 'missing-form':
        suite.reactions[2].crossSection.forms.pop('hot')
    elif failure == 'temperature':
        suite.styles['hot'].temperature = PhysicalQuantity(float('nan'), 'K')
    else:
        suite.reactions[2].crossSection['hot'].xs[-1] = 11.
    path = tmp_path/'keep.endf'
    path.write_text('keep')
    with pytest.raises(ValueError):
        kika.write(suite, path, format='endf', label='hot')
    assert path.read_text() == 'keep'

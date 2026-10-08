"""A scoped consumer never silently exempts an unclassified conversion issue."""
from kika.nuclear_data.model import ConversionReport


def test_scoped_issue_remains_visible_and_globally_unclean():
    report = ConversionReport()
    report.unsupportedNode('energy metadata', unaffectedScopes=('cross-sections',))
    assert not report.isClean
    assert report.isCleanFor('cross-sections')
    assert not report.isCleanFor('transport')
    combined = ConversionReport()
    combined.extend(report)
    assert combined.isCleanFor('cross-sections')
    assert combined.unsupported == ['energy metadata']


def test_unknown_duplicate_loss_and_approximation_still_block():
    for add in ('unsupportedNode', 'lost', 'approximated'):
        report = ConversionReport()
        report.unsupportedNode('same text', unaffectedScopes=('cross-sections',))
        getattr(report, add)('same text')
        assert not report.isCleanFor('cross-sections')

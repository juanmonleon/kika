"""The ACE decoder names the neutron-side blocks it does not convert.

Runs without an ACE file: the check reads only which blocks are present.
"""
from types import SimpleNamespace

from kika.ace.model_adapter.decode import _reportUnconverted
from kika.nuclear_data.model import ConversionReport


def test_every_unconverted_block_is_named_and_leaves_cross_sections_clean():
    ace = SimpleNamespace(
        angular_distributions=SimpleNamespace(elastic=object(), incident_neutron={51: 1, 16: 1},
                                              particle_production=[]),
        energy_distributions=SimpleNamespace(incident_neutron={16: [1]}, delayed_neutron=[1],
                                             particle_production=[]),
        nubar=SimpleNamespace(prompt=object(), total=object(), delayed=None),
        delayed_neutron_data=SimpleNamespace(precursors=[1] * 6),
    )
    report = ConversionReport()
    _reportUnconverted(ace, report)
    text = "\n".join(report.unsupported)
    assert "angular distributions (AND block, MT 2, 16, 51)" in text
    assert "energy distributions (DLW block, MT 16)" in text
    assert "delayed-neutron energy spectra" in text
    assert "nu-bar (prompt, total;" in text
    assert "6 groups" in text
    assert not report.isClean and report.isCleanFor("cross-sections")


def test_an_absent_block_says_nothing():
    report = ConversionReport()
    _reportUnconverted(SimpleNamespace(), report)
    assert report.isEmpty

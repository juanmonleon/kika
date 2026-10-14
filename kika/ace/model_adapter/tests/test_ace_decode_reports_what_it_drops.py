"""The ACE decoder names the neutron-side blocks it does not convert.

Runs without an ACE file: the check reads only which blocks are present.
"""
from types import SimpleNamespace

from kika.ace.model_adapter.decode import _reportUnconverted
from kika.nuclear_data.model import ConversionReport


def test_only_the_secondary_particle_blocks_are_unconverted():
    # Since A3 of the ACE roadmap AND, DLW, NU, BDD and DNED reach the model;
    # the charged-particle blocks (IXS) do not, as in OpenMC.
    ace = SimpleNamespace(
        angular_distributions=SimpleNamespace(elastic=object(), incident_neutron={51: 1, 16: 1},
                                              particle_production=[{5: 1}]),
        energy_distributions=SimpleNamespace(incident_neutron={16: [1]}, delayed_neutron=[1],
                                             particle_production=[{5: [1]}]),
    )
    report = ConversionReport()
    _reportUnconverted(ace, report)
    (text,) = report.unsupported
    assert "secondary-particle data (IXS" in text
    assert not report.isClean and report.isCleanFor("cross-sections")


def test_neutron_blocks_alone_say_nothing():
    ace = SimpleNamespace(
        angular_distributions=SimpleNamespace(elastic=object(), incident_neutron={16: 1},
                                              particle_production=[]),
        energy_distributions=SimpleNamespace(incident_neutron={16: [1]}, particle_production=[]))
    report = ConversionReport()
    _reportUnconverted(ace, report)
    assert report.isEmpty


def test_an_absent_block_says_nothing():
    report = ConversionReport()
    _reportUnconverted(SimpleNamespace(), report)
    assert report.isEmpty

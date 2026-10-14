"""Writing a thermal evaluation as HDF5 hands the principal nuclide's tape to the processing.

The table's header AWR is ACER's, read from that neutron tape (graphite 11.89365);
without it the header carries the scattering law's B(3) (11.898). ``kika.write``
used to drop the ``neutron=`` option before it reached
:func:`kika.hdf5.thermal_processing.processThermal`.
"""
from __future__ import annotations

import pytest

from kika.hdf5 import thermal_processing, writer


class _Stop(Exception):
    pass


def test_the_neutron_tape_reaches_the_thermal_processing(monkeypatch, tmp_path):
    seen = {}

    def fake(thermal, neutron=None, **kwargs):
        seen.update(thermal=thermal, neutron=neutron, name=kwargs.get("name"))
        raise _Stop

    monkeypatch.setattr(thermal_processing, "processThermal", fake)

    class _Suite:
        interaction = "thermalNeutronScatteringLaw"
        styles = ()

    with pytest.raises(_Stop):
        writer.writeSuite(_Suite(), tmp_path, endf="tsl-graphite.endf",
                          neutron="n-006_C_012.endf", name="c_Graphite")
    assert seen == dict(thermal="tsl-graphite.endf", neutron="n-006_C_012.endf",
                        name="c_Graphite")

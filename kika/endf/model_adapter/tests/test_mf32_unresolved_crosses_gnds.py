"""MF32's unresolved-range covariance crosses kika's GNDS and comes back as the tape states it."""
from __future__ import annotations

import pytest

import kika

URR_RANGE = f"{2:11d}{1:11d}{0:11d}{0:11d}"


def _mf32(path):
    with open(path, encoding="latin-1") as handle:
        return [line[:66] for line in handle if line[70:75] == "32151"]


def _unresolvedBody(lines):
    return lines[max(i for i, line in enumerate(lines) if i > 1 and line[22:66] == URR_RANGE):]


@pytest.mark.slow
def test_th232_unresolved_covariance_comes_back_through_gnds(th232_b81_tape, tmp_path):
    gnds = tmp_path / "th232.gnds.xml"
    kika.write(kika.read(str(th232_b81_tape), covariances=True), gnds, format="gnds")
    suite = kika.read(str(gnds), covariances=True)
    averages = [c for c in suite.covarianceSuite.parameterCovariances
                if type(c).__name__ == "AverageParameterCovariance"]
    assert len(averages) == 60
    assert sum(c.crossTerm for c in averages) == 45

    written = tmp_path / "th232.endf"
    kika.write(suite, str(written), format="endf")
    assert _unresolvedBody(_mf32(written)) == _unresolvedBody(_mf32(th232_b81_tape))
